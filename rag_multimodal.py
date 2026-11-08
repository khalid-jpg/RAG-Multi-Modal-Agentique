"""TP : extraction PDF, embeddings CLIP et recherche multimodale persistante."""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pymupdf
from PIL import Image


def charger_rgb(chemin):
    """Composer les images transparentes sur blanc, comme dans le PDF."""
    with Image.open(chemin) as image:
        rgba = image.convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")


# %% Étape 1 — Extraction
@dataclass
class Element:
    id: str
    contenu: str
    metadata: dict


def sha256_fichier(chemin: str | Path) -> str:
    empreinte = hashlib.sha256()
    with Path(chemin).open("rb") as fichier:
        for bloc in iter(lambda: fichier.read(1024 * 1024), b""):
            empreinte.update(bloc)
    return empreinte.hexdigest()


def extraire_pdf(pdf: str | Path, sortie: str | Path = "resultats_pdf") -> dict:
    """Sauvegarde texte, images intégrées et grands schémas vectoriels.

    Les chemins sauvegardés sont relatifs au projet, donc transportables.
    Les figures vectorielles sont des rendus de zones détectées, pas des
    images intégrées. Le champ origine permet de les distinguer.
    """
    pdf = Path(pdf)
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF introuvable : {pdf}")
    document_id = sha256_fichier(pdf)
    dossier = Path(sortie) / document_id[:16]
    textes_dir, images_dir = dossier / "textes", dossier / "images"
    textes_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    pages, images, avertissements = [], [], []
    with pymupdf.open(pdf) as document:
        for numero, page in enumerate(document, 1):
            texte = page.get_text("text", sort=True)
            chemin = textes_dir / f"page_{numero:03d}.txt"
            chemin.write_text(texte, encoding="utf-8")
            pages.append({"page": numero, "texte": texte, "source": str(chemin)})
            if not texte.strip():
                avertissements.append(f"Page {numero} sans texte : OCR nécessaire.")

            # Dédupliquer une image répétée sur une même page.
            xrefs_vus = set()
            for information in page.get_images(full=True):
                xref = information[0]
                if xref in xrefs_vus:
                    continue
                xrefs_vus.add(xref)
                pixmap = pymupdf.Pixmap(document, xref)
                if information[1]:  # conserver le masque de transparence
                    masque = pymupdf.Pixmap(document, information[1])
                    pixmap = pymupdf.Pixmap(pixmap, masque)
                if pixmap.colorspace and pixmap.colorspace.n != 3:
                    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
                chemin_image = images_dir / f"page_{numero:03d}_xref_{xref}.png"
                pixmap.save(str(chemin_image))
                charger_rgb(chemin_image).save(chemin_image)
                images.append({"page": numero, "source": str(chemin_image),
                               "origine": "image_integree", "xref": xref})

            # Les dessins vectoriels ne sont pas renvoyés par get_images().
            # Regrouper les zones proches, puis écarter lignes et petits signes.
            for indice, rectangle in enumerate(page.cluster_drawings(
                x_tolerance=10, y_tolerance=10
            )):
                if rectangle.width < 100 or rectangle.height < 70:
                    continue
                zone = (rectangle + (-8, -8, 8, 8)) & page.rect
                # Un schéma déjà couvert par une image intégrée est ignoré.
                couvert = any(
                    (zone & r).get_area() >= 0.8 * zone.get_area()
                    for xref in xrefs_vus for r in page.get_image_rects(xref)
                )
                if couvert:
                    continue
                chemin_image = images_dir / f"page_{numero:03d}_vecteur_{indice}.png"
                page.get_pixmap(clip=zone, dpi=160, alpha=False).save(str(chemin_image))
                images.append({"page": numero, "source": str(chemin_image),
                               "origine": "rendu_vectoriel", "bbox": list(zone)})

    texte_complet = "\n\n".join(
        f"===== PAGE {p['page']} =====\n{p['texte']}" for p in pages
    )
    (dossier / "texte_complet.txt").write_text(texte_complet, encoding="utf-8")
    manifeste = {"document_id": document_id, "pdf": pdf.name,
                 "pages": pages, "images": images, "avertissements": avertissements}
    (dossier / "manifest.json").write_text(
        json.dumps(manifeste, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifeste


# %% Étape 2 — Espace latent commun
def normaliser(vecteurs: np.ndarray) -> np.ndarray:
    vecteurs = np.asarray(vecteurs, dtype=np.float32)
    normes = np.linalg.norm(vecteurs, axis=-1, keepdims=True)
    if not np.isfinite(vecteurs).all() or np.any(normes < 1e-12):
        raise ValueError("Embedding non fini ou de norme nulle.")
    return vecteurs / normes


def decouper_texte(texte: str, bpe, taille: int = 60,
                   chevauchement: int = 10) -> list[str]:
    """Fragments bornés en tokens CLIP ; conserve le texte original si possible."""
    if type(taille) is not int or not 1 <= taille <= 75:
        raise ValueError("taille doit être un entier entre 1 et 75.")
    if type(chevauchement) is not int or not 0 <= chevauchement < taille:
        raise ValueError("Chevauchement invalide.")
    # Découper sur les mots préserve casse, ponctuation et équations.
    mots = list(re.finditer(r"\S+", texte))
    fragments, debut = [], 0
    while debut < len(mots):
        fin = debut
        while fin < len(mots):
            candidat = texte[mots[debut].start():mots[fin].end()]
            if len(bpe.encode(candidat)) > taille:
                break
            fin += 1
        if fin == debut:
            # Cas rare : un seul mot / URL dépasse la limite du modèle.
            tokens = bpe.encode(mots[debut].group())
            for position in range(0, len(tokens), taille):
                fragments.append(bpe.decode(tokens[position:position + taille]).strip())
            debut += 1
            continue
        fragments.append(texte[mots[debut].start():mots[fin - 1].end()])
        if fin == len(mots):
            break
        prochain = fin
        while prochain > debut + 1:
            recouvrement = texte[mots[prochain - 1].start():mots[fin - 1].end()]
            if len(bpe.encode(recouvrement)) > chevauchement:
                break
            prochain -= 1
        debut = prochain
    return [f for f in fragments if f.strip()]


class EncodeurCLIP:
    """Texte et images encodés par le même CLIP ViT-B-32 (512 dimensions)."""
    def __init__(self, modele="ViT-B-32", poids="laion2b_s34b_b79k",
                 cache=".cache/clip", device=None):
        import torch
        import open_clip
        from open_clip.tokenizer import SimpleTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Éviter la surallocation de threads sur un ordinateur portable.
        if self.device == "cpu":
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            modele, pretrained=poids, cache_dir=cache
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(modele)
        self.bpe = SimpleTokenizer()
        # Inclure la version du découpage et de l'extraction dans l'identité.
        self.signature = f"{modele}|{poids}|chunks60-overlap10|extraction-v1"

    def encoder_textes(self, textes: list[str], lot=32) -> np.ndarray:
        if not textes:
            return np.empty((0, 512), dtype=np.float32)
        if any(len(self.bpe.encode(t)) > 75 for t in textes):
            raise ValueError("Texte trop long pour CLIP : utiliser decouper_texte().")
        vecteurs = []
        with self.torch.inference_mode():
            for debut in range(0, len(textes), lot):
                tokens = self.tokenizer(textes[debut:debut + lot]).to(self.device)
                sortie = self.model.encode_text(tokens).float().cpu().numpy()
                vecteurs.append(normaliser(sortie))
        return np.concatenate(vecteurs)

    def encoder_images(self, chemins: list[str | Path], lot=8) -> np.ndarray:
        if not chemins:
            return np.empty((0, 512), dtype=np.float32)
        vecteurs = []
        with self.torch.inference_mode():
            for debut in range(0, len(chemins), lot):
                tenseurs = []
                for chemin in chemins[debut:debut + lot]:
                    tenseurs.append(self.preprocess(charger_rgb(chemin)))
                batch = self.torch.stack(tenseurs).to(self.device)
                sortie = self.model.encode_image(batch).float().cpu().numpy()
                vecteurs.append(normaliser(sortie))
        return np.concatenate(vecteurs)

    def encoder_requete(self, texte=None, image=None, poids_texte=0.5):
        if not 0 < poids_texte < 1:
            raise ValueError("poids_texte doit être strictement entre 0 et 1.")
        vecteurs = []
        if texte is not None:
            if not isinstance(texte, str) or not texte.strip():
                raise ValueError("La requête textuelle doit être non vide.")
            fragments = decouper_texte(texte, self.bpe)
            vecteurs.append(normaliser(self.encoder_textes(fragments).mean(axis=0)))
        if image is not None:
            vecteurs.append(self.encoder_images([image])[0])
        if not vecteurs:
            raise ValueError("Fournir du texte, une image ou les deux.")
        if len(vecteurs) == 2:
            return normaliser(poids_texte * vecteurs[0] + (1 - poids_texte) * vecteurs[1])
        return vecteurs[0]


def preparer_elements(manifeste: dict, bpe) -> list[Element]:
    elements = []
    document_id = manifeste["document_id"]
    commun = {"document_id": document_id, "pdf": manifeste["pdf"]}
    for page in manifeste["pages"]:
        for indice, fragment in enumerate(decouper_texte(page["texte"], bpe)):
            elements.append(Element(
                f"{document_id}:texte:{page['page']}:{indice}", fragment,
                {**commun, "type": "texte", "page": page["page"],
                 "fragment": indice, "source": page["source"]},
            ))
    for indice, image in enumerate(manifeste["images"]):
        elements.append(Element(
            f"{document_id}:image:{image['page']}:{indice}",
            f"Figure de {manifeste['pdf']}, page {image['page']}.",
            {**commun, "type": "image", "page": image["page"],
             "source": image["source"], "origine": image["origine"]},
        ))
    return elements


# %% Étape 3 — Indexation persistante et idempotente
class IndexMultimodal:
    def __init__(self, encodeur, chemin="chroma_pdf"):
        import chromadb
        from chromadb.config import Settings

        self.encodeur = encodeur
        self.client = chromadb.PersistentClient(
            path=str(chemin), settings=Settings(anonymized_telemetry=False)
        )
        signature = hashlib.sha256(encodeur.signature.encode()).hexdigest()[:16]
        self.collection = self.client.get_or_create_collection(
            name=f"pdf_clip_{signature}", embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"encodeur": encodeur.signature},
        )

    def indexer(self, elements: list[Element], lot=32) -> dict:
        if not elements:
            raise ValueError("Aucun élément à indexer.")
        document_ids = {e.metadata["document_id"] for e in elements}
        if len(document_ids) != 1:
            raise ValueError("Indexer un seul PDF par appel.")
        document_id = next(iter(document_ids))
        attendus = {e.id for e in elements}
        existants = set(self.collection.get(where={"document_id": document_id})["ids"])
        nouveaux = [e for e in elements if e.id not in existants]
        for type_contenu in ("texte", "image"):
            groupe = [e for e in nouveaux if e.metadata["type"] == type_contenu]
            for debut in range(0, len(groupe), lot):
                batch = groupe[debut:debut + lot]
                if type_contenu == "texte":
                    vecteurs = self.encodeur.encoder_textes([e.contenu for e in batch])
                else:
                    vecteurs = self.encodeur.encoder_images([e.metadata["source"] for e in batch])
                self.collection.upsert(
                    ids=[e.id for e in batch], documents=[e.contenu for e in batch],
                    metadatas=[e.metadata for e in batch], embeddings=vecteurs.tolist(),
                )
        # Retirer les anciens fragments seulement après la réussite des ajouts.
        obsoletes = sorted(existants - attendus)
        if obsoletes:
            self.collection.delete(ids=obsoletes)
        # Rafraîchir les chemins si le projet a été déplacé, sans réencoder.
        for debut in range(0, len(elements), lot):
            batch = elements[debut:debut + lot]
            self.collection.update(ids=[e.id for e in batch],
                                   metadatas=[e.metadata for e in batch])
        return {"elements": len(elements), "nouveaux": len(nouveaux),
                "supprimes": len(obsoletes), "total_collection": self.collection.count()}


    # Étape 4 — Recherche cosinus texte / image / hybride
    def rechercher(self, texte=None, image=None, k=5,
                   type_resultat: Literal["tous", "texte", "image"] = "tous",
                   document_id=None, poids_texte=0.5) -> list[dict]:
        if type(k) is not int or not 1 <= k <= 20:
            raise ValueError("k doit être un entier entre 1 et 20.")
        if type_resultat not in ("tous", "texte", "image"):
            raise ValueError("Type de résultat invalide.")
        vecteur = self.encodeur.encoder_requete(texte, image, poids_texte)
        conditions = []
        if type_resultat != "tous":
            conditions.append({"type": type_resultat})
        if document_id:
            conditions.append({"document_id": document_id})
        filtre = ({"$and": conditions} if len(conditions) > 1
                  else conditions[0] if conditions else None)
        options = {"where": filtre} if filtre else {}
        # Limiter k au nombre réellement présent après filtrage.
        nombre = len(self.collection.get(include=[], **options)["ids"])
        if not nombre:
            return []
        sortie = self.collection.query(
            query_embeddings=[vecteur.tolist()], n_results=min(k, nombre),
            include=["documents", "metadatas", "distances"], **options,
        )
        return [{"id": identifiant, "contenu": contenu, **metadata,
                 "similarite": float(np.clip(1 - distance, -1, 1))}
                for identifiant, contenu, metadata, distance in zip(
                    sortie["ids"][0], sortie["documents"][0],
                    sortie["metadatas"][0], sortie["distances"][0])]


def afficher_resultats(resultats: list[dict]):
    from IPython.display import Markdown, display

    if not resultats:
        print("Aucun résultat pour ce filtre.")
    for rang, resultat in enumerate(resultats, 1):
        print(f"{rang}. {resultat['type']} | {resultat['pdf']} | p. {resultat['page']}"
              f" | cosinus={resultat['similarite']:.3f}")
        if resultat["type"] == "texte":
            display(Markdown(resultat["contenu"]))
        else:
            apercu = charger_rgb(resultat["source"])
            apercu.thumbnail((700, 500))
            display(apercu)
