"""Services de l'interface : fichiers chargés, index local et accès à l'agent."""
from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
import re
from threading import RLock

from PIL import Image
import pymupdf

from rag_multimodal import EncodeurCLIP, IndexMultimodal, extraire_pdf, preparer_elements
from agent_rag import creer_llm, interroger_agent, trace_execution

ROOT = Path(__file__).resolve().parent


def enregistrer_pdf(contenu: bytes, nom: str, destination=None) -> Path:
    """Vérifier le PDF, puis le stocker dans un dossier identifié par son contenu."""
    if not contenu or len(contenu) > 30 * 1024 * 1024:
        raise ValueError("Choisis un PDF non vide de 30 Mo maximum.")
    try:
        with pymupdf.open(stream=contenu, filetype="pdf") as document:
            if document.needs_pass:
                raise ValueError("Ce PDF est protégé par un mot de passe. Utilise une copie déverrouillée.")
            if len(document) == 0:
                raise ValueError("Ce PDF ne contient aucune page.")
    except (pymupdf.FileDataError, pymupdf.EmptyFileError) as erreur:
        raise ValueError("Le fichier ne peut pas être lu comme un PDF.") from erreur
    nom = re.split(r"[/\\]", nom)[-1]
    nom = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", nom).strip(" .")
    if not nom.lower().endswith(".pdf"):
        nom = "document.pdf"
    # Préfixer évite également les noms de périphériques réservés sous Windows.
    nom = "pdf_" + nom[:120]
    empreinte = hashlib.sha256(contenu).hexdigest()
    dossier = Path(destination or ROOT / ".cache/uploads/pdfs") / empreinte
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / nom
    if not chemin.exists():
        chemin.write_bytes(contenu)
    return chemin


def enregistrer_image(contenu: bytes, destination=None) -> Path:
    if not contenu or len(contenu) > 10 * 1024 * 1024:
        raise ValueError("Choisis une image non vide de 10 Mo maximum.")
    try:
        with Image.open(BytesIO(contenu)) as image:
            image.load()
            rgba = image.convert("RGBA")
            image_rgb = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as erreur:
        raise ValueError("Cette image ne peut pas être lue. Utilise PNG, JPEG ou WebP.") from erreur
    empreinte = hashlib.sha256(contenu).hexdigest()
    dossier = Path(destination or ROOT / ".cache/uploads/images")
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / f"{empreinte}.png"
    if not chemin.exists():
        image_rgb.save(chemin)
    return chemin


class MoteurInterface:
    """Une ressource partagée ; verrouiller indexation et inférence concurrentes."""
    def __init__(self):
        self.verrou = RLock()
        self.encodeur = EncodeurCLIP(cache=str(ROOT / ".cache/clip"))
        self.index = IndexMultimodal(self.encodeur, ROOT / "chroma_pdf")

    def preparer(self, pdf):
        with self.verrou:
            # Séparer les documents importés des résultats de remise du TP.
            pdf = Path(pdf).resolve()
            sortie = ROOT / "resultats_pdf" if pdf.parent == ROOT / "pdfs" else ROOT / ".cache/documents"
            manifeste = extraire_pdf(pdf, sortie)
            elements = preparer_elements(manifeste, self.encodeur.bpe)
            bilan = self.index.indexer(elements)
            return {"manifeste": manifeste, "bilan": bilan, "pdf_path": str(pdf),
                    "textes": sum(e.metadata["type"] == "texte" for e in elements),
                    "images": sum(e.metadata["type"] == "image" for e in elements)}

    def rechercher(self, document_id, question="", image=None, k=5, type_resultat="tous"):
        with self.verrou:
            return self.index.rechercher(texte=question.strip() or None, image=image, k=k,
                                        type_resultat=type_resultat, document_id=document_id)

    def interroger(self, document_id, question="", image=None, api_key=None,
                   modele=None, max_appels=4):
        # Ne jamais placer les clés d'une session dans os.environ ou un cache global.
        llm = creer_llm(api_key=api_key, modele=modele)
        with self.verrou:
            sortie = interroger_agent(self.index, question=question, image=image,
                                      document_id=document_id, llm=llm, max_appels=max_appels)
        reponse = sortie["reponse"]
        if not isinstance(reponse, str):
            reponse = "\n".join(b.get("text", "") for b in reponse if isinstance(b, dict))
        return {"reponse": reponse, "sources": sortie["sources"],
                "trace": [e for e in trace_execution(sortie) if e["etape"] != "Réponse finale"]}


def creer_moteur():
    return MoteurInterface()
