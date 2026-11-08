"""Tests des invariants du RAG et du contrôle d'agent, sans appel API payant."""
import json
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from langchain_core.messages import AIMessage, ToolMessage

from rag_multimodal import (
    Element, EncodeurCLIP, IndexMultimodal, decouper_texte,
    extraire_pdf, normaliser, sha256_fichier,
)
from agent_rag import creer_retriever_tool, interroger_agent, trace_execution


class EncodeurTest:
    """Vecteurs contrôlés pour vérifier l'index, pas la qualité de CLIP."""
    signature = "encodeur-unitaire-v1"

    def encoder_textes(self, textes):
        return normaliser(np.array([[1, 0, 0] for _ in textes], dtype=np.float32))

    def encoder_images(self, chemins):
        return normaliser(np.array([[0, 1, 0] for _ in chemins], dtype=np.float32))

    def encoder_requete(self, texte=None, image=None, poids_texte=0.5):
        if texte is None and image is None:
            raise ValueError("Requête vide.")
        if texte and image:
            return normaliser(np.array([poids_texte, 1 - poids_texte, 0]))
        return np.array([1, 0, 0] if texte else [0, 1, 0], dtype=np.float32)


@pytest.fixture
def index(tmp_path):
    index = IndexMultimodal(EncodeurTest(), tmp_path / "chroma")
    commun = {"document_id": "a" * 64, "pdf": "test.pdf", "page": 1}
    elements = [
        Element("texte-1", "Multi-head attention.",
                {**commun, "type": "texte", "source": "page1.txt"}),
        Element("texte-2", "Positional encoding.",
                {**commun, "type": "texte", "source": "page1.txt"}),
        Element("image-1", "Figure 1.",
                {**commun, "type": "image", "source": "figure.png"}),
    ]
    index.indexer(elements)
    return index, elements


def test_extraction_pdf_et_vecteurs(tmp_path):
    chemin = tmp_path / "source.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 50), "Attention is all you need")
        page.draw_rect(pymupdf.Rect(100, 100, 350, 250), color=(0, 0, 1))
        doc.save(chemin)
    manifeste = extraire_pdf(chemin, tmp_path / "sortie")
    assert len(manifeste["pages"]) == 1
    assert "Attention" in manifeste["pages"][0]["texte"]
    assert manifeste["document_id"] == sha256_fichier(chemin)
    assert any(i["origine"] == "rendu_vectoriel" for i in manifeste["images"])
    assert all(Path(i["source"]).is_file() for i in manifeste["images"])


def test_decoupage_clip_sans_perte():
    from open_clip.tokenizer import SimpleTokenizer
    bpe = SimpleTokenizer()
    texte = "The Transformer uses Multi-Head Attention. " * 80
    fragments = decouper_texte(texte, bpe)
    assert len(fragments) > 2
    assert all(0 < len(bpe.encode(f)) <= 60 for f in fragments)
    assert set(bpe.encode(texte)) <= set(t for f in fragments for t in bpe.encode(f))
    assert "Transformer" in fragments[0]
    with pytest.raises(ValueError):
        decouper_texte(texte, bpe, chevauchement=60)


def test_normalisation():
    vecteurs = normaliser(np.array([[3, 4], [0, 2]]))
    np.testing.assert_allclose(np.linalg.norm(vecteurs, axis=1), [1, 1])
    with pytest.raises(ValueError):
        normaliser(np.array([[0, 0]]))
    with pytest.raises(ValueError):
        normaliser(np.array([[np.nan, 1]]))


def test_reindexation_et_nettoyage(index):
    index, elements = index
    bilan = index.indexer(elements)
    assert bilan["nouveaux"] == 0
    assert bilan["total_collection"] == 3
    bilan = index.indexer(elements[:1])
    assert bilan["supprimes"] == 2
    assert index.collection.count() == 1


def test_recherche_cosinus_filtres_et_persistance(index, tmp_path):
    index, _ = index
    textes = index.rechercher(texte="attention", type_resultat="texte", k=10)
    assert len(textes) == 2
    assert all(r["type"] == "texte" for r in textes)
    assert textes[0]["similarite"] == pytest.approx(1, abs=1e-5)
    images = index.rechercher(image="figure.png", type_resultat="image", k=10)
    assert len(images) == 1 and images[0]["id"] == "image-1"
    assert index.rechercher(texte="x", document_id="inconnu") == []
    hybride = index.rechercher(texte="x", image="figure.png")
    assert hybride[0]["similarite"] == pytest.approx(2 ** -0.5, abs=1e-5)
    rouvert = IndexMultimodal(EncodeurTest(), tmp_path / "chroma")
    assert rouvert.collection.count() == 3
    with pytest.raises(ValueError):
        index.rechercher(texte="x", k=True)


class ModeleScenario:
    """Double de test : séquence d'appels déterministe, ce n'est pas un LLM."""
    def __init__(self, appels=2, groupe=1):
        self.tour = 0
        self.appels = appels
        self.groupe = groupe

    def bind_tools(self, outils):
        return self

    def invoke(self, messages):
        observations = [m for m in messages if isinstance(m, ToolMessage)]
        if len(observations) < self.appels:
            self.tour += 1
            return AIMessage(content="", tool_calls=[{
                "name": "retriever_tool",
                "args": {"requete": "multi-head attention", "type_resultat": "texte", "k": 2},
                "id": f"test-{self.tour}-{i}", "type": "tool_call",
            } for i in range(self.groupe)])
        return AIMessage(content="Synthèse de test à partir des observations [test.pdf, p. 1].")


def test_outil_retourne_sources_et_artifact(index):
    index, _ = index
    outil = creer_retriever_tool(index)
    observation = outil.invoke({"name": "retriever_tool", "id": "appel-1",
                               "type": "tool_call", "args": {"requete": "attention", "type_resultat": "texte"}})
    assert len(json.loads(observation.content)["sources"]) == 2
    assert observation.artifact[0]["page"] == 1


def test_agent_action_observation_reformulation(index):
    index, _ = index
    sortie = interroger_agent(index, "Explique l'attention", llm=ModeleScenario(2))
    assert len(sortie["appels"]) == 2
    assert len(sortie["sources"]) == 2
    assert [e["etape"] for e in trace_execution(sortie)] == [
        "Action", "Observation", "Action", "Observation", "Réponse finale"]


def test_budget_impose_meme_avec_appels_multiples(index):
    index, _ = index
    sortie = interroger_agent(index, "Question", llm=ModeleScenario(100, groupe=5), max_appels=2)
    observations = [m for m in sortie["execution"]["messages"] if isinstance(m, ToolMessage)]
    assert sum(bool(m.artifact) for m in observations) == 2
    assert "Budget" in sortie["reponse"]


def test_image_absente_et_schema(index):
    index, _ = index
    outil = creer_retriever_tool(index)
    resultat = json.loads(outil.invoke({"modalite": "image"}))
    assert "erreur" in resultat
    with pytest.raises(ValueError):
        outil.invoke({"requete": "x", "k": 1000})


def test_requete_mixte_utilise_deux_encodeurs():
    # Vérifier la véritable fonction de fusion de CLIP sans télécharger ses poids.
    encodeur = object.__new__(EncodeurCLIP)
    class BPE:
        def encode(self, texte):
            return texte.split()
    encodeur.bpe = BPE()
    encodeur.encoder_textes = lambda textes: np.array([[1, 0]], dtype=np.float32)
    encodeur.encoder_images = lambda images: np.array([[0, 1]], dtype=np.float32)
    np.testing.assert_allclose(encodeur.encoder_requete("attention", "figure.png"),
                               np.array([1, 1]) / np.sqrt(2), atol=1e-6)
    with pytest.raises(ValueError):
        encodeur.encoder_requete(" ")
