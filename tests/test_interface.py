"""Validation des fichiers importés et des parcours Streamlit sans appel API."""
from io import BytesIO
from pathlib import Path
import os

from PIL import Image
import pymupdf
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import streamlit_backend as backend

ROOT = Path(__file__).resolve().parents[1]


def test_import_pdf_verifie_et_confine_le_chemin(tmp_path):
    with pymupdf.open() as document:
        document.new_page().insert_text((40, 40), "Document de test")
        contenu = document.tobytes()
    chemin = backend.enregistrer_pdf(contenu, "../../.git/cours.pdf", tmp_path)
    assert chemin.resolve().is_relative_to(tmp_path.resolve())
    assert chemin.name == "pdf_cours.pdf"
    assert backend.enregistrer_pdf(contenu, "../../.git/cours.pdf", tmp_path) == chemin
    with pytest.raises(ValueError, match="PDF"):
        backend.enregistrer_pdf(b"ceci n'est pas un PDF", "x.pdf", tmp_path)
    with pytest.raises(ValueError):
        backend.enregistrer_pdf(b"", "x.pdf", tmp_path)


def test_import_image_conserve_le_fond_blanc(tmp_path):
    image = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    tampon = BytesIO()
    image.save(tampon, format="PNG")
    chemin = backend.enregistrer_image(tampon.getvalue(), tmp_path)
    with Image.open(chemin) as resultat:
        assert resultat.mode == "RGB"
        assert resultat.getpixel((0, 0)) == (255, 255, 255)
    with pytest.raises(ValueError):
        backend.enregistrer_image(b"image invalide", tmp_path)


@pytest.fixture
def interface(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    st.cache_resource.clear()
    image_path = tmp_path / "figure.png"
    Image.new("RGB", (30, 20), "white").save(image_path)
    source = {"id": "texte-1", "pdf": "test.pdf", "type": "texte", "page": 1,
              "contenu": "L'attention combine les représentations.", "similarite": 0.8,
              "source": "page1.txt"}

    class MoteurTest:
        def __init__(self):
            self.appels = []

        def preparer(self, pdf):
            return {"manifeste": {"document_id": "a" * 64, "pdf": "test.pdf",
                    "pages": [{"page": 1, "texte": "Texte de la page 1"}],
                    "images": [{"page": 1, "source": str(image_path), "origine": "image_integree"}],
                    "avertissements": []},
                    "bilan": {"elements": 2}, "textes": 1, "images": 1, "pdf_path": str(pdf)}

        def rechercher(self, document_id, question, image, k, type_resultat):
            self.appels.append({"question": question, "image": image, "k": k,
                                "type_resultat": type_resultat, "document_id": document_id})
            return [source]

        def interroger(self, document_id, **kwargs):
            self.appels.append(kwargs)
            return {"reponse": "Réponse de test [test.pdf, p. 1].", "sources": [source],
                    "trace": [{"etape": "Action", "outil": "retriever_tool"}]}

    moteur = MoteurTest()
    monkeypatch.setattr(backend, "creer_moteur", lambda: moteur)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    yield app, moteur
    st.cache_resource.clear()


def bouton(app, texte):
    return next(b for b in app.button if b.label == texte)


def preparer(app):
    app.button(key="preparer").click().run()
    assert not app.exception
    return app


def test_interface_accueil_et_recherche_texte(interface):
    app, moteur = interface
    assert not app.exception
    assert bouton(app, "Rechercher dans le PDF").disabled
    preparer(app)
    assert [m.value for m in app.metric] == ["1", "1", "1"]
    app.text_area(key="question").set_value("multi-head attention")
    bouton(app, "Rechercher dans le PDF").click().run()
    assert not app.exception
    assert moteur.appels[0]["question"] == "multi-head attention"
    assert moteur.appels[0]["document_id"] == "a" * 64
    assert app.session_state["historique"][0]["sources"][0]["page"] == 1


def test_requete_vide_et_image_seule(interface):
    app, moteur = interface
    preparer(app)
    bouton(app, "Rechercher dans le PDF").click().run()
    assert any("question ou joins une image" in w.value for w in app.warning)
    assert not moteur.appels
    app.selectbox(key="figure_" + "a" * 64).select(1).run()
    bouton(app, "Rechercher dans le PDF").click().run()
    assert not app.exception
    assert moteur.appels[-1]["image"].endswith("figure.png")
    assert moteur.appels[-1]["question"] == ""


def test_agent_exige_cle_et_ne_modifie_pas_environnement(interface):
    app, moteur = interface
    preparer(app)
    app.radio(key="mode").set_value("Agent IA").run()
    assert bouton(app, "Interroger l'agent").disabled
    app.text_input(key="api_key").set_value("cle-factice-pour-test").run()
    assert not bouton(app, "Interroger l'agent").disabled
    app.text_area(key="question").set_value("Explique le document")
    bouton(app, "Interroger l'agent").click().run()
    assert not app.exception
    assert moteur.appels[-1]["api_key"] == "cle-factice-pour-test"
    assert "OPENAI_API_KEY" not in os.environ
    assert any("Réponse de test" in m.value for m in app.markdown)
    assert "api_key" not in app.session_state["historique"][0]


def test_changement_document_efface_contexte_et_historique(interface):
    app, _ = interface
    preparer(app)
    app.session_state["historique"] = [{"question": "ancienne question"}]
    app.radio(key="source_pdf").set_value("Importer un PDF").run()
    assert not app.exception
    assert app.session_state["document"] is None
    assert app.session_state["historique"] == []
    assert bouton(app, "Rechercher dans le PDF").disabled
