"""Parcours Streamlit avec le véritable CLIP, sans double et sans clé API."""
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from streamlit.testing.v1 import AppTest

racine = Path(__file__).resolve().parents[1]
os.chdir(racine)
app = AppTest.from_file(str(racine / "app.py"), default_timeout=180).run()
assert not app.exception
app.button(key="preparer").click().run(timeout=180)
assert not app.exception
document = app.session_state["document"]
assert document and document["images"] > 0
print("Vrai index prêt.", flush=True)
cle_figure = "figure_" + document["manifeste"]["document_id"]
app.selectbox(key=cle_figure).select(1).run()
soumettre = next(b for b in app.button if b.label == "Rechercher dans le PDF")
soumettre.click().run(timeout=90)
assert not app.exception
echange = app.session_state["historique"][-1]
premier = echange["sources"][0]
assert premier["type"] == "image"
assert Path(premier["source"]) == Path(echange["image"])
assert premier["similarite"] > 0.99
print(f"Image seule : page {premier['page']}, cosinus {premier['similarite']:.6f}.", flush=True)
bilan = {"url_locale": "http://127.0.0.1:8502", "pages": len(document["manifeste"]["pages"]),
         "fragments_texte": document["textes"], "images": document["images"],
         "recherche_image_ui_clip_reel": True, "cosinus_image_connue": premier["similarite"],
         "generation_llm_reelle_executee": False}
preuve_navigateur = racine / ".cache/qa_streamlit/verification.json"
if preuve_navigateur.exists():
    bilan["verification_navigateur"] = json.loads(preuve_navigateur.read_text(encoding="utf-8"))
preuve_tests = racine / ".cache/pytest-streamlit.xml"
if preuve_tests.exists():
    suite = ET.parse(preuve_tests).getroot().find("testsuite")
    assert int(suite.attrib["failures"]) == int(suite.attrib["errors"]) == 0
    bilan["tests_automatiques_reussis"] = int(suite.attrib["tests"])
(racine / "validation_interface.json").write_text(json.dumps(bilan, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(bilan, ensure_ascii=False, indent=2), flush=True)
