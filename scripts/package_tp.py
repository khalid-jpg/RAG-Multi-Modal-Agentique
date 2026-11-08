"""Archive de remise : sources, données fournies et résultats, sans caches ni clé."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

racine = Path(__file__).resolve().parents[1]
fichiers = [
    "README.md", "requirements.txt", "requirements-lock.txt", "pytest.ini",
    "TP_RAG_multimodal_agentique.ipynb", "TP_RAG_multimodal_agentique.html",
    "rag_multimodal.py", "agent_rag.py", "validation.json", "validation_retrieval.json",
    "tests/test_tp.py", "scripts/build_notebook.py", "scripts/execute_notebook.py",
    "scripts/inspect_pdf.py", "scripts/package_tp.py",
    "app.py", "streamlit_backend.py", "lancer_interface.bat",
    ".streamlit/config.toml", "tests/test_interface.py",
    "scripts/smoke_interface.py", "validation_interface.json",
    ".gitignore", "scripts/build_test_report.py",
]
selection = [racine / f for f in fichiers]
selection.extend(p for p in (racine / "docs").rglob("*") if p.is_file())
selection.extend((racine / "pdfs").glob("*.pdf"))
selection.extend(p for p in (racine / "resultats_pdf").rglob("*") if p.is_file())
destination = racine / "TP_RAG_multimodal_agentique_complet.zip"
with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
    for fichier in selection:
        archive.write(fichier, fichier.relative_to(racine).as_posix())
with ZipFile(destination) as archive:
    assert archive.testzip() is None
    assert len(archive.namelist()) == len(selection)
print(f"Archive vérifiée : {destination.name} — {len(selection)} fichiers — {destination.stat().st_size:,} octets")
