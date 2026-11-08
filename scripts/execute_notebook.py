"""Exécute dans le noyau Python courant et exporte un aperçu HTML autonome."""
import os
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager

racine = Path(__file__).resolve().parents[1]
os.chdir(racine)
for nom, sous_dossier in [("JUPYTER_RUNTIME_DIR", "jupyter"), ("IPYTHONDIR", "ipython"),
                          ("TEMP", "tmp"), ("TMP", "tmp")]:
    dossier = racine / ".cache" / sous_dossier
    dossier.mkdir(parents=True, exist_ok=True)
    os.environ[nom] = str(dossier)
chemin = racine / "TP_RAG_multimodal_agentique.ipynb"
notebook = nbformat.read(chemin, as_version=4)
print("Initialisation du noyau Jupyter…", flush=True)
km = KernelManager(kernel_name="python3")
km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]

def progression(cell, cell_index, **kwargs):
    if cell.cell_type == "code":
        print(f"Cellule {cell_index + 1}/{len(notebook.cells)} : {cell.source.splitlines()[0][:75]}", flush=True)

client = NotebookClient(notebook, km=km, timeout=600,
                        resources={"metadata": {"path": str(racine)}},
                        on_cell_start=progression)
try:
    client.execute()
finally:
    nbformat.write(notebook, chemin)
    html, _ = HTMLExporter(embed_images=True).from_notebook_node(notebook)
    (racine / "TP_RAG_multimodal_agentique.html").write_text(html, encoding="utf-8")
print("Notebook exécuté et aperçu HTML exporté.", flush=True)
