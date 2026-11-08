"""Produit un notebook autonome à partir du code testé (pas de JSON manuel)."""
from pathlib import Path
import re
import textwrap
import json
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(texte):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(texte).strip()))


def code(texte):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(texte).strip()))


def sections(fichier):
    return re.split(r"^# %%[^\n]*\n", (ROOT / fichier).read_text(encoding="utf-8"), flags=re.M)


rag = sections("rag_multimodal.py")
agent = sections("agent_rag.py")
md("""
# TP — RAG multimodal et agentique
### Application à *Attention Is All You Need*

Ce notebook réalise les **sept étapes** de l'énoncé : extraction du PDF,
embeddings CLIP, index ChromaDB, recherche texte/image, outil documentaire,
agent ReAct avec LangGraph, puis tests et analyse.

**Livrable :** code complet, démonstrations et sorties conservées. Les définitions
sont incluses dans ce notebook ; les fichiers `.py` fournissent la même solution
réutilisable. Le notebook de cours fourni sert de référence pédagogique.

**Exécution :** Python 3.11+, PDF dans `pdfs/attention is all you need.pdf`.
CLIP et la recherche tournent localement sur CPU ou CUDA. Le premier chargement
télécharge les poids CLIP. La génération réelle de l'agent nécessite
`OPENAI_API_KEY`. Sans clé, la validation du graphe utilise un **double de test
explicitement simulé** ; aucune réponse générative n'est présentée comme exécutée.
""")
validation_path = ROOT / "validation.json"
if validation_path.exists():
    bilan = json.loads(validation_path.read_text(encoding="utf-8"))
    md(f"""
    **Résultats de l'exécution locale :** {bilan['pages']} pages traitées,
    {bilan['fragments_texte']} fragments texte et {bilan['images']} images indexés
    ({bilan['elements_indexes']} éléments au total). Les quatre questions annotées
    retrouvent une page attendue dans le top-5. La réindexation est sans doublons.
    Le graphe a été validé avec un scénario simulé ; les essais du vrai LLM
    restent à exécuter avec une clé API.
    """)
md("""
## Installation et paramètres

Dans un terminal : `python -m pip install -r requirements.txt`, puis ouvrir ce
notebook avec le noyau de cet environnement. Sous Windows, l'environnement livré
se trouve dans `.venv/Scripts/python.exe`. `requirements-lock.txt` enregistre les
versions exactes de la validation. Pour les roues CPU PyTorch, voir le README.

Les numéros de page des sources sont les numéros physiques du PDF, à partir de 1.
Le PDF est en anglais : l'agent reformule ses recherches en anglais et répond en français.
""")
code("""
import os
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore", message="IProgress not found.*")
os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
PDF_PATH = Path("pdfs/attention is all you need.pdf")
OUTPUT_DIR = Path("resultats_pdf")
EXECUTER_LLM = bool(os.getenv("OPENAI_API_KEY"))
print("Python :", sys.version.split()[0])
print("PDF présent :", PDF_PATH.is_file())
print("Génération LLM activée :", EXECUTER_LLM)
""")
code(rag[0])
md("""
## Étape 1 — Extraire le contenu multimodal

Chaque page produit un fichier texte. Les images intégrées sont sauvegardées avec
leur numéro de page ; les grands dessins vectoriels sont également rendus en PNG.
La transparence est composée sur fond blanc pour conserver la lisibilité des schémas.
Un manifeste lie chaque objet au SHA-256 intégral du PDF. Aucun OCR n'est exécuté :
une page sans texte est signalée. La détection de zones vectorielles reste heuristique.
""")
code(rag[1])
code("""
manifeste = extraire_pdf(PDF_PATH, OUTPUT_DIR)
document_id = manifeste["document_id"]
print("SHA-256 :", document_id)
print("Pages :", len(manifeste["pages"]))
print("Images intégrées :", sum(i["origine"] == "image_integree" for i in manifeste["images"]))
print("Zones vectorielles rendues :", sum(i["origine"] == "rendu_vectoriel" for i in manifeste["images"]))
print("Avertissements :", manifeste["avertissements"] or "aucun")
print("\\nExtrait de la première page :\\n", manifeste["pages"][0]["texte"][:850])
""")
code("""
from IPython.display import display, Markdown, HTML
for source in manifeste["images"]:
    print(f"Page {source['page']} — {source['origine']}")
    apercu = charger_rgb(source["source"])
    apercu.thumbnail((600, 430))
    display(apercu)
IMAGE_REQUETE = next(i["source"] for i in manifeste["images"] if i["page"] == 3)
""")
md("""
## Étape 2 — Embeddings CLIP dans un espace commun

CLIP ViT-B-32 encode textes et images en **512 dimensions**. Les textes sont
découpés à 60 tokens BPE avec un recouvrement maximal de 10 tokens, en conservant
le texte original. Les 2 tokens spéciaux laissent la séquence sous la limite de 77.
Les requêtes longues sont découpées puis moyennées au lieu d'être tronquées.

Tous les vecteurs sont normalisés : $\\hat v=v/\\|v\\|_2$.
Pour une requête mixte, $q=\\operatorname{norm}(0.5q_{texte}+0.5q_{image})$.
La recherche hybride de ce TP combine les modalités texte et image dans CLIP.
""")
code(rag[2])
code("""
encodeur = EncodeurCLIP()
elements = preparer_elements(manifeste, encodeur.bpe)
elements_texte = [e for e in elements if e.metadata["type"] == "texte"]
elements_image = [e for e in elements if e.metadata["type"] == "image"]
vecteur_texte = encodeur.encoder_textes([elements_texte[0].contenu])[0]
vecteur_image = encodeur.encoder_images([IMAGE_REQUETE])[0]
assert vecteur_texte.shape == vecteur_image.shape == (512,)
assert np.isclose(np.linalg.norm(vecteur_texte), 1, atol=1e-5)
assert np.isclose(np.linalg.norm(vecteur_image), 1, atol=1e-5)
assert all(len(encodeur.bpe.encode(e.contenu)) <= 60 for e in elements_texte)
print("Appareil :", encodeur.device)
print("Fragments texte :", len(elements_texte), "| Images :", len(elements_image))
print("Dimensions :", vecteur_texte.shape, vecteur_image.shape)
print("Normes :", round(float(np.linalg.norm(vecteur_texte)), 6),
      round(float(np.linalg.norm(vecteur_image)), 6))
""")
md("""
## Étape 3 — Indexer et stocker dans ChromaDB

La collection persistante utilise la **distance cosinus**. Chaque objet conserve
son type, sa page, le nom du PDF, son SHA-256 et son chemin source. Les identifiants
sont déterministes. Une réexécution réutilise les embeddings et supprime les
fragments devenus obsolètes après les ajouts réussis. L'identité de collection
inclut le modèle et la configuration d'extraction/découpage.

La classe ci-dessous contient également la méthode de recherche utilisée à l'étape 4.
""")
code(rag[3])
code("""
index = IndexMultimodal(encodeur)
bilan_index = index.indexer(elements)
bilan_reindex = index.indexer(elements)
assert bilan_reindex["nouveaux"] == 0
assert bilan_reindex["total_collection"] == len(elements)
print("Indexation :", bilan_index)
print("Deuxième passage (absence de doublons) :", bilan_reindex)
""")
md("""
## Étape 4 — Recherche multimodale et affichage du top-k

Pour des vecteurs normalisés, $\\cos(q,v)=q^Tv$. Chroma renvoie une distance
$d=1-\\cos(q,v)$ ; le score affiché est donc $1-d$. **Ce score n'est pas une
probabilité de vérité.** Le filtre porte sur le type de contenu et le PDF.
La valeur de `k` est bornée par le nombre de résultats disponibles après filtrage.
""")
code("""
resultats_texte = index.rechercher(
    texte="Why use multiple attention heads with different representation subspaces?",
    type_resultat="texte", k=3, document_id=document_id)
afficher_resultats(resultats_texte)
""")
md("### Texte → images : retrouver les schémas d'attention")
code("""
resultats_figures = index.rechercher(
    texte="Scaled dot-product attention and multi-head attention diagram",
    type_resultat="image", k=3, document_id=document_id)
afficher_resultats(resultats_figures)
""")
md("### Image → images : contrôler la correspondance visuelle")
code("""
resultats_visuels = index.rechercher(image=IMAGE_REQUETE, type_resultat="image", k=2,
                                    document_id=document_id)
assert Path(resultats_visuels[0]["source"]) == Path(IMAGE_REQUETE)
assert resultats_visuels[0]["similarite"] > 0.99
afficher_resultats(resultats_visuels[:1])
print("Cette requête réutilise une image indexée : c'est un contrôle technique,")
print("pas une mesure de généralisation à des images nouvelles.")
""")
md("### Texte + image → passages : recherche hybride")
code("""
resultats_hybrides = index.rechercher(
    texte="Transformer encoder decoder residual connections and positional encoding",
    image=IMAGE_REQUETE, type_resultat="texte", k=3, document_id=document_id)
afficher_resultats(resultats_hybrides)
""")
md("""
## Étape 5 — Encapsuler le retriever dans un outil agentique

`retriever_tool` expose un schéma Pydantic : requête, modalité, type de résultat
et top-k. L'image d'entrée est liée par l'application ; l'agent ne choisit pas de
chemin local. Le contenu JSON contient les extraits et les références stables.
L'artefact conserve les métadonnées nécessaires à l'affichage et à la vision.
L'outil récupère les preuves ; la synthèse appartient au LLM de l'agent.
""")
# Le notebook contient déjà charger_rgb ; supprimer l'import du module compagnon.
code(agent[0].replace("from rag_multimodal import charger_rgb\n", ""))
code(agent[1])
code("""
retriever_tool = creer_retriever_tool(index, document_id, IMAGE_REQUETE)
print("Nom :", retriever_tool.name)
print(json.dumps(retriever_tool.args_schema.model_json_schema()["properties"],
                 ensure_ascii=False, indent=2))
observation = retriever_tool.invoke({"type": "tool_call", "id": "demo-1",
    "name": "retriever_tool", "args": {
        "requete": "multi-head attention", "modalite": "texte",
        "type_resultat": "texte", "k": 2}})
print("Sources retournées :", len(observation.artifact))
print("Première source :", observation.artifact[0]["pdf"],
      "p.", observation.artifact[0]["page"])
""")
md("""
## Étape 6 — Construire l'agent ReAct avec LangGraph

**MIND** : le LLM choisit un outil ou produit sa réponse. **BODY** : la fonction
de recherche exécute l'action et renvoie une observation. Le graphe compilé boucle
jusqu'à la synthèse : `START → mind → body → mind → … → END`.

Les figures retournées sont jointes au prochain contexte du LLM, à raison de trois
au maximum. Le budget est imposé par le code, même si le modèle propose plusieurs
appels dans un seul message. À épuisement, le modèle est invoqué sans outils.
Le journal expose les actions et observations, sans demander de raisonnement interne.

La clé se lit dans l'environnement. Une saisie locale possible est
`os.environ["OPENAI_API_KEY"] = getpass.getpass("Clé API : ")` ; ne jamais écrire
une clé littérale dans une cellule. Le modèle par défaut, `gpt-4.1-mini`, est
configurable via `OPENAI_MODEL` et doit accepter vision et appels d'outils.
""")
code(agent[2])
code("""
# Construire et inspecter le graphe ne déclenche aucun appel distant.
if EXECUTER_LLM:
    agent = creer_agent(creer_llm(), retriever_tool)
    print(agent.get_graph().draw_mermaid())
else:
    print("Graphe : START → mind → body → mind → END")
    print("LLM réel non exécuté : OPENAI_API_KEY absente.")
""")
md("""
## Étape 7 — Tests et validation

### Contrôles exécutés sans clé

On vérifie l'extraction, les deux modalités, la normalisation, les références de
pages, le classement d'une image connue et l'idempotence. L'évaluation suivante
mesure le retour d'au moins une page attendue dans le top-5 de quatre questions
textuelles. Les pages attendues sont des annotations manuelles du PDF ; ce petit
jeu ne mesure ni la qualité des réponses du LLM ni la généralisation.
""")
code("""
cas_retrieval = [
    ("Attention multi-têtes", "multiple attention heads different representation subspaces", {4, 5}),
    ("Division par racine de dk", "scaled dot product attention square root key dimension softmax gradients", {4}),
    ("Encodage positionnel", "sine cosine positional encoding frequencies token positions", {6}),
    ("Architecture encodeur-décodeur", "encoder decoder stack six layers residual connections layer normalization", {3}),
]
mesures = []
for nom, requete, pages_attendues in cas_retrieval:
    trouves = index.rechercher(texte=requete, type_resultat="texte", k=5,
                              document_id=document_id)
    pages_trouvees = sorted({r["page"] for r in trouves})
    succes = bool(pages_attendues & set(pages_trouvees))
    mesures.append({"test": nom, "pages_attendues": sorted(pages_attendues),
                    "pages_top5": pages_trouvees, "hit@5": succes})
    print(nom, "| pages :", pages_trouvees, "| hit@5 :", succes)
taux_hit = sum(m["hit@5"] for m in mesures) / len(mesures)
print(f"\\nHit@5 (4 questions annotées) : {taux_hit:.0%}")
_ = Path("validation_retrieval.json").write_text(
    json.dumps(mesures, ensure_ascii=False, indent=2), encoding="utf-8")
""")
md("""
### Valider la boucle de l'agent sans API — scénario simulé

Le double ci-dessous programme deux actions de recherche sur le **vrai index
CLIP**, puis une réponse de test. Il vérifie le câblage du graphe et les observations.
Il **ne valide pas** les décisions autonomes, la compréhension visuelle ou la qualité
d'un LLM. La section suivante contient les essais génératifs réels.
""")
code("""
class ModeleScenario:
    def bind_tools(self, outils):
        return self

    def invoke(self, messages):
        observations = [m for m in messages if isinstance(m, ToolMessage)]
        if len(observations) < 2:
            requete = ("multi-head attention representation subspaces" if not observations
                       else "sinusoidal positional encoding")
            return AIMessage(content="", tool_calls=[{
                "name": "retriever_tool", "args": {
                    "requete": requete, "type_resultat": "texte", "k": 3},
                "id": f"simulation-{len(observations)}", "type": "tool_call"}])
        return AIMessage(content="Scénario simulé terminé : deux observations reçues. Aucune synthèse LLM.")

simulation = interroger_agent(index, "Compare attention et positions", document_id=document_id,
                              llm=ModeleScenario())
assert len(simulation["appels"]) == 2
assert simulation["sources"]
for evenement in trace_execution(simulation):
    print(evenement)
""")
md("""
### Batterie de tests génératifs réels — clé API requise

Les tests couvrent explication, recherche de figures, entrée texte + image,
recherches multiples et information absente. La dernière question demande une
mesure GPT-4 absente du PDF ; la réponse doit signaler l'absence de preuve.
Pour chaque exécution, lire les actions, vérifier les pages citées et contrôler
la réponse avec les extraits et images. Les ressemblances CLIP ne suffisent pas
à prouver une affirmation.
""")
code("""
tests_agent = [
    {"nom": "Attention multi-têtes", "question": "Pourquoi plusieurs têtes d'attention sont-elles utilisées ?"},
    {"nom": "Schéma et texte", "question": "Retrouve les schémas de la figure 2 et explique les opérations avec les passages du PDF."},
    {"nom": "Entrée multimodale", "question": "Explique cette architecture avec des preuves textuelles et visuelles du PDF.", "image": IMAGE_REQUETE},
    {"nom": "Recherche en plusieurs étapes", "question": "Recherche séparément attention multi-têtes et encodage positionnel, puis compare leurs rôles avec les sources."},
    {"nom": "Information absente", "question": "Quel score MMLU de GPT-4 est rapporté dans cet article ?"},
]
sorties_agent = []
if EXECUTER_LLM:
    llm = creer_llm()
    for cas in tests_agent:
        print("\\nTEST :", cas["nom"])
        sortie = interroger_agent(index, cas["question"], image=cas.get("image"),
                                  document_id=document_id, llm=llm)
        sorties_agent.append(sortie)
        display(Markdown(str(sortie["reponse"])))
        for evenement in trace_execution(sortie):
            if evenement["etape"] != "Réponse finale":
                print(evenement)
        afficher_resultats([s for s in sortie["sources"] if s["type"] == "image"][:2])
else:
    print("5 scénarios génératifs prêts. Non exécutés : OPENAI_API_KEY absente.")
""")
md("""
### Analyse du comportement et limites

- La trace simulée contrôle la transition **Action → Observation → nouvelle Action → réponse**.
  Seules les exécutions avec un vrai LLM permettent d'évaluer la décision d'appeler
  l'outil et la pertinence des reformulations.
- La recherche CLIP utilise un espace commun ; elle peut confondre des schémas
  proches et privilégier l'apparence. Elle est moins adaptée aux équations et aux
  passages techniques qu'un encodeur textuel spécialisé. Le modèle choisi est
  surtout anglophone ; les requêtes françaises peuvent donner un classement différent.
- Les fragments courts respectent la limite CLIP mais peuvent couper une explication.
  L'agent peut chercher plusieurs fois ; un élargissement aux fragments voisins
  serait une amélioration possible.
- Le top-k global `tous` ne garantit pas la présence des deux types. Pour une
  question exigeant une figure et du texte, demander deux recherches filtrées.
- Aucun seuil de similarité arbitraire ne garantit l'absence d'hallucination.
  L'agent doit distinguer preuves pertinentes et informations non attestées.
- Les citations sont demandées au modèle ; leur fidélité exige un contrôle sur les
  sources retournées. La validation sans clé ne mesure pas leur exactitude.

### Références techniques

Source documentaire : le PDF fourni *Attention Is All You Need*, Vaswani et al.
Les dépendances suivent les API documentées par
[OpenCLIP](https://github.com/mlfoundations/open_clip),
[ChromaDB](https://docs.trychroma.com/docs/collections/configure),
[PyMuPDF](https://pymupdf.readthedocs.io/en/latest/page.html) et
[LangGraph](https://docs.langchain.com/oss/python/langgraph/workflows-agents).
Le modèle génératif utilise la vision et les appels d'outils documentés dans
[OpenAI Docs](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
""")
code("""
resume_validation = {
    "pages": len(manifeste["pages"]),
    "images": len(manifeste["images"]),
    "fragments_texte": len(elements_texte),
    "elements_indexes": index.collection.count(),
    "reindexation_sans_doublons": bilan_reindex["nouveaux"] == 0,
    "dimension_clip": len(vecteur_texte),
    "hit_at_5": taux_hit,
    "graphe_scenario_simule": len(simulation["appels"]) == 2,
    "generation_llm_executee": bool(sorties_agent),
}
print(json.dumps(resume_validation, ensure_ascii=False, indent=2))
_ = Path("validation.json").write_text(json.dumps(resume_validation, ensure_ascii=False, indent=2), encoding="utf-8")
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3 (TP RAG)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
nbf.validate(nb)
destination = ROOT / "TP_RAG_multimodal_agentique.ipynb"
nbf.write(nb, destination)
print(f"Notebook généré : {len(cells)} cellules — {destination}")
