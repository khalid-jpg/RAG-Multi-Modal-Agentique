"""Créer le rapport partageable depuis un véritable résultat JUnit de pytest."""
from __future__ import annotations

import argparse
from datetime import datetime
from html import escape
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
LIBELLES = {
    "test_extraction_pdf_et_vecteurs": "Extraction du texte et des dessins vectoriels",
    "test_decoupage_clip_sans_perte": "Découpage conforme à la limite de tokens CLIP",
    "test_normalisation": "Normalisation et rejet des vecteurs invalides",
    "test_reindexation_et_nettoyage": "Réindexation sans doublons et nettoyage",
    "test_recherche_cosinus_filtres_et_persistance": "Cosinus, filtres et persistance de l’index",
    "test_outil_retourne_sources_et_artifact": "Outil : sources et métadonnées retournées",
    "test_agent_action_observation_reformulation": "Graphe : actions, observations et nouvel appel",
    "test_budget_impose_meme_avec_appels_multiples": "Budget imposé malgré les appels multiples",
    "test_image_absente_et_schema": "Validation du schéma et de l’image requise",
    "test_requete_mixte_utilise_deux_encodeurs": "Fusion des modalités texte et image",
    "test_import_pdf_verifie_et_confine_le_chemin": "Import PDF : contenu valide et chemin borné",
    "test_import_image_conserve_le_fond_blanc": "Image importée : transparence sur fond blanc",
    "test_interface_accueil_et_recherche_texte": "Interface : préparation et recherche textuelle",
    "test_requete_vide_et_image_seule": "Interface : requête vide et recherche par image",
    "test_agent_exige_cle_et_ne_modifie_pas_environnement": "Agent : clé requise et propre à la session",
    "test_changement_document_efface_contexte_et_historique": "Nouveau PDF : contexte et historique effacés",
}


def generer(junit: Path, destination: Path):
    arbre = ET.parse(junit)
    suites = list(arbre.getroot().iter("testsuite"))
    if not suites:
        raise ValueError("Le fichier JUnit ne contient pas de suite de tests.")
    cas = []
    for suite in suites:
        for test in suite.findall("testcase"):
            statut = ("échec" if test.find("failure") is not None
                      else "erreur" if test.find("error") is not None
                      else "ignoré" if test.find("skipped") is not None else "réussi")
            cas.append({"module": test.get("classname", ""), "nom": test.get("name", ""),
                        "duree_secondes": float(test.get("time", "0")), "statut": statut})
    # L'heure vient de l'exécution JUnit ; afficher la date dans le fuseau du projet.
    instant = datetime.fromisoformat(suites[0].get("timestamp", datetime.now().isoformat()))
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=ZoneInfo("Africa/Casablanca"))
    instant = instant.astimezone(ZoneInfo("Africa/Casablanca"))
    bilan = {"date_execution": instant.isoformat(), "fuseau": "Africa/Casablanca",
             "commande": "python -m pytest -q --junitxml=.cache/pytest-readme.xml",
             "total": len(cas), "reussis": sum(t["statut"] == "réussi" for t in cas),
             "echecs": sum(t["statut"] == "échec" for t in cas),
             "erreurs": sum(t["statut"] == "erreur" for t in cas),
             "ignores": sum(t["statut"] == "ignoré" for t in cas),
             "duree_secondes": sum(float(s.get("time", "0")) for s in suites), "tests": cas}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "resultats.json").write_text(json.dumps(bilan, ensure_ascii=False, indent=2), encoding="utf-8")
    groupes = []
    for module in sorted({t["module"] for t in cas}, reverse=True):
        selection = [t for t in cas if t["module"] == module]
        titre = "Moteur RAG et agent" if module.endswith("test_tp") else "Interface Streamlit et imports"
        lignes = []
        for test in selection:
            valide = test["statut"] == "réussi"
            couleur = "pass" if valide else "fail"
            libelle = LIBELLES.get(test["nom"], test["nom"])
            lignes.append(f'<li><span class="mark {couleur}">{"✓" if valide else "!"}</span>'
                          f'<span title="{escape(test["nom"])}">{escape(libelle)}</span>'
                          f'<small>{test["duree_secondes"]:.2f} s</small></li>')
        groupes.append(f'<section class="panel"><h2>{escape(titre)} <span>{len(selection)} tests</span></h2>'
                       f'<ul>{"".join(lignes)}</ul></section>')
    tous_valides = bilan["reussis"] == bilan["total"] and bilan["total"] > 0
    titre = f'{bilan["reussis"]} tests réussis' if tous_valides else "Validation à vérifier"
    contenu = """<!doctype html>
<html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Atelier RAG — Rapport des tests</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f1f5f6;color:#183039;font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1240px;margin:0 auto;padding:42px 36px}header .label{color:#147d72;font-size:12px;letter-spacing:2px;font-weight:750}
h1{font-size:42px;letter-spacing:-1.5px;line-height:1.2;margin:10px 0}header p{color:#5b7079;margin:10px 0 24px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px}.stat{background:white;border:1px solid #dce6e8;border-radius:14px;padding:17px 20px}
.stat b{display:block;font-size:30px;line-height:1.4}.stat span{font-size:13px;color:#5b7079}.stat:first-child b{color:#147d72}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.panel{background:white;border:1px solid #dce6e8;border-radius:14px;overflow:hidden}
h2{font-size:18px;margin:0;padding:17px 20px;border-bottom:1px solid #e7edef}h2 span{float:right;color:#6a7e85;font-size:12px;font-weight:500}
ul{list-style:none;margin:0;padding:4px 20px 10px}li{display:flex;align-items:center;gap:10px;padding:10px 0;border-bottom:1px solid #edf1f2;font-size:13px}
li:last-child{border:0}li small{margin-left:auto;white-space:nowrap;color:#71858c;font-size:11px}.mark{flex-shrink:0;width:21px;height:21px;text-align:center;border-radius:50%;font-weight:800}
.pass{color:#147d72;background:#e5f5ef}.fail{color:#b34432;background:#fbe9e5}.note{margin-top:20px;border-left:4px solid #147d72;background:#e8f3f1;padding:13px 18px;font-size:13px}
footer{margin-top:20px;display:flex;justify-content:space-between;gap:16px;color:#6a7e85;font-size:12px}code{font:11px/1.4 ui-monospace,Consolas,monospace}
@media(max-width:850px){.grid{grid-template-columns:1fr}.stats{grid-template-columns:1fr 1fr}main{padding:25px 18px}h1{font-size:32px}footer{display:block}}
</style><main><header><div class="label">ATELIER RAG · VALIDATION LOCALE</div>
<h1>__TITRE__</h1><p>Rapport généré depuis une exécution réelle de pytest · __DATE__ · Afrique/Casablanca</p></header>
<div class="stats"><div class="stat"><b>__REUSSIS__/__TOTAL__</b><span>Tests réussis</span></div>
<div class="stat"><b>__ECHECS__</b><span>Échecs et erreurs</span></div>
<div class="stat"><b>__DUREE__ s</b><span>Durée de la suite</span></div>
<div class="stat"><b>Sans clé API</b><span>Aucun appel génératif payant</span></div></div>
<div class="grid">__GROUPES__</div>
<div class="note"><strong>Portée de la validation.</strong> Les tests de contrôle du graphe et de l’interface utilisent des doubles.
Les essais du vrai modèle CLIP sont documentés séparément dans <code>validation_interface.json</code>.
La génération par un vrai LLM reste non exécutée sans clé API.</div>
<footer><span>Source : rapport JUnit de pytest · <a href="resultats.json">Résultats JSON détaillés</a></span>
<code>python -m pytest -q</code></footer></main></html>
"""
    for cle, valeur in {"TITRE": escape(titre), "DATE": instant.strftime("%d/%m/%Y à %H:%M"),
                        "REUSSIS": bilan["reussis"], "TOTAL": bilan["total"],
                        "ECHECS": bilan["echecs"] + bilan["erreurs"],
                        "DUREE": f'{bilan["duree_secondes"]:.2f}', "GROUPES": "".join(groupes)}.items():
        contenu = contenu.replace(f"__{cle}__", str(valeur))
    (destination / "rapport-tests.html").write_text(contenu, encoding="utf-8")
    print(f"Rapport généré : {bilan['reussis']}/{bilan['total']} tests réussis, "
          f"{bilan['echecs']} échec(s), {bilan['erreurs']} erreur(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path, default=ROOT / ".cache/pytest-readme.xml")
    parser.add_argument("--sortie", type=Path, default=ROOT / "docs/test-results")
    arguments = parser.parse_args()
    generer(arguments.junit, arguments.sortie)
