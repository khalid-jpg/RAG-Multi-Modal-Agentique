"""Interface Streamlit du TP. Lancer avec : python -m streamlit run app.py."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import warnings

import streamlit as st

from streamlit_backend import ROOT, creer_moteur, enregistrer_image, enregistrer_pdf

os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
warnings.filterwarnings("ignore", message="IProgress not found.*")

st.set_page_config(page_title="Atelier RAG · Explorer un PDF", page_icon="📚",
                   layout="wide", initial_sidebar_state="expanded")


@st.cache_resource(show_spinner=False)
def moteur_local():
    return creer_moteur()


def afficher_sources(sources):
    if not sources:
        st.info("Aucune source retrouvée avec ces critères.")
        return
    for numero, source in enumerate(sources, 1):
        with st.container(border=True):
            st.markdown(f"**Source {numero} · Page {source['page']}**")
            type_source = "Passage de texte" if source["type"] == "texte" else "Image du document"
            st.caption(f"{source['pdf']} · {type_source} · Similarité {source['similarite']:.3f}")
            if source["type"] == "image":
                chemin = Path(source["source"])
                if chemin.is_file():
                    st.image(str(chemin), width=460)
                else:
                    st.warning("L'image source a été déplacée. Prépare à nouveau le document.")
            else:
                st.text(source["contenu"])


def afficher_echange(echange, numero):
    with st.chat_message("user"):
        st.write(echange["question"] or "Recherche à partir de l'image jointe")
        if echange.get("image"):
            st.image(echange["image"], width=260)
    with st.chat_message("assistant"):
        if echange["mode"] == "Agent IA":
            st.markdown(echange["reponse"])
        else:
            st.markdown("**Voici les sources les plus proches de ta recherche.**")
            st.caption("Recherche locale : les extraits sont affichés sans synthèse générée.")
        with st.expander(f"Sources retrouvées ({len(echange['sources'])})", expanded=True):
            afficher_sources(echange["sources"])
        if echange.get("trace"):
            with st.expander("Voir les actions de l'agent"):
                for evenement in echange["trace"]:
                    st.json(evenement)
        # Export uniquement du contenu utile ; aucune clé ni objet de modèle.
        export = {"question": echange["question"], "mode": echange["mode"],
                  "reponse": echange.get("reponse", ""), "sources": [
                      {k: v for k, v in s.items() if k != "source"} for s in echange["sources"]],
                  "actions": echange.get("trace", [])}
        st.download_button("Télécharger ces résultats", json.dumps(export, ensure_ascii=False, indent=2),
                           file_name=f"recherche_{numero + 1}.json", mime="application/json",
                           key=f"export_{numero}")


def main():
    st.html("""<style>
    .stMainBlockContainer {max-width:1200px;padding-top:2.3rem;padding-bottom:3rem}
    h1 {font-weight:750;letter-spacing:-1.2px}
    h2,h3 {letter-spacing:-.35px}
    [data-testid="stSidebar"] {border-right:1px solid #dce6e8}
    [data-testid="stMetric"] {background:white;border:1px solid #e0e9eb;border-radius:12px;padding:14px 18px}
    .eyebrow {color:#147d72;font-size:.76rem;letter-spacing:1.8px;font-weight:750;margin-bottom:8px}
    </style>""")
    st.session_state.setdefault("document", None)
    st.session_state.setdefault("historique", [])
    st.session_state.setdefault("question", "")

    with st.sidebar:
        st.markdown("## 📚 Atelier RAG")
        st.caption("Un PDF, ses textes et ses images.")
        st.divider()
        st.markdown("**1 · Choisir le document**")
        choix = st.radio("Source du PDF", ["PDF du TP", "Importer un PDF"], key="source_pdf",
                         label_visibility="collapsed")
        pdf = None
        if choix == "PDF du TP":
            disponibles = sorted((ROOT / "pdfs").glob("*.pdf"))
            if disponibles:
                pdf = st.selectbox("Document disponible", disponibles,
                                   format_func=lambda p: p.name, key="pdf_existant")
            else:
                st.info("Ajoute le PDF du TP dans le dossier pdfs ou importe un document.")
        else:
            fichier = st.file_uploader("Ton document PDF", type=["pdf"], key="pdf_upload")
            if fichier is not None:
                try:
                    pdf = enregistrer_pdf(fichier.getvalue(), fichier.name)
                except ValueError as erreur:
                    st.error(str(erreur))

        selection = str(pdf) if pdf else None
        if selection != st.session_state.get("selection_pdf"):
            st.session_state.selection_pdf = selection
            st.session_state.document = None
            st.session_state.historique = []

        if st.button("Préparer le document", type="primary", width="stretch",
                     disabled=pdf is None, key="preparer"):
            try:
                with st.status("Préparation du document…", expanded=True) as statut:
                    st.write("Chargement du modèle local…")
                    moteur = moteur_local()
                    st.write("Extraction des pages et des images, puis indexation…")
                    document = moteur.preparer(pdf)
                    st.session_state.document = document
                    st.session_state.historique = []
                    statut.update(label="Document prêt", state="complete", expanded=False)
            except Exception:
                logging.exception("Échec de préparation du PDF")
                st.error("La préparation a échoué. Vérifie que le PDF est lisible et que les poids CLIP sont disponibles ; le premier chargement nécessite Internet.")

        pret = st.session_state.document is not None
        if pret:
            st.success("Document prêt à interroger")
        else:
            st.caption("Le premier chargement du modèle peut prendre quelques minutes.")

        st.divider()
        st.markdown("**2 · Choisir le mode**")
        mode = st.radio("Mode de réponse", ["Recherche locale", "Agent IA"], key="mode")
        k, filtre, api_key, modele, max_appels = 5, "tous", None, None, 4
        if mode == "Recherche locale":
            st.caption("Retrouve les passages et les figures, sans clé API.")
            filtre = st.selectbox("Sources à retrouver", ["tous", "texte", "image"],
                                 format_func=lambda t: {"tous": "Textes et images", "texte": "Textes", "image": "Images"}[t],
                                 key="filtre")
            k = st.slider("Nombre de résultats", 1, 10, 5, key="nombre_resultats")
        else:
            st.caption("L'agent recherche dans le PDF et rédige une réponse avec ses sources.")
            api_key = st.text_input("Clé API OpenAI", type="password", key="api_key",
                                    help="Conservée dans cette session. Tu peux aussi définir OPENAI_API_KEY avant le lancement.")
            modele = st.text_input("Modèle", value=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), key="modele")
            max_appels = st.slider("Nombre maximal de recherches", 1, 6, 4, key="max_appels")
            if not (api_key or os.getenv("OPENAI_API_KEY")):
                st.info("Renseigne une clé pour activer l'agent, ou utilise la recherche locale.")
            st.caption("La question et les sources utiles sont envoyées au modèle. Les appels peuvent être facturés.")
        st.divider()
        if st.button("Effacer les résultats", width="stretch", key="effacer"):
            st.session_state.historique = []
        st.caption("TP · RAG multimodal et agentique")

    st.html('<div class="eyebrow">EXPLORATION DOCUMENTAIRE · TEXTE & IMAGE</div>')
    st.title("Explore ton PDF autrement.")
    st.markdown("Pose une question, joins une image et retrouve les passages qui comptent.")
    document = st.session_state.document
    if document:
        manifeste = document["manifeste"]
        st.caption(f"Document actif : {manifeste['pdf']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Pages", len(manifeste["pages"]))
        c2.metric("Passages indexés", document["textes"])
        c3.metric("Images et figures", document["images"])
        for avertissement in manifeste["avertissements"]:
            st.warning(avertissement)
    else:
        with st.container(border=True):
            st.subheader("Ton espace de recherche est prêt.")
            a, b, c = st.columns(3)
            a.markdown("**01 · Choisis un PDF**\n\nUtilise l'article du TP ou importe ton document.")
            b.markdown("**02 · Prépare le document**\n\nSes textes et ses images deviennent interrogeables.")
            c.markdown("**03 · Explore les sources**\n\nRecherche par mots, par image ou avec l'agent.")
        st.info("Pour commencer, clique sur « Préparer le document » dans la barre latérale.")

    recherche, figures, pages = st.tabs(["🔎 Interroger", "🖼️ Figures du document", "📄 Texte par page"])
    with recherche:
        st.subheader("Que veux-tu comprendre ?")
        exemples = st.columns(3)
        for colonne, titre, question in zip(exemples,
                ["L'attention multi-têtes", "L'architecture du Transformer", "L'encodage positionnel"],
                ["Why does the Transformer use multi-head attention?",
                 "Explain the Transformer encoder-decoder architecture.",
                 "How does sinusoidal positional encoding work?"]):
            if colonne.button(titre, width="stretch", disabled=not pret, key=f"exemple_{titre}"):
                st.session_state.question = question
        if mode == "Recherche locale":
            st.caption("Pour l'article du TP, les recherches en anglais donnent généralement de meilleurs résultats.")

        image = None
        image_valide = True
        with st.expander("Joindre une image à la recherche", expanded=False):
            fichier_image = st.file_uploader("Image de référence", type=["png", "jpg", "jpeg", "webp"],
                                             disabled=not pret, key="image_upload")
            images_document = document["manifeste"]["images"] if document else []
            figure = st.selectbox("Ou utiliser une figure du PDF", range(len(images_document) + 1),
                                  format_func=lambda i: "Aucune figure sélectionnée" if i == 0 else f"Figure {i} · Page {images_document[i - 1]['page']}",
                                  disabled=not pret, key=f"figure_{document['manifeste']['document_id'] if document else 'vide'}")
            try:
                if fichier_image is not None:
                    image = str(enregistrer_image(fichier_image.getvalue()))
                elif figure:
                    image = images_document[figure - 1]["source"]
            except ValueError as erreur:
                image_valide = False
                st.error(str(erreur))
            if image:
                st.image(image, width=300)
                st.caption("La recherche combine cette image avec ta question ; une image seule est aussi acceptée.")

        agent_disponible = mode != "Agent IA" or bool(api_key or os.getenv("OPENAI_API_KEY"))
        with st.form("recherche_form"):
            question = st.text_area("Ta question ou tes mots-clés", key="question", height=100,
                                    placeholder="Ex. : Pourquoi le Transformer utilise-t-il plusieurs têtes d'attention ?",
                                    disabled=not pret)
            envoyer = st.form_submit_button("Interroger l'agent" if mode == "Agent IA" else "Rechercher dans le PDF",
                                            type="primary", disabled=not pret or not agent_disponible or not image_valide,
                                            width="stretch")
        if envoyer:
            if not question.strip() and image is None:
                st.warning("Écris une question ou joins une image pour lancer la recherche.")
            else:
                try:
                    with st.spinner("L'agent consulte le PDF…" if mode == "Agent IA" else "Recherche des sources…"):
                        moteur = moteur_local()
                        identifiant = document["manifeste"]["document_id"]
                        echange = {"question": question.strip(), "image": image, "mode": mode}
                        if mode == "Agent IA":
                            echange.update(moteur.interroger(identifiant, question=question.strip(), image=image,
                                                            api_key=api_key, modele=modele, max_appels=max_appels))
                        else:
                            echange["sources"] = moteur.rechercher(identifiant, question, image, k, filtre)
                        st.session_state.historique.append(echange)
                        st.session_state.historique = st.session_state.historique[-10:]
                except Exception as erreur:
                    # Ne pas afficher les détails d'un fournisseur susceptibles de contenir une clé.
                    logging.error("Échec d'une requête documentaire (%s)", type(erreur).__name__)
                    st.error("La recherche n'a pas abouti. Vérifie le document et l'image ; en mode agent, vérifie aussi la clé API, le modèle et la connexion.")
        if st.session_state.historique:
            st.divider()
            for numero, echange in enumerate(reversed(st.session_state.historique)):
                afficher_echange(echange, numero)

    with figures:
        if not document:
            st.info("Prépare le document pour afficher ses figures.")
        elif not document["manifeste"]["images"]:
            st.info("Aucune image exploitable n'a été trouvée dans ce PDF.")
        else:
            st.caption("Images intégrées et grandes zones vectorielles détectées dans le document.")
            colonnes = st.columns(3)
            for numero, source in enumerate(document["manifeste"]["images"]):
                with colonnes[numero % 3]:
                    with st.container(border=True):
                        st.markdown(f"**Figure {numero + 1} · Page {source['page']}**")
                        st.image(source["source"], width="stretch")

    with pages:
        if not document:
            st.info("Prépare le document pour lire le texte extrait.")
        else:
            textes = document["manifeste"]["pages"]
            numero_page = st.selectbox("Page à consulter", range(1, len(textes) + 1),
                                      key=f"page_texte_{document['manifeste']['document_id']}")
            st.text(textes[numero_page - 1]["texte"] or "Aucun texte détecté sur cette page.")
            st.download_button("Télécharger le PDF actif", Path(document["pdf_path"]).read_bytes(),
                               file_name=document["manifeste"]["pdf"], mime="application/pdf")
    st.divider()
    st.caption("Les scores indiquent une ressemblance entre contenus. Vérifie les sources avant de retenir une affirmation.")


if __name__ == "__main__":
    main()
