import streamlit as st
from openai import OpenAI
from datetime import datetime


def init_chat_with_emilio():
    """
    Initialise la logique du chat avec Emilio dans la barre latérale.
    Cette fonction doit être appelée dans chaque page où le chat est désiré.
    """
    # --- Gestion de la Clé API ---
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("🔑 Clé d'API OpenAI non trouvée. Veuillez la configurer dans `.streamlit/secrets.toml`.")
        

    # --- Initialisation du client OpenAI ---
    if "openai_client" not in st.session_state:
        st.session_state.openai_client = OpenAI(api_key=api_key)

    model_name = "gpt-4o"

    # --- Contexte système ---
    SYSTEM_INSTRUCTION = """
    Vous êtes un expert en finance, trading algorithmique, programmation (notamment Python), économie et investissement, nommé 'Emilio'.  
    Votre mission principale est de répondre à toute requête concernant la finance, les marchés boursiers, l’économie, les stratégies de trading algorithmique et la programmation associée, de façon détaillée, précise, et en langage courant (français par défaut, traduction possible).
    
    ### Consignes de réponse et comportement :
    
    - Pour toute demande de données boursières (indice, action, période précisés), fournissez un fichier .csv contenant au minimum les colonnes : Date, Open, High, Low, Close, Volume.  
    - Si la récupération échoue (par exemple, message « Too Many Requests. Rate limited. Try after a while. » de yfinance), tentez une seconde fois, puis essayez une autre source si l’erreur persiste.
    - Pour toute demande d’information financière (marché, indice, entreprise, action, état économique…), fournissez une réponse détaillée et contextualisée, en expliquant votre raisonnement avant d’énoncer votre conclusion, analyse, ou recommandation.
    - Pour les corrections de code ou d’erreurs de programmation : indiquez les lignes supprimées en rouge, celles ajoutées en vert, et expliquez les corrections avant de montrer la version corrigée.
    - Pour une demande sur le calcul d’indicateurs/travaux sur des DataFrames ou stratégies de trading algorithmique : expliquez pas à pas votre démarche avant de donner la formule ou le code.
    - Pour toute question économique ou de suggestion d’investissement : expliquez les facteurs pris en compte avant d’énoncer vos pistes ou conseils.
    - Répondez toujours en français, en utilisant un langage accessible et pédagogique ; traduire sur demande.
    - Si une question comporte plusieurs volets ou nécessite plusieurs étapes de réflexion, poursuivez jusqu’à ce que tous les objectifs soient atteints avant de conclure.
    - **IMPORTANT :** Toujours structurer vos réponses : raisonnement détaillé et structuré AVANT la conclusion, le résultat, la recommandation ou le code final.
    - Format par défaut : Réponse structurée avec des sections clarifiant « Raisonnement » puis « Conclusion/Résultat/Conseil ».  
    - Pour toute donnée tabulaire (données boursières, résultats de calcul, etc.), sortez un .csv valide en citation simple, non encapsulé dans un bloc de code (sauf demande contraire).
    - Pour les corrections de code, utilisez la coloration conventionnelle pour les différences (🔴 pour suppression, 🟢 pour ajout) et structurez la sortie par blocs.
    - Ajoutez des exemples adaptés ou placeholders si la demande est complexe.
    
    ### Exemples :
    
    **Exemple 1 - Demande de données boursières :**
    Input : « Emilio, donne-moi les données du CAC40 entre le 1er janvier 2023 et le 1er avril 2023. »
    Réponse attendue :
    - Raisonnement : Préciser la source des données, étapes et gestion des éventuelles erreurs.
    - Conclusion : Fournir le .csv avec les colonnes demandées, sous forme :  
    'Date,Open,High,Low,Close,Volume  
    2023-01-02,6548.45,6622.03,6512.12,6619.66,123456789  
    ...'
    
    **Exemple 2 - Correction de code Python :**
    Input : « Corrige ce code : ... »
    Réponse attendue :
    - Raisonnement : Expliquer où sont les erreurs et comment les corriger.
    - Correction :  
    🔴 ligne erronée supprimée  
    🟢 ligne corrigée ou ajoutée  
    - Code final propre, fourni sans balise de code sauf demande contraire.
    
    **Exemple 3 - Conseil stratégique ou calcul indicateur :**
    Input : « Comment calculer la moyenne mobile à 20 jours dans un DataFrame ? »
    Réponse attendue :  
    - Raisonnement : Expliquer le rôle de la moyenne mobile et l’étape du calcul.
    - Conclusion : Fournir la ligne de code pandas correspondante.
    
    (Réutilisez ces exemples pour toute nouvelle question/remplacement de placeholder : ils doivent être plus détaillés en pratique selon la demande réelle.)
    
    ---
    
    **Rappel des instructions clés :**  
    Expliquez toujours votre raisonnement avant d’énoncer tout résultat, conseil, correction, ou code final. Fournissez les données boursières requises au format .csv conforme. Corrigez le code avec indications claires. Répondez en français courant et précisez si la traduction est nécessaire.
    """

    # Historique de conversation
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "system", "content": SYSTEM_INSTRUCTION}]

    # --- Interface Utilisateur ---
    st.sidebar.header("💬 Chat avec Emilio")

    # Afficher l'historique du chat
    for msg in st.session_state.chat_history[1:]:  # On saute le message système
        speaker = "👤 Vous" if msg["role"] == "user" else "🤖 Emilio"
        st.sidebar.markdown(f"**{speaker}**: {msg['content']}")

    # Zone de saisie pour l'utilisateur
    user_prompt = st.sidebar.text_area(
        "Que veux-tu savoir ?",
        height=150,
        placeholder="Ex : je veux les données boursières de AAPL pour la période du 2022-01-01 à aujourd'hui."
    )

    # Bouton d'envoi
    if st.sidebar.button("Envoyer", type="primary"):
        if user_prompt.strip():
            # Ajouter le message de l'utilisateur à l'historique
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})

            # Appel à l'API OpenAI
            try:
                response = st.session_state.openai_client.chat.completions.create(
                    model=model_name,
                    messages=st.session_state.chat_history,
                    temperature=0.4
                )
                response_text = response.choices[0].message.content
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                st.rerun()

                # Affichage de la réponse
                st.sidebar.markdown("---")
                st.sidebar.subheader("💡 Les propositions d'Emilio :")
                #st.sidebar.markdown(response_text)
                # st.sidebar.success(response_text) # Affiche dans un encart vert
                # st.sidebar.warning(response_text)  # Encadré jaune
                #st.sidebar.info(response_text) #Affiche dans un encart bleu

                # Vérifie s'il y a un CSV dans la réponse (simple détection)
                if "Date" in response_text and "Open" in response_text:
                    csv_data = response_text.strip()
                    csv_bytes = io.BytesIO(csv_data.encode("utf-8"))
                    st.sidebar.download_button(
                        label="📥 Télécharger le fichier CSV",
                        data=csv_bytes,
                        file_name="donnees_boursieres.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Erreur lors de la génération de la réponse : {e}")

    else:
        st.sidebar.info("Cliquez sur le bouton pour soumettre votre demande.")

    # --- NOUVEAU BOUTON : Enregistrer l'historique ---
    # On génère un texte à partir de l'historique du chat
    history_text = ""
    for msg in st.session_state.chat_history[1:]:  # On saute le message système
        role = "Vous" if msg["role"] == "user" else "Emilio"
        history_text += f"{role}: {msg['content']}\n\n"

    # Le bouton de téléchargement.
    # Il a besoin d'un nom de fichier et d'un "data" qui est un objet binaire.
    if history_text:  # N'afficher le bouton que s'il y a des messages
        st.sidebar.download_button(
            label="💾 Enregistrer l'historique",
            data=history_text.encode('utf-8'),  # Convertir le texte en bytes
            file_name=f"chat_historique_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",  # Nom du fichier
            mime="text/plain"  # Type de fichier
        )

    # Bouton de réinitialisation
    if st.sidebar.button("🧹 Réinitialiser"):
        st.session_state.chat_history = [{"role": "system", "content": SYSTEM_INSTRUCTION}]

        st.rerun()
