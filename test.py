import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Récupérer la clé API depuis les variables d'environnement
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configurer la clé API d'OpenAI
client_openai = OpenAI(api_key=OPENAI_API_KEY)

def get_openai_response(user_message):
    try:
        # Définir le prompt pour l'IA
        system_prompt = "Tu es un assistant amical."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Envoyer la requête à l'API d'OpenAI
        response = client_openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=100
        )

        # Retourner la réponse de l'IA
        return response.choices[0].message.content

    except Exception as e:
        return f"Erreur lors de l'appel à l'API OpenAI : {e}"

# Configuration de la page Streamlit
st.title("Chatbot Interactif avec OpenAI")

# Zone de texte pour poser des questions
user_input = st.text_input("Posez votre question ici :", "")

# Bouton pour envoyer la question
if st.button("Envoyer"):
    if user_input:
        # Obtenir la réponse de l'IA
        response = get_openai_response(user_input)
        # Afficher la réponse
        st.write("Réponse de l'IA :", response)
    else:
        st.write("Veuillez entrer une question.")
