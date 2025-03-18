import os
import json
import streamlit as st
import chromadb
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import sqlite3
from dotenv import load_dotenv

print(sqlite3.sqlite_version)

# 🔑 Configuration de l'API Groq (à remplacer par votre clé)
# Charger les variables d'environnement depuis .env
load_dotenv()

# Récupérer la clé API
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("La clé API GROQ_API_KEY n'est pas définie.")

# 🟢 Initialisation de ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="seatech_chunks")

# 📂 Chargement des données JSON
def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

data = load_json("donnees_seatech.json")
chunks = [entry['markdown'] for entry in data]  # Assurez-vous que la clé 'markdown' existe

# 🔄 Ajout des documents à ChromaDB
collection.upsert(
    documents=chunks,
    ids=[str(i) for i in range(len(chunks))]
)

# 📜 Création du modèle de réponse
template = """
Tu es un assistant intelligent et utile. Réponds aux questions de l'utilisateur en français de manière claire et concise.

**Question de l'utilisateur :** {question}

📌 **Documents pertinents :**
{documents}

💡 **Réponse :**
"""

prompt = PromptTemplate(template=template, input_variables=["question", "documents"])
llm = ChatGroq(api_key=API_KEY, temperature=0, model="llama3-70b-8192")
chain = prompt | llm

# 🎨 **Interface Streamlit**
st.set_page_config(page_title="Chatbot Seatech", page_icon="💬", layout="wide")

# 📌 **En-tête**
st.markdown("<h1 style='text-align: center; color: #0066cc;'>🔹 Chatbot Seatech 🔹</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Posez votre question sur l'école et obtenez une réponse instantanée !</p>", unsafe_allow_html=True)

# 📝 **Champ de saisie**
question = st.text_input("💬 Posez votre question ici :", placeholder="Exemple : Admission sur dossier")

if st.button("🔍 Rechercher"):
    if question:
        with st.spinner("🔄 Recherche en cours..."):
            # 🔎 Interrogation de ChromaDB
            results = collection.query(query_texts=[question], n_results=3)
            retrieved_docs = "\n".join(results['documents'][0])

            # 🔥 Génération de la réponse avec LangChain
            response = chain.invoke({"question": question, "documents": retrieved_docs})

        # 📝 **Affichage de la réponse**
        st.markdown("### ✨ Réponse :")
        st.success(response.content)

        # 📚 **Affichage des documents pertinents**
        with st.expander("📂 Documents similaires retrouvés"):
            for i, doc in enumerate(results['documents'][0]):
                st.write(f"**{i+1}.** {doc}")

    else:
        st.warning("⚠️ Veuillez entrer une question avant de rechercher.")

# 📌 **Pied de page**
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>© 2025 Seatech AI | Développé en Python</p>", unsafe_allow_html=True)