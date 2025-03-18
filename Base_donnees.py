import json
import sys
import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st

# Charger le modèle d'embedding
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Charger les données JSON
with open("documents.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

# Initialiser ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")  # Stocke en local
collection = client.get_or_create_collection(name="json_collection")

# Ajouter les documents à la base vectorielle
for item in json_data:
    markdown_text = item['markdown']
    url = item["metadata"]["url"]  # Utiliser l'URL dans les métadonnées comme ID unique
    vector = embedding_model.encode(markdown_text).tolist()  # Convertir en vecteur

    # Ajouter dans la base vectorielle
    collection.add(
        ids=[url],  # L'URL comme ID unique
        embeddings=[vector],  # Vecteur du markdown
        documents=[markdown_text]  # Texte markdown original
    )

print("Indexation terminée ✅")

# Interface utilisateur avec Streamlit
st.title("🤖 Chatbot Seatech")
st.write("Posez votre question ci-dessous :")

# Champ de saisie pour l'utilisateur
user_query = st.text_input("Votre question :", "")

# Fonction de recherche dans la base de données ChromaDB
def search_question(query):
    # Convertir la question en vecteur
    query_vector = embedding_model.encode(query).tolist()

    # Rechercher dans la collection ChromaDB
    results = collection.query(
        query_embeddings=[query_vector],  # Vecteur de la question
        n_results=3  # Nombre de résultats que l'on veut récupérer
    )

    # Récupérer les résultats
    return results

# Fonction d'interaction avec le modèle LLM Groq
def query_llm_with_passages(query, passing_texts):
    # Préparer la requête avec la question et les passages pertinents
    combined_text = f"Question: {query}\n\n" + "\n\n".join(passing_texts)
    
    # Préparer la requête pour Groq (remplacer par ton modèle et clé)
    client = Groq(api_key="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")
    
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": combined_text},
                  {"role": "system", "content":" 1:Si la question ne concerne pas Seatech, répond 'Désolé, je ne peux pas répondre' 2: Tu es une IA d'aide pour une école d'ingenieur Francaise, Seatech, tu parles en francais de facon sympatique simple et direct. Tu te base sur les sources données après la question de l'utilisateur. Répond au demandes de l'utilisateur,  " }],
        temperature=0.9,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    # # Récupérer la réponse générée
    # for chunk in completion:
    #     print(chunk.choices[0].delta.content or "", end="")
    #     # ne marche pas
    #     # print(f'"c est ma complétion"{completion}')  

    response_text = ""  # Stocke la réponse complète

    for chunk in completion:
        if chunk.choices[0].delta.content:
            response_text += chunk.choices[0].delta.content  # Ajouter chaque morceau de texte

    return response_text  # Retourner la réponse finale

def main():
    # Exemple d'interrogation
    query = "la capital de paris ? ?"
    results = search_question(query)

    # Obtenir les passages pertinents
    passages = []
    for result, url in zip(results['documents'], results['ids']):
        markdown_text = result
        # Ajouter chaque passage au tableau des passages
        passages.append(f"URL: {url}\nMarkdown: {markdown_text}")

    # Envoyer la question et les passages au LLM pour obtenir une réponse
    query_llm_with_passages(query, passages)

# Exécution si l'utilisateur pose une question
if user_query:
    st.write("🔎 Recherche des documents pertinents...")
    
    results = search_question(user_query)
    # st.write("Résultats de la recherche : ", results)

    
    # Extraction des passages pertinents
    passages = []
    for result, url in zip(results['documents'], results['ids']):
        markdown_text = result
        passages.append(f"URL: {url}\nMarkdown: {markdown_text}")

    # # Affichage des documents trouvés
    # st.write("📚 **Documents pertinents trouvés :**")
    # for passage in passages:
    #     st.markdown(f"🔗 **{passage}**")

    # Génération de la réponse du chatbot
    st.write("🤖 **Réponse du chatbot :**")
    response = query_llm_with_passages(user_query, passages)
    st.success(response)

    # # Debug : afficher la réponse pour vérifier
    # st.write("Réponse générée par le chatbot : ", response)


# if __name__ == "__main__":
#     main()