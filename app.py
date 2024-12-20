import streamlit as st
import pandas as pd
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain, LLMChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain

# Récupération sécurisée de la clé API
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

class StreamlitChatbot:
    def __init__(self):
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.messages = []
            st.session_state.rag_chain = None
            st.session_state.quiz_chain = None
            st.session_state.videos_robotics = None
            st.session_state.model = None

        if not st.session_state.initialized:
            self._initialize_system()
            st.session_state.initialized = True

    def _initialize_system(self):
        """Initialize the system configuration"""
        try:
            # Initialisation du modèle
            st.session_state.model = ChatOpenAI(
                temperature=0.7,
                model_name="gpt-3.5-turbo"
            )

            # Initialisation des embeddings
            self.embeddings = OpenAIEmbeddings()

            # Chargement des données
            try:
                st.session_state.videos_robotics = pd.read_csv("final_videos_robotics (2).csv")
            except FileNotFoundError as e:
                st.error(f"Erreur de chargement du fichier CSV: {str(e)}")
                return

            # Préparation des documents
            self.docs = self._prepare_documents(st.session_state.videos_robotics)
            
            if not self.docs:
                st.error("❌ Aucun document n'a été préparé.")
                return

            # Création du vectorstore et setup des chaînes
            self.vectorstore = self._create_vectorstore(self.docs)
            self.setup_chains()
            
            st.success("✅ Système initialisé avec succès!")

        except Exception as e:
            st.error(f"❌ Erreur d'initialisation: {str(e)}")
            raise

    def setup_chains(self):
        """Configure les chaînes nécessaires"""
        try:
            # Configuration du retriever
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
            
            # Configuration du prompt pour la chaîne QA
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that answers questions based on the provided context."),
                ("human", "Context: {context}\n\nQuestion: {question}"),
            ])

            # Configuration de la chaîne QA
            qa_chain = load_qa_chain(
                llm=st.session_state.model,
                chain_type="stuff",
                prompt=qa_prompt,
            )

            # Configuration de la chaîne RAG
            st.session_state.rag_chain = RetrievalQAWithSourcesChain(
                combine_documents_chain=qa_chain,
                retriever=retriever,
            )

            # Configuration de la chaîne de quiz
            quiz_prompt = PromptTemplate.from_template("""
            Based on the following video content, create 5 multiple-choice questions in English.
            Make the questions focused and specific to the video content.

            Video Content: {content}

            Format EXACTLY as follows:

            Q1: [Question]
            A) [Option A]
            B) [Option B]
            C) [Option C]
            D) [Option D]
            Correct Answer: [Letter]

            [Continue for all 5 questions]
            """)
            
            st.session_state.quiz_chain = LLMChain(llm=st.session_state.model, prompt=quiz_prompt)

        except Exception as e:
            st.error(f"❌ Erreur lors de la configuration des chaînes: {str(e)}")
            raise

    def _prepare_documents(self, videos_robotics, max_tokens=1000):
        docs = []
        for _, video in videos_robotics.iterrows():
            try:
                if pd.notna(video["transcript_text"]):
                    transcript = video["transcript_text"][:max_tokens]
                    docs.append(Document(
                        page_content=transcript,
                        metadata={"video_id": video["video_id"], "title": video["title"]}
                    ))
            except KeyError as e:
                st.error(f"Erreur de clé dans la vidéo: {e}")
        return docs

    def _create_vectorstore(self, docs):
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        return FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )

    def _generate_quiz(self, video_title):
        """Génère un quiz basé sur le titre de la vidéo"""
        if st.session_state.videos_robotics is None:
            return "❌ Erreur: Les données des vidéos n'ont pas pu être chargées."

        matching_videos = st.session_state.videos_robotics[
            st.session_state.videos_robotics['title'].str.contains(video_title, case=False, na=False)
        ]

        if matching_videos.empty:
            return "❌ Aucun résultat trouvé pour ce titre."

        video = matching_videos.iloc[0]
        
        try:
            response = st.session_state.quiz_chain.invoke({"content": video['transcript_text'][:2000]})
            return response['text']
        except Exception as e:
            st.error(f"Erreur détaillée: {str(e)}")
            return f"Erreur lors de la génération du quiz: {str(e)}"

    def run(self):
        """Lance l'interface Streamlit"""
        st.title("🤖 Assistant Robotique")

        # Sidebar pour le mode quiz
        with st.sidebar:
            st.title("Options")
            mode = st.radio("Mode", ["Chat", "Quiz"])

        if mode == "Chat":
            # Interface de chat
            query = st.text_input("Posez votre question:", key="query_input")
            
            if query and st.session_state.rag_chain:
                try:
                    response = st.session_state.rag_chain({"question": query})
                    
                    # Ajouter les messages à l'historique
                    st.session_state.messages.append({"role": "user", "content": query})
                    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
                    
                    # Afficher l'historique des messages
                    for message in st.session_state.messages:
                        if message["role"] == "user":
                            st.write(f"👤 Vous: {message['content']}")
                        else:
                            st.write(f"🤖 Assistant: {message['content']}")
                            
                    # Afficher la source
                    if response.get("sources"):
                        st.info(f"📚 Sources: {response['sources']}")

                except Exception as e:
                    st.error(f"❌ Une erreur s'est produite: {str(e)}")

        else:  # Mode Quiz
            video_title = st.text_input("Entrez le titre de la vidéo pour générer un quiz:")
            if video_title:
                quiz = self._generate_quiz(video_title)
                st.markdown(f"## Quiz pour '{video_title}':")
                st.write(quiz)

def main():
    chatbot = StreamlitChatbot()
    chatbot.run()

if __name__ == "__main__":
    main()
