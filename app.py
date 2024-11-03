import streamlit as st
import pandas as pd
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
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

        if not st.session_state.initialized:
            self._initialize_system()
            st.session_state.initialized = True

    def _initialize_system(self):
        """Initialize the system configuration"""
        try:
            # Initialisation du modèle
            self.model = ChatOpenAI(
                temperature=0.7,
                model_name="gpt-3.5-turbo"
            )

            # Initialisation des embeddings
            self.embeddings = OpenAIEmbeddings()

            # Chargement des données
            try:
                self.videos_robotics = pd.read_csv("final_videos_robotics (2).csv")
            except FileNotFoundError as e:
                st.error(f"Erreur de chargement du fichier CSV: {str(e)}")
                self.videos_robotics = None
                return

            # Préparation des documents
            self.docs = self._prepare_documents(self.videos_robotics) if self.videos_robotics is not None else []

            # Création du vectorstore
            if self.docs:
                self.vectorstore = self._create_vectorstore(self.docs)
                self.setup_chains()
            else:
                st.error("❌ Impossible d'initialiser le système car aucun document n'est chargé.")
                
            st.success("✅ Système initialisé avec succès!")

        except Exception as e:
            st.error(f"❌ Erreur d'initialisation: {str(e)}")

    def setup_chains(self):
        """Configure les chaînes nécessaires"""
        # Configuration de la chaîne RAG
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
        
        # Updated RAG chain initialization
        qa_chain = load_qa_chain(
            llm=self.model,
            chain_type="stuff",
            prompt=ChatPromptTemplate.from_template("""
            Answer the question based on the provided context. Be concise.
            Context: {context}
            Question: {question}
            """)
        )
        
        self.rag_chain = RetrievalQAWithSourcesChain(
            combine_documents_chain=qa_chain,
            retriever=retriever
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
        
        self.quiz_chain = LLMChain(llm=self.model, prompt=quiz_prompt)

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
        if self.videos_robotics is None:
            return "❌ Erreur: Les données des vidéos n'ont pas pu être chargées. Veuillez vérifier le fichier CSV."

        matching_videos = self.videos_robotics[
            self.videos_robotics['title'].str.contains(video_title, case=False, na=False)
        ]

        if matching_videos.empty:
            return "❌ Aucun résultat trouvé pour ce titre. Essayez un autre titre."

        video = matching_videos.iloc[0]
        
        try:
            response = self.quiz_chain.invoke({"content": video['transcript_text'][:2000]})
            return response
        except Exception as e:
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
            
            if query:
                try:
                    response = self.rag_chain({"question": query})
                    
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
