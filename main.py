"""
Enhanced Human-like Legal Expert Chat Application
Streamlit-based chat interface with natural conversation flow
"""

import os
import json
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import time
import re

# Lang/OpenAI/Pinecone imports
from langchain.agents import tool
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Configure Streamlit page
st.set_page_config(
    page_title="🏛️ Maître Khalil - Expert Juridique",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more professional, human-like interface
st.markdown("""
<style>
    .chat-message {
        padding: 1.2rem; 
        border-radius: 1rem; 
        margin-bottom: 1rem; 
        display: flex;
        flex-direction: column;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 15%;
        border-bottom-right-radius: 0.3rem;
    }
    .chat-message.assistant {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 15%;
        border-bottom-left-radius: 0.3rem;
    }
    .expert-name {
        font-weight: bold;
        font-size: 1.1em;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    .thinking-indicator {
        font-style: italic;
        opacity: 0.7;
        font-size: 0.9em;
    }
    .source-citation {
        background: rgba(255,255,255,0.2);
        padding: 0.8rem;
        border-left: 4px solid #fff;
        margin: 1rem 0;
        border-radius: 0.5rem;
        backdrop-filter: blur(10px);
    }
    .legal-reference {
        background: rgba(255,255,255,0.15);
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration ---
@st.cache_resource
def initialize_connections():
    """Initialize all connections with caching for better performance"""
    
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
        PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    except:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")


    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        st.error("⚠️ Les clés API ne sont pas configurées. Veuillez définir OPENAI_API_KEY et PINECONE_API_KEY.")
        st.stop()
    
    # Configuration
    PINECONE_ENV = "us-east-1"
    PINECONE_INDEX = "tunisia-laws"
    
    # Models for different purposes
    CHAT_MODEL = "gpt-4o"  # For natural conversation
    ANALYSIS_MODEL = "gpt-4o"  # For deep legal analysis
    RERANK_MODEL = "gpt-4o-mini"  # For document ranking
    EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
    
    # Initialize connections
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL_NAME)
    
    # Fix: Create custom vector store that handles the content field correctly
    class TunisianLegalVectorStore(PineconeVectorStore):

        def similarity_search_with_score(self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs) -> List[Tuple]:
            """Override to handle our document structure"""
            results = []
            try:
                # Query Pinecone directly
                query_vector = self._embedding.embed_query(query)
                pinecone_results = self._index.query(
                    vector=query_vector,
                    top_k=k,
                    include_metadata=True,
                    filter=filter
                )
                
                for match in pinecone_results.matches:
                    metadata = match.metadata
                    
                    # Extract content based on available fields
                    content = ""
                    if 'content_fr' in metadata and metadata['content_fr']:
                        content = metadata['content_fr']
                    elif 'content_ar' in metadata and metadata['content_ar']:
                        content = metadata['content_ar']
                    elif 'summary' in metadata:
                        content = metadata['summary']
                    else:
                        continue  # Skip if no content found
                    
                    # Create document-like object
                    doc = type('Document', (), {
                        'page_content': content,
                        'metadata': metadata
                    })()
                    
                    results.append((doc, match.score))
                    
            except Exception as e:
                st.error(f"Erreur lors de la recherche: {e}")
                
            return results
        
        def similarity_search(self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs):
            """Override similarity_search to use our custom method"""
            docs_and_scores = self.similarity_search_with_score(query, k=k, filter=filter, **kwargs)
            return [doc for doc, score in docs_and_scores]
    
    vector_store = TunisianLegalVectorStore(index=index, embedding=embeddings, text_key="content")
    
    # LLM clients
    llm_chat = ChatOpenAI(api_key=OPENAI_API_KEY, model=CHAT_MODEL, temperature=0.7)
    llm_analysis = ChatOpenAI(api_key=OPENAI_API_KEY, model=ANALYSIS_MODEL, temperature=0.3)
    llm_rerank = ChatOpenAI(api_key=OPENAI_API_KEY, model=RERANK_MODEL, temperature=0)
    
    return {
        "vector_store": vector_store,
        "llm_chat": llm_chat,
        "llm_analysis": llm_analysis,
        "llm_rerank": llm_rerank,
        "index": index
    }

# Initialize connections
connections = initialize_connections()
vector_store = connections["vector_store"]
llm_chat = connections["llm_chat"]
llm_analysis = connections["llm_analysis"]
llm_rerank = connections["llm_rerank"]

# --- Enhanced Data Classes ---
@dataclass
class LegalDocument:
    id: str
    title: str
    article_number: int
    content_fr: str
    content_ar: str
    legal_code: str
    summary: str
    tags: List[str]
    status: str
    relevance_score: float = 0.0
    
@dataclass
class ConversationContext:
    user_question: str
    detected_language: str
    legal_domain: str
    complexity_level: str
    follow_up_questions: List[str]

# --- Human-like Legal Expert Class ---
class MaitreKhalil:
    def __init__(self):
        self.name = "Maître Khalil Ben Ahmed"
        self.specialties = [
            "Droit commercial tunisien",
            "Droit civil et obligations",
            "Droit pénal des affaires",
            "Comptabilité publique",
            "Procédures juridiques"
        ]
        self.vector_store = vector_store
        self.llm_chat = llm_chat
        self.llm_analysis = llm_analysis
        self.llm_rerank = llm_rerank
        
    def _detect_language_and_context(self, query: str) -> ConversationContext:
        """Analyze the query to understand context and user needs"""
        
        # Simple language detection
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', query))
        total_chars = len(query.replace(' ', ''))
        detected_language = "ar" if arabic_chars > total_chars * 0.3 else "fr"
        
        # Detect legal domain based on keywords
        domain_keywords = {
            "commercial": ["commerce", "entreprise", "société", "contrat commercial", "تجارة", "شركة"],
            "civil": ["mariage", "divorce", "succession", "propriété", "زواج", "طلاق", "ميراث"],
            "penal": ["crime", "délit", "sanction", "prison", "جريمة", "عقوبة"],
            "public": ["administration", "finances publiques", "comptabilité", "إدارة", "مالية عامة"],
            "procedure": ["procédure", "tribunal", "recours", "إجراءات", "محكمة"]
        }
        
        legal_domain = "general"
        for domain, keywords in domain_keywords.items():
            if any(keyword.lower() in query.lower() for keyword in keywords):
                legal_domain = domain
                break
        
        # Assess complexity
        complexity_indicators = ["exception", "recours", "cassation", "constitutionnel", "استثناء", "طعن"]
        complexity_level = "complex" if any(ind.lower() in query.lower() for ind in complexity_indicators) else "standard"
        
        return ConversationContext(
            user_question=query,
            detected_language=detected_language,
            legal_domain=legal_domain,
            complexity_level=complexity_level,
            follow_up_questions=[]
        )
    
    def _search_legal_documents(self, query: str, legal_code: Optional[str] = None, top_k: int = 15) -> List[LegalDocument]:
        """Search for relevant legal documents"""
        
        try:
            # Build filter
            filter_dict = {}
            if legal_code and legal_code != "all":
                filter_dict["legal_code"] = {"$eq": legal_code}
            
            # Search with our custom vector store
            docs_with_scores = vector_store.similarity_search_with_score(
                query, 
                k=top_k, 
                filter=filter_dict
            )
            
            legal_docs = []
            for doc, score in docs_with_scores:
                metadata = doc.metadata
                
                legal_doc = LegalDocument(
                    id=metadata.get("id", "unknown"),
                    title=metadata.get("title", "Document sans titre"),
                    article_number=metadata.get("article_index", 0),
                    content_fr=metadata.get("content_fr", ""),
                    content_ar=metadata.get("content_ar", ""),
                    legal_code=metadata.get("legal_code", ""),
                    summary=metadata.get("summary", ""),
                    tags=metadata.get("tags", []),
                    status=metadata.get("status", ""),
                    relevance_score=float(score)
                )
                legal_docs.append(legal_doc)
            
            return legal_docs
            
        except Exception as e:
            st.error(f"Erreur lors de la recherche: {e}")
            return []
    
    def _rerank_documents(self, context: ConversationContext, documents: List[LegalDocument]) -> List[LegalDocument]:
        """Intelligently rerank documents based on context"""
        
        if not documents:
            return []
        
        # Prepare documents for ranking
        docs_for_ranking = []
        for doc in documents:
            content = doc.content_fr if context.detected_language == "fr" else doc.content_ar
            if not content:  # Fallback to the other language
                content = doc.content_ar if context.detected_language == "fr" else doc.content_fr
            
            docs_for_ranking.append({
                "id": doc.id,
                "title": doc.title,
                "article_number": doc.article_number,
                "legal_code": doc.legal_code,
                "content": content[:800],  # Truncate for ranking
                "summary": doc.summary,
                "tags": doc.tags
            })
        
        rerank_prompt = f"""En tant qu'expert juridique tunisien, évaluez la pertinence de chaque document pour cette question:

Question: {context.user_question}
Domaine juridique détecté: {context.legal_domain}
Niveau de complexité: {context.complexity_level}

Documents à évaluer:
{json.dumps(docs_for_ranking, ensure_ascii=False, indent=2)}

Donnez un score de pertinence (0-100) pour chaque document. Considérez:
- Pertinence directe au problème juridique
- Applicabilité en droit tunisien
- Niveau de détail approprié

Répondez uniquement en JSON:
{{"rankings": [{{"id": "...", "score": 0-100, "rationale": "..."}}]}}"""

        try:
            response = self.llm_rerank.invoke([{"role": "user", "content": rerank_prompt}])
            rankings_data = json.loads(response.content.strip())
            
            # Apply new scores
            id_to_score = {r["id"]: r["score"] for r in rankings_data.get("rankings", [])}
            for doc in documents:
                doc.relevance_score = id_to_score.get(doc.id, doc.relevance_score)
            
            # Sort by new relevance score
            return sorted(documents, key=lambda x: x.relevance_score, reverse=True)[:5]
            
        except Exception as e:
            st.warning(f"Problème avec le re-classement: {e}")
            return documents[:5]
    
    def _generate_human_response(self, context: ConversationContext, relevant_docs: List[LegalDocument]) -> str:
        """Generate a natural, human-like response"""
        
        if not relevant_docs:
            return self._generate_no_results_response(context)
        
        # Prepare legal context
        legal_context = []
        for doc in relevant_docs:
            content = doc.content_fr if context.detected_language == "fr" else doc.content_ar
            if not content:
                content = doc.content_ar if context.detected_language == "fr" else doc.content_fr
            
            legal_context.append({
                "article": doc.article_number,
                "title": doc.title,
                "code": doc.legal_code,
                "content": content,
                "summary": doc.summary,
                "status": doc.status,
                "relevance": doc.relevance_score
            })
        
        # Create human-like persona prompt
        persona_prompt = f"""Vous êtes Maître Khalil Ben Ahmed, un avocat tunisien expérimenté avec 20 ans d'expérience.

VOTRE PERSONNALITÉ:
- Chaleureux et accessible, mais professionnel
- Vous expliquez le droit de manière claire et pédagogique
- Vous utilisez des exemples concrets
- Vous êtes prudent et mentionnez quand consulter un avocat
- Vous maîtrisez parfaitement le français et l'arabe
- Vous avez une connaissance approfondie du droit tunisien

STYLE DE COMMUNICATION:
- Commencez par une salutation personnelle
- Structurez votre réponse clairement
- Utilisez "En droit tunisien..." ou "Selon la législation..."
- Terminez par des conseils pratiques
- Soyez empathique aux préoccupations du client

QUESTION DU CLIENT: {context.user_question}

CONTEXTE JURIDIQUE PERTINENT:
{json.dumps(legal_context, ensure_ascii=False, indent=2)}

Répondez comme un vrai avocat tunisien le ferait lors d'une consultation. Soyez naturel, professionnel et rassurant."""

        try:
            response = self.llm_chat.invoke([{"role": "user", "content": persona_prompt}])
            base_response = response.content.strip()
            
            # Add legal references in a natural way
            if relevant_docs:
                base_response += "\n\n📋 **Références juridiques consultées:**\n"
                for i, doc in enumerate(relevant_docs[:3], 1):
                    base_response += f"{i}. Article {doc.article_number} - {doc.title} ({doc.legal_code})\n"
                    if doc.status:
                        base_response += f"   *{doc.status}*\n"
            
            return base_response
            
        except Exception as e:
            return f"Je suis désolé, j'ai rencontré un problème technique. Pouvez-vous reformuler votre question ? (Erreur: {e})"
    
    def _generate_no_results_response(self, context: ConversationContext) -> str:
        """Generate a helpful response when no relevant documents are found"""
        
        responses = [
            f"Je comprends votre question sur {context.legal_domain}, mais je n'ai pas trouvé de textes juridiques spécifiques dans ma base de données actuelle.",
            "Permettez-moi de vous orienter différemment. Pouvez-vous me donner plus de détails sur votre situation ?",
            "En attendant, je vous recommande de consulter directement un confrère spécialisé pour une analyse personnalisée."
        ]
        
        if context.legal_domain != "general":
            responses.append(f"Pour les questions de {context.legal_domain}, il serait peut-être utile de consulter directement le code concerné.")
        
        return "\n\n".join(responses)
    
    def respond_to_question(self, user_input: str, legal_code_filter: Optional[str] = None) -> Tuple[str, Dict]:
        """Main method to process user question and generate response"""
        
        # Analyze context
        context = self._detect_language_and_context(user_input)
        
        # Search for relevant documents
        documents = self._search_legal_documents(
            user_input, 
            legal_code=legal_code_filter,
            top_k=15
        )
        
        # Rerank based on context
        relevant_docs = self._rerank_documents(context, documents)
        
        # Generate human-like response
        response = self._generate_human_response(context, relevant_docs)
        
        # Prepare metadata for UI
        metadata = {
            "detected_language": context.detected_language,
            "legal_domain": context.legal_domain,
            "documents_found": len(documents),
            "documents_used": len(relevant_docs),
            "complexity": context.complexity_level
        }
        
        return response, metadata

# --- Streamlit Interface ---
def main():
    # Header with expert introduction
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1>🏛️ Maître Khalil Ben Ahmed</h1>
        <h3 style='color: #666; font-weight: normal;'>Avocat & Expert en Droit Tunisien</h3>
        <p style='font-style: italic; color: #888;'>Spécialisé en droit commercial, civil et comptabilité publique</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with expert profile
    with st.sidebar:
        st.markdown("""
        ### 👨‍⚖️ Profil de l'Expert
        
        **Maître Khalil Ben Ahmed**
        - 🎓 20+ ans d'expérience
        - 🏛️ Droit tunisien
        - 📚 Spécialités multiples
        - 🌐 Bilingue FR/AR
        
        ---
        """)
        
        # Legal code filter
        st.subheader("🔍 Filtrer par domaine")
        legal_codes = {
            "Tous les domaines": None,
            "📊 Comptabilité publique": "code-comptabilite-publique",
            "🏢 Droit commercial": "code-commerce",
            "👥 Droit civil": "code-civil",
            "⚖️ Droit pénal": "code-penal",
            "💼 Droit du travail": "code-travail",
            "📋 Procédure civile": "code-procedure-civile"
        }
        
        selected_code_display = st.selectbox(
            "Domaine juridique",
            list(legal_codes.keys()),
            help="Concentrer la recherche sur un domaine spécifique"
        )
        selected_code = legal_codes[selected_code_display]
        
        # Expert availability status
        st.markdown("""
        ---
        ### 📊 Statut
        🟢 **En ligne** - Réponse immédiate  
        📖 Base juridique à jour  
        🔒 Consultations confidentielles
        """)
        
        # Quick tips
        with st.expander("💡 Conseils pour de meilleures réponses"):
            st.markdown("""
            - Soyez spécifique dans vos questions
            - Mentionnez le contexte (personnel/entreprise)
            - Précisez si c'est urgent
            - N'hésitez pas à demander des clarifications
            """)
    
    # Initialize chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        welcome_msg = """Bonjour ! Je suis Maître Khalil Ben Ahmed, avocat spécialisé en droit tunisien.

Je suis là pour vous aider avec vos questions juridiques, que ce soit en droit commercial, civil, pénal, ou comptabilité publique. 

N'hésitez pas à me poser votre question en français ou en arabe. Je vous donnerai une réponse claire avec les références juridiques appropriées.

Comment puis-je vous aider aujourd'hui ? 😊"""
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": welcome_msg,
            "metadata": {}
        })
    
    if "expert" not in st.session_state:
        st.session_state.expert = MaitreKhalil()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(f'<div class="expert-name">👨‍⚖️ Maître Khalil</div>', unsafe_allow_html=True)
            
            st.markdown(message["content"])
            
            # Show metadata for assistant messages (except welcome)
            if (message["role"] == "assistant" and 
                message.get("metadata") and 
                len(message["metadata"]) > 0):
                
                meta = message["metadata"]
                with st.expander("📊 Détails de l'analyse", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🌐 Langue détectée", meta.get("detected_language", "fr").upper())
                    with col2:
                        st.metric("⚖️ Domaine", meta.get("legal_domain", "general"))
                    with col3:
                        st.metric("📄 Documents analysés", f"{meta.get('documents_used', 0)}/{meta.get('documents_found', 0)}")
    
    # Chat input
    if prompt := st.chat_input("Tapez votre question juridique..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt, "metadata": {}})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate expert response
        with st.chat_message("assistant"):
            st.markdown(f'<div class="expert-name">👨‍⚖️ Maître Khalil</div>', unsafe_allow_html=True)
            
            # Show thinking indicator
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown('<div class="thinking-indicator">🤔 Analyse de votre question et recherche dans la jurisprudence...</div>', unsafe_allow_html=True)
            
            try:
                # Get response from expert
                response, metadata = st.session_state.expert.respond_to_question(
                    prompt, 
                    legal_code_filter=selected_code
                )
                
                # Clear thinking indicator and show response
                thinking_placeholder.empty()
                st.markdown(response)
                
                # Store the response
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "metadata": metadata
                })
                
                # Show analysis details
                if metadata:
                    with st.expander("📊 Détails de l'analyse", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🌐 Langue détectée", metadata.get("detected_language", "fr").upper())
                        with col2:
                            st.metric("⚖️ Domaine", metadata.get("legal_domain", "general"))
                        with col3:
                            st.metric("📄 Documents analysés", f"{metadata.get('documents_used', 0)}/{metadata.get('documents_found', 0)}")
                
            except Exception as e:
                thinking_placeholder.empty()
                error_msg = f"Je suis désolé, j'ai rencontré un problème technique. Pouvez-vous réessayer ? 🔧\n\nDétails: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg,
                    "metadata": {}
                })
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("""
        <div style='font-size: 0.8em; color: #666;'>
        ⚠️ <strong>Avertissement:</strong> Ces réponses sont à titre informatif uniquement. 
        Pour des conseils juridiques personnalisés, consultez un avocat.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🔄 Nouvelle consultation"):
            st.session_state.messages = []
            st.rerun()
    
    with col3:
        if st.button("📞 Contact direct"):
            st.info("📧 khalil.ahmed@avocat.tn\n📱 +216 XX XXX XXX")

if __name__ == "__main__":
    main()