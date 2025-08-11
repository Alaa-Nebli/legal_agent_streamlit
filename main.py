"""
Enhanced Legal Expert Chat Application
Streamlit-based chat interface with optimized legal search and analysis
"""

import os
import json
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio
import time

# Lang/OpenAI/Pinecone imports
from langchain.agents import tool
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Configure Streamlit page
st.set_page_config(
    page_title="🏛️ Expert Juridique Tunisien",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .chat-message {
        padding: 1rem; 
        border-radius: 0.5rem; 
        margin-bottom: 1rem; 
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
        color: white;
        margin-left: 20%;
    }
    .chat-message.assistant {
        background-color: #475063;
        color: white;
        margin-right: 20%;
    }
    .chat-message .avatar {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        object-fit: cover;
        margin-bottom: 10px;
    }
    .source-citation {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-left: 4px solid #ff6b6b;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration ---
@st.cache_resource
def initialize_connections():
    """Initialize all connections with caching for better performance"""
    
    # Get API keys from Streamlit secrets or environment
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
        PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    except:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        st.error("⚠️ API keys not configured. Please set OPENAI_API_KEY and PINECONE_API_KEY in secrets.toml or environment variables.")
        st.stop()
    
    # Configuration
    PINECONE_ENV = "us-east-1"
    PINECONE_INDEX = "tunisia-laws"
    
    # Models
    RERANK_MODEL = "gpt-5-mini"
    SYNTHESIS_MODEL = "gpt-5"
    EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
    
    # Initialize connections
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL_NAME)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings, text_key="content")
    
    # LLM clients
    llm_rerank = ChatOpenAI(api_key=OPENAI_API_KEY, model=RERANK_MODEL, temperature=0)
    llm_synth = ChatOpenAI(api_key=OPENAI_API_KEY, model=SYNTHESIS_MODEL, temperature=0.1)
    
    return {
        "vector_store": vector_store,
        "llm_rerank": llm_rerank,
        "llm_synth": llm_synth,
        "index": index
    }

# Initialize connections
connections = initialize_connections()
vector_store = connections["vector_store"]
llm_rerank = connections["llm_rerank"]
llm_synth = connections["llm_synth"]
index = connections["index"]

# --- Enhanced Data Classes ---
@dataclass
class SearchResult:
    id: str
    score: float
    metadata: Dict[str, Any]
    excerpt: str
    full_content: str
    relevance_score: float = 0.0
    rationale: str = ""

@dataclass
class ChatMetrics:
    search_time: float
    rerank_time: float
    synthesis_time: float
    total_docs_retrieved: int
    final_docs_used: int

# --- Enhanced Core Functions ---
class LegalExpertChatbot:
    def __init__(self):
        self.vector_store = vector_store
        self.llm_rerank = llm_rerank
        self.llm_synth = llm_synth
        
    def _build_candidates_from_search(self, docs: List[Any]) -> List[SearchResult]:
        """Convert langchain Document results to SearchResult objects"""
        candidates = []
        for i, d in enumerate(docs):
            meta = d.metadata if hasattr(d, "metadata") else {}
            text = d.page_content if hasattr(d, "page_content") else str(d)
            excerpt = self._create_smart_excerpt(text, max_length=600)
            
            candidates.append(SearchResult(
                id=str(meta.get("id", i)),
                score=getattr(d, 'score', 0.0),
                metadata=meta,
                excerpt=excerpt,
                full_content=text
            ))
        return candidates
    
    def _create_smart_excerpt(self, text: str, max_length: int = 600) -> str:
        """Create intelligent excerpt that preserves sentence boundaries"""
        if len(text) <= max_length:
            return text.strip()
        
        # Try to cut at sentence boundary
        excerpt = text[:max_length]
        last_period = excerpt.rfind('.')
        last_question = excerpt.rfind('?')
        last_exclamation = excerpt.rfind('!')
        
        last_sentence_end = max(last_period, last_question, last_exclamation)
        
        if last_sentence_end > max_length * 0.7:  # If we can preserve 70% and get full sentence
            return text[:last_sentence_end + 1].strip()
        else:
            return excerpt.strip() + "..."
    
    def enhanced_rerank_with_llm(self, user_query: str, candidates: List[SearchResult], top_n: int = 5) -> Tuple[List[SearchResult], float]:
        """Enhanced reranking with timing and better prompts"""
        start_time = time.time()
        
        if not candidates:
            return [], 0.0
        
        # Prepare items for ranking
        items_for_ranking = []
        for c in candidates:
            title = c.metadata.get("title") or c.metadata.get("article_number") or f"Document {c.id}"
            legal_code = c.metadata.get("legal_code", "")
            
            items_for_ranking.append({
                "id": c.id,
                "title": title,
                "legal_code": legal_code,
                "excerpt": c.excerpt
            })
        
        # Enhanced reranking prompt
        system_prompt = """Vous êtes un expert en droit tunisien spécialisé dans l'évaluation de la pertinence juridique.

Analysez chaque document par rapport à la question de l'utilisateur et attribuez un score de pertinence (0-100) avec une justification concise.

Critères d'évaluation:
- Pertinence directe (50 points max): Le document répond-il directement à la question?
- Pertinence contextuelle (30 points max): Le document fournit-il un contexte juridique utile?
- Qualité juridique (20 points max): Le document est-il une source juridique fiable?

Répondez en JSON valide uniquement:
{"rankings": [{"id": "...", "score": 0-100, "rationale": "justification en une phrase"}, ...]}"""
        
        user_prompt = f"""Question de l'utilisateur: {user_query}

Documents à évaluer:
{json.dumps(items_for_ranking, ensure_ascii=False, indent=2)}"""
        
        try:
            response = self.llm_rerank.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            content = response.content.strip()
            
            # Parse JSON response
            try:
                rankings_data = json.loads(content)
            except json.JSONDecodeError:
                # Fallback parsing
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    rankings_data = json.loads(json_match.group())
                else:
                    # Fallback: return original order with equal scores
                    for c in candidates:
                        c.relevance_score = 50.0
                    return candidates[:top_n], time.time() - start_time
            
            # Apply scores and rationales
            id_to_ranking = {str(r["id"]): r for r in rankings_data.get("rankings", [])}
            
            for c in candidates:
                ranking_info = id_to_ranking.get(str(c.id), {})
                c.relevance_score = float(ranking_info.get("score", 0))
                c.rationale = ranking_info.get("rationale", "")
            
            # Sort by relevance score
            reranked = sorted(candidates, key=lambda x: x.relevance_score, reverse=True)
            
            return reranked[:top_n], time.time() - start_time
            
        except Exception as e:
            st.error(f"Erreur lors du re-classement: {str(e)}")
            return candidates[:top_n], time.time() - start_time
    
    def intelligent_search(self, query: str, legal_code: Optional[str] = None, top_k: int = 20, return_top: int = 5) -> Tuple[List[SearchResult], ChatMetrics]:
        """Enhanced search with comprehensive metrics"""
        start_time = time.time()
        
        # Search phase
        search_start = time.time()
        filter_dict = {"legal_code": {"$eq": legal_code}} if legal_code else {}
        docs = self.vector_store.similarity_search(query, k=top_k, filter=filter_dict)
        search_time = time.time() - search_start
        
        # Convert to candidates
        candidates = self._build_candidates_from_search(docs)
        
        # Rerank phase
        rerank_start = time.time()
        reranked_candidates, rerank_duration = self.enhanced_rerank_with_llm(query, candidates, top_n=return_top)
        
        metrics = ChatMetrics(
            search_time=search_time,
            rerank_time=rerank_duration,
            synthesis_time=0.0,  # Will be updated later
            total_docs_retrieved=len(docs),
            final_docs_used=len(reranked_candidates)
        )
        
        return reranked_candidates, metrics
    
    def generate_expert_answer(self, user_question: str, legal_code: Optional[str] = None, max_docs: int = 5) -> Tuple[str, ChatMetrics]:
        """Generate comprehensive answer with citations and metrics"""
        
        # Search and rerank
        search_results, metrics = self.intelligent_search(
            user_question, 
            legal_code=legal_code, 
            top_k=20, 
            return_top=max_docs
        )
        
        if not search_results:
            return "Désolé, je n'ai pas trouvé de documents pertinents pour répondre à votre question.", metrics
        
        # Synthesis phase
        synthesis_start = time.time()
        
        # Prepare context for synthesis
        context_documents = []
        for result in search_results:
            doc_info = {
                "id": result.id,
                "title": result.metadata.get("title", "Document sans titre"),
                "article_number": result.metadata.get("article_number", ""),
                "legal_code": result.metadata.get("legal_code", ""),
                "url": result.metadata.get("url_fr") or result.metadata.get("url", ""),
                "relevance_score": result.relevance_score,
                "excerpt": result.excerpt,
                "rationale": result.rationale
            }
            context_documents.append(doc_info)
        
        # Enhanced synthesis prompt
        synthesis_system = """Vous êtes un expert juriste tunisien hautement qualifié avec une expertise approfondie en droit tunisien.

Votre mission:
1. Analyser la question juridique posée
2. Fournir une réponse précise et complète basée sur les documents fournis
3. Citer explicitement les sources avec le format: [Article X — Code Y]
4. Inclure des extraits pertinents entre guillemets quand c'est utile
5. Signaler les zones d'incertitude juridique s'il y en a
6. Recommander des étapes supplémentaires si nécessaire

Structure de réponse:
- Réponse directe à la question
- Analyse juridique détaillée
- Citations et références
- Recommandations pratiques (si applicables)

Ton: Professionnel, précis, et accessible."""
        
        synthesis_user = f"""Question: {user_question}

Documents juridiques pertinents:
{json.dumps(context_documents, ensure_ascii=False, indent=2)}

Fournissez une analyse juridique complète avec citations appropriées."""
        
        try:
            response = self.llm_synth.invoke([
                {"role": "system", "content": synthesis_system},
                {"role": "user", "content": synthesis_user}
            ])
            
            answer = response.content.strip()
            
            # Add sources section
            sources_section = "\n\n📚 **Sources consultées:**\n"
            for i, result in enumerate(search_results, 1):
                title = result.metadata.get("title", "Document sans titre")
                code = result.metadata.get("legal_code", "Code non spécifié")
                url = result.metadata.get("url_fr") or result.metadata.get("url", "")
                score = result.relevance_score
                
                sources_section += f"{i}. **{title}** | {code} | Score: {score:.1f}/100"
                if url:
                    sources_section += f" | [Lien]({url})"
                sources_section += "\n"
            
            final_answer = answer + sources_section
            
            metrics.synthesis_time = time.time() - synthesis_start
            
            return final_answer, metrics
            
        except Exception as e:
            metrics.synthesis_time = time.time() - synthesis_start
            return f"Erreur lors de la génération de la réponse: {str(e)}", metrics

# --- Streamlit Interface ---
def main():
    st.title("🏛️ Expert Juridique Tunisien")
    st.markdown("*Assistant juridique intelligent spécialisé en droit tunisien*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Legal code filter
        legal_codes = [
            "Tous les codes",
            "code-commerce",
            "code-civil",
            "code-penal",
            "code-travail",
            "code-procedure-civile"
        ]
        
        selected_code = st.selectbox(
            "Code juridique",
            legal_codes,
            help="Filtrer par type de code juridique"
        )
        
        legal_code_filter = None if selected_code == "Tous les codes" else selected_code
        
        # Advanced settings
        with st.expander("🔧 Paramètres avancés"):
            max_docs = st.slider("Nombre max de documents", 3, 10, 5)
            show_metrics = st.checkbox("Afficher les métriques", value=True)
            show_sources_detail = st.checkbox("Détails des sources", value=False)
        
        # Statistics
        st.markdown("---")
        st.markdown("📊 **Statistiques de session**")
        if 'chat_count' not in st.session_state:
            st.session_state.chat_count = 0
        st.metric("Questions posées", st.session_state.chat_count)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_metrics = []
    
    # Initialize chatbot
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = LegalExpertChatbot()
    
    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show metrics if available and enabled
            if (message["role"] == "assistant" and 
                show_metrics and 
                i < len(st.session_state.chat_metrics)):
                
                metrics = st.session_state.chat_metrics[i // 2]  # Every other message is assistant
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("⏱️ Recherche", f"{metrics.search_time:.2f}s")
                with col2:
                    st.metric("🔄 Re-classement", f"{metrics.rerank_time:.2f}s")
                with col3:
                    st.metric("✍️ Synthèse", f"{metrics.synthesis_time:.2f}s")
                with col4:
                    st.metric("📄 Documents", f"{metrics.final_docs_used}/{metrics.total_docs_retrieved}")
    
    # Chat input
    if prompt := st.chat_input("Posez votre question juridique..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Recherche dans la base juridique..."):
                try:
                    answer, metrics = st.session_state.chatbot.generate_expert_answer(
                        prompt, 
                        legal_code=legal_code_filter,
                        max_docs=max_docs
                    )
                    
                    st.markdown(answer)
                    
                    # Store metrics
                    st.session_state.chat_metrics.append(metrics)
                    st.session_state.chat_count += 1
                    
                    # Show metrics in real-time if enabled
                    if show_metrics:
                        st.markdown("---")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("⏱️ Recherche", f"{metrics.search_time:.2f}s")
                        with col2:
                            st.metric("🔄 Re-classement", f"{metrics.rerank_time:.2f}s")
                        with col3:
                            st.metric("✍️ Synthèse", f"{metrics.synthesis_time:.2f}s")
                        with col4:
                            st.metric("📄 Documents", f"{metrics.final_docs_used}/{metrics.total_docs_retrieved}")
                    
                except Exception as e:
                    error_message = f"❌ Erreur: {str(e)}"
                    st.error(error_message)
                    answer = error_message
                    # Create dummy metrics for error case
                    metrics = ChatMetrics(0, 0, 0, 0, 0)
                    st.session_state.chat_metrics.append(metrics)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Footer with clear chat button
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Effacer l'historique"):
            st.session_state.messages = []
            st.session_state.chat_metrics = []
            st.session_state.chat_count = 0
            st.rerun()

if __name__ == "__main__":
    main()