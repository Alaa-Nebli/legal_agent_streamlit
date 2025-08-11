"""
Advanced Legal Expert AI System
Professional-grade legal assistant for enterprises and individuals
"""

import os
import json
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import time
import re
from enum import Enum

# Lang/OpenAI/Pinecone imports
from langchain.agents import tool
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Configure Streamlit page
st.set_page_config(
    page_title="🏛️ Legaly Pro - Expert Juridique Tunisien",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional interface
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem; 
        border-radius: 1rem; 
        margin-bottom: 1.5rem; 
        display: flex;
        flex-direction: column;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .chat-message:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .chat-message.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 10%;
        border-bottom-right-radius: 0.3rem;
    }
    .chat-message.assistant {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 10%;
        border-bottom-left-radius: 0.3rem;
    }
    .expert-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        font-weight: bold;
        font-size: 1.1em;
    }
    .thinking-indicator {
        font-style: italic;
        opacity: 0.8;
        font-size: 0.95em;
        padding: 0.5rem;
        background: rgba(255,255,255,0.1);
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .legal-section {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-left: 4px solid #fff;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .contract-clause {
        background: rgba(255,255,255,0.15);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-family: 'Georgia', serif;
        border-left: 3px solid #ffd700;
    }
    .citation {
        font-size: 0.9em;
        opacity: 0.9;
        margin-top: 0.5rem;
        font-style: italic;
    }
    .complexity-indicator {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.8em;
        font-weight: bold;
    }
    .complexity-basic { background: #28a745; color: white; }
    .complexity-intermediate { background: #ffc107; color: black; }
    .complexity-advanced { background: #dc3545; color: white; }
</style>
""", unsafe_allow_html=True)

# --- Enhanced Data Models ---
class QueryType(Enum):
    QUESTION_ANSWER = "qa"
    CONTRACT_DRAFT = "contract"
    LEGAL_ANALYSIS = "analysis"
    COMPLIANCE_CHECK = "compliance"
    PRECEDENT_SEARCH = "precedent"

class ExpertiseLevel(Enum):
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"

@dataclass
class EnhancedLegalDocument:
    id: str
    title: str
    article_number: int
    content_fr: str
    content_ar: str
    legal_code: str
    summary: str
    tags: List[str]
    status: str
    country: str = "Tunisia"
    relevance_score: float = 0.0
    semantic_score: float = 0.0
    legal_weight: float = 0.0
    
@dataclass
class QueryContext:
    original_query: str
    reformulated_queries: List[str]
    detected_language: str
    query_type: QueryType
    legal_domain: str
    complexity_level: str
    user_expertise: ExpertiseLevel
    requires_citations: bool = True
    requires_examples: bool = False
    business_context: bool = False

@dataclass
class LegalResponse:
    primary_answer: str
    detailed_analysis: str
    legal_references: List[Dict]
    practical_advice: str
    risk_assessment: str
    next_steps: List[str]
    confidence_score: float
    processing_metadata: Dict

# --- Configuration ---
@st.cache_resource
def initialize_connections():
    """Initialize all connections with enhanced error handling"""
    
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
        PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    except:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        st.error("⚠️ Configuration requise: OPENAI_API_KEY et PINECONE_API_KEY")
        st.stop()
    
    # Enhanced model configuration
    MODELS = {
        "query_understanding": "gpt-4o",
        "legal_analysis": "gpt-4o", 
        "contract_drafting": "gpt-4o",
        "document_ranking": "gpt-4o-mini",
        "quality_control": "gpt-4o-mini"
    }
    
    PINECONE_ENV = "us-east-1"
    PINECONE_INDEX = "tunisia-laws"
    EMBEDDING_MODEL = "text-embedding-ada-002"
    
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)
    
    # Enhanced Vector Store with better document handling
    class ProfessionalLegalVectorStore(PineconeVectorStore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
        def enhanced_similarity_search(self, query: str, k: int = 20, filter: Optional[dict] = None) -> List[Tuple]:
            """Enhanced search with better document processing"""
            results = []
            try:
                query_vector = self._embedding.embed_query(query)
                search_results = self._index.query(
                    vector=query_vector,
                    top_k=k,
                    include_metadata=True,
                    filter=filter
                )
                
                for match in search_results.matches:
                    metadata = match.metadata
                    
                    # Smart content extraction with fallbacks
                    content = self._extract_best_content(metadata)
                    if not content:
                        continue
                    
                    # Create enhanced document object
                    doc = type('EnhancedDocument', (), {
                        'page_content': content,
                        'metadata': metadata,
                        'score': match.score,
                        'id': metadata.get('id', 'unknown')
                    })()
                    
                    results.append((doc, match.score))
                    
            except Exception as e:
                st.error(f"Erreur de recherche vectorielle: {e}")
                
            return results
        
        def _extract_best_content(self, metadata: Dict) -> str:
            """Extract the best available content from metadata"""
            content_fields = ['content_fr', 'content_ar', 'summary', 'title']
            
            for field in content_fields:
                content = metadata.get(field, '')
                if content and len(content.strip()) > 10:
                    return content.strip()
            
            return ""
    
    vector_store = ProfessionalLegalVectorStore(
        index=index, 
        embedding=embeddings, 
        text_key="content"
    )
    
    # Initialize LLM clients with specific configurations
    llm_clients = {}
    for purpose, model in MODELS.items():
        temperature = 0.1 if purpose in ["legal_analysis", "document_ranking"] else 0.3
        llm_clients[purpose] = ChatOpenAI(
            api_key=OPENAI_API_KEY, 
            model=model, 
            temperature=temperature,
            max_tokens=4000 if purpose == "contract_drafting" else 2000
        )
    
    return {
        "vector_store": vector_store,
        "llm_clients": llm_clients,
        "index": index
    }

# Initialize connections
connections = initialize_connections()
vector_store = connections["vector_store"]
llm_clients = connections["llm_clients"]

# --- Advanced Legal Expert System ---
class AdvancedLegalExpert:
    def __init__(self):
        self.name = "LegalGPT Pro"
        self.version = "2.0"
        self.vector_store = vector_store
        self.llm_clients = llm_clients
        self.expertise_areas = {
            "commercial": "Droit Commercial et des Sociétés",
            "civil": "Droit Civil et Obligations", 
            "penal": "Droit Pénal et Procédure Pénale",
            "public": "Droit Public et Administratif",
            "labor": "Droit du Travail et Sécurité Sociale",
            "tax": "Droit Fiscal et Comptabilité",
            "international": "Droit International des Affaires"
        }
        
    def analyze_query_intent(self, query: str) -> QueryContext:
        """Advanced query analysis with intent detection"""
        
        intent_prompt = f"""Analysez cette requête juridique et déterminez:

REQUÊTE: "{query}"

Répondez en JSON uniquement:
{{
    "detected_language": "fr|ar|mixed",
    "query_type": "qa|contract|analysis|compliance|precedent",
    "legal_domain": "commercial|civil|penal|public|labor|tax|international|general",
    "complexity_level": "basic|intermediate|advanced",
    "user_expertise": "novice|intermediate|expert", 
    "business_context": true|false,
    "requires_citations": true|false,
    "requires_examples": true|false,
    "reformulated_queries": ["alternative query 1", "alternative query 2"]
}}

Critères:
- query_type: qa=question simple, contract=rédaction, analysis=analyse approfondie
- complexity_level: basic=concepts de base, intermediate=cas pratiques, advanced=jurisprudence complexe
- user_expertise: basé sur le vocabulaire et la précision de la question
- business_context: true si contexte entrepreneurial/commercial
"""

        try:
            response = self.llm_clients["query_understanding"].invoke([
                {"role": "system", "content": "Vous êtes un expert en analyse d'intentions juridiques."},
                {"role": "user", "content": intent_prompt}
            ])
            
            # Robust JSON parsing with fallbacks
            content = response.content.strip()
            
            # Try direct JSON parsing
            try:
                analysis = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                else:
                    # Fallback to default analysis
                    analysis = self._create_fallback_analysis(query)
            
            return QueryContext(
                original_query=query,
                reformulated_queries=analysis.get("reformulated_queries", [query]),
                detected_language=analysis.get("detected_language", "fr"),
                query_type=QueryType(analysis.get("query_type", "qa")),
                legal_domain=analysis.get("legal_domain", "general"),
                complexity_level=analysis.get("complexity_level", "basic"),
                user_expertise=ExpertiseLevel(analysis.get("user_expertise", "novice")),
                business_context=analysis.get("business_context", False),
                requires_citations=analysis.get("requires_citations", True),
                requires_examples=analysis.get("requires_examples", False)
            )
            
        except Exception as e:
            st.warning(f"Analyse d'intention échouée, utilisation des paramètres par défaut: {e}")
            return self._create_fallback_analysis(query)
    
    def _create_fallback_analysis(self, query: str) -> QueryContext:
        """Create fallback analysis when intent detection fails"""
        
        # Simple language detection
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', query))
        total_chars = len(query.replace(' ', ''))
        detected_language = "ar" if arabic_chars > total_chars * 0.3 else "fr"
        
        # Simple domain detection
        domain_keywords = {
            "commercial": ["entreprise", "société", "commerce", "contrat", "شركة", "تجارة"],
            "civil": ["mariage", "divorce", "propriété", "succession", "زواج", "ملكية"],
            "penal": ["crime", "sanction", "prison", "جريمة", "عقوبة"],
            "public": ["administration", "service public", "إدارة"],
            "labor": ["travail", "employé", "salaire", "عمل", "موظف"]
        }
        
        detected_domain = "general"
        for domain, keywords in domain_keywords.items():
            if any(keyword.lower() in query.lower() for keyword in keywords):
                detected_domain = domain
                break
        
        return QueryContext(
            original_query=query,
            reformulated_queries=[query],
            detected_language=detected_language,
            query_type=QueryType.QUESTION_ANSWER,
            legal_domain=detected_domain,
            complexity_level="basic",
            user_expertise=ExpertiseLevel.NOVICE,
            business_context=False,
            requires_citations=True,
            requires_examples=True
        )
    
    def enhanced_document_search(self, context: QueryContext, legal_code_filter: Optional[str] = None) -> List[EnhancedLegalDocument]:
        """Advanced multi-query document search with semantic ranking"""
        
        all_documents = {}  # Use dict to avoid duplicates
        
        # Search with original and reformulated queries
        queries_to_search = [context.original_query] + context.reformulated_queries
        
        for query in queries_to_search:
            try:
                # Build search filter
                filter_dict = {}
                if legal_code_filter and legal_code_filter != "all":
                    filter_dict["legal_code"] = {"$eq": legal_code_filter}
                
                # Enhanced search
                docs_with_scores = self.vector_store.enhanced_similarity_search(
                    query=query,
                    k=15,
                    filter=filter_dict
                )
                
                # Process and deduplicate results
                for doc, score in docs_with_scores:
                    metadata = doc.metadata
                    doc_id = metadata.get('id', 'unknown')
                    
                    if doc_id not in all_documents:
                        enhanced_doc = EnhancedLegalDocument(
                            id=doc_id,
                            title=metadata.get('title', 'Document sans titre'),
                            article_number=metadata.get('article_index', 0),
                            content_fr=metadata.get('content_fr', ''),
                            content_ar=metadata.get('content_ar', ''),
                            legal_code=metadata.get('legal_code', ''),
                            summary=metadata.get('summary', ''),
                            tags=metadata.get('tags', []),
                            status=metadata.get('status', ''),
                            country=metadata.get('country', 'Tunisia'),
                            semantic_score=float(score)
                        )
                        all_documents[doc_id] = enhanced_doc
                    else:
                        # Update score if higher
                        if float(score) > all_documents[doc_id].semantic_score:
                            all_documents[doc_id].semantic_score = float(score)
                
            except Exception as e:
                st.warning(f"Erreur de recherche pour '{query}': {e}")
                continue
        
        return list(all_documents.values())
    
    def intelligent_document_ranking(self, context: QueryContext, documents: List[EnhancedLegalDocument]) -> List[EnhancedLegalDocument]:
        """Advanced document ranking with legal expertise"""
        
        if not documents:
            return []
        
        # Prepare documents for ranking with enhanced metadata
        ranking_data = []
        for doc in documents:
            content = doc.content_fr if context.detected_language == "fr" else doc.content_ar
            if not content:
                content = doc.content_ar if context.detected_language == "fr" else doc.content_fr
            
            ranking_data.append({
                "id": doc.id,
                "title": doc.title,
                "article_number": doc.article_number,
                "legal_code": doc.legal_code,
                "content_preview": content[:1000],
                "summary": doc.summary,
                "tags": doc.tags,
                "status": doc.status,
                "semantic_score": doc.semantic_score
            })
        
        ranking_prompt = f"""En tant qu'expert juridique tunisien de niveau international, analysez et classez ces documents juridiques.

CONTEXTE DE LA REQUÊTE:
- Question: {context.original_query}
- Domaine: {context.legal_domain}
- Complexité: {context.complexity_level}
- Expertise utilisateur: {context.user_expertise.value}
- Contexte business: {context.business_context}

DOCUMENTS À CLASSER:
{json.dumps(ranking_data, ensure_ascii=False, indent=2)}

CRITÈRES DE CLASSEMENT (total 100 points):
1. Pertinence directe (40 points): Répond directement à la question
2. Autorité juridique (25 points): Importance hiérarchique du texte
3. Actualité (20 points): Statut actuel et modifications récentes
4. Applicabilité pratique (15 points): Utilité pour le cas d'usage

Répondez UNIQUEMENT en JSON valide:
{{
    "rankings": [
        {{
            "id": "document_id",
            "total_score": 0-100,
            "pertinence_directe": 0-40,
            "autorite_juridique": 0-25,
            "actualite": 0-20,
            "applicabilite_pratique": 0-15,
            "justification": "Analyse en 1-2 phrases"
        }}
    ]
}}"""

        try:
            response = self.llm_clients["document_ranking"].invoke([
                {"role": "system", "content": "Vous êtes un expert en classification de documents juridiques tunisiens avec 25 ans d'expérience."},
                {"role": "user", "content": ranking_prompt}
            ])
            
            content = response.content.strip()
            
            # Robust JSON parsing
            try:
                rankings_result = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    rankings_result = json.loads(json_match.group())
                else:
                    # Fallback: return documents sorted by semantic score
                    return sorted(documents, key=lambda x: x.semantic_score, reverse=True)[:8]
            
            # Apply rankings
            id_to_ranking = {r["id"]: r for r in rankings_result.get("rankings", [])}
            
            for doc in documents:
                ranking_info = id_to_ranking.get(doc.id, {})
                doc.legal_weight = float(ranking_info.get("total_score", 50))
                doc.relevance_score = doc.legal_weight  # Update relevance score
            
            # Sort by legal weight and return top documents
            ranked_docs = sorted(documents, key=lambda x: x.legal_weight, reverse=True)
            return ranked_docs[:8]
            
        except Exception as e:
            st.warning(f"Erreur de classement intelligent: {e}")
            # Fallback to semantic scoring
            return sorted(documents, key=lambda x: x.semantic_score, reverse=True)[:8]
    
    def generate_comprehensive_response(self, context: QueryContext, documents: List[EnhancedLegalDocument]) -> LegalResponse:
        """Generate comprehensive legal response based on query type"""
        
        if context.query_type == QueryType.CONTRACT_DRAFT:
            return self._generate_contract_response(context, documents)
        elif context.query_type == QueryType.LEGAL_ANALYSIS:
            return self._generate_analysis_response(context, documents)
        else:
            return self._generate_qa_response(context, documents)
    
    def _generate_qa_response(self, context: QueryContext, documents: List[EnhancedLegalDocument]) -> LegalResponse:
        """Generate Q&A response with appropriate detail level"""
        
        if not documents:
            return self._generate_no_documents_response(context)
        
        # Prepare legal context
        legal_context = self._prepare_legal_context(context, documents)
        
        # Adapt response complexity to user expertise
        complexity_instructions = {
            ExpertiseLevel.NOVICE: "Expliquez de manière simple et pédagogique, avec des exemples concrets. Évitez le jargon juridique ou expliquez-le.",
            ExpertiseLevel.INTERMEDIATE: "Fournissez une explication équilibrée avec les concepts juridiques appropriés et des références pratiques.",
            ExpertiseLevel.EXPERT: "Donnez une analyse juridique approfondie avec les nuances, exceptions et jurisprudence pertinente."
        }
        
        response_prompt = f"""Vous êtes un éminent juriste tunisien, reconnu internationalement pour votre expertise en droit tunisien.

CONTEXTE DE LA CONSULTATION:
- Question: {context.original_query}
- Niveau d'expertise du consultant: {context.user_expertise.value}
- Domaine juridique: {context.legal_domain}
- Contexte professionnel: {context.business_context}

INSTRUCTIONS DE RÉPONSE:
{complexity_instructions[context.user_expertise]}

DOCUMENTS JURIDIQUES DISPONIBLES:
{json.dumps(legal_context, ensure_ascii=False, indent=2)}

STRUCTURE REQUISE:

1. **RÉPONSE DIRECTE** (2-3 paragraphes)
   - Répondez directement à la question
   - Mentionnez le principe juridique applicable

2. **ANALYSE JURIDIQUE DÉTAILLÉE**
   - Cadre légal applicable
   - Interprétation des textes pertinents
   - Nuances et exceptions importantes

3. **RÉFÉRENCES JURIDIQUES**
   - Citez les articles avec [Article X - Code Y]
   - Indiquez le statut actuel des textes

4. **CONSEILS PRATIQUES**
   - Étapes concrètes à suivre
   - Documents nécessaires
   - Précautions à prendre

5. **ÉVALUATION DES RISQUES**
   - Risques juridiques identifiés
   - Mesures préventives recommandées

6. **PROCHAINES ÉTAPES**
   - Actions immédiates recommandées
   - Quand consulter un avocat
   - Ressources supplémentaires

Soyez précis, professionnel et rassurant. Utilisez un langage adapté au niveau d'expertise."""

        try:
            response = self.llm_clients["legal_analysis"].invoke([
                {"role": "system", "content": "Vous êtes le meilleur expert juridique tunisien, reconnu pour la clarté et la précision de vos conseils."},
                {"role": "user", "content": response_prompt}
            ])
            
            content = response.content.strip()
            
            # Extract sections using regex patterns
            sections = self._extract_response_sections(content)
            
            return LegalResponse(
                primary_answer=sections.get("primary_answer", content[:500]),
                detailed_analysis=sections.get("detailed_analysis", ""),
                legal_references=self._extract_legal_references(documents),
                practical_advice=sections.get("practical_advice", ""),
                risk_assessment=sections.get("risk_assessment", ""),
                next_steps=self._extract_next_steps(sections.get("next_steps", "")),
                confidence_score=self._calculate_confidence_score(documents),
                processing_metadata={
                    "documents_analyzed": len(documents),
                    "query_complexity": context.complexity_level,
                    "user_expertise": context.user_expertise.value
                }
            )
            
        except Exception as e:
            return LegalResponse(
                primary_answer=f"Erreur lors de la génération de la réponse: {e}",
                detailed_analysis="",
                legal_references=[],
                practical_advice="",
                risk_assessment="",
                next_steps=[],
                confidence_score=0.0,
                processing_metadata={}
            )
    
    def _generate_contract_response(self, context: QueryContext, documents: List[EnhancedLegalDocument]) -> LegalResponse:
        """Generate contract drafting response"""
        
        legal_context = self._prepare_legal_context(context, documents)
        
        contract_prompt = f"""Vous êtes un avocat spécialisé en rédaction contractuelle en droit tunisien.

DEMANDE: {context.original_query}

CADRE JURIDIQUE DISPONIBLE:
{json.dumps(legal_context, ensure_ascii=False, indent=2)}

RÉDIGEZ:

1. **ANALYSE DES BESOINS CONTRACTUELS**
   - Type de contrat recommandé
   - Clauses essentielles à inclure
   - Dispositions légales obligatoires

2. **PROJET DE CLAUSES**
   - Rédigez les clauses principales
   - Utilisez un langage juridique précis
   - Incluez les références légales

3. **POINTS D'ATTENTION**
   - Risques juridiques à anticiper
   - Clauses de protection recommandées
   - Conformité réglementaire

4. **CONSEILS DE NÉGOCIATION**
   - Points négociables
   - Positions de force/faiblesse
   - Alternatives juridiques

Soyez précis et professionnel. Chaque clause doit être justifiée juridiquement."""

        try:
            response = self.llm_clients["contract_drafting"].invoke([
                {"role": "system", "content": "Vous êtes un expert en rédaction contractuelle, reconnu pour la qualité de vos projets de contrats."},
                {"role": "user", "content": contract_prompt}
            ])
            
            content = response.content.strip()
            sections = self._extract_response_sections(content)
            
            return LegalResponse(
                primary_answer=sections.get("primary_answer", content),
                detailed_analysis=sections.get("detailed_analysis", ""),
                legal_references=self._extract_legal_references(documents),
                practical_advice=sections.get("practical_advice", ""),
                risk_assessment=sections.get("risk_assessment", ""),
                next_steps=["Révision par un avocat", "Adaptation au cas spécifique", "Négociation des termes"],
                confidence_score=self._calculate_confidence_score(documents),
                processing_metadata={
                    "contract_type": "custom",
                    "documents_referenced": len(documents)
                }
            )
            
        except Exception as e:
            return self._generate_error_response(str(e))
    
    def _prepare_legal_context(self, context: QueryContext, documents: List[EnhancedLegalDocument]) -> List[Dict]:
        """Prepare structured legal context from documents"""
        
        legal_context = []
        for doc in documents:
            content = doc.content_fr if context.detected_language == "fr" else doc.content_ar
            if not content:
                content = doc.content_ar if context.detected_language == "fr" else doc.content_fr
            
            legal_context.append({
                "article_number": doc.article_number,
                "title": doc.title,
                "legal_code": doc.legal_code,
                "content": content,
                "summary": doc.summary,
                "status": doc.status,
                "relevance_score": doc.relevance_score,
                "tags": doc.tags
            })
        
        return legal_context
    
    def _extract_response_sections(self, content: str) -> Dict[str, str]:
        """Extract different sections from the response"""
        
        sections = {}
        
        # Define section patterns
        patterns = {
            "primary_answer": r"(?:RÉPONSE DIRECTE|1\.\s*\*\*RÉPONSE DIRECTE\*\*)(.*?)(?=\n\n(?:\d+\.|$))",
            "detailed_analysis": r"(?:ANALYSE JURIDIQUE|2\.\s*\*\*ANALYSE JURIDIQUE)(.*?)(?=\n\n(?:\d+\.|$))",
            "practical_advice": r"(?:CONSEILS PRATIQUES|4\.\s*\*\*CONSEILS PRATIQUES)(.*?)(?=\n\n(?:\d+\.|$))",
            "risk_assessment": r"(?:ÉVALUATION DES RISQUES|5\.\s*\*\*ÉVALUATION DES RISQUES)(.*?)(?=\n\n(?:\d+\.|$))",
            "next_steps": r"(?:PROCHAINES ÉTAPES|6\.\s*\*\*PROCHAINES ÉTAPES)(.*?)(?=\n\n(?:\d+\.|$))"
        }
        
        for section_name, pattern in patterns.items():
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                sections[section_name] = match.group(1).strip()
            else:
                sections[section_name] = ""
        
        # If no structured sections found, use first part as primary answer
        if not any(sections.values()):
            sections["primary_answer"] = content[:1000] if len(content) > 1000 else content
        
        return sections
    
    def _extract_legal_references(self, documents: List[EnhancedLegalDocument]) -> List[Dict]:
        """Extract formatted legal references"""
        
        references = []
        for doc in documents:
            references.append({
                "article": f"Article {doc.article_number}",
                "code": doc.legal_code,
                "title": doc.title,
                "status": doc.status,
                "relevance": doc.relevance_score
            })
        
        return references
    
    def _extract_next_steps(self, next_steps_text: str) -> List[str]:
        """Extract next steps as a list"""
        
        if not next_steps_text:
            return []
        
        # Split by lines and extract meaningful steps
        lines = next_steps_text.split('\n')
        steps = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('6.') and not line.startswith('**'):
                # Remove bullet points and numbering
                line = re.sub(r'^[-*•\d.)\s]+', '', line).strip()
                if len(line) > 10:  # Only meaningful steps
                    steps.append(line)
        
        return steps[:5]  # Limit to 5 steps
    
    def _calculate_confidence_score(self, documents: List[EnhancedLegalDocument]) -> float:
        """Calculate confidence score based on document quality and relevance"""
        
        if not documents:
            return 0.0
        
        # Factors affecting confidence
        avg_relevance = sum(doc.relevance_score for doc in documents) / len(documents)
        num_documents = min(len(documents), 10)  # Cap at 10 for scoring
        status_bonus = sum(1 for doc in documents if "actif" in doc.status.lower()) / len(documents)
        
        # Calculate weighted confidence score
        confidence = (avg_relevance * 0.5) + (num_documents * 5) + (status_bonus * 20)
        
        return min(confidence, 95.0)  # Cap at 95%
    
    def _generate_no_documents_response(self, context: QueryContext) -> LegalResponse:
        """Generate response when no relevant documents are found"""
        
        no_docs_prompt = f"""Aucun document spécifique n'a été trouvé pour cette question juridique:

"{context.original_query}"

En tant qu'expert juridique tunisien, fournissez:

1. **ANALYSE GÉNÉRALE**
   - Cadre juridique général applicable
   - Principes juridiques fondamentaux

2. **ORIENTATION PROCÉDURALE**
   - Démarches recommandées
   - Autorités compétentes à consulter

3. **RESSOURCES SUGGÉRÉES**
   - Codes juridiques à consulter
   - Jurisprudence pertinente

4. **RECOMMANDATIONS**
   - Consultation d'avocat spécialisé
   - Documents à préparer

Soyez informatif malgré l'absence de sources spécifiques."""

        try:
            response = self.llm_clients["legal_analysis"].invoke([
                {"role": "system", "content": "Vous êtes un expert juridique capable de donner des orientations même sans documents spécifiques."},
                {"role": "user", "content": no_docs_prompt}
            ])
            
            return LegalResponse(
                primary_answer=f"Je n'ai pas trouvé de documents spécifiques pour votre question, mais voici mon analyse basée sur les principes généraux du droit tunisien:\n\n{response.content}",
                detailed_analysis="",
                legal_references=[],
                practical_advice="Consultez un avocat spécialisé pour une analyse approfondie de votre situation spécifique.",
                risk_assessment="Sans documents juridiques précis, il est recommandé d'obtenir un avis juridique professionnel.",
                next_steps=["Consulter un avocat spécialisé", "Rechercher dans la jurisprudence", "Vérifier les textes réglementaires récents"],
                confidence_score=30.0,
                processing_metadata={"no_documents_found": True}
            )
            
        except Exception as e:
            return self._generate_error_response(str(e))
    
    def _generate_error_response(self, error_message: str) -> LegalResponse:
        """Generate error response"""
        
        return LegalResponse(
            primary_answer=f"Je rencontre des difficultés techniques pour traiter votre demande. Erreur: {error_message}",
            detailed_analysis="",
            legal_references=[],
            practical_advice="Veuillez reformuler votre question ou contacter le support technique.",
            risk_assessment="",
            next_steps=["Reformuler la question", "Contacter le support"],
            confidence_score=0.0,
            processing_metadata={"error": True, "error_message": error_message}
        )
    
    def process_legal_query(self, user_input: str, legal_code_filter: Optional[str] = None) -> Tuple[LegalResponse, QueryContext]:
        """Main method to process legal queries with full pipeline"""
        
        # 1. Analyze query intent and context
        context = self.analyze_query_intent(user_input)
        
        # 2. Enhanced document search
        documents = self.enhanced_document_search(context, legal_code_filter)
        
        # 3. Intelligent document ranking
        ranked_documents = self.intelligent_document_ranking(context, documents)
        
        # 4. Generate comprehensive response
        legal_response = self.generate_comprehensive_response(context, ranked_documents)
        
        return legal_response, context

# --- Enhanced Streamlit Interface ---
def main():
    # Professional header
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 1rem; margin-bottom: 2rem; color: white;'>
        <h1>🏛️ LegalGPT Pro</h1>
        <h3 style='margin: 0; font-weight: normal;'>Expert Juridique Intelligent • Droit Tunisien</h3>
        <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Assistant juridique professionnel pour entreprises et particuliers</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced sidebar
    with st.sidebar:
        st.markdown("### 🎯 Configuration Expert")
        
        # Legal domain filter
        legal_codes = {
            "🌐 Tous les domaines": None,
            "📊 Comptabilité publique": "code-comptabilite-publique",
            "🏢 Droit commercial": "code-commerce", 
            "👥 Droit civil": "code-civil",
            "⚖️ Droit pénal": "code-penal",
            "💼 Droit du travail": "code-travail",
            "📋 Procédure civile": "code-procedure-civile"
        }
        
        selected_domain = st.selectbox(
            "Domaine juridique",
            list(legal_codes.keys()),
            help="Concentrer la recherche sur un domaine spécifique"
        )
        legal_code_filter = legal_codes[selected_domain]
        
        # User expertise level
        expertise_levels = {
            "🔰 Débutant": ExpertiseLevel.NOVICE,
            "📚 Intermédiaire": ExpertiseLevel.INTERMEDIATE,
            "🎓 Expert": ExpertiseLevel.EXPERT
        }
        
        user_expertise = st.selectbox(
            "Votre niveau d'expertise",
            list(expertise_levels.keys()),
            help="Adapte la complexité des réponses"
        )
        
        # System status
        st.markdown("---")
        st.markdown("### 📊 Statut du Système")
        st.markdown("""
        🟢 **Opérationnel**  
        📚 Base juridique tunisienne  
        🤖 IA juridique avancée  
        🔒 Consultations sécurisées  
        """)
        
        # Quick help
        with st.expander("💡 Guide d'utilisation"):
            st.markdown("""
            **Types de requêtes supportées:**
            - ❓ Questions juridiques générales
            - 📝 Rédaction de contrats
            - 🔍 Analyse de conformité
            - ⚖️ Recherche de jurisprudence
            
            **Pour de meilleurs résultats:**
            - Soyez précis et détaillé
            - Mentionnez le contexte
            - Spécifiez le domaine si connu
            - Indiquez si c'est urgent
            """)
    
    # Initialize advanced expert system
    if "legal_expert" not in st.session_state:
        st.session_state.legal_expert = AdvancedLegalExpert()
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
        # Enhanced welcome message
        welcome_response = LegalResponse(
            primary_answer="""Bonjour et bienvenue sur LegalGPT Pro ! 

Je suis votre assistant juridique intelligent spécialisé en droit tunisien. Avec accès à une base de données juridique complète et des capacités d'analyse avancées, je peux vous aider avec:

🔹 **Consultations juridiques** - Questions sur le droit tunisien  
🔹 **Rédaction contractuelle** - Projets de contrats et clauses  
🔹 **Analyses de conformité** - Vérifications réglementaires  
🔹 **Recherche juridique** - Jurisprudence et doctrine  

Mon analyse s'adapte automatiquement à votre niveau d'expertise et au contexte de votre demande. Toutes mes réponses incluent des références juridiques précises et des conseils pratiques.

**Comment puis-je vous accompagner dans vos démarches juridiques aujourd'hui ?** 🤝""",
            detailed_analysis="",
            legal_references=[],
            practical_advice="",
            risk_assessment="",
            next_steps=[],
            confidence_score=100.0,
            processing_metadata={"welcome_message": True}
        )
        
        st.session_state.chat_messages.append({
            "role": "assistant", 
            "response": welcome_response,
            "context": None
        })
    
    # Display chat history with enhanced formatting
    for message in st.session_state.chat_messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(f'<div class="expert-header">🤖 LegalGPT Pro</div>', unsafe_allow_html=True)
                
                response = message["response"]
                context = message.get("context")
                
                # Display primary answer
                st.markdown(response.primary_answer)
                
                # Display additional sections if available
                if response.detailed_analysis:
                    with st.expander("📋 Analyse juridique détaillée", expanded=False):
                        st.markdown(response.detailed_analysis)
                
                if response.practical_advice:
                    with st.expander("💡 Conseils pratiques", expanded=False):
                        st.markdown(response.practical_advice)
                
                if response.risk_assessment:
                    with st.expander("⚠️ Évaluation des risques", expanded=False):
                        st.markdown(response.risk_assessment)
                
                if response.next_steps:
                    with st.expander("📋 Prochaines étapes recommandées", expanded=False):
                        for i, step in enumerate(response.next_steps, 1):
                            st.markdown(f"{i}. {step}")
                
                if response.legal_references:
                    with st.expander("📚 Références juridiques", expanded=False):
                        for ref in response.legal_references:
                            st.markdown(f"• **{ref['article']}** - {ref['code']}")
                            if ref.get('title'):
                                st.markdown(f"  *{ref['title']}*")
                
                # Display metadata
                if context and response.processing_metadata:
                    with st.expander("🔍 Métadonnées de l'analyse", expanded=False):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            complexity_class = f"complexity-{context.complexity_level}" if context else "complexity-basic"
                            st.markdown(f'<span class="complexity-indicator {complexity_class}">{context.complexity_level.upper() if context else "BASIC"}</span>', unsafe_allow_html=True)
                        
                        with col2:
                            st.metric("🎯 Confiance", f"{response.confidence_score:.1f}%")
                        
                        with col3:
                            docs_count = response.processing_metadata.get("documents_analyzed", 0)
                            st.metric("📄 Documents", docs_count)
                        
                        with col4:
                            if context:
                                st.metric("🧠 Type", context.query_type.value.upper())
    
    # Enhanced chat input with processing indicators
    if user_query := st.chat_input("Posez votre question juridique ou décrivez votre besoin..."):
        # Add user message
        st.session_state.chat_messages.append({
            "role": "user", 
            "content": user_query
        })
        
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Process expert response
        with st.chat_message("assistant"):
            st.markdown(f'<div class="expert-header">🤖 LegalGPT Pro</div>', unsafe_allow_html=True)
            
            # Progressive thinking indicators
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            try:
                # Step 1: Query analysis
                status_placeholder.markdown('<div class="thinking-indicator">🧠 Analyse de votre demande et détection d\'intention...</div>', unsafe_allow_html=True)
                progress_bar.progress(20)
                time.sleep(0.5)
                
                # Step 2: Document search
                status_placeholder.markdown('<div class="thinking-indicator">🔍 Recherche dans la base juridique tunisienne...</div>', unsafe_allow_html=True)
                progress_bar.progress(40)
                time.sleep(0.5)
                
                # Step 3: Document ranking
                status_placeholder.markdown('<div class="thinking-indicator">📊 Classification intelligente des documents juridiques...</div>', unsafe_allow_html=True)
                progress_bar.progress(60)
                time.sleep(0.5)
                
                # Step 4: Response generation
                status_placeholder.markdown('<div class="thinking-indicator">✍️ Génération de la réponse juridique experte...</div>', unsafe_allow_html=True)
                progress_bar.progress(80)
                
                # Generate response
                legal_response, query_context = st.session_state.legal_expert.process_legal_query(
                    user_query, 
                    legal_code_filter=legal_code_filter
                )
                
                progress_bar.progress(100)
                
                # Clear progress indicators
                status_placeholder.empty()
                progress_bar.empty()
                
                # Display response
                st.markdown(legal_response.primary_answer)
                
                # Store response
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "response": legal_response,
                    "context": query_context
                })
                
                # Display additional sections
                if legal_response.detailed_analysis:
                    with st.expander("📋 Analyse juridique détaillée", expanded=False):
                        st.markdown(legal_response.detailed_analysis)
                
                if legal_response.practical_advice:
                    with st.expander("💡 Conseils pratiques", expanded=False):
                        st.markdown(legal_response.practical_advice)
                
                if legal_response.risk_assessment:
                    with st.expander("⚠️ Évaluation des risques", expanded=False):
                        st.markdown(legal_response.risk_assessment)
                
                if legal_response.next_steps:
                    with st.expander("📋 Prochaines étapes recommandées", expanded=False):
                        for i, step in enumerate(legal_response.next_steps, 1):
                            st.markdown(f"{i}. {step}")
                
                if legal_response.legal_references:
                    with st.expander("📚 Références juridiques", expanded=False):
                        for ref in legal_response.legal_references:
                            st.markdown(f"• **{ref['article']}** - {ref['code']}")
                            if ref.get('title'):
                                st.markdown(f"  *{ref['title']}*")
                
                # Display analysis metadata
                with st.expander("🔍 Métadonnées de l'analyse", expanded=False):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        complexity_class = f"complexity-{query_context.complexity_level}"
                        st.markdown(f'<span class="complexity-indicator {complexity_class}">{query_context.complexity_level.upper()}</span>', unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("🎯 Confiance", f"{legal_response.confidence_score:.1f}%")
                    
                    with col3:
                        docs_count = legal_response.processing_metadata.get("documents_analyzed", 0)
                        st.metric("📄 Documents", docs_count)
                    
                    with col4:
                        st.metric("🧠 Type", query_context.query_type.value.upper())
                
            except Exception as e:
                status_placeholder.empty()
                progress_bar.empty()
                
                error_msg = f"""🔧 **Erreur technique détectée**

Je rencontre des difficultés pour traiter votre demande. Détails techniques: `{str(e)}`

**Suggestions:**
- Reformulez votre question de manière plus simple
- Vérifiez la connexion internet
- Contactez le support si le problème persiste

**Support technique:** legalgpt.support@example.com"""
                
                st.error(error_msg)
                
                # Store error response
                error_response = LegalResponse(
                    primary_answer=error_msg,
                    detailed_analysis="",
                    legal_references=[],
                    practical_advice="",
                    risk_assessment="",
                    next_steps=["Reformuler la question", "Contacter le support"],
                    confidence_score=0.0,
                    processing_metadata={"error": True}
                )
                
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "response": error_response,
                    "context": None
                })
    
    # Enhanced footer with actions
    st.markdown("---")
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        st.markdown("""
        <div style='font-size: 0.85em; color: #666; line-height: 1.4;'>
        ⚠️ <strong>Avertissement légal:</strong> Les réponses fournies sont à titre informatif et ne constituent pas un avis juridique personnalisé. 
        Pour des situations complexes, consultez un avocat qualifié.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🗑️ Nouvelle session"):
            st.session_state.chat_messages = []
            st.rerun()
    
    with col3:
        if st.button("📊 Statistiques"):
            total_queries = len([m for m in st.session_state.chat_messages if m["role"] == "user"])
            avg_confidence = np.mean([
                m["response"].confidence_score 
                for m in st.session_state.chat_messages 
                if m["role"] == "assistant" and hasattr(m["response"], "confidence_score")
            ]) if total_queries > 0 else 0
            
            st.info(f"""📈 **Session Statistics**
            - Requêtes traitées: {total_queries}
            - Confiance moyenne: {avg_confidence:.1f}%
            - Système: Opérationnel ✅""")
    
    with col4:
        if st.button("📞 Support Pro"):
            st.info("""🏢 **Support Professionnel**
            
            📧 support@legalgpt.tn  
            📱 +216 XX XXX XXX  
            💬 Chat en direct 24/7  
            
            🎯 Support entreprise disponible""")

if __name__ == "__main__":
    # Add numpy import for statistics
    import numpy as np
    main()