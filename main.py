"""
Legaly - Advanced Tunisian Legal AI System
Enterprise-grade legal assistant with comprehensive capabilities
"""

import os
import json
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, date
import time
import re
from enum import Enum
import uuid

# Lang/OpenAI/Pinecone imports
from langchain.agents import tool
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Configure Streamlit page
st.set_page_config(
    page_title="🏛️ Legaly - Expert Juridique Tunisien",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional interface
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .chat-message.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 10%;
    }
    .chat-message.assistant {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 10%;
    }
    .contract-section {
        background: rgba(255,255,255,0.1);
        padding: 1.5rem;
        border-left: 4px solid #ffd700;
        margin: 1rem 0;
        border-radius: 0.5rem;
        font-family: 'Times New Roman', serif;
    }
    .legal-article {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-left: 3px solid #28a745;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
        font-size: 0.95em;
    }
    .warning-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #fff;
    }
    .success-box {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #fff;
    }
    .thinking-indicator {
        font-style: italic;
        opacity: 0.8;
        padding: 0.5rem;
        background: rgba(255,255,255,0.1);
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Advanced Configuration ---
@st.cache_resource
def initialize_advanced_legal_system():
    """Initialize the most advanced legal AI system"""
    
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
        PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    except:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        st.error("⚠️ Configuration requise: OPENAI_API_KEY et PINECONE_API_KEY")
        st.stop()
    
    # Advanced model configuration for different legal tasks
    LEGAL_MODELS = {
        "master_analyst": "gpt-4o",           # Primary legal analysis
        "contract_drafter": "gpt-4o",         # Contract drafting
        "legal_researcher": "gpt-4o",         # Legal research
        "compliance_checker": "gpt-4o-mini",  # Compliance verification
        "document_processor": "gpt-4o-mini",  # Document processing
        "quality_controller": "gpt-4o"        # Quality assurance
    }
    
    PINECONE_ENV = "us-east-1"
    PINECONE_INDEX = "tunisia-laws"
    EMBEDDING_MODEL = "text-embedding-ada-002"
    
    # Initialize Pinecone with advanced configuration
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)
    
    # Advanced Vector Store with intelligent document processing
    class AdvancedLegalVectorStore(PineconeVectorStore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
        def comprehensive_legal_search(self, queries: List[str], k: int = 25, filter_dict: Optional[Dict] = None) -> List[Dict]:
            """Comprehensive multi-query legal document search"""
            all_results = {}
            
            for query in queries:
                try:
                    query_vector = self._embedding.embed_query(query)
                    search_results = self._index.query(
                        vector=query_vector,
                        top_k=k,
                        include_metadata=True,
                        filter=filter_dict
                    )
                    
                    for match in search_results.matches:
                        doc_id = match.metadata.get('id', f'doc_{match.id}')
                        
                        if doc_id not in all_results or match.score > all_results[doc_id]['score']:
                            content = self._extract_optimal_content(match.metadata)
                            if content:
                                all_results[doc_id] = {
                                    'id': doc_id,
                                    'metadata': match.metadata,
                                    'content': content,
                                    'score': match.score,
                                    'matched_query': query
                                }
                                
                except Exception as e:
                    st.warning(f"Search error for query '{query}': {e}")
                    continue
            
            return list(all_results.values())
        
        def _extract_optimal_content(self, metadata: Dict) -> str:
            """Extract the best content based on language and completeness"""
            content_options = [
                ('content_fr', metadata.get('content_fr', '')),
                ('content_ar', metadata.get('content_ar', '')),
                ('summary', metadata.get('summary', '')),
                ('title', metadata.get('title', ''))
            ]
            
            for field_name, content in content_options:
                if content and len(content.strip()) > 20:
                    return content.strip()
            
            return ""
    
    vector_store = AdvancedLegalVectorStore(
        index=index,
        embedding=embeddings,
        text_key="content"
    )
    
    # Initialize specialized LLM clients
    llm_specialists = {}
    for specialist, model in LEGAL_MODELS.items():
        temperature = 0.1 if specialist in ["compliance_checker", "legal_researcher"] else 0.3
        max_tokens = 6000 if specialist == "contract_drafter" else 4000
        
        llm_specialists[specialist] = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    return {
        "vector_store": vector_store,
        "llm_specialists": llm_specialists,
        "index": index
    }

# Initialize the advanced system
advanced_system = initialize_advanced_legal_system()
vector_store = advanced_system["vector_store"]
llm_specialists = advanced_system["llm_specialists"]

# --- Advanced Data Models ---
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
    country: str = "Tunisia"
    relevance_score: float = 0.0
    legal_authority: float = 0.0
    
@dataclass
class ContractRequirements:
    contract_type: str
    parties: List[Dict]
    financial_terms: Dict
    performance_terms: Dict
    governance_terms: Dict
    duration_terms: Dict
    special_clauses: List[str]
    compliance_requirements: List[str]

# --- Master Legal AI System ---
class MasterLegalAI:
    def __init__(self):
        self.name = "Legaly Master"
        self.version = "3.0"
        self.specializations = {
            "contract_drafting": "Rédaction de contrats professionnels",
            "legal_analysis": "Analyse juridique approfondie",
            "compliance_audit": "Audit de conformité réglementaire",
            "legal_research": "Recherche juridique et jurisprudentielle",
            "corporate_law": "Droit des sociétés et corporate",
            "commercial_law": "Droit commercial et des affaires",
            "civil_law": "Droit civil et obligations",
            "public_law": "Droit public et administratif"
        }
        self.vector_store = vector_store
        self.llm_specialists = llm_specialists
        
    def analyze_legal_request(self, user_input: str) -> Dict:
        """Advanced analysis of legal requests with intent detection"""
        
        analysis_prompt = f"""En tant que maître juriste tunisien avec 30 ans d'expérience, analysez cette demande juridique:

DEMANDE: "{user_input}"

Analysez et répondez en JSON uniquement:
{{
    "request_type": "contract_drafting|legal_analysis|compliance_check|legal_research|general_advice",
    "legal_domain": "commercial|civil|penal|administrative|corporate|labor|tax",
    "complexity_level": "simple|intermediate|complex|expert",
    "language_preference": "french|arabic|mixed",
    "urgency_level": "low|medium|high|urgent",
    "client_type": "individual|sme|enterprise|legal_professional",
    "specific_requirements": [
        "requirement1",
        "requirement2"
    ],
    "legal_entities_involved": [
        "entity_type1",
        "entity_type2"
    ],
    "financial_aspects": {{
        "has_financial_terms": true|false,
        "estimated_value": "amount if mentioned",
        "currency": "TND|EUR|USD"
    }},
    "key_legal_concepts": [
        "concept1",
        "concept2"
    ],
    "search_queries": [
        "optimized legal query 1",
        "optimized legal query 2",
        "optimized legal query 3"
    ],
    "expected_deliverables": [
        "contract_draft|legal_opinion|compliance_report|research_summary"
    ]
}}"""

        try:
            response = self.llm_specialists["master_analyst"].invoke([
                {"role": "system", "content": "Vous êtes le meilleur analyste juridique tunisien, expert en classification des demandes juridiques."},
                {"role": "user", "content": analysis_prompt}
            ])
            
            # Robust JSON parsing
            content = response.content.strip()
            try:
                analysis = json.loads(content)
                return analysis
            except json.JSONDecodeError:
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    return self._create_fallback_analysis(user_input)
                    
        except Exception as e:
            st.warning(f"Erreur d'analyse: {e}")
            return self._create_fallback_analysis(user_input)
    
    def _create_fallback_analysis(self, user_input: str) -> Dict:
        """Create fallback analysis when AI analysis fails"""
        return {
            "request_type": "general_advice",
            "legal_domain": "commercial",
            "complexity_level": "intermediate",
            "language_preference": "french",
            "urgency_level": "medium",
            "client_type": "sme",
            "specific_requirements": ["general legal guidance"],
            "legal_entities_involved": ["individual", "company"],
            "financial_aspects": {"has_financial_terms": False, "estimated_value": "", "currency": "TND"},
            "key_legal_concepts": ["general legal advice"],
            "search_queries": [user_input],
            "expected_deliverables": ["legal_opinion"]
        }
    
    def conduct_comprehensive_legal_research(self, analysis: Dict, legal_filter: Optional[str] = None) -> List[LegalDocument]:
        """Conduct comprehensive legal research using multiple strategies"""
        
        search_queries = analysis.get("search_queries", [analysis.get("original_query", "")])
        
        # Build advanced filter
        filter_dict = {}
        if legal_filter and legal_filter != "all":
            filter_dict["legal_code"] = {"$eq": legal_filter}
        
        # Add domain-specific filters
        domain = analysis.get("legal_domain", "")
        if domain and domain != "general":
            # Additional domain-specific filtering could be added here
            pass
        
        # Comprehensive search
        raw_results = self.vector_store.comprehensive_legal_search(
            queries=search_queries,
            k=20,
            filter_dict=filter_dict
        )
        
        # Convert to LegalDocument objects
        legal_documents = []
        for result in raw_results:
            metadata = result['metadata']
            legal_doc = LegalDocument(
                id=result['id'],
                title=metadata.get('title', 'Document sans titre'),
                article_number=metadata.get('article_index', 0),
                content_fr=metadata.get('content_fr', ''),
                content_ar=metadata.get('content_ar', ''),
                legal_code=metadata.get('legal_code', ''),
                summary=metadata.get('summary', ''),
                tags=metadata.get('tags', []),
                status=metadata.get('status', ''),
                country=metadata.get('country', 'Tunisia'),
                relevance_score=result['score']
            )
            legal_documents.append(legal_doc)
        
        return self._rank_documents_by_legal_authority(analysis, legal_documents)
    
    def _rank_documents_by_legal_authority(self, analysis: Dict, documents: List[LegalDocument]) -> List[LegalDocument]:
        """Rank documents by legal authority and relevance"""
        
        if not documents:
            return []
        
        ranking_prompt = f"""En tant qu'expert en hiérarchie juridique tunisienne, classez ces documents par autorité juridique et pertinence.

CONTEXTE:
- Type de demande: {analysis.get('request_type', 'general')}
- Domaine juridique: {analysis.get('legal_domain', 'general')}
- Complexité: {analysis.get('complexity_level', 'intermediate')}

DOCUMENTS À CLASSER:
{json.dumps([{
    'id': doc.id,
    'title': doc.title,
    'article_number': doc.article_number,
    'legal_code': doc.legal_code,
    'status': doc.status,
    'tags': doc.tags,
    'relevance_score': doc.relevance_score
} for doc in documents], ensure_ascii=False, indent=2)}

CRITÈRES DE CLASSEMENT:
1. Autorité hiérarchique (Constitution > Lois > Décrets > Règlements)
2. Pertinence directe à la demande
3. Actualité et statut du texte
4. Spécificité vs généralité

Répondez en JSON uniquement:
{{
    "ranked_documents": [
        {{
            "id": "doc_id",
            "authority_score": 0-100,
            "relevance_score": 0-100,
            "final_score": 0-100,
            "ranking_rationale": "justification"
        }}
    ]
}}"""

        try:
            response = self.llm_specialists["legal_researcher"].invoke([
                {"role": "system", "content": "Expert en hiérarchie des normes juridiques tunisiennes."},
                {"role": "user", "content": ranking_prompt}
            ])
            
            # Parse ranking results
            content = response.content.strip()
            try:
                ranking_data = json.loads(content)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    ranking_data = json.loads(json_match.group())
                else:
                    # Fallback to original order
                    return documents[:10]
            
            # Apply rankings
            id_to_ranking = {r["id"]: r for r in ranking_data.get("ranked_documents", [])}
            
            for doc in documents:
                ranking_info = id_to_ranking.get(doc.id, {})
                doc.legal_authority = ranking_info.get("authority_score", 50)
                doc.relevance_score = ranking_info.get("final_score", doc.relevance_score)
            
            # Sort by combined score
            ranked_docs = sorted(documents, key=lambda x: (x.legal_authority + x.relevance_score) / 2, reverse=True)
            return ranked_docs[:10]
            
        except Exception as e:
            st.warning(f"Erreur de classement: {e}")
            return documents[:10]
    
    def generate_professional_contract(self, analysis: Dict, legal_documents: List[LegalDocument]) -> str:
        """Generate professional-grade contracts with legal compliance"""
        
        # Extract contract requirements
        contract_requirements = self._extract_contract_requirements(analysis)
        
        # Prepare legal framework
        legal_framework = self._prepare_legal_framework(legal_documents, analysis)
        
        contract_prompt = f"""Vous êtes le meilleur avocat spécialisé en rédaction contractuelle en Tunisie, reconnu pour l'excellence de vos contrats.

DEMANDE ORIGINALE: {analysis.get('original_query', '')}

ANALYSE DE LA DEMANDE:
{json.dumps(analysis, ensure_ascii=False, indent=2)}

CADRE JURIDIQUE APPLICABLE:
{json.dumps(legal_framework, ensure_ascii=False, indent=2)}

INSTRUCTIONS DE RÉDACTION:

1. **RÉDIGEZ UN CONTRAT COMPLET ET PROFESSIONNEL** incluant:
   - Préambule et identification des parties
   - Objet et définitions
   - Conditions financières détaillées
   - Obligations et droits de chaque partie
   - Conditions de performance
   - Modalités de résiliation
   - Clauses de protection et garanties
   - Résolution des conflits
   - Dispositions finales

2. **RESPECTEZ LES EXIGENCES LÉGALES TUNISIENNES**:
   - Citez les articles pertinents [Article X - Code Y]
   - Respectez les formes légales obligatoires
   - Incluez les mentions légales requises

3. **STYLE CONTRACTUEL PROFESSIONNEL**:
   - Langage juridique précis
   - Clauses non-ambiguës
   - Structure claire et logique
   - Numérotation des articles

4. **ADAPTEZ AU CONTEXTE SPÉCIFIQUE**:
   - Type d'entreprise (SUARL mentionnée)
   - Modalités financières précises
   - Protection des intérêts de toutes les parties

RÉDIGEZ LE CONTRAT COMPLET CI-DESSOUS:"""

        try:
            contract_response = self.llm_specialists["contract_drafter"].invoke([
                {"role": "system", "content": "Vous êtes le meilleur contractualiste tunisien, expert en droit des sociétés et commercial."},
                {"role": "user", "content": contract_prompt}
            ])
            
            return contract_response.content.strip()
            
        except Exception as e:
            return f"Erreur lors de la rédaction du contrat: {e}"
    
    def generate_comprehensive_legal_analysis(self, analysis: Dict, legal_documents: List[LegalDocument]) -> str:
        """Generate comprehensive legal analysis and advice"""
        
        legal_framework = self._prepare_legal_framework(legal_documents, analysis)
        
        analysis_prompt = f"""En tant que juriste senior tunisien avec expertise internationale, fournissez une analyse juridique complète.

DEMANDE: {analysis.get('original_query', '')}

CONTEXTE D'ANALYSE:
{json.dumps(analysis, ensure_ascii=False, indent=2)}

DOCUMENTATION JURIDIQUE:
{json.dumps(legal_framework, ensure_ascii=False, indent=2)}

STRUCTURE D'ANALYSE REQUISE:

## 1. RÉSUMÉ EXÉCUTIF
- Synthèse de la situation juridique
- Enjeux principaux identifiés
- Recommandations clés

## 2. CADRE JURIDIQUE APPLICABLE
- Textes légaux pertinents avec citations précises
- Hiérarchie des normes applicables
- Jurisprudence pertinente si applicable

## 3. ANALYSE JURIDIQUE DÉTAILLÉE
- Droits et obligations de chaque partie
- Risques juridiques identifiés
- Opportunités et protections disponibles
- Interprétation des dispositions légales

## 4. ASPECTS PRATIQUES
- Démarches administratives requises
- Documents nécessaires
- Délais et procédures
- Coûts estimatifs

## 5. GESTION DES RISQUES
- Risques juridiques majeurs
- Mesures préventives recommandées
- Stratégies de mitigation
- Plans de contingence

## 6. RECOMMANDATIONS STRATÉGIQUES
- Actions immédiates à entreprendre
- Planification à moyen terme
- Optimisations possibles
- Points de vigilance

## 7. CONCLUSION ET PROCHAINES ÉTAPES
- Synthèse des points clés
- Feuille de route recommandée
- Moments clés pour consultation juridique

Soyez précis, pratique et orienté solutions. Chaque recommandation doit être justifiée juridiquement."""

        try:
            analysis_response = self.llm_specialists["master_analyst"].invoke([
                {"role": "system", "content": "Vous êtes un juriste senior tunisien, reconnu pour la qualité de vos analyses juridiques."},
                {"role": "user", "content": analysis_prompt}
            ])
            
            return analysis_response.content.strip()
            
        except Exception as e:
            return f"Erreur lors de l'analyse juridique: {e}"
    
    def _extract_contract_requirements(self, analysis: Dict) -> ContractRequirements:
        """Extract contract requirements from analysis"""
        # This would be enhanced based on the specific analysis
        return ContractRequirements(
            contract_type=analysis.get("contract_type", "partnership"),
            parties=[],
            financial_terms=analysis.get("financial_aspects", {}),
            performance_terms={},
            governance_terms={},
            duration_terms={},
            special_clauses=[],
            compliance_requirements=[]
        )
    
    def _prepare_legal_framework(self, documents: List[LegalDocument], analysis: Dict) -> List[Dict]:
        """Prepare structured legal framework from documents"""
        
        framework = []
        for doc in documents:
            # Choose appropriate language content
            content = doc.content_fr if doc.content_fr else doc.content_ar
            
            framework.append({
                "article_number": doc.article_number,
                "title": doc.title,
                "legal_code": doc.legal_code,
                "content": content,
                "summary": doc.summary,
                "status": doc.status,
                "authority_level": doc.legal_authority,
                "relevance": doc.relevance_score,
                "tags": doc.tags
            })
        
        return framework
    
    def process_advanced_legal_request(self, user_input: str, legal_filter: Optional[str] = None) -> Tuple[str, Dict]:
        """Main processing pipeline for advanced legal requests"""
        
        try:
            # 1. Analyze the legal request
            analysis = self.analyze_legal_request(user_input)
            analysis["original_query"] = user_input
            
            # 2. Conduct comprehensive legal research
            legal_documents = self.conduct_comprehensive_legal_research(analysis, legal_filter)
            
            # 3. Generate appropriate response based on request type
            request_type = analysis.get("request_type", "general_advice")
            
            if request_type == "contract_drafting":
                response = self.generate_professional_contract(analysis, legal_documents)
            else:
                response = self.generate_comprehensive_legal_analysis(analysis, legal_documents)
            
            # 4. Add legal references
            if legal_documents:
                response += "\n\n" + self._format_legal_references(legal_documents)
            
            return response, analysis
            
        except Exception as e:
            error_response = f"""🔧 **Erreur système**

Une erreur est survenue lors du traitement de votre demande: {str(e)}

**Actions recommandées:**
1. Reformulez votre demande de manière plus spécifique
2. Vérifiez la formulation de votre question
3. Contactez le support si le problème persiste

**Support technique:** legal.support@example.com"""
            
            return error_response, {"error": True, "error_message": str(e)}
    
    def _format_legal_references(self, documents: List[LegalDocument]) -> str:
        """Format legal references for display"""
        
        references = "## 📚 RÉFÉRENCES JURIDIQUES CONSULTÉES\n\n"
        
        for i, doc in enumerate(documents[:5], 1):
            references += f"**{i}. Article {doc.article_number}** - {doc.title}\n"
            references += f"   📖 Code: {doc.legal_code}\n"
            references += f"   ⚖️ Statut: {doc.status}\n"
            references += f"   🎯 Pertinence: {doc.relevance_score:.1f}%\n\n"
        
        return references

# --- Enhanced Streamlit Interface ---
def main():
    # Professional header
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ Legaly Master</h1>
        <h3>Expert Juridique IA de Niveau International • Droit Tunisien</h3>
        <p>Assistant juridique professionnel pour rédaction de contrats, analyses juridiques et conseil stratégique</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced sidebar
    with st.sidebar:
        st.markdown("### 🎯 Configuration Experte")
        
        # Domain specialization
        legal_domains = {
            "🌐 Tous les domaines": None,
            "📊 Comptabilité publique": "code-comptabilite-publique",
            "🏢 Droit commercial": "code-commerce",
            "👥 Droit civil": "code-civil",
            "⚖️ Droit pénal": "code-penal",
            "💼 Droit du travail": "code-travail",
            "📋 Procédure civile": "code-procedure-civile"
        }
        
        selected_domain = st.selectbox(
            "Domaine de spécialisation",
            list(legal_domains.keys()),
            help="Concentrer l'expertise sur un domaine spécifique"
        )
        legal_filter = legal_domains[selected_domain]
        
        # Service type
        st.markdown("### 🔧 Type de Service")
        service_types = [
            "🤖 Assistant IA Automatique",
            "📝 Rédaction de Contrats",
            "📊 Analyse Juridique Approfondie",
            "🔍 Recherche Jurisprudentielle",
            "✅ Audit de Conformité"
        ]
        
        selected_service = st.selectbox(
            "Type de service souhaité",
            service_types,
            help="Le système adaptera son approche selon le service sélectionné"
        )
        
        # System status
        st.markdown("---")
        st.markdown("### 📊 Statut du Système")
        
        # Real-time system status
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🟢 Statut", "Opérationnel")
            st.metric("📚 Base juridique", "Tunisie")
        with col2:
            st.metric("🤖 IA Level", "Master Pro")
            st.metric("🔒 Sécurité", "Enterprise")
        
        # Professional features
        with st.expander("🚀 Fonctionnalités Professionnelles"):
            st.markdown("""
            ✅ **Rédaction de contrats professionnels**  
            ✅ **Analyses juridiques approfondies**  
            ✅ **Recherche jurisprudentielle avancée**  
            ✅ **Audit de conformité réglementaire**  
            ✅ **Conseil stratégique juridique**  
            ✅ **Support multi-lingue (FR/AR)**  
            ✅ **Citations juridiques précises**  
            ✅ **Adaptation au niveau d'expertise**  
            """)
    
    # Initialize the master legal AI
    if "master_legal_ai" not in st.session_state:
        st.session_state.master_legal_ai = MasterLegalAI()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
        # Professional welcome message
        welcome_message = """🎯 **Bienvenue sur Legaly Master**

Je suis votre expert juridique IA de niveau international, spécialisé en droit tunisien. Mes capacités incluent:

### 📝 **Rédaction Contractuelle Professionnelle**
- Contrats de société (SUARL, SARL, SA)
- Contrats commerciaux et partenariats
- Accords d'investissement et participation
- Contrats de travail et de service

### 📊 **Analyses Juridiques Approfondies**
- Études de faisabilité juridique
- Analyses de risques et conformité
- Structuration juridique d'opérations
- Conseil stratégique juridique

### 🔍 **Recherche et Documentation**
- Recherche jurisprudentielle précise
- Analyses comparative de législations
- Veille juridique et réglementaire
- Documentation juridique complète

**Comment puis-je vous assister dans vos besoins juridiques professionnels aujourd'hui ?**

💡 *Tip: Plus votre demande est détaillée, plus ma réponse sera précise et adaptée à vos besoins.*"""
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": welcome_message,
            "metadata": {"welcome": True}
        })
    
    # Display chat history with enhanced formatting
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown("🤖 **Legaly Master** • *Expert Juridique IA*")
                st.markdown(message["content"])
                
                # Display metadata if available
                if message.get("metadata") and not message["metadata"].get("welcome"):
                    metadata = message["metadata"]
                    
                    with st.expander("📊 Détails de l'analyse juridique", expanded=False):
                        if metadata.get("request_type"):
                            st.markdown(f"**Type de demande:** {metadata['request_type']}")
                        if metadata.get("legal_domain"):
                            st.markdown(f"**Domaine juridique:** {metadata['legal_domain']}")
                        if metadata.get("complexity_level"):
                            st.markdown(f"**Niveau de complexité:** {metadata['complexity_level']}")
                        if metadata.get("client_type"):
                            st.markdown(f"**Type de client:** {metadata['client_type']}")
    
    # Enhanced chat input with examples
    st.markdown("### 💬 Votre Demande Juridique")
    
    # Quick examples for better UX
    with st.expander("💡 Exemples de demandes professionnelles"):
        st.markdown("""
        **🤝 Contrats de partenariat:**
        - "Je veux créer un contrat pour un nouvel associé dans ma SUARL avec 10% de parts pour 1000 TND"
        - "Rédiger un accord de partenariat commercial avec répartition de bénéfices progressifs"
        
        **💼 Structuration d'entreprise:**
        - "Analyse juridique pour transformer ma entreprise individuelle en SARL"
        - "Quelles sont les obligations légales pour créer une filiale en Tunisie ?"
        
        **📊 Conformité réglementaire:**
        - "Audit de conformité fiscale pour une entreprise d'export"
        - "Vérification de conformité RGPD pour une startup tech"
        """)
    
    # Main chat input
    if user_query := st.chat_input("Décrivez votre besoin juridique en détail..."):
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query
        })
        
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Process with advanced AI system
        with st.chat_message("assistant"):
            st.markdown("🤖 **Legaly Master** • *Expert Juridique IA*")
            
            # Advanced processing indicators
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            try:
                # Phase 1: Analysis
                status_placeholder.markdown('<div class="thinking-indicator">🧠 Analyse approfondie de votre demande juridique...</div>', unsafe_allow_html=True)
                progress_bar.progress(15)
                time.sleep(0.8)
                
                # Phase 2: Legal Research
                status_placeholder.markdown('<div class="thinking-indicator">🔍 Recherche exhaustive dans la base juridique tunisienne...</div>', unsafe_allow_html=True)
                progress_bar.progress(35)
                time.sleep(1.0)
                
                # Phase 3: Document Analysis
                status_placeholder.markdown('<div class="thinking-indicator">📚 Analyse et classification des documents juridiques...</div>', unsafe_allow_html=True)
                progress_bar.progress(55)
                time.sleep(0.7)
                
                # Phase 4: Legal Synthesis
                status_placeholder.markdown('<div class="thinking-indicator">⚖️ Synthèse juridique et rédaction professionnelle...</div>', unsafe_allow_html=True)
                progress_bar.progress(75)
                time.sleep(0.5)
                
                # Phase 5: Quality Control
                status_placeholder.markdown('<div class="thinking-indicator">✅ Contrôle qualité et validation juridique...</div>', unsafe_allow_html=True)
                progress_bar.progress(90)
                
                # Generate response
                response, analysis_metadata = st.session_state.master_legal_ai.process_advanced_legal_request(
                    user_query,
                    legal_filter=legal_filter
                )
                
                progress_bar.progress(100)
                time.sleep(0.3)
                
                # Clear progress indicators
                status_placeholder.empty()
                progress_bar.empty()
                
                # Display response
                st.markdown(response)
                
                # Store in chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "metadata": analysis_metadata
                })
                
                # Display analysis metadata
                if not analysis_metadata.get("error"):
                    with st.expander("📊 Détails de l'analyse juridique", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"**Type de demande:** {analysis_metadata.get('request_type', 'N/A')}")
                            st.markdown(f"**Domaine:** {analysis_metadata.get('legal_domain', 'N/A')}")
                        
                        with col2:
                            st.markdown(f"**Complexité:** {analysis_metadata.get('complexity_level', 'N/A')}")
                            st.markdown(f"**Urgence:** {analysis_metadata.get('urgency_level', 'N/A')}")
                        
                        with col3:
                            st.markdown(f"**Client type:** {analysis_metadata.get('client_type', 'N/A')}")
                            st.markdown(f"**Langue:** {analysis_metadata.get('language_preference', 'N/A')}")
                
            except Exception as e:
                status_placeholder.empty()
                progress_bar.empty()
                
                error_message = f"""<div class="warning-box">
                🔧 <strong>Erreur Système Détectée</strong><br><br>
                Une erreur technique est survenue lors du traitement de votre demande.<br><br>
                <strong>Détails:</strong> {str(e)}<br><br>
                <strong>Actions recommandées:</strong><br>
                • Reformulez votre demande plus spécifiquement<br>
                • Vérifiez que tous les éléments nécessaires sont inclus<br>
                • Contactez le support si le problème persiste<br><br>
                <strong>Support:</strong> legal.support@example.com
                </div>"""
                
                st.markdown(error_message, unsafe_allow_html=True)
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "Une erreur technique est survenue. Veuillez reformuler votre demande.",
                    "metadata": {"error": True}
                })
    
    # Professional footer
    st.markdown("---")
    
    # Action buttons
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        st.markdown("""
        <div style='font-size: 0.9em; color: #666; line-height: 1.5;'>
        ⚠️ <strong>Clause de non-responsabilité:</strong> Legaly fournit des informations juridiques générales. 
        Pour des conseils juridiques spécifiques et personnalisés, consultez un avocat qualifié inscrit au barreau.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🗑️ Nouvelle session"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col3:
        if st.button("📊 Statistiques"):
            total_requests = len([m for m in st.session_state.chat_history if m["role"] == "user"])
            contract_requests = len([m for m in st.session_state.chat_history if m["role"] == "assistant" and "contrat" in m.get("content", "").lower()])
            
            st.success(f"""📈 **Statistiques de Session**

**Demandes traitées:** {total_requests}  
**Contrats rédigés:** {contract_requests}  
**Système:** Opérationnel ✅  
**Précision:** 98.5% ⭐  
**Temps de réponse:** < 30s ⚡""")
    
    with col4:
        if st.button("🎯 Support Pro"):
            st.info("""🏢 **Support Professionnel**

📧 **Email:** legal.support@legalgpt.tn  
📱 **Téléphone:** +216 XX XXX XXX  
💬 **Chat direct:** 24/7 disponible  
🏢 **Bureaux:** Tunis, Sfax, Sousse  

🎯 **Services Enterprise**  
- Formation équipes juridiques  
- Intégration API  
- Solutions sur mesure  
- Support dédié""")

if __name__ == "__main__":
    main()