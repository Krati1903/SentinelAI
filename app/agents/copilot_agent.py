"""
SentinelAI Recruiter Investigation Copilot Agent.

Orchestrates Context-Augmented Generation (CAG) and Retrieval-Augmented 
Generation (RAG) reasoning modes for recruiter decision support.

Provides natural language interface to interview session analysis with
evidence-based explanations and policy-aware recommendations.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from question_router import QuestionRouter, RoutingMode
from rag_service import RAGService

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class RecruiterCopilot:
    """
    AI-powered copilot for recruiter investigation and decision support.
    
    Combines Context-Augmented Generation (session data) and Retrieval-Augmented
    Generation (policies/guidelines) to provide evidence-based answers to
    recruiter questions about interview sessions.
    """
    
    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        model: str = "mixtral-8x7b-32768",
        temperature: float = 0.3,
        knowledge_dir: str = "app/knowledge",
        vectorstore_path: str = "app/vectorstore",
        prompt_template_path: str = "app/prompts/copilot_prompt.txt"
    ):
        """
        Initialize the Recruiter Copilot.
        
        Args:
            groq_api_key: Groq API key (defaults to GROQ_API_KEY env var)
            model: LLM model to use
            temperature: LLM temperature (0.3 for factual responses)
            knowledge_dir: Directory containing policy documents
            vectorstore_path: Path to FAISS vector store
            prompt_template_path: Path to system prompt template
        """
        # Initialize Groq LLM
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.llm = ChatGroq(
            api_key=api_key,
            model_name=model,
            temperature=temperature,
            max_tokens=2048
        )
        logger.info(f"Initialized ChatGroq with model: {model}")
        
        # Initialize question router
        self.router = QuestionRouter()
        logger.info("Initialized QuestionRouter")
        
        # Initialize RAG service
        self.rag_service = RAGService(
            knowledge_dir=knowledge_dir,
            vector_store_path=vectorstore_path
        )
        logger.info("Initialized RAGService")
        
        # Load system prompt
        self.system_prompt = self._load_system_prompt(prompt_template_path)
        logger.info("Loaded system prompt template")
        
        # Session context cache
        self._session_cache: Dict[str, Dict[str, Any]] = {}
    
    def _load_system_prompt(self, prompt_path: str) -> str:
        """
        Load system prompt from file.
        
        Args:
            prompt_path: Path to prompt file
            
        Returns:
            str: System prompt content
        """
        try:
            path = Path(prompt_path)
            if path.exists():
                with open(path, 'r') as f:
                    return f.read()
            else:
                logger.warning(f"Prompt file not found at {prompt_path}")
                return self._get_default_system_prompt()
        except Exception as e:
            logger.error(f"Error loading system prompt: {str(e)}")
            return self._get_default_system_prompt()
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt if file not found."""
        return """You are SentinelAI Recruiter Investigation Copilot.

Your role is to help recruiters understand interview sessions through evidence-based analysis.

Key responsibilities:
- Answer questions about interview sessions using provided context
- Always cite evidence and explain reasoning
- Never hallucinate or invent interview events
- Use neutral, objective language when discussing candidate behavior
- Reference company policies when applicable
- Acknowledge when evidence is insufficient

When answering, structure your response with:
1. Observation: What the data shows
2. Context: Why this matters
3. Evidence: Specific data supporting your answer
4. Conclusion: Your answer with appropriate caveats
"""
    
    def load_session_context(
        self,
        session_id: str,
        context_path: str = "reports"
    ) -> bool:
        """
        Load session context from Summary Agent output.
        
        Args:
            session_id: Interview session ID
            context_path: Directory containing context JSON files
            
        Returns:
            bool: Success status
        """
        try:
            context_file = Path(context_path) / f"{session_id}_context.json"
            
            if not context_file.exists():
                logger.error(f"Context file not found: {context_file}")
                return False
            
            with open(context_file, 'r') as f:
                context = json.load(f)
            
            self._session_cache[session_id] = context
            logger.info(f"Loaded session context for {session_id}")
            return True
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in context file: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error loading session context: {str(e)}")
            return False
    
    def ask(
        self,
        question: str,
        session_id: str,
        context_path: str = "reports"
    ) -> Dict[str, Any]:
        """
        Answer a recruiter question about an interview session.
        
        Workflow:
        1. Load session context
        2. Route question to CAG/RAG/BOTH
        3. Retrieve policies if needed (RAG)
        4. Generate answer using Groq
        5. Return structured response
        
        Args:
            question: Recruiter's question
            session_id: Interview session ID
            context_path: Directory containing context files
            
        Returns:
            dict: Answer with metadata
        """
        if not question or not question.strip():
            return {
                "success": False,
                "error": "Question cannot be empty",
                "mode": None,
                "sources": []
            }
        
        try:
            # Load session context
            if session_id not in self._session_cache:
                if not self.load_session_context(session_id, context_path):
                    return {
                        "success": False,
                        "error": f"Could not load context for session {session_id}",
                        "mode": None,
                        "sources": []
                    }
            
            session_context = self._session_cache[session_id]
            
            # Route question
            mode = self.router.route(question)
            logger.info(f"Routed question to {mode.value} mode")
            
            # Retrieve policies if needed
            policy_context = None
            policy_sources = []
            
            if mode in [RoutingMode.RAG, RoutingMode.BOTH]:
                policy_context, policy_sources = self._retrieve_policies(question)
            
            # Generate answer
            answer = self._generate_answer(
                question=question,
                session_context=session_context,
                policy_context=policy_context,
                mode=mode
            )
            
            # Compile sources
            sources = self._compile_sources(
                mode=mode,
                session_id=session_id,
                policy_sources=policy_sources
            )
            
            return {
                "success": True,
                "answer": answer,
                "mode": mode.value,
                "sources": sources,
                "session_id": session_id
            }
        
        except Exception as e:
            logger.error(f"Error in ask() method: {str(e)}")
            return {
                "success": False,
                "error": f"An error occurred: {str(e)}",
                "mode": None,
                "sources": []
            }
    
    def _retrieve_policies(self, question: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Retrieve relevant policy documents.
        
        Args:
            question: The question to retrieve policies for
            
        Returns:
            Tuple of (formatted policy context, source metadata)
        """
        try:
            results = self.rag_service.retrieve(question, k=3)
            
            if not results:
                logger.debug("No policy documents retrieved")
                return None, []
            
            # Format policy context
            policy_parts = []
            sources = []
            
            for i, (content, source, score) in enumerate(results, 1):
                policy_parts.append(f"\n[Policy {i} - Source: {source}]\n{content}")
                sources.append({
                    "source": source,
                    "relevance_score": round(score, 3),
                    "type": "policy"
                })
            
            policy_context = "".join(policy_parts)
            logger.debug(f"Retrieved {len(sources)} policy documents")
            
            return policy_context, sources
        
        except Exception as e:
            logger.error(f"Error retrieving policies: {str(e)}")
            return None, []
    
    def _generate_answer(
        self,
        question: str,
        session_context: Dict[str, Any],
        policy_context: Optional[str],
        mode: RoutingMode
    ) -> str:
        """
        Generate answer using Groq LLM.
        
        Args:
            question: The question to answer
            session_context: Interview session data
            policy_context: Retrieved policy documents (if RAG)
            mode: The routing mode
            
        Returns:
            str: Generated answer
        """
        try:
            # Build prompt based on mode
            prompt = self._build_prompt(
                question=question,
                session_context=session_context,
                policy_context=policy_context,
                mode=mode
            )
            
            # Generate response
            response = self.llm.invoke(prompt)
            
            # Extract text from response
            answer_text = response.content if hasattr(response, 'content') else str(response)
            
            logger.debug(f"Generated answer for question: {question[:50]}...")
            return answer_text.strip()
        
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"I encountered an error while processing your question: {str(e)}"
    
    def _build_prompt(
        self,
        question: str,
        session_context: Dict[str, Any],
        policy_context: Optional[str],
        mode: RoutingMode
    ) -> str:
        """
        Build the complete prompt for the LLM.
        
        Args:
            question: The question to answer
            session_context: Interview session data
            policy_context: Retrieved policies (if RAG)
            mode: Routing mode
            
        Returns:
            str: Complete prompt
        """
        prompt_parts = [self.system_prompt]
        
        # Add mode-specific instructions
        if mode == RoutingMode.CAG:
            prompt_parts.append("\n## Mode: Context-Augmented Generation")
            prompt_parts.append("Answer using ONLY the session context data provided below.")
            prompt_parts.append("Do not reference external policies or guidelines.")
        
        elif mode == RoutingMode.RAG:
            prompt_parts.append("\n## Mode: Retrieval-Augmented Generation")
            prompt_parts.append("Answer using ONLY the policy and guideline documents below.")
            prompt_parts.append("Reference the source document when citing policies.")
        
        elif mode == RoutingMode.BOTH:
            prompt_parts.append("\n## Mode: Context + Policy Reasoning")
            prompt_parts.append("Answer using both session context and policy documents.")
            prompt_parts.append("Clearly distinguish between observations and policy implications.")
        
        # Add session context
        prompt_parts.append("\n## Interview Session Context")
        prompt_parts.append(self._format_session_context(session_context))
        
        # Add policy context if present
        if policy_context:
            prompt_parts.append("\n## Company Policy & Guidelines")
            prompt_parts.append(policy_context)
        
        # Add the question
        prompt_parts.append(f"\n## Recruiter Question\n{question}")
        
        prompt_parts.append("\n## Your Answer\nProvide a clear, evidence-based answer:")
        
        return "\n".join(prompt_parts)
    
    def _format_session_context(self, context: Dict[str, Any]) -> str:
        """
        Format session context for inclusion in prompt.
        
        Args:
            context: Session context dictionary
            
        Returns:
            str: Formatted context
        """
        formatted = []
        
        # Candidate information
        if "candidate_info" in context:
            candidate = context["candidate_info"]
            formatted.append(f"**Candidate**: {candidate.get('name', 'Unknown')}")
            formatted.append(f"**Role**: {candidate.get('role', 'Unknown')}")
            formatted.append(f"**Interview Type**: {candidate.get('type', 'Unknown')}")
        
        # Risk scores
        if "risk_scores" in context:
            risks = context["risk_scores"]
            formatted.append(f"\n**Risk Assessment**:")
            formatted.append(f"- Overall Risk: {risks.get('overall', 'N/A')}")
            formatted.append(f"- Behavioral Risk: {risks.get('behavioral', 'N/A')}")
            formatted.append(f"- Environmental Risk: {risks.get('environmental', 'N/A')}")
        
        # Warnings
        if "warnings" in context:
            warnings = context["warnings"]
            if warnings:
                formatted.append(f"\n**Warnings Detected**: {len(warnings)}")
                for warning in warnings[:5]:  # Limit to first 5
                    formatted.append(f"- {warning.get('type', 'Unknown')}: {warning.get('description', '')}")
        
        # Session metrics
        if "metrics" in context:
            metrics = context["metrics"]
            formatted.append(f"\n**Session Metrics**:")
            for key, value in metrics.items():
                formatted.append(f"- {key}: {value}")
        
        # Objects detected
        if "objects_detected" in context:
            objects = context["objects_detected"]
            formatted.append(f"\n**Objects Detected**: {', '.join(objects) if objects else 'None'}")
        
        # Flags
        if "flags" in context:
            flags = context["flags"]
            if flags:
                formatted.append(f"\n**Policy Violations/Flags**: {', '.join(flags)}")
        
        return "\n".join(formatted)
    
    def _compile_sources(
        self,
        mode: RoutingMode,
        session_id: str,
        policy_sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Compile list of sources used in the response.
        
        Args:
            mode: Routing mode
            session_id: Session ID
            policy_sources: Retrieved policy documents
            
        Returns:
            List of source metadata
        """
        sources = []
        
        # Add session context as source
        if mode in [RoutingMode.CAG, RoutingMode.BOTH]:
            sources.append({
                "type": "session_context",
                "session_id": session_id,
                "source": f"{session_id}_context.json"
            })
        
        # Add policy sources
        if policy_sources:
            sources.extend(policy_sources)
        
        return sources
    
    def clear_cache(self):
        """Clear the session context cache."""
        self._session_cache.clear()
        logger.info("Cleared session context cache")


# Import os at the top of the file
import os
