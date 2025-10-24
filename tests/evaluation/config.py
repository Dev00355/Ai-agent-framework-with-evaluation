from dataclasses import dataclass
from typing import List, Dict, Optional
from azure.ai.evaluation import GroundednessEvaluator, RelevanceEvaluator, CoherenceEvaluator, FluencyEvaluator

@dataclass
class EvaluationConfig:
    """Configuration for RAG evaluation"""
    azure_openai_endpoint: str
    azure_openai_key: str
    azure_openai_deployment: str
    azure_search_endpoint: str
    azure_search_key: str
    
    # Evaluation metrics to use
    metrics: List[str] = None
    
    # Thresholds
    groundedness_threshold: float = 0.7
    relevance_threshold: float = 0.7
    coherence_threshold: float = 0.7
    fluency_threshold: float = 0.7
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["groundedness", "relevance", "coherence", "fluency"]

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        import os
        return cls(
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_openai_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            azure_search_key=os.getenv("AZURE_SEARCH_KEY"),
        )