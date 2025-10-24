from typing import Dict, List, Any
from azure.ai.evaluation import (
    GroundednessEvaluator,
    RelevanceEvaluator,
    CoherenceEvaluator,
    FluencyEvaluator,
    SimilarityEvaluator,
    RetrievalEvaluator
)
from .config import EvaluationConfig

class RAGEvaluationFramework:
    """Unified framework for RAG evaluation"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.evaluators = self._initialize_evaluators()
    
    def _initialize_evaluators(self) -> Dict[str, Any]:
        """Initialize all evaluators based on config"""
        model_config = {
            "azure_endpoint": self.config.azure_openai_endpoint,
            "api_key": self.config.azure_openai_key,
            "azure_deployment": self.config.azure_openai_deployment,
        }
        
        evaluators = {}
        
        if "groundedness" in self.config.metrics:
            evaluators["groundedness"] = GroundednessEvaluator(model_config)
        
        if "relevance" in self.config.metrics:
            evaluators["relevance"] = RelevanceEvaluator(model_config)
        
        if "coherence" in self.config.metrics:
            evaluators["coherence"] = CoherenceEvaluator(model_config)
        
        if "fluency" in self.config.metrics:
            evaluators["fluency"] = FluencyEvaluator(model_config)
        
        if "similarity" in self.config.metrics:
            evaluators["similarity"] = SimilarityEvaluator(model_config)
        
        if "retrieval" in self.config.metrics:
            evaluators["retrieval"] = RetrievalEvaluator()
        
        return evaluators
    
    def evaluate_response(
        self,
        query: str,
        response: str,
        context: str,
        ground_truth: str = None,
        retrieved_documents: List[Dict] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single RAG response
        
        Args:
            query: User query
            response: Generated response
            context: Retrieved context
            ground_truth: Expected answer (optional)
            retrieved_documents: List of retrieved docs for retrieval metrics
            
        Returns:
            Dictionary of metric scores
        """
        results = {}
        
        # Groundedness: Is response grounded in context?
        if "groundedness" in self.evaluators:
            groundedness_result = self.evaluators["groundedness"](
                query=query,
                response=response,
                context=context
            )
            results["groundedness"] = groundedness_result["groundedness"]
        
        # Relevance: Is response relevant to query?
        if "relevance" in self.evaluators:
            relevance_result = self.evaluators["relevance"](
                query=query,
                response=response,
                context=context
            )
            results["relevance"] = relevance_result["relevance"]
        
        # Coherence: Is response coherent?
        if "coherence" in self.evaluators:
            coherence_result = self.evaluators["coherence"](
                query=query,
                response=response
            )
            results["coherence"] = coherence_result["coherence"]
        
        # Fluency: Is response fluent?
        if "fluency" in self.evaluators:
            fluency_result = self.evaluators["fluency"](
                query=query,
                response=response
            )
            results["fluency"] = fluency_result["fluency"]
        
        # Similarity: Compare with ground truth
        if "similarity" in self.evaluators and ground_truth:
            similarity_result = self.evaluators["similarity"](
                query=query,
                response=response,
                ground_truth=ground_truth
            )
            results["similarity"] = similarity_result["similarity"]
        
        # Retrieval metrics
        if "retrieval" in self.evaluators and retrieved_documents:
            retrieval_result = self.evaluators["retrieval"](
                query=query,
                retrieved_documents=retrieved_documents,
                ground_truth=ground_truth
            )
            results.update(retrieval_result)
        
        return results
    
    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple test cases
        
        Args:
            test_cases: List of dicts with keys: query, response, context, ground_truth
            
        Returns:
            List of evaluation results
        """
        results = []
        for test_case in test_cases:
            eval_result = self.evaluate_response(
                query=test_case["query"],
                response=test_case["response"],
                context=test_case["context"],
                ground_truth=test_case.get("ground_truth"),
                retrieved_documents=test_case.get("retrieved_documents")
            )
            results.append({
                **test_case,
                "metrics": eval_result
            })
        return results
    
    def check_thresholds(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Check if metrics meet thresholds"""
        threshold_map = {
            "groundedness": self.config.groundedness_threshold,
            "relevance": self.config.relevance_threshold,
            "coherence": self.config.coherence_threshold,
            "fluency": self.config.fluency_threshold,
        }
        
        return {
            metric: score >= threshold_map.get(metric, 0.7)
            for metric, score in metrics.items()
            if metric in threshold_map
        }