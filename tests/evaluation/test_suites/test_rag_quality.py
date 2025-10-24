import pytest
from typing import Dict, List

class TestRAGQuality:
    """Test suite for RAG quality metrics"""
    
    def test_groundedness_threshold(self, rag_evaluator, fastapi_client, test_queries):
        """Test that responses are grounded in retrieved context"""
        for query_data in test_queries:
            # Call your API
            response = fastapi_client.post("/query", json={"query": query_data["query"]})
            assert response.status_code == 200
            
            result = response.json()
            
            # Evaluate
            metrics = rag_evaluator.evaluate_response(
                query=query_data["query"],
                response=result["answer"],
                context=result["context"]
            )
            
            # Assert groundedness meets threshold
            assert metrics["groundedness"] >= rag_evaluator.config.groundedness_threshold, \
                f"Groundedness {metrics['groundedness']} below threshold for query: {query_data['query']}"
    
    def test_relevance_threshold(self, rag_evaluator, fastapi_client, test_queries):
        """Test that responses are relevant to queries"""
        for query_data in test_queries:
            response = fastapi_client.post("/query", json={"query": query_data["query"]})
            result = response.json()
            
            metrics = rag_evaluator.evaluate_response(
                query=query_data["query"],
                response=result["answer"],
                context=result["context"]
            )
            
            assert metrics["relevance"] >= rag_evaluator.config.relevance_threshold
    
    @pytest.mark.parametrize("metric_name", ["coherence", "fluency"])
    def test_response_quality_metrics(self, rag_evaluator, fastapi_client, test_queries, metric_name):
        """Test coherence and fluency of responses"""
        failures = []
        
        for query_data in test_queries:
            response = fastapi_client.post("/query", json={"query": query_data["query"]})
            result = response.json()
            
            metrics = rag_evaluator.evaluate_response(
                query=query_data["query"],
                response=result["answer"],
                context=result["context"]
            )
            
            threshold = getattr(rag_evaluator.config, f"{metric_name}_threshold")
            if metrics[metric_name] < threshold:
                failures.append({
                    "query": query_data["query"],
                    "score": metrics[metric_name],
                    "threshold": threshold
                })
        
        assert len(failures) == 0, f"Failed {metric_name} checks: {failures}"
    
    def test_similarity_with_ground_truth(self, rag_evaluator, fastapi_client, test_queries, ground_truth_data):
        """Test similarity of responses with ground truth answers"""
        for query_data in test_queries:
            query = query_data["query"]
            
            if query not in ground_truth_data:
                pytest.skip(f"No ground truth for query: {query}")
            
            response = fastapi_client.post("/query", json={"query": query})
            result = response.json()
            
            metrics = rag_evaluator.evaluate_response(
                query=query,
                response=result["answer"],
                context=result["context"],
                ground_truth=ground_truth_data[query]
            )
            
            # Lower threshold for similarity as exact matches are rare
            assert metrics.get("similarity", 0) >= 0.6, \
                f"Similarity too low for query: {query}"