import pytest
import json
from typing import List, Dict
from pathlib import Path

@pytest.fixture
def evaluation_config():
    """Provide evaluation configuration"""
    from .config import EvaluationConfig
    return EvaluationConfig.from_env()

@pytest.fixture
def rag_evaluator(evaluation_config):
    """Provide RAG evaluation framework"""
    from .evaluators import RAGEvaluationFramework
    return RAGEvaluationFramework(evaluation_config)

@pytest.fixture
def test_queries() -> List[Dict]:
    """Load test queries from file"""
    test_data_path = Path(__file__).parent.parent.parent / "evaluation_data" / "test_queries.jsonl"
    
    queries = []
    if test_data_path.exists():
        with open(test_data_path, 'r') as f:
            for line in f:
                queries.append(json.loads(line))
    
    return queries

@pytest.fixture
def ground_truth_data() -> Dict[str, str]:
    """Load ground truth answers"""
    gt_path = Path(__file__).parent.parent.parent / "evaluation_data" / "ground_truth.jsonl"
    
    ground_truth = {}
    if gt_path.exists():
        with open(gt_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                ground_truth[data["query"]] = data["answer"]
    
    return ground_truth

@pytest.fixture
def fastapi_client():
    """Provide FastAPI test client"""
    from fastapi.testclient import TestClient
    from src.api.main import app  # Adjust import path
    return TestClient(app)