# Ai-agent-framework-with-evaluation
Ai-agent-framework-with-evaluation/
├── src/
│   ├── agents/          # LangGraph agents
│   ├── chains/          # LangChain chains
│   ├── api/             # FastAPI endpoints
│   └── rag/             # RAG components
├── tests/
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── config.py           # Evaluation configurations
│   │   ├── evaluators.py       # Custom evaluator classes
│   │   ├── fixtures.py         # Test data and fixtures
│   │   ├── metrics.py          # Metric calculators
│   │   ├── test_suites/
│   │   │   ├── test_rag_quality.py
│   │   │   ├── test_retrieval.py
│   │   │   └── test_generation.py
│   │   └── utils.py            # Helper functions
│   └── integration/
└── evaluation_data/
    ├── ground_truth.jsonl
    └── test_queries.jsonl