# Examples & Reports

This directory contains example scripts and generated reports for testing and demonstrating the RAG system.

## 📊 RAG Report Generator

### `generate_rag_report.py`

Comprehensive testing script that generates detailed reports showing:
- Query processing and retrieval
- Retrieved document chunks with ColBERT reranker scores
- Generated answers using retrieved context
- RAG Triad evaluation metrics (Groundedness, Context Relevance, Answer Relevance)
- Score distributions and quality assessments

### Usage

```bash
# Start the Flask API (in one terminal)
cd /home/philipp/workspace/playground/rag_voyage_demo
source .venv/bin/activate
python apps/api.py

# Run the report generator (in another terminal)
cd /home/philipp/workspace/playground/rag_voyage_demo
source .venv/bin/activate
python examples/generate_rag_report.py
```

### Requirements

- Flask API running on `http://localhost:8000`
- `VOYAGE_API_KEY` set in environment (for retrieval)
- `OPENAI_API_KEY` set in environment (for RAG Triad metrics)
- Config must have `reranker.enabled: true` for real ColBERT scores

### Output

The script generates:
1. **Console output**: Detailed, formatted report with visual indicators
2. **JSON file**: Structured data saved as `rag_report.json`

## 📄 Sample Report

### `rag_report.json`

Sample output from running the report generator. Contains:
- Query details and timestamps
- Retrieved documents with full text and scores
- Generated answers
- RAG Triad metrics for each query
- Latency measurements

**Structure:**
```json
[
  {
    "query": "What is ColBERT?",
    "timestamp": "2025-10-08T17:02:37.872783",
    "retrieval": {
      "num_results": 10,
      "latency": 3.92,
      "results": [...]
    },
    "answer": "ColBERT is a retrieval model...",
    "metrics": {
      "groundedness": 1.0,
      "context_relevance": 1.0,
      "answer_relevance": 1.0
    }
  },
  ...
]
```

## 🎯 Understanding the Metrics

### Retrieval Scores (ColBERT Reranker)
- **Scale**: 0-100+ (higher = more relevant)
- **Source**: Late interaction neural model (ColBERT)
- **Measures**: Document relevance to query based on token-level interactions

**Typical ranges:**
- 50-60: Excellent match
- 40-50: Very good match
- 30-40: Good match
- 20-30: Moderate relevance
- <20: Weak relevance

### RAG Triad Metrics
- **Scale**: 0.0-1.0 (higher = better)
- **Source**: GPT-4o-mini as LLM judge
- **Measures**: Quality of full RAG pipeline (retrieval + generation)

| Metric | What it Measures |
|--------|------------------|
| **Groundedness** | Is the answer fully supported by retrieved documents? (prevents hallucination) |
| **Context Relevance** | Are the retrieved documents relevant to the question? |
| **Answer Relevance** | Does the answer directly address the user's question? |

**Quality ranges:**
- 0.95-1.0: Excellent
- 0.85-0.95: Very good
- 0.70-0.85: Good
- 0.50-0.70: Needs improvement
- <0.50: Poor

## 🔧 Customization

To test with your own queries, edit the `test_queries` list in `generate_rag_report.py`:

```python
test_queries = [
    "What is ColBERT and how does it work?",
    "Your custom query here...",
]
```
