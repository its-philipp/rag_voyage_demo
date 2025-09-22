import os
import sys
from pathlib import Path
import yaml
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI


# Ensure project root is in path for imports
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.pipeline import query_system  # noqa: E402
from eval.feedback import groundedness_score, relevance_score  # noqa: E402
from src.agentic_rag import AgenticRAG  # noqa: E402

# --- App components ---


class RAGApp:
    def __init__(self, config):
        self.config = config
        self.oa_client = OpenAI()

    def retrieve(self, query: str) -> List[str]:
        """Retrieve contexts for a query."""
        top_k = query_system(query, self.config)
        return [c[1]["text"] for c in top_k]

    def synthesize(self, query: str, contexts: List[str]) -> str:
        """Synthesize an answer from contexts."""
        context_str = "\n\n".join(contexts[:10])
        prompt = (
            "You are a helpful assistant. Answer ONLY using the provided context. "
            "Cite evidence by quoting short snippets and include their doc_id if available. "
            "If context is insufficient, say so explicitly. Keep the answer concise.\n\n"
            f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"
        )
        resp = self.oa_client.chat.completions.create(
            model=os.getenv("TRULENS_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()


class AgenticRAGApp:
    def __init__(self, config):
        self.config = config
        self.agent = AgenticRAG(config)
        # Store results of the last run to avoid re-computing
        self._last_run_results: Dict[str, Any] = {}

    def _run_agent_if_needed(self, query: str):
        """Helper to run the agent only once per query."""
        if self._last_run_results.get("original_query") != query:
            self._last_run_results = self.agent.run(query)

    def retrieve(self, query: str) -> List[str]:
        """Retrieve contexts for a query using the agentic pipeline."""
        self._run_agent_if_needed(query)
        contexts = self._last_run_results.get("retrieved_contexts", [])
        return [c["text"] for c in contexts]

    def synthesize(self, query: str, contexts: List[str]) -> str:
        """Synthesize an answer using the agentic pipeline."""
        # The contexts argument is unused here because the agent handles synthesis internally,
        # but we keep it for compatibility with the evaluation harness.
        self._run_agent_if_needed(query)
        return self._last_run_results.get("final_answer", "")


# --- Evaluation Runner ---


def run_evaluation(app_class):
    """Runs evaluation for a given app class (RAGApp or AgenticRAGApp)."""
    load_dotenv()
    config = yaml.safe_load(open("config.yaml"))

    rag_app = app_class(config)

    eval_questions = [
        "Compare and contrast ColBERT with traditional cross-encoder and bi-encoder models, highlighting its unique mechanism.",
        "How does the FAISS IVFPQ index work, and what are the trade-offs compared to a Flat index?",
    ]

    results: List[Dict[str, Any]] = []

    for question in eval_questions:
        print(f"\n--- Processing question: {question} ---")
        contexts = rag_app.retrieve(question)
        answer = rag_app.synthesize(question, contexts)

        joined_context = "\n\n".join(contexts)

        # Calculate metrics using custom feedback functions
        g_score = groundedness_score(source=joined_context, statement=answer)

        ctx_scores = [relevance_score(question=question, text=ctx) for ctx in contexts[:5]]
        avg_ctx_relevance = sum(ctx_scores) / len(ctx_scores) if ctx_scores else 0.0

        ans_rel_score = relevance_score(question=question, text=answer)

        item = {
            "question": question,
            "answer": answer,
            "metrics": {
                "groundedness": g_score,
                "context_relevance_avg": avg_ctx_relevance,
                "answer_relevance": ans_rel_score,
            },
            "contexts": contexts,
        }
        results.append(item)
        print(f"  - Groundedness: {g_score:.2f}")
        print(f"  - Avg. Context Relevance: {avg_ctx_relevance:.2f}")
        print(f"  - Answer Relevance: {ans_rel_score:.2f}")

    # Write the final summary report
    report_name = f"{app_class.__name__.lower()}_summary.yaml"
    os.makedirs("eval/reports", exist_ok=True)
    out_path = f"eval/reports/{report_name}"
    with open(out_path, "w") as fh:
        yaml.safe_dump({"results": results}, fh, indent=2, sort_keys=False)
    print(f"\nWrote final summary to {out_path}")


if __name__ == "__main__":
    print("===== Running Baseline RAG Evaluation =====")
    run_evaluation(RAGApp)

    print("\n\n===== Running Agentic RAG Evaluation =====")
    run_evaluation(AgenticRAGApp)
