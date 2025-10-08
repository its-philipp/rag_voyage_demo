#!/usr/bin/env python3
"""
Comprehensive RAG Report Generator
Displays: Queries → Retrieved Chunks → Generated Answers → All Scores & Metrics
"""
import requests
import json
import time
import os
import sys
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Add project root to path
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

API_URL = "http://localhost:8000"


def wait_for_api(max_retries=15, timeout=2):
    """Wait for Flask API to be ready"""
    print("⏳ Waiting for Flask API to be ready...")
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_URL}/health", timeout=timeout)
            if response.status_code == 200:
                print("✅ API is ready!\n")
                return True
        except requests.exceptions.ConnectionError:
            if i < max_retries - 1:
                time.sleep(2)
    print("❌ Could not connect to API. Make sure it's running on port 8000")
    return False


def retrieve_documents(query: str) -> Optional[Dict]:
    """Retrieve documents from the search endpoint"""
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/search",
            json={"query": query},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        latency = time.time() - start_time

        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            return None

        data = response.json()
        data["latency"] = latency
        return data
    except Exception as e:
        print(f"❌ Error retrieving documents: {e}")
        return None


def generate_answer(query: str, contexts: List[str]) -> Optional[str]:
    """Generate an answer using OpenAI"""
    try:
        from openai import OpenAI

        client = OpenAI()

        context_str = "\n\n".join(contexts[:10])
        prompt = (
            "You are a helpful assistant. Answer ONLY using the provided context. "
            "Cite evidence by quoting short snippets and include document numbers when referencing. "
            "If context is insufficient, say so explicitly. Keep the answer concise but comprehensive.\n\n"
            f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️  Error generating answer: {e}")
        return None


def compute_rag_metrics(query: str, contexts: List[str], answer: str) -> Optional[Dict]:
    """Compute RAG Triad metrics"""
    try:
        from eval.feedback import groundedness_score, relevance_score

        joined_context = "\n\n".join(contexts)

        print("    Computing Groundedness...")
        g_score = groundedness_score(source=joined_context, statement=answer)

        print("    Computing Context Relevance...")
        c_score = relevance_score(question=query, text=joined_context)

        print("    Computing Answer Relevance...")
        a_score = relevance_score(question=query, text=answer)

        return {"groundedness": g_score, "context_relevance": c_score, "answer_relevance": a_score}
    except Exception as e:
        print(f"⚠️  Error computing metrics: {e}")
        return None


def format_score_interpretation(score: float, metric_name: str) -> str:
    """Format score with interpretation"""
    if score >= 0.95:
        return f"🌟 {score:.3f} - Excellent"
    elif score >= 0.85:
        return f"✅ {score:.3f} - Very Good"
    elif score >= 0.70:
        return f"👍 {score:.3f} - Good"
    elif score >= 0.50:
        return f"⚠️  {score:.3f} - Needs Improvement"
    else:
        return f"❌ {score:.3f} - Poor"


def generate_report_for_query(query: str, query_num: int, total_queries: int) -> Dict:
    """Generate complete report for a single query"""
    print(f"\n{'='*120}")
    print(f"QUERY {query_num}/{total_queries}: {query}")
    print(f"{'='*120}\n")

    report = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "retrieval": None,
        "answer": None,
        "metrics": None,
    }

    # Step 1: Retrieve documents
    print("📥 STEP 1: RETRIEVING DOCUMENTS...")
    print("-" * 120)
    retrieval_data = retrieve_documents(query)

    if not retrieval_data:
        print("❌ Retrieval failed\n")
        return report

    results = retrieval_data.get("results", [])
    latency = retrieval_data.get("latency", 0)

    # Display retrieval metrics
    print(f"✓ Retrieved {len(results)} documents in {latency:.2f}s")
    if results:
        scores = [r["score"] for r in results]
        print(
            f"✓ Score Range: {min(scores):.2f} - {max(scores):.2f} (avg: {sum(scores)/len(scores):.2f})\n"
        )

    report["retrieval"] = {"num_results": len(results), "latency": latency, "results": results}

    # Display top retrieved chunks
    print("📄 RETRIEVED CHUNKS (with scores):")
    print("-" * 120)
    for i, result in enumerate(results[:10], 1):
        score = result["score"]
        doc_id = result.get("doc_id", "N/A")
        title = result.get("title", "N/A")
        text = result.get("text", "")

        # Show more text for better context
        text_preview = text[:300] + "..." if len(text) > 300 else text

        print(f"\n[Chunk {i}] Score: {score:.2f}")
        print(f"  Doc ID: {doc_id}")
        print(f"  Title:  {title}")
        print(f"  Text:   {text_preview}")

    print(f"\n{'-'*120}\n")

    # Step 2: Generate answer
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found. Skipping answer generation and metrics.\n")
        return report

    print("🤖 STEP 2: GENERATING ANSWER...")
    print("-" * 120)
    contexts = [r.get("text", "") for r in results[:5]]
    answer = generate_answer(query, contexts)

    if not answer:
        print("❌ Answer generation failed\n")
        return report

    print("✓ Answer generated\n")
    print("📝 GENERATED ANSWER:")
    print("-" * 120)
    print(answer)
    print(f"\n{'-'*120}\n")

    report["answer"] = answer

    # Step 3: Compute RAG Triad metrics
    print("📊 STEP 3: COMPUTING RAG TRIAD METRICS...")
    print("-" * 120)
    metrics = compute_rag_metrics(query, contexts, answer)

    if not metrics:
        print("❌ Metrics computation failed\n")
        return report

    print("✓ Metrics computed\n")
    report["metrics"] = metrics

    # Display comprehensive scores
    print("=" * 120)
    print("📊 COMPREHENSIVE SCORE SUMMARY")
    print("=" * 120)

    print("\n🔍 RETRIEVAL SCORES (ColBERT Reranker):")
    print("-" * 120)
    for i, result in enumerate(results[:5], 1):
        score = result["score"]
        title = result.get("title", "N/A")
        print(f"  [{i}] {score:6.2f} - {title}")

    if len(results) > 5:
        print(f"  ... and {len(results) - 5} more documents")

    print(f"\n  Retrieval Latency:  {latency:.3f} seconds")
    print(f"  Total Retrieved:    {len(results)} documents")
    print(f"  Top Score:          {max(scores):.2f}")
    print(f"  Lowest Score:       {min(scores):.2f}")
    print(f"  Average Score:      {sum(scores)/len(scores):.2f}")
    print(f"  Score Spread:       {max(scores) - min(scores):.2f}")

    print("\n🎯 RAG TRIAD METRICS (LLM Judge: GPT-4o-mini):")
    print("-" * 120)
    print(
        f"  Groundedness:       {format_score_interpretation(metrics['groundedness'], 'Groundedness')}"
    )
    print("                      → Is the answer fully supported by the retrieved documents?")
    print(
        f"\n  Context Relevance:  {format_score_interpretation(metrics['context_relevance'], 'Context Relevance')}"
    )
    print("                      → Are the retrieved documents relevant to the question?")
    print(
        f"\n  Answer Relevance:   {format_score_interpretation(metrics['answer_relevance'], 'Answer Relevance')}"
    )
    print("                      → Does the answer directly address the question?")

    # Overall assessment
    avg_metric = (
        metrics["groundedness"] + metrics["context_relevance"] + metrics["answer_relevance"]
    ) / 3
    print(f"\n  Overall RAG Score:  {format_score_interpretation(avg_metric, 'Overall')}")

    print("\n" + "=" * 120)

    return report


def save_report_to_file(reports: List[Dict], filename: str = "rag_report.json"):
    """Save all reports to a JSON file"""
    try:
        with open(filename, "w") as f:
            json.dump(reports, f, indent=2)
        print(f"\n💾 Full report saved to: {filename}")
    except Exception as e:
        print(f"\n⚠️  Could not save report: {e}")


def main():
    """Generate comprehensive RAG reports for multiple queries"""
    print("\n" + "=" * 120)
    print(" " * 40 + "COMPREHENSIVE RAG REPORT GENERATOR")
    print("=" * 120)
    print("\nThis report shows:")
    print("  1. Query details")
    print("  2. Retrieved chunks with ColBERT reranker scores")
    print("  3. Generated answer from retrieved context")
    print("  4. RAG Triad evaluation metrics (Groundedness, Context Relevance, Answer Relevance)")
    print("=" * 120)

    # Check API availability
    if not wait_for_api():
        return

    # Test queries
    test_queries = [
        "What is ColBERT and how does it work?",
        "How does hybrid search combine dense and sparse retrieval?",
        "Explain the benefits of using FAISS for vector search",
        "What are the key considerations for chunking strategies in RAG?",
    ]

    print(f"\n📋 Processing {len(test_queries)} queries...\n")

    reports = []
    for i, query in enumerate(test_queries, 1):
        report = generate_report_for_query(query, i, len(test_queries))
        reports.append(report)

        if i < len(test_queries):
            print("\n" + "▼" * 120 + "\n")
            time.sleep(1)  # Brief pause between queries

    # Save reports
    save_report_to_file(reports)

    # Summary
    print("\n" + "=" * 120)
    print(" " * 45 + "📊 EXECUTIVE SUMMARY")
    print("=" * 120)

    successful_queries = [r for r in reports if r.get("metrics")]

    if successful_queries:
        avg_groundedness = sum(r["metrics"]["groundedness"] for r in successful_queries) / len(
            successful_queries
        )
        avg_context_rel = sum(r["metrics"]["context_relevance"] for r in successful_queries) / len(
            successful_queries
        )
        avg_answer_rel = sum(r["metrics"]["answer_relevance"] for r in successful_queries) / len(
            successful_queries
        )

        avg_retrieval_scores = []
        for r in successful_queries:
            if r.get("retrieval") and r["retrieval"]["results"]:
                scores = [res["score"] for res in r["retrieval"]["results"]]
                avg_retrieval_scores.append(sum(scores) / len(scores))

        avg_retrieval = (
            sum(avg_retrieval_scores) / len(avg_retrieval_scores) if avg_retrieval_scores else 0
        )

        print(f"\nQueries Processed:           {len(test_queries)}")
        print(f"Successful Evaluations:      {len(successful_queries)}")
        print("\nAverage Metrics Across All Queries:")
        print(f"  Groundedness:              {avg_groundedness:.3f}")
        print(f"  Context Relevance:         {avg_context_rel:.3f}")
        print(f"  Answer Relevance:          {avg_answer_rel:.3f}")
        print(
            f"  Overall RAG Quality:       {((avg_groundedness + avg_context_rel + avg_answer_rel) / 3):.3f}"
        )
        print(f"\nAverage Retrieval Score:     {avg_retrieval:.2f}")
    else:
        print("\n⚠️  No successful evaluations completed.")

    print("\n" + "=" * 120 + "\n")


if __name__ == "__main__":
    main()
