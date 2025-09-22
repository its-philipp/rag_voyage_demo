import os
import json
from typing import List, Dict, Any
from openai import OpenAI

from src.pipeline import query_system

# --- Agent Components ---


class AgenticRAG:
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Agentic RAG system.

        Args:
            config: The application configuration dictionary.
        """
        self.config = config
        self.client = OpenAI()
        self.model = os.getenv("TRULENS_MODEL", "gpt-4o-mini")

    def decompose_query(self, query: str) -> List[str]:
        """
        Decomposes a complex query into a series of simpler, answerable sub-queries.

        Args:
            query: The user's original query.

        Returns:
            A list of sub-queries.
        """
        prompt = f"""
        You are a helpful research assistant. A user has asked the following complex question:
        "{query}"

        Your task is to break this down into a series of simpler, independent sub-questions that can be answered by a search engine.
        The goal is to gather all the necessary facts to fully answer the original question.

        Please provide the sub-questions as a JSON list of strings.
        For example:
        ["What is a foobar?", "How does a foobar relate to a widget?", "What are the key features of a super-foobar?"]

        JSON Sub-questions:
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        try:
            # The response object holds the content in a string, which needs to be parsed.
            content = response.choices[0].message.content
            if content:
                # The LLM often returns a JSON object with a key, like {"questions": [...]}.
                # We need to find the list within that object.
                data = json.loads(content)
                # Find the first value in the JSON dict that is a list
                for value in data.values():
                    if isinstance(value, list):
                        return value
                # Fallback if the structure is unexpected but contains a list
                raise ValueError("No list found in the JSON response.")
            return []
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Error parsing LLM response for query decomposition: {e}")
            # Fallback to using the original query if decomposition fails
            return [query]

    def synthesize_answer(
        self, query: str, contexts: List[Dict[str, Any]], sub_queries: List[str]
    ) -> str:
        """
        Synthesizes a final answer based on the original query and retrieved contexts.

        Args:
            query: The user's original query.
            contexts: A list of context dictionaries, each with a 'text' key.
            sub_queries: The list of sub-queries that were executed.

        Returns:
            The synthesized final answer as a string.
        """
        context_str = "\n\n---\n\n".join([c["text"] for c in contexts])
        sub_queries_str = "\n- ".join(sub_queries)

        prompt = f"""
        You are an expert research assistant. Provide an answer using ONLY the provided context.

        Original Question: "{query}"

        Sub-questions investigated:
        - {sub_queries_str}

        Context:
        ---
        {context_str}
        ---

        Instructions:
        - Cite evidence by quoting short snippets and, when possible, include the source doc_id.
        - If the context is insufficient for any part, state that explicitly.
        - Keep the answer concise and grounded.

        Final Answer:
        """

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()

    def run(self, query: str) -> Dict[str, Any]:
        """
        Runs the full agentic RAG pipeline: decompose, retrieve for each, and synthesize.

        Args:
            query: The user's original query.

        Returns:
            A dictionary containing the final answer and intermediate steps.
        """
        print(f"\nOriginal Query: {query}")
        sub_queries = self.decompose_query(query)
        print(f"Decomposed Sub-queries: {sub_queries}")

        # Retrieve context for each sub-query and collect unique results
        all_contexts: Dict[str, Dict[str, Any]] = {}  # Use dict to handle duplicates
        for sub_q in sub_queries:
            print(f"  > Retrieving for: '{sub_q}'")
            retrieved = query_system(sub_q, self.config)
            for _, context_dict in retrieved:
                all_contexts[context_dict["doc_id"]] = context_dict

        contexts_list = list(all_contexts.values())
        print(f"Retrieved {len(contexts_list)} unique contexts.")

        final_answer = self.synthesize_answer(query, contexts_list, sub_queries)
        print(f"Final Answer: {final_answer}")

        return {
            "original_query": query,
            "sub_queries": sub_queries,
            "retrieved_contexts": contexts_list,
            "final_answer": final_answer,
        }
