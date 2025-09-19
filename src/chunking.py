import re
from typing import List, Dict


def simple_sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def make_chunks_with_context(doc: Dict, max_sentences: int = 3, overlap: int = 1) -> List[Dict]:
    """Create small sentence-window chunks; attach title as lightweight context."""
    sents = simple_sentence_split(doc["text"])
    chunks = []
    i = 0
    while i < len(sents):
        window = sents[i : i + max_sentences]
        context = doc.get("title", "")  # lightweight context; you can extend to neighboring chunks
        chunk_text = " ".join(window)
        chunks.append(
            {
                "doc_id": doc["doc_id"],
                "chunk_id": f"{doc['doc_id']}::chunk_{len(chunks):04d}",
                "title": doc.get("title", ""),
                "text": chunk_text,
                "context": context,
            }
        )
        if i + max_sentences >= len(sents):
            break
        i += max_sentences - overlap
    return chunks
