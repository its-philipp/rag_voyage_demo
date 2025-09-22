# Context Windowing

Rerankers benefit from clean passages. Keep chunks within model context limits; for ColBERT, doc_maxlen ~180 is common. For generator LLMs, prune low-signal passages and summarize when necessary to fit within context.
