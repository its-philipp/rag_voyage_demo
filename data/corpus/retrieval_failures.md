# Common Retrieval Failures

Failure modes include:
- Vocabulary mismatch: dense helps; add synonyms, augment data.
- Ambiguity: query decomposition or disambiguation prompts.
- Over-chunking: too small windows fragment context.
- Poor metadata: missing titles harm reranking.
Diagnostics: manual query audits, coverage tests, and recall at K.
