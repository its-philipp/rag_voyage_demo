# Prompting for Grounding

Grounded answers cite or reflect evidence. Techniques:
- Ask the model to quote, cite, or list supporting snippets.
- Penalize unsupported claims in system prompts.
- Use constrained generation with retrieved spans (extract-then-generate).
These often improve groundedness in the RAG triad.
