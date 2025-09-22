# ColBERT vs Cross-Encoders

Cross-encoders jointly encode the [query, document] pair with full attention, enabling deep interactions but at high computational cost. They excel in reranking but are infeasible for exhaustive retrieval at scale. ColBERT decouples encoding and uses a MaxSim late interaction at query time, yielding faster retrieval while retaining fine-grained matching. Cross-encoders often achieve the highest accuracy per pair, but ColBERT provides a practical accuracy/speed trade-off for retrieval and reranking.
