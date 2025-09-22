import os
import logging
from flask import Flask, request, jsonify
import yaml
from dotenv import load_dotenv


def create_app(query_func=None) -> Flask:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    app = Flask(__name__)

    cfg = yaml.safe_load(open("config.yaml"))

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/search")
    def search():
        data = request.get_json(force=True, silent=True) or {}
        q = data.get("query")
        if not q:
            return jsonify({"error": "Missing 'query'"}), 400
        # Lazy injection to avoid heavy imports during testing
        nonlocal query_func
        if query_func is None:
            from src.pipeline import query_system as _query_system  # type: ignore

            query_func = _query_system
        results = query_func(q, cfg)
        out = [
            {
                "score": float(score),
                "doc_id": item.get("doc_id"),
                "title": item.get("title"),
                "text": item.get("text"),
            }
            for score, item in results
        ]
        return jsonify({"results": out})

    return app


if __name__ == "__main__":
    app = create_app()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    app.run(host=host, port=port)
