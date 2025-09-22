import os
import logging
from flask import Flask, request, jsonify
import yaml
from dotenv import load_dotenv

from query import query_system


def create_app() -> Flask:
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
        results = query_system(q, cfg)
        # results are List[Tuple[score, dict]]
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
