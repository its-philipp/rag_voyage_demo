import json
from typing import List, Tuple, Dict

from apps.api import create_app


def test_health_endpoint():
    app = create_app(query_func=lambda q, cfg: [])
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data == {"status": "ok"}


def test_search_endpoint_mocked():
    # Mock query_system to avoid real embedding/reranking
    mock_results: List[Tuple[float, Dict]] = [
        (1.0, {"doc_id": "d1", "title": "T", "text": "Hello world"}),
        (0.9, {"doc_id": "d2", "title": "U", "text": "Goodbye world"}),
    ]
    app = create_app(query_func=lambda q, cfg: mock_results)
    client = app.test_client()

    resp = client.post(
        "/search",
        data=json.dumps({"query": "test"}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    payload = resp.get_json()
    assert "results" in payload
    assert len(payload["results"]) == 2
    assert payload["results"][0]["doc_id"] == "d1"
