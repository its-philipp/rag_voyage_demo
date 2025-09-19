import httpx
import numpy as np


class VoyageClient:
    """Thin client for Voyage embeddings. Replace with your own client if needed."""

    def __init__(self, api_key: str, model: str = "voyage-context-3", timeout=30.0):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.base_url = "https://api.voyageai.com/v1/embeddings"

    def embed(self, texts):
        if not self.api_key:
            raise RuntimeError("VOYAGE_API_KEY missing. Set it in env or config.")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": texts}
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(self.base_url, headers=headers, json=payload)
            # If the API returned an error, include the response body for debugging
            if r.status_code >= 400:
                body = None
                try:
                    body = r.json()
                except Exception:
                    body = r.text
                raise RuntimeError(f"Voyage API error {r.status_code}: {body}")
            data = r.json()
        vecs = [np.array(item["embedding"], dtype=np.float32) for item in data["data"]]
        return np.vstack(vecs)
