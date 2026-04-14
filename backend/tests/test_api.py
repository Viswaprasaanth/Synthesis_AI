from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app

client = TestClient(app)


def test_health():
    """Health endpoint should always respond."""
    with patch("app.main.get_client"):
        response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "papers_loaded" in data


def test_synthesise_needs_papers():
    """Synthesis should fail without enough papers."""
    response = client.post(
        "/synthesise/",
        headers={"X-API-Key": "default-key"},
    )
    assert response.status_code == 400
    assert "at least 2" in response.json()["detail"].lower()


def test_ingest_rejects_non_pdf():
    """Ingest should reject non-PDF files."""
    response = client.post(
        "/ingest/",
        headers={"X-API-Key": "default-key"},
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400