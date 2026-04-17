"""Integration tests for API endpoints.

Uses FastAPI TestClient — no real OpenAI/Pinecone calls (we test route
scaffolding, auth, validation, and error paths).
"""
from unittest.mock import AsyncMock, patch

import pytest


class TestHealth:
    def test_health_no_auth_required(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestAuthMiddleware:
    def test_query_rejects_bad_key_when_configured(self, client, monkeypatch):
        monkeypatch.setenv("API_KEY", "secret-key-123")
        from app.utils.config import get_settings
        get_settings.cache_clear()

        r = client.post("/query", json={"query": "test"}, headers={"X-API-Key": "wrong"})
        assert r.status_code == 401

        get_settings.cache_clear()

    def test_query_allows_when_no_key_configured(self, client, monkeypatch):
        monkeypatch.setenv("API_KEY", "")
        from app.utils.config import get_settings
        get_settings.cache_clear()

        with patch("app.routes.query.handle_query", new_callable=AsyncMock) as mock_hq:
            mock_hq.return_value = {"answer": "test", "sources": [], "confidence": "high"}
            r = client.post("/query", json={"query": "test"})
            assert r.status_code == 200

        get_settings.cache_clear()


class TestQueryValidation:
    def test_empty_query_rejected(self, client):
        r = client.post("/query", json={"query": ""})
        assert r.status_code == 422

    def test_missing_query_rejected(self, client):
        r = client.post("/query", json={})
        assert r.status_code == 422


class TestIngestValidation:
    def test_non_pdf_rejected(self, client):
        r = client.post(
            "/ingest",
            files={"file": ("report.txt", b"text content", "text/plain")},
            data={"company": "Test", "year": "2024"},
        )
        assert r.status_code == 400

    def test_empty_file_rejected(self, client):
        r = client.post(
            "/ingest",
            files={"file": ("report.pdf", b"", "application/pdf")},
            data={"company": "Test", "year": "2024"},
        )
        assert r.status_code == 400


class TestExtractValidation:
    def test_year_must_be_4_digits(self, client):
        r = client.post("/extract", json={"company": "Apple", "year": "24"})
        assert r.status_code == 422

    def test_company_required(self, client):
        r = client.post("/extract", json={"year": "2024"})
        assert r.status_code == 422
