import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-key")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    return TestClient(app)
