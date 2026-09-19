from fastapi.testclient import TestClient

from app.main import app

c = TestClient(app)


def test_health_and_search():
    assert c.get("/health").json()["status"] == "ok"
    hits = c.get("/kb/search", params={"q": "agroforestry soil carbon"}).json()["hits"]
    assert hits[0]["source_id"] == "destefano2018"


def test_chat_endpoint():
    r = c.post("/chat", json={"message": "SOC 0.3%, low rainfall, semi-arid, monoculture wheat"}).json()
    assert r["kind"] == "recommendations"
    assert c.post("/chat", json={}).status_code == 400
