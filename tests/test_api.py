"""Tests for the FastAPI app: the page in web/ is served next to the API routes."""

from fastapi.testclient import TestClient

from stormlattice.api.app import app

client = TestClient(app)


def test_root_serves_the_page():
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>StormLattice</title>" in response.text


def test_static_mount_does_not_hide_the_api():
    # The mount on "/" matches every path, so it only works if the API routes come first.
    assert client.get("/health").json()["status"] == "ok"
    assert "disclaimer" in client.get("/api/meta").json()


def test_unknown_path_is_404():
    assert client.get("/no-such-file.png").status_code == 404
