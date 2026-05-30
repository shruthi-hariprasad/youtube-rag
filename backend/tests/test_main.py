from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from backend.main import app
import time

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_VIDEO_ID = "dQw4w9WgXcQ"
FAKE_URL = f"https://www.youtube.com/watch?v={FAKE_VIDEO_ID}"

FAKE_SEGMENTS = [
    {"text": "Hello world this is a test transcript segment.", "start": 0.0},
    {"text": "It contains enough words to exercise the chunker.", "start": 5.0},
]

FAKE_OEMBED = {
    "title": "Test Video",
    "author_name": "Test Channel",
    "thumbnail_url": "https://img.youtube.com/vi/dQw4w9WgXcQ/0.jpg",
}


def _register_and_login(username: str | None = None) -> str:
    username = username or unique_username()
    client.post("/auth/register", json={"username": username, "password": "pw"})
    resp = client.post("/auth/login", json={"username": username, "password": "pw"})
    return resp.json()["access_token"]


def unique_username():
    return f"testuser_{int(time.time() * 1000)}"


def test_register():
    response = client.post("/auth/register", json={
        "username": unique_username(),
        "password": "testpassword"
    })
    assert response.status_code == 200
    assert response.json()["message"] == "User created successfully"


def test_register_duplicate_username():
    username = unique_username()
    client.post("/auth/register", json={"username": username, "password": "pw"})
    response = client.post("/auth/register", json={"username": username, "password": "pw"})
    assert response.status_code == 400
    assert "taken" in response.json()["detail"].lower()


def test_register_invalid_username():
    response = client.post("/auth/register", json={
        "username": "no spaces allowed",
        "password": "pw"
    })
    assert response.status_code == 400


def test_login():
    username = unique_username()
    client.post("/auth/register", json={"username": username, "password": "testpassword"})
    response = client.post("/auth/login", json={"username": username, "password": "testpassword"})
    assert response.status_code == 200
    assert "access_token" in response.json()


def test_login_wrong_password():
    username = unique_username()
    client.post("/auth/register", json={"username": username, "password": "correct"})
    response = client.post("/auth/login", json={"username": username, "password": "wrong"})
    assert response.status_code == 401


def test_add_video_no_auth():
    response = client.post("/videos?url=https://www.youtube.com/watch?v=test")
    assert response.status_code in (401, 403)


def test_get_videos_no_auth():
    response = client.get("/videos")
    assert response.status_code in (401, 403)


def test_query_no_auth():
    response = client.post("/query", json={"question": "test"})
    assert response.status_code in (401, 403)


def test_full_auth_flow():
    username = unique_username()
    client.post("/auth/register", json={"username": username, "password": "flowpassword"})
    login_response = client.post("/auth/login", json={"username": username, "password": "flowpassword"})
    token = login_response.json()["access_token"]
    assert token and len(token) > 0

    videos_response = client.get("/videos", headers={"Authorization": f"Bearer {token}"})
    assert videos_response.status_code == 200


# ---------------------------------------------------------------------------
# Video ingestion integration tests (external calls mocked)
# ---------------------------------------------------------------------------

def _mock_transcript(segments):
    """Build a mock object that YouTubeTranscriptApi().fetch() returns."""
    items = []
    for s in segments:
        m = MagicMock()
        m.text = s["text"]
        m.start = s["start"]
        items.append(m)
    return items


def _add_video_patched(headers: dict) -> dict:
    """POST /videos with all external calls mocked. Returns the response JSON."""
    oembed_resp = MagicMock(status_code=200)
    oembed_resp.json.return_value = FAKE_OEMBED
    with (
        patch("backend.main.requests.get", return_value=oembed_resp),
        patch("backend.main.YouTubeTranscriptApi") as mock_ytt,
        patch("backend.main._ingest_video"),  # skip background DB/embedding work
    ):
        mock_ytt.return_value.fetch.return_value = _mock_transcript(FAKE_SEGMENTS)
        resp = client.post(f"/videos?url={FAKE_URL}", headers=headers)
    return resp


def test_add_video_and_list():
    token = _register_and_login()
    headers = {"Authorization": f"Bearer {token}"}

    resp = _add_video_patched(headers)

    assert resp.status_code == 202
    data = resp.json()
    assert data["youtube_video_id"] == FAKE_VIDEO_ID
    assert data["title"] == "Test Video"

    list_resp = client.get("/videos", headers=headers)
    assert list_resp.status_code == 200
    assert any(v["youtube_video_id"] == FAKE_VIDEO_ID for v in list_resp.json())


def test_add_duplicate_video():
    token = _register_and_login()
    headers = {"Authorization": f"Bearer {token}"}

    _add_video_patched(headers)
    resp = _add_video_patched(headers)

    assert resp.status_code == 400
    assert "already added" in resp.json()["detail"].lower()


def test_delete_video():
    token = _register_and_login()
    headers = {"Authorization": f"Bearer {token}"}

    add_resp = _add_video_patched(headers)
    video_db_id = add_resp.json()["id"]

    with patch("backend.main.delete_chunks"):
        del_resp = client.delete(f"/videos/{video_db_id}", headers=headers)

    assert del_resp.status_code == 200
    list_resp = client.get("/videos", headers=headers)
    assert not any(v["id"] == video_db_id for v in list_resp.json())


def test_query_after_add():
    token = _register_and_login()
    headers = {"Authorization": f"Bearer {token}"}

    fake_chunk = {
        "video_id": FAKE_VIDEO_ID,
        "text": "Hello world this is a test transcript segment.",
        "score": 0.9,
        "start_time": 0.0,
    }

    _add_video_patched(headers)

    with (
        patch("backend.main.retrieve_chunks", return_value=[fake_chunk]),
        patch("backend.main.generate_answer", return_value={"answer": "test answer", "sources": []}),
    ):
        query_resp = client.post("/query", json={"question": "what is this about?"}, headers=headers)

    assert query_resp.status_code == 200
    assert "answer" in query_resp.json()
