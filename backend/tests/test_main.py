from fastapi.testclient import TestClient
from backend.main import app
import time

client = TestClient(app)


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
    assert response.status_code == 403


def test_get_videos_no_auth():
    response = client.get("/videos")
    assert response.status_code == 403


def test_query_no_auth():
    response = client.post("/query", json={"question": "test"})
    assert response.status_code == 403


def test_full_auth_flow():
    username = unique_username()
    client.post("/auth/register", json={"username": username, "password": "flowpassword"})
    login_response = client.post("/auth/login", json={"username": username, "password": "flowpassword"})
    token = login_response.json()["access_token"]
    assert token and len(token) > 0

    videos_response = client.get("/videos", headers={"Authorization": f"Bearer {token}"})
    assert videos_response.status_code == 200
