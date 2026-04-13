from fastapi.testclient import TestClient
from backend.main import app
import time

client = TestClient(app)

# Test 1: Register a new user
def test_register():
    email = f"test_{int(time.time())}@example.com"
    response = client.post("/auth/register", json={
        "email": email,
        "password": "testpassword"
    })
    assert response.status_code == 200
    assert "message" in response.json()

# Test 2: Register with same email twice should fail
def test_register_duplicate_email():
    client.post("/auth/register", json={
        "email": "duplicate@example.com",
        "password": "testpassword"
    })
    response = client.post("/auth/register", json={
        "email": "duplicate@example.com",
        "password": "testpassword"
    })
    assert response.status_code == 400

# Test 3: Login with correct credentials
def test_login():
    client.post("/auth/register", json={
        "email": "login@example.com",
        "password": "testpassword"
    })
    response = client.post("/auth/login", json={
        "email": "login@example.com",
        "password": "testpassword"
    })
    assert response.status_code == 200
    assert "access_token" in response.json()

# Test 4: Login with wrong password
def test_login_wrong_password():
    client.post("/auth/register", json={
        "email": "wrong@example.com",
        "password": "correctpassword"
    })
    response = client.post("/auth/login", json={
        "email": "wrong@example.com",
        "password": "wrongpassword"
    })
    assert response.status_code == 401

# Test 5: Add video without auth token
def test_add_video_no_auth():
    response = client.post("/videos?url=https://www.youtube.com/watch?v=test")
    assert response.status_code == 401

# Test 6: Get videos without auth token
def test_get_videos_no_auth():
    response = client.get("/videos")
    assert response.status_code == 401

# Test 7: Query without auth token
def test_query_no_auth():
    response = client.post("/query?question=test")
    assert response.status_code == 401

# Test 8: Full auth flow - register, login, get token
def test_full_auth_flow():
    client.post("/auth/register", json={
        "email": "flow@example.com",
        "password": "flowpassword"
    })
    login_response = client.post("/auth/login", json={
        "email": "flow@example.com",
        "password": "flowpassword"
    })
    token = login_response.json()["access_token"]
    assert token is not None
    assert len(token) > 0