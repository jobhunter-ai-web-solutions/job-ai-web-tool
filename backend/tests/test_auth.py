def test_register_success(client):
    res = client.post("/api/auth/register", json={
        "full_name": "Test User",
        "email": "test@example.com",
        "password": "Password123"
    })
    assert res.status_code == 200
    assert "token" in res.json


def test_register_duplicate_email(client):
    client.post("/api/auth/register", json={
        "full_name": "Test User",
        "email": "test@example.com",
        "password": "Password123"
    })

    res = client.post("/api/auth/register", json={
        "full_name": "Test Again",
        "email": "test@example.com",
        "password": "Password123"
    })
    assert res.status_code == 409


def test_login_success(client):
    # First register
    client.post("/api/auth/register", json={
        "full_name": "Test User",
        "email": "login@example.com",
        "password": "Password123"
    })
    res = client.post("/api/auth/login", json={
        "email": "login@example.com",
        "password": "Password123"
    })
    assert res.status_code == 200
    assert "token" in res.json


def test_login_bad_password(client):
    client.post("/api/auth/register", json={
        "full_name": "Test User",
        "email": "wrongpass@example.com",
        "password": "Password123"
    })
    res = client.post("/api/auth/login", json={
        "email": "wrongpass@example.com",
        "password": "badpass"
    })
    assert res.status_code == 401
