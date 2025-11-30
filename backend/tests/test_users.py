def test_get_user_profile_unauthorized(client):
    res = client.get("/api/users/me")
    assert res.status_code == 401


def test_update_user_profile_success(client, auth_header):
    # Register first
    client.post("/api/auth/register", json={
        "full_name": "Test User",
        "email": "profile@example.com",
        "password": "Password123"
    })

    res = client.put("/api/users/me/profile", json={
        "name": "Updated User"
    }, headers={"X-User-Id": "1"})
    assert res.status_code == 200
