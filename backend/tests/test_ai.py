def test_chat_ai_success(client, mock_gemini):
    res = client.post("/api/chat", json={
        "message": "Hello!"
    })
    assert res.status_code == 200
    assert "reply" in res.json
    assert res.json["reply"] == "Mocked AI response"
