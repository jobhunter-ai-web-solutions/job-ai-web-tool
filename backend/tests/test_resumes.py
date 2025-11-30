def test_upload_resume_text(client, auth_header):
    res = client.post("/api/resumes", json={
        "text": "This is my resume",
        "meta": {"name": "Napoleon"}
    }, headers=auth_header)

    assert res.status_code == 200
    assert "resume_id" in res.json


def test_upload_resume_file_pdf(client, auth_header):
    with open("backend/tests/sample_resume.txt", "rb") as f:
        res = client.post("/api/resumes",
            headers=auth_header,
            data={"file": (f, "resume.txt")}
        )
    assert res.status_code == 200
    assert "resume_id" in res.json
