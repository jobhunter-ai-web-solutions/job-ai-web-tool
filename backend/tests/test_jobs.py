from unittest.mock import patch

def test_job_search_success(client):
    fake_response = {
        "results": [{
            "title": "Data Analyst",
            "company": {"display_name": "MockCorp"},
            "location": {"display_name": "New York"},
            "description": "Analyze data.",
            "salary_min": 60000,
            "salary_max": 80000,
            "redirect_url": "http://example.com/job",
            "category": {"label": "Tech"}
        }],
        "count": 1
    }

    with patch("requests.get") as mock:
        mock.return_value.json.return_value = fake_response
        mock.return_value.status_code = 200

        res = client.post("/api/jobs/search", json={"query": "Data"})
        assert res.status_code == 200
        assert res.json["count"] == 1
