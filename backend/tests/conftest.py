import os
import pytest
from app import app as flask_app
from unittest.mock import patch

@pytest.fixture
def client():
    """
    Creates a test client with a test database configuration. 
    Uses memory mode  unless DB variables are provided.
    """
    flask_app.config["TESTING"] = True
    flask_app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

    # Force memory DB mode
    os.environ["DB_HOST"] = ""
    os.environ["DB_USER"] = ""
    os.environ["DB_NAME"] = ""

    with flask_app.test_client() as client:
        yield client


@pytest.fixture
def auth_header():
    """Fake authentication header using a dummy JWT."""
    return {"Authorization": "Bearer testtoken"}


# Mock Gemini so tests never hit external API
@pytest.fixture
def mock_gemini():
    with patch("google.generativeai.GenerativeModel") as mock:
        instance = mock.return_value
        instance.generate_content.return_value.text = "Mocked AI response"
        yield mock
