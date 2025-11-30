# Development Guide — jobhunter-ai-web-tool

This document describes the API routes, request/response shapes, required environment variables, how to run the frontend/backend/database locally, and a collection of example curl requests for quick testing.

Base URLs 
## 1) API Routes
Below is a simple, non-technical description of what the backend can do and how to use it from the app.

- Check server status
  - URL: `GET /api/health`
  - What it does: Quickly tells you if the backend is running and whether it has a database connected.

- Create an account (Sign up)
  - URL: `POST /api/auth/register`
  - What to send: your full name, email and password.
  - What you get back: a short-lived login token and your user info. The app stores the token so you stay signed in.

- Sign in
  - URL: `POST /api/auth/login`
  - What to send: your email and password.
  - What you get back: a login token and your user info.

- Your profile (view / update)
  - URL: `GET /api/users/me` and `PUT /api/users/me/profile`
  - What it does: view or edit your name, contact info, location, and job preferences. You must be signed in.

- Saved jobs
  - URL: `GET /api/users/me/saved-jobs` — view jobs you've saved
  - URL: `POST /api/users/me/saved-jobs` — save a job (send job id)
  - URL: `DELETE /api/users/me/saved-jobs/<job_id>` — remove a saved job
  - What it does: keeps a list of jobs you want to review later. You must be signed in.

- Applied jobs
  - URL: `POST /api/users/me/applied-jobs` — mark a job as applied
  - URL: `GET /api/users/me/applied-jobs` — view applied jobs
  - What it does: track jobs you've applied to. You must be signed in.

- Search for jobs
  - URL: `POST /api/jobs/search`
  - What to send: keywords (like "data analyst"), optional location, and simple filters (experience level, job type).
  - What you get back: a list of matching jobs with brief details. You can click into a job to see the full description.

- Job details
  - URL: `GET /api/jobs/<job_id>`
  - What it does: returns the full job description and other details for display in a modal or detail view.

- Upload or paste a resume
  - URL: `POST /api/resumes`
  - What to send: either a pasted resume text (JSON) or a file (PDF/DOCX/TXT).
  - What you get back: an id for the uploaded resume.

- Generate a cover letter
  - URL: `POST /api/ai/cover-letter`
  - What to send: a job (or job id) and a resume (or resume id). Optionally include your name.
  - What you get back: a generated cover letter and suggested resume bullets.

- Ask the assistant (chat)
  - URL: `POST /api/chat`
  - What it does: sends a short message to the server assistant and returns a text reply. Note: this requires the server to be configured with a provider key.

Authentication note 
- When you sign in, the app gets a token and keeps it for you. You don't need to copy anything manually — the app will send the token with requests that need it.

Developer note
- For local development the server can run without a database and will store data in memory (useful for quick testing). If you run with a database, data will persist between restarts.

---

## 2) .env variables (important)
Place a `.env` file in the project root (do NOT commit it). Example variables used by the app:

- JWT_SECRET
  - Description: HS256 symmetric secret used to sign/verify JWTs. Use a random 256-bit value.
  - Example: `JWT_SECRET=__REPLACE_WITH_RANDOM__`

- PORT (optional)
  - Backend port (default 5001)
  - Example: `PORT=5001`

- DB_HOST, DB_USER, DB_PASSWORD, DB_NAME
  - MySQL connection parameters. If not set, the server uses in-memory fallback for development.
  - Example:
    ```env
    DB_HOST=127.0.0.1
    DB_USER=root
    DB_PASSWORD=supersecret
    DB_NAME=jobhunter
    ```

- GEMINI_API_KEY (optional)
  - If set, powers `/api/chat` and AI cover-letter generation.

- ADZUNA_APP_ID, ADZUNA_APP_KEY (optional)
  - For job provider integration used by `POST /api/jobs/search`.

- CORS_ALLOW_ORIGINS (optional)
  - Comma-separated list of allowed origins, e.g. `http://localhost:5173`.

- Frontend (Vite) env (optional):
  - `VITE_API_BASE` — override the API base URL the frontend uses (defaults to http://localhost:5001/api in dev).

---

## Environment / JWT secret

To generate a secure JWT secret locally:

```bash
# 256-bit (32-byte) hex string
openssl rand -hex 32
```

Or with Python:

```bash
python - <<'PY'
import secrets
print(secrets.token_urlsafe(48))
PY
```

Notes:
- If you rotate `JWT_SECRET`, existing tokens will become invalid.

## 3) Setup and run locally

Prereqs
- Node.js (18+ recommended), npm or pnpm
- Python 3.10+ and `virtualenv`/venv
- (Optional) MySQL server or Docker to run MySQL

Backend (Python / Flask)

1. Create & activate virtualenv

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install Python deps

```bash
pip install -r requirements.txt
```

3. Create a `.env` in the project root with at least `JWT_SECRET` (and DB_* if using MySQL)

```bash
# copy and edit
cp .env.example .env  # if you have an example file
# or create manually
export JWT_SECRET=$(openssl rand -hex 32)
# (or add to .env file)
```

4. (Optional) Start MySQL (local) or using Docker

- Docker command (example):

```bash
docker run --name jobhunter-mysql -e MYSQL_ROOT_PASSWORD=pass -e MYSQL_DATABASE=jobhunter -p 3306:3306 -d mysql:8.0
```

- Wait for MySQL to be ready and then import schema (if you have `backend/database/scripts/schema.sql`):

```bash
mysql -u root -p jobhunter < backend/database/scripts/schema.sql
```

5. Run the backend

```bash
# from repo root
cd backend
python app.py
# or
python -m backend.app
```

The server listens on port set by `PORT` env (default 5001). API base will be `http://localhost:5001/api`.

Frontend (React + Vite)

1. Install dependencies

```bash
cd frontend
npm install
```

2. Run development server

```bash
npm run dev
```

3. Build for production

```bash
npm run build
```

Make sure `VITE_API_BASE` in the frontend env (or default) points to the backend API (http://localhost:5001/api).

---

## 4) Example curl requests

Note: replace `http://localhost:5001` with your API base and `<token>` with token returned from login/register.

Register

```bash
curl -s -X POST http://localhost:5001/api/auth/register \
  -H 'Content-Type: application/json' \
  -d '{ "full_name": "Alice Example", "email": "alice@example.com", "password": "testpass123" }'
```

Login

```bash
curl -s -X POST http://localhost:5001/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{ "email": "alice@example.com", "password": "testpass123" }'
```

Get profile (protected)

```bash
curl -s -X GET http://localhost:5001/api/users/me \
  -H "Authorization: Bearer <token>"
```

Search jobs

```bash
curl -s -X POST http://localhost:5001/api/jobs/search \
  -H 'Content-Type: application/json' \
  -d '{ "query": "data analyst", "location": "Chicago" }'
```

Get job details

```bash
curl -s http://localhost:5001/api/jobs/101
```

Save a job (protected)

```bash
curl -s -X POST http://localhost:5001/api/users/me/saved-jobs \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -d '{ "job_id": 101, "notes": "save for later" }'
```

List saved jobs (protected)

```bash
curl -s -X GET http://localhost:5001/api/users/me/saved-jobs \
  -H 'Authorization: Bearer <token>'
```

Apply to a job (protected)

```bash
curl -s -X POST http://localhost:5001/api/users/me/applied-jobs \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -d '{ "job_id": 101 }'
```

Upload resume (file)

```bash
curl -s -X POST http://localhost:5001/api/resumes \
  -H 'Authorization: Bearer <token>' \
  -F file=@/path/to/resume.pdf
```

Generate cover letter

```bash
curl -s -X POST http://localhost:5001/api/ai/cover-letter \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -d '{ "job_id": 101, "resume_id": 5, "candidate_name": "Alice" }'
```

Chat (requires provider configured)

```bash
curl -s -X POST http://localhost:5001/api/chat \
  -H 'Content-Type: application/json' \
  -d '{ "message": "Hello assistant" }'
```

---

## Troubleshooting & tips

- If you get 401 on protected endpoints:
  - Ensure the frontend sent `Authorization: Bearer <token>`.
  - Ensure `JWT_SECRET` used by the server matches the secret used to sign tokens (tokens issued after login/register use the server's `JWT_SECRET`).
  - Check token expiry (tokens are issued with a 7-day expiry in the current code).

- If you prefer not to run a local MySQL server while developing, the backend will operate with an in-memory fallback (`MEM`) but data will not persist across restarts.

- To regenerate a secure secret:

```bash
openssl rand -hex 32 # 256-bit secret
```

---
