# Developer Guide — jobhunter-ai-web-tool

This document is a technical reference for developers working on the codebase. It focuses on the backend parsing and matching pipeline, field profiles, environment configuration, local development, example requests, and troubleshooting.

Summary
- Backend: `backend/app.py` (Flask API)
- Parser: `ml/pre_llm_filter_functions.py` (pre-LLM parsing helpers)
- Field profiles: `backend/field_profiles.py` (+ optional exported `backend/field_profiles.json`)
- Frontend: `frontend/` (React + Vite)
- DB schema: `backend/database/scripts/schema.sql`

Principles
- Matching is deterministic and rule-based; LLMs are usedfor cover-letter generation and optional parsing refinement.
- Field-aware matching uses curated `FIELD_PROFILES` to bias scoring toward priority skills.
- Lightweight fuzzy matching tolerates common near-miss tokens without using heavy NLP models.

Contents
1. Architecture overview
2. API reference (developer)
3. Field profiles
4. Parsing and resume skill derivation
5. Matching pipeline & scoring
6. Environment variables and feature flags
7. Local development: backend, frontend, DB, venv
8. Example requests and smoke test
9. Troubleshooting and tuning
10. Contribution notes

---

1) Architecture overview

- The repository separates concerns:
  - `backend/`: Flask app exposing REST endpoints for auth, resumes, jobs, matching, and AI cover-letter/chat.
  - `ml/`: parsing helpers and any lightweight ML utilities. The function `ParsingFunctionsPreLLM` lives in `ml/pre_llm_filter_functions.py`.
  - `frontend/`: React + Vite UI that calls the backend API.
- Data storage: optional MySQL (schema available in `backend/database/scripts/schema.sql`). If DB env vars are not set, the app uses an in-memory store called `MEM` for development.

---

2) API reference (developer)

Key endpoints the guide focuses on for parsing and matching:

- `GET /api/health` — health check (includes DB status when configured).
- `POST /api/resumes` — upload a resume (file or text). Returns `resume_id`.
- `GET /api/resumes/<id>/parsed` — returns parsed sections and contacts from the ML microservice.
- `POST /api/match` — score a single `resume` (or `resume_id`) against a single `job` object or `job_id`.
  - Response includes: `score_percent`, `matched_skills`, `gaps`, `derived_resume_skills`, `job_phrases`.
- `POST /api/recommend` — batch score multiple jobs for a given resume. Supports a field-aware matcher via flags.
- `POST /api/ai/cover-letter` and `POST /api/chat` — require `GEMINI_API_KEY` and are used for LLM functions only.

For full endpoint details and auth, read `backend/app.py` and the route docstrings.

---

3) Field profiles

Location: `backend/field_profiles.py`.

- Purpose: centralize domain knowledge per "field" (e.g., `data`, `engineering`, `product`) so matching can emphasize priority skills.
- Structure for each profile:
  - `weights`: weight multipliers for `skills`, `title`, and `description` similarity.
  - `priority_skills`: a short list of high-importance tokens that should boost a candidate's score when present.
  - `synonyms`: mapping to canonical tokens (e.g., `js` -> `javascript`).

APIs in the module:
- `to_dict()` — returns the full profiles as a plain dict.
- `export_to(path)` — writes `FIELD_PROFILES` to JSON at the given path.

Runtime behavior:
- The backend prefers importing the module directly at startup. If importing fails, it falls back to loading `backend/field_profiles.json` if present.
- Use `export_to()` to regenerate the JSON after you update the module when you want a standalone JSON snapshot.

Editing tips:
- Keep `priority_skills` short and precise (3–7 tokens).
- Add common synonyms under `synonyms` to improve exact-token mapping.

---

4) Parsing and resume skill derivation

Location: `ml/pre_llm_filter_functions.py` and integration in `backend/app.py`.

Pipeline summary:
1. File extraction: PDFs/DOCX/TXT are converted to raw text by utilities in `ml/`.
2. Section detection: the parser attempts to segment text into sections (education, experience, skills, etc.). This is heuristic-based.
3. Contact extraction: basic heuristics (regex) pull emails and phone numbers.
4. Resume skill derivation: `_extract_resume_skills(text, user_list)` (in `backend/app.py`) finds tokens by scanning `BASE_SKILLS` + `user_list` and also tries to include `FIELD_PROFILES` tokens where appropriate.

Notes:
- Matching uses case-insensitive substring presence for simple token hits. The pipeline also applies a lightweight RAKE-like extraction (`_rake_extract`) to build `job_phrases` from job title/description/company, which improves phrase-level matching.
- The parser purposely avoids overfitting to section names; instead it focuses on extracting candidate tokens and phrases.

---

5) Matching pipeline & scoring

Major functions (see `backend/app.py`):

- `_tokenize_and_map(s, synonyms)`: lowercase, remove punctuation and stopwords, split to tokens, apply `synonyms` mapping.
- `_skill_overlap(resume_tokens, job_tokens, synonyms)`: counts matches between resume tokens and job/profile tokens. It accepts exact token matches and substring presence (e.g., `postgres` matches `postgresql`). It also applies lightweight fuzzy matching for common near-misses.
- `_rake_extract(text, top_n)`: extracts candidate phrases for `job_phrases`.
- `_jaccard(a, b)`: computes Jaccard similarity between token sets for title/description similarity.
- `_simple_match_score(resume_text, job, use_loose=False)`: orchestrates the scoring:
  - Detect job field with `_detect_field(job)` using `_FIELD_KEYWORDS` from `field_profiles.py`.
  - Load profile `weights` and `priority_skills` for field if available.
  - Compute `skill_sim`, `title_sim`, and `desc_sim`; combine with weights.
  - Optionally apply `USE_LOOSE_SCORING` and `SCORE_LOOSENESS` to be more generous.
  - Map the combined 0..1 similarity into an integer `score_percent` (0–100) with clamping.

Gaps and matched skills:
- `matched_skills` lists resume-derived skills found in the job's skill set (after synonyms mapping).
- `gaps` lists high-priority profile skills that the resume does not contain.

Tuning:
- `weights` per profile determine sensitivity to `skills` vs `title` vs `description`.
- `SCORE_LOOSENESS` increases the mapped output to favor higher scores.
- `USE_LOOSE_SCORING=true` is useful in early-stage product UX testing to increase recall.

---

6) Environment variables and feature flags

Create a `.env` at the project root with required keys. Important keys for dev and matching:

- `JWT_SECRET` — required for auth.
- `PORT` — defaults to `5001`.
- `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` — optional; if omitted, `MEM` in-memory store is used.
- `GEMINI_API_KEY`, `GEMINI_MODEL` — optional; for LLM cover-letter/chat features.
- `ADZUNA_APP_ID`, `ADZUNA_APP_KEY` — optional; for job search API.

Feature flags (env or request-level overrides):
- `USE_FIELD_AWARE_MATCHER` — when `true`, `POST /api/recommend` and internal scoring prefer field-based weights.
- `USE_LOOSE_SCORING` — when `true`, scoring mapping is more generous.
- `SCORE_LOOSENESS` — numeric float to tune looseness (e.g., `0.1` to add 10% looseness).

You can also pass `use_field_aware` in request bodies to toggle field-aware matching per-request.

---

7) Local development: backend, frontend, DB, venv

Prereqs: Python 3.10+ (the repo may work with newer Python versions), Node.js + npm, optional MySQL.

Backend (recommended):

```bash
cd /path/to/job-ai-web-tool
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# set up .env (see section 6)
cd backend
python app.py
```

If you use the MySQL DB, import the schema:

```bash
mysql -u root -p < backend/database/scripts/schema.sql
# or
mysql -u jobhunter -p jobhunter < backend/database/scripts/schema.sql
```

Frontend:

```bash
cd frontend
npm install
npm run dev
# set VITE_API_BASE to http://localhost:5001 or whatever your backend uses
```

ML parser (optional local dev):
- `ml/` contains helpers for parsing. If you need a separate microservice, create a small Flask/FastAPI wrapper that calls functions in `ml/pre_llm_filter_functions.py` and exposes `/parse-resume`.

---

8) Example requests and smoke test

Score a single job (match):

```bash
curl -s -X POST http://localhost:5001/api/match \
  -H 'Content-Type: application/json' \
  -d '{"resume_text":"Experienced data engineer with Python, SQL, PostgreSQL.", "job": {"title":"Data Engineer","description":"We need Python and Postgres experience","skills":["python","postgresql"]}}'
```

Batch recommend (recommend):

```bash
curl -s -X POST http://localhost:5001/api/recommend \
  -H 'Content-Type: application/json' \
  -d '{"resume_text":"Python, SQL, Spark","jobs":[{"id":1,"title":"Data Engineer","description":"...","skills":["python","spark"]}], "use_field_aware": true}'
```

Quick Python smoke test (save as `scripts/smoke_match.py` and run from project root while backend is running):

```python
import requests
url = "http://localhost:5001/api/match"
payload = {
  "resume_text": "SQL, Python, pandas, aws",
  "job": {"title":"Data Analyst","description":"Experience with Python and SQL","skills":["python","sql"]}
}
print(requests.post(url, json=payload).json())
```

---

9) Troubleshooting and tuning

- If `backend/app.py` fails to import `backend/field_profiles.py`, the app will try to load `backend/field_profiles.json`.
- If you see score behavior that is too strict: enable `USE_LOOSE_SCORING=true` or increase `SCORE_LOOSENESS`.
- If scores are too permissive: lower `SCORE_LOOSENESS` and ensure `priority_skills` are accurate and minimal.
- For parsing issues (mis-detected sections), inspect `GET /api/resumes/<id>/parsed` when running with the in-memory `MEM` store.
- If a package in `.venv` was modified unexpectedly (e.g., WatchFiles reloads), re-create the venv and reinstall with `pip install -r requirements.txt` to ensure integrity.

---

10) Contribution notes

- Keep changes small and well-scoped. For matching logic updates:
  - Add unit tests or a smoke script showing before/after example payloads.
  - Update `backend/field_profiles.py` and call `export_to('backend/field_profiles.json')` if you want the JSON snapshot updated.
- Style: follow existing repo patterns. Avoid adding heavy ML dependencies for matching; prefer lightweight, explainable heuristics.

---

End of developer guide.
