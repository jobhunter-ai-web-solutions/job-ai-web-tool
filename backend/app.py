# backend/app.py
from __future__ import annotations
import os, re, json, random, string
import requests
from datetime import datetime, timezone 
from typing import Any, Dict, List, Tuple

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml.pre_llm_filter_functions import ParsingFunctionsPreLLM
from ml.cover_letter_generator import CoverLetterGenerator

from pathlib import Path
import mammoth
import pdfplumber
from io import BytesIO
import html as _html

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from dotenv import load_dotenv

# Optional MySQL
import mysql.connector
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException

# Gemini
import google.generativeai as genai
import bcrypt
import jwt
from datetime import timedelta

# ML imports
from ml.pre_llm_filter_functions import ParsingFunctionsPreLLM
from ml.cover_letter_generator import CoverLetterGenerator
from ml.gemini_client import GeminiClient
from auth import require_auth 

# -----------------------------
# Env & app
# -----------------------------
load_dotenv()
app = Flask(__name__)

# Config: allow configuring how many characters of job description to return
JOB_DESCRIPTION_MAX_CHARS = int(os.getenv("JOB_DESCRIPTION_MAX_CHARS", "2000"))

# Log whether Adzuna creds are available (do not print the secrets)
if os.getenv("ADZUNA_APP_ID") and os.getenv("ADZUNA_APP_KEY"):
    app.logger.info("Adzuna credentials found in environment")
else:
    app.logger.warning(
        "Adzuna credentials not found in environment; "
        "set ADZUNA_APP_ID and ADZUNA_APP_KEY in .env or env"
    )

# CORS allowlist (env) or permissive for dev
_allow = os.getenv("CORS_ALLOW_ORIGINS", "")
origins = [o.strip() for o in _allow.split(",") if o.strip()]

# Always give Flask-CORS a list (never None)
if not origins:
    origins = ["*"]  # Allow all for local dev

# JWT Utilities
JWT_SECRET = os.getenv("JWT_SECRET", "super-secret-key")
JWT_ALGO = "HS256"
JWT_EXPIRE_MINUTES = 30     # access token duartion

@app.errorhandler(413)
def too_large(e):
    return bad("File too large (max 5MB)", 413)

@app.errorhandler(Exception)
def handle_uncaught(e):
    if isinstance(e, HTTPException):
        return e
    app.logger.exception("Unhandled exception", exc_info=e)
    return bad("Server error", 500)

CORS(app, origins=origins, supports_credentials=True)

# -----------------------------
# Gemini config
# -----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    app.logger.info(f"Gemini model set to {GEMINI_MODEL}")
else:
    app.logger.warning("GEMINI_API_KEY not set; /api/chat will return an error")

# -----------------------------
# DB helpers (uses your global connection style, but with safe fallback)
# -----------------------------
USE_DB = all(os.getenv(k) for k in ["DB_HOST", "DB_USER", "DB_NAME"])
_db = None
_cursor = None


def get_db():
    global _db, _cursor
    if not USE_DB:
        return None, None
    try:
        if _db is None or not _db.is_connected():
            _db = mysql.connector.connect(
                host=os.getenv("DB_HOST"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                database=os.getenv("DB_NAME"),
                autocommit=True,
                use_pure=True,
            )
            _cursor = _db.cursor(dictionary=True)
        return _db, _cursor
    except Exception as e:
        app.logger.warning(f"MySQL unavailable, using memory store. Error: {e}")
        return None, None


# Try to connect once at startup
# get_db()

# -----------------------------
# Resume Upload Helpers
# -----------------------------

# Define a max file size of 5MB
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# Only allow PDFs, Open XML, and TXT files 
ALLOWED_MIME = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
}

# *** CHANGE LATER IF NEEDED ***
def _get_user_id():
    """
    Grab user id for the authenticated user.
    """
    # Prefer Authorization: Bearer <token> (JWT). Fall back to X-User-Id header (dev/testing).
    auth = request.headers.get("Authorization") or request.headers.get("authorization")
    if auth and auth.startswith("Bearer "):
        token = auth.split(None, 1)[1].strip()
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
            uid = payload.get("user_id")
            try:
                return int(uid) if uid is not None else None
            except Exception:
                return None
        except Exception:
            # invalid token — fall back to X-User-Id below
            pass

    h = request.headers.get("X-User-Id")
    try:
        return int(h) if h is not None else None
    except Exception:
        return None

# -----------------------------
# In-memory fallback store
# -----------------------------
MEM = {
    "resumes": {},     # resume_id -> {"user_id":..., "text":..., "file_name":..., "name":..., "skills":[...]}
    "jobs": {},        # job_id -> job dict
    "next_resume_id": 1,
    "next_job_id": 1,
}


def _gen_id(prefix="J", n=6):
    return prefix + "".join(random.choices(string.ascii_uppercase + string.digits, k=n))


def ok(data: Dict[str, Any] | List[Any] | str = "ok", code: int = 200):
    if isinstance(data, str):
        data = {"status": data}
    return jsonify(data), code


def bad(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


# Simple skill extraction / scoring utilities
_BASE_SKILLS = {
    "python", "sql", "excel", "power bi", "tableau", "snowflake", "pandas", "numpy", "r",
    "java", "javascript", "react", "node", "api", "rest", "fastapi", "flask",
    "dashboards", "kpi", "etl", "data pipeline", "airflow", "docker", "kubernetes",
    "git", "jira", "experimentation", "a b testing", "a/b testing", "statistics",
    "forecast", "supply chain", "sap", "ibp", "ml", "machine learning", "genai",
}


def _extract_resume_skills(text: str, user_list: List[str] | None = None) -> List[str]:
    text_l = text.lower()
    hits = []
    for sk in _BASE_SKILLS.union({s.lower() for s in (user_list or [])}):
        if sk and sk in text_l:
            hits.append(sk)
    hits.sort(key=lambda k: text_l.find(k))
    return [h.upper() if h in {"sql", "r"} else h.title() for h in hits]


def _match_score(resume_text: str, job_skills: List[str]) -> Tuple[int, List[str]]:
    text = resume_text.lower()
    hits = [k for k in job_skills if k.lower() in text]
    score = max(40, min(99, 60 + 7 * len(hits)))
    gaps = [k for k in job_skills if k.lower() not in text]
    return score, gaps


def _make_bullets(job_title: str, job_company: str, matched: List[str]) -> List[str]:
    top = matched[:3] if matched else []
    bullets = [
        "Improved process efficiency through data analysis and clear metrics reporting.",
        "Built concise status updates and dashboards to support decision making.",
        "Partnered with stakeholders to clarify requirements and reduce cycle time.",
    ]
    if top:
        bullets.insert(
            0,
            f"Applied {', '.join(top)} to tasks relevant to the {job_title} role at {job_company}.",
        )
    return bullets[:4]


def _make_cover_letter(
    candidate_name: str,
    job_title: str,
    company: str,
    matched: List[str],
    gaps: List[str],
) -> str:
    who = candidate_name or "Candidate"
    have = ", ".join(matched[:3]) if matched else "relevant tools"
    need = ", ".join(gaps[:2]) if gaps else "the listed requirements"
    return (
        f"Dear Hiring Manager,\n\n"
        f"I am interested in the {job_title} role at {company}. My background includes practical experience with {have} "
        f"and a consistent focus on clear communication and measurable outcomes. I understand the importance of {need} "
        f"and learn quickly to meet team goals.\n\n"
        f"Thank you for your time and consideration.\nSincerely,\n{who}"
    )

def _parse_pdf_with_pre_llm(content: bytes):
    """
    Use ParsingFunctionsPreLLM to get clean text + sections + contacts from PDF.
    """
    parser = ParsingFunctionsPreLLM(path="<in-memory>")
    raw_text = parser.extract_text_from_pdf_bytes(content)
    cleaned_text = parser.clean_up_text(raw_text)
    sections = parser.define_sections(cleaned_text)
    contacts = parser.gather_contact_info_from_text(cleaned_text)

    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "sections": sections,
        "contacts": contacts,
    }
    
def _parse_plain_text_with_pre_llm(text: str):
    """
    Uses the same logic as _parse_pdf_with_pre_llm except starts from already extracted plain text (DOCX/TXT).
    """
    parser = ParsingFunctionsPreLLM(path="<in-memory>")
    cleaned = parser.clean_up_text(text or "")
    sections = parser.define_sections(cleaned)
    contacts = parser.gather_contact_info_from_text(cleaned)
    return {
        "raw_text": text or "",
        "cleaned_text": cleaned,
        "sections": sections,
        "contacts": contacts,
    }

def _get_resume_record(resume_id: int):
    """Fetch resume text, parsed sections, and parsed contacts."""
    db, cursor = get_db()
    if db:
        try:
            cursor.execute("""
                SELECT id, user_id, resume_text, parsed_sections, parsed_contacts
                FROM resumes WHERE id=%s
            """, (resume_id,))
            row = cursor.fetchone()
            if row:
                # Normalize database data into json for consistency
                def _as_json(v):
                    if v is None: return None
                    if isinstance(v, (dict, list)): return v
                    try: return json.loads(v)
                    except Exception: return None
                return {
                    "resume_id": row["id"],
                    "user_id": row.get("user_id"),
                    "text": row.get("resume_text") or "",
                    "sections": _as_json(row.get("parsed_sections")),
                    "contacts": _as_json(row.get("parsed_contacts")),
                }
        except Exception as e:
            app.logger.exception(e)

    r = MEM["resumes"].get(resume_id) or {}
    return {
        "resume_id": resume_id,
        "user_id": r.get("user_id"),
        "text": r.get("text") or "",
        "sections": r.get("parsed_sections"),
        "contacts": r.get("parsed_contacts"),
    }

    
def _null_if_blank(v):
    """
    Returns null if string entry is blank.
    """
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    return v

def _decimal_or_null(v):
    """Returns null if decimal entry is blank. Rounds to whole number."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(float(v))  
    s = str(v).strip()
    if not s:
        return None
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        return int(float(s))
    except Exception:
        return None
    
# -----------------------------
# Original routes
# -----------------------------
@app.route("/")
def home():
    return "JobHunter.ai Backend Running Successfully!"


@app.route("/db-check")
def db_check():
    db, cursor = get_db()
    if not db:
        return "Database connection not configured; using memory store."
    try:
        cursor.execute("SELECT DATABASE() AS db;")
        result = cursor.fetchone()
        return f"Connected successfully to database: {result['db']}"
    except Exception as e:
        return f"Database connection failed: {e}"


@app.route("/users")
def get_users():
    db, cursor = get_db()
    if not db:
        return jsonify({"error": "DB not configured; no users table in memory mode"})
    try:
        cursor.execute("SELECT * FROM users;")
        users = cursor.fetchall()
        return jsonify(users)
    except Exception as e:
        return jsonify({"error": str(e)})


# -----------------------------
# New routes
# -----------------------------
@app.get("/api/health")
def health():
    db, _ = get_db()
    info = {
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "use_db": bool(db),
        "env": os.getenv("FLASK_ENV", "unknown"),
    }
    return ok(info)


# POST /api/resumes  JSON {"text": "...", "meta":{"name":"...", "skills":[...], "experience":"..."}}
# or multipart 'file'
@app.post("/api/resumes")
def upload_resume():
    db, cursor = get_db()
    uid = _get_user_id()

    if request.is_json:
        body = request.get_json(force=True) or {}
        text = _normalize_ws(body.get("text", ""))
        meta = body.get("meta") or {}
        name = _normalize_ws(meta.get("name", ""))
        u_skills = meta.get("skills") or []
        experience = _normalize_ws(meta.get("experience", ""))
        if not text and not experience:
            return bad("Provide 'text' or 'meta.experience' or upload a file")

        text_to_store = text or experience

        if db:
            try:
                cursor.execute(
                    "INSERT INTO resumes (user_id, resume_text, created_at) "
                    "VALUES (%s, %s, NOW())",
                    (uid, text_to_store),
                )
                cursor.execute("SELECT LAST_INSERT_ID() AS id")
                resume_id = cursor.fetchone()["id"]
                # keep meta in memory even if DB does not have columns
                MEM["resumes"][resume_id] = {
                    "user_id": uid,
                    "text": text_to_store,
                    "file_name": None,
                    "name": name,
                    "skills": u_skills,
                    "experience": experience,
                }
                return ok({"resume_id": resume_id})
            except Exception as e:
                app.logger.exception(e)

        rid = MEM["next_resume_id"]
        MEM["next_resume_id"] += 1
        MEM["resumes"][rid] = {
            "user_id": uid,
            "text": text_to_store,
            "file_name": None,
            "name": name,
            "skills": u_skills,
            "experience": experience,
        }
        return ok({"resume_id": rid})

    # file path
    if "file" not in request.files:
        return bad("No file part. Use 'file' field for upload or send JSON with 'text'")
    file = request.files["file"]
    fname = secure_filename(file.filename or "resume.pdf")
    mime = file.mimetype or ""
    if mime not in ALLOWED_MIME:
        return bad("Only PDF, DOCX, or TXT allowed")

    content = file.read() or b""

    parsed = None
    if mime == "application/pdf":
        parsed = _parse_pdf_with_pre_llm(content)
        text = parsed["cleaned_text"]
    elif mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        result = mammoth.extract_raw_text(BytesIO(content))
        raw = (result.value or "").strip()
        parsed = _parse_plain_text_with_pre_llm(raw)
        text = parsed["cleaned_text"]
    else:  # text/plain
        raw = content.decode("utf-8", errors="ignore")
        parsed = _parse_plain_text_with_pre_llm(raw)
        text = parsed["cleaned_text"]

    meta_blob = {}
    if parsed:
        meta_blob = {
            "sections": parsed["sections"],
            "contacts": parsed["contacts"],
        }

    if db:
        try:
            cursor.execute(
                "INSERT INTO resumes (user_id, resume_text, file_name, parsed_sections, parsed_contacts, created_at) "
                "VALUES (%s, %s, %s, %s, %s, NOW())",
                (
                    uid,
                    text,
                    fname,
                    json.dumps(meta_blob.get("sections")) if meta_blob else None,
                    json.dumps(meta_blob.get("contacts")) if meta_blob else None,
                ),
            )
            cursor.execute("SELECT LAST_INSERT_ID() AS id")
            resume_id = cursor.fetchone()["id"]

            MEM["resumes"][resume_id] = {
                "user_id": uid,
                "text": text,
                "file_name": fname,
                "name": "",
                "skills": [],
                "experience": "",
                "parsed_sections": meta_blob.get("sections"),
                "parsed_contacts": meta_blob.get("contacts"),
            }
            return ok({"resume_id": resume_id})
        except Exception as e:
            app.logger.exception(e)

    # memory fallback
    rid = MEM["next_resume_id"]
    MEM["next_resume_id"] += 1
    MEM["resumes"][rid] = {
        "user_id": uid,
        "text": text,
        "file_name": fname,
        "name": "",
        "skills": [],
        "experience": "",
        "parsed_sections": meta_blob.get("sections"),
        "parsed_contacts": meta_blob.get("contacts"),
    }

    app.logger.info(f"Upload: name={fname}, mime={mime}, size={len(content)}")

    return ok({"resume_id": rid})

@app.get("/api/resumes")
def resume_get_latest_meta():
    """
    Returns the latest resume metadata for the current user.
    """
    uid = _get_user_id()
    db, cursor = get_db()

    if db:
        cursor.execute(
            """
            SELECT id, COALESCE(file_name, 'pasted-text') AS name, created_at
            FROM resumes
            WHERE (user_id <=> %s)
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (uid,),
        )
        row = cursor.fetchone()
        if not row:
            return bad("No resume", 404)
        return ok({
            "resume_id": row["id"],
            "name": row["name"],
            "uploaded_at": (row["created_at"].isoformat() if row.get("created_at") else datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
        })

    # Find most recent by insertion order.
    if not MEM["resumes"]:
        return bad("No resume", 404)
    # Prefer last created for the same user if present; else any last.
    user_items = [(rid, r) for rid, r in MEM["resumes"].items() if r.get("user_id") == uid]
    rid, r = (user_items[-1] if user_items else list(MEM["resumes"].items())[-1])
    return ok({
        "resume_id": rid,
        "name": r.get("file_name") or "pasted-text",
        "uploaded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    })


@app.put("/api/resumes/<int:rid>")
def resume_replace_existing(rid: int):
    """
    Replaces an existing resume with a new resume. Updates file info accordingly.
    """
    uid = _get_user_id()
    db, cursor = get_db()

    if "file" not in request.files:
        return bad("Missing 'file' in form-data")
    f = request.files["file"]
    if not f or not f.filename:
        return bad("Empty file")

    mime = f.mimetype or ""
    if mime not in ALLOWED_MIME:
        return bad("Only PDF, DOCX, or TXT allowed")

    content = f.read() or b""

    parsed = None
    if mime == "application/pdf":
        parsed = _parse_pdf_with_pre_llm(content)
        text = parsed["cleaned_text"]
    elif mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        result = mammoth.extract_raw_text(BytesIO(content))
        raw = (result.value or "").strip()
        parsed = _parse_plain_text_with_pre_llm(raw)
        text = parsed["cleaned_text"]
    else:  # text/plain
        raw = content.decode("utf-8", errors="ignore")
        parsed = _parse_plain_text_with_pre_llm(raw)
        text = parsed["cleaned_text"]

    safe_name = secure_filename(f.filename)

    if db:
        # Ensure the row exists and belongs to this user
        cursor.execute(
            """
            UPDATE resumes
            SET
                resume_text = %s,
                file_name = %s,
                parsed_sections = %s,
                parsed_contacts = %s,
                created_at = NOW()
            WHERE id=%s AND (user_id <=> %s)
            """,
            (
                text,
                safe_name,
                json.dumps(parsed["sections"]) if parsed else None,
                json.dumps(parsed["contacts"]) if parsed else None,
                rid,
                uid,
            ),
        )

        # Fetch fresh metadata
        cursor.execute(
            "SELECT id, COALESCE(file_name, 'pasted-text') AS name, created_at FROM resumes WHERE id=%s",
            (rid,),
        )
        fresh = cursor.fetchone()
        return ok({
            "resume_id": fresh["id"],
            "name": fresh["name"],
            "uploaded_at": fresh["created_at"].isoformat()
        })

    if rid not in MEM["resumes"] or MEM["resumes"][rid].get("user_id") != uid:
        # Accept null user match in dev if uid is None and record has None
        if not (rid in MEM["resumes"] and MEM["resumes"][rid].get("user_id") is None and uid is None):
            return bad("Not found", 404)
    MEM["resumes"][rid]["text"] = text
    MEM["resumes"][rid]["file_name"] = safe_name

    if parsed:
        MEM["resumes"][rid]["parsed_sections"] = parsed["sections"]
        MEM["resumes"][rid]["parsed_contacts"] = parsed["contacts"]

    return ok({
        "resume_id": rid,
        "name": safe_name,
        "uploaded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    })


@app.delete("/api/resumes/<int:rid>")
def resume_delete_existing(rid: int):
    """
    Delete a resume row. 204 on success.
    """
    uid = _get_user_id()
    db, cursor = get_db()

    if db:
        cursor.execute(
            "SELECT id FROM resumes WHERE id=%s AND (user_id <=> %s)",
            (rid, uid),
        )
        row = cursor.fetchone()
        if not row:
            return bad("Not found", 404)
        cursor.execute(
            "DELETE FROM resumes WHERE id=%s AND (user_id <=> %s)",
            (rid, uid),
        )
        return "", 204

    r = MEM["resumes"].get(rid)
    if not r:
        return bad("Not found", 404)
    if not (r.get("user_id") == uid or (r.get("user_id") is None and uid is None)):
        return bad("Not found", 404)
    del MEM["resumes"][rid]
    return "", 204

@app.get("/api/resumes/<int:rid>/parsed")
def resume_get_parsed(rid: int):
    uid = _get_user_id()
    db, cursor = get_db()

    if db:
        cursor.execute(
            """
            SELECT id, parsed_sections, parsed_contacts
            FROM resumes
            WHERE id=%s AND (user_id <=> %s)
            """,
            (rid, uid),
        )
        row = cursor.fetchone()
        if not row:
            return bad("Not found", 404)

        def _maybe_load(v):
            if v is None:
                return None
            if isinstance(v, (dict, list)):
                return v
            try:
                return json.loads(v)
            except Exception:
                return None

        return ok({
            "resume_id": row["id"],
            "parsed_sections": _maybe_load(row.get("parsed_sections")),
            "parsed_contacts": _maybe_load(row.get("parsed_contacts")),
        })

    # memory fallback
    r = MEM["resumes"].get(rid)
    if not r or not (r.get("user_id") == uid or (r.get("user_id") is None and uid is None)):
        return bad("Not found", 404)

    return ok({
        "resume_id": rid,
        "parsed_sections": r.get("parsed_sections"),
        "parsed_contacts": r.get("parsed_contacts"),
    })

def extract_experience_level_helper(text: str) -> str:
    """Heuristic to infer experience level from free text.

    Returns one of: 'entry', 'mid', 'senior', or 'unknown'.
    """
    if not text:
        return "unknown"

    s = text.lower()

    # entry keywords
    if re.search(
        r"\b(intern(ship)?|intern|fresher|new\s+grad|graduate|entry[- ]?level|junior)\b",
        s,
    ):
        return "entry"

    # senior keywords (include common abbreviations)
    if re.search(
        r"\b(senior|sr\.?|lead|principal|director|vp\b|vice\s+president|manager)\b",
        s,
    ):
        return "senior"

    # numeric years: map to categories
    m = re.search(r"(\d+)\s*(\+|plus)?\s*(years|yrs|y)\b", s)
    if m:
        try:
            years = int(m.group(1))
        except Exception:
            years = 0
        if years <= 1:
            return "entry"
        if 2 <= years <= 4:
            return "mid"
        if years >= 5:
            return "senior"

    return "unknown"


def _normalize_contract_type(job_raw: dict, title: str, description: str) -> str:
    """FALL BACK heuristic to:

    Normalize job type into friendly labels.
    Returns one of: 'Full-time', 'Part-time', 'Contract', 'Internship', 'Remote', ''
    """
    # Ensure job_raw is a dict-like object
    if not isinstance(job_raw, dict):
        job_raw = {}

    # Combine structured contract fields with title/description for a single search space
    parts: List[str] = []
    for k in ("contract_time", "contract_type"):
        v = job_raw.get(k)
        if v:
            parts.append(str(v))
    parts.append(str(title or ""))
    parts.append(str(description or ""))
    txt = " ".join(parts).lower()

    # Phrase-first patterns (use word boundaries to avoid accidental matches like 'part' in 'participants')
    patterns = [
        (r"\b(full[\s-]*time|fte|full\s+time|permanent)\b", "Full-time"),
        (r"\b(part[\s-]*time)\b", "Part-time"),
        (r"\b(contract|temporary)\b", "Contract"),
        (r"\b(intern(ship)?|intern)\b", "Internship"),
        (r"\bremote\b", "Remote"),
        (r"\bhybrid\b", "Hybrid"),
    ]

    for pattern, label in patterns:
        try:
            if re.search(pattern, txt, re.IGNORECASE):
                return label
        except re.error:
            # if a regex fails for any reason, skip and continue
            continue

    return ""


def _canonicalize_type_input(s: str) -> str:
    """Map various user-provided job type labels into canonical labels used by the API.

    Returns values like: 'Full-time', 'Part-time', 'Contract', 'Internship', 'Remote', 'Hybrid', or ''
    """
    if not s:
        return ""
    s2 = s.strip().lower()
    if re.search(r"full[\s_-]*time|^fulltime$|^full-time$|^fte|permanent", s2):
        return "Full-time"
    if re.search(r"part[\s_-]*time|^parttime$|^part-time$", s2):
        return "Part-time"
    if re.search(r"contract|temporary|temp|c2h|c2c|contract-to-hire|contract to hire", s2):
        return "Contract"
    if re.search(r"intern(ship)?|^intern$", s2):
        return "Internship"
    if re.search(r"\bremote\b", s2):
        return "Remote"
    if re.search(r"\bhybrid\b", s2):
        return "Hybrid"
    return s.strip().replace("_", " ").replace("-", " ").title()


def _canonicalize_experience_input(s: str) -> str:
    """Map various user-provided experience labels into canonical experience_level values.

    Returns one of: 'entry', 'mid', 'senior', or '' (empty string for unknown).
    """
    if not s:
        return ""
    t = s.strip().lower()
    if re.search(r"entry|intern|junior|fresher|new\s+grad|graduate", t):
        return "entry"
    if re.search(r"mid|associate|intermediate", t):
        return "mid"
    if re.search(r"senior|sr\.?|lead|principal|director|vp|manager", t):
        return "senior"
    return ""


# -----------------------------
# Authentication endpoints
# -----------------------------
@app.post("/api/auth/register")
def auth_register():
    data = request.get_json(force=True) or {}
    full_name = (data.get("full_name") or data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    # Basic validation
    if not full_name:
        return bad("Missing 'full_name'")
    if not email or "@" not in email:
        return bad("Provide a valid email address")
    if not password or len(password) < 8:
        return bad("Password must be at least 8 characters")

    db, cursor = get_db()

    # Check uniqueness
    if db:
        try:
            cursor.execute("SELECT id FROM users WHERE email=%s", (email,))
            if cursor.fetchone():
                return bad("Email already registered", 409)
        except Exception as e:
            app.logger.exception(e)

    else:
        MEM.setdefault("users", {})
        # case-insensitive check
        for u in MEM["users"].values():
            if (u.get("email") or "").lower() == email:
                return bad("Email already registered", 409)

    # Hash password with bcrypt
    try:
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    except Exception as e:
        app.logger.exception(e)
        return bad("Failed to hash password", 500)

    # Insert user
    if db:
        try:
            cursor.execute(
                "INSERT INTO users (full_name, email, password_hash) VALUES (%s, %s, %s)",
                (full_name, email, hashed),
            )
            cursor.execute("SELECT LAST_INSERT_ID() AS id")
            uid = cursor.fetchone()["id"]
            db.commit()
            # Optionally ensure user_profiles exists (schema trigger should create it)
            cursor.execute("SELECT id, full_name, email, role FROM users WHERE id=%s", (uid,))
            row = cursor.fetchone()
            user_obj = {"user_id": row["id"], "name": row["full_name"], "email": row["email"], "role": row.get("role")}
        except Exception as e:
            app.logger.exception(e)
            return bad("Failed to create user", 500)
    else:
        uid = MEM.get("next_user_id", 1)
        MEM["next_user_id"] = uid + 1
        MEM.setdefault("users", {})
        MEM["users"][uid] = {"id": uid, "full_name": full_name, "email": email, "password_hash": hashed, "role": "jobseeker"}
        # create profile placeholder
        MEM.setdefault("user_profiles", {})
        MEM["user_profiles"][uid] = {"user_id": uid}
        user_obj = {"user_id": uid, "name": full_name, "email": email, "role": "jobseeker"}

    # Issue JWT (7 day expiry)
    try:
        exp = datetime.now(timezone.utc) + timedelta(days=7)
        token = jwt.encode({"user_id": user_obj.get("user_id"), "exp": int(exp.timestamp())}, JWT_SECRET, algorithm=JWT_ALGO)
    except Exception as e:
        app.logger.exception(e)
        return bad("Failed to create token", 500)

    return ok({"token": token, "user": user_obj})


@app.post("/api/auth/login")
def auth_login():
    data = request.get_json(force=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return bad("Provide 'email' and 'password'")

    db, cursor = get_db()
    user_row = None
    if db:
        try:
            cursor.execute("SELECT id, full_name, email, password_hash, role FROM users WHERE email=%s", (email,))
            row = cursor.fetchone()
            if not row:
                return bad("Invalid credentials", 401)
            user_row = row
            stored = row.get("password_hash")
        except Exception as e:
            app.logger.exception(e)
            return bad("Error during authentication", 500)
    else:
        for uid, u in (MEM.get("users") or {}).items():
            if (u.get("email") or "").lower() == email:
                user_row = u
                stored = u.get("password_hash")
                break
        if not user_row:
            return bad("Invalid credentials", 401)

    # Verify password
    try:
        if not stored or not bcrypt.checkpw(password.encode('utf-8'), stored.encode('utf-8')):
            return bad("Invalid credentials", 401)
    except Exception:
        return bad("Invalid credentials", 401)

    user_obj = {"user_id": user_row.get("id") or user_row.get("user_id"), "name": user_row.get("full_name") or user_row.get("full_name"), "email": user_row.get("email"), "role": user_row.get("role")}

    try:
        exp = datetime.now(timezone.utc) + timedelta(days=7)
        token = jwt.encode({"user_id": user_obj.get("user_id"), "exp": int(exp.timestamp())}, JWT_SECRET, algorithm=JWT_ALGO)
    except Exception as e:
        app.logger.exception(e)
        return bad("Failed to create token", 500)

    return ok({"token": token, "user": user_obj})


# POST /api/jobs/search { "inputs": ["https://...", "data analyst chicago", ...] }
@app.post("/api/jobs/search")
def jobs_search():
    ## Read search inputs
    data = request.get_json(force=True) or {}
    query = (data.get("query") or "").strip()
    location = (data.get("location") or "").strip()
    # Optional filters (type and experience now provided by frontend)
    job_type_filter = (data.get("type") or "").strip()
    experience_filter = (data.get("experience") or "").strip()
    page = int(data.get("page", 1))
    # Enforce a single allowed page size for Adzuna results: 30 per user requirement
    # Ignore any client-provided value and always request 30 results per page
    results_per_page = 30
    salary_min = data.get("salaryMin")
    salary_max = data.get("salaryMax")

    if not query:
        return bad("Provide 'query' as a non-empty string")

    ## Load Adzuna credentials
    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")
    country = os.getenv("ADZUNA_COUNTRY", "us")

    if not app_id or not app_key:
        return bad("Missing Adzuna credentials")

    ## Build Adzuna request
    url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"

    # Build 'what' by appending textual qualifiers so Adzuna performs a best-effort filtered search.
    # NOTE: Do NOT append the client's `type` filter here — Appending literal labels
    # like "Full-time" to Adzuna's free-text query often over-constrains results and
    # returns zero matches. We perform deterministic server-side post-filtering on
    # our normalized `type` instead. Only append `experience` as a textual qualifier.
    # Canonicalize experience before appending to avoid over-constraining the provider
    # (send 'entry'|'mid'|'senior' tokens rather than display labels like 'Entry').
    qualifiers = []
    canon_exp = _canonicalize_experience_input(experience_filter)
    if canon_exp:
        qualifiers.append(canon_exp)

    what_query = " ".join([query] + qualifiers).strip()

    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": results_per_page,
        "what": what_query,
        "where": location,
    }
    if salary_min:
        params["salary_min"] = salary_min
    if salary_max:
        params["salary_max"] = salary_max

    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        app.logger.exception(f"Adzuna API error: {e}")
        return bad("Failed to fetch jobs from Adzuna")

    ## Database connection
    db, cursor = get_db()
    results, job_ids = [], []

    ## Iterate over each job result
    for job in data.get("results", []):
        job_id = None
        title = job.get("title")
        company = job.get("company", {}).get("display_name")
        location_name = job.get("location", {}).get("display_name")
        url_job = job.get("redirect_url")
        raw_description = job.get("description", "") or ""
        # Unescape HTML entities then strip tags
        unescaped_description = _html.unescape(raw_description)
        description = re.sub(r"<[^>]+>", "", unescaped_description)
        job_salary_min = job.get("salary_min")
        job_salary_max = job.get("salary_max")
        category = job.get("category", {}).get("label", "")
        source = "api"

        # Infer experience level from title+description
        exp_text = f"{title or ''} {description or ''}"
        experience_level = extract_experience_level_helper(exp_text)

        ## UPSERT (insert or update existing)
        if db:
            try:
                cursor.execute(
                    '''
                    INSERT INTO jobs (title, company_name, industry, description, location, salary_range, source, url)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        description = VALUES(description),
                        salary_range = VALUES(salary_range),
                        posted_at = CURRENT_TIMESTAMP
                    ''',
                    (title, company, category, description, location_name, f"{salary_min}-{salary_max}", source, url_job)
                )
                cursor.execute("SELECT job_id FROM jobs WHERE title=%s AND company_name=%s AND location=%s AND url=%s",
                               (title, company, location_name, url_job))
                job_id = cursor.fetchone()["job_id"]
                job_ids.append(job_id)
            except Exception as e:
                app.logger.warning(f"Job UPSERT failed: {e}")

        # Normalize contract/type and include raw fields so clients can rely on a consistent `type` value
        # Prefer Adzuna's structured contract fields when present. Only fall back to
        # our heuristic `_normalize_contract_type` when Adzuna provides no useful value.
        adz_contract_time = job.get("contract_time")
        adz_contract_type = job.get("contract_type")
        adz_type_raw = None
        for k in ("type", "employment_type"):
            v = job.get(k)
            if v:
                adz_type_raw = str(v).strip()
                break

        # Map common Adzuna contract_time values to our labels
        if adz_contract_time:
            act = str(adz_contract_time).lower()
            if act in {"full_time", "full-time", "full time", "permanent", "fte"}:
                job_type = "Full-time"
            elif act in {"part_time", "part-time", "part time"}:
                job_type = "Part-time"
            elif act in {"contract", "temporary", "temp"}:
                job_type = "Contract"
            elif act in {"internship", "intern"}:
                job_type = "Internship"
            else:
                # use fallback normalization on the raw value
                job_type = _normalize_contract_type(job, title, description)
        elif adz_contract_type:
            # contract_type can be values like 'permanent' or 'contract'
            act = str(adz_contract_type).lower()
            if act in {"permanent", "permanent contract", "permanent-hire"}:
                job_type = "Full-time"
            elif act in {"contract", "temporary", "temp"}:
                job_type = "Contract"
            else:
                job_type = _normalize_contract_type(job, title, description)
        elif adz_type_raw:
            # Map textual type/employment_type values
            s = adz_type_raw.lower()
            if re.search(r"(full[\s_-]*time|fte|permanent|full time|fulltime)", s):
                job_type = "Full-time"
            elif re.search(r"(part[\s_-]*time|part time|parttime)", s):
                job_type = "Part-time"
            elif re.search(r"(contract|temporary|temp|c2h|c2c|contract-to-hire|contract to hire)", s):
                job_type = "Contract"
            elif re.search(r"(intern(ship)?|intern)", s):
                job_type = "Internship"
            elif re.search(r"\bremote\b", s):
                job_type = "Remote"
            elif re.search(r"\bhybrid\b", s):
                job_type = "Hybrid"
            else:
                job_type = str(adz_type_raw).replace("_", " ").replace("-", " ").title()
        else:
            # No Adzuna-provided type info — run our heuristic on title/description
            job_type = _normalize_contract_type(job, title, description)

        ## Return clean job JSON
        results.append({
            "job_id": job_id,
            "title": title,
            "company": company,
            "location": location_name,
            "url": url_job,
            "description": (description or "")[:JOB_DESCRIPTION_MAX_CHARS],
            # Full cleaned description (not truncated). Frontend can show this in a modal/detail view.
            "full_description": (description or ""),
            "salary_min": job_salary_min,
            "salary_max": job_salary_max,
            "category": category,
            "type": job_type,
            # Provide both keys so frontend can consume either one
            "experience": experience_level,
            "experience_level": experience_level,
            "raw": job,
        })


    # Deterministic post-filtering: apply server-side filters for type and
    # experience so the returned result set strictly matches requested filters.
    # This operates on our normalized values (the `type` field and
    # `experience_level` field we set above).
    post_filtered_results = list(results)
    want_label = _canonicalize_type_input(job_type_filter)
    want_exp = _canonicalize_experience_input(experience_filter)

    # If a type was requested, keep only jobs with the normalized type
    if want_label:
        post_filtered_results = [r for r in post_filtered_results if (r.get("type") or "") == want_label]

    # If an experience was requested, keep only jobs with matching experience_level
    if want_exp:
        post_filtered_results = [r for r in post_filtered_results if (r.get("experience_level") or "").lower() == want_exp]

    filtered_total = len(post_filtered_results)

    # If the client requested qualifiers (type/experience) and Adzuna returned zero
    # results for that qualified query, do a best-effort fallback: re-query without
    # the qualifiers to obtain an unfiltered count so the UI can show an alternative.
    unfiltered_total = None
    filter_applied_but_no_results = False
    if (job_type_filter or experience_filter) and (filtered_total == 0):
        try:
            # Build params without qualifiers (just the base query)
            params_no_qual = params.copy()
            params_no_qual["what"] = query
            res2 = requests.get(url, params=params_no_qual, timeout=10)
            res2.raise_for_status()
            data2 = res2.json()
            unfiltered_total = data2.get("count", 0)
            filter_applied_but_no_results = True if (unfiltered_total and unfiltered_total > 0) else False
        except Exception as e:
            app.logger.info(f"Fallback unfiltered Adzuna query failed: {e}")
            unfiltered_total = None

    # Adzuna's provider count for the original query we issued (best-effort)
    adzuna_count = data.get("count", len(results)) if isinstance(data, dict) else len(results)

    # Decide what to report as the authoritative total_results:
    # - If the client did not request any filters (type/experience), prefer
    #   Adzuna's provider count (adzuna_count) since it reflects the total
    #   upstream matching jobs across pages.
    # - If the client requested filters and we applied deterministic
    #   post-filtering, report the filtered_total (which currently reflects
    #   matches within the fetched page). Computing the exact filtered total
    #   across all pages would require additional provider queries and is
    #   intentionally left out for performance; instead we also include
    #   `provider_total_results` for transparency.
    if not (job_type_filter or experience_filter):
        total_results = int(adzuna_count or 0)
    else:
        total_results = filtered_total

    total_pages = (total_results + results_per_page - 1) // results_per_page

    # If the client requested a job type, return the post-filtered results
    # (deterministic). Otherwise, return the raw results from Adzuna.
    results_to_return = post_filtered_results if want_label or want_exp else results

    ## Return API response
    return ok({
        "query": query,
        "location": location,
        "type": job_type_filter,
        "experience": experience_filter,
        "page": page,
        "results_per_page": results_per_page,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "total_results": total_results,
        "total_pages": total_pages,
        # Surface the Adzuna provider's raw count so the UI can explain when
        # our deterministic post-filtering reduced the returned set.
        "provider_total_results": adzuna_count,
        # When filters were applied but produced no filtered results, provide unfiltered count if available
        "filtered_total_results": filtered_total,
        "unfiltered_total_results": unfiltered_total,
        "filter_applied_but_no_results": filter_applied_but_no_results,
        "count": len(results_to_return),
        "persisted": len(job_ids),
        "job_ids": job_ids,
        "results": results_to_return,
    })


# POST /api/recommend { "resume_id": int, "job_ids": [int] }
@app.post("/api/recommend")
def recommend():
    data = request.get_json(force=True) or {}
    resume_id = data.get("resume_id")
    job_ids = data.get("job_ids", [])

    if not resume_id:
        return bad("Missing 'resume_id'")
    if not job_ids:
        return bad("Provide non-empty 'job_ids' array")

    # fetch resume text (from DB if available, else memory)
    db, cursor = get_db()
    resume_text = ""
    candidate_name = ""
    user_listed_skills: List[str] = []

    if db:
        try:
            cursor.execute(
                "SELECT resume_text FROM resumes WHERE id=%s", (resume_id,)
            )
            row = cursor.fetchone()
            if row:
                resume_text = row.get("resume_text") or ""
        except Exception as e:
            app.logger.exception(e)

    if not resume_text:
        r = MEM["resumes"].get(resume_id, {})
        resume_text = r.get("text", "")
        candidate_name = r.get("name", "") or ""
        user_listed_skills = r.get("skills", []) or []

    if not resume_text:
        return bad("Resume not found")

    derived = _extract_resume_skills(resume_text, user_listed_skills)

    results = []
    for jid in job_ids:
        job = MEM["jobs"].get(jid)
        if not job:
            continue
        score, gaps = _match_score(resume_text, job["skills"])
        matched = [s for s in job["skills"] if s.lower() in resume_text.lower()] or derived[:3]
        bullets = _make_bullets(job["title"], job["company"], matched)
        cover = _make_cover_letter(
            candidate_name, job["title"], job["company"], matched, gaps
        )
        results.append(
            {
                "job_id": jid,
                "title": job["title"],
                "company": job["company"],
                "location": job["location"],
                "score": score,
                "gaps": gaps[:3],
                "resume_bullets": bullets,
                "cover_letter": cover,
                "matched_skills": matched,
                "derived_resume_skills": derived[:8],
            }
        )

    return ok({"results": results})


# POST /api/jobs/recommend { "job_title": "...", "skills": [...], "location": "..." }
@app.post("/api/jobs/recommend")
def job_recommend_mock():
    data = request.get_json(force=True) or {}
    job_title = data.get("job_title")
    skills = data.get("skills", [])
    location = data.get("location")

    # Validate required fields
    if not job_title or not skills or not location:
        return bad("Missing required field(s): job_title, skills, and location are required")

    # Mock response
    mock_score = 85 if "data analysis" in [s.lower() for s in skills] else 70
    return ok({
        "message": "Job recommendation generated successfully",
        "input": data,
        "mock_score": mock_score,
    })


@app.get('/api/jobs/<int:job_id>')
def get_job(job_id: int):
    """
    Return job details by job_id. Tries DB first, otherwise falls back to in-memory store.
    """
    db, cursor = get_db()
    if db:
        try:
            cursor.execute(
                "SELECT job_id, title, company_name AS company, description, location, url, salary_range, source, type FROM jobs WHERE job_id=%s",
                (job_id,),
            )
            row = cursor.fetchone()
            if row:
                # Normalize to the frontend-friendly shape
                return ok({
                    "job_id": row.get("job_id"),
                    "title": row.get("title"),
                    "company": row.get("company"),
                    "location": row.get("location"),
                    "description": (row.get("description") or "")[:JOB_DESCRIPTION_MAX_CHARS],
                    "full_description": row.get("description") or "",
                    "url": row.get("url"),
                    "salary_range": row.get("salary_range"),
                    "type": row.get("type") or "",
                    "source": row.get("source"),
                })
        except Exception as e:
            app.logger.exception(e)

    # memory fallback
    j = MEM.get("jobs", {}).get(job_id)
    if j:
        return ok({
            "job_id": job_id,
            "title": j.get("title"),
            "company": j.get("company"),
            "location": j.get("location"),
            "description": (j.get("description") or "")[:JOB_DESCRIPTION_MAX_CHARS],
            "full_description": j.get("full_description") or j.get("description") or "",
            "url": j.get("url"),
            "salary_range": j.get("salary_range"),
            "type": j.get("type") or "",
            "source": j.get("source") or "",
        })

    return bad("Not found", 404)


# -----------------------------
# Chat endpoint for landing page
# -----------------------------
@app.post("/api/chat")
def chat():
    """Simple chat endpoint for the landing-page assistant."""
    if not GEMINI_API_KEY:
        return bad("Gemini API key is not configured on the server")

    body = request.get_json(force=True) or {}
    messages = body.get("messages") or []
    user_text = (body.get("message") or "").strip()

    if not user_text:
        return bad("Missing 'message'")

    # Build prompt history
    history = []
    for m in messages[-10:]:
        role = m.get("role", "user")
        content = m.get("content", "")
        history.append({"role": role, "parts": [content]})

    history.append({"role": "user", "parts": [user_text]})

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(history)
        reply = (response.text or "").strip()
    except Exception as e:
        app.logger.exception(f"Gemini chat error: {e}")
        return bad("Chat service failed")

    return ok({"reply": reply})

## User Profile Endpoints (GET / PUT)
def _get_current_user_id():
    """Helper to extract the current user ID from headers (temporary mock auth).

    NOTE: legacy helper — routes should prefer `_get_user_id()` which supports
    Authorization: Bearer <token> (JWT) and falls back to X-User-Id. We keep this
    function for backward compatibility but prefer `_get_user_id()` in new code.
    """
    uid = request.headers.get("X-User-Id")
    if not uid:
        return None
    try:
        return int(uid)
    except ValueError:
        return None


@app.get("/api/users/me")
def get_user_profile():
    """Fetch current user's profile data (names mapped for UI).

    This endpoint requires authentication. Prefer JWT via Authorization: Bearer
    header — `_get_user_id()` will extract and verify it. Fall back to
    X-User-Id for local/dev convenience.
    """
    user_id = _get_user_id()
    if not user_id:
        return bad("Unauthorized", 401)

    db, cursor = get_db()
    if not db:
        return bad("Database not configured", 500)

    try:
        cursor.execute("""
            SELECT 
                u.id AS user_id,
                u.full_name,
                u.title,
                u.email,
                p.location,
                p.phone,
                p.experience_level,
                p.job_preferences,
                p.desired_salary
            FROM users u
            LEFT JOIN user_profiles p ON u.id = p.user_id
            WHERE u.id = %s
        """, (user_id,))
        row = cursor.fetchone()
        if not row:
            return bad("User not found", 404)

        # Use JSON if needed
        prefs = row.get("job_preferences")
        if isinstance(prefs, str):
            try:
                prefs = json.loads(prefs) or {}
            except Exception:
                prefs = {}
        elif not isinstance(prefs, dict):
            prefs = {}

        payload = {
            "user_id": row["user_id"],
            "name": row.get("full_name") or "",
            "title": row.get("title") or "",
            "email": row.get("email") or "",
            "location": row.get("location") or "",
            "phone": row.get("phone") or "",
            "desired_salary": str(row.get("desired_salary") or ""),
            "job_preferences": prefs,
            "pref_locations": ", ".join(prefs.get("locations", [])) if isinstance(prefs.get("locations"), list) else "",
            "pref_type": prefs.get("type", "") or "",
            "pref_salary": str(row.get("desired_salary") or ""),
        }
        return ok(payload)
    except Exception as e:
        app.logger.exception(e)
        return bad("Error fetching user profile", 500)



@app.put("/api/users/me/profile")
def update_user_profile():
    """Update the current user's profile fields (keys aligned with UI)."""
    user_id = _get_user_id()
    if not user_id:
        return bad("Unauthorized", 401)

    data = request.get_json(force=True) or {}

    name = _null_if_blank(data.get("name"))
    title = _null_if_blank(data.get("title"))
    email = _null_if_blank(data.get("email"))
    location = _null_if_blank(data.get("location"))
    phone = _null_if_blank(data.get("phone"))

    desired_salary = _decimal_or_null(data.get("desired_salary")) \
        if data.get("desired_salary") is not None else _decimal_or_null(data.get("pref_salary"))

    job_prefs = data.get("job_preferences")
    if not isinstance(job_prefs, dict):
        pref_locs_raw = _null_if_blank(data.get("pref_location"))  
        locations = []
        if pref_locs_raw:
            locations = [p.strip() for p in pref_locs_raw.split(",") if p.strip()]
        pref_type = _null_if_blank(data.get("pref_type"))
        job_prefs = {
            "locations": locations,
            "type": pref_type or ""
        }

    db, cursor = get_db()
    if not db:
        return bad("Database not configured", 500)

    try:
        # Update user info
        if any([name is not None, title is not None, email is not None]):
            cursor.execute("""
                UPDATE users
                SET 
                    full_name = COALESCE(%s, full_name),
                    title      = COALESCE(%s, title),
                    email      = COALESCE(%s, email)
                WHERE id = %s
            """, (name, title, email, user_id))

        # Update profile row
        cursor.execute("""
            UPDATE user_profiles
            SET 
                location       = COALESCE(%s, location),
                phone          = COALESCE(%s, phone),
                desired_salary = COALESCE(%s, desired_salary),
                job_preferences = COALESCE(%s, job_preferences)
            WHERE user_id = %s
        """, (location, phone, desired_salary, json.dumps(job_prefs), user_id))


        db.commit()
        return ok({"message": "Profile updated successfully"})
    except Exception as e:
        db.rollback()
        app.logger.exception(e)
        return bad("Error updating profile", 500)

# Saved Jobs endpoints
@app.get("/api/users/me/saved-jobs")
def get_saved_jobs():
    user_id = _get_user_id()
    if not user_id:
        return bad("Unauthorized", 401)

    db, cursor = get_db()
    if not db:
        return ok([])

    cursor.execute("""
        SELECT sj.saved_job_id, sj.date_saved, sj.notes,
               j.job_id, j.title, j.company_name, j.location,
               j.industry, j.salary_range, j.url, j.source, j.description
        FROM saved_jobs sj
        JOIN jobs j ON sj.job_id = j.job_id
        WHERE sj.user_id = %s
        ORDER BY sj.date_saved DESC
    """, (user_id,))
    rows = cursor.fetchall()
    return ok(rows)


@app.post("/api/users/me/saved-jobs")
def save_job():
    user_id = _get_user_id()
    if not user_id:
        return bad("Unauthorized", 401)

    data = request.get_json(force=True)
    job_id = data.get("job_id")
    notes = data.get("notes")

    if not job_id:
        return bad("Missing job_id")

    db, cursor = get_db()
    if not db:
        return bad("Database not configured", 500)

    try:
        cursor.execute("""
            INSERT INTO saved_jobs (user_id, job_id, notes)
            VALUES (%s, %s, %s)
        """, (user_id, job_id, notes))
        db.commit()
        return ok({"message": "Job saved"})
    except mysql.connector.IntegrityError:
        return bad("Job already saved", 409)


@app.delete("/api/users/me/saved-jobs/<int:job_id>")
def delete_saved_job(job_id):
    user_id = _get_user_id()
    if not user_id:
        return bad("Unauthorized", 401)

    db, cursor = get_db()
    if not db:
        return bad("Database not configured", 500)

    cursor.execute("""
        DELETE FROM saved_jobs
        WHERE user_id=%s AND job_id=%s
    """, (user_id, job_id))

    db.commit()

    return ok({"message": "Removed job"})

## Applied Jobs Endpoints
@app.post("/api/users/me/applied-jobs")
def apply_to_job():
    user_id = _get_user_id()
    if not user_id:
        return bad("Unauthorized", 401)

    data = request.get_json(force=True)
    job_id = data.get("job_id")

    if not job_id:
        return bad("Missing job_id")

    db, cursor = get_db()

    try:
        cursor.execute("""
            INSERT INTO applied_jobs (user_id, job_id)
            VALUES (%s, %s)
        """, (user_id, job_id))

        db.commit()
        return ok({"message": "Job marked as applied"})
    except mysql.connector.IntegrityError:
        return bad("You already applied to this job", 409)

@app.get("/api/users/me/applied-jobs")
def get_applied_jobs():
    user_id = _get_user_id()
    if not user_id:
        return bad("Unauthorized", 401)

    db, cursor = get_db()

    cursor.execute("""
        SELECT aj.applied_id, aj.applied_at,
            j.job_id, j.title, j.company_name, j.location,
            j.salary_range, j.url, j.source
        FROM applied_jobs aj
        JOIN jobs j ON aj.job_id = j.job_id
        WHERE aj.user_id = %s
        ORDER BY aj.applied_at DESC
    """, (user_id,))

    return ok(cursor.fetchall())


@app.post("/api/cover_letter")
def generate_cover_letter_api():
    """
    Generates a cover letter according to a resume and job listing input.
    """
    body = request.get_json(force=True) or {}
    persist = body.get("persist", True)

    job_id = body.get("job_id")
    job_obj = body.get("job") or {} #atleast job_id or job_obj should exist
    resume_id = body.get("resume_id")
    match_score = body.get("match_score")
    candidate_name = (body.get("candidate_name") or "").strip()

    contacts = None
    sections = None
    user_id_for_rec = _get_user_id()

    # Pulls job data from payload 'job' as 1st priority and fall back on data from database
    title = job_obj.get("title")
    company = job_obj.get("company") or job_obj.get("company_name")
    description = job_obj.get("full_description") or job_obj.get("description")
    location = job_obj.get("location")
    url = job_obj.get("url")

    if (not title or not company or not description) and job_id:
        db, cursor = get_db()
        if db:
            try:
                cursor.execute("""
                  SELECT job_id, title, company_name, description, location, url
                  FROM jobs WHERE job_id=%s
                """, (job_id,))
                row = cursor.fetchone()
                if row:
                    title = title or row.get("title")
                    company = company or row.get("company_name")
                    description = description or row.get("description")
                    location = location or row.get("location")
                    url = url or row.get("url")
            except Exception as e:
                app.logger.exception(e)
        # fall back to in-memory store
        if not description:
            j = MEM["jobs"].get(job_id)
            if j:
                title = title or j.get("title")
                company = company or j.get("company")
                description = description or j.get("description") or j.get("full_description")
                location = location or j.get("location")
                url = url or j.get("url")

    if resume_id:
        r = _get_resume_record(resume_id)
        resume_text = r.get("text") or ""
        contacts = r.get("contacts")
        sections = r.get("sections")
        if r.get("user_id") is not None:
            user_id_for_rec = r.get("user_id")

    # Generate the cover letter
    use_gemini = bool(GEMINI_API_KEY)
    cover_letter_text = None
    resume_bullets = []

    if use_gemini:
        try:
            generator = CoverLetterGenerator()
            cover_letter_text = generator.generate_cover_letter(
                contacts=contacts,
                sections=sections,
                tone="professional",
                job_title=title,
                company=company,
                job_description=description,
                job_board=(job_obj.get("job_board") or None),
            )
        except Exception as e:
            app.logger.exception(f"Gemini generation failed, falling back: {e}")

    # Optionally persist to job_recommendations
    if persist:
        db, cursor = get_db()
        if db and user_id_for_rec and job_id:
            try:
                cursor.execute("""
                    INSERT INTO job_recommendations
                      (user_id, job_id, match_score, generated_resume, generated_cover_letter, recommended_at)
                    VALUES
                      (%s, %s, %s, %s, %s, NOW())
                """, (
                    user_id_for_rec,
                    int(job_id),
                    float(match_score) if match_score is not None else None,
                    None,  # we don't generate resumes here, it happens in another function
                    cover_letter_text,
                ))
                db.commit()
            except Exception as e:
                app.logger.exception(f"Failed to persist job_recommendation: {e}")
        else:
            # memory fallback
            MEM.setdefault("job_recommendations", []).append({
                "user_id": user_id_for_rec,
                "job_id": job_id,
                "match_score": match_score,
                "generated_cover_letter": cover_letter_text,
                "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            })

    return ok({
        "cover_letter": cover_letter_text,
        "resume_bullets": resume_bullets
    })


@app.post("/api/ai/cover-letter")
def api_ai_cover_letter():
    """
    Backwards-compatible wrapper that exposes the same behavior under
    `/api/ai/cover-letter` — frontends may call this path.
    """
    # Delegate to the existing implementation which reads from request.json
    return generate_cover_letter_api()

@app.route("/api/ml/clean-resume", methods=["POST"])
@require_auth
def ml_clean_resume():
    data = request.json or {}

    text = data.get("text")
    if not text:
        return jsonify({"error": "Missing 'text'"}), 400

    try:
        parser = ParsingFunctionsPreLLM(None)   
        cleaned = parser.clean_up_text(text)

        return jsonify({"resume_text": cleaned}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ml/cover-letter", methods=["POST"])
@require_auth
def ml_cover_letter():
    data = request.json or {}

    resume = data.get("resume")
    job_desc = data.get("job_description")

    if not resume or not job_desc:
        return jsonify({"error": "resume and job_description are required"}), 400

    try:
        letter = generate_cover_letter(resume, job_desc)
        return jsonify({"cover_letter": letter}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ml/gemini", methods=["POST"])
@require_auth
def ml_gemini():
    prompt = request.json.get("prompt")

    if not prompt:
        return jsonify({"error": "Missing 'prompt'"}), 400

    try:
        result = GeminiClient(prompt)
        return jsonify({"response": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

from werkzeug.utils import secure_filename

@app.route("/api/ml/upload-resume", methods=["POST"])
@require_auth
def ml_upload_resume():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    filepath = f"/tmp/{filename}"
    file.save(filepath)

    try:
        with open(filepath, "r", errors="ignore") as f:
            text = f.read()

        parser = ParsingFunctionsPreLLM(None)   
        cleaned = parser.clean_up_text(text)

        return jsonify({"resume_text": cleaned}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ml/job-match", methods=["POST"])
@require_auth
def ml_job_match():
    data = request.json or {}

    resume = data.get("resume_text")
    jobs = data.get("job_list")

    if not resume or not jobs:
        return jsonify({"error": "resume_text and job_list required"}), 400

    try:
        matches = match_jobs(resume, jobs)
        return jsonify({"matches": matches}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)