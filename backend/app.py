# backend/app.py
from __future__ import annotations
import os, re, json, random, string
import requests
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Tuple
import base64
import bcrypt
import jwt

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml.pre_llm_filter_functions import ParsingFunctionsPreLLM
from ml.cover_letter_generator import CoverLetterGenerator

from pathlib import Path
import mammoth
import pdfplumber
from io import BytesIO
import html as _html

from flask import Flask, request, jsonify, make_response, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

# MySQL
import mysql.connector
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException

# Gemini
import google.generativeai as genai

# ML Microservice configuration
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8002").rstrip("/")
ML_TIMEOUT = int(os.getenv("ML_TIMEOUT", "20"))

def ml_post(path: str, payload: dict):
    """Send POST request to ML microservice."""
    url = f"{ML_SERVICE_URL}/{path.lstrip('/')}"
    try:
        r = requests.post(url, json=payload, timeout=ML_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        app.logger.error(f"[ML ERROR] {e}")
        raise

# Environment Loading and Flask App Setup
load_dotenv()
app = Flask(__name__)

# Rate Limiter (security)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

# General App Config
JOB_DESCRIPTION_MAX_CHARS = int(os.getenv("JOB_DESCRIPTION_MAX_CHARS", "2000"))

# Logging: External API Credentials
if os.getenv("ADZUNA_APP_ID") and os.getenv("ADZUNA_APP_KEY"):
    app.logger.info("Adzuna credentials found in environment")
else:
    app.logger.warning(
        "Adzuna credentials missing. Set ADZUNA_APP_ID and ADZUNA_APP_KEY"
    )

# CORS Configuration
_allow = os.getenv("CORS_ALLOW_ORIGINS", "")
# Convert comma separated list into python list
origins = [o.strip() for o in _allow.split(",") if o.strip()]

# Default for dev
if not origins:
    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

CORS(
    app,
    resources={r"/api/*": {"origins": origins}},
    supports_credentials=True,
    expose_headers=["Content-Type", "Authorization"],
)

# JWT Utilities
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    # Fallback for class and dev so the app still boots
    JWT_SECRET = "dev-insecure-jwt-secret-change-me"
    app.logger.warning(
        "JWT_SECRET not set in environment; using insecure development default. "
        "Set JWT_SECRET in Render or your environment for production."
    )

JWT_ALGO = "HS256"
JWT_EXPIRE_MINUTES = 30

# Global Error Handlers
@app.errorhandler(413)
def too_large(e):
    return bad("File too large (max 5MB)", 413)

@app.errorhandler(Exception)
def handle_uncaught(e):
    if isinstance(e, HTTPException):
        return e
    app.logger.exception("Unhandled exception", exc_info=e)
    return bad("Server error", 500)

# -----------------------------
# Gemini config
# -----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Feature flags
USE_FIELD_AWARE_MATCHER = str(os.getenv("USE_FIELD_AWARE_MATCHER", "false") or "").lower() in ("1", "true", "yes")
USE_LOOSE_SCORING = str(os.getenv("USE_LOOSE_SCORING", "true") or "").lower() in ("1", "true", "yes")
SCORE_LOOSENESS = float(os.getenv("SCORE_LOOSENESS", "1.55"))
FIELD_MATCH_BOOST = float(os.getenv("FIELD_MATCH_BOOST", ".15"))
FIELD_MISMATCH_PENALTY = float(os.getenv("FIELD_MISMATCH_PENALTY", "0.4"))
BASE_SKILL_WEIGHT = float(os.getenv("BASE_SKILL_WEIGHT", "0.4"))

FAVOR_FIELDS = [f.strip().lower() for f in os.getenv("FAVOR_FIELDS", "").split(",") if f.strip()]
FAVOR_FIELD_BOOST = float(os.getenv("FAVOR_FIELD_BOOST", "0.45"))
TITLE_MATCH_BOOST = float(os.getenv("TITLE_MATCH_BOOST", "0.1"))
TITLE_SPREAD_PER_SCORE = float(os.getenv("TITLE_SPREAD_PER_SCORE", "0.05"))

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    app.logger.info(f"Gemini model set to {GEMINI_MODEL}")
else:
    app.logger.warning("GEMINI_API_KEY not set; /api/chat will return an error")

# -----------------------------
# DB helpers (uses thread-local Flask 'g' for connection safety)
# -----------------------------
USE_DB = all(os.getenv(k) for k in ["DB_HOST", "DB_USER", "DB_NAME"])

def get_db():
    """
    Prefer PyMySQL connection when available.
    Fall back to mysql-connector-python if PyMySQL is not installed or fails.
    Returns (db_conn, cursor) or (None, None) on failure.
    """
    if not USE_DB:
        return None, None

    # Reuse thread local connection if possible
    if hasattr(g, "_db") and g._db is not None:
        try:
            if hasattr(g._db, "ping"):
                try:
                    g._db.ping(reconnect=True)
                except TypeError:
                    g._db.ping()
            elif hasattr(g._db, "is_connected"):
                if not g._db.is_connected():
                    raise Exception("stale connection")
            return g._db, g._cursor
        except Exception as e:
            app.logger.warning(f"DB connection lost, reconnecting: {e}")
            try:
                g._db.close()
            except Exception:
                pass
            g._db = None
            g._cursor = None

    # Try PyMySQL first
    try:
        import pymysql
        import pymysql.cursors

        g._db = pymysql.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            db=os.getenv("DB_NAME"),
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
        )
        g._cursor = g._db.cursor()
        app.logger.info("Connected to MySQL via PyMySQL (thread local)")
        return g._db, g._cursor
    except Exception as e_py:
        app.logger.info(f"PyMySQL connect failed (will try mysql-connector): {e_py}")

    # Fallback to mysql-connector
    try:
        g._db = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            autocommit=True,
            use_pure=True,
        )
        g._cursor = g._db.cursor(dictionary=True)
        app.logger.info("Connected to MySQL via mysql-connector-python (thread local)")
        return g._db, g._cursor
    except Exception as e:
        app.logger.warning(f"MySQL unavailable, using memory store. Error: {e}")
        return None, None

@app.teardown_appcontext
def close_db(exception=None):
    """Close the database connection at the end of each request."""
    db = g.pop("_db", None)
    if db is not None:
        try:
            db.close()
        except Exception:
            pass

# -----------------------------
# Resume Upload Helpers
# -----------------------------
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB

ALLOWED_MIME = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
}

def _get_user_id():
    """
    Grab user id for the authenticated user.
    """
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
    "resumes": {},
    "jobs": {},
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
    "java", "javascript", "react", "reactjs", "node", "nodejs", "api", "rest", "fastapi", "flask",
    "dashboards", "kpi", "etl", "data pipeline", "airflow", "docker", "kubernetes", "k8s",
    "git", "github", "gitlab", "jira", "confluence", "experimentation", "a b testing", "a/b testing", "statistics",
    "forecast", "supply chain", "sap", "ibp", "ml", "machine learning", "genai",
    "tensorflow", "pytorch", "scikit-learn", "sklearn", "keras",
    "spark", "hadoop", "hive", "bigquery", "redshift", "dbt",
    "aws", "azure", "gcp", "terraform", "ansible", "helm",
    "docker-compose", "prometheus", "grafana",
    "typescript", "graphql", "webpack", "vite", "nextjs", "gatsby",
    "html", "css", "sass", "tailwind", "bootstrap",
    "c++", "c#", "golang", "go",
    "microservices", "ci/cd", "restapi", "restful",
    "tableau", "powerbi", "lookml", "metabase",
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

# -----------------------------
# Simple, field-aware matching utilities
# -----------------------------
import importlib.util
_FIELD_KEYWORDS = {}
FIELD_PROFILES = {}

_profiles_json = os.path.join(os.path.dirname(__file__), "field_profiles.json")
_profiles_path = os.path.join(os.path.dirname(__file__), "field_profiles.py")

_loaded_profiles = False

if os.path.exists(_profiles_path):
    try:
        spec = importlib.util.spec_from_file_location("backend_field_profiles", _profiles_path)
        _fp_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_fp_mod)

        if hasattr(_fp_mod, "to_dict"):
            try:
                _data = _fp_mod.to_dict()
                _FIELD_KEYWORDS = _data.get("field_keywords", {}) or {}
                FIELD_PROFILES = _data.get("profiles", {}) or {}
                _BASE_SKILLS = set([s.lower() for s in (_data.get("base_skills") or [])])
            except Exception:
                _FIELD_KEYWORDS = getattr(_fp_mod, "_FIELD_KEYWORDS", {}) or getattr(_fp_mod, "field_keywords", {})
                FIELD_PROFILES = getattr(_fp_mod, "FIELD_PROFILES", {}) or getattr(_fp_mod, "profiles", {})
                _BASE_SKILLS = set([s.lower() for s in (getattr(_fp_mod, "BASE_SKILLS", None) or getattr(_fp_mod, "_BASE_SKILLS", []) or [])])
        else:
            _FIELD_KEYWORDS = getattr(_fp_mod, "_FIELD_KEYWORDS", {}) or getattr(_fp_mod, "field_keywords", {})
            FIELD_PROFILES = getattr(_fp_mod, "FIELD_PROFILES", {}) or getattr(_fp_mod, "profiles", {})
            _BASE_SKILLS = set([s.lower() for s in (getattr(_fp_mod, "BASE_SKILLS", None) or getattr(_fp_mod, "_BASE_SKILLS", []) or [])])

        _loaded_profiles = True
        app.logger.info(f"Loaded field profiles from module: {_profiles_path}")
    except Exception as e:
        app.logger.warning(f"Failed to load field profiles from module {_profiles_path}: {e}")

if not _loaded_profiles and os.path.exists(_profiles_json):
    try:
        with open(_profiles_json, "r", encoding="utf-8") as fh:
            _data = json.load(fh)
        _FIELD_KEYWORDS = _data.get("field_keywords", {}) or {}
        FIELD_PROFILES = _data.get("profiles", {}) or {}
        _BASE_SKILLS = set([s.lower() for s in (_data.get("base_skills") or [])])
        _loaded_profiles = True
        app.logger.info(f"Loaded field profiles from JSON: {_profiles_json}")
    except Exception as e:
        app.logger.warning(f"Failed to load profiles from JSON {_profiles_json}: {e}")

_STOPWORDS = {
    "the", "and", "a", "an", "of", "in", "on", "for", "with", "to", "from", "by", "at", "as", "is", "are",
    "be", "this", "that", "it", "its", "our", "you", "we", "your", "i",
}

def _tokenize_and_map(s: str, synonyms: dict | None = None) -> List[str]:
    if not s:
        return []
    synonyms = synonyms or {}
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    toks = [t for t in s.split() if t and t not in _STOPWORDS and len(t) > 1]
    mapped = [synonyms.get(t, t) for t in toks]
    return mapped

def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = sa.intersection(sb)
    union = sa.union(sb)
    return len(inter) / (len(union) or 1)

def _skill_overlap(resume_tokens: List[str], job_skills: List[str], synonyms: dict | None = None) -> float:
    if not job_skills:
        return 0.0
    synonyms = synonyms or {}
    skill_tokens = []
    for s in job_skills:
        skill_tokens.extend(_tokenize_and_map(s, synonyms))
    if not skill_tokens:
        return 0.0
    found = 0
    for skill in set(skill_tokens):
        if skill in resume_tokens:
            found += 1
        else:
            if any(skill in rt for rt in resume_tokens):
                found += 1
    return found / len(set(skill_tokens))

def _detect_field(job: dict) -> str | None:
    explicit = (job.get("field") or job.get("category") or "").strip().lower()
    if explicit and explicit in FIELD_PROFILES:
        return explicit
    txt = " ".join(filter(None, [job.get("title", ""), job.get("description", "")])).lower()
    for fld, kws in _FIELD_KEYWORDS.items():
        for kw in kws:
            if kw in txt:
                return fld
    return None

def _simple_match_score(resume_text: str, job: dict, use_loose: bool = False) -> Tuple[int, List[str]]:
    field = _detect_field(job) or "data"
    profile = FIELD_PROFILES.get(field, {})
    weights = profile.get("weights", {"skill": 0.6, "title": 0.25, "desc": 0.15})
    synonyms = profile.get("synonyms", {})
    priority = profile.get("priority_skills", [])

    resume_tokens = _tokenize_and_map(resume_text or "", synonyms)
    title_tokens = _tokenize_and_map(job.get("title", ""), synonyms)
    desc_tokens = _tokenize_and_map(job.get("description") or job.get("full_description") or "", synonyms)
    job_skills = job.get("skills") or priority

    skill_overlap = _skill_overlap(resume_tokens, job_skills, synonyms)
    try:
        base_overlap = 0.0
        if isinstance(_BASE_SKILLS, (set, list)) and len(_BASE_SKILLS) > 0:
            base_overlap = _skill_overlap(resume_tokens, list(_BASE_SKILLS), synonyms)
        skill_overlap = min(1.0, (skill_overlap + (BASE_SKILL_WEIGHT * base_overlap)))
    except Exception:
        pass
    title_sim = _jaccard(resume_tokens, title_tokens)
    desc_sim = _jaccard(resume_tokens, desc_tokens)

    if not desc_tokens:
        total = weights.get("skill", 0.6) + weights.get("title", 0.25)
        w_skill = weights.get("skill", 0.6) / (total or 1)
        w_title = weights.get("title", 0.25) / (total or 1)
        w_desc = 0.0
    else:
        w_skill = weights.get("skill", 0.6)
        w_title = weights.get("title", 0.25)
        w_desc = weights.get("desc", 0.15)

    combined = (skill_overlap * w_skill) + (title_sim * w_title) + (desc_sim * w_desc)

    try:
        resume_field = _detect_field({"title": resume_text or "", "description": resume_text or ""})
    except Exception:
        resume_field = None
    if resume_field and field and resume_field == field:
        combined = min(1.0, combined + FIELD_MATCH_BOOST)
    elif resume_field and field and resume_field != field:
        combined = combined * FIELD_MISMATCH_PENALTY

    try:
        if field and field.lower() in FAVOR_FIELDS:
            fk = []
            try:
                fk = list(_FIELD_KEYWORDS.get(field, []) or [])
            except Exception:
                fk = []
            cand_keywords = fk + list(priority or [])

            hits = 0
            text_l = (resume_text or "").lower()
            for kw in cand_keywords:
                if not kw:
                    continue
                k = str(kw).lower()
                if k in text_l:
                    hits += 1
                    if hits >= 5:
                        break

            per_hit_bonus = float(os.getenv("FAVOR_FIELD_PER_HIT", "0.02"))
            combined = min(1.0, combined + FAVOR_FIELD_BOOST + (per_hit_bonus * hits))

            spread_multiplier = 1.0 + (0.06 * min(hits, 5))
        else:
            spread_multiplier = 1.0
    except Exception:
        spread_multiplier = 1.0

    multiplier = 59
    min_allowed = 35
    base = 40
    if use_loose:
        loos = SCORE_LOOSENESS if SCORE_LOOSENESS > 0 else 1.0
        multiplier = int(round(85 * loos))
        min_allowed = 25
        base = 35
        combined = min(1.0, combined + (0.08 * loos))

    gamma = 1.5
    raw = int(round((combined ** gamma) * multiplier * (spread_multiplier if "spread_multiplier" in locals() else 1.0)))
    score = max(min_allowed, min(99, base + raw))

    gaps = []
    for s in (job.get("skills") or priority):
        s_toks = _tokenize_and_map(s, synonyms)
        present = any(tok in resume_tokens for tok in s_toks) if s_toks else False
        if not present:
            gaps.append(s)

    return score, gaps

def _parse_pdf_with_pre_llm(content: bytes):
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

_RAKE_STOPWORDS = {
    "a", "an", "and", "the", "or", "for", "to", "of", "in", "on", "with", "is", "are",
    "at", "by", "be", "from", "as", "that", "this", "we", "you", "your", "our", "us",
}

def _candidate_phrases(text: str) -> List[str]:
    words = re.split(r"\s+", (text or "").lower())
    phrases = []
    cur = []
    for w in words:
        w = re.sub(r"[^a-z0-9+-]", "", w)
        if not w:
            if cur:
                phrases.append(" ".join(cur))
                cur = []
            continue
        if w in _RAKE_STOPWORDS:
            if cur:
                phrases.append(" ".join(cur))
                cur = []
            continue
        cur.append(w)
    if cur:
        phrases.append(" ".join(cur))
    seen = set()
    out = []
    for p in phrases:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out

def _rake_score_phrases(cands: List[str]) -> List[Tuple[str, float]]:
    scores = []
    for p in cands:
        toks = [t for t in re.split(r"\W+", p) if t]
        score = len(toks) + 0.1 * len(set(toks))
        scores.append((p, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def _rake_extract(text: str, top_n: int = 8) -> List[str]:
    if not text:
        return []
    cands = _candidate_phrases(text)
    scored = _rake_score_phrases(cands)
    return [p for p, s in scored[:top_n]]

def _get_resume_record(resume_id: int):
    db, cursor = get_db()
    if db:
        try:
            cursor.execute("""
                SELECT id, user_id, resume_text, parsed_sections, parsed_contacts
                FROM resumes WHERE id=%s
            """, (resume_id,))
            row = cursor.fetchone()
            if row:
                def _as_json(v):
                    if v is None:
                        return None
                    if isinstance(v, (dict, list)):
                        return v
                    try:
                        return json.loads(v)
                    except Exception:
                        return None
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
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    return v

def _decimal_or_null(v):
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

@limiter.limit("10 per minute")
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

    if "file" not in request.files:
        return bad("No file part. Use 'file' field for upload or send JSON with 'text'")
    file = request.files["file"]
    fname = secure_filename(file.filename or "resume.pdf")
    mime = file.mimetype or ""
    if mime not in ALLOWED_MIME:
        return bad("Only PDF, DOCX, or TXT allowed")

    content = file.read() or b""

    safe_name = secure_filename(fname)
    payload = {
        "file_b64": base64.b64encode(content).decode("utf-8"),
        "filename": safe_name,
    }

    if mime == "application/pdf":
        parsed = ml_post("parse-resume", payload)
        text = parsed.get("cleaned_text", "")
    elif mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        result = mammoth.extract_raw_text(BytesIO(content))
        raw = (result.value or "").strip()
        payload["raw_text"] = raw
        parsed = ml_post("parse-resume", payload)
        text = parsed["cleaned_text"]
    else:
        raw = content.decode("utf-8", errors="ignore")
        payload["raw_text"] = raw
        parsed = ml_post("parse-resume", payload)
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

    if not MEM["resumes"]:
        return bad("No resume", 404)
    user_items = [(rid, r) for rid, r in MEM["resumes"].items() if r.get("user_id") == uid]
    rid, r = (user_items[-1] if user_items else list(MEM["resumes"].items())[-1])
    return ok({
        "resume_id": rid,
        "name": r.get("file_name") or "pasted-text",
        "uploaded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    })

@app.put("/api/resumes/<int:rid>")
def resume_replace_existing(rid: int):
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
    safe_name = secure_filename(f.filename)

    payload = {
        "file_b64": base64.b64encode(content).decode("utf-8"),
        "filename": safe_name,
    }

    if mime == "application/pdf":
        parsed = ml_post("parse-resume", payload)
        text = parsed.get("cleaned_text", "")
    elif mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        result = mammoth.extract_raw_text(BytesIO(content))
        raw = (result.value or "").strip()
        payload["raw_text"] = raw
        parsed = ml_post("parse-resume", payload)
        text = parsed["cleaned_text"]
    else:
        raw = content.decode("utf-8", errors="ignore")
        payload["raw_text"] = raw
        parsed = ml_post("parse-resume", payload)
        text = parsed["cleaned_text"]

    safe_name = secure_filename(f.filename)

    if db:
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

    r = MEM["resumes"].get(rid)
    if not r or not (r.get("user_id") == uid or (r.get("user_id") is None and uid is None)):
        return bad("Not found", 404)

    return ok({
        "resume_id": rid,
        "parsed_sections": r.get("parsed_sections"),
        "parsed_contacts": r.get("parsed_contacts"),
    })

def extract_experience_level_helper(text: str) -> str:
    if not text:
        return "unknown"

    s = text.lower()

    if re.search(
        r"\b(intern(ship)?|intern|fresher|new\s+grad|graduate|entry[- ]?level|junior)\b",
        s,
    ):
        return "entry"

    if re.search(
        r"\b(senior|sr\.?|lead|principal|director|vp\b|vice\s+president|manager)\b",
        s,
    ):
        return "senior"

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
    if not isinstance(job_raw, dict):
        job_raw = {}

    parts: List[str] = []
    for k in ("contract_time", "contract_type"):
        v = job_raw.get(k)
        if v:
            parts.append(str(v))
    parts.append(str(title or ""))
    parts.append(str(description or ""))
    txt = " ".join(parts).lower()

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
            continue

    return ""

def _canonicalize_type_input(s: str) -> str:
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
@limiter.limit("10 per minute")
@app.post("/api/auth/register")
def auth_register():
    data = request.get_json(force=True) or {}
    full_name = (data.get("full_name") or data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not full_name:
        return bad("Missing 'full_name'")
    if not email or "@" not in email:
        return bad("Provide a valid email address")
    if not password or len(password) < 8:
        return bad("Password must be at least 8 characters")

    db, cursor = get_db()

    if db:
        try:
            cursor.execute("SELECT id FROM users WHERE email=%s", (email,))
            if cursor.fetchone():
                return bad("Email already registered", 409)
        except Exception as e:
            app.logger.exception(e)
    else:
        MEM.setdefault("users", {})
        for u in MEM["users"].values():
            if (u.get("email") or "").lower() == email:
                return bad("Email already registered", 409)

    try:
        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    except Exception as e:
        app.logger.exception(e)
        return bad("Failed to hash password", 500)

    if db:
        try:
            cursor.execute(
                "INSERT INTO users (full_name, email, password_hash) VALUES (%s, %s, %s)",
                (full_name, email, hashed),
            )
            cursor.execute("SELECT LAST_INSERT_ID() AS id")
            uid = cursor.fetchone()["id"]
            db.commit()
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
        MEM.setdefault("user_profiles", {})
        MEM["user_profiles"][uid] = {"user_id": uid}
        user_obj = {"user_id": uid, "name": full_name, "email": email, "role": "jobseeker"}

    try:
        exp = datetime.now(timezone.utc) + timedelta(days=7)
        token = jwt.encode({"user_id": user_obj.get("user_id"), "exp": int(exp.timestamp())}, JWT_SECRET, algorithm=JWT_ALGO)
    except Exception as e:
        app.logger.exception(e)
        return bad("Failed to create token", 500)

    return ok({"token": token, "user": user_obj})

@limiter.limit("10 per minute")
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

    try:
        if not stored or not bcrypt.checkpw(password.encode("utf-8"), stored.encode("utf-8")):
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

# ... (the rest of your routes from jobs_search onward remain exactly as in your original file)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)