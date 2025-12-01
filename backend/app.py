# backend/app.py
from __future__ import annotations
import os, re, json, random, string
import requests
from datetime import datetime, timezone 
from typing import Any, Dict, List, Tuple
import base64
import bcrypt
import jwt

import sys, os
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
import bcrypt
import jwt
from datetime import timedelta

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

# Environment Loading & Flask App Setup
load_dotenv()
app = Flask(__name__)

# Rate Limiter (security)
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
)

# General App Config
JOB_DESCRIPTION_MAX_CHARS = int(os.getenv("JOB_DESCRIPTION_MAX_CHARS", "2000"))

# Logging: External API Credentials
if os.getenv("ADZUNA_APP_ID") and os.getenv("ADZUNA_APP_KEY"):
    app.logger.info("Adzuna credentials found in environment")
else:
    app.logger.warning(
        "Adzuna credentials missing — set ADZUNA_APP_ID and ADZUNA_APP_KEY"
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
        "http://localhost:3000"
    ]

CORS(
    app,
    resources={r"/api/*": {"origins": origins}},
    supports_credentials=True,
    expose_headers=["Content-Type", "Authorization"],
)

#JWT Utilities
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET is not set. Add it to your environment variables.")
# JWT / Auth Utilities
JWT_SECRET = os.getenv("JWT_SECRET", "super-secret-key")
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

# Feature flag: enable field-aware simple matcher
USE_FIELD_AWARE_MATCHER = str(os.getenv("USE_FIELD_AWARE_MATCHER", "false") or "").lower() in ("1", "true", "yes")
# Feature flag: produce looser (more generous) scoring when enabled
USE_LOOSE_SCORING = str(os.getenv("USE_LOOSE_SCORING", "true") or "").lower() in ("1", "true", "yes")
# Feature flag: produce looser (more generous) scoring when enabled
SCORE_LOOSENESS = float(os.getenv("SCORE_LOOSENESS", "1.55"))
# Field-aware tuning: small boost when resume and job share a detected field,
# and a multiplier penalty when they differ. Tune via env vars.
FIELD_MATCH_BOOST = float(os.getenv("FIELD_MATCH_BOOST", ".15"))
FIELD_MISMATCH_PENALTY = float(os.getenv("FIELD_MISMATCH_PENALTY", "0.4"))
# Weight to apply to matches from the `base_skills` bucket (lower than priority_skills)
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
    Prefer PyMySQL connection when available (pure-Python, robust).
    Fall back to mysql-connector-python if PyMySQL is not installed or fails.
    Returns (db_conn, cursor) or (None, None) on failure.
    
    Uses Flask's 'g' object for thread-local storage to avoid thread-safety issues.
    """
    if not USE_DB:
        return None, None

    # Check if we have a thread-local connection
    if hasattr(g, '_db') and g._db is not None:
        try:
            # Test if connection is still alive
            if hasattr(g._db, "ping"):
                # pymysql exposes ping(reconnect=True) - auto-reconnect!
                try:
                    g._db.ping(reconnect=True)  # Auto-reconnect if connection lost
                except TypeError:
                    # some connectors have different signature
                    g._db.ping()
            elif hasattr(g._db, "is_connected"):
                if not g._db.is_connected():
                    raise Exception("stale connection")
            return g._db, g._cursor
        except Exception as e:
            # Connection is dead and couldn't reconnect, log and clear
            app.logger.warning(f"DB connection lost, reconnecting: {e}")
            try:
                g._db.close()
            except Exception:
                pass
            g._db = None
            g._cursor = None

    # Try PyMySQL first (pure-Python, avoids mysql_native_password .so issues)
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
        app.logger.info("Connected to MySQL via PyMySQL (thread-local)")
        return g._db, g._cursor
    except Exception as e_py:
        app.logger.info(f"PyMySQL connect failed (will try mysql-connector): {e_py}")

    # Fallback: try mysql-connector (use_pure=True) as before
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
        app.logger.info("Connected to MySQL via mysql-connector-python (thread-local)")
        return g._db, g._cursor
    except Exception as e:
        app.logger.warning(f"MySQL unavailable, using memory store. Error: {e}")
        return None, None


@app.teardown_appcontext
def close_db(exception=None):
    """Close the database connection at the end of each request."""
    db = g.pop('_db', None)
    if db is not None:
        try:
            db.close()
        except Exception:
            pass

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
# Simple, field-aware non-LLM matching utilities
# Profiles and keywords are loaded from `backend/field_profiles.py` to keep
# `app.py` smaller and make profiles easy to export/edit.
# -----------------------------
import importlib.util
_FIELD_KEYWORDS = {}
FIELD_PROFILES = {}

_profiles_json = os.path.join(os.path.dirname(__file__), "field_profiles.json")
_profiles_path = os.path.join(os.path.dirname(__file__), "field_profiles.py")

_loaded_profiles = False

# Prefer loading the Python module directly (so you don't need to regenerate JSON).
# Fall back to the exported JSON if the module is unavailable or fails to load.
if os.path.exists(_profiles_path):
    try:
        spec = importlib.util.spec_from_file_location("backend_field_profiles", _profiles_path)
        _fp_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_fp_mod)

        # If the module exposes a `to_dict()` helper (recommended), use it to
        # obtain the canonical exported structure (field_keywords, profiles, base_skills).
        if hasattr(_fp_mod, "to_dict"):
            try:
                _data = _fp_mod.to_dict()
                _FIELD_KEYWORDS = _data.get("field_keywords", {}) or {}
                FIELD_PROFILES = _data.get("profiles", {}) or {}
                _BASE_SKILLS = set([s.lower() for s in (_data.get("base_skills") or [])])
            except Exception:
                # If to_dict() fails for any reason, fall back to inspecting module attrs
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
        app.logger.info(f"Loaded field profiles from JSON: {_profiles_json}")
        # Load base skills from exported JSON if present
        _BASE_SKILLS = set([s.lower() for s in (_data.get("base_skills") or [])])
        _loaded_profiles = True
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
    # normalize job skills into tokens
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
            # check partial presence
            if any(skill in rt for rt in resume_tokens):
                found += 1
    return found / len(set(skill_tokens))

def _detect_field(job: dict) -> str | None:
    explicit = (job.get("field") or job.get("category") or "").strip().lower()
    if explicit and explicit in FIELD_PROFILES:
        return explicit
    txt = " ".join(filter(None, [job.get("title",""), job.get("description","")])).lower()
    for fld, kws in _FIELD_KEYWORDS.items():
        for kw in kws:
            if kw in txt:
                return fld
    return None

def _simple_match_score(resume_text: str, job: dict, use_loose: bool = False) -> Tuple[int, List[str]]:
    """
    Lightweight, field-aware score. Returns (score:int, gaps:list[str])
    """
    field = _detect_field(job) or "data"  # default to data if uncertain
    profile = FIELD_PROFILES.get(field, {})
    weights = profile.get("weights", {"skill": 0.6, "title": 0.25, "desc": 0.15})
    synonyms = profile.get("synonyms", {})
    priority = profile.get("priority_skills", [])

    resume_tokens = _tokenize_and_map(resume_text or "", synonyms)
    title_tokens = _tokenize_and_map(job.get("title", ""), synonyms)
    desc_tokens = _tokenize_and_map(job.get("description") or job.get("full_description") or "", synonyms)
    job_skills = job.get("skills") or priority

    # Primary priority-skill overlap
    skill_overlap = _skill_overlap(resume_tokens, job_skills, synonyms)
    # Secondary base-skill overlap (lower weight). _BASE_SKILLS is loaded from profiles JSON.
    try:
        base_overlap = 0.0
        if isinstance(_BASE_SKILLS, (set, list)) and len(_BASE_SKILLS) > 0:
            base_overlap = _skill_overlap(resume_tokens, list(_BASE_SKILLS), synonyms)
        # Combine priority and base overlaps into an effective skill signal
        skill_overlap = min(1.0, (skill_overlap + (BASE_SKILL_WEIGHT * base_overlap)))
    except Exception:
        pass
    title_sim = _jaccard(resume_tokens, title_tokens)
    desc_sim = _jaccard(resume_tokens, desc_tokens)

    # If description missing, reweight to rely more on skills/title
    if not desc_tokens:
        total = weights.get("skill", 0.6) + weights.get("title", 0.25)
        w_skill = weights.get("skill", 0.6) / (total or 1)
        w_title = weights.get("title", 0.25) / (total or 1)
        w_desc = 0.0
    else:
        w_skill = weights.get("skill", 0.6)
        w_title = weights.get("title", 0.25)
        w_desc = weights.get("desc", 0.15)

    # Combine component similarities into a single 0..1 value
    combined = (skill_overlap * w_skill) + (title_sim * w_title) + (desc_sim * w_desc)

    # Field-aware adjustment: detect the dominant field for the resume and apply
    # a small boost when the resume and job share the same detected field,
    # or apply a penalty multiplier when they differ. These are tunable
    # via env vars `FIELD_MATCH_BOOST` and `FIELD_MISMATCH_PENALTY`.
    try:
        resume_field = _detect_field({"title": resume_text or "", "description": resume_text or ""})
    except Exception:
        resume_field = None
    if resume_field and field and resume_field == field:
        combined = min(1.0, combined + FIELD_MATCH_BOOST)
    elif resume_field and field and resume_field != field:
        # Apply multiplicative penalty to reduce cross-field high scores
        combined = combined * FIELD_MISMATCH_PENALTY
    # If repository/environment favors certain fields, give those jobs an extra bump
    # We also count keyword/synonym hits for the field to slightly increase spread
    try:
        if field and field.lower() in FAVOR_FIELDS:
            # gather candidate keywords: field keywords + priority skills
            fk = []
            try:
                fk = list(_FIELD_KEYWORDS.get(field, []) or [])
            except Exception:
                fk = []
            cand_keywords = fk + list(priority or [])

            # count how many field keywords appear in the resume (cap at 5)
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

            # small additive boost based on configured FAVOR_FIELD_BOOST + per-hit bonus
            per_hit_bonus = float(os.getenv("FAVOR_FIELD_PER_HIT", "0.02"))
            combined = min(1.0, combined + FAVOR_FIELD_BOOST + (per_hit_bonus * hits))

            # apply a small spread multiplier to increase variation for favored fields
            spread_multiplier = 1.0 + (0.06 * min(hits, 5))
        else:
            spread_multiplier = 1.0
    except Exception:
        spread_multiplier = 1.0

    # Allow looser scoring mode: increase multiplier and slightly boost skill overlap
    multiplier = 59
    min_allowed = 35
    base = 40
    if use_loose:
        # Apply a stronger looseness multiplier and boost overlap
        loos = SCORE_LOOSENESS if SCORE_LOOSENESS > 0 else 1.0
        multiplier = int(round(85 * loos))
        min_allowed = 25
        base = 35
        # larger boost to combined similarity to reward partial/related matches
        combined = min(1.0, combined + (0.08 * loos))

    # Increase score spread by applying a gamma (exponent) transform to combined
    # This exaggerates higher similarities and compresses lower ones, producing
    # greater variation between jobs. Tunable via `gamma`.
    gamma = 1.5
    # Apply spread multiplier (favored-field hits increase dispersion)
    raw = int(round((combined ** gamma) * multiplier * (spread_multiplier if 'spread_multiplier' in locals() else 1.0)))
    score = max(min_allowed, min(99, base + raw))

    # Build gaps ordered by priority then others
    gaps = []
    for s in (job.get("skills") or priority):
        s_toks = _tokenize_and_map(s, synonyms)
        present = any(tok in resume_tokens for tok in s_toks) if s_toks else False
        if not present:
            gaps.append(s)

    return score, gaps


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


# Lightweight RAKE-like extractor (no external deps)
_RAKE_STOPWORDS = set(
    [
        "a","an","and","the","or","for","to","of","in","on","with","is","are",
        "at","by","be","from","as","that","this","we","you","your","our","us",
    ]
)

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
    # dedupe while preserving order
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
        # score by length and uniqueness
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

    # Shared payload for ML service
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

    else:  # text/plain
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

    else:  # text/plain
        raw = content.decode("utf-8", errors="ignore")
        payload["raw_text"] = raw
        parsed = ml_post("parse-resume", payload)
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
@limiter.limit("10 per minute")
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
        adzuna_id = job.get("id")  # Adzuna's unique job ID
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

        ## UPSERT (insert or update existing based on external_id from Adzuna)
        if db:
            try:
                # Use external_id (Adzuna ID) as the primary matching key
                # This ensures the same job gets the same job_id across searches
                cursor.execute(
                    '''
                    INSERT INTO jobs (external_id, title, company_name, industry, description, location, salary_range, source, url)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        title = VALUES(title),
                        company_name = VALUES(company_name),
                        description = VALUES(description),
                        salary_range = VALUES(salary_range),
                        url = VALUES(url),
                        posted_at = CURRENT_TIMESTAMP
                    ''',
                    (adzuna_id, title, company, category, description, location_name, f"{salary_min}-{salary_max}", source, url_job)
                )
                # Get the job_id (either newly inserted or existing)
                cursor.execute("SELECT job_id FROM jobs WHERE external_id = %s", (adzuna_id,))
                result = cursor.fetchone()
                if result:
                    job_id = result["job_id"]
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
            "job_id": job_id,  # Database auto-increment ID
            "adzuna_id": adzuna_id,  # Adzuna's external ID
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
    user_id_for_rec = _get_user_id()
    persist_recs = bool(data.get("persist", False))
    provided_jobs_list = data.get("jobs") or []
    # Build a lookup for provided job objects by job_id and external_id (if present)
    provided_jobs = {}
    for j in provided_jobs_list:
        try:
            if isinstance(j, dict):
                if j.get("job_id") is not None:
                    provided_jobs[int(j.get("job_id"))] = j
                if j.get("external_id"):
                    provided_jobs[str(j.get("external_id"))] = j
        except Exception:
            continue

    # Per-request override to enable field-aware matcher for quick testing.
    use_field_override = data.get("use_field_aware", None)
    if use_field_override is not None:
        try:
            use_field_override = str(use_field_override).lower() in ("1", "true", "yes")
        except Exception:
            use_field_override = False
    use_field = use_field_override if use_field_override is not None else USE_FIELD_AWARE_MATCHER

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

    # Use derived skills only; remove substring matching / scoring logic.
    # This avoids any heuristic matching and returns results based on
    # skills extracted from the resume (or empty if none).
    derived = _extract_resume_skills(resume_text, user_listed_skills)

    results = []
    for jid in job_ids:
        # allow jid to be passed as string or int
        job = None
        try:
            key = int(jid)
        except Exception:
            key = jid
        # first try in-memory store
        job = MEM["jobs"].get(key) if key is not None else None
        # then try provided jobs payload (fallback)
        if not job:
            job = provided_jobs.get(key) or provided_jobs.get(str(jid))
        if not job:
            # nothing we can score without a job object; skip
            continue
        if use_field:
            # Field-aware scoring (non-LLM)
            try:
                    # Determine per-request loose scoring override
                    use_loose_override = data.get("use_loose_scoring", None)
                    if use_loose_override is not None:
                        try:
                            use_loose_override = str(use_loose_override).lower() in ("1", "true", "yes")
                        except Exception:
                            use_loose_override = False
                    use_loose = use_loose_override if use_loose_override is not None else USE_LOOSE_SCORING

                    score, gaps = _simple_match_score(resume_text, job, use_loose=use_loose)
            except Exception as e:
                app.logger.exception(f"_simple_match_score failed: {e}")
                score, gaps = None, []
            # matched skills prefer explicit job.skills overlap, else derived
            matched = [s for s in (job.get("skills") or []) if s.lower() in (resume_text or "").lower()] or derived[:3]
        else:
            score, gaps = None, []
            matched = derived[:3]

        bullets = _make_bullets(job.get("title", ""), job.get("company", ""), matched)
        cover = _make_cover_letter(candidate_name, job.get("title", ""), job.get("company", ""), matched, gaps[:3])
        results.append(
            {
                "job_id": jid,
                "title": job.get("title"),
                "company": job.get("company"),
                "location": job.get("location"),
                "score": score,
                "gaps": gaps[:3],
                "resume_bullets": bullets,
                "cover_letter": cover,
                "matched_skills": matched,
                "derived_resume_skills": derived[:8],
            }
        )

    # Optionally persist recommendations into `job_recommendations` so
    # saved/applied views (or other queries) can surface the score later.
    if persist_recs:
        db2, cursor2 = get_db()
        for r in results:
            try:
                jid = r.get("job_id")
                score = r.get("score")
                cover = r.get("cover_letter")
                # Persist only when we have a DB and a user context
                if db2 and user_id_for_rec and jid is not None:
                    try:
                        # delete any existing recommendation for this user+job, then insert
                        cursor2.execute("DELETE FROM job_recommendations WHERE user_id=%s AND job_id=%s", (user_id_for_rec, int(jid)))
                        cursor2.execute(
                            """
                            INSERT INTO job_recommendations
                              (user_id, job_id, match_score, generated_resume, generated_cover_letter, recommended_at)
                            VALUES (%s, %s, %s, %s, %s, NOW())
                            """,
                            (user_id_for_rec, int(jid), float(score) if score is not None else None, None, cover),
                        )
                        if hasattr(db2, 'commit'):
                            db2.commit()
                    except Exception:
                        try:
                            db2.rollback()
                        except Exception:
                            pass
                else:
                    # memory fallback
                    MEM.setdefault("job_recommendations", []).append({
                        "user_id": user_id_for_rec,
                        "job_id": jid,
                        "match_score": r.get("score"),
                        "generated_cover_letter": r.get("cover_letter"),
                        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    })
            except Exception as e:
                app.logger.exception(f"Failed to persist recommendation: {e}")

    return ok({"results": results})


@app.post("/api/match")
def match_job():
    """
    Body JSON:
      - resume_id: int (required)
      - job_id: int (optional)  OR
      - job: { "title":..., "company":..., "description":..., "skills": [...] } (optional)

    Returns relevance percentage (0-100), matched skills, and gaps.
    """
    data = request.get_json(force=True) or {}
    resume_id = data.get("resume_id")
    resume_text_provided = (data.get("resume_text") or "").strip()
    job_id = data.get("job_id")
    job_obj = data.get("job")

    # Accept either an inline resume_text (useful for quick tests) or a resume_id
    db, cursor = get_db()
    resume_text = ""
    user_listed_skills: List[str] = []

    if resume_text_provided:
        resume_text = resume_text_provided
    elif resume_id:
        if db:
            try:
                cursor.execute("SELECT resume_text FROM resumes WHERE id=%s", (resume_id,))
                row = cursor.fetchone()
                if row:
                    resume_text = row.get("resume_text") or ""
            except Exception as e:
                app.logger.exception(e)

        if not resume_text:
            r = MEM["resumes"].get(resume_id, {})
            resume_text = r.get("text", "")
            user_listed_skills = r.get("skills", []) or []

    if not resume_text:
        return bad("Missing 'resume_id' or 'resume_text'")

    # Determine job data
    job_skills: List[str] = []
    job_title = ""
    job_company = ""
    desc = ""

    # If job_id provided, fetch from DB or MEM
    if job_id and not job_obj:
        if db:
            try:
                cursor.execute(
                    "SELECT job_id, title, company_name AS company, description, location FROM jobs WHERE job_id=%s",
                    (job_id,),
                )
                row = cursor.fetchone()
                if row:
                    job_title = row.get("title") or ""
                    job_company = row.get("company") or ""
                    desc = row.get("description") or ""
                else:
                    desc = ""
            except Exception as e:
                app.logger.exception(e)
                desc = ""
        else:
            j = MEM["jobs"].get(job_id, {})
            job_title = j.get("title", "")
            job_company = j.get("company", "")
            desc = j.get("description", "")

        # Try to get explicit skills from DB if stored as JSON under 'skills'
        if db:
            try:
                cursor.execute("SELECT skills FROM jobs WHERE job_id=%s", (job_id,))
                r2 = cursor.fetchone()
                if r2 and r2.get("skills"):
                    job_skills = r2.get("skills") if isinstance(r2.get("skills"), list) else json.loads(r2.get("skills") or "[]")
            except Exception:
                pass

    # If job object provided directly
    if job_obj:
        job_title = job_obj.get("title", "")
        job_company = job_obj.get("company", "")
        desc = job_obj.get("description", "") or job_obj.get("summary", "")
        job_skills = job_obj.get("skills") or []

    # Derive resume skills from resume text and optional user-listed skills
    derived = _extract_resume_skills(resume_text, user_listed_skills)

    # Extract job phrases via RAKE (use title+description+company as source)
    combined_job_text = " ".join([str(x or "") for x in [job_title, desc, job_company]])
    job_phrases = _rake_extract(combined_job_text, top_n=16)

    # Normalize lists for comparison
    resume_skills_norm = [s.lower() for s in derived]
    job_phrases_norm = [p.lower() for p in job_phrases]
    desc_norm = (desc or "").lower()

    matched = []
    gaps = []

    for i, rs in enumerate(resume_skills_norm):
        # exact phrase match against job_phrases
        if rs in job_phrases_norm:
            matched.append(derived[i])
            continue

        # word-boundary search in full description
        tokens = [t for t in re.split(r"\W+", rs) if t]
        found = False
        for t in tokens:
            if re.search(rf"\b{re.escape(t)}\b", desc_norm):
                matched.append(derived[i])
                found = True
                break

        if not found:
            # also check against job_skills provided explicitly
            for js in (job_skills or []):
                try:
                    if js and js.lower() == rs:
                        matched.append(derived[i])
                        found = True
                        break
                except Exception:
                    continue

        if not found:
            gaps.append(derived[i])

    # Compute score using same heuristic as before
    score = max(40, min(99, 60 + 7 * len(matched)))

    result = {
        "resume_id": resume_id,
        "job_id": job_id,
        "title": job_title,
        "company": job_company,
        "score_percent": score,
        "matched_skills": matched,
        "gaps": gaps,
        "derived_resume_skills": derived[:8],
        "job_phrases": job_phrases,
    }
    return ok(result)

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
@limiter.limit("20 per minute")
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
               j.industry, j.salary_range, j.url, j.source, j.description,
               (SELECT match_score FROM job_recommendations jr WHERE jr.user_id = sj.user_id AND jr.job_id = j.job_id ORDER BY jr.recommended_at DESC LIMIT 1) AS match_score
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
    if not db:
        return bad("Database not available", 500)

    try:
        cursor.execute("""
            INSERT INTO applied_jobs (user_id, job_id)
            VALUES (%s, %s)
        """, (user_id, job_id))

        # Commit is redundant with autocommit=True but harmless
        if hasattr(db, 'commit'):
            db.commit()
        return ok({"message": "Job marked as applied"})
    except Exception as e:
        # Handle both PyMySQL and mysql-connector IntegrityError
        error_str = str(e).lower()
        if 'duplicate' in error_str or 'integrity' in error_str:
            return bad("You already applied to this job", 409)
        app.logger.error(f"Error applying to job: {e}")
        return bad(f"Failed to apply: {e}", 500)

@app.delete("/api/users/me/applied-jobs/<int:job_id>")
def delete_applied_job(job_id):
    user_id = _get_user_id()
    if not user_id:
        return bad("Unauthorized", 401)

    db, cursor = get_db()
    if not db:
        return bad("Database not configured", 500)

    try:
        cursor.execute("""
            DELETE FROM applied_jobs
            WHERE user_id=%s AND job_id=%s
        """, (user_id, job_id))

        # Commit is redundant with autocommit=True but harmless
        if hasattr(db, 'commit'):
            db.commit()

        return ok({"message": "Removed applied job"})
    except Exception as e:
        app.logger.error(f"Error deleting applied job: {e}")
        return bad(f"Failed to delete: {e}", 500)

@app.get("/api/users/me/applied-jobs")
def get_applied_jobs():
    user_id = _get_user_id()
    if not user_id:
        return bad("Unauthorized", 401)

    db, cursor = get_db()

    cursor.execute("""
        SELECT aj.applied_id, aj.applied_at,
            j.job_id, j.title, j.company_name, j.location,
            j.salary_range, j.url, j.source,
            (SELECT match_score FROM job_recommendations jr WHERE jr.user_id = aj.user_id AND jr.job_id = j.job_id ORDER BY jr.recommended_at DESC LIMIT 1) AS match_score
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
            cover_letter_text = ml_post("generate-cover-letter", {
                "job_title": title,
                "company": company,
                "job_description": description,
                "contacts": contacts,
                "sections": sections,
                "candidate_name": candidate_name,
            }).get("cover_letter")

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

@app.get("/api/health/ml")
def ml_health():
    try:
        r = requests.get(f"{ML_SERVICE_URL}/health", timeout=5)
        r.raise_for_status()
        return ok({
            "ml_status": "up",
            "details": r.json()
        })
    except Exception as e:
        app.logger.error(f"[ML HEALTH ERROR] {e}")
        return bad("ML service unavailable", 503)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)