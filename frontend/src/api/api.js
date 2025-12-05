// src/api/api.js

// Base API URL:
// - Use VITE_API_BASE_URL when set
// - In production, default to the Render backend URL
// - In development, default to localhost
const API_BASE =
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.PROD
    ? "https://job-ai-web-tool.onrender.com/api"
    : "http://localhost:5001/api");

// Helper: handle errors and responses in a consistent way
async function handleResponse(res) {
  if (res.ok) {
    try {
      return await res.json();
    } catch {
      return await res.text();
    }
  }

  let errBody = null;
  try {
    errBody = await res.json();
  } catch {
    try {
      errBody = await res.text();
    } catch {
      errBody = null;
    }
  }

  const message =
    (errBody && (errBody.error || errBody.message || JSON.stringify(errBody))) ||
    res.statusText ||
    `HTTP ${res.status}`;

  throw new Error(message);
}

// Canonicalize experience labels from UI to backend friendly tokens
function canonicalizeExperience(exp) {
  if (typeof exp !== "string") return "";
  const s = exp.trim().toLowerCase();
  if (!s) return "";

  const map = {
    entry: "entry",
    "entry-level": "entry",
    "entry level": "entry",
    junior: "entry",
    mid: "mid",
    "mid-level": "mid",
    "mid level": "mid",
    senior: "senior",
    "senior-level": "senior",
    "senior level": "senior",
    lead: "senior",
  };

  return map[s] || s;
}

// Canonicalize job type labels from UI
function canonicalizeType(t) {
  if (typeof t !== "string") return "";
  const s = t.trim().toLowerCase();
  if (!s) return "";

  const map = {
    "full-time": "Full-time",
    "full time": "Full-time",
    fulltime: "Full-time",
    "part-time": "Part-time",
    "part time": "Part-time",
    parttime: "Part-time",
    contract: "Contract",
    temporary: "Contract",
    intern: "Internship",
    internship: "Internship",
    remote: "Remote",
    hybrid: "Hybrid",
  };

  return map[s] || t;
}

// ----------------------------------------------------
// Auth token handling (module level, optional to use)
// ----------------------------------------------------

let authToken = null;

export function setAuthToken(t) {
  authToken = t || null;
}

function getToken() {
  return (
    authToken ??
    (typeof localStorage !== "undefined" ? localStorage.getItem("token") : null)
  );
}

// -----------------
// Basic endpoints
// -----------------

// Health check
export async function getHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return handleResponse(res);
}

// Upload resume as plain text JSON
export async function uploadResume(text, meta = {}) {
  const res = await fetch(`${API_BASE}/resumes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, meta }),
  });
  return handleResponse(res);
}

// Upload resume as file
export async function uploadResumeFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/resumes`, {
    method: "POST",
    body: formData,
  });
  return handleResponse(res);
}

// -----------------
// Job search
// -----------------

export async function searchJobs(options = {}) {
  const payload = {
    ...(options.inputs ? { inputs: options.inputs } : {}),
    ...(options.query ? { query: options.query } : {}),
    ...(options.location ? { location: options.location } : {}),
    page: options.page || 1,
    resultsPerPage: 30,
    ...(Number.isFinite(options.salaryMin)
      ? { salaryMin: options.salaryMin }
      : {}),
    ...(Number.isFinite(options.salaryMax)
      ? { salaryMax: options.salaryMax }
      : {}),
    type:
      typeof options.type !== "undefined"
        ? canonicalizeType(options.type)
        : "",
    experience:
      typeof options.experience !== "undefined"
        ? canonicalizeExperience(options.experience)
        : "",
  };

  const res = await fetch(`${API_BASE}/jobs/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await handleResponse(res);

  if (data && Array.isArray(data.results)) {
    const mapped = data.results.map((j) => ({
      job_id: j.job_id,
      external_id: j.external_id,
      title: j.title,
      company: j.company,
      location: j.location,
      description: j.description,
      skills: Array.isArray(j.skills) ? j.skills : j.skills || [],
      salaryMin: Number.isFinite(j.salary_min)
        ? j.salary_min
        : Number.isFinite(j.salaryMin)
        ? j.salaryMin
        : null,
      salaryMax: Number.isFinite(j.salary_max)
        ? j.salary_max
        : Number.isFinite(j.salaryMax)
        ? j.salaryMax
        : null,
      category: j.category,
      url: j.url,
      raw: j.raw || {},
      type: j.type || "",
      experience: j.experience_level || j.experience || "",
    }));

    const respPage = Number(data.page || 1);
    const respRpp = Number(
      data.resultsPerPage ?? data.results_per_page ?? 30
    );
    const respTotal = Number(
      data.totalResults ?? data.total_results ?? 0
    );

    return {
      page: respPage,
      resultsPerPage: respRpp,
      totalResults: respTotal,
      _raw: data,
      results: mapped,
    };
  }

  return data;
}

// -----------------
// Recommendation
// -----------------

export async function getRecommendations(resumeId, jobIdsOrJobs, opts = {}) {
  const first =
    Array.isArray(jobIdsOrJobs) && jobIdsOrJobs.length
      ? jobIdsOrJobs[0]
      : null;

  let jobIds = [];
  let jobsPayload;

  if (first && typeof first === "object") {
    jobsPayload = jobIdsOrJobs;
    jobIds = jobIdsOrJobs
      .map((j) => j.job_id || j.external_id)
      .filter(Boolean);
  } else {
    jobIds = Array.isArray(jobIdsOrJobs) ? jobIdsOrJobs : [];
  }

  const body = { resume_id: resumeId, job_ids: jobIds };
  if (jobsPayload) body.jobs = jobsPayload;
  if (opts.use_field_aware !== undefined)
    body.use_field_aware = !!opts.use_field_aware;

  const res = await fetch(`${API_BASE}/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const data = await handleResponse(res);

  if (data && Array.isArray(data.results)) {
    const mapped = data.results.map((r) => ({
      job_id: r.job_id,
      title: r.title,
      company: r.company,
      location: r.location,
      score: r.score,
      gaps: r.gaps || [],
      resume_bullets: r.resume_bullets || [],
      cover_letter: r.cover_letter || "",
      matched_skills: r.matched_skills || [],
      derived_resume_skills: r.derived_resume_skills || [],
    }));
    return { ...data, results: mapped };
  }

  return data;
}

// Match a single resume with a single job
export async function matchResumeToJob(payload = {}) {
  const res = await fetch(`${API_BASE}/match`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await handleResponse(res);

  if (data && typeof data.score !== "undefined") {
    return { score: Number(data.score) };
  }

  return data;
}

// Fetch a single job by ID
export async function getJob(jobId) {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`);
  return handleResponse(res);
}

// -----------------
// AI cover letters
// -----------------

export async function generateCoverLetter(payload) {
  const res = await fetch(`${API_BASE}/ai/cover-letter`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await handleResponse(res);

  if (!data || typeof data !== "object") {
    return { cover_letter: "", resume_bullets: [] };
  }

  return {
    cover_letter:
      data.cover_letter || data.coverLetter || data.cover || "",
    resume_bullets: Array.isArray(data.resume_bullets)
      ? data.resume_bullets
      : Array.isArray(data.resumeBullets)
      ? data.resumeBullets
      : [],
    _raw: data,
  };
}

// -----------------
// Saved and applied jobs
// -----------------

export async function saveJob(jobId, notes = "") {
  const token = getToken();
  const headers = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}/users/me/saved-jobs`, {
    method: "POST",
    headers,
    body: JSON.stringify({ job_id: jobId, notes }),
  });

  return handleResponse(res);
}

export async function applyJob(jobId) {
  const token = getToken();
  const headers = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}/users/me/applied-jobs`, {
    method: "POST",
    headers,
    body: JSON.stringify({ job_id: jobId }),
  });

  return handleResponse(res);
}

export async function getAppliedJobs() {
  const token = getToken();
  const headers = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}/users/me/applied-jobs`, {
    headers,
    credentials: "include",
  });

  return handleResponse(res);
}

export async function getSavedJobs() {
  const token = getToken();
  const headers = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}/users/me/saved-jobs`, { headers });
  return handleResponse(res);
}

export async function deleteSavedJob(jobId) {
  const token = getToken();
  const headers = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}/users/me/saved-jobs/${jobId}`, {
    method: "DELETE",
    headers,
  });

  return handleResponse(res);
}

export async function deleteAppliedJob(jobId) {
  const token = getToken();
  const headers = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}/users/me/applied-jobs/${jobId}`, {
    method: "DELETE",
    headers,
  });

  return handleResponse(res);
}

// -----------------
// Auth
// -----------------

export async function authRegister(payload) {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  return handleResponse(res);
}

export async function authLogin(payload) {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  return handleResponse(res);
}