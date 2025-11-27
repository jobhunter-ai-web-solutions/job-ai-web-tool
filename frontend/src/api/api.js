// src/api/api.js

// Use Vite-provided env or sensible defaults. Prefer VITE_API_BASE when set.
const API_BASE =
  import.meta.env.VITE_API_BASE || (import.meta.env.PROD ? "https://api.yourdomain.com/api" : "http://localhost:5001/api");

// Helper: Handle errors & responses
async function handleResponse(res) {
  if (res.ok) {
    // Prefer JSON but fall back to text when server returns plain text
    try {
      return await res.json();
    } catch {
      return await res.text();
    }
  }

  // Non-OK: try to parse JSON error body, otherwise return text/status
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

// Small helper to canonicalize experience labels from the UI into tokens
// so the backend/provider queries aren't over-constrained by display text.
function canonicalizeExperience(exp) {
  if (typeof exp !== 'string') return '';
  const s = exp.trim().toLowerCase();
  if (!s) return '';
  const map = {
    'entry': 'entry',
    'entry-level': 'entry',
    'entry level': 'entry',
    'junior': 'entry',
    'mid': 'mid',
    'mid-level': 'mid',
    'mid level': 'mid',
    'senior': 'senior',
    'senior-level': 'senior',
    'senior level': 'senior',
    'lead': 'senior',
  };
  return map[s] || s;
}

// Small helper to canonicalize job contract/type labels from the UI into
// a predictable set the backend expects. Maps common display labels to
// tokens we use server-side (e.g. 'Full-time' -> 'Full-time').
function canonicalizeType(t) {
  if (typeof t !== 'string') return '';
  const s = t.trim().toLowerCase();
  if (!s) return '';
  const map = {
    'full-time': 'Full-time',
    'full time': 'Full-time',
    'fulltime': 'Full-time',
    'part-time': 'Part-time',
    'part time': 'Part-time',
    'parttime': 'Part-time',
    'contract': 'Contract',
    'temporary': 'Contract',
    'intern': 'Internship',
    'internship': 'Internship',
    'remote': 'Remote',
    'hybrid': 'Hybrid',
  };
  return map[s] || t;
}

// --------------------
// Example API methods
// --------------------

// Module-level auth token. Prefer this when set by the application (AuthContext)
// to avoid relying on localStorage at call-time and to make the helpers
// testable and deterministic.
let authToken = null

export function setAuthToken(t) {
  authToken = t || null
}

function getToken() {
  return authToken ?? (typeof localStorage !== 'undefined' ? localStorage.getItem('token') : null)
}

// Check backend health
export async function getHealth() {
  const res = await fetch(`${API_BASE}/health`); // ex: http://localhost:5000/api/health
  return await handleResponse(res);
}

// Upload resume (JSON)
export async function uploadResume(text, meta = {}) {
  const res = await fetch(`${API_BASE}/resumes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, meta }),
  });
  return handleResponse(res);
}

// Upload resume (file)
export async function uploadResumeFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/resumes`, {
    method: "POST",
    body: formData,
  });
  return handleResponse(res);
}

// Search jobs
export async function searchJobs(options = {}) {
  // options: { inputs, query, location, page, resultsPerPage, salaryMin, salaryMax, type, experience }
  const payload = {
    ...(options.inputs ? { inputs: options.inputs } : {}),
    ...(options.query ? { query: options.query } : {}),
    ...(options.location ? { location: options.location } : {}),
  page: options.page || 1,
  // Enforce 30 results per page (server expects and backend will also enforce this)
  resultsPerPage: 30,
  ...(Number.isFinite(options.salaryMin) ? { salaryMin: options.salaryMin } : {}),
  ...(Number.isFinite(options.salaryMax) ? { salaryMax: options.salaryMax } : {}),
  // Always include type and experience (server is authoritative). Fall back to empty strings so the backend receives explicit keys.
  type: typeof options.type !== 'undefined' ? canonicalizeType(options.type) : '',
  experience: typeof options.experience !== 'undefined' ? canonicalizeExperience(options.experience) : '',
  };

  const res = await fetch(`${API_BASE}/jobs/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  // Use handleResponse to throw on HTTP errors and parse body
  const data = await handleResponse(res);

  // If backend returned the expected structure, map job fields to frontend-friendly shape
  if (data && Array.isArray(data.results)) {
    const mapped = data.results.map((j) => ({
      job_id: j.job_id,
      external_id: j.external_id,
      title: j.title,
      company: j.company,
      location: j.location,
      description: j.description,
      skills: Array.isArray(j.skills) ? j.skills : (j.skills || []),
      salaryMin: Number.isFinite(j.salary_min) ? j.salary_min : (Number.isFinite(j.salaryMin) ? j.salaryMin : null),
      salaryMax: Number.isFinite(j.salary_max) ? j.salary_max : (Number.isFinite(j.salaryMax) ? j.salaryMax : null),
      category: j.category,
      url: j.url,
      raw: j.raw || {},
      // Prefer server-provided normalized type (backend provides `type`). Do not infer on frontend.
      type: j.type || '',
      // Normalize experience to a simple returned value (lowercase from backend). The frontend will format/display it.
      experience: j.experience_level || j.experience || '',
    }))

    // Normalize paging metadata to camelCase for consumers
    const respPage = Number(data.page || 1)
    const respRpp = Number(data.resultsPerPage ?? data.results_per_page ?? 30);
    const respTotal = Number(data.totalResults ?? data.total_results ?? 0)

    return {
      // normalized
      page: respPage,
      resultsPerPage: respRpp,
      totalResults: respTotal,
      // raw backend payload (included for debugging if needed)
      _raw: data,
      results: mapped,
    };
  }

  return data;
}

// Get recommendations
export async function getRecommendations(resumeId, jobIds) {
  const res = await fetch(`${API_BASE}/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ resume_id: resumeId, job_ids: jobIds }),
  });

  const data = await handleResponse(res);

  // Map results array (if present) so callers get predictable keys
  if (data && Array.isArray(data.results)) {
    const mapped = data.results.map((r) => ({
      job_id: r.job_id,
      title: r.title,
      company: r.company,
      location: r.location,
      score: r.score,
      gaps: r.gaps || [],
      resume_bullets: r.resume_bullets || [],
      cover_letter: r.cover_letter || '',
      matched_skills: r.matched_skills || [],
      derived_resume_skills: r.derived_resume_skills || [],
    }))
    return { ...data, results: mapped };
  }

  return data;
}

// Match a single resume with a single job and return match metadata + generated cover letter
// payload: { resume_id?, resume_text?, name?, job_id?, job? }
export async function matchResumeToJob(payload = {}) {
  const res = await fetch(`${API_BASE}/match`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await handleResponse(res);
  // The backend `match` endpoint now returns only a numeric score. Normalize to { score } when present.
  if (data && typeof data.score !== 'undefined') {
    return { score: Number(data.score) };
  }
  return data;
}

// Fetch a single job by ID
export async function getJob(jobId) {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`)
  return handleResponse(res)
}

// Generate a cover letter for a resume/job pair. Returns { cover_letter, resume_bullets }
// payload: same shape as matchResumeToJob
export async function generateCoverLetter(payload) {
  const res = await fetch(`${API_BASE}/ai/cover-letter`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  // Normalize AI response so callers always get { cover_letter, resume_bullets }
  const data = await handleResponse(res);
  if (!data || typeof data !== 'object') return { cover_letter: '', resume_bullets: [] };

  return {
    cover_letter: data.cover_letter || data.coverLetter || data.cover || '',
    resume_bullets: Array.isArray(data.resume_bullets)
      ? data.resume_bullets
      : (Array.isArray(data.resumeBullets) ? data.resumeBullets : []),
    _raw: data,
  };
}

// Save a job for the current user
export async function saveJob(jobId, notes = '') {
  const token = getToken()
  const headers = { 'Content-Type': 'application/json' }
  if (token) headers['Authorization'] = `Bearer ${token}`
  const res = await fetch(`${API_BASE}/users/me/saved-jobs`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ job_id: jobId, notes }),
  });
  return handleResponse(res);
}

// Mark a job as applied for the current user
export async function applyJob(jobId) {
  const token = getToken()
  const headers = { 'Content-Type': 'application/json' }
  if (token) headers['Authorization'] = `Bearer ${token}`
  const res = await fetch(`${API_BASE}/users/me/applied-jobs`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ job_id: jobId }),
  });
  return handleResponse(res);
}

// Get applied jobs for the current user (returns an array of applied job objects)
export async function getAppliedJobs() {
  const token = getToken()
  const headers = { 'Content-Type': 'application/json' }
  if (token) headers['Authorization'] = `Bearer ${token}`
  const res = await fetch(`${API_BASE}/users/me/applied-jobs`, { headers, credentials: 'include' })
  return handleResponse(res)
}

// Register a new user
export async function authRegister(payload) {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return handleResponse(res)
}

// Login
export async function authLogin(payload) {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return handleResponse(res)
}

// Get saved jobs for current user
export async function getSavedJobs() {
  const token = getToken()
  const headers = { 'Content-Type': 'application/json' }
  if (token) headers['Authorization'] = `Bearer ${token}`
  const res = await fetch(`${API_BASE}/users/me/saved-jobs`, { headers })
  return handleResponse(res)
}

// Delete a saved job
export async function deleteSavedJob(jobId) {
  const token = getToken()
  const headers = { 'Content-Type': 'application/json' }
  if (token) headers['Authorization'] = `Bearer ${token}`
  const res = await fetch(`${API_BASE}/users/me/saved-jobs/${jobId}`, {
    method: 'DELETE',
    headers,
  })
  return handleResponse(res)
}

export async function deleteAppliedJob(jobId) {
  const token = getToken()
  const headers = { 'Content-Type': 'application/json' }
  if (token) headers['Authorization'] = `Bearer ${token}`
  const res = await fetch(`${API_BASE}/users/me/applied-jobs/${jobId}`, {
    method: 'DELETE',
    headers,
  })
  return handleResponse(res)
}