import { useState, useEffect} from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'
import './jobAI.css'
import { getSavedJobs, deleteSavedJob, applyJob } from '../api/api'

const formatSalary = (val) => `$${Math.round((val || 0) / 1000)}k`

const getSalaryText = (job) => {
  // Prefer server-provided salary_range string when available
  if (job?.salary_range) {
    // Try to parse ranges like "70000-90000" into $70k - $90k
    try {
      const parts = String(job.salary_range).split(/[-–—]/).map(p => p.replace(/[^0-9]/g, '').trim()).filter(Boolean)
      if (parts.length >= 2) {
        const a = Number(parts[0]) || 0
        const b = Number(parts[1]) || 0
        return `${formatSalary(a)} - ${formatSalary(b)}`
      }
    } catch (e) {
      // fall back to raw string
    }
    return String(job.salary_range)
  }

  // Fallback to numeric mins/maxes
  if (job?.salaryMin || job?.salaryMax) {
    const a = job.salaryMin || 0
    const b = job.salaryMax || 0
    return `${formatSalary(a)} - ${formatSalary(b)}`
  }

  return 'Not specified'
}

export default function SavedJobs() {
  useEffect(() => { document.title = 'Saved Jobs – jobhunter.ai' }, [])

  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const { token } = useAuth() || {}
  const navigate = useNavigate()
  const [unauthorized, setUnauthorized] = useState(false)

  

  const fetchSaved = async () => {
    setLoading(true)
    setError('')
    if (!token) {
      setUnauthorized(true)
      setJobs([])
      setLoading(false)
      return
    }
    try {
      const data = await getSavedJobs()
      // backend returns an array of rows; normalize if needed
      setJobs(Array.isArray(data) ? data : (data.results || []))
    } catch (err) {
      console.error('getSavedJobs failed', err)
      setError(String(err?.message || err || 'Failed to load saved jobs'))
      setJobs([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { setUnauthorized(false); fetchSaved() }, [token])

  const handleRemove = async (job) => {
    if (!job || !job.job_id) return
    try {
      await deleteSavedJob(job.job_id)
      // refresh list
      await fetchSaved()
    } catch (err) {
      console.error('deleteSavedJob failed', err)
      setError(String(err?.message || err || 'Failed to remove saved job'))
    }
  }

  const handleApply = async (job) => {
    if (!job || !job.job_id) return
    try {
      await applyJob(job.job_id)
      // Optionally remove from saved after applying
      try { await deleteSavedJob(job.job_id) } catch (e) { /* ignore */ }
      await fetchSaved()
    } catch (err) {
      console.error('applyJob failed', err)
      setError(String(err?.message || err || 'Failed to apply to job'))
    }
  }

  return (
    <div>
      <div>
        <a href="/" className="brand-right" aria-label="jobhunter.ai home">
          <span className="brand-dot-right" aria-hidden="true"></span>
          <span>jobhunter.ai</span>
        </a>
      </div>

      <div className="app-row">
        <nav className="sidebar">
          <ul>
            <li>
              <Link to="/" className="nav-link">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
                     strokeWidth="1.5" stroke="currentColor" className="nav-icon">
                  <path strokeLinecap="round" strokeLinejoin="round"
                        d="m2.25 12 8.954-8.955c.44-.439 1.152-.439 1.591 0L21.75 12M4.5 9.75v10.125c0 .621.504 1.125 1.125 1.125H9.75v-4.875c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125V21h4.125c.621 0 1.125-.504 1.125-1.125V9.75M8.25 21h8.25"/>
                </svg>
                Home
              </Link>
            </li>
            <li><Link to="/resume_upload">Upload Resume</Link></li>
            <li><Link to="/matchedAI">AI Matched Jobs</Link></li>
            <li><Link to="/savedJobs">Saved Jobs</Link></li>
            <li><Link to="/appliedJobs">Applied Jobs</Link></li>
            <li><Link to="/profile">Profile</Link></li>
          </ul>
        </nav>

        <main className="container">
          <h1>Saved Jobs</h1>
          <p className="text-muted-foreground">Jobs you marked to review or apply later.</p>

          {error ? <div style={{ color: 'crimson', marginBottom: 8 }}>{error}</div> : null}

          {unauthorized ? (
            <div style={{ padding: 24, borderRadius: 8, border: '1px solid var(--border)', textAlign: 'center' }}>
              <h3 style={{ marginTop: 0 }}>Sign in to view your saved jobs</h3>
              <p style={{ color: 'var(--muted)' }}>Saved jobs are persisted to your account. Please sign in or create an account to see them.</p>
              <div style={{ display: 'flex', gap: 8, justifyContent: 'center', marginTop: 12 }}>
                <button onClick={() => navigate('/signIn')} style={{ padding: '8px 12px' }}>Sign In</button>
                <button onClick={() => navigate('/createNewUser')} style={{ padding: '8px 12px' }}>Create Account</button>
              </div>
            </div>
          ) : (
            <section id="savedList" className="jobs" style={{ width: '100vw', maxWidth: '100%', display: 'grid', gap: '14px', marginTop: '18px', boxSizing: 'border-box' }}>
              {loading && (<div>Loading…</div>)}
              {!loading && jobs.length === 0 && (
                <div id="savedEmpty" style={{ padding: '36px', borderRadius: '12px', border: '1px dashed var(--border)', textAlign: 'center', color: 'var(--muted)' }}>
                  <p style={{ margin: '0 0 8px' }}>No saved jobs yet.</p>
                  <p style={{ margin: '0' }}>
                    <Link to="/matchedAI" style={{ color: 'var(--brand)', textDecoration: 'none' }}>
                      Browse matched jobs →
                    </Link>
                  </p>
                </div>
              )}

              {/* Populate job cards */}
              {jobs.map((job,index) => (
                <JobCard key={job.job_id || index} job={job} formatSalary={formatSalary} onRemove={() => handleRemove(job)} onApply={() => handleApply(job)} />
              ))}
            </section>
          )}
        </main>
      </div>
    </div>
  )
}

function JobCard({job, formatSalary, onRemove, onApply}) {
  return (
    <div data-slot="card" className="bg-card text-card-foreground border" style={{padding: '12px 16px'}}>
      <div data-slot="card-header" style={{textAlign: 'left'}}>
        <div style={{display:'flex', justifyContent:'space-between', gap:12, alignItems:'flex-start'}}>
          <div style={{minWidth:0}}>
            <h4 data-slot="card-title" style={{margin:0}}>{job.title}</h4>
            <p data-slot="card-description">{job.company || job.company_name} {job.type ? `• ${job.type}` : ''}</p>

            <div className="text-muted-foreground" style={{marginTop:6}}>
              {(() => {
                const r = job?.raw || {};
                const loc = job?.location || (r.location && (r.location.display_name || r.location.area)) || '';
                const created = r.created || r.created_at || r.date || r.created_date || job?.created || job?.posted_at;
                let days = null;
                if (created) {
                  const d = new Date(created);
                  if (!isNaN(d)) {
                    days = Math.floor((Date.now() - d.getTime()) / (1000 * 60 * 60 * 24));
                  }
                }
                const timeVal = r.Time ?? job?.time ?? job?.posted_days;
                return (
                  <>
                    <div className="location"><strong>Location:</strong> {loc || 'N/A'}</div>
                    <div className="days-posted"><strong>Days Posted:</strong> {days !== null ? `${days} day${days === 1 ? '' : 's'}` : (timeVal || timeVal === 0 ? String(timeVal) : 'N/A')}</div>
                  </>
                )
              })()}

              <div className="salary-range"><strong>Salary:</strong> {formatSalary(job.salaryMin)} - {formatSalary(job.salaryMax)}</div>
              <div className='job-experience'><strong>Experience:</strong> {job.experience}</div>
            </div>
          </div>

          <div style={{textAlign:'right', minWidth:72}}>
            <div className="text-muted-foreground" style={{marginBottom:6}}>Match</div>
            <div><strong>{job.score ?? job.matchScore ?? job.match_score ?? 0}%</strong></div>
          </div>
        </div>
      </div>

      <div data-slot="card-content">
        <div style={{ display:'flex', gap:8, justifyContent:'flex-end', paddingTop:6}}>
          <a href={job.url || '#'}><button>View</button></a>
          <button onClick={() => onRemove && onRemove()}>Remove</button>
          <button onClick={() => onApply && onApply()}>Confirm Applied</button>
        </div>
      </div>
    </div>
  )
}