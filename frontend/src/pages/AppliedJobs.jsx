import { useState, useEffect} from 'react'
import { Link } from 'react-router-dom'
import { getAppliedJobs } from '../api/api'
import { deleteAppliedJob } from '../api/api'
import './jobAI.css'

export default function AppliedJobs() {
  useEffect(() => { document.title = 'Applied Jobs – jobhunter.ai' }, [])

  useEffect(() => {
    async function loadAppliedJobs() {
      try {
        const data = await getAppliedJobs()
        setJobs(Array.isArray(data) ? data : [])
      } catch (err) {
        console.error('Error fetching applied jobs', err)
        setJobs([])
      }
    }

    loadAppliedJobs()
    
    // Listen for changes from MatchedAI tab (when jobs are added)
    const handleAppliedJobsChange = (event) => {
      if (event.detail?.action === 'added') {
        // Refresh the list when a job is added from another tab
        loadAppliedJobs()
      }
    }
    
    window.addEventListener('appliedJobsChanged', handleAppliedJobsChange)
    
    return () => {
      window.removeEventListener('appliedJobsChanged', handleAppliedJobsChange)
    }
  }, [])

  const [jobs, setJobs] = useState([]);
  
  const handleRemove = async (job) => {
    if (!job || !job.job_id) return
    try {
      await deleteAppliedJob(job.job_id)
      setJobs(prev => prev.filter(j => j.job_id !== job.job_id))
      
      // Notify other tabs/components that applied jobs changed
      window.dispatchEvent(new CustomEvent('appliedJobsChanged', { 
        detail: { action: 'removed', job_id: job.job_id } 
      }))
    } catch (err) {
      console.error('deleteAppliedJob failed', err)
    }
  }

  const formatSalary = (val) => `$${Math.round(val / 1000)}k`

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
          <h1>Applied Jobs</h1>
          <p className="text-muted-foreground">Track jobs you've applied to.</p>

          <section id="appliedList" className="jobs" style={{ width: '100vw', maxWidth: '100%', display: 'grid', gap: '14px', marginTop: '18px', boxSizing: 'border-box' }}>
            {jobs.length === 0 && (
              <div id="appliedEmpty" style={{
              padding: '36px',
              borderRadius: '12px',
              border: '1px dashed var(--border)',
              textAlign: 'center',
              color: 'var(--muted)'
            }}>
              <p style={{ margin: '0 0 8px' }}>You haven't applied to any jobs yet.</p>
              <p style={{ margin: '0' }}>
                <Link to="/matchedAI" style={{ color: 'var(--brand)', textDecoration: 'none' }}>
                  Browse matched jobs →
                </Link>
              </p>
            </div>
          )}
          {/* Populate job cards */}
            {jobs.map((job,index) => (
              <JobCard
                key={job.job_id || index}
                job={job}
                formatSalary={formatSalary}
                onRemove={() => handleRemove(job)}
              />
            ))}
          </section>
        </main>
      </div>
    </div>
  )
}

function JobCard({job, formatSalary, onRemove}) {
  const title = job.title
  const company = job.company_name || job.company
  const location = job.location
  const salaryText =
    job.salary_range ||
    (job.salaryMin && job.salaryMax
      ? `${formatSalary(job.salaryMin)} - ${formatSalary(job.salaryMax)}`
      : 'Not specified')

  return (
    <div data-slot="card" className="bg-card text-card-foreground border" style={{padding: '12px 16px'}}>
      <div data-slot="card-header" style={{textAlign: 'left'}}>
        <div style={{display:'flex', justifyContent:'space-between', gap:12, alignItems:'flex-start'}}>
          <div style={{minWidth:0}}>
            <h4 data-slot="card-title" style={{margin:0}}>{job.title}</h4>
            <p data-slot="card-description">{job.company} · {job.location} {job.type ? `• ${job.type}` : ''}</p>

            <div className="text-muted-foreground" style={{marginTop:6}}>
              <div className="skills">Skills: {Array.isArray(job.skills) ? job.skills.join(', ') : (job.skills || '')}</div>
              <div className="salary-range">Salary: {formatSalary(job.salaryMin)} - {formatSalary(job.salaryMax)}</div>
              <div className='job-experience'>Experience: {job.experience}</div>
            </div>
          </div>

          <div style={{textAlign:'right', minWidth:72}}>
            <div className="text-muted-foreground" style={{marginBottom:6}}>Match</div>
            <div><strong>{job.matchScore ?? 0}%</strong></div>
          </div>
        </div>
      </div>

      <div data-slot="card-content">
        <div style={{ display:'flex', gap:8, justifyContent:'flex-end', paddingTop:6}}>
          <a
            href={job.url || '#'}
            target="_blank"
            rel="noopener noreferrer"
          >
            <button>View</button>
          </a>
            <button onClick={() => onRemove && onRemove()}>
              Remove from Applied
            </button>
        </div>
      </div>
    </div>
  )
}