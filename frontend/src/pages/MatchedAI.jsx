import { useState, useEffect, useMemo, useCallback } from 'react'
import DOMPurify from 'dompurify'
import { Link } from 'react-router-dom'
import './jobAI.css'
import { searchJobs, getRecommendations, generateCoverLetter, saveJob, applyJob, getJob, getAppliedJobs } from '../api/api'

export default function MatchedAI() {
  useEffect(() => { document.title = 'AI Matched Jobs – jobhunter.ai' }, [])

  const [jobs, setJobs] = useState([])
  const [type, setType] = useState("")
  const [experience, setExp] = useState("")
  const [location, setLoc] = useState("")
  const [searchQuery, setSearchQuery] = useState('data analyst')

  const [salaryMin, setSalaryMin] = useState(30000)
  const [salaryMax, setSalaryMax] = useState(200000)
  const [page, setPage] = useState(1)
  // Backend enforces page size; keep a local constant but don't expose it in the UI
  const RESULTS_PER_PAGE = 30
  const [totalResults, setTotalResults] = useState(0)
  const [coverModalOpen, setCoverModalOpen] = useState(false)
  const [coverModalLoading, setCoverModalLoading] = useState(false)
  const [coverModalError, setCoverModalError] = useState('')
  const [coverModalContent, setCoverModalContent] = useState('')
  const [coverModalBullets, setCoverModalBullets] = useState([])
  const [selectedJob, setSelectedJob] = useState(null)
  const [jobModalOpen, setJobModalOpen] = useState(false)
  const [jobModalMessage, setJobModalMessage] = useState('')
  const [jobActionLoading, setJobActionLoading] = useState(false)
  const [filterAppliedButNoResults, setFilterAppliedButNoResults] = useState(false)
  const [unfilteredTotalResults, setUnfilteredTotalResults] = useState(null)
  const [copyStatus, setCopyStatus] = useState('')
  const [appliedJobIds, setAppliedJobIds] = useState(new Set())

  // Debug: Log when appliedJobIds changes
  useEffect(() => {
    console.log('appliedJobIds state updated:', Array.from(appliedJobIds))
  }, [appliedJobIds])

  // Helper function to check if a job is applied
  const isJobApplied = useCallback((jobId) => {
    return appliedJobIds.has(String(jobId))
  }, [appliedJobIds])

  const formatSalary = (val) => `$${Math.round((val || 0) / 1000)}k`

  const handleSalaryMinChange = (e) => {
    const newMin = parseInt(e.target.value || 0)
    if (newMin <= salaryMax) setSalaryMin(newMin)
  }
  const handleSalaryMaxChange = (e) => {
    const newMax = parseInt(e.target.value || 0)
    if (newMax >= salaryMin) setSalaryMax(newMax)
  }

  const normalizeExperience = (expLevel) => {
    if (!expLevel) return ''
    const s = String(expLevel).toLowerCase()
    if (s.includes('entry')) return 'Entry'
    if (s.includes('mid')) return 'Mid'
    if (s.includes('senior')) return 'Senior'
    return expLevel
  }

  const refreshAppliedJobs = useCallback(async () => {
    try {
      const data = await getAppliedJobs()
      console.log('refreshAppliedJobs - Raw data from server:', data)
      // server-provided applied rows should be authoritative
      const serverIds = new Set(
        (Array.isArray(data) ? data : [])
          .filter(row => row && row.job_id != null)
          .map(row => String(row.job_id))
      )
      console.log('refreshAppliedJobs - Setting appliedJobIds:', Array.from(serverIds))
      setAppliedJobIds(serverIds)
    } catch (err) {
      console.error('Failed to load applied jobs', err)
      setAppliedJobIds(new Set())
    }
  }, [])


  useEffect(() => {
    refreshAppliedJobs()
    
    // Listen for changes from AppliedJobs tab (when jobs are deleted)
    const handleAppliedJobsChange = (event) => {
      if (event.detail?.action === 'removed' && event.detail?.job_id) {
        // Remove from local applied set
        setAppliedJobIds((prev) => {
          const next = new Set(prev)
          next.delete(String(event.detail.job_id))
          return next
        })
        // Update jobs list to remove applied flag
        setJobs((prev) => prev.map((j) => 
          j.job_id === event.detail.job_id ? { ...j, applied: false } : j
        ))
      }
    }
    
    window.addEventListener('appliedJobsChanged', handleAppliedJobsChange)
    
    return () => {
      window.removeEventListener('appliedJobsChanged', handleAppliedJobsChange)
    }
  }, [refreshAppliedJobs])

  const fetchJobs = useCallback(async (opts = {}) => {
    const payload = {
      query: opts.query ?? searchQuery ?? 'data analyst',
      location: opts.location ?? location ?? '',
      page: opts.page ?? page,
      // Keep sending an explicit page size to the API for clarity, but this is not
      // controlled by the user in the UI anymore.
      resultsPerPage: opts.resultsPerPage ?? RESULTS_PER_PAGE,
      salaryMin: opts.salaryMin ?? salaryMin,
      salaryMax: opts.salaryMax ?? salaryMax,
      type: opts.type ?? type,
      experience: opts.experience ?? (experience || ''),
    }

    try {
      // Fetch search results
      const data = await searchJobs(payload)

      const mapped = (data.results || []).map((j) => ({
        ...j,
        experience: normalizeExperience(j.experience),
        skills: Array.isArray(j.skills) ? j.skills : (j.skills || []),
        salaryMin: Number.isFinite(j.salaryMin) ? j.salaryMin : null,
        salaryMax: Number.isFinite(j.salaryMax) ? j.salaryMax : null,
      }))

      // ALWAYS fetch fresh applied jobs from server to ensure button state is correct
      let appliedIds = new Set()
      try {
        const appliedData = await getAppliedJobs()
        const appliedArray = Array.isArray(appliedData) ? appliedData : []
        
        // Extract all applied job IDs and update the Set
        appliedIds = new Set(
          appliedArray
            .filter(row => row && row.job_id != null)
            .map(row => String(row.job_id))
        )
        
        // Update state immediately - this is the source of truth
        setAppliedJobIds(appliedIds)
        
        console.log('Applied job IDs from server:', Array.from(appliedIds))
      } catch (err) {
        console.error('Failed to fetch applied jobs in fetchJobs', err)
        // If fetch fails, keep current state
        appliedIds = appliedJobIds
      }

      // Mark jobs with applied flag based on server data
      const mergedJobs = mapped.map((job) => ({
        ...job,
        applied: appliedIds.has(String(job.job_id))
      }))

      setJobs(mergedJobs)

      if (data) {
        setPage(Number(data.page || 1));
        setTotalResults(Number((data.totalResults ?? data.total_results) || 0));
        const raw = data._raw || {};
        setFilterAppliedButNoResults(Boolean(raw.filter_applied_but_no_results));
        setUnfilteredTotalResults(raw.unfiltered_total_results ?? null);

      }

      const rid = parseInt(localStorage.getItem('resume_id') || '', 10)
      if (rid && mapped.length) {
        try {
          const rec = await getRecommendations(rid, mapped.map((m) => m.job_id))
          if (rec && Array.isArray(rec.results)) {
            const scoresById = {}
            rec.results.forEach((r) => { scoresById[r.job_id] = r })
            const merged = mapped.map((m) => ({ ...m, ...(scoresById[m.job_id] || {}) }))
            setJobs(merged)
          }
        } catch (err) {
          console.warn('Recommend call failed', err)
        }
      }

    } catch (err) {
      console.error('Failed to fetch jobs', err)
      setJobs([])
    }
  }, [searchQuery, location, salaryMin, salaryMax, type, experience, page])

  useEffect(() => { fetchJobs({ query: searchQuery, page }) }, [fetchJobs, page, searchQuery])

  // Debug: log selected job when modal opens to help diagnose missing fields
  useEffect(() => {
    if (jobModalOpen) {
      console.debug('Job modal opened, selectedJob=', selectedJob)
    }
  }, [jobModalOpen, selectedJob])

  // Generate cover letter for a selected job and show modal
  const handleGenerateCoverLetter = async (job) => {
    setSelectedJob(job)
    setCoverModalError('')
    setCoverModalContent('')
    setCoverModalBullets([])
    setCopyStatus('')
    setCoverModalOpen(true)
    setCoverModalLoading(true)

    try {
      const resumeId = Number(localStorage.getItem('resume_id') || 0) || undefined
      const payload = { resume_id: resumeId, job_id: job.job_id, job, persist: false, match_score: job.score ?? job.matchScore ?? undefined }
      const data = await generateCoverLetter(payload)
      setCoverModalContent(data.cover_letter || data.coverLetter || '')
      setCoverModalBullets(Array.isArray(data.resume_bullets) ? data.resume_bullets : (data.resumeBullets || []))
    } catch (err) {
      console.error('generateCoverLetter failed', err)
      setCoverModalError(String(err?.message || err || 'Failed to generate cover letter'))
    } finally {
      setCoverModalLoading(false)
    }
  }

  // Open job detail modal
  const openJobModal = (job) => {
    setJobModalMessage('')
    setJobModalOpen(true)
    // If job_id present, fetch fresh details from backend
    if (job && job.job_id) {
      getJob(job.job_id).then((data) => {
        // Preserve any known applied state from the current list
        const knownApplied = jobs.find((j) => j.job_id === data.job_id)?.applied
        // Also preserve the original raw provider payload (Adzuna) when available
        const rawFromList = job && job.raw ? job.raw : undefined
        const merged = rawFromList ? { ...data, raw: rawFromList } : data
        setSelectedJob(knownApplied ? { ...merged, applied: true } : merged)
      }).catch((err) => {
        console.error('getJob failed', err)
        // fall back to using provided job object
        setSelectedJob(job)
      })
    } else {
      setSelectedJob(job)
    }
  }

  const closeJobModal = () => {
    setJobModalOpen(false)
    setSelectedJob(null)
  }

  const handleSaveJob = async (job) => {
    setJobActionLoading(true)
    setJobModalMessage('')
    try {
      await saveJob(job.job_id)
      setJobModalMessage('Saved')
    } catch (err) {
      console.error('saveJob failed', err)
      setJobModalMessage(String(err?.message || 'Failed to save job'))
    } finally {
      setJobActionLoading(false)
    }
  }

  const handleApplyJob = async (job) => {
    if (!job || !job.job_id) return
    setJobActionLoading(true)
    setJobModalMessage('')
    try {
      await applyJob(job.job_id)
      setJobModalMessage('Marked as applied')
      
      // Update applied IDs set first (this controls button state)
      setAppliedJobIds((prev) => {
        const next = new Set(prev)
        next.add(String(job.job_id))
        return next
      })
      
      // Update local jobs state so UI reflects applied status
      setJobs((prev) => prev.map((j) => (j.job_id === job.job_id ? { ...j, applied: true } : j)))
      
      // If the job is currently selected in the modal, update that too
      setSelectedJob((s) => (s && s.job_id === job.job_id ? { ...s, applied: true } : s))
      
      // Notify other tabs/components that applied jobs changed
      window.dispatchEvent(new CustomEvent('appliedJobsChanged', { 
        detail: { action: 'added', job_id: job.job_id } 
      }))
      
      // Refresh applied jobs from server to ensure consistency
      await refreshAppliedJobs()
    } catch (err) {
      console.error('applyJob failed', err)
      setJobModalMessage(String(err?.message || 'Failed to apply to job'))
    } finally {
      setJobActionLoading(false)
    }
  }


  const filterJobs = useMemo(() => {
    return jobs.filter((j) => {
      const jMin = j.salaryMin || 0
      const jMax = j.salaryMax || 0
      const hasMin = jMin > 0
      const hasMax = jMax > 0
      const hasAnySalary = hasMin || hasMax

      const passSalary = !hasAnySalary
        || (hasMax && jMax >= salaryMin) && (!hasMin || jMin <= salaryMax)

      const jobExp = String(j.experience || '')
      const passExp = !experience || jobExp.toLowerCase() === experience.toLowerCase() || jobExp.toLowerCase() === 'unknown' || jobExp === ''

      // Location filtering is handled server-side via the Adzuna `where` param.
      // Adzuna's returned `location.display_name` may contain neighborhoods or
      // county names rather than the literal city (e.g. "Tarrytown, Travis County").
      // To avoid accidentally hiding valid server-returned matches, do not
      // re-filter by location in the client — assume the backend already applied it.
      const passLoc = true

      return passSalary && passExp && passLoc
    })
  }, [jobs, experience, salaryMin, salaryMax, location])

  const handleCopyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(coverModalContent || '')
      setCopyStatus('copied')
      setTimeout(() => setCopyStatus(''), 1500)
    } catch {
      setCopyStatus('error')
      setTimeout(() => setCopyStatus(''), 1500)
    }
  }

  const handleDownloadTxt = () => {
    const text = coverModalContent || ''
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const safeCompany = (selectedJob?.company || 'Company').replace(/[^\w.-]+/g, '_')
    const safeTitle = (selectedJob?.title || 'Role').replace(/[^\w.-]+/g, '_')
    const a = document.createElement('a')
    a.href = url
    a.download = `${safeCompany}-${safeTitle}-cover-letter.txt`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
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
                        d="m2.25 12 8.954-8.955c.44-.439 1.152-.439 1.591 0L21.75 12M4.5 9.75v10.125c0 .621.504 1.125 1.125 1.125H9.75
v-4.875c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125V21h4.125c.621 0 1.125-.504 1.125-1.125V9.75M8.25 21h8.25"/>
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
          <h1>AI Matched Jobs</h1>
          <p className="text-muted-foreground">Jobs matched to your resume and preferences.</p>
          <p className="text-muted-foreground">Use filters to help narrow results.</p>

          <form id="filtersForm" className="filters" onSubmit={(e) => { e.preventDefault(); setPage(1); fetchJobs({ query: searchQuery, location, page: 1, salaryMin, salaryMax, type, experience }); }} style={{
            width: '100vw',
            maxWidth: '100%',
            margin: '18px 0',
            display: 'flex',
            gap: '12px',
            flexWrap: 'wrap',
            justifyContent: 'center',
            alignItems: 'center',
            boxSizing: 'border-box'
          }}>
            <input
              type="search"
              name="q"
              placeholder="Search job title, company or skill"
              className="filter-input"
              style={{ flex: 1, minWidth: '220px' }}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />

            <input
              type="search"
              name="location"
              id="locationSearch"
              className="filter-input"
              placeholder="Location (city, zip or Remote)"
              list="location-list"
              style={{ minWidth: '180px' }}
              value={location}
              onChange={(e) => setLoc(e.target.value)}
            />
            <datalist id="location-list">
              <option value="Remote" />
              <option value="San Francisco, CA" />
              <option value="New York, NY" />
              <option value="Austin, TX" />
            </datalist>

            <select name="type" className="filter-input" style={{ minWidth: '150px' }} value={type} onChange={(e) => { const v = e.target.value; setType(v); setPage(1); fetchJobs({ query: searchQuery, location, page: 1, resultsPerPage: RESULTS_PER_PAGE, salaryMin, salaryMax, type: v, experience }); }}>
              <option value="">Any type</option>
              <option>Full-time</option>
              <option>Part-time</option>
              <option>Contract</option>
              <option>Remote</option>
              <option>Hybrid</option>
              <option>Internship</option>
            </select>

            <select name="exp" className="filter-input" style={{ minWidth: '140px' }} value={experience} onChange={(e) => { const v = e.target.value; setExp(v); setPage(1); fetchJobs({ query: searchQuery, location, page: 1, resultsPerPage: RESULTS_PER_PAGE, salaryMin, salaryMax, type, experience: v }); }}>
              <option value="">Experience</option>
              <option>Entry</option>
              <option>Mid</option>
              <option>Senior</option>
            </select>

            <div className="range-group" style={{
              width: '100%',
              maxWidth: '420px',
              display: 'flex',
              flexDirection: 'column',
              gap: '12px',
              marginTop: '6px'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px' }}>
                <label style={{ margin: 0, fontSize: '14px', color: 'var(--muted)', fontWeight: 500 }}>Salary Range</label>
                <div style={{ fontSize: '13px', color: 'var(--muted)', fontWeight: 500 }}>
                  <span id="salaryLabel">{formatSalary(salaryMin)} - {formatSalary(salaryMax)}</span>
                </div>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <label htmlFor="salaryMin" style={{ fontSize: '12px', color: 'var(--muted)', minWidth: '30px' }}>Min</label>
                  <input
                    type="range"
                    id="salaryMin"
                    name="salaryMin"
                    min="10000"
                    max="700000"
                    step="5000"
                    value={salaryMin}
                    onChange={handleSalaryMinChange}
                    className="range-input"
                    aria-label="Minimum salary"
                    style={{
                      width: '100%'
                    }}
                  />
                  <span style={{ fontSize: '12px', color: 'var(--muted)', minWidth: '50px', textAlign: 'right' }}>
                    {formatSalary(salaryMin)}
                  </span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <label htmlFor="salaryMax" style={{ fontSize: '12px', color: 'var(--muted)', minWidth: '30px' }}>Max</label>
                  <input
                    type="range"
                    id="salaryMax"
                    name="salaryMax"
                    min="10000"
                    max="700000"
                    step="5000"
                    value={salaryMax}
                    onChange={handleSalaryMaxChange}
                    className="range-input"
                    aria-label="Maximum salary"
                    style={{
                      width: '100%'
                    }}
                  />
                  <span style={{ fontSize: '12px', color: 'var(--muted)', minWidth: '50px', textAlign: 'right' }}>
                    {formatSalary(salaryMax)}
                  </span>
                </div>
              </div>
            </div>

            <button type="submit" style={{
              padding: '10px 14px',
              borderRadius: '10px',
              background: 'var(--brand)',
              border: 0,
              color: '#07120c',
              fontWeight: 600,
              minWidth: '96px'
            }}>
              Apply Filters
            </button>
            {/* results-per-page control removed from UI (backend controls page size) */}
          </form>

          {/* Banner when backend indicates filters produced no filtered results */}
          {filterAppliedButNoResults ? (
            <div style={{ maxWidth: '820px', margin: '8px auto', padding: 12, borderRadius: 8, background: '#fff3cd', color: '#664d03', display: 'flex', gap: 12, alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <strong>No exact matches for your filters.</strong>
                <div style={{ fontSize: 13, marginTop: 4 }}>
                  We couldn't find jobs that matched your filters exactly. There {unfilteredTotalResults === 1 ? 'is' : 'are'} {unfilteredTotalResults ?? 'some'} job{(unfilteredTotalResults === 1 ? '' : 's')} without the selected filters.
                </div>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button onClick={() => { setType(''); setExp(''); setPage(1); setFilterAppliedButNoResults(false); fetchJobs({ query: searchQuery, location, page: 1, resultsPerPage: RESULTS_PER_PAGE, salaryMin, salaryMax, type: '', experience: '' }); }} style={{ padding: '8px 10px', borderRadius: 8, border: '1px solid #c69500', background: '#fff', cursor: 'pointer' }}>Remove Type & Experience Filters</button>
                <button onClick={() => { setType(''); setExp(''); setPage(1); setFilterAppliedButNoResults(false); fetchJobs({ query: searchQuery, location, page: 1, resultsPerPage: RESULTS_PER_PAGE, salaryMin, salaryMax, type: '', experience: '' }); }} style={{ padding: '8px 10px', borderRadius: 8, border: '1px solid #c69500', background: '#c69500', color: '#fff', cursor: 'pointer' }}>Show {unfilteredTotalResults ?? 'All'} Jobs</button>
              </div>
            </div>
          ) : null}

          <section id="jobsList" className="jobs" style={{
            width: '100%',
            maxWidth: '820px',
            display: 'grid',
            gap: '14px',
            marginTop: '8px'
          }}>
            {filterJobs.map((job,index) => (
              <JobCard
                key={job.job_id || index}
                job={job}
                formatSalary={formatSalary}
                onView={() => openJobModal(job)}
                onSave={() => handleSaveJob(job)}
                onApply={() => handleApplyJob(job)}
                onGenerateCover={() => handleGenerateCoverLetter(job)}
                isApplied={appliedJobIds.has(String(job.job_id))}
              />
            ))}
          </section>

          {/* Pagination footer: show server total and filtered count when client-side filters are active */}
          {(() => {
            const filteredCount = filterJobs.length || 0
            // Prefer server-provided totalResults when available; fall back to client-side filtered count
            const hasServerTotal = Boolean(totalResults && Number(totalResults) > 0)
      const pagesFromServer = Math.max(1, Math.ceil((hasServerTotal ? Number(totalResults) : (jobs.length || 0)) / RESULTS_PER_PAGE))
      const pagesFromFiltered = Math.max(1, Math.ceil(filteredCount / RESULTS_PER_PAGE))
            const totalPages = hasServerTotal ? pagesFromServer : pagesFromFiltered

            const prevDisabled = page <= 1
            const nextDisabled = page >= totalPages

            return (
              <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 12, marginTop: 12 }}>
                <button onClick={() => { if (!prevDisabled) { const np = page - 1; setPage(np); fetchJobs({ page: np, query: searchQuery, location, salaryMin, salaryMax, type, experience }); } }} disabled={prevDisabled}>Prev</button>
                <div style={{ fontSize: 14, color: 'var(--muted)' }}>
                  Page {page} of {totalPages}
                  {hasServerTotal ? ` — ${totalResults} total results` : ` — showing ${filteredCount} result${filteredCount === 1 ? '' : 's'}`}
                </div>
                <button onClick={() => { if (!nextDisabled) { const np = page + 1; setPage(np); fetchJobs({ page: np, query: searchQuery, location, salaryMin, salaryMax, type, experience }); } }} disabled={nextDisabled}>Next</button>
              </div>
            )
          })()}
    {/* Cover letter modal */}
            {/* Job modal (View) */}
            {jobModalOpen ? (
              <div role="dialog" aria-modal="true" className="cover-modal-overlay" style={{position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 9999}} onClick={() => { if (!jobActionLoading) closeJobModal() }}>
                <div className="cover-modal" style={{ background: '#fff', color: '#0b0b0b', maxWidth: 800, width: '95%', maxHeight: '90vh', overflow: 'auto', borderRadius: 8, padding: 18 }} onClick={(e) => e.stopPropagation()}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
                    <h3 style={{ margin: 0 }}>{selectedJob?.title} — {selectedJob?.company}</h3>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                      <button onClick={() => { if (!jobActionLoading) closeJobModal() }}>Close</button>
                    </div>
                  </div>
                  <div style={{ marginTop: 12 }}>
                    <div style={{ color: 'var(--muted)', lineHeight: 1.4 }}>
                      {(() => {
                        const r = selectedJob?.raw || {};
                        const loc = selectedJob?.location || (r.location && (r.location.display_name || r.location.area)) || '';

                        const created = r.created || r.created_at || r.date || r.created_date || selectedJob?.created || selectedJob?.posted_at;
                        let days = null;
                        if (created) {
                          const d = new Date(created);
                          if (!isNaN(d)) {
                            days = Math.floor((Date.now() - d.getTime()) / (1000 * 60 * 60 * 24));
                          }
                        }

                        const timeVal = r.Time ?? selectedJob?.time ?? selectedJob?.posted_days;

                        return (
                          <div>
                            <div><strong>Location:</strong> {loc || 'N/A'}</div>
                            <div>
                              <strong>Days Posted:</strong> {days !== null ? `${days} day${days === 1 ? '' : 's'}` : (timeVal || timeVal === 0 ? String(timeVal) : 'N/A')}{selectedJob?.type ? ` • ${selectedJob.type}` : ''}
                            </div>
                          </div>
                        )
                      })()}
                    </div>
                    <div style={{ marginTop: 12 }}>
                      {selectedJob?.full_description || selectedJob?.description ? (
                        <div style={{ lineHeight: 1.5 }} dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(selectedJob?.full_description || selectedJob?.description || '') }} />
                      ) : null}
                    </div>
                    <div style={{ marginTop: 12, display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
                      <button onClick={() => handleSaveJob(selectedJob)} disabled={jobActionLoading}>Save</button>
                      <button
                        onClick={() => handleApplyJob(selectedJob)}
                        disabled={
                          jobActionLoading ||
                          (selectedJob && appliedJobIds.has(String(selectedJob.job_id)))
                        }
                      >
                        {selectedJob && appliedJobIds.has(String(selectedJob.job_id))
                          ? 'Confirmed'
                          : 'Confirm Applied'}
                      </button>
                      <button onClick={() => { if (!jobActionLoading) { closeJobModal(); handleGenerateCoverLetter(selectedJob); } }}>Cover Letter</button>
                      <a href={selectedJob?.url || selectedJob?.url || '#'} target="_blank" rel="noreferrer"><button>Open Original</button></a>
                    </div>
                    {jobModalMessage ? (<div style={{ marginTop: 8, color: jobModalMessage.startsWith('Failed') ? 'crimson' : 'green' }}>{jobModalMessage}</div>) : null}
                  </div>
                </div>
              </div>
            ) : null}

            {coverModalOpen ? (
              <div role="dialog" aria-modal="true" className="cover-modal-overlay" style={{
                position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 9999
              }} onClick={() => { if (!coverModalLoading) setCoverModalOpen(false) }}>
                <div className="cover-modal" style={{ background: '#fff', color: '#0b0b0b', maxWidth: 800, width: '95%', maxHeight: '90vh', overflow: 'auto', borderRadius: 8, padding: 18 }} onClick={(e) => e.stopPropagation()}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
                    <h3 style={{ margin: 0 }}>Cover Letter Preview</h3>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                      <button
                        onClick={handleCopyToClipboard}
                        disabled={!coverModalContent}
                        aria-live="polite"
                        title="Copy to Clipboard"
                      >
                        {copyStatus === 'copied' ? 'Copied!' : copyStatus === 'error' ? 'Copy failed' : 'Copy to Clipboard'}
                      </button>

                      <button
                        onClick={handleDownloadTxt}
                        disabled={!coverModalContent}
                        title="Download .txt"
                      >
                        Download .txt
                      </button>

                      <button onClick={() => { if (!coverModalLoading) setCoverModalOpen(false) }}>
                        Close
                      </button>
                    </div>
                  </div>

                  <div style={{ marginTop: 12 }}>
                    {coverModalLoading ? (
                      <div style={{ padding: 24 }}>Generating cover letter…</div>
                    ) : coverModalError ? (
                      <div style={{ color: 'crimson' }}>Error: {coverModalError}</div>
                    ) : (
                      <>
                        {coverModalContent ? (
                          <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.5 }}>{coverModalContent}</div>
                        ) : null}

                        {coverModalBullets && coverModalBullets.length ? (
                          <div style={{ marginTop: 16 }}>
                            <h4>Suggested resume bullets</h4>
                            <ul>
                              {coverModalBullets.map((b, i) => <li key={i}>{b}</li>)}
                            </ul>
                          </div>
                        ) : null}
                      </>
                    )}
                  </div>
                </div>
              </div>
            ) : null}

          </main>
        </div>
      </div>
  )
}

function JobCard({job, formatSalary, onView, onSave, onApply, onGenerateCover, isApplied}) {
  console.log('JobCard render:', {
    job_id: job.job_id,
    title: job.title,
    isApplied: isApplied,
    applied_flag: job.applied
  })
  return (
    <div data-slot="card" className="bg-card text-card-foreground border" style={{padding: '12px 16px'}}>
      <div data-slot="card-header" style={{textAlign: 'left'}}>
        <div style={{display:'flex', justifyContent:'space-between', gap:12, alignItems:'flex-start'}}>
          <div style={{minWidth:0}}>
            <h4 data-slot="card-title" style={{margin:0}}>{job.title}</h4>
            <p data-slot="card-description">{job.company} {job.type ? `• ${job.type}` : ''}</p>

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
            <div><strong>{job.score ?? job.matchScore ?? 0}%</strong></div>
          </div>
        </div>
      </div>

  <div data-slot="card-content">
        <div style={{ display:'flex', gap:8, justifyContent:'flex-end', paddingTop:6}}>
          <button onClick={() => onView && onView()}>View</button>
        <button onClick={() => onSave && onSave()}>Save</button>
        <button
          onClick={() => !isApplied && onApply && onApply()}
          disabled={isApplied}
        >
          {isApplied ? 'Confirmed' : 'Confirm Applied'}
        </button>
        <button onClick={() => onGenerateCover && onGenerateCover()}>Cover Letter</button>
        </div>
      </div>
    </div>
  )
}
