import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'
import './jobAI.css'

export default function Profile() {
  const DEV_HEADERS = { 'X-User-Id': '1' }
  const { logout } = useAuth()

  const [editing, setEditing] = useState(false)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [err, setErr] = useState('')
  const [resumeMeta, setResumeMeta] = useState(null)

  const [profileData, setProfileData] = useState({
    name: '—',
    title: '—',
    email: '—',
    location: '—',
    phone: '—',
    desired_salary: ''
  })

  // server-saved vs in-progress job preferences
  const [savedPrefs, setSavedPrefs] = useState({
    pref_location: '',
    pref_type: '',
    pref_salary: ''
  })
  const [draftPrefs, setDraftPrefs] = useState({
    pref_location: '',
    pref_type: '',
    pref_salary: ''
  })
  const [prefsEditing, setPrefsEditing] = useState(false)

  const handlePrefsEditToggle = () => {
    // reset draft to saved when entering edit mode
    setDraftPrefs(savedPrefs)
    setPrefsEditing(v => !v)
  }

  useEffect(() => { document.title = 'Profile – jobhunter.ai' }, [])

  useEffect(() => {
    (async () => {
      setLoading(true)
      setErr('')
      try {
        // Profile
        const res = await fetch('/api/users/me', { headers: DEV_HEADERS })
        if (!res.ok) throw new Error(await res.text())
        const data = await res.json()

        setProfileData({
          name: data.name ?? '—',
          title: data.title ?? '—',
          email: data.email ?? '—',
          location: data.location ?? '—',
          phone: data.phone ?? '—',
          desired_salary: data.desired_salary ?? ''
        })

        const initialSaved = {
          pref_location: data.pref_locations ?? '',
          pref_type: data.pref_type ?? '',
          pref_salary: data.pref_salary ?? ''
        }
        setSavedPrefs(initialSaved)
        setDraftPrefs(initialSaved)

        // Resume meta
        try {
          const r = await fetch('/api/resumes', { headers: DEV_HEADERS })
          setResumeMeta(r.ok ? await r.json() : null)
        } catch {
          setResumeMeta(null)
        }
      } catch (e) {
        console.error('Failed to load profile:', e)
        setErr('Failed to load profile.')
      } finally {
        setLoading(false)
      }
    })()
  }, [])

  const handleEditToggle = () => setEditing(v => !v)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setErr('')
    setSaving(true)

    const formData = new FormData(e.target)
    const next = {
      name: formData.get('name') || '',
      title: formData.get('title') || '',
      email: formData.get('email') || '',
      location: formData.get('location') || '',
      phone: formData.get('phone') || '',
      desired_salary: formData.get('desired_salary') || ''
    }

    try {
      const res = await fetch('/api/users/me/profile', {
        method: 'PUT',
        headers: { ...DEV_HEADERS, 'Content-Type': 'application/json' },
        body: JSON.stringify(next)
      })
      if (!res.ok) throw new Error(await res.text())

      // Update UI after success
      setProfileData(p => ({ ...p, ...next }))
      setEditing(false)
      console.log('Profile saved successfully.')
    } catch (e) {
      console.error('Save failed:', e)
      setErr('Save failed.')
    } finally {
      setSaving(false)
    }
  }

  const handlePrefsSubmit = async (e) => {
    e.preventDefault()
    setErr('')
    setSaving(true)

    const pref_location_raw = draftPrefs.pref_location.trim()
    const pref_type = draftPrefs.pref_type.trim()
    const pref_salary_raw = draftPrefs.pref_salary.trim()
    const pref_salary = pref_salary_raw.replace(/[^\d.\-]/g, '')

    try {
      const res = await fetch('/api/users/me/profile', {
        method: 'PUT',
        headers: { ...DEV_HEADERS, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pref_location: pref_location_raw,
          pref_type,
          pref_salary,
          job_preferences: {
            locations: pref_location_raw
              ? pref_location_raw.split(',').map(s => s.trim()).filter(Boolean)
              : [],
            type: pref_type || ''
          }
        })
      })
      if (!res.ok) throw new Error(await res.text())

      console.log('Preferences saved.')

      const updated = {
        pref_location: pref_location_raw,
        pref_type,
        pref_salary
      }
      setSavedPrefs(updated)
      setDraftPrefs(updated)
      setPrefsEditing(false)
    } catch (e) {
      console.error('Save prefs failed:', e)
      setErr('Saving preferences failed.')
    } finally {
      setSaving(false)
    }
  }

  const handleLogout = () => {
    // Clear any local data
    localStorage.removeItem('resume_id')
    // Call the auth context logout which clears token and redirects
    logout()
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
          <div data-slot="card" className="bg-card text-card-foreground flex flex-col gap-6 rounded-xl border" style={{ width: '100vw', maxWidth: '100%', boxSizing: 'border-box' }}>
            <div data-slot="card-header" className="px-6 pt-6 text-center">
              <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-4" style={{ width: '80px', height: '80px' }}>
                <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                  <circle cx="12" cy="7" r="4"></circle>
                </svg>
              </div>

              <h4 data-slot="card-title" className="leading-none" id="profileName">{profileData.name}</h4>
              <p data-slot="card-description" className="text-muted-foreground" id="profileRole">{profileData.title}</p>
            </div>

            <div data-slot="card-content" className="px-6 pb-6 space-y-4" style={{ width: '100%', maxWidth: '680px', margin: '0 auto' }}>
              {loading && <div className="text-muted-foreground">Loading profile…</div>}
              {err && <div style={{ color: 'crimson' }}>{err}</div>}

              {/* Personal information */}
              <section aria-labelledby="personal-info">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '12px' }}>
                  <h3 id="personal-info" style={{ margin: 0, fontWeight: 700 }}>Personal information</h3>
                  <div>
                    <button
                      id="editProfileBtn"
                      onClick={handleEditToggle}
                      disabled={loading || saving}
                      style={{ padding: '8px 10px', borderRadius: '8px', border: '1px solid var(--border)', background: 'transparent', color: 'var(--ink)' }}
                    >
                      {editing ? 'Cancel' : 'Edit'}
                    </button>
                  </div>
                </div>

                {!editing && (
                  <div id="profileView" style={{ marginTop: '12px', display: 'grid', gap: '10px' }}>
                    <div style={{ display: 'flex', gap: '12px', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' }}>
                      <div style={{ textAlign: 'left', minWidth: '160px', flex: 1 }}>
                        <div style={{ fontWeight: 600 }}>Email</div>
                        <div className="text-muted-foreground" id="profileEmail">{profileData.email}</div>
                      </div>
                      <div style={{ textAlign: 'left', minWidth: '160px', flex: 1 }}>
                        <div style={{ fontWeight: 600 }}>Location</div>
                        <div className="text-muted-foreground" id="profileLocation">{profileData.location}</div>
                      </div>
                      <div style={{ textAlign: 'left', minWidth: '160px', flex: 1 }}>
                        <div style={{ fontWeight: 600 }}>Phone</div>
                        <div className="text-muted-foreground" id="profilePhone">{profileData.phone}</div>
                      </div>
                    </div>
                  </div>
                )}

                {editing && (
                  <form id="profileEdit" onSubmit={handleSubmit} style={{ marginTop: '12px', gap: '8px' }}>
                    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                      <input name="name" placeholder="Full name" defaultValue={profileData.name === '—' ? '' : profileData.name} style={{ flex: 1, padding: '8px', borderRadius: '8px', border: '1px solid var(--border)', background: 'transparent', color: 'var(--ink)' }} />
                      <input name="title" placeholder="Role / Title" defaultValue={profileData.title === '—' ? '' : profileData.title} style={{ flex: 1, padding: '8px', borderRadius: '8px', border: '1px solid var(--border)', background: 'transparent', color: 'var(--ink)' }} />
                    </div>
                    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '8px' }}>
                      <input name="email" placeholder="Email" defaultValue={profileData.email === '—' ? '' : profileData.email} style={{ flex: 1, padding: '8px', borderRadius: '8px', border: '1px solid var(--border)', background: 'transparent', color: 'var(--ink)' }} />
                      <input name="location" placeholder="Location" defaultValue={profileData.location === '—' ? '' : profileData.location} style={{ flex: 1, padding: '8px', borderRadius: '8px', border: '1px solid var(--border)', background: 'transparent', color: 'var(--ink)' }} />
                      <input name="phone" placeholder="Phone" defaultValue={profileData.phone === '—' ? '' : profileData.phone} style={{ flex: 1, padding: '8px', borderRadius: '8px', border: '1px solid var(--border)', background: 'transparent', color: 'var(--ink)' }} />
                    </div>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                      <label style={{ margin: 0, fontSize: '14px', color: 'var(--muted)', minWidth: '120px' }}>Desired salary</label>
                      <input name="desired_salary" placeholder="$80,000" defaultValue={profileData.desired_salary || ''} style={{ flex: 1, padding: '8px', borderRadius: '8px', border: '1px solid var(--border)', background: 'transparent', color: 'var(--ink)' }} />
                    </div>
                    <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end', marginTop: '10px' }}>
                      <button type="button" id="cancelEdit" onClick={() => setEditing(false)} style={{ padding: '8px 12px', borderRadius: '8px', border: '1px solid var(--border)', background: 'transparent', color: 'var(--ink)' }}>Cancel</button>
                      <button type="submit" style={{ padding: '8px 12px', borderRadius: '8px', background: 'var(--brand)', border: 0, color: '#07120c', fontWeight: 600 }}>Save</button>
                    </div>
                  </form>
                )}
              </section>

              {/* Resumes */}
              <section aria-labelledby="resumes" style={{ marginTop: '6px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <h3 id="resumes" style={{ margin: 0, fontWeight: 700 }}>Resumes</h3>
                  <div>
                    <Link to="/resume_upload" className="nav-link" style={{ padding: '6px 10px', border: '1px solid var(--border)', borderRadius: '8px' }}>
                      {resumeMeta ? 'Re-upload' : 'Upload'}
                    </Link>
                  </div>
                </div>

                <ul id="resumeList" style={{ listStyle: 'none', padding: 0, margin: '12px 0', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {resumeMeta ? (
                    <li style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '8px' }}>{resumeMeta.name}</div>
                        <div className="text-muted-foreground" style={{ fontSize: '12px' }}>
                          Uploaded at {new Date(resumeMeta.uploaded_at).toLocaleString()}
                        </div>
                      </div>
                    </li>
                  ) : (
                    <li className="text-muted-foreground">No resume on file.</li>
                  )}
                </ul>
              </section>

              {/* Job Preferences */}
              <section aria-labelledby="preferences" style={{ marginTop: '6px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '12px' }}>
                  <h3 id="preferences" style={{ margin: '0 0 8px 0', fontWeight: 700 }}>Job Preferences</h3>
                  <div>
                    <button
                      onClick={handlePrefsEditToggle}
                      disabled={loading || saving}
                      style={{ padding: '8px 10px', borderRadius: '8px', border: '1px solid var(--border)', background: 'transparent', color: 'var(--ink)' }}
                    >
                      {prefsEditing ? 'Cancel' : 'Edit'}
                    </button>
                  </div>
                </div>

                {/* View Edited Job Preferences) */}
                {!prefsEditing && (
                  <div className="text-muted-foreground" style={{ marginTop: '12px' }}>
                    <div><strong>Saved locations:</strong> {savedPrefs.pref_location || '—'}</div>
                    <div><strong>Saved type:</strong> {savedPrefs.pref_type || 'Any type'}</div>
                    <div><strong>Saved desired salary:</strong> {savedPrefs.pref_salary || '—'}</div>
                  </div>
                )}

                {/* Edit Job Preferences */}
                {prefsEditing && (
                  <form id="prefsForm" onSubmit={handlePrefsSubmit} style={{ marginTop: '12px', gap: '8px' }}>
                    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                      <input
                        name="pref_location"
                        placeholder="Preferred locations (comma separated)"
                        value={draftPrefs.pref_location}
                        onChange={(e) => setDraftPrefs(p => ({ ...p, pref_location: e.target.value }))}
                        style={{ flex: 1, padding: '8px', borderRadius: '8px', border: '1px solid var(--border)', background: 'transparent', color: 'var(--ink)' }}
                      />
                      <select
                        name="pref_type"
                        value={draftPrefs.pref_type}
                        onChange={(e) => setDraftPrefs(p => ({ ...p, pref_type: e.target.value }))}
                        style={{ minWidth: '160px', padding: '8px', borderRadius: '8px', border: '1px solid var(--border)', background: 'transparent', color: 'var(--ink)' }}
                      >
                        <option value="">Any type</option>
                        <option>Full-time</option>
                        <option>Part-time</option>
                        <option>Contract</option>
                        <option>Remote</option>
                      </select>
                    </div>

                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginTop: '8px' }}>
                      <label style={{ margin: 0, fontSize: '14px', color: 'var(--muted)', minWidth: '120px' }}>Desired salary</label>
                      <input
                        name="pref_salary"
                        placeholder="$80,000"
                        value={draftPrefs.pref_salary}
                        onChange={(e) => setDraftPrefs(p => ({ ...p, pref_salary: e.target.value }))}
                        style={{ flex: 1, padding: '8px', borderRadius: '8px', border: '1px solid var(--border)', background: 'transparent', color: 'var(--ink)' }}
                      />
                    </div>

                    <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end', marginTop: '10px' }}>
                      <button type="button" onClick={handlePrefsEditToggle} style={{ padding: '8px 12px', borderRadius: '8px', border: '1px solid var(--border)', background: 'transparent', color: 'var(--ink)' }}>
                        Cancel
                      </button>
                      <button type="submit" disabled={saving} style={{ padding: '8px 12px', borderRadius: '8px', background: 'var(--brand)', border: 0, color: '#07120c', fontWeight: 600 }}>
                        Save
                      </button>
                    </div>
                  </form>
                )}
              </section>

              {/* Security */}
              <section style={{ marginTop: '6px' }}>
                <h5 style={{ margin: '0 0 8px 0', fontWeight: 700 }}>Security</h5>
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  <a href="#" id="changePassword" style={{ padding: '8px 10px', borderRadius: '8px', border: '1px solid var(--border)', background: 'transparent', color: 'var(--ink)', textDecoration: 'none' }}>Change password</a>
                  <button 
                    onClick={handleLogout}
                    style={{ 
                      padding: '8px 10px', 
                      borderRadius: '8px', 
                      border: '1px solid #dc2626', 
                      background: 'transparent', 
                      color: '#dc2626',
                      cursor: 'pointer',
                      fontWeight: 500
                    }}
                  >
                    Logout
                  </button>
                </div>
              </section>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
