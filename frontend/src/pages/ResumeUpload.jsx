import { useEffect, useRef, useState } from 'react'
import './jobAI.css'

export default function ResumeUpload() {
  const [current, setCurrent] = useState(null)
  const [loading, setLoading] = useState(false)
  const [selected, setSelected] = useState(null);
  const [err, setErr] = useState('')
  const fileRef = useRef(null)

  const fmtSize = (n) => {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n/1024).toFixed(1)} KB`;
  return `${(n/1024/1024).toFixed(1)} MB`;
};

  // ** ADD AUTH USER WIRING HERE **
  const DEV_HEADERS = { 'X-User-Id': '1' }

  useEffect(() => { document.title = 'Resume Upload – jobhunter.ai' }, [])

  useEffect(() => {
    async function fetchData() {
      setErr('')
      try {
        const res = await fetch('/api/resumes', { headers: DEV_HEADERS })
        if (res.status === 404) { setCurrent(null); return }
        if (!res.ok) throw new Error(await res.text())
        const data = await res.json()
        setCurrent(data)
      } catch (e) {
        setErr(e.message || 'Failed to fetch resume.')
      }
    }
    fetchData()
  }, [])

  function pickFile() {
    fileRef.current?.click()
  }

  async function onSubmit(e) {
    e.preventDefault()
    setErr('')

    const file = fileRef.current?.files?.[0]
    if (!file) {
      if (current?.resume_id) {
        setErr('Please choose a new file before replacing your resume.')
      } else {
        setErr('Please choose a file to upload.')
      }
      return
    }

    const okTypes = [
      'application/pdf',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'text/plain',
    ]
    if (!okTypes.includes(file.type)) {
      setErr('Only PDF, DOCX, or TXT allowed.')
      return
    }
    if (file.size > 5 * 1024 * 1024) {
      setErr('Max file size is 5MB.')
      return
    }

    const form = new FormData()
    form.append('file', file)

    setLoading(true)
    try {
      let res
      if (current?.resume_id) {
        // Replace
        res = await fetch(`/api/resumes/${current.resume_id}`, {
          method: 'PUT',
          headers: DEV_HEADERS,
          body: form,
        })
      } else {
        // Create
        res = await fetch('/api/resumes', {
          method: 'POST',
          headers: DEV_HEADERS,
          body: form,
        })
      }
      if (!res.ok) throw new Error(await res.text())

      // After create/replace, refresh the latest metadata
      const latest = await fetch('/api/resumes', { headers: DEV_HEADERS })
      if (!latest.ok) throw new Error(await latest.text())
      const data = await latest.json()
      setCurrent(data)
      if (data?.resume_id) localStorage.setItem('resume_id', String(data.resume_id))
      if (fileRef.current) fileRef.current.value = ''
      setSelected(null)
    } catch (e) {
      setErr(e.message || 'Upload failed.')
    } finally {
      setLoading(false)
    }
  }

  async function onDelete() {
    if (!current?.resume_id) return
    setErr('')
    setLoading(true)
    try {
      const res = await fetch(`/api/resumes/${current.resume_id}`, {
        method: 'DELETE',
        headers: DEV_HEADERS,
      })
      if (res.status !== 204 && !res.ok) throw new Error(await res.text())
      setCurrent(null)
      localStorage.removeItem('resume_id')
      if (fileRef.current) fileRef.current.value = ''
    } catch (e) {
      setErr(e.message || 'Delete failed.')
    } finally {
      setLoading(false)
    }
  }

  function onFileChange(e) {
  setErr('');
  const f = e.target.files?.[0];
  if (!f) { setSelected(null); return; }

  const okTypes = [
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'text/plain',
  ];
  if (!okTypes.includes(f.type)) {
    setErr('Only PDF, DOCX, or TXT allowed.');
    setSelected(null);
    return;
  }
  if (f.size > 5 * 1024 * 1024) {
    setErr('Max file size is 5MB.');
    setSelected(null);
    return;
  }

  setSelected({ name: f.name, size: f.size, type: f.type });
}
  return (
    <div>
      {/* Top-right brand */}
      <div>
        <a href="/" className="brand-right" aria-label="jobhunter.ai home">
          <span className="brand-dot-right" aria-hidden="true"></span>
          <span>jobhunter.ai</span>
        </a>
      </div>

      <div className="app-row">
        {/* Sidebar */}
        <nav className="sidebar">
          <ul>
            <li>
              <a href="/" className="nav-link">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
                     strokeWidth="1.5" stroke="currentColor" className="nav-icon">
                  <path strokeLinecap="round" strokeLinejoin="round"
                        d="m2.25 12 8.954-8.955c.44-.439 1.152-.439 1.591 0L21.75 12M4.5 9.75v10.125c0 .621.504 1.125 1.125 1.125H9.75v-4.875c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125V21h4.125c.621 0 1.125-.504 1.125-1.125V9.75M8.25 21h8.25"/>
                </svg>
                Home
              </a>
            </li>
            <li><a href="/resume_upload">Upload Resume</a></li>
            <li><a href="/matchedAI">Search</a></li>
            <li><a href="/savedJobs">Saved Jobs</a></li>
            <li><a href="/appliedJobs">Applied Jobs</a></li>
            <li><a href="/profile">Profile</a></li>
          </ul>
        </nav>

        {/* Main Content */}
        <main className="container" style={{ width: '100vw', maxWidth: '100%', boxSizing: 'border-box' }}>
          <h1 className="page-title">Upload Your Resume</h1>
          <p className="text-muted-foreground">We'll check your resume and give you tips to make it better</p>

          <div data-slot="card" className="bg-card text-card-foreground flex flex-col gap-6 rounded-xl border">
            <div data-slot="card-header" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px', paddingTop: '24px', paddingBottom: '12px' }}>
              <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mb-2">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-upload w-8 h-8 text-primary">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="17 8 12 3 7 8"></polyline>
                  <line x1="12" x2="12" y1="3" y2="15"></line>
                </svg>
              </div>
              <h4 data-slot="card-title" className="leading-none" style={{ margin: 0 }}>Upload Your Resume</h4>
            </div>

            <div data-slot="card-content" className="px-6 [&:last-child]:pb-6 space-y-4" style={{ paddingBottom: '24px' }}>
              {/* Metadata (if any) */}
              {current && (
                <div className="rounded-md border p-3" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
                  <div>
                    <div><strong>Filename:</strong> {current.name}</div>
                    <div><strong>Uploaded:</strong> {new Date(current.uploaded_at).toLocaleString()}</div>
                  </div>
                  <button
                    className="btn btn-danger"
                    onClick={onDelete}
                    disabled={loading || (!selected && !current)}
                    title="Delete current resume"
                    style={{ padding: '10px 24px', borderRadius: '8px', background: 'var(--brand)', color: '#000000ff', fontWeight: 600, cursor: 'pointer', display: 'inline-block' }}
                  >
                    {loading ? 'Deleting…' : 'Delete'}
                  </button>
                </div>
              )}

              {/* Upload / Replace */}
              <form onSubmit={onSubmit} className="border-dashed" style={{ textAlign: 'center' }}>
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
                     viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                     strokeLinecap="round" strokeLinejoin="round"
                     className="lucide lucide-file-text w-12 h-12 text-muted-foreground mx-auto mb-4">
                  <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"></path>
                  <path d="M14 2v4a2 2 0 0 0 2 2h4"></path>
                  <path d="M10 9H8"></path>
                  <path d="M16 13H8"></path>
                  <path d="M16 17H8"></path>
                </svg>

                <p className="mb-4">Drop your resume here or click below</p>

                <button
                  type="button"
                  onClick={pickFile}
                  style={{ padding: '10px 24px', borderRadius: '8px', background: 'var(--brand)', color: '#000000ff', fontWeight: 600, cursor: 'pointer', display: 'inline-block' }}
                >
                  Choose File
                </button>

                <input
                  id="fileUpload"
                  ref={fileRef}
                  type="file"
                  accept=".pdf,.docx,.txt"
                  style={{ display: 'none' }}
                  onChange={onFileChange}
                />

                {selected && (
                  <div
                    role="status"
                    aria-live="polite"
                    className="rounded-md border p-2 mt-3"
                    style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}
                  >
                    <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
                      <path d="M20 6L9 17l-5-5" fill="none" stroke="currentColor" strokeWidth="2" />
                    </svg>
                    <span><strong>Ready:</strong> {selected.name} ({fmtSize(selected.size)})</span>
                    <button
                      type="button"
                      onClick={() => { setSelected(null); if (fileRef.current) fileRef.current.value = ''; }}
                      className="btn"
                      style={{ marginLeft: 8, padding: '4px 10px', color: '#ffffffff'  }}
                      title="Clear selected file"
                    >
                      Clear
                    </button>
                  </div>
                )}

                <div style={{ marginTop: 12 }}>
                  <button className="btn" type="submit" disabled={loading} style={{ minWidth: 120, color: '#ffffffff' }}>
                    {loading ? (current ? 'Replacing…' : 'Uploading…') : (current ? 'Replace' : 'Upload')}
                  </button>
                </div>
                
              {/* Errors */}
              {err && <div style={{ color: 'crimson', padding: '10px 24px' }}>{err}</div>}

                <p className="text-sm text-muted-foreground mt-4">PDF, DOCX, or TXT — max 5MB</p>
              </form>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}