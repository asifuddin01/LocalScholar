import { useCallback, useEffect, useState } from 'react'
import * as api from './api.js'
import UploadZone from './components/UploadZone.jsx'
import DocumentList from './components/DocumentList.jsx'
import DocumentViewer from './components/DocumentViewer.jsx'
import SearchPanel from './components/SearchPanel.jsx'
import AskPanel from './components/AskPanel.jsx'
import ResearchPanel from './components/ResearchPanel.jsx'

const POLL_INTERVAL_MS = 1500

export default function App() {
  const [documents, setDocuments] = useState([])
  const [selectedIds, setSelectedIds] = useState([])
  const [detail, setDetail] = useState(null)
  const [notices, setNotices] = useState([])
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)

  const refresh = useCallback(async () => {
    try {
      setDocuments(await api.listDocuments())
      setError(null)
    } catch (err) {
      setError(`Cannot reach the LocalScholar API — is the backend running? (${err.message})`)
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  // Parsing happens in a background task, so poll only while something is
  // actually in flight. An idle library makes no requests.
  useEffect(() => {
    if (!documents.some((doc) => doc.status === 'processing')) return undefined
    const timer = setInterval(refresh, POLL_INTERVAL_MS)
    return () => clearInterval(timer)
  }, [documents, refresh])

  async function handleUpload(files) {
    setUploading(true)
    setNotices([])
    try {
      const result = await api.uploadDocuments(files)
      const messages = [
        ...result.rejected.map((r) => ({ kind: 'error', text: `${r.filename}: ${r.reason}` })),
        ...result.results
          .filter((r) => r.duplicate)
          .map((r) => ({ kind: 'info', text: `${r.document.filename} is already in your library.` })),
      ]
      setNotices(messages)
      await refresh()
    } catch (err) {
      setNotices([{ kind: 'error', text: err.message }])
    } finally {
      setUploading(false)
    }
  }

  async function handleSelect(id) {
    setDetail(await api.getDocument(id))
  }

  function toggleDocument(id) {
    setSelectedIds((current) =>
      current.includes(id) ? current.filter((x) => x !== id) : [...current, id],
    )
  }

  async function handleDelete(id) {
    await api.deleteDocument(id)
    setSelectedIds((current) => current.filter((x) => x !== id))
    if (detail?.document.id === id) setDetail(null)
    await refresh()
  }

  return (
    <div className="app">
      <header className="app__header">
        <h1>LocalScholar</h1>
        <p>Local LLM research paper assistant — your papers never leave this machine.</p>
      </header>

      {error && <div className="notice notice--error">{error}</div>}

      <section className="panel">
        <h2>Upload papers</h2>
        <UploadZone onUpload={handleUpload} busy={uploading} />
        {notices.map((notice, index) => (
          <div key={index} className={`notice notice--${notice.kind}`}>{notice.text}</div>
        ))}
      </section>

      <section className="panel">
        <h2>Your papers <span className="count">{documents.length}</span></h2>
        <DocumentList
          documents={documents}
          selectedId={detail?.document.id}
          onSelect={handleSelect}
          onDelete={handleDelete}
        />
      </section>

      <AskPanel
        documents={documents}
        selectedIds={selectedIds}
        onToggleDocument={toggleDocument}
      />

      <ResearchPanel documents={documents} selectedIds={selectedIds} />

      <SearchPanel
        documents={documents}
        selectedIds={selectedIds}
        onToggleDocument={toggleDocument}
      />

      {detail && <DocumentViewer detail={detail} onClose={() => setDetail(null)} />}
    </div>
  )
}
