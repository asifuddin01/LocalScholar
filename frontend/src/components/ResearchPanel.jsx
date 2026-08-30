import { useState } from 'react'
import * as api from '../api.js'

// "Not reported" is rendered as a distinct, muted state rather than an empty
// cell. An empty cell reads as a bug; "Not reported" is a finding — the paper
// genuinely does not say.
function Value({ value, citations }) {
  const missing = !value || /^not reported$/i.test(value.trim())
  if (missing) return <span className="missing">Not reported</span>
  return (
    <>
      <span>{value}</span>
      {citations?.length > 0 && (
        <span className="cites">
          {citations.map((c) => (
            <span key={c} className="cite cite--static">{c}</span>
          ))}
        </span>
      )}
    </>
  )
}

export default function ResearchPanel({ documents, selectedIds }) {
  const [details, setDetails] = useState(null)
  const [table, setTable] = useState(null)
  const [busy, setBusy] = useState(null)
  const [error, setError] = useState(null)

  const ready = documents.filter((d) => d.status === 'ready')
  const chosen = selectedIds.length
    ? ready.filter((d) => selectedIds.includes(d.id))
    : ready

  async function loadDetails(doc) {
    setBusy(`Extracting details from ${doc.title || doc.filename}…`)
    setError(null)
    setTable(null)
    try {
      setDetails(await api.paperDetails(doc.id))
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(null)
    }
  }

  async function runCompare() {
    setBusy(`Comparing ${chosen.length} papers… (first run extracts each one)`)
    setError(null)
    setDetails(null)
    try {
      setTable(await api.compare(chosen.map((d) => d.id)))
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(null)
    }
  }

  if (!ready.length) return null

  return (
    <section className="panel">
      <h2>Paper details &amp; comparison</h2>

      <div className="actions">
        {ready.map((doc) => (
          <button key={doc.id} className="ghost" onClick={() => loadDetails(doc)}>
            {doc.title ? doc.title.slice(0, 40) : doc.filename.slice(0, 40)}
          </button>
        ))}
      </div>

      {chosen.length >= 2 && (
        <div className="actions">
          <button className="primary" onClick={runCompare}>
            Compare {chosen.length} selected papers
          </button>
        </div>
      )}

      {busy && <div className="notice notice--info">{busy}</div>}
      {error && <div className="notice notice--error">{error}</div>}

      {details && (
        <div className="details">
          <h3>{details.title || details.filename}</h3>
          <table className="kv">
            <tbody>
              {details.fields.map((f) => (
                <tr key={f.name}>
                  <th>{f.label}</th>
                  <td><Value value={f.value} citations={f.citations} /></td>
                </tr>
              ))}
            </tbody>
          </table>
          <details className="evidence">
            <summary>{details.sources.length} supporting excerpts</summary>
            {details.sources.map((s) => (
              <div className="source" key={s.chunk_id}>
                <header>
                  <span className="source__index">[{s.index}]</span>
                  <span className="source__where">
                    Page {s.page_number}{s.section ? ` — ${s.section}` : ''}
                  </span>
                </header>
                <p>{s.text}</p>
              </div>
            ))}
          </details>
        </div>
      )}

      {table && (
        <div className="compare-wrap">
          <table className="compare">
            <thead>
              <tr>
                <th />
                {table.columns.map((c) => (
                  <th key={c.document_id}>{c.title || c.filename}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {table.rows.map((row) => (
                <tr key={row.field}>
                  <th>{row.label}</th>
                  {row.cells.map((cell, i) => (
                    <td key={i}>
                      <Value value={cell.value} citations={cell.citations} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
