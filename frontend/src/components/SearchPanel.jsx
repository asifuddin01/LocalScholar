import { useState } from 'react'
import * as api from '../api.js'

// Dense and lexical retrieval are exposed as an explicit toggle rather than
// blended behind one button. They fail in opposite directions and it is worth
// being able to see that: ask "what optimizer was used?" and dense retrieval
// returns plausible-looking prose about models, while BM25 lands directly on
// the hyperparameter table containing the word "AdamW".

const METHODS = [
  { id: 'dense', label: 'Semantic', hint: 'Embeddings — finds paraphrases' },
  { id: 'bm25', label: 'Keyword', hint: 'BM25 — finds exact terms' },
]

export default function SearchPanel({ documents, selectedIds, onToggleDocument }) {
  const [query, setQuery] = useState('')
  const [method, setMethod] = useState('dense')
  const [results, setResults] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const ready = documents.filter((d) => d.status === 'ready')

  async function run(event) {
    event.preventDefault()
    if (!query.trim()) return
    setBusy(true)
    setError(null)
    try {
      setResults(await api.search({ query, method, documentIds: selectedIds }))
    } catch (err) {
      setError(err.message)
      setResults(null)
    } finally {
      setBusy(false)
    }
  }

  if (!ready.length) return null

  return (
    <section className="panel">
      <h2>Search your papers</h2>

      <form className="search" onSubmit={run}>
        <input
          type="text"
          value={query}
          placeholder="e.g. what dataset did they use?"
          onChange={(e) => setQuery(e.target.value)}
        />
        <button type="submit" disabled={busy || !query.trim()}>
          {busy ? 'Searching…' : 'Search'}
        </button>
      </form>

      <div className="methods">
        {METHODS.map((m) => (
          <button
            key={m.id}
            type="button"
            title={m.hint}
            className={`pill ${method === m.id ? 'pill--on' : ''}`}
            onClick={() => setMethod(m.id)}
          >
            {m.label}
          </button>
        ))}
        <span className="methods__hint">
          {METHODS.find((m) => m.id === method).hint}
        </span>
      </div>

      {ready.length > 1 && (
        <div className="scope">
          <span className="scope__label">
            {selectedIds.length ? `Searching ${selectedIds.length} paper(s)` : 'Searching all papers'}
          </span>
          {ready.map((doc) => (
            <label key={doc.id} className="scope__item">
              <input
                type="checkbox"
                checked={selectedIds.includes(doc.id)}
                onChange={() => onToggleDocument(doc.id)}
              />
              {doc.title || doc.filename}
            </label>
          ))}
        </div>
      )}

      {error && <div className="notice notice--error">{error}</div>}

      {results && (
        <div className="results">
          <div className="results__meta">
            {results.results.length} result(s) · {results.took_ms}ms
          </div>
          {results.results.length === 0 && (
            <p className="empty">
              No matching passages. Nothing in these papers uses those terms.
            </p>
          )}
          {results.results.map((r) => (
            <article className="result" key={r.chunk_id}>
              <header>
                <span className="result__source">
                  {r.filename} — Page {r.page_number}
                  {r.section ? ` — ${r.section}` : ''}
                </span>
                <span className="result__score">{r.score.toFixed(3)}</span>
              </header>
              <p>{r.text}</p>
            </article>
          ))}
        </div>
      )}
    </section>
  )
}
