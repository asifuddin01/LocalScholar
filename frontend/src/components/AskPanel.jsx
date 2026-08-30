import { useEffect, useState } from 'react'
import * as api from '../api.js'

// Renders [1] / [2] in the answer as clickable chips that jump to the matching
// source card. Citations are the point of this product, so they have to be
// followable, not just printed.
function AnswerText({ text, onJump }) {
  const parts = text.split(/(\[\d{1,2}\])/g)
  return (
    <p className="answer__text">
      {parts.map((part, index) => {
        const match = /^\[(\d{1,2})\]$/.exec(part)
        if (!match) return <span key={index}>{part}</span>
        const number = Number(match[1])
        return (
          <button
            key={index}
            className="cite"
            onClick={() => onJump(number)}
            title={`Jump to source ${number}`}
          >
            {number}
          </button>
        )
      })}
    </p>
  )
}

function SourceCard({ source, cited, highlighted }) {
  return (
    <article
      id={`source-${source.index}`}
      className={`source ${cited ? '' : 'source--uncited'} ${highlighted ? 'source--flash' : ''}`}
    >
      <header>
        <span className="source__index">[{source.index}]</span>
        <span className="source__where">
          {source.filename} — Page {source.page_number}
          {source.section ? ` — ${source.section}` : ''}
        </span>
        {!cited && <span className="source__tag">retrieved, not cited</span>}
      </header>
      <p>{source.text}</p>
    </article>
  )
}

export default function AskPanel({ documents, selectedIds, onToggleDocument }) {
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [llm, setLlm] = useState(null)
  const [showAll, setShowAll] = useState(false)
  const [flash, setFlash] = useState(null)

  const ready = documents.filter((d) => d.status === 'ready')

  useEffect(() => {
    api.llmStatus().then(setLlm).catch(() => setLlm(null))
  }, [])

  async function run(event) {
    event.preventDefault()
    if (!question.trim()) return
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      setResult(await api.ask({ question, documentIds: selectedIds }))
      setShowAll(false)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  function jumpTo(number) {
    setShowAll(true)
    setFlash(number)
    setTimeout(() => {
      document.getElementById(`source-${number}`)?.scrollIntoView({
        behavior: 'smooth', block: 'center',
      })
    }, 30)
    setTimeout(() => setFlash(null), 1600)
  }

  if (!ready.length) return null

  const cited = new Set(result?.cited_indexes || [])
  const visibleSources = (result?.sources || []).filter(
    (s) => showAll || cited.has(s.index),
  )

  return (
    <section className="panel">
      <h2>Ask your papers</h2>

      {llm && !llm.available && (
        <div className="notice notice--error">
          {llm.detail} LocalScholar needs a local model to answer questions; retrieval
          still works without one.
        </div>
      )}

      <form className="search" onSubmit={run}>
        <input
          type="text"
          value={question}
          placeholder="e.g. what dataset did they use, and how large was it?"
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button type="submit" disabled={busy || !question.trim()}>
          {busy ? 'Thinking…' : 'Ask'}
        </button>
      </form>

      {ready.length > 1 && (
        <div className="scope">
          <span className="scope__label">
            {selectedIds.length
              ? `Asking ${selectedIds.length} selected paper(s)`
              : 'Asking all papers'}
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

      {busy && <div className="notice notice--info">Generating…</div>}
      {error && <div className="notice notice--error">{error}</div>}

      {result && (
        <div className="answer">
          <div className={`answer__body ${result.found ? '' : 'answer__body--empty'}`}>
            {result.found ? (
              <AnswerText text={result.answer} onJump={jumpTo} />
            ) : (
              <p className="answer__text">{result.answer}</p>
            )}
          </div>

          <div className="answer__meta">
            {result.model} · {(result.took_ms / 1000).toFixed(1)}s ·
            {' '}{result.sources.length} passages retrieved
            {result.found ? `, ${cited.size} cited` : ''}
            {result.sources.length > cited.size && (
              <button className="linkish" onClick={() => setShowAll((v) => !v)}>
                {showAll ? 'show cited only' : 'show all retrieved evidence'}
              </button>
            )}
          </div>

          {visibleSources.length > 0 && (
            <div className="sources">
              <h3>Sources</h3>
              {visibleSources.map((source) => (
                <SourceCard
                  key={source.chunk_id}
                  source={source}
                  cited={cited.has(source.index)}
                  highlighted={flash === source.index}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  )
}
