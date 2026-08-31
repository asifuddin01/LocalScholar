import { useEffect, useState } from 'react'
import * as api from '../api.js'

// Renders the answer: [1] markers become clickable chips that jump to the
// matching source card, **bold** section labels are honoured (summaries come
// back with them), and blank lines become paragraphs. Citations are the point
// of this product, so they have to be followable, not just printed.
function renderInline(text, onJump, keyPrefix) {
  return text.split(/(\[\d{1,2}\]|\*\*[^*]+\*\*|_[^_]+_)/g).map((part, index) => {
    const key = `${keyPrefix}-${index}`
    const citation = /^\[(\d{1,2})\]$/.exec(part)
    if (citation) {
      const number = Number(citation[1])
      return (
        <button
          key={key}
          className="cite"
          onClick={() => onJump(number)}
          title={`Jump to source ${number}`}
        >
          {number}
        </button>
      )
    }
    const bold = /^\*\*([^*]+)\*\*$/.exec(part)
    if (bold) return <strong key={key}>{bold[1]}</strong>
    const italic = /^_([^_]+)_$/.exec(part)
    if (italic) return <em key={key} className="note">{italic[1]}</em>
    return <span key={key}>{part}</span>
  })
}

// Comparisons come back as a markdown table, which is the one piece of
// markdown worth supporting properly: a side-by-side comparison rendered as
// run-together prose is not a comparison anyone can read.
function MarkdownTable({ lines, onJump, keyPrefix }) {
  const cells = (line) =>
    line.replace(/^\||\|$/g, '').split('|').map((c) => c.trim())
  const [head, , ...body] = lines
  return (
    <div className="answer__tablewrap">
      <table className="answer__table">
        <thead>
          <tr>{cells(head).map((c, i) => <th key={i}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {body.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {cells(row).map((cell, cellIndex) =>
                cellIndex === 0 ? (
                  <th key={cellIndex}>{cell}</th>
                ) : (
                  <td key={cellIndex}>
                    {renderInline(cell, onJump, `${keyPrefix}-${rowIndex}-${cellIndex}`)}
                  </td>
                ),
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function isTableBlock(block) {
  const lines = block.split('\n')
  return lines.length >= 3 && lines[1].replace(/[\s|:-]/g, '') === ''
}

function AnswerText({ text, onJump }) {
  const blocks = text.split(/\n{2,}/)
  return (
    <div className="answer__text">
      {blocks.map((block, blockIndex) => {
        if (isTableBlock(block)) {
          return (
            <MarkdownTable
              key={blockIndex}
              lines={block.split('\n')}
              onJump={onJump}
              keyPrefix={blockIndex}
            />
          )
        }
        return (
          <p key={blockIndex}>
            {block.split('\n').map((line, lineIndex, lines) => (
              <span key={lineIndex}>
                {renderInline(line, onJump, `${blockIndex}-${lineIndex}`)}
                {lineIndex < lines.length - 1 && <br />}
              </span>
            ))}
          </p>
        )
      })}
    </div>
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
              <div className="answer__text"><p>{result.answer}</p></div>
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
