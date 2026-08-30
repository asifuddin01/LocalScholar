const STATUS_LABEL = {
  processing: 'Processing…',
  ready: 'Indexed',
  failed: 'Failed',
}

function formatSize(bytes) {
  if (!bytes) return ''
  const mb = bytes / (1024 * 1024)
  return mb >= 1 ? `${mb.toFixed(1)} MB` : `${Math.round(bytes / 1024)} KB`
}

export default function DocumentList({ documents, selectedId, onSelect, onDelete }) {
  if (!documents.length) {
    return <p className="empty">No papers yet. Upload a PDF to build your library.</p>
  }

  return (
    <ul className="paper-list">
      {documents.map((doc) => (
        <li
          key={doc.id}
          className={`paper ${selectedId === doc.id ? 'paper--selected' : ''}`}
          onClick={() => doc.status === 'ready' && onSelect(doc.id)}
        >
          <div className="paper__main">
            <span className="paper__title">{doc.title || doc.filename}</span>
            {doc.title && <span className="paper__file">{doc.filename}</span>}
            <span className={`badge badge--${doc.status}`}>
              {STATUS_LABEL[doc.status] || doc.status}
            </span>
          </div>

          {doc.status === 'ready' && (
            <div className="paper__meta">
              {doc.page_count} pages · {doc.sections.length} sections · {formatSize(doc.size_bytes)}
            </div>
          )}
          {doc.status === 'failed' && <div className="paper__error">{doc.error}</div>}

          <button
            className="paper__delete"
            title="Remove from library"
            onClick={(e) => { e.stopPropagation(); onDelete(doc.id) }}
          >
            ×
          </button>
        </li>
      ))}
    </ul>
  )
}
