// Shows exactly what the parser extracted, page by page. This is a debugging
// view as much as a reading view: if a citation later points at page 4, this
// is where you confirm that page 4 really says what the answer claims.

export default function DocumentViewer({ detail, onClose }) {
  if (!detail) return null
  const { document: doc, pages } = detail

  return (
    <section className="viewer">
      <header className="viewer__header">
        <div>
          <h2>{doc.title || doc.filename}</h2>
          <p className="viewer__sub">
            {doc.filename} · {doc.page_count} pages
          </p>
        </div>
        <button className="ghost" onClick={onClose}>Close</button>
      </header>

      {doc.sections.length > 0 && (
        <div className="viewer__sections">
          <h3>Detected sections</h3>
          <div className="chips">
            {doc.sections.map((section) => (
              <span className="chip" key={section}>{section}</span>
            ))}
          </div>
        </div>
      )}

      <div className="viewer__pages">
        {pages.map((page) => (
          <article className="page" key={page.page_number}>
            <h4>Page {page.page_number}</h4>
            <pre>{page.text || '(no extractable text on this page)'}</pre>
          </article>
        ))}
      </div>
    </section>
  )
}
