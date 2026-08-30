// All requests are same-origin: Vite proxies /api to the backend in dev, and
// the Docker image serves the built frontend from the backend itself.

async function request(path, options) {
  const response = await fetch(path, options)
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      if (body.detail) detail = body.detail
    } catch {
      // Non-JSON error body; the status line is the best we have.
    }
    throw new Error(detail)
  }
  return response.status === 204 ? null : response.json()
}

export function listDocuments() {
  return request('/api/documents')
}

export function getDocument(id) {
  return request(`/api/documents/${id}`)
}

export function deleteDocument(id) {
  return request(`/api/documents/${id}`, { method: 'DELETE' })
}

export function uploadDocuments(files) {
  const form = new FormData()
  for (const file of files) form.append('files', file)
  return request('/api/documents', { method: 'POST', body: form })
}

export function search({ query, method = 'dense', documentIds = [], topK = null }) {
  return request('/api/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      method,
      document_ids: documentIds,
      top_k: topK,
    }),
  })
}

export function ask({ question, documentIds = [], topK = null }) {
  return request('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, document_ids: documentIds, top_k: topK }),
  })
}

export function llmStatus() {
  return request('/api/llm/status')
}

export function paperDetails(id, refresh = false) {
  return request(`/api/documents/${id}/details${refresh ? '?refresh=true' : ''}`)
}

export function compare(documentIds) {
  return request('/api/compare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ document_ids: documentIds }),
  })
}
