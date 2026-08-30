import { useRef, useState } from 'react'

export default function UploadZone({ onUpload, busy }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  function handleFiles(fileList) {
    const files = Array.from(fileList || [])
    if (files.length) onUpload(files)
  }

  return (
    <div
      className={`dropzone ${dragging ? 'dropzone--active' : ''} ${busy ? 'dropzone--busy' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        handleFiles(e.dataTransfer.files)
      }}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click() }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf,.pdf"
        multiple
        hidden
        onChange={(e) => { handleFiles(e.target.files); e.target.value = '' }}
      />
      <strong>{busy ? 'Uploading…' : 'Drop PDFs here'}</strong>
      <span>or click to choose files — nothing leaves your machine</span>
    </div>
  )
}
