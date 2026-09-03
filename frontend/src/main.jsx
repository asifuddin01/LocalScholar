import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import './styles.css'

// The published demo runs this exact interface with no backend behind it, so
// the /api routes are answered from a recording. Guarded by a build flag, and
// tree-shaken out of every build that does not set it — a local instance must
// never be talking to a fixture without knowing.
if (import.meta.env.VITE_DEMO === '1') {
  const { installDemoBackend } = await import('./demo/offline.js')
  installDemoBackend()
}

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
