import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The API runs as a separate process in development. Proxying keeps the
    // frontend talking to a same-origin /api in both dev and the Docker build,
    // so no environment-specific base URL is needed anywhere in the code.
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
  build: { outDir: 'dist' },
})
