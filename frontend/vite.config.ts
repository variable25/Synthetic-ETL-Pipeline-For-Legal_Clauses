import { resolve } from 'node:path'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const page = (path: string) => resolve(import.meta.dirname, path)

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Unknown paths return 404 in dev too, like FastAPI's StaticFiles in production.
  appType: 'mpa',
  build: {
    rolldownOptions: {
      input: {
        main: page('index.html'),
        privacy: page('privacy/index.html'),
        terms: page('terms/index.html'),
        notFound: page('404.html'),
      },
    },
  },
  server: {
    // Dev only: forward API calls to FastAPI (uvicorn serve.app:app on :8000).
    // In production FastAPI serves the built app itself, so no proxy is needed.
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})