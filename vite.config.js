import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

// https://vite.dev/config/
export default defineConfig({
  // tell Vite that the app lives in ./frontend
  root: 'frontend',
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:5001', // Flask backend
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',      // output dir at repo root: job-ai-web-tool/dist
    emptyOutDir: true,   // clear dist before each build
  },
})