# DeepFold Web Application

A full-stack web application for predicting miRNA SNP pathogenicity. It uses a Next.js 14 frontend with Three.js WebGL visualizations, and a Python FastAPI backend that interfaces with an ensemble of structural, sequence, and thermodynamic machine learning models.

## Project Structure
- `/frontend`: Next.js 14 App Router, TailwindCSS, Framer Motion, React Three Fiber
- `/backend`: FastAPI, Pydantic, Python ML Libraries

## Demo Mode
If the trained models are not placed in `DeepFold_models/` at the backend root, the backend will automatically start in **DEMO_MODE**. In this mode, the API returns realistic mock predictions and ensemble probabilities to allow for UI/UX development and testing without requiring the massive 2GB ML model payloads.

## Running Locally

### Backend (FastAPI)
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the development server (runs on `localhost:8000`):
   ```bash
   uvicorn main:app --reload
   ```

### Frontend (Next.js)
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server (runs on `localhost:3000`):
   ```bash
   npm run dev
   ```

## API Endpoints
- `GET /health` - System status and model registry check.
- `GET /model-stats` - Returns benchmarked metrics (AUC, F1, etc) for all models in the ensemble.
- `GET /history` - Returns the last 50 queries from in-memory cache.
- `POST /predict` - Accepts `{"mirna_id", "seq_healthy", "seq_mutant", "snp_pos"}`. Returns probability, pathogenic label, model breakdown, and thermodynamics.
