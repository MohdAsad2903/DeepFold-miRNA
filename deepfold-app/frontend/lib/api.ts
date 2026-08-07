import { PredictionRequest, PredictionResponse, ModelStats, PredictionHistoryEntry } from './types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function predictVariant(req: PredictionRequest): Promise<PredictionResponse> {
  const response = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Prediction failed');
  }
  
  return response.json();
}

export async function getModelStats(): Promise<ModelStats[]> {
  const response = await fetch(`${API_BASE}/model-stats`);
  if (!response.ok) throw new Error('Failed to fetch model stats');
  return response.json();
}

export async function getHealth(): Promise<{ status: string; models_loaded: boolean; model_count: number }> {
  try {
    const response = await fetch(`${API_BASE}/health`);
    if (!response.ok) return { status: 'error', models_loaded: false, model_count: 0 };
    return response.json();
  } catch (error) {
    return { status: 'offline', models_loaded: false, model_count: 0 };
  }
}

export async function getPredictionHistory(): Promise<PredictionHistoryEntry[]> {
  const response = await fetch(`${API_BASE}/history`);
  if (!response.ok) return [];
  return response.json();
}

export async function getAnalytics(): Promise<any> {
  try {
    const response = await fetch(`${API_BASE}/analytics`);
    if (!response.ok) return {};
    const data = await response.json();
    return data;
  } catch {
    return {};
  }
}

export async function predictBatch(formData: FormData): Promise<any> {
  const response = await fetch(`${API_BASE}/predict/batch`, {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Batch prediction failed');
  }
  
  return response.json();
}

export async function getVerifiedExamples(): Promise<any[]> {
  const response = await fetch(`${API_BASE}/examples`);
  if (!response.ok) return [];
  return response.json();
}

export async function getValidationData(): Promise<any> {
  const response = await fetch(`${API_BASE}/validation`);
  if (!response.ok) return { error: 'Failed to fetch validation data' };
  return response.json();
}
