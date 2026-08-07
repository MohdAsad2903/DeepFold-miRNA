"use client";

import React, { createContext, useContext, useState, useCallback, useEffect, ReactNode } from 'react';
import { PredictionResponse } from './types';

export interface PredictionEntry {
  id: string;
  timestamp: string;
  mirna_id: string;
  seq_healthy: string;
  seq_mutant: string;
  prob_disease: number;
  label: 'Pathogenic' | 'Benign';
  confidence: string;
  base_probs: Record<string, number>;
}

export interface PredictionStoreContextType {
  history: PredictionEntry[];
  addPrediction: (p: PredictionEntry) => void;
  clearHistory: () => void;
  stats: {
    total: number;
    pathogenic: number;
    benign: number;
    mean_prob: number;
    high_confidence: number;
  };
}

const PredictionContext = createContext<PredictionStoreContextType | undefined>(undefined);

export function PredictionProvider({ children }: { children: ReactNode }) {
  const [history, setHistory] = useState<PredictionEntry[]>([]);

  // Load from local storage
  useEffect(() => {
    try {
      const stored = localStorage.getItem('deepfold_history');
      if (stored) {
        setHistory(JSON.parse(stored));
      }
    } catch(e) {
      console.warn("Failed to read local storage", e);
    }
  }, []);

  const addPrediction = useCallback((p: PredictionEntry) => {
    setHistory((prev) => {
      const updated = [p, ...prev].slice(0, 50); // Keep last 50
      localStorage.setItem('deepfold_history', JSON.stringify(updated));
      return updated;
    });
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
    localStorage.removeItem('deepfold_history');
  }, []);

  const stats = React.useMemo(() => {
    const total = history.length;
    let pathogenic = 0;
    let sumProb = 0;
    let highConf = 0;

    history.forEach(h => {
      if (h.label === 'Pathogenic') pathogenic++;
      sumProb += h.prob_disease;
      if (h.confidence === 'High') highConf++;
    });

    return {
      total,
      pathogenic,
      benign: total - pathogenic,
      mean_prob: total > 0 ? sumProb / total : 0,
      high_confidence: highConf
    };
  }, [history]);

  return (
    <PredictionContext.Provider value={{ history, addPrediction, clearHistory, stats }}>
      {children}
    </PredictionContext.Provider>
  );
}

export function usePredictionStore() {
  const context = useContext(PredictionContext);
  if (context === undefined) {
    throw new Error('usePredictionStore must be used within a PredictionProvider');
  }
  return context;
}
