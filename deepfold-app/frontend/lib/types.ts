export interface PredictionRequest {
  mirna_id: string;
  seq_healthy: string;
  seq_mutant: string;
  snp_pos?: number;
}

export interface BaseModelProbs {
  CNN_v4: number;
  GCN: number;
  kmer_XGB: number;
  MFE_XGB: number;
  RNAFM_XGB: number;
}

export interface ShapFeature {
  feature_name: string;
  plain_label: string;
  shap_value: number;
  direction: 'pathogenic' | 'benign';
}

export interface PredictionResponse {
  prob_disease: number;
  label: 'Pathogenic' | 'Benign';
  confidence: 'High' | 'Medium' | 'Low';
  base_probs: {
    CNN_v4: number;
    GCN: number;
    kmer_XGB: number;
    MFE_XGB: number;
    RNAFM_XGB: number;
  };
  delta_mfe?: number;
  snp_in_stem?: boolean;
  processing_time_ms: number;
  shap_explanation?: ShapFeature[];
  verdict_explanation?: string;
  disagreement_score: number;
  disagreement_level: 'Low' | 'Moderate' | 'High';
  ood_flag: boolean;
  ood_message?: string;
}

export interface PredictionHistoryEntry extends PredictionResponse {
  id: string;
  timestamp: string;
  mirna_id: string;
  seq_healthy: string;
  seq_mutant: string;
}

export interface ModelStats {
  name: string;
  auc_mean: number;
  auc_std: number;
  accuracy: number;
  f1: number;
}
