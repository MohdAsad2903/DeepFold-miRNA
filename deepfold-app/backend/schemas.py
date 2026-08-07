from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, List

class PredictionRequest(BaseModel):
    mirna_id: str
    seq_healthy: str
    seq_mutant: str
    snp_pos: Optional[int] = None

class BaseModelProbs(BaseModel):
    CNN_v4: float
    GCN: float
    kmer_XGB: float
    MFE_XGB: float
    RNAFM_XGB: float

class ShapFeature(BaseModel):
    feature_name: str
    plain_label: str
    shap_value: float
    direction: str   # 'pathogenic' or 'benign'

class PredictionResponse(BaseModel):
    prob_disease: float
    label: str           # 'Pathogenic' | 'Benign'
    confidence: str      # 'High' | 'Medium' | 'Low'
    base_probs: BaseModelProbs
    delta_mfe: Optional[float] = None
    snp_in_stem: Optional[bool] = None
    processing_time_ms: int
    # New additions
    shap_explanation: Optional[List[ShapFeature]] = None
    verdict_explanation: Optional[str] = None
    disagreement_score: float
    disagreement_level: str
    ood_flag: bool
    ood_message: Optional[str] = None

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    models_loaded: bool
    model_count: int
