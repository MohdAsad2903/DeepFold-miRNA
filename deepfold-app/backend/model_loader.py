import logging
import os
import joblib
import torch
from typing import List, Dict, Any
from nn_models import DeepFoldCNN_v4, SiameseGCN

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.demo_mode = False
        
        self.cnn_models = []
        self.gcn_models = []
        self.kmer_models = []
        self.mfe_models = []
        self.rnafm_models = []
        self.meta_model = None
        self.config = {}
        
        self._load_models()

    def _load_models(self):
        logger.info(f"Loading DeepFold Registry from {self.models_dir}...")
        if not os.path.exists(self.models_dir):
            logger.warning(f"Models directory {self.models_dir} not found. Running in DEMO_MODE.")
            self.demo_mode = True
            return

        try:
            # Load CNN Folds
            for i in range(1, 6):
                path = os.path.join(self.models_dir, f"cnn_v4_fold{i}.pt")
                if os.path.exists(path):
                    model = DeepFoldCNN_v4().to(self.device)
                    model.load_state_dict(torch.load(path, map_location=self.device))
                    model.eval()
                    self.cnn_models.append(model)
            
            # Load GCN Folds
            for i in range(1, 6):
                path = os.path.join(self.models_dir, f"gcn_fold{i}.pt")
                if os.path.exists(path):
                    model = SiameseGCN().to(self.device)
                    model.load_state_dict(torch.load(path, map_location=self.device))
                    model.eval()
                    self.gcn_models.append(model)
            
            # Load XGBoost Folds
            for i in range(1, 6):
                kp = os.path.join(self.models_dir, f"kmer_xgb_fold{i}.pkl")
                if os.path.exists(kp):
                    self.kmer_models.append(joblib.load(kp))
                
                mp = os.path.join(self.models_dir, f"mfe_xgb_fold{i}.pkl")
                if os.path.exists(mp):
                    self.mfe_models.append(joblib.load(mp))
                
                rp = os.path.join(self.models_dir, f"rnafm_xgb_fold{i}.pkl")
                if os.path.exists(rp):
                    self.rnafm_models.append(joblib.load(rp))
            
            # Load Meta Learner
            meta_path = os.path.join(self.models_dir, "meta_learner_v2.pkl")
            if os.path.exists(meta_path):
                self.meta_model = joblib.load(meta_path)
            
            # Load Config
            config_path = os.path.join(self.models_dir, "pipeline_config.pkl")
            if os.path.exists(config_path):
                self.config = joblib.load(config_path)

            total = len(self.cnn_models) + len(self.gcn_models) + len(self.kmer_models) + \
                    len(self.mfe_models) + len(self.rnafm_models)
            
            if total == 0:
                logger.warning("No model files found. DEMO_MODE activated.")
                self.demo_mode = True
            else:
                logger.info(f"Successfully loaded {total} models across 5 architectures. Meta-learner active.")
                
        except Exception as e:
            logger.error(f"Failed to load registry: {str(e)}")
            self.demo_mode = True
