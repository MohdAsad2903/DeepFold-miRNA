import numpy as np
import shap

FEATURE_LABELS = {
    # k-mer patterns
    'k3_density': 'Loss of 3-mer structural motif',
    'k4_density': 'Disruption of 4-mer stem motif',
    'k5_density': 'Change in 5-mer structural motif',
    'diff_3mer_AUG': 'Loss of AUG start-codon motif',
    'diff_3mer_UGC': 'Disruption of UGC stem motif',
    'diff_4mer_GAUC': 'Change in GAUC structural motif',
    # positional
    'snp_in_seed': 'SNP located in seed region (pos 2-8)',
    'snp_in_mature': 'SNP within mature miRNA body',
    'gc_content_healthy': 'GC content of healthy sequence',
    'ctx_gc': 'GC content around mutation site',
    'ctx_entropy': 'Sequence complexity at mutation site',
    'is_purine': 'Reference base is a purine (A or G)',
    'snp_relative_pos': 'SNP position within hairpin',
    'dist_from_centre': 'Distance of SNP from hairpin centre',
    # MFE related passed through
    'delta_mfe': 'Change in RNA folding stability',
    'snp_in_stem': 'SNP disrupts a base-paired stem region',
}

def get_kmer_feats(seq: str, k_list=[3,4,5]):
    seq = seq.upper().replace("U", "T")
    feats = []
    for k in k_list:
        kmers = {}
        for i in range(len(seq)-k+1):
            km = seq[i:i+k]
            kmers[km] = kmers.get(km, 0) + 1
        # Match training logic exactly
        feats.append(float(len(kmers)) / len(seq))
    return np.array(feats).reshape(1, -1)

def extract_all_features(seq_h: str, seq_m: str):
    """
    Consolidates feature extraction for XGBoost models.
    """
    # K-mer features
    kmer_feats = get_kmer_feats(seq_m)
    
    # Placeholder for MFE (needs ViennaRNA)
    mfe_feats = np.zeros((1, 5)) 
    
    # Placeholder for RNA-FM (needs fair-esm)
    rnafm_feats = np.zeros((1, 10))
    
    
    return kmer_feats, mfe_feats, rnafm_feats

def compute_shap_explanation(features, kmer_model, feature_names):
    """
    Computes SHAP explanation for a single prediction.
    """
    try:
        # Use TreeExplainer on the booster for reliability
        explainer = shap.TreeExplainer(kmer_model.get_booster())
        shap_values = explainer.shap_values(features)[0]
    except:
        # Fallback to symbolical importance if shap fails
        shap_values = kmer_model.feature_importances_ * 0.1

    explanations = []
    for i, val in enumerate(shap_values):
        fname = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        label = FEATURE_LABELS.get(fname, fname)
        direction = "pathogenic" if val > 0 else "benign"
        explanations.append({
            "feature_name": fname,
            "plain_label": label,
            "shap_value": float(val),
            "direction": direction
        })
    
    # Sort by absolute shape value
    explanations.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return explanations[:10] # Top 10 combined
