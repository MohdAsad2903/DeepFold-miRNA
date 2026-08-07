// lib/modelNames.ts  — Mapping for model display names and descriptions

export const MODEL_DISPLAY_NAMES: Record<string, string> = {
  // Standard Keys (Production/Predict)
  CNN_v4:     'Structure Analysis Model',
  GCN:        'Graph Structure Model',
  kmer_XGB:   'Sequence Pattern Model',
  MFE_XGB:    'Stability Analysis Model',
  RNAFM_XGB:  'Evolutionary Pattern Model',

  // Aliases (API/Dashboard fallback keys)
  'cnn':      'Structure Analysis Model',
  'gcn':      'Graph Structure Model',
  'kmer':     'Sequence Pattern Model',
  'mfe':      'Stability Analysis Model',
  'rna-fm':   'Evolutionary Pattern Model',
  'rnafm':    'Evolutionary Pattern Model',
  'ensemble': 'DeepFold Ensemble (Meta-Learner)',
  'Meta-Learner Ensemble': 'DeepFold Ensemble (Meta-Learner)'
}

export const MODEL_DESCRIPTIONS: Record<string, string> = {
  CNN_v4:     'Analyzes the 2D contact map of the RNA structure as an image to detect folding disruptions',
  GCN:        'Represents the miRNA as a molecular graph and compares healthy vs mutant topology',
  kmer_XGB:   'Identifies which short sequence patterns (3–5 nucleotides) are created or destroyed by the mutation',
  MFE_XGB:    'Calculates the thermodynamic stability change caused by the mutation using folding energy',
  RNAFM_XGB:  'Analyzes deep evolutionary patterns across millions of RNA sequences to identify conservation signals',
}

/**
 * Returns the user-friendly display name for a model key.
 */
export function getModelName(key: string): string {
  return MODEL_DISPLAY_NAMES[key] ?? key;
}

/**
 * Returns the detailed description for a model key.
 */
export function getModelDescription(key: string): string {
  return MODEL_DESCRIPTIONS[key] ?? '';
}
