'use client';

import { ExternalLink } from 'lucide-react';

export default function AboutPage() {
  return (
    <div className="container mx-auto px-4 py-12 max-w-5xl">
      <div className="mb-12">
        <h1 className="text-4xl font-display font-bold text-white mb-4">DeepFold Architecture</h1>
        <p className="text-xl text-gray-400">Understanding the 5-phase ensemble ML pipeline.</p>
      </div>

      {/* Pipeline Diagram (CSS only) */}
      <section className="mb-16">
        <h2 className="text-2xl font-display font-semibold text-white mb-6">Pipeline Workflow</h2>
        
        <div className="flex flex-col lg:flex-row items-center justify-between gap-4 w-full">
          {/* Stage 1 */}
          <div className="glass-card w-full lg:w-48 text-center p-4">
            <div className="w-10 h-10 mx-auto bg-cyan-500/20 rounded-full flex items-center justify-center text-cyan-400 font-bold mb-2">1</div>
            <h3 className="font-semibold text-sm">Input Arrays</h3>
            <p className="text-xs text-gray-400 mt-1">Healthy & Mutant Sequences</p>
          </div>
          
          <Arrow />
          
          {/* Stage 2 */}
          <div className="glass-card w-full lg:w-48 text-center p-4 border-emerald-500/30">
            <div className="w-10 h-10 mx-auto bg-emerald-500/20 rounded-full flex items-center justify-center text-emerald-400 font-bold mb-2">2</div>
            <h3 className="font-semibold text-sm">UFold Inference</h3>
            <p className="text-xs text-gray-400 mt-1">128x128 Contact Maps</p>
          </div>
          
          <Arrow />
          
          {/* Stage 3 (Parallel) */}
          <div className="glass-card w-full lg:w-64 text-center p-4 border-violet-500/30 bg-violet-500/5">
            <div className="w-10 h-10 mx-auto bg-violet-500/20 rounded-full flex items-center justify-center text-violet-400 font-bold mb-2">3</div>
            <h3 className="font-semibold text-sm mb-2">Parallel Encoders</h3>
            <div className="text-xs text-left space-y-1 text-gray-300 bg-deepspace/50 p-2 rounded">
              <p>• <b>Structure Analysis:</b> Multi-channel CNN v4</p>
              <p>• <b>Graph Structure:</b> Siamese Graph ConvNet</p>
              <p>• <b>Sequence Pattern:</b> 1,995-dim k-mer frequencies</p>
              <p>• <b>Evolutionary Pattern:</b> 640-dim latent Evolutionary Pattern Model embeddings</p>
              <p>• <b>Stability Analysis:</b> Thermodynamic MFE change</p>
            </div>
          </div>
          
          <Arrow />
          
          {/* Stage 4 */}
          <div className="glass-card w-full lg:w-48 text-center p-4 border-amber-500/30">
            <div className="w-10 h-10 mx-auto bg-amber-500/20 rounded-full flex items-center justify-center text-amber-500 font-bold mb-2">4</div>
            <h3 className="font-semibold text-sm">Stacking</h3>
            <p className="text-xs text-gray-400 mt-1">Rank-Normalised OOF Meta-Learner</p>
          </div>
          
          <Arrow />
          
          {/* Stage 5 */}
          <div className="glass-card w-full lg:w-48 text-center p-4 border-red-500/30 bg-red-500/5">
            <h3 className="font-bold text-lg text-white">Prediction</h3>
            <p className="text-xs text-red-300 mt-1 font-mono">&gt; 0.65 Pathogenic</p>
          </div>
        </div>
      </section>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
        <section className="glass-card">
          <h2 className="text-xl font-display font-semibold text-white mb-4">Dataset Construction</h2>
          <p className="text-gray-300 mb-2">
            <strong>COSMIC (Disease):</strong> 6,512 disease-associated SNPs from the COSMIC database, deduplicated against dbSNP.
          </p>
          <p className="text-gray-300 mb-2">
            <strong>gnomAD (Benign):</strong> Streaming 150GB of human VCF files to extract natural population variants with AF_popmax &gt; 0.005. 
          </p>
          <p className="text-gray-300">
            <strong>Final Output:</strong> Exactly balanced 2,372 sample pairs, no synthetic data generated.
          </p>
        </section>

        <section className="glass-card">
          <h2 className="text-xl font-display font-semibold text-white mb-4">Rigorous Evaluation</h2>
          <p className="text-gray-300 mb-2">
            Traditional k-fold validation leads to catastrophic data leakage in bioinformatics because highly similar sequences (like `hsa-mir-21` and `hsa-mir-21b`) can bleed across train/test splits.
          </p>
          <p className="text-cyan-300 font-medium">
            We implemented StratifiedGroupKFold grouped by miRNA prefix family to ensure zero sequence leakage during the 5-Fold cross-validation strategy.
          </p>
        </section>
      </div>

      <section className="mb-16">
         <h2 className="text-2xl font-display font-semibold text-white mb-6">References & Technologies</h2>
         <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
            <a href="#" className="glass-card p-4 hover:bg-cyan-500/10 flex justify-between items-center group">
              <span>UFold PyTorch</span>
              <ExternalLink size={16} className="text-gray-500 group-hover:text-cyan-400" />
            </a>
            <a href="#" className="glass-card p-4 hover:bg-cyan-500/10 flex justify-between items-center group">
              <span>Evolutionary Model (fair-esm)</span>
              <ExternalLink size={16} className="text-gray-500 group-hover:text-cyan-400" />
            </a>
            <a href="#" className="glass-card p-4 hover:bg-cyan-500/10 flex justify-between items-center group">
              <span>ViennaRNA</span>
              <ExternalLink size={16} className="text-gray-500 group-hover:text-cyan-400" />
            </a>
            <a href="#" className="glass-card p-4 hover:bg-cyan-500/10 flex justify-between items-center group">
              <span>XGBoost</span>
              <ExternalLink size={16} className="text-gray-500 group-hover:text-cyan-400" />
            </a>
            <a href="#" className="glass-card p-4 hover:bg-cyan-500/10 flex justify-between items-center group">
              <span>Optuna</span>
              <ExternalLink size={16} className="text-gray-500 group-hover:text-cyan-400" />
            </a>
         </div>
      </section>

    </div>
  );
}

function Arrow() {
  return (
    <div className="flex justify-center py-2 lg:py-0">
      {/* Mobile down arrow */}
      <div className="h-6 w-0.5 bg-cyan-500/30 lg:hidden"></div>
      {/* Desktop right arrow */}
      <div className="hidden lg:flex items-center w-8">
        <div className="h-0.5 w-full bg-cyan-500/30"></div>
        <div className="w-2 h-2 border-t-2 border-r-2 border-cyan-500/30 rotate-45 -ml-1.5"></div>
      </div>
    </div>
  );
}
