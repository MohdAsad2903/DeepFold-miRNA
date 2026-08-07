'use client';

import { useState } from 'react';
import dynamic from 'next/dynamic';
import { PredictionResponse } from '@/lib/types';
import { motion, AnimatePresence } from 'framer-motion';

const RNAHelix3D = dynamic(() => import('@/components/visualization/RNAHelix3D'), {
  ssr: false,
  loading: () => <div className="w-full h-[450px] skeleton rounded-xl" />
});

const ResultCard = dynamic(() => import('@/components/prediction/ResultCard'), {
  ssr: false,
  loading: () => <div className="w-full h-[300px] skeleton rounded-xl" />
});

const ConfidenceMeter = dynamic(() => import('@/components/prediction/ConfidenceMeter'), {
  ssr: false,
  loading: () => <div className="w-full h-[180px] skeleton rounded-xl" />
});

const ModelBreakdown = dynamic(() => import('@/components/prediction/ModelBreakdown'), {
  ssr: false,
  loading: () => <div className="w-full h-[250px] skeleton rounded-xl" />
});

const MutationSite3D = dynamic(() => import('@/components/visualization/MutationSite3D'), {
  ssr: false,
  loading: () => <div className="w-full h-[300px] skeleton rounded-xl" />
});

const SequenceInput = dynamic(() => import('@/components/prediction/SequenceInput'), {
  ssr: false,
  loading: () => <div className="w-full h-[500px] skeleton rounded-xl" />
});

export default function PredictPage() {
  const [mirnaId, setMirnaId]   = useState('');
  const [healthy, setHealthy]   = useState('');
  const [mutant,  setMutant]    = useState('');
  const [snpPos,  setSnpPos]    = useState(0);
  const [loading, setLoading]   = useState(false);
  const [result,  setResult]    = useState<PredictionResponse | null>(null);

  let varType: 'pathogenic' | 'benign' | 'VUS' | null = null;
  if (result) {
    if (result.prob_disease >= 0.5) varType = 'pathogenic';
    else varType = 'benign';
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-[1400px] relative z-10">
      <div className="mb-8 pl-4 border-l-4 border-cyan-500 shadow-[0_0_15px_rgba(0,229,255,0.1)]">
        <h1 className="text-3xl font-display font-black text-white tracking-widest uppercase">Predictive Console</h1>
        <div className="flex items-center gap-4 mt-2">
          <span className="text-gray-500 font-mono text-[10px] tracking-widest uppercase">
            Status: 0x00_READY_WAITING_FOR_VECTOR
          </span>
          <div className="flex-1 h-[2px] bg-cyan-700/20" />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-10 gap-6 mb-6">

        {/* Left Panel — Stepped Inputs (30%) */}
        <div className="col-span-1 lg:col-span-3">
          <SequenceInput
            mirnaId={mirnaId} setMirnaId={setMirnaId}
            healthy={healthy} setHealthy={setHealthy}
            mutant={mutant}   setMutant={setMutant}
            snpPos={snpPos}   setSnpPos={setSnpPos}
            onPredict={(data) => { setResult(data); }}
          />
        </div>

        {/* Center Panel — Structural Representation (40%) */}
        <div className="col-span-1 lg:col-span-4 flex flex-col gap-6">
          <RNAHelix3D healthySeq={healthy} mutantSeqType={varType} snpPos={snpPos} />
          <AnimatePresence>
            {result && !loading && (
              <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }}>
                <MutationSite3D snpPos={snpPos} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Right Panel — Ensemble Diagnostics (30%) */}
        <div className="col-span-1 lg:col-span-3 flex flex-col gap-6">
          {!result && !loading && (
            <div className="h-[250px] glass-card flex flex-col items-center justify-center border-dashed opacity-50 border-cyan-500/20">
              <span className="text-[10px] font-mono text-cyan-500/50 uppercase tracking-[0.2em] text-center">
                Awaiting Tensor Compute<br />Input required in step 3
              </span>
            </div>
          )}

          {loading && (
            <div className="h-full glass-card p-10 flex flex-col items-center justify-center gap-6">
              <div className="w-16 h-16 rounded-full border-2 border-cyan-500 border-t-transparent animate-spin" />
              <div className="text-xs font-mono text-cyan-400 animate-pulse tracking-widest">EXECUTING_ENSEMBLE_STACK...</div>
            </div>
          )}

          {result && !loading && (
            <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="space-y-6">
              <ResultCard result={result} />
              <ConfidenceMeter prob={result.prob_disease} />
              <ModelBreakdown probs={result.base_probs} />
            </motion.div>
          )}
        </div>

      </div>
    </div>
  );
}
