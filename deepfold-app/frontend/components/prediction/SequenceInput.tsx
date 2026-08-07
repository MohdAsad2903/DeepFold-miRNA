'use client';

import { useState, useEffect, useCallback } from 'react';
import { ChevronDown, Check, Info, FileText, Edit3 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { PredictionResponse } from '@/lib/types';
import { getVerifiedExamples, predictVariant } from '@/lib/api';
import { usePredictionStore } from '@/lib/predictionStore';

// ─── Local fallback sequences extracted from final_dataset.csv ───────────────
// Used when GET /examples returns empty (backend CWD issue)
const LOCAL_FALLBACK: Record<string, any> = {
  'hsa-mir-21':   { seq_healthy: 'TGTCGGGTAGCTTATCAGACTGATGTTGACTGTTGAATCTCATGGCAACACCAGTCGATGGGCTGTCTGACA', seq_mutant: 'TTTCGGGTAGCTTATCAGACTGATGTTGACTGTTGAATCTCATGGCAACACCAGTCGATGGGCTGTCTGACA', snp_pos: 1, category: 'pathogenic' },
  'hsa-mir-155':  { seq_healthy: 'CCTTAGCAGAGCTGTGGAGTGTGACAATGGTGTTTGTGTCTAAACTATCAAACGCCATTA', seq_mutant: 'CCTTAGCAGAGCTGTGGAGTGTGACAATGGTGTTTGTGTATAAACTATCAAACGCCATTA', snp_pos: 39, category: 'pathogenic' },
  'hsa-mir-17':   { seq_healthy: 'CCGATGTGTATCCTCAGCTTTGAGAACTGAATTCCATGGGTTGTGTCAGTGTCAGACCTC', seq_mutant: 'CCGATGTGTATCCTCAGCTTTGAGAACTGAATTCCATGGGTTGTGTCAGTGTCAGAACTC', snp_pos: 56, category: 'pathogenic' },
  'hsa-mir-122':  { seq_healthy: 'CCTTAGCAGAGCTGTGGAGTGTGACAATGGTGTTTGTGTCTAAACTATCAAACGCCATTATCACACTAAATAGCTACTGCTAGGC', seq_mutant: 'CCTTAGCAGAGCTGTGGAGTGTGACAATGGTGTTTGTGTCTAAACTATCAAATGCCATTATCACACTAAATAGCTACTGCTAGGC', snp_pos: 52, category: 'pathogenic' },
  'hsa-mir-34a':  { seq_healthy: 'GGGCCAAGGTGGGCCAGGGGTGGTGTTGGGACAGCTCCGTTTAAAAAGGCATCTCCAAGA', seq_mutant: 'GGGCCAAGGTGGGCCAGGGGTGGTGTTGGGACAGCTCCGTTTAAAAAGGCATCTCCAATA', snp_pos: 58, category: 'pathogenic' },
  'hsa-mir-146a': { seq_healthy: 'CCGATGTGTATCCTCAGCTTTGAGAACTGAATTCCATGGGTTGTGTCAGTGTCAGACCTCTGAAATTCAGTTCTTCAGCTGGGATATCTCTGTCATCGT', seq_mutant: 'CCGATGTGTATCCTCAGCTTTGAGAACTGAATTCCATGGGTTGTGTCAGTGTCAGACCTGTGAAATTCAGTTCTTCAGCTGGGATATCTCTGTCATCGT', snp_pos: 59, category: 'pathogenic' },
  'hsa-mir-196a': { seq_healthy: 'ATAAAGGAAGTTAGGCTGAGGGGCAGAGAGCGAGACTTTTCTATTTTCCAAAAGCTCGGT', seq_mutant: 'ATAAAGGAAGTTAGGCTGAGGGGCAGAGAGCGCGACTTTTCTATTTTCCAAAAGCTCGGT', snp_pos: 32, category: 'benign' },
  'hsa-mir-499':  { seq_healthy: 'TGTCGGGTAGCTTATCAGACTGATGTTGACTGTTGAATCTCATGGCAACACCAGTCGATG', seq_mutant: 'TGTCGGGTAGCTTATCAGACTGATGTTGACTGTTGAATGTCATGGCAACACCAGTCGATG', snp_pos: 38, category: 'benign' },
  'hsa-mir-608':  { seq_healthy: 'GGGCCAAGGTGGGCCAGGGGTGGTGTTGGGACAGCTCCGTTTAAAAAGGCATCTCCAAGAGCTTCCATCAAAGGCTGCCTCTTGGTGCAGCACAGGTAGA', seq_mutant: 'GGGCCAAGGTGGGCCAGGGGTGGTGTTGGGACAGCTGCGTTTAAAAAGGCATCTCCAAGAGCTTCCATCAAAGGCTGCCTCTTGGTGCAGCACAGGTAGA', snp_pos: 36, category: 'benign' },
};

interface ExampleItem {
  id: string;
  label: string;
  category: 'pathogenic' | 'benign';
  mirna_id: string;
  seq_healthy: string;
  seq_mutant: string;
  snp_pos: number;
  expected: string;
}

interface Props {
  mirnaId: string;
  setMirnaId: (v: string) => void;
  healthy: string;
  setHealthy: (v: string) => void;
  mutant: string;
  setMutant: (v: string) => void;
  snpPos: number;
  setSnpPos: (v: number) => void;
  onPredict: (result: PredictionResponse) => void;
}

const sanitise = (v: string) => v.toUpperCase().replace(/[^AUCGT]/g, '');

export default function SequenceInput({
  mirnaId, setMirnaId, healthy, setHealthy, mutant, setMutant, snpPos, setSnpPos, onPredict
}: Props) {
  const [step, setStep] = useState(1);
  const [method, setMethod] = useState<'example' | 'custom' | 'fasta' | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fastaInput, setFastaInput] = useState('');

  // Examples
  const [examples, setExamples] = useState<ExampleItem[]>([]);
  const [selectedExample, setSelectedExample] = useState<ExampleItem | null>(null);

  // Custom input
  const [customMirnaId, setCustomMirnaId] = useState('');
  const [customHealthy, setCustomHealthy] = useState('');
  const [customMutant, setCustomMutant] = useState('');
  const [customSnpPos, setCustomSnpPos] = useState('0');
  const [preloadId, setPreloadId] = useState('');

  const { addPrediction } = usePredictionStore();

  // Fetch examples on mount; merge with LOCAL_FALLBACK
  useEffect(() => {
    getVerifiedExamples().then((remote: any[]) => {
      if (remote && remote.length > 0) {
        setExamples(remote as ExampleItem[]);
      } else {
        // Build from LOCAL_FALLBACK
        const fallback: ExampleItem[] = Object.entries(LOCAL_FALLBACK).map(([id, d]) => ({
          id, label: id, category: d.category,
          mirna_id: id, seq_healthy: d.seq_healthy,
          seq_mutant: d.seq_mutant, snp_pos: d.snp_pos,
          expected: d.category
        }));
        setExamples(fallback);
      }
    }).catch(() => {
      const fallback: ExampleItem[] = Object.entries(LOCAL_FALLBACK).map(([id, d]) => ({
        id, label: id, category: d.category,
        mirna_id: id, seq_healthy: d.seq_healthy,
        seq_mutant: d.seq_mutant, snp_pos: d.snp_pos,
        expected: d.category
      }));
      setExamples(fallback);
    });
  }, []);

  // ─── Custom diff detection ────────────────────────────────────────────────
  let diffCount = -1;
  let diffPos = -1;
  let diffMsg = 'Enter both sequences to compare';
  let diffColor = 'text-gray-500';

  if (customHealthy && customMutant) {
    if (customHealthy.length !== customMutant.length) {
      diffCount = -2;
      diffMsg = '✗ Sequences must be the same length';
      diffColor = 'text-red-400';
    } else {
      diffCount = 0;
      for (let i = 0; i < customHealthy.length; i++) {
        if (customHealthy[i] !== customMutant[i]) { diffCount++; diffPos = i; }
      }
      if (diffCount === 1) {
        diffMsg = `✓ 1 difference found at position ${diffPos}`;
        diffColor = 'text-green-400';
      } else if (diffCount === 0) {
        diffMsg = '✗ Sequences are identical — must have exactly 1 difference';
        diffColor = 'text-red-400';
      } else {
        diffMsg = `✗ ${diffCount} differences found — must be exactly 1`;
        diffColor = 'text-red-400';
      }
    }
  }

  // Auto-fill SNP pos from diff
  useEffect(() => {
    if (diffCount === 1) setCustomSnpPos(diffPos.toString());
  }, [diffCount, diffPos]);

  const isCustomValid = customMirnaId.trim() !== '' &&
    customHealthy.length >= 15 &&
    customMutant.length >= 15 &&
    /^[AUCGT]+$/.test(customHealthy) &&
    /^[AUCGT]+$/.test(customMutant) &&
    diffCount === 1;

  // ─── FASTA parser ─────────────────────────────────────────────────────────
  const parseFasta = useCallback((input: string) => {
    const lines = input.trim().split('\n');
    let h = '', m = '', name = '';
    let reading: 'h' | 'm' | null = null;
    for (const line of lines) {
      if (line.startsWith('>healthy')) { reading = 'h'; continue; }
      if (line.startsWith('>mutant'))  { reading = 'm'; continue; }
      if (line.startsWith('>'))        { name = line.slice(1).trim(); continue; }
      if (reading === 'h') h += line.trim();
      if (reading === 'm') m += line.trim();
    }
    setCustomHealthy(sanitise(h));
    setCustomMutant(sanitise(m));
    setMirnaId(name || 'custom-fasta');
    setCustomMirnaId(name || 'custom-fasta');
    if (h.length === m.length && h.length > 0) {
      let pos = -1, count = 0;
      for (let i = 0; i < h.length; i++) {
        if (h[i].toUpperCase() !== m[i].toUpperCase()) { pos = i; count++; }
      }
      if (count === 1) setCustomSnpPos(pos.toString());
    }
  }, [setMirnaId]);

  // ─── Preload example into custom form ────────────────────────────────────
  const handlePreload = (id: string) => {
    setPreloadId(id);
    const ex = examples.find(e => e.id === id);
    if (!ex) return;
    setCustomMirnaId(ex.mirna_id);
    setCustomHealthy(ex.seq_healthy.toUpperCase());
    setCustomMutant(ex.seq_mutant.toUpperCase());
    setCustomSnpPos(ex.snp_pos.toString());
  };

  // ─── Main predict handler ─────────────────────────────────────────────────
  const handlePredict = async () => {
    setIsSubmitting(true);
    setError(null);
    try {
      let finalMirnaId = '';
      let finalHealthy = '';
      let finalMutant = '';
      let finalSnp = 0;

      if (method === 'example') {
        if (!selectedExample) return;
        finalMirnaId = selectedExample.mirna_id;
        finalHealthy = selectedExample.seq_healthy.toUpperCase();
        finalMutant  = selectedExample.seq_mutant.toUpperCase();
        finalSnp     = selectedExample.snp_pos;
      } else {
        finalMirnaId = customMirnaId || 'custom-query';
        finalHealthy = customHealthy.toUpperCase();
        finalMutant  = customMutant.toUpperCase();
        finalSnp     = parseInt(customSnpPos) || 0;
      }

      setMirnaId(finalMirnaId);
      setHealthy(finalHealthy);
      setMutant(finalMutant);
      setSnpPos(finalSnp);

      const res = await predictVariant({
        mirna_id:    finalMirnaId,
        seq_healthy: finalHealthy,
        seq_mutant:  finalMutant,
        snp_pos:     finalSnp,
      });

      addPrediction({
        id:          'pred_' + Date.now(),
        timestamp:   new Date().toISOString(),
        mirna_id:    finalMirnaId,
        seq_healthy: finalHealthy,
        seq_mutant:  finalMutant,
        prob_disease: res.prob_disease,
        label:       res.label,
        confidence:  res.confidence,
        base_probs:  res.base_probs,
      });

      onPredict(res);
    } catch (err: any) {
      setError('Prediction failed. Make sure the backend is running.');
    } finally {
      setIsSubmitting(false);
    }
  };

  // ─── Step indicator ───────────────────────────────────────────────────────
  const pathogenic = examples.filter(e => e.category === 'pathogenic');
  const benign     = examples.filter(e => e.category === 'benign');

  return (
    <div className="w-full flex flex-col gap-6">

      {/* Progress indicator */}
      <div className="flex items-center justify-between px-2 mb-2">
        {[1, 2, 3].map(i => (
          <div key={i} className="flex items-center gap-2">
            <div className={`w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold border transition-all ${
              step > i  ? 'bg-cyan-500 border-cyan-500 text-[#020818]' :
              step === i ? 'border-cyan-500 text-cyan-400' :
                          'border-cyan-500/30 text-cyan-500/30'
            }`}>
              {step > i ? <Check size={12} strokeWidth={3} /> : i}
            </div>
            {i < 3 && <div className={`w-8 h-[1px] ${step > i ? 'bg-cyan-500' : 'bg-cyan-500/20'}`} />}
          </div>
        ))}
      </div>

      <AnimatePresence mode="wait">

        {/* ── STEP 1: Choose method ── */}
        {step === 1 && (
          <motion.div key="step1" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} className="flex flex-col gap-4">
            <h3 className="text-sm font-bold text-cyan-400/70 uppercase tracking-widest mb-2 px-1">Choose Input Method</h3>

            <button onClick={() => { setMethod('example'); setStep(2); }}
              className="glass-card p-4 text-left border-l-4 border-l-cyan-500 flex items-center gap-4 hover:bg-cyan-500/5 transition-colors">
              <div className="w-10 h-10 rounded-lg bg-cyan-500/10 flex items-center justify-center text-cyan-400 shrink-0">
                <Info size={20} />
              </div>
              <div>
                <div className="font-bold text-white">Example miRNA</div>
                <div className="text-xs text-gray-500">Choose from {examples.length || 9} pre-loaded benchmarks</div>
              </div>
            </button>

            <button onClick={() => { setMethod('custom'); setStep(2); }}
              className="glass-card p-4 text-left border-l-4 border-l-violet-500 flex items-center gap-4 hover:bg-violet-500/5 transition-colors">
              <div className="w-10 h-10 rounded-lg bg-violet-500/10 flex items-center justify-center text-violet-400 shrink-0">
                <Edit3 size={20} />
              </div>
              <div>
                <div className="font-bold text-white">Custom Input</div>
                <div className="text-xs text-gray-500">Enter sequences and SNP manually</div>
              </div>
            </button>

            <button onClick={() => { setMethod('fasta'); setStep(2); }}
              className="glass-card p-4 text-left border-l-4 border-l-amber-500 flex items-center gap-4 hover:bg-amber-500/5 transition-colors">
              <div className="w-10 h-10 rounded-lg bg-amber-500/10 flex items-center justify-center text-amber-400 shrink-0">
                <FileText size={20} />
              </div>
              <div>
                <div className="font-bold text-white">Paste FASTA</div>
                <div className="text-xs text-gray-500">Extract from raw genomic files</div>
              </div>
            </button>
          </motion.div>
        )}

        {/* ── STEP 2: Data entry ── */}
        {step === 2 && (
          <motion.div key="step2" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }} className="flex flex-col gap-5">

            {/* ── EXAMPLE flow ── */}
            {method === 'example' && (
              <div className="flex flex-col gap-4">
                <label className="text-[10px] font-bold text-cyan-500/60 uppercase tracking-widest">Benchmark Selection</label>

                {/* Pathogenic group */}
                {pathogenic.length > 0 && (
                  <div className="flex flex-col gap-1">
                    <div className="text-[9px] font-bold text-red-400/60 uppercase tracking-widest px-1 pb-1">── Pathogenic ──</div>
                    {pathogenic.map(ex => (
                      <button key={ex.id}
                        onClick={() => setSelectedExample(ex)}
                        className={`w-full text-left p-3 rounded-lg border-l-2 flex items-center justify-between transition-all ${
                          selectedExample?.id === ex.id
                            ? 'border-l-red-400 bg-red-500/10 text-white'
                            : 'border-l-red-400/30 glass-card hover:bg-red-500/5 text-gray-300 hover:text-white'
                        }`}>
                        <span className="text-xs font-mono">{ex.id}</span>
                        <span className="text-[9px] px-2 py-0.5 rounded border border-red-500/30 text-red-400">PATHOGENIC</span>
                      </button>
                    ))}
                  </div>
                )}

                {/* Benign group */}
                {benign.length > 0 && (
                  <div className="flex flex-col gap-1 mt-1">
                    <div className="text-[9px] font-bold text-green-400/60 uppercase tracking-widest px-1 pb-1">── Benign ──</div>
                    {benign.map(ex => (
                      <button key={ex.id}
                        onClick={() => setSelectedExample(ex)}
                        className={`w-full text-left p-3 rounded-lg border-l-2 flex items-center justify-between transition-all ${
                          selectedExample?.id === ex.id
                            ? 'border-l-green-400 bg-green-500/10 text-white'
                            : 'border-l-green-400/30 glass-card hover:bg-green-500/5 text-gray-300 hover:text-white'
                        }`}>
                        <span className="text-xs font-mono">{ex.id}</span>
                        <span className="text-[9px] px-2 py-0.5 rounded border border-green-500/30 text-green-400">BENIGN</span>
                      </button>
                    ))}
                  </div>
                )}

                {/* Confirmation card */}
                {selectedExample && (
                  <div className="mt-2 p-4 glass-card bg-[#020818]/60 border-cyan-500/20 font-mono text-[11px] space-y-2 rounded-lg">
                    <div className="flex justify-between border-b border-cyan-500/5 pb-1">
                      <span className="text-gray-500">miRNA ID</span>
                      <span className="text-cyan-400">{selectedExample.mirna_id}</span>
                    </div>
                    <div className="flex justify-between border-b border-cyan-500/5 pb-1">
                      <span className="text-gray-500">Category</span>
                      <span className={selectedExample.category === 'pathogenic' ? 'text-red-400' : 'text-green-400'}>
                        {selectedExample.category === 'pathogenic' ? 'Pathogenic' : 'Benign'}
                      </span>
                    </div>
                    <div className="flex justify-between border-b border-cyan-500/5 pb-1">
                      <span className="text-gray-500">SNP Position</span>
                      <span className="text-white">{selectedExample.snp_pos}</span>
                    </div>
                    <div className="text-[9px] text-gray-600 pt-1">Sequences loaded from training dataset</div>
                  </div>
                )}
              </div>
            )}

            {/* ── CUSTOM flow ── */}
            {method === 'custom' && (
              <div className="flex flex-col gap-4">

                {/* Preload dropdown */}
                <div className="flex flex-col gap-1">
                  <label className="text-[10px] font-bold text-cyan-500/50 uppercase tracking-widest">
                    Load a known sequence as starting point (optional)
                  </label>
                  <select value={preloadId} onChange={e => handlePreload(e.target.value)}
                    className="w-full p-2 glass-card bg-[#0a1229] border border-cyan-500/30 rounded-md text-xs focus:border-cyan-400 outline-none">
                    <option value="">— Select to prefill sequences —</option>
                    <optgroup label="── Pathogenic ──">
                      {pathogenic.map(ex => <option key={ex.id} value={ex.id}>{ex.id}</option>)}
                    </optgroup>
                    <optgroup label="── Benign ──">
                      {benign.map(ex => <option key={ex.id} value={ex.id}>{ex.id}</option>)}
                    </optgroup>
                  </select>
                </div>

                {/* miRNA ID */}
                <div className="flex flex-col gap-1">
                  <label className="text-[10px] font-bold text-cyan-500/50 uppercase tracking-widest">miRNA ID</label>
                  <input type="text" placeholder="e.g. hsa-mir-21"
                    value={customMirnaId} onChange={e => setCustomMirnaId(e.target.value)}
                    className="w-full p-2 glass-card bg-transparent border border-cyan-500/30 rounded-md focus:border-cyan-400 outline-none text-sm" />
                </div>

                {/* Healthy sequence */}
                <div className="flex flex-col gap-1">
                  <label className="text-[10px] font-bold text-cyan-500/50 uppercase tracking-widest">Healthy Sequence</label>
                  <textarea rows={3}
                    placeholder="Enter the original RNA sequence (A U C G)"
                    value={customHealthy}
                    onChange={e => setCustomHealthy(sanitise(e.target.value))}
                    className={`w-full p-2 glass-card bg-transparent border rounded-md outline-none font-mono text-xs ${
                      customHealthy && !/^[AUCGT]+$/.test(customHealthy)
                        ? 'border-red-500' : 'border-cyan-500/30 focus:border-cyan-400'
                    }`} />
                  <div className="text-[9px] text-gray-600 text-right">{customHealthy.length} chars</div>
                </div>

                {/* Diff indicator */}
                <div className={`text-[11px] font-bold text-center py-1 px-2 rounded bg-[#020818]/50 border border-white/5 ${diffColor}`}>
                  {diffMsg}
                </div>

                {/* Mutant sequence */}
                <div className="flex flex-col gap-1">
                  <label className="text-[10px] font-bold text-cyan-500/50 uppercase tracking-widest">Mutant Sequence</label>
                  <textarea rows={3}
                    placeholder="Enter the mutated RNA sequence (one base different)"
                    value={customMutant}
                    onChange={e => setCustomMutant(sanitise(e.target.value))}
                    className={`w-full p-2 glass-card bg-transparent border rounded-md outline-none font-mono text-xs ${
                      customMutant && !/^[AUCGT]+$/.test(customMutant)
                        ? 'border-red-500' : 'border-cyan-500/30 focus:border-cyan-400'
                    }`} />
                  <div className="text-[9px] text-gray-600 text-right">{customMutant.length} chars</div>
                </div>

                {/* SNP position */}
                <div className="flex flex-col gap-1">
                  <label className="text-[10px] font-bold text-cyan-500/50 uppercase tracking-widest">SNP Position (auto-detected)</label>
                  <input type="number" value={customSnpPos}
                    onChange={e => setCustomSnpPos(e.target.value)}
                    className="w-full p-2 glass-card bg-[#020818]/50 border border-cyan-500/30 rounded-md text-sm text-white focus:border-cyan-400 outline-none" />
                </div>
              </div>
            )}

            {/* ── FASTA flow ── */}
            {method === 'fasta' && (
              <div className="flex flex-col gap-4">
                <div className="text-[10px] font-mono text-gray-500 p-2 border border-gray-500/20 rounded bg-black/40">
                  Expected format:<br />
                  {`>healthy`}<br />[sequence]<br />{`>mutant`}<br />[sequence]
                </div>
                <textarea rows={8}
                  placeholder="Paste FASTA content here..."
                  value={fastaInput}
                  onChange={e => { setFastaInput(e.target.value); parseFasta(e.target.value); }}
                  className="w-full p-3 glass-card bg-transparent border border-cyan-500/30 focus:border-cyan-400 outline-none text-xs font-mono rounded-lg" />
                {customHealthy && (
                  <div className="text-[10px] text-green-400 font-mono flex items-center gap-2">
                    <Check size={10} /> Parsed: {customMirnaId || 'unnamed'}, {customHealthy.length}nt, diff@pos {customSnpPos}
                  </div>
                )}
              </div>
            )}

            {/* Navigation */}
            <div className="flex gap-3 mt-2">
              <button onClick={() => { setStep(1); setSelectedExample(null); setError(null); }}
                className="p-3 glass-card flex-1 hover:bg-gray-500/10 transition-colors text-xs font-bold uppercase tracking-wider text-gray-500">
                ← Back
              </button>
              <button
                disabled={
                  method === 'example'
                    ? selectedExample === null    // enabled as soon as selection made
                    : !isCustomValid              // custom requires valid sequences
                }
                onClick={() => setStep(3)}
                className="p-3 rounded-lg flex-[2] font-bold uppercase tracking-widest text-xs transition-all disabled:bg-gray-800 disabled:text-gray-600 disabled:cursor-not-allowed enabled:bg-cyan-500 enabled:text-[#020818] enabled:shadow-[0_0_20px_rgba(0,229,255,0.4)] enabled:hover:bg-cyan-400">
                Review Step →
              </button>
            </div>
          </motion.div>
        )}

        {/* ── STEP 3: Review & Submit ── */}
        {step === 3 && (
          <motion.div key="step3" initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.98 }} className="flex flex-col gap-6">
            <div className="p-5 glass-card border-cyan-500/30 bg-cyan-500/5 space-y-3 rounded-lg">
              <div className="text-[10px] font-bold text-cyan-400 uppercase tracking-widest mb-3">Vector Configuration Summary</div>
              <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                <div className="flex flex-col gap-1">
                  <span className="text-gray-500 text-[9px] uppercase">Target ID</span>
                  <span className="text-white truncate">
                    {method === 'example' ? selectedExample?.mirna_id : customMirnaId}
                  </span>
                </div>
                <div className="flex flex-col gap-1">
                  <span className="text-gray-500 text-[9px] uppercase">Method</span>
                  <span className="text-white capitalize">{method}</span>
                </div>
                <div className="flex flex-col gap-1">
                  <span className="text-gray-500 text-[9px] uppercase">Sequence Length</span>
                  <span className="text-white">
                    {method === 'example'
                      ? `${selectedExample?.seq_healthy.length ?? 0} nt`
                      : `${customHealthy.length} nt`}
                  </span>
                </div>
                <div className="flex flex-col gap-1">
                  <span className="text-gray-500 text-[9px] uppercase">SNP Index</span>
                  <span className="text-cyan-400 font-bold">
                    {method === 'example' ? selectedExample?.snp_pos : customSnpPos}
                  </span>
                </div>
              </div>
            </div>

            {error && (
              <div className="p-3 bg-red-500/10 border border-red-500/30 text-red-400 text-xs rounded-lg flex items-start gap-2">
                <span>⚠</span> {error}
              </div>
            )}

            <div className="flex flex-col gap-3">
              <button onClick={handlePredict} disabled={isSubmitting}
                className="w-full p-4 bg-[#00e5ff] text-[#020818] font-black uppercase tracking-[0.2em] rounded-xl hover:scale-[1.02] active:scale-[0.98] transition-all shadow-[0_0_30px_rgba(0,212,255,0.3)] disabled:opacity-60 disabled:cursor-not-allowed flex items-center justify-center gap-2">
                {isSubmitting
                  ? <><div className="w-4 h-4 border-2 border-[#020818] border-t-transparent rounded-full animate-spin" />Processing Tensors...</>
                  : 'Run Prediction Engine'}
              </button>
              <button onClick={() => { setStep(2); setError(null); }}
                className="text-gray-500 text-xs font-bold uppercase tracking-wider hover:text-gray-300 transition-colors">
                ← Modify Parameters
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
