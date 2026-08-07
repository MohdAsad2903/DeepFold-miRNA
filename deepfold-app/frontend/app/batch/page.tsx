'use client';

import { useState } from 'react';
import { Upload, FileText, Download, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { predictBatch } from '@/lib/api';

export default function BatchPage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const data = await predictBatch(formData);
      setResults(data.results);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const downloadResults = () => {
    if (!results) return;
    const headers = Object.keys(results[0]).join(',');
    const rows = results.map(r => Object.values(r).join(',')).join('\n');
    const blob = new Blob([`${headers}\n${rows}`], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `deepfold_batch_${Date.now()}.csv`;
    a.click();
  };

  return (
    <div className="container mx-auto px-4 py-12 max-w-5xl relative z-10">
      <div className="mb-12 text-center">
        <h1 className="text-4xl font-display font-black text-white mb-4 tracking-tight">High-Throughput Batch Engine</h1>
        <p className="text-gray-400 max-w-2xl mx-auto">
          Upload a CSV of miRNA variants for concurrent ensemble processing. Supports up to 500 rows per batch.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Upload Panel */}
        <div className="glass-card p-8 flex flex-col items-center justify-center min-h-[300px] border-dashed border-cyan-500/30">
          <Upload className="w-12 h-12 text-cyan-500/50 mb-4" />
          <input 
            type="file" 
            id="file-upload" 
            className="hidden" 
            accept=".csv" 
            onChange={handleFileChange}
          />
          <label 
            htmlFor="file-upload"
            className="px-6 py-3 bg-cyan-500/10 border border-cyan-500/30 rounded-lg text-cyan-400 font-bold cursor-pointer hover:bg-cyan-500/20 transition-all mb-4"
          >
            {file ? file.name : 'Select CSV File'}
          </label>
          <p className="text-[10px] text-gray-500 uppercase tracking-widest mb-8">Format: mirna_id, seq_healthy, seq_mutant</p>
          
          <button
            onClick={handleUpload}
            disabled={!file || loading}
            className="w-full py-4 bg-cyan-500 text-[#020818] font-black uppercase tracking-widest rounded-xl disabled:opacity-50 shadow-[0_0_20px_rgba(34,211,238,0.2)]"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <Loader2 className="animate-spin" /> RUNNING_ENSEMBLE_STACK...
              </span>
            ) : 'Execute Batch Analysis'}
          </button>
        </div>

        {/* Status/Results Panel */}
        <div className="glass-card p-8 flex flex-col">
          <h3 className="text-sm font-bold text-white uppercase tracking-widest mb-6 flex items-center gap-2">
            <FileText size={16} className="text-cyan-500" /> Process_Monitor
          </h3>

          <div className="flex-1 flex flex-col items-center justify-center text-center">
            <AnimatePresence mode="wait">
              {!results && !error && !loading && (
                <motion.div initial={{ opacity:0 }} animate={{ opacity:1 }} className="text-gray-600">
                  <p className="text-xs italic">Awaiting file upload...</p>
                </motion.div>
              )}

              {loading && (
                <motion.div initial={{ opacity:0 }} animate={{ opacity:1 }} className="space-y-4 w-full">
                  <div className="h-2 w-full bg-gray-800 rounded-full overflow-hidden">
                    <motion.div 
                      className="h-full bg-cyan-500"
                      animate={{ x: ['-100%', '100%'] }}
                      transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                    />
                  </div>
                  <p className="text-[10px] text-cyan-500 font-mono">PARALLEL_VECTOR_PROCESSING_IN_PROGRESS</p>
                </motion.div>
              )}

              {error && (
                <motion.div initial={{ opacity:0 }} animate={{ opacity:1 }} className="text-red-500 flex flex-col items-center">
                  <AlertCircle size={32} className="mb-2" />
                  <p className="text-xs font-bold uppercase">{error}</p>
                </motion.div>
              )}

              {results && (
                <motion.div initial={{ opacity:0, scale:0.9 }} animate={{ opacity:1, scale:1 }} className="w-full flex flex-col items-center">
                  <CheckCircle2 size={48} className="text-green-500 mb-4" />
                  <h4 className="text-xl font-bold text-white mb-1">{results.length} Variants Processed</h4>
                  <p className="text-xs text-gray-500 mb-8 uppercase tracking-widest font-bold">100% SUCCESS_RATE</p>
                  
                  <button 
                    onClick={downloadResults}
                    className="flex items-center gap-2 px-8 py-3 bg-green-500/10 border border-green-500/30 text-green-400 rounded-lg font-bold hover:bg-green-500/20 transition-all"
                  >
                    <Download size={18} /> Export Diagnostic CSV
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>

      {results && (
        <div className="mt-12 glass-card overflow-hidden">
          <div className="p-4 border-b border-white/5 bg-white/5 flex items-center justify-between">
            <span className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">Preview_Top_10_Results</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-[11px] border-collapse">
              <thead>
                <tr className="border-b border-white/5 bg-black/20">
                  <th className="p-4 text-cyan-500 uppercase tracking-widest">miRNA_ID</th>
                  <th className="p-4 text-white uppercase tracking-widest">Label</th>
                  <th className="p-4 text-white uppercase tracking-widest text-right">Patho_Prob</th>
                  <th className="p-4 text-white uppercase tracking-widest text-right">Agreement</th>
                </tr>
              </thead>
              <tbody>
                {results.slice(0, 10).map((r, i) => (
                  <tr key={i} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                    <td className="p-4 font-mono text-gray-400">{r.mirna_id}</td>
                    <td className="p-4 font-bold">
                       <span className={r.label.toLowerCase().includes('pathogenic') ? 'text-red-400' : 'text-green-400'}>
                         {r.label.toUpperCase()}
                       </span>
                    </td>
                    <td className="p-4 font-mono text-right">{r.prob_disease.toFixed(4)}</td>
                    <td className="p-4 text-right">
                       <span className={`px-2 py-0.5 rounded text-[9px] font-bold ${
                         r.disagreement_level === 'Low' ? 'bg-green-500/10 text-green-400' : 'bg-amber-500/10 text-amber-400'
                       }`}>
                         {r.disagreement_level}
                       </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
