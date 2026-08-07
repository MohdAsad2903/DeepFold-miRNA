'use client';

import { PredictionResponse } from '@/lib/types';
import { getModelName } from '@/lib/modelNames';

interface ResultCardProps {
  result: PredictionResponse;
}

export const ResultSkeleton = () => (
  <div className="glass-card p-6 space-y-6 w-full">
    <div className="w-32 h-32 rounded-full skeleton mx-auto" />
    <div className="h-8 w-48 rounded skeleton mx-auto" />
    <div className="space-y-3">
      <div className="h-4 w-full rounded skeleton" />
      <div className="h-4 w-5/6 rounded skeleton" />
      <div className="h-4 w-4/6 rounded skeleton" />
    </div>
  </div>
);

export default function ResultCard({ result }: ResultCardProps) {
  const isPathogenic   = result.prob_disease >= 0.5;
  const colorClass     = isPathogenic ? 'text-[#ef4444]' : 'text-[#10b981]';
  const borderClass    = isPathogenic ? 'border-red-500/40 bg-red-950/10' : 'border-green-500/40 bg-green-950/10';
  const barColor       = isPathogenic ? 'bg-[#ef4444]' : 'bg-[#10b981]';
  const verdictLabel   = isPathogenic ? 'PATHOGENIC' : 'BENIGN';
  const percentage     = Math.round(result.prob_disease * 100);

  // Build explanation bullets
  const bullets = result.shap_explanation && result.shap_explanation.length > 0
    ? result.shap_explanation.slice(0, 3)
    : null;

  return (
    <div className={`glass-card flex flex-col overflow-hidden ${borderClass} border`}>
      <h3 className="w-full text-center text-[10px] text-cyan-500/70 uppercase tracking-[0.2em] py-3 border-b border-cyan-500/5 font-bold bg-[#0a1229]/50">
        Prediction Engine Result
      </h3>

      <div className="px-6 pt-6 pb-2">
        {/* Probability bar */}
        <div className="flex items-center gap-3 mb-5">
          <div className="flex-1 h-4 bg-gray-900 rounded-full overflow-hidden">
            <div className={`h-full rounded-full transition-all duration-700 ${barColor}`}
              style={{ width: `${percentage}%` }} />
          </div>
          <span className={`text-lg font-black font-mono ${colorClass}`}>{percentage}%</span>
        </div>

        {/* Verdict */}
        <div className={`text-3xl font-black tracking-[0.15em] mb-1 ${colorClass}`}>
          {verdictLabel}
        </div>
        <div className={`text-sm mb-5 font-mono ${colorClass} opacity-75`}>
          {percentage}% confidence — {isPathogenic ? 'Pathogenic' : 'Benign'}
        </div>

        {/* Model scores */}
        <div className="space-y-[6px] mb-6 text-xs text-gray-400 font-mono">
          {[
            [getModelName('CNN_v4'), result.base_probs.CNN_v4],
            [getModelName('kmer_XGB'),  result.base_probs.kmer_XGB],
            [getModelName('MFE_XGB'), result.base_probs.MFE_XGB],
            [getModelName('GCN'),     result.base_probs.GCN],
            [getModelName('RNAFM_XGB'), result.base_probs.RNAFM_XGB],
          ].map(([name, val]) => (
            <div key={name as string} className="flex justify-between items-center">
              <span className="text-gray-500">{name as string}</span>
              <span className="text-gray-300">{(val as number).toFixed(2)}</span>
            </div>
          ))}
        </div>

        {/* Why this prediction */}
        <div className="pt-4 border-t border-white/5">
          <div className={`text-xs font-bold uppercase tracking-widest mb-2 ${colorClass}`}>
            Why this prediction?
          </div>

          {/* Verdict sentence */}
          {result.verdict_explanation && (
            <p className="text-xs text-gray-300 leading-relaxed mb-4 p-3 bg-[#020818]/60 rounded-lg border border-cyan-500/10">
              {result.verdict_explanation}
            </p>
          )}

          {/* Bullet points with direction */}
          {bullets && (
            <div className="flex flex-col gap-2">
              {bullets.map((exp, i) => (
                <div key={i} className="flex items-start gap-2 text-xs">
                  <span className={`mt-0.5 font-bold shrink-0 ${
                    exp.direction === 'pathogenic' ? 'text-[#ef4444]' : 'text-[#10b981]'
                  }`}>
                    {exp.direction === 'pathogenic' ? '↑' : '↓'}
                  </span>
                  <span className="text-gray-300 leading-snug flex-1">{exp.plain_label}</span>
                  <span className={`text-[9px] font-mono shrink-0 mt-0.5 px-1.5 py-0.5 rounded ${
                    exp.direction === 'pathogenic'
                      ? 'bg-red-900/40 text-red-400'
                      : 'bg-green-900/40 text-green-400'
                  }`}>
                    {exp.direction === 'pathogenic' ? 'path' : 'benign'}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Fallback if no SHAP */}
          {!bullets && !result.verdict_explanation && (
            <p className="text-xs text-gray-500 italic">No explanation available for this prediction.</p>
          )}
        </div>
      </div>

      <div className="px-6 pb-4 pt-2 text-[9px] text-gray-600 font-mono flex justify-between">
        <span>Confidence: {result.confidence}</span>
        <span>{result.processing_time_ms}ms</span>
      </div>
    </div>
  );
}
