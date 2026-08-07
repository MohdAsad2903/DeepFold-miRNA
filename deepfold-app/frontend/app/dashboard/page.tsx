'use client';

import { useEffect, useState } from 'react';
import { getModelStats } from '@/lib/api';
import { ModelStats } from '@/lib/types';
import { usePredictionStore } from '@/lib/predictionStore';
import StatsCards from '@/components/dashboard/StatsCards';
import PredictionHistory from '@/components/dashboard/PredictionHistory';
import AUCChart from '@/components/dashboard/AUCChart';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell
} from 'recharts';

// ──────────────────────────────────────────────
// Chart 1: Score distribution histogram
// ──────────────────────────────────────────────
function ScoreDistributionChart({ history }: { history: any[] }) {
  const bins = Array.from({ length: 10 }, (_, i) => ({
    range: `${(i * 0.1).toFixed(1)}–${((i + 1) * 0.1).toFixed(1)}`,
    count: 0,
    isPath: i >= 5,
  }));
  history.forEach(h => {
    const idx = Math.min(9, Math.floor(h.prob_disease * 10));
    bins[idx].count++;
  });

  return (
    <div className="glass-card p-6 flex flex-col gap-4">
      <h3 className="text-lg font-bold text-white uppercase tracking-widest flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-cyan-500 animate-pulse" />
        Prediction Score Distribution
      </h3>
      <p className="text-[10px] text-gray-500 uppercase tracking-tighter -mt-2">
        Distribution of pathogenicity probability across all stored predictions.
      </p>
      {history.length === 0 ? (
        <div className="h-[260px] flex items-center justify-center text-gray-600 text-sm">
          Make predictions to see distribution
        </div>
      ) : (
        <div className="h-[260px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={bins} margin={{ top: 5, right: 5, left: -20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#22d3ee15" />
              <XAxis dataKey="range" stroke="#555" fontSize={9} angle={-45} textAnchor="end" />
              <YAxis stroke="#555" fontSize={10} allowDecimals={false} />
              <Tooltip
                contentStyle={{ backgroundColor: '#020818', border: '1px solid #22d3ee40', fontSize: '11px' }}
                formatter={(v: any) => [v, 'Count']}
              />
              <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                {bins.map((b, i) => (
                  <Cell key={i} fill={b.isPath ? '#ef4444' : '#10b981'} fillOpacity={0.8} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

// ──────────────────────────────────────────────
// Chart 2: Prediction history timeline
// ──────────────────────────────────────────────
function HistoryTimelineChart({ history }: { history: any[] }) {
  const data = [...history].reverse().map((h, i) => ({
    idx: i + 1,
    prob: h.prob_disease,
    color: h.prob_disease >= 0.5 ? '#ef4444' : '#10b981',
    label: h.label,
  }));

  const CustomDot = (props: any) => {
    const { cx, cy, payload } = props;
    return <circle cx={cx} cy={cy} r={4} fill={payload.color} stroke="none" />;
  };

  return (
    <div className="glass-card p-6 flex flex-col gap-4">
      <h3 className="text-lg font-bold text-white uppercase tracking-widest flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-cyan-500 animate-pulse" />
        Prediction History
      </h3>
      <p className="text-[10px] text-gray-500 uppercase tracking-tighter -mt-2">
        Pathogenicity score over time. Red = Pathogenic, Green = Benign.
      </p>
      {data.length < 2 ? (
        <div className="h-[260px] flex items-center justify-center text-gray-600 text-sm">
          {data.length === 0 ? 'Make predictions to see trend' : 'Make more predictions to see the trend'}
        </div>
      ) : (
        <div className="h-[260px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#22d3ee15" />
              <XAxis dataKey="idx" stroke="#555" fontSize={10} label={{ value: 'Prediction #', position: 'insideBottomRight', offset: -5, fill: '#555', fontSize: 10 }} />
              <YAxis stroke="#555" fontSize={10} domain={[0, 1]} />
              <Tooltip
                contentStyle={{ backgroundColor: '#020818', border: '1px solid #22d3ee40', fontSize: '11px' }}
                formatter={(v: any) => [Number(v).toFixed(3), 'Score']}
              />
              {/* Decision boundary */}
              <Line type="monotone" dataKey={() => 0.5} stroke="#555" strokeDasharray="4 4" dot={false} name="Threshold" strokeWidth={1} />
              <Line type="monotone" dataKey="prob" stroke="#22d3ee" strokeWidth={2} dot={<CustomDot />} name="Score" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

// ──────────────────────────────────────────────
// Chart 3: Label distribution donut
// ──────────────────────────────────────────────
function LabelDonutChart({ history }: { history: any[] }) {
  const pathogenic = history.filter(h => h.label === 'Pathogenic').length;
  const benign     = history.filter(h => h.label === 'Benign').length;
  const data = [
    { name: 'Pathogenic', value: pathogenic, color: '#ef4444' },
    { name: 'Benign',     value: benign,     color: '#10b981' },
  ];

  return (
    <div className="glass-card p-6 flex flex-col gap-4">
      <h3 className="text-lg font-bold text-white uppercase tracking-widest flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-cyan-500 animate-pulse" />
        Classification Summary
      </h3>
      <p className="text-[10px] text-gray-500 uppercase tracking-tighter -mt-2">
        Split of Pathogenic vs Benign predictions.
      </p>
      {history.length === 0 ? (
        <div className="h-[260px] flex items-center justify-center text-gray-600 text-sm">
          Make predictions to see breakdown
        </div>
      ) : (
        <div className="h-[260px] w-full relative">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={data} cx="50%" cy="50%" innerRadius={60} outerRadius={90} paddingAngle={4} dataKey="value">
                {data.map((entry, i) => <Cell key={i} fill={entry.color} fillOpacity={0.85} />)}
              </Pie>
              <Tooltip
                contentStyle={{ backgroundColor: '#020818', border: '1px solid #22d3ee40', fontSize: '11px' }}
              />
              <Legend iconType="circle" iconSize={10} wrapperStyle={{ fontSize: '11px', color: '#aaa' }} />
            </PieChart>
          </ResponsiveContainer>
          {/* Center text */}
          <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
            <span className="text-2xl font-black text-white">{history.length}</span>
            <span className="text-[9px] text-gray-500 uppercase tracking-widest">total</span>
          </div>
        </div>
      )}
    </div>
  );
}

// ──────────────────────────────────────────────
// Main Dashboard Page
// ──────────────────────────────────────────────
export default function DashboardPage() {
  const [stats, setStats]       = useState<ModelStats[]>([]);
  const { history }             = usePredictionStore();

  useEffect(() => {
    getModelStats().then(setStats).catch(() => {});
  }, []);

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <div className="mb-8 flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-display font-bold text-white mb-2">Analytics Dashboard</h1>
          <p className="text-gray-400">Prediction history and model performance metrics.</p>
        </div>
      </div>

      {/* Live stats cards */}
      <div className="mb-8">
        <StatsCards />
      </div>

      {/* Row 1: recent predictions + model comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div className="flex flex-col gap-4">
          <h3 className="text-xl font-display font-semibold text-white">Recent Predictions</h3>
          <PredictionHistory />
        </div>
        <div className="flex flex-col gap-4">
          <h3 className="text-xl font-display font-semibold text-white">Model Comparison</h3>
          {stats.length > 0 ? (
            <AUCChart data={stats} />
          ) : (
            <div className="glass-card h-[400px] flex items-center justify-center text-gray-500 text-sm">
              Loading metrics...
            </div>
          )}
        </div>
      </div>

      {/* Row 2: three new charts from stored predictions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
        <ScoreDistributionChart history={history} />
        <HistoryTimelineChart   history={history} />
        <LabelDonutChart        history={history} />
      </div>

    </div>
  );
}

