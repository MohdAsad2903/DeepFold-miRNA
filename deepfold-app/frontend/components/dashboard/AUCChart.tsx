'use client';

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { ModelStats } from '@/lib/types';
import { useMemo } from 'react';

import { getModelName } from '@/lib/modelNames';

interface Props {
  data: ModelStats[];
}

export default function AUCChart({ data }: Props) {
  const sortedData = useMemo(() => {
    return [...data].sort((a, b) => a.auc_mean - b.auc_mean);
  }, [data]);

  const getColor = (auc: number) => {
    if (auc >= 0.70) return '#10b981'; // Emerald
    if (auc >= 0.62) return '#f59e0b'; // Amber
    return '#ef4444'; // Red
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const d = payload[0].payload;
      return (
        <div className="bg-deepspace-card border border-cyan-500/30 p-3 rounded shadow-xl max-w-xs">
          <p className="font-semibold text-white mb-2">{getModelName(d.name)}</p>
          <div className="text-xs text-gray-300 space-y-1">
            <p>AUC: <span className="text-cyan-400 font-mono">{(d.auc_mean ?? 0).toFixed(3)}</span> ± {(d.auc_std ?? 0).toFixed(3)}</p>
            <p>F1 Score: <span className="font-mono">{(d.f1 ?? 0).toFixed(3)}</span></p>
            <p>Accuracy: <span className="font-mono">{(d.accuracy ?? 0).toFixed(3)}</span></p>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="glass-card h-[400px] w-full p-6">
      <h3 className="text-lg font-display font-semibold mb-4 text-white">ROC-AUC Performance (5-Fold CV)</h3>
      <div className="h-full pb-8">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart 
            layout="vertical" 
            data={sortedData} 
            margin={{ top: 0, right: 30, left: 160, bottom: 0 }}
          >
            <XAxis type="number" domain={[0.5, 0.8]} hide />
            <YAxis 
              dataKey="name" 
              type="category" 
              axisLine={false} 
              tickLine={false}
              tick={{ fill: '#9ca3af', fontSize: 11 }}
              tickFormatter={(val) => getModelName(val)}
              width={160}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(0, 212, 255, 0.05)' }} />
            <Bar dataKey="auc_mean" radius={[0, 4, 4, 0]} barSize={16}>
              {sortedData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={getColor(entry.auc_mean)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
