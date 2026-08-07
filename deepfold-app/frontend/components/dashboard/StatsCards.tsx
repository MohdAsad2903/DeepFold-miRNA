'use client';

import { Activity, Beaker, Zap, Clock } from 'lucide-react';
import { usePredictionStore } from '@/lib/predictionStore';

export default function StatsCards() {
  const { stats } = usePredictionStore();
  
  const pathRate = stats.total > 0 ? Math.round((stats.pathogenic / stats.total) * 100) : 0;
  const confRate = stats.total > 0 ? Math.round((stats.high_confidence / stats.total) * 100) : 0;
  
  const statCards = [
    { title: 'Local Predictions', value: stats.total, icon: <Activity className="text-cyan-400" size={20} /> },
    { title: 'Pathogenic Rate', value: `${pathRate}%`, icon: <Beaker className="text-red-400" size={20} /> },
    { title: 'High Confidence', value: `${confRate}%`, icon: <Zap className="text-amber-400" size={20} /> },
    { title: 'Avg Prob', value: `${stats.mean_prob.toFixed(2)}`, icon: <Clock className="text-emerald-400" size={20} /> },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 w-full">
      {statCards.map((stat, i) => (
        <div key={i} className="glass-card flex items-center p-4 shadow-[0_0_15px_rgba(0,0,0,0.5)] bg-[#020818]/60">
          <div className="p-3 bg-[#0a1229] border border-cyan-500/20 rounded-lg mr-4">
            {stat.icon}
          </div>
          <div>
            <p className="text-[10px] text-gray-400 uppercase font-bold tracking-widest">{stat.title}</p>
            <p className="text-2xl font-bold text-white font-mono mt-1">{stat.value}</p>
          </div>
        </div>
      ))}
    </div>
  );
}
