import { PredictionHistoryEntry } from '@/lib/types';
import { PlayCircle } from 'lucide-react';
import Link from 'next/link';
import { usePredictionStore } from '@/lib/predictionStore';

export default function PredictionHistory() {
  const { history } = usePredictionStore();

  if (history.length === 0) {
    return (
      <div className="glass-card flex flex-col items-center justify-center py-16 text-center">
        <p className="text-gray-400 mb-4">No predictions made yet in this session.</p>
        <Link href="/predict" className="flex items-center gap-2 text-cyan-400 hover:text-cyan-300">
          <PlayCircle /> Run your first prediction
        </Link>
      </div>
    );
  }

  return (
    <div className="glass-card overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="bg-[#111827] text-gray-400 uppercase border-b border-cyan-500/10">
            <tr>
              <th className="px-4 py-3">Time</th>
              <th className="px-4 py-3">miRNA ID</th>
              <th className="px-4 py-3">Probability</th>
              <th className="px-4 py-3">Result</th>
              <th className="px-4 py-3">Confidence</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-cyan-500/5">
            {history.map((item) => (
              <tr key={item.id} className="hover:bg-cyan-500/5 text-gray-300 transition-colors">
                <td className="px-4 py-3 font-mono text-xs">{new Date(item.timestamp).toLocaleTimeString()}</td>
                <td className="px-4 py-3 font-medium text-white">{item.mirna_id || 'Unknown'}</td>
                <td className="px-4 py-3 font-mono text-cyan-200">{item.prob_disease.toFixed(4)}</td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-1 rounded text-xs ${
                    item.prob_disease >= 0.50 ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'
                  }`}>
                    {item.label}
                  </span>
                </td>
                <td className="px-4 py-3 text-xs">{item.confidence}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
