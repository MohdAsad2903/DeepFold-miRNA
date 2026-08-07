import { Loader2 } from 'lucide-react';

export default function LoadingSpinner({ text = "Processing..." }: { text?: string }) {
  return (
    <div className="flex flex-col items-center justify-center p-8 space-y-4">
      <div className="relative">
        <div className="absolute inset-0 rounded-full blur-md bg-cyan-500/50 animate-pulse"></div>
        <Loader2 className="relative animate-spin text-cyan-400" size={48} />
      </div>
      <p className="text-cyan-200 animate-pulse font-medium">{text}</p>
    </div>
  );
}
