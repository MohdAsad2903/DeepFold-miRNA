export default function ContactMapViewer() {
  return (
    <div className="flex flex-col items-center justify-center h-full bg-deepspace-cardSolid/50 rounded-xl border border-cyan-500/10 p-4">
      <div className="w-full aspect-square border border-cyan-500/30 rounded flex items-center justify-center relative overflow-hidden bg-[#0a0f1c]">
        {/* Mock 2D Grid representation */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#00d4ff0a_1px,transparent_1px),linear-gradient(to_bottom,#00d4ff0a_1px,transparent_1px)] bg-[size:10px_10px]"></div>
        <div className="absolute top-1/4 left-1/4 w-1/2 h-1/2 rounded-full blur-[40px] bg-cyan-500/10"></div>
        <div className="absolute z-10 text-xs font-mono text-cyan-400/50 rotate-[-45deg]">
          UFold 128x128 Tensor Viewer
        </div>
      </div>
      <p className="text-xs text-gray-500 mt-2">Simulated Contact Predictor</p>
    </div>
  );
}
