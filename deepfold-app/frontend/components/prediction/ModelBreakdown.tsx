'use client';

import { BaseModelProbs } from '@/lib/types';
import { useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Html, OrbitControls } from '@react-three/drei';
import { useSpring, animated } from '@react-spring/three';
import { getModelName, getModelDescription } from '@/lib/modelNames';

function Bar3D({ model, index, total }: { model: any, index: number, total: number }) {
  const [hovered, setHover] = useState(false);
  
  const { height, color } = useSpring({
    height: model.prob * 3 + 0.1, // mapped visually
    color: hovered ? '#ffffff' : model.color,
    config: { tension: 120, friction: 14 }
  });

  const xPos = (index - total / 2 + 0.5) * 1.2;

  return (
    <group position={[xPos, 0, 0]}>
      <animated.mesh 
         position-y={height.to(h => h / 2)}
         onPointerOver={(e) => { e.stopPropagation(); setHover(true); }}
         onPointerOut={() => setHover(false)}
      >
         <boxGeometry args={[0.8, 1, 0.8]} />
         <animated.meshStandardMaterial color={color} roughness={0.2} metalness={0.8} emissive={color} emissiveIntensity={0.2} />
      </animated.mesh>

      {/* Probability Label Above Bar */}
      <animated.group position-y={height.to(h => h + 0.2)}>
        <Html center className="pointer-events-none">
           <span className="text-[10px] text-gray-300 font-mono font-bold drop-shadow-md">
             {model.prob.toFixed(2)}
           </span>
        </Html>
      </animated.group>

      {/* Model Name Below Bar */}
      <Html position={[0, -0.3, 0]} center className="pointer-events-none">
         <div className="w-16 text-center text-[9px] text-gray-500 uppercase tracking-widest font-bold leading-tight break-words">
            {model.shortLabel}
         </div>
      </Html>

      {hovered && (
        <Html position={[0, 2.5, 0]} center className="pointer-events-none z-50">
           <div className="w-32 bg-[#020818]/90 backdrop-blur-md border border-[#00fff0]/50 p-2 rounded text-[10px] text-[#00fff0] text-center shadow-[0_0_15px_rgba(0,255,240,0.3)]">
              {model.desc}
           </div>
        </Html>
      )}
    </group>
  );
}

interface Props {
  probs: BaseModelProbs;
}

export default function ModelBreakdown({ probs }: Props) {
  const models = [
    { key: 'CNN_v4',   color: '#0ea5e9' },
    { key: 'GCN',      color: '#a855f7' },
    { key: 'kmer_XGB', color: '#22d3ee' },
    { key: 'RNAFM_XGB', color: '#fbbf24' },
    { key: 'MFE_XGB',  color: '#ff2d55' }
  ].map(m => ({
    ...m,
    shortLabel: getModelName(m.key).split(' ')[0], // First word as short label
    fullLabel: getModelName(m.key),
    desc: getModelDescription(m.key)
  }));

  const data = models.map(m => ({ ...m, prob: probs[m.key as keyof BaseModelProbs] }));

  return (
    <div className="glass-card flex flex-col h-[250px] bg-[#020818] border-[#00fff0]/20 relative">
      <h3 className="text-xs text-cyan-500 uppercase tracking-widest px-4 py-3 border-b border-cyan-500/10 font-bold bg-[#0a1229]/50">
         Base Model Stack
      </h3>
      
      <div className="flex-1 w-full h-full cursor-move">
         <Canvas camera={{ position: [0, 4, 6], fov: 40 }} frameloop="demand">
            <ambientLight intensity={0.5} />
            <directionalLight position={[5, 10, 5]} intensity={1} color="#ffffff" />
            <group position={[0, -1, 0]}>
               {data.map((m, i) => (
                  <Bar3D key={m.key} model={m} index={i} total={data.length} />
               ))}
               {/* Grid floor underneath bars */}
               <gridHelper args={[8, 8, '#334155', '#1e293b']} position={[0, 0, 0]} />
            </group>
            <OrbitControls enableZoom={false} enablePan={false} maxPolarAngle={Math.PI/2 - 0.1} minPolarAngle={0} autoRotate autoRotateSpeed={0.5} />
         </Canvas>
      </div>

      <div className="absolute top-10 right-2 flex gap-1 items-end h-8">
         <div className="w-1 h-2 bg-gray-700"></div><div className="w-1 h-4 bg-gray-600"></div><div className="w-1 h-6 bg-cyan-500"></div>
      </div>
    </div>
  );
}
