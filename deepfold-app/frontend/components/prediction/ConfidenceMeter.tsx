'use client';

import { Canvas, useFrame } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import { useRef, useEffect } from 'react';
import * as THREE from 'three';
import { useSpring, animated } from '@react-spring/three';

interface Props {
  prob: number;
}

function GaugeModel({ prob }: { prob: number }) {
  const needleRef = useRef<THREE.Group>(null);
  
  // Spring animate the probability mapping (0 = Math.PI, 1 = 0)
  const { rotationZ } = useSpring({
    rotationZ: Math.PI - (prob * Math.PI),
    config: { tension: 80, friction: 15 }
  });

  return (
    <group position={[0, -0.5, 0]}>
       {/* Dial Arc Background */}
       <mesh rotation={[0, 0, 0]}>
          <ringGeometry args={[1.8, 2.0, 64, 1, 0, Math.PI]} />
          <meshBasicMaterial transparent opacity={0.15} color="#00fff0" side={THREE.DoubleSide} />
       </mesh>

       {/* Green Segment (0 - 0.50) */}
       <mesh rotation={[0, 0, Math.PI * 0.50]}>
          <ringGeometry args={[1.85, 1.95, 32, 1, 0, Math.PI * 0.50]} />
          <meshBasicMaterial color="#22d3ee" side={THREE.DoubleSide} />
       </mesh>
       {/* Red Segment (0.50 - 1.0) */}
       <mesh rotation={[0, 0, 0]}>
          <ringGeometry args={[1.85, 1.95, 32, 1, 0, Math.PI * 0.50]} />
          <meshBasicMaterial color="#ff2d55" side={THREE.DoubleSide} />
       </mesh>

       {/* Animated Needle */}
       <animated.group rotation-z={rotationZ}>
          <mesh position={[0.9, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
             <cylinderGeometry args={[0.02, 0.08, 1.8, 8]} />
             <meshStandardMaterial color="#ffffff" metalness={0.8} roughness={0.2} />
          </mesh>
          <mesh>
             <sphereGeometry args={[0.15, 16, 16]} />
             <meshStandardMaterial color="#00fff0" metalness={1} roughness={0} />
          </mesh>
       </animated.group>

       {/* Labels */}
       <Html position={[-2.2, 0.2, 0]} center><span className="text-[10px] text-[#22d3ee] font-mono font-bold">Benign</span></Html>
       <Html position={[2.2, 0.2, 0]} center><span className="text-[10px] text-[#ff2d55] font-mono font-bold">Pathogenic</span></Html>
       <Html position={[0, 2.0, 0]} center><span className="text-[10px] text-gray-500 font-mono">0.5</span></Html>
    </group>
  );
}

export default function ConfidenceMeter({ prob }: Props) {
  return (
    <div className="glass-card flex flex-col h-[180px] bg-[#020818] border-[#00fff0]/20 relative overflow-hidden mt-6">
      <div className="absolute top-3 left-4 z-10 text-xs text-cyan-500 uppercase tracking-widest font-bold">
        Threshold Gauge
      </div>
      
      <div className="flex-1 w-full h-full mt-4">
         <Canvas camera={{ position: [0, 0, 5], fov: 45 }} frameloop="demand">
            <ambientLight intensity={1.0} />
            <directionalLight position={[0, 0, 5]} intensity={1.5} />
            <GaugeModel prob={prob} />
         </Canvas>
      </div>
    </div>
  );
}
