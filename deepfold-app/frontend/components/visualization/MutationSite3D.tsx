'use client';

import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

const TerrainShaderMaterial = new THREE.ShaderMaterial({
  uniforms: { maxHeight: { value: 2.0 } },
  vertexShader: `
    varying vec3 vPos;
    void main() {
      vPos = position;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    varying vec3 vPos;
    uniform float maxHeight;
    void main() {
      float h = vPos.z / maxHeight;
      vec3 deep = vec3(0.01, 0.03, 0.1);
      vec3 mid = vec3(0.0, 0.9, 0.85);
      vec3 peak = vec3(1.0, 1.0, 1.0);
      vec3 color = h < 0.5 ? mix(deep, mid, h * 2.0) : mix(mid, peak, (h-0.5)*2.0);
      float gridLine = max(smoothstep(0.95, 1.0, mod(vPos.x * 2.0, 1.0)), smoothstep(0.95, 1.0, mod(vPos.y * 2.0, 1.0)));
      gl_FragColor = vec4(mix(color, vec3(0.0), gridLine * 0.2), 0.9);
    }
  `,
  transparent: true,
  side: THREE.DoubleSide
});

function Terrain({ snpPos }: { snpPos?: number }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const scanPlaneRef = useRef<THREE.Mesh>(null);
  
  // Reduced segmentation from 127 to 64
  const size = 63;
  const geometry = useMemo(() => {
    const geo = new THREE.PlaneGeometry(16, 16, size, size);
    const pos = geo.attributes.position;
    for (let i = 0; i <= size; i++) {
      for (let j = 0; j <= size; j++) {
        const idx = i * (size+1) + j;
        let h = 0.1;
        if (Math.abs(i - j) < 3) h += 0.4;
        const dX = Math.abs(i - (snpPos !== undefined ? (snpPos % 64) : 32));
        const dY = Math.abs(j - (snpPos !== undefined ? Math.floor(snpPos / 2) : 32));
        if (dX < 2 || dY < 2) h += 0.8;
        pos.setZ(idx, h);
      }
    }
    geo.computeVertexNormals();
    return geo;
  }, [snpPos]);

  const lastInteraction = useRef(Date.now());
  useFrame((state) => {
    const idle = Date.now() - lastInteraction.current > 3000;
    if (idle && Math.random() > 0.6) return; // Heavy throttling on terrain
    
    if (meshRef.current) meshRef.current.rotation.z = state.clock.elapsedTime * 0.03;
    if (scanPlaneRef.current) scanPlaneRef.current.position.y = (state.clock.elapsedTime * 3) % 16 - 8;
  });

  return (
    <group rotation={[-Math.PI/4, 0, 0]} onPointerMove={() => (lastInteraction.current = Date.now())}>
      <mesh ref={meshRef} geometry={geometry}>
        <primitive object={TerrainShaderMaterial} attach="material" />
      </mesh>
      <mesh ref={scanPlaneRef} rotation={[Math.PI/2, 0, 0]} position={[0,0,0.5]}>
         <planeGeometry args={[16, 0.08]} />
         <meshBasicMaterial color="#00fff0" transparent opacity={0.4} />
      </mesh>
    </group>
  );
}

export default function MutationSite3D({ snpPos }: { snpPos?: number }) {
  return (
    <div className="w-full h-[300px] glass-card border-white/5 bg-[#020818]/80 overflow-hidden">
      <Canvas camera={{ position: [0, -12, 10], fov: 45 }} frameloop="demand" gl={{ antialias: false }}>
        <ambientLight intensity={0.5} />
        <Terrain snpPos={snpPos} />
        <OrbitControls enableZoom={false} enablePan={false} />
      </Canvas>
      <div className="absolute top-2 left-2 text-[9px] font-mono text-cyan-500/50 uppercase tracking-widest pointer-events-none">
        [STRUCTURAL_TOPOLOGY_VIEWER]
      </div>
    </div>
  );
}
