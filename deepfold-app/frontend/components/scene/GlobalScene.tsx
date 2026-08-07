'use client';

import { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { Bloom, EffectComposer, Noise, Vignette } from '@react-three/postprocessing';
import { usePathname } from 'next/navigation';

function Starfield() {
  const count = 2000;
  const [positions, sizes] = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const siz = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const radius = 20 + Math.random() * 80;
      const theta = 2 * Math.PI * Math.random();
      const phi = Math.acos(2 * Math.random() - 1);
      pos[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      pos[i * 3 + 2] = radius * Math.cos(phi);
      siz[i] = Math.random() * 0.12;
    }
    return [pos, siz];
  }, [count]);

  const pointsRef = useRef<THREE.Points>(null);
  useFrame((state) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y = state.clock.elapsedTime * 0.005;
    }
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={count} array={positions} itemSize={3} />
        <bufferAttribute attach="attributes-size" count={count} array={sizes} itemSize={1} />
      </bufferGeometry>
      <pointsMaterial size={0.12} color="#00fff0" transparent opacity={0.4} sizeAttenuation={true} blending={THREE.AdditiveBlending} />
    </points>
  );
}

function GridFloor() {
  const gridRef = useRef<THREE.GridHelper>(null);
  useFrame((state) => {
    if (gridRef.current) {
      gridRef.current.position.z = (state.clock.elapsedTime * 1.5) % 10;
    }
  });
  return (
    <group position={[0, -15, 0]}>
      <gridHelper ref={gridRef} args={[200, 40, '#00fff0', '#0ea5e9']} />
    </group>
  );
}

function SceneContents() {
  return (
    <>
      <color attach="background" args={['#020818']} />
      <fogExp2 attach="fog" args={['#020818', 0.015]} />
      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 10, 5]} intensity={1} color="#00fff0" />
      <Starfield />
      <GridFloor />
      <EffectComposer>
        <Bloom luminanceThreshold={0.4} mipmapBlur intensity={0.5} />
        <Vignette eskil={false} offset={0.1} darkness={1.1} />
        <Noise opacity={0.02} />
      </EffectComposer>
    </>
  );
}

const CSSBackground = () => (
  <div className="fixed inset-0 z-[-1] bg-[#020818] overflow-hidden pointer-events-none">
    <div className="bg-pulse-orb w-[600px] h-[600px] bg-cyan-500/10 top-[-10%] left-[-10%]" />
    <div className="bg-pulse-orb w-[800px] h-[800px] bg-violet-500/10 bottom-[-20%] right-[-10%]" style={{ animationDelay: '-5s' }} />
    <div className="bg-pulse-orb w-[400px] h-[400px] bg-blue-500/10 top-[40%] right-[20%]" style={{ animationDelay: '-2s' }} />
  </div>
);

export default function GlobalScene() {
  const pathname = usePathname();
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const handleVisibility = () => setIsVisible(document.visibilityState === 'visible');
    document.addEventListener('visibilitychange', handleVisibility);
    return () => document.removeEventListener('visibilitychange', handleVisibility);
  }, []);

  if (pathname !== '/') {
    return <CSSBackground />;
  }

  return (
    <div className="fixed inset-0 z-0 pointer-events-none">
      <Canvas 
        frameloop={isVisible ? "always" : "demand"}
        camera={{ position: [0, 0, 10], fov: 60 }}
        dpr={[1, 1.2]} 
        gl={{ powerPreference: "high-performance", antialias: false }}
      >
        {isVisible && <SceneContents />}
      </Canvas>
    </div>
  );
}
