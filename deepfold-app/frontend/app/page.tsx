'use client';

import Link from 'next/link';
import { ArrowRight } from 'lucide-react';
import { Suspense, useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { Bloom, EffectComposer } from '@react-three/postprocessing';
import { motion } from 'framer-motion';
import dynamic from 'next/dynamic';

// Optimized Hero Molecule with instancedMesh
function HeroMolecule() {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const count = 80;
  const dummy = useMemo(() => new THREE.Object3D(), []);

  const particles = useMemo(() => {
    const temp = [];
    for (let i = 0; i < count; i++) {
      const angle = (i / count) * Math.PI * 10;
      const x = Math.cos(angle) * 3;
      const z = Math.sin(angle) * 3;
      const y = (i - count / 2) * 0.2;
      temp.push({ x, y, z, color: new THREE.Color(['#00fff0', '#a855f7', '#22d3ee', '#0ea5e9'][i % 4]) });
    }
    return temp;
  }, []);

  useFrame((state, delta) => {
    if (!meshRef.current) return;
    const time = state.clock.elapsedTime;
    
    particles.forEach((p, i) => {
      dummy.position.set(p.x + Math.sin(time + i * 0.2) * 0.2, p.y, p.z + Math.cos(time + i * 0.3) * 0.2);
      dummy.rotation.y = time * 0.2;
      dummy.updateMatrix();
      meshRef.current!.setMatrixAt(i, dummy.matrix);
      meshRef.current!.setColorAt(i, p.color);
    });
    
    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true;
    meshRef.current.rotation.y += delta * 0.1;
  });

  return (
    <instancedMesh ref={meshRef} args={[null as any, null as any, count]}>
      <sphereGeometry args={[0.06, 6, 6]} />
      <meshStandardMaterial roughness={0.2} metalness={0.8} />
    </instancedMesh>
  );
}

const PipelineGraph = dynamic(() => import('@/components/visualization/PipelineGraph'), { 
  ssr: false, 
  loading: () => <div className="w-full h-full skeleton rounded-2xl" /> 
});

export default function LandingPage() {
  return (
    <div className="relative min-h-screen flex flex-col pt-16">
      <section className="relative flex-1 flex flex-col items-center justify-center py-20 px-4 z-10 text-center min-h-[80vh]">
        <div className="absolute inset-0 pointer-events-none z-0 opacity-60">
           <Canvas camera={{ position: [0, 0, 10], fov: 50 }} dpr={[1, 1]} gl={{ antialias: false }}>
              <ambientLight intensity={0.5} />
              <pointLight position={[10, 10, 10]} intensity={1} color="#00fff0" />
              <Suspense fallback={null}>
                 <HeroMolecule />
                 <EffectComposer>
                   <Bloom luminanceThreshold={0.4} mipmapBlur intensity={0.5} />
                 </EffectComposer>
              </Suspense>
           </Canvas>
        </div>

        <div className="max-w-4xl mx-auto space-y-6 relative z-10 pointer-events-auto flex flex-col items-center">
          <div className="relative flex items-center justify-center py-10">
             <h1 className="text-7xl md:text-8xl font-display font-black tracking-tighter heading-gradient drop-shadow-[0_0_30px_rgba(0,255,240,0.2)]">
               DeepFold
             </h1>
          </div>
          
          <motion.p 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-lg md:text-xl text-gray-300 max-w-2xl mx-auto leading-relaxed"
          >
            Predict miRNA SNP pathogenicity using a deeply integrated workflow driven by evolutionary embeddings and thermodynamics.
          </motion.p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-6 pt-8">
            <Link 
              href="/predict" 
              className="group relative flex items-center gap-2 px-10 py-5 bg-[#00e5ff] text-[#020818] font-bold rounded-xl overflow-hidden hover:scale-105 transition-all shadow-[0_0_30px_rgba(0,229,255,0.3)] active:translate-y-1"
            >
              Initialize Prediction <ArrowRight size={20} className="group-hover:translate-x-1 transition-transform" />
            </Link>
          </div>
        </div>
      </section>

      <section className="relative z-10 py-24 border-t border-[#00fff0]/10 bg-[#020818]/40 backdrop-blur-sm">
        <div className="container mx-auto px-4 max-w-6xl">
          <div className="text-center mb-16 relative">
            <h2 className="text-3xl font-display font-bold text-white mb-4">Architectural Flow</h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
               Our ensemble extracts structural maps, topological graphs, and k-mer vectors for a unified pathogenicity assessment.
            </p>
          </div>
          <div className="w-full h-[500px] glass-card overflow-hidden cursor-move relative">
             <Suspense fallback={<div className="w-full h-full skeleton" />}>
                <PipelineGraph />
             </Suspense>
          </div>
        </div>
      </section>
    </div>
  );
}
