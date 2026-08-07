'use client';

import { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Environment, Html } from '@react-three/drei';
import * as THREE from 'three';
import { useSpring, animated } from '@react-spring/three';

interface Props {
  healthySeq: string;
  mutantSeqType?: 'pathogenic' | 'benign' | 'VUS' | null;
  snpPos?: number;
}

function generateHairpinCurve(nodes: {x:number, y:number, z:number}[]) {
  const pts = nodes.filter(n => n && typeof n.x === 'number' && !isNaN(n.x)).map(n => new THREE.Vector3(n.x, n.y, n.z));
  if (pts.length < 2) {
    // Return a dummy curve if not enough points to prevent crash
    return new THREE.CatmullRomCurve3([new THREE.Vector3(0,0,0), new THREE.Vector3(0,0.1,0)]);
  }
  return new THREE.CatmullRomCurve3(pts);
}

function HolographicMolecule({ healthySeq, mutantSeqType, snpPos }: Props) {
  const groupRef = useRef<THREE.Group>(null);
  const { gl } = useThree();
  
  const seq = healthySeq || "AUGCAUGCAUGCAUGC";
  const L = seq.length;
  const stemLen = Math.floor(L * 0.4);
  const loopLen = L - (stemLen * 2);

  const { nodes, pairs, curvePts } = useMemo(() => {
    const nData = [];
    const pData = [];
    const yStep = stemLen > 0 ? 8.0 / stemLen : 1.0;

    for (let i = 0; i < L; i++) {
        let x, y, z;
        if (i < stemLen) {
            x = -1.5; y = i * yStep - 4; z = Math.sin(i * 0.5) * 0.5;
        } else if (i >= L - stemLen) {
            const pairIdx = L - 1 - i;
            x = 1.5; y = pairIdx * yStep - 4; z = -Math.sin(pairIdx * 0.5) * 0.5;
            pData.push([pairIdx, i]);
        } else {
            const loopIdx = i - stemLen;
            const denominator = Math.max(1, loopLen - 1);
            const loopAngle = (loopIdx / denominator) * Math.PI;
            x = -1.5 * Math.cos(loopAngle);
            y = (stemLen - 1) * yStep - 4 + Math.sin(loopAngle) * 2.5;
            z = 0; 
        }

        const base = seq[i]?.toUpperCase() ?? 'A';
        let color = '#00e5ff'; // A (Saturated)
        if (base === 'U' || base === 'T') color = '#cc44ff'; // U
        if (base === 'C') color = '#ffcc00'; // C
        if (base === 'G') color = '#44ff88'; // G
        
        nData.push({ x, y, z, color, isSnp: i === snpPos });
    }
    return { nodes: nData, pairs: pData, curvePts: nData };
  }, [seq, snpPos, stemLen, L, loopLen]);

  const curveLeft = useMemo(() => generateHairpinCurve(curvePts.slice(0, stemLen + Math.ceil(loopLen/2) + 1)), [curvePts, stemLen, loopLen]);
  const curveRight = useMemo(() => generateHairpinCurve(curvePts.slice(Math.floor(stemLen + loopLen/2), L).reverse()), [curvePts, stemLen, loopLen, L]);

  const { snpColor } = useSpring({
    snpColor: mutantSeqType === 'pathogenic' ? '#ff2d55' : 
              mutantSeqType === 'benign' ? '#44ff88' : 
              mutantSeqType === 'VUS' ? '#ffcc00' : '#ffffff',
    config: { tension: 100, friction: 20 }
  });

  // Performance throttling: 30fps when idle
  const lastInteraction = useRef(Date.now());
  useFrame((state, delta) => {
    const idle = Date.now() - lastInteraction.current > 3000;
    if (idle && Math.random() > 0.5) return; // Skip 50% frames if idle
    
    if (groupRef.current) groupRef.current.rotation.y += delta * 0.1;
  });

  return (
    <group ref={groupRef} onPointerMove={() => (lastInteraction.current = Date.now())}>
      {/* Backbones */}
      <mesh>
        <tubeGeometry args={[curveLeft, 64, 0.12, 6, false]} />
        <meshToonMaterial color="#0ea5e9" transparent opacity={0.3} />
      </mesh>
      <mesh>
        <tubeGeometry args={[curveRight, 64, 0.12, 6, false]} />
        <meshToonMaterial color="#0ea5e9" transparent opacity={0.3} />
      </mesh>

      {/* Nodes */}
      {nodes.map((node, i) => (
        <SnpNode key={i} node={node} isSnp={node.isSnp} animColor={snpColor} snpType={mutantSeqType} />
      ))}
      
      {/* Rungs */}
      {pairs.map(([i, j], idx) => {
        const n1 = nodes[i]; const n2 = nodes[j];
        if (!n1 || !n2) return null;
        const start = new THREE.Vector3(n1.x, n1.y, n1.z);
        const end = new THREE.Vector3(n2.x, n2.y, n2.z);
        const distance = start.distanceTo(end);
        const direction = new THREE.Vector3().subVectors(end, start).normalize();
        const quaternion = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);

        return (
          <mesh key={`pair-${idx}`} position={start.clone().lerp(end, 0.5)} quaternion={quaternion}>
            <cylinderGeometry args={[0.04, 0.04, distance, 4]} />
            <meshToonMaterial color="#334155" transparent opacity={0.4} />
          </mesh>
        );
      })}
    </group>
  );
}

function SnpNode({ node, isSnp, animColor, snpType }: any) {
  const meshRef = useRef<THREE.Mesh>(null);
  const shockwaveRef = useRef<THREE.Mesh>(null);
  const [shockTime, setShockTime] = useState(0);

  useFrame((state) => {
    if (isSnp && meshRef.current) {
        const pulse = 1.0 + Math.sin(state.clock.elapsedTime * 4) * 0.1;
        meshRef.current.scale.setScalar(pulse);
        //@ts-ignore
        meshRef.current.material.emissiveIntensity = pulse * 1.2;

        if (snpType && shockwaveRef.current) {
          setShockTime(t => (t + 0.04) % 1);
          const scale = 1 + shockTime * 5;
          const opacity = Math.max(0, 0.8 - shockTime);
          shockwaveRef.current.scale.setScalar(scale);
          //@ts-ignore
          shockwaveRef.current.material.opacity = opacity;
        }
    }
  });

  return (
    <animated.mesh position={[node.x, node.y, node.z]} ref={meshRef}>
      <sphereGeometry args={[isSnp ? 0.4 : 0.2, 6, 6]} />
      <animated.meshToonMaterial color={isSnp ? animColor : node.color} />
      {isSnp && (
         <>
           <Html position={[0.5, 0.5, 0]} center className="pointer-events-none select-none">
              <div className="bg-[#020818]/80 backdrop-blur px-2 py-0.5 rounded border border-cyan-500/50 text-[9px] text-cyan-400 font-mono shadow-xl">
                 SNP SITE
              </div>
           </Html>
           <mesh ref={shockwaveRef} rotation={[Math.PI / 2, 0, 0]}>
              <ringGeometry args={[0.3, 0.4, 32]} />
              <meshBasicMaterial color={animColor} side={THREE.DoubleSide} transparent opacity={0.6} />
           </mesh>
         </>
      )}
    </animated.mesh>
  );
}

export default function RNAHelix3D(props: Props) {
  return (
    <div className="w-full h-[450px] bg-[#020818] rounded-xl border border-cyan-500/10 relative overflow-hidden">
      <Canvas camera={{ position: [0, 0, 15], fov: 40 }} frameloop="demand" gl={{ antialias: false }}>
        <ambientLight intensity={0.8} />
        <pointLight position={[10, 10, 10]} intensity={1.5} color="#00e5ff" />
        <pointLight position={[-10, 10, 10]} intensity={1.0} color="#cc44ff" />
        <HolographicMolecule {...props} />
        <OrbitControls autoRotate={!props.mutantSeqType} autoRotateSpeed={1.0} enableZoom={false} enablePan={false} />
        {process.env.NODE_ENV === 'development' && <axesHelper args={[5]} />}
      </Canvas>
      <div className="absolute top-4 left-4 z-10 font-mono text-[10px] text-cyan-500/50 uppercase tracking-widest pointer-events-none">
        [3D_STRUCTURE_RENDER]
      </div>
    </div>
  );
}
