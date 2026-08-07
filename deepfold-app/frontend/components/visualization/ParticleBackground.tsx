'use client';

import { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';

function Particles() {
  const count = 200;
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const linesRef = useRef<THREE.LineSegments>(null);
  
  // Create particle positions and colors
  const { positions, colors, velocities } = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const cols = new Float32Array(count * 3);
    const vels = [];
    
    const colorCyan = new THREE.Color('#00d4ff');
    const colorViolet = new THREE.Color('#7c3aed');
    
    for (let i = 0; i < count; i++) {
      // Random position in 30x30x30 cube
      pos[i * 3] = (Math.random() - 0.5) * 30;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 30;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 30;
      
      // Random velocities
      vels.push({
        x: (Math.random() - 0.5) * 0.02,
        y: (Math.random() - 0.5) * 0.02,
        z: (Math.random() - 0.5) * 0.02
      });
      
      // Alternate colors
      const color = i % 2 === 0 ? colorCyan : colorViolet;
      cols[i * 3] = color.r;
      cols[i * 3 + 1] = color.g;
      cols[i * 3 + 2] = color.b;
    }
    return { positions: pos, colors: cols, velocities: vels };
  }, [count]);
  
  // Create edges geometry
  const maxLines = count * count;
  const linePositions = useMemo(() => new Float32Array(maxLines * 6), [maxLines]);
  const edgesGeometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
    return geo;
  }, [linePositions]);

  // Flash state
  const timeRef = useRef(0);
  
  useFrame((state, delta) => {
    if (!meshRef.current || !linesRef.current) return;
    
    timeRef.current += delta;
    
    // Rotate entire system
    const group = meshRef.current.parent;
    if (group) group.rotation.y += 0.001;
    
    let lineIdx = 0;
    const dummy = new THREE.Object3D();
    
    // Update particle positions
    for (let i = 0; i < count; i++) {
      positions[i * 3] += velocities[i].x;
      positions[i * 3 + 1] += velocities[i].y;
      positions[i * 3 + 2] += velocities[i].z;
      
      // Bounce off walls
      if (Math.abs(positions[i * 3]) > 15) velocities[i].x *= -1;
      if (Math.abs(positions[i * 3 + 1]) > 15) velocities[i].y *= -1;
      if (Math.abs(positions[i * 3 + 2]) > 15) velocities[i].z *= -1;
      
      dummy.position.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
      
      // Random flash effect every ~3 seconds
      let scale = 1.0;
      if (timeRef.current > 3.0 && Math.random() > 0.99) {
        scale = 2.5;
        if (Math.random() > 0.9) timeRef.current = 0; // reset
      }
      dummy.scale.setScalar(scale);
      
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
    }
    meshRef.current.instanceMatrix.needsUpdate = true;
    
    // Recompute edges
    for (let i = 0; i < count; i++) {
      for (let j = i + 1; j < count; j++) {
        const dx = positions[i * 3] - positions[j * 3];
        const dy = positions[i * 3 + 1] - positions[j * 3 + 1];
        const dz = positions[i * 3 + 2] - positions[j * 3 + 2];
        const distSq = dx*dx + dy*dy + dz*dz;
        
        if (distSq < 16.0) { // distance threshold 4.0
          linePositions[lineIdx++] = positions[i * 3];
          linePositions[lineIdx++] = positions[i * 3 + 1];
          linePositions[lineIdx++] = positions[i * 3 + 2];
          linePositions[lineIdx++] = positions[j * 3];
          linePositions[lineIdx++] = positions[j * 3 + 1];
          linePositions[lineIdx++] = positions[j * 3 + 2];
        }
      }
    }
    
    linesRef.current.geometry.setDrawRange(0, lineIdx / 3);
    linesRef.current.geometry.attributes.position.needsUpdate = true;
  });

  return (
    <group>
      <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
        <sphereGeometry args={[0.05, 8, 8]} />
        <meshBasicMaterial vertexColors toneMapped={false} />
        <instancedBufferAttribute attach="attributes-color" args={[colors, 3]} />
      </instancedMesh>
      <lineSegments ref={linesRef} geometry={edgesGeometry}>
        <lineBasicMaterial color="#00d4ff" transparent opacity={0.15} />
      </lineSegments>
    </group>
  );
}

export default function ParticleBackground() {
  return (
    <div className="fixed inset-0 z-0 pointer-events-none opacity-50">
      <Canvas camera={{ position: [0, 0, 20], fov: 60 }}>
        <Particles />
      </Canvas>
    </div>
  );
}
