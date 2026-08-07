'use client';

import { useRef, useState, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Line } from '@react-three/drei';
import * as THREE from 'three';
import { useSpring, animated } from '@react-spring/three';

const PIpelineShaderMaterial = new THREE.ShaderMaterial({
  uniforms: { time: { value: 0 }, color: { value: new THREE.Color() } },
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform float time;
    uniform vec3 color;
    varying vec2 vUv;
    void main() {
      float noise = sin(vUv.x * 10.0 + time) * cos(vUv.y * 10.0 + time);
      vec3 finalColor = mix(color, vec3(1.0), noise * 0.5 + 0.5);
      float glow = max(0.0, 1.0 - length(vUv - vec2(0.5)) * 2.0);
      gl_FragColor = vec4(finalColor * glow, glow);
    }
  `,
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending
});

function PipelineNode({ node }: { node: any }) {
  const [hovered, setHover] = useState(false);
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  
  useFrame((state) => {
     if (materialRef.current) {
       materialRef.current.uniforms.time.value = state.clock.elapsedTime;
     }
  });

  const { scale } = useSpring({ scale: hovered ? 1.5 : 1.0 });

  return (
    <animated.mesh 
      position={node.pos} 
      scale={scale}
      onPointerOver={(e) => { e.stopPropagation(); setHover(true); }}
      onPointerOut={() => setHover(false)}
    >
      <sphereGeometry args={[0.8, 16, 16]} />
      <primitive object={PIpelineShaderMaterial.clone()} ref={materialRef} uniforms-color-value={new THREE.Color(node.color)} />
      {hovered && (
        <Text position={[0, 1.5, 0]} fontSize={0.5} color="white" outlineWidth={0.05} outlineColor="#000">
          {node.label}
        </Text>
      )}
    </animated.mesh>
  );
}

function Scene() {
  const groupRef = useRef<THREE.Group>(null);
  
  const nodes = [
    { id: 'ufold', pos: [-5, 2, 0], color: '#00fff0', label: 'UFold Maps' },
    { id: 'rnafm', pos: [-3, -3, 2], color: '#a855f7', label: 'Evolutionary Pattern Model' },
    { id: 'thermo', pos: [-4, 5, -2], color: '#fbbf24', label: 'Stability Analysis Model' },
    { id: 'kmer', pos: [3, 4, 1], color: '#22d3ee', label: 'Sequence Pattern Model' },
    { id: 'gcn', pos: [4, -2, -1], color: '#ff2d55', label: 'Graph Structure Model' },
    { id: 'meta', pos: [0, 0, 0], color: '#ffffff', label: 'Decision Stacking' }
  ];

  const bonds = [[0, 5], [1, 5], [2, 5], [3, 5], [4, 5]];

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.elapsedTime * 0.1;
    }
  });

  return (
    <group ref={groupRef} scale={[0.8, 0.8, 0.8]}>
      {nodes.map((n, i) => <PipelineNode key={i} node={n} />)}
      {bonds.map(([i, j], idx) => (
        <Line key={idx} points={[nodes[i].pos as any, nodes[j].pos as any]} color="#0ea5e9" lineWidth={1} transparent opacity={0.3} />
      ))}
    </group>
  );
}

export default function PipelineGraph() {
  return (
    <Canvas 
      camera={{ position: [0, 0, 12], fov: 45 }} 
      frameloop="demand" 
      gl={{ antialias: false }}
    >
      <Scene />
      <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={0.5} />
    </Canvas>
  );
}
