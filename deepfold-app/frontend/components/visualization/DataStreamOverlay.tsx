'use client';

import { usePathname } from 'next/navigation';
import { useMemo } from 'react';

export default function DataStreamOverlay() {
  const pathname = usePathname();
  const visible = pathname === '/predict' || pathname === '/dashboard';

  const columns = useMemo(() => {
    const chars = ['A', 'U', 'C', 'G'];
    return Array.from({ length: 12 }, (_, i) => {
      const left = Math.random() * 100;
      const duration = 8 + Math.random() * 12; // 8s to 20s
      const delay = Math.random() * -20;
      const opacity = 0.05 + Math.random() * 0.1;
      
      // Generate a string of random characters for this column
      const content = Array.from({ length: 50 }, () => chars[Math.floor(Math.random() * 4)]).join('\n');
      
      return { id: i, left, duration, delay, opacity, content };
    });
  }, []);

  if (!visible) return null;

  return (
    <div className="fixed inset-0 pointer-events-none z-[-1] overflow-hidden">
      {columns.map((col) => (
        <div 
          key={col.id}
          className="data-stream-col leading-none"
          style={{ 
            left: `${col.left}%`, 
            animationDuration: `${col.duration}s`,
            animationDelay: `${col.delay}s`,
            opacity: col.opacity
          }}
        >
          {col.content}
        </div>
      ))}
    </div>
  );
}
