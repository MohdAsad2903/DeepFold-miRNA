'use client';

import Link from 'next/link';
import { Activity, Menu, X } from 'lucide-react';
import { useEffect, useState } from 'react';
import { getHealth } from '@/lib/api';
import { motion, AnimatePresence } from 'framer-motion';

export default function Navbar() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [healthStatus, setHealthStatus] = useState<'checking' | 'ok' | 'demo' | 'error'>('checking');

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await getHealth();
        if (res.status === 'error' || res.status === 'offline') setHealthStatus('error');
        else if (res.models_loaded === false) setHealthStatus('demo');
        else setHealthStatus('ok');
      } catch (e) {
        setHealthStatus('error');
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 60000); // 1 min health check
    return () => clearInterval(interval);
  }, []);

  const navLinks = [
    { name: 'Home', href: '/' },
    { name: 'Predict', href: '/predict', primary: true },
    { name: 'Research', href: '/research' },
    { name: 'Dashboard', href: '/dashboard' },
    { name: 'About', href: '/about' },
  ];

  return (
    <nav className="sticky top-0 z-[100] w-full backdrop-blur-[8px] bg-[#020818]/60 border-b border-cyan-500/10 h-16 flex items-center">
      <div className="container mx-auto px-4 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2 group z-50">
          <div className="w-8 h-8 rounded bg-cyan-500/10 flex items-center justify-center border border-cyan-500/30">
            <Activity className="w-5 h-5 text-cyan-400 group-hover:scale-110 transition-transform" />
          </div>
          <span className="font-display font-black text-xl tracking-tighter text-white">
            DeepFold
          </span>
        </Link>
        
        {/* Desktop Nav */}
        <div className="hidden md:flex items-center gap-8">
          <div className="flex gap-6 text-xs font-bold uppercase tracking-widest">
            {navLinks.map((link) => (
              <Link 
                key={link.name} 
                href={link.href} 
                className={`transition-colors py-2 relative group ${link.primary ? 'text-cyan-400' : 'text-gray-400 hover:text-white'}`}
              >
                {link.name}
                <span className={`absolute bottom-0 left-0 h-[2px] bg-cyan-400 transition-all duration-300 ${link.primary ? 'w-full' : 'w-0 group-hover:w-full'}`} />
              </Link>
            ))}
          </div>

          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/5 border border-cyan-500/20">
            <div className={`w-1.5 h-1.5 rounded-full ${
              healthStatus === 'ok' ? 'bg-green-500' : healthStatus === 'demo' ? 'bg-amber-500' : 'bg-red-500'
            } animate-pulse`} />
            <span className="text-[9px] font-bold text-cyan-500/70 uppercase select-none">{healthStatus}</span>
          </div>
        </div>

        {/* Mobile Trigger */}
        <button className="md:hidden text-white" onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}>
          {isMobileMenuOpen ? <X /> : <Menu />}
        </button>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isMobileMenuOpen && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="absolute top-16 left-0 right-0 bg-[#020818]/95 backdrop-blur-xl border-b border-cyan-500/10 overflow-hidden md:hidden">
             <div className="flex flex-col p-4 gap-4">
                {navLinks.map(link => (
                  <Link key={link.name} href={link.href} onClick={() => setIsMobileMenuOpen(false)} className="text-sm font-bold uppercase tracking-widest p-2 hover:bg-cyan-500/10 rounded transition-colors">
                    {link.name}
                  </Link>
                ))}
             </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
}
