import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        deepspace: {
          DEFAULT: '#050912',
          card: 'rgba(13,20,36,0.8)',
          cardSolid: '#0d1424',
        },
        cyan: {
          400: '#22d3ee',
          500: '#00d4ff',
          hover: 'rgba(0, 212, 255, 0.15)',
        },
        violet: {
          500: '#7c3aed',
        },
        emerald: {
          500: '#10b981',
          bg: 'rgba(16, 185, 129, 0.1)',
        },
        red: {
          500: '#ef4444',
          bg: 'rgba(239, 68, 68, 0.1)',
        },
        amber: {
          500: '#f59e0b',
          bg: 'rgba(245, 158, 11, 0.1)',
        }
      },
      fontFamily: {
        sans: ['var(--font-inter)', 'system-ui', 'sans-serif'],
        display: ['var(--font-space-grotesk)', 'system-ui', 'sans-serif'],
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'hero-grid': 'linear-gradient(to right, rgba(255,255,255,0.05) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.05) 1px, transparent 1px)',
      },
    },
  },
  plugins: [],
};
export default config;
