import { Github } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="w-full border-t border-cyan-500/10 bg-deepspace-card py-8 mt-auto">
      <div className="container mx-auto px-4 flex flex-col md:flex-row justify-between items-center gap-4">
        <p className="text-sm text-gray-400">
          © {new Date().getFullYear()} DeepFold Project. All rights reserved.
        </p>
        <div className="flex gap-4">
          <a href="https://github.com/uci-cbcl/UFold" target="_blank" rel="noreferrer" className="text-gray-400 hover:text-cyan-400 transition-colors flex items-center gap-1 text-sm">
            Powered by UFold & Evolutionary Pattern Model
          </a>
        </div>
      </div>
    </footer>
  );
}
