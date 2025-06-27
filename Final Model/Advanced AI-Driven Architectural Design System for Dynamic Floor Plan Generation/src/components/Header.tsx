import React from 'react';
import { motion } from 'framer-motion';
import { Zap } from 'lucide-react';

interface HeaderProps {
  activeSection: string;
  onNavigate: (section: 'home' | 'generate' | 'projects') => void;
}

const Header: React.FC<HeaderProps> = ({ activeSection, onNavigate }) => {
  return (
    <motion.header
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      className="fixed top-0 left-0 right-0 z-50 backdrop-blur-xl bg-slate-900/70 border-b border-cyan-500/20"
    >
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <motion.div
            className="flex items-center space-x-3"
            whileHover={{ scale: 1.05 }}
            onClick={() => onNavigate('home')}
          >
            <div className="relative cursor-pointer">
              <div className="w-10 h-10 bg-gradient-to-r from-cyan-400 to-purple-600 rounded-lg flex items-center justify-center">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-400 to-purple-600 rounded-lg blur-lg opacity-30" />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                GenArch
              </h1>
              <p className="text-xs text-slate-400">Floor Plan Generator</p>
            </div>
          </motion.div>

          {/* Simple Navigation */}
          <nav className="flex items-center space-x-1">
            <motion.button
              onClick={() => onNavigate('generate')}
              className={`relative px-6 py-2 rounded-lg flex items-center space-x-2 transition-all duration-200 ${
                activeSection === 'generate'
                  ? 'text-cyan-400 bg-cyan-500/10'
                  : 'text-slate-300 hover:text-cyan-400 hover:bg-slate-800/50'
              }`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <span className="text-sm font-medium">Generate Floor Plans</span>
              {activeSection === 'generate' && (
                <motion.div
                  layoutId="activeTab"
                  className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 rounded-lg border border-cyan-500/30"
                />
              )}
            </motion.button>
          </nav>
        </div>
      </div>
    </motion.header>
  );
};

export default Header;