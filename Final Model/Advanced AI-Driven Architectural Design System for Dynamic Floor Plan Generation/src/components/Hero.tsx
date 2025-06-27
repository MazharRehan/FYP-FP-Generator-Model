import React from 'react';
import { motion } from 'framer-motion';
import { Sparkles, ArrowRight, Zap, Layers, Eye } from 'lucide-react';

interface HeroProps {
  onGetStarted: () => void;
}

const Hero: React.FC<HeroProps> = ({ onGetStarted }) => {
  const features = [
    {
      icon: Zap,
      title: 'AI-Powered Design',
      description: 'Generate stunning floor plans in seconds using advanced AI algorithms'
    },
    {
      icon: Layers,
      title: 'Perfect Layouts',
      description: 'Zero overlapping rooms with intelligent spacing and professional quality'
    },
    {
      icon: Eye,
      title: 'Multiple Variations',
      description: 'Create unique layout variations with different architectural strategies'
    }
  ];

  return (
    <div className="min-h-screen pt-24 pb-16">
      <div className="max-w-7xl mx-auto px-6">
        {/* Hero Section */}
        <div className="text-center mb-20">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="mb-8"
          >
            <div className="inline-flex items-center space-x-2 bg-gradient-to-r from-cyan-500/10 to-purple-500/10 backdrop-blur-xl rounded-full px-6 py-3 border border-cyan-500/20 mb-6">
              <Sparkles className="w-5 h-5 text-cyan-400" />
              <span className="text-sm font-medium text-cyan-400">Powered by Advanced AI</span>
            </div>
            
            <h1 className="text-6xl md:text-8xl font-bold mb-6 leading-tight">
              <span className="bg-gradient-to-r from-white via-cyan-200 to-purple-400 bg-clip-text text-transparent">
                Design Your
              </span>
              <br />
              <span className="bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                Perfect Floor Plan
              </span>
            </h1>
            
            <p className="text-xl text-slate-300 max-w-3xl mx-auto leading-relaxed">
              Create professional floor plans with AI-powered intelligence. Perfect room placement, 
              zero overlapping, and multiple architectural strategies for unique designs.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-6"
          >
            <motion.button
              onClick={onGetStarted}
              className="group px-8 py-4 bg-gradient-to-r from-cyan-500 to-purple-600 rounded-xl font-semibold text-white shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/40 transition-all duration-300 flex items-center space-x-3"
              whileHover={{ scale: 1.05, boxShadow: "0 0 40px rgba(0, 212, 255, 0.4)" }}
              whileTap={{ scale: 0.95 }}
            >
              <span>Start Designing</span>
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </motion.button>
          </motion.div>
        </div>

        {/* Features Grid */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="grid md:grid-cols-3 gap-8 mb-20"
        >
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <motion.div
                key={index}
                className="group relative p-8 rounded-2xl bg-gradient-to-br from-slate-800/40 to-slate-900/40 backdrop-blur-xl border border-slate-700/30 hover:border-cyan-500/30 transition-all duration-300"
                whileHover={{ scale: 1.02, rotateY: 5 }}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.6 + index * 0.1 }}
              >
                <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-purple-500/5 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                
                <div className="relative z-10">
                  <div className="w-12 h-12 bg-gradient-to-r from-cyan-400 to-purple-600 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  
                  <h3 className="text-xl font-bold text-white mb-4 group-hover:text-cyan-400 transition-colors">
                    {feature.title}
                  </h3>
                  
                  <p className="text-slate-400 leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              </motion.div>
            );
          })}
        </motion.div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-8 text-center"
        >
          {[
            { number: '7+', label: 'Layout Strategies' },
            { number: '0%', label: 'Room Overlapping' },
            { number: '100%', label: 'Connected Layouts' },
            { number: '∞', label: 'Unique Variations' }
          ].map((stat, index) => (
            <div key={index} className="group">
              <div className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent mb-2 group-hover:scale-110 transition-transform">
                {stat.number}
              </div>
              <div className="text-slate-400 text-sm">{stat.label}</div>
            </div>
          ))}
        </motion.div>
      </div>
    </div>
  );
};

export default Hero;