import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Header from './components/Header';
import Hero from './components/Hero';
import FloorPlanGenerator from './components/FloorPlanGenerator';
import Footer from './components/Footer';

function App() {
  const [activeSection, setActiveSection] = useState<'home' | 'generate'>('home');
  const [currentProject, setCurrentProject] = useState<any>(null);

  const renderActiveSection = () => {
    switch (activeSection) {
      case 'generate':
        return <FloorPlanGenerator onProjectUpdate={setCurrentProject} />;
      default:
        return <Hero onGetStarted={() => setActiveSection('generate')} />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white overflow-hidden">
      {/* Animated background particles */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        {[...Array(50)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-cyan-400 rounded-full opacity-30"
            animate={{
              x: [0, Math.random() * window.innerWidth],
              y: [0, Math.random() * window.innerHeight],
            }}
            transition={{
              duration: Math.random() * 20 + 10,
              repeat: Infinity,
              repeatType: "reverse"
            }}
            style={{
              left: Math.random() * window.innerWidth,
              top: Math.random() * window.innerHeight,
            }}
          />
        ))}
      </div>

      {/* Grid overlay */}
      <div className="fixed inset-0 opacity-10 pointer-events-none">
        <div className="w-full h-full bg-[linear-gradient(rgba(0,212,255,0.1)_1px,transparent_1px),linear-gradient(90deg,rgba(0,212,255,0.1)_1px,transparent_1px)] bg-[size:20px_20px]" />
      </div>

      <Header 
        activeSection={activeSection} 
        onNavigate={setActiveSection}
      />

      <main className="relative z-10">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeSection}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {renderActiveSection()}
          </motion.div>
        </AnimatePresence>
      </main>

      {activeSection === 'home' && <Footer />}
    </div>
  );
}

export default App;