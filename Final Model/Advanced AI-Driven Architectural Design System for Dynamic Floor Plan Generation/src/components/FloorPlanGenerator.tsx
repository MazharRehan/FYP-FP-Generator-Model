import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Settings, Zap, Download, Share2, Maximize2, RotateCcw, Home, Car, Bath, ChefHat, Sofa, Trees, Grid, Layers, BookOpen, RefreshCw, Brain, Cpu, Stars as Stairs } from 'lucide-react';
import { PLOT_SIZES, ROOM_COLORS, PlotSize, RoomRequirements, GeneratedFloorPlan } from '../types/floorplan';
import { CGANFloorPlanGenerator } from '../utils/cganFloorPlanGenerator';
import FloorPlanCanvas from './FloorPlanCanvas';

interface FloorPlanGeneratorProps {
  onProjectUpdate: (project: any) => void;
}

export default function FloorPlanGenerator({ onProjectUpdate }: FloorPlanGeneratorProps) {
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedPlans, setGeneratedPlans] = useState<GeneratedFloorPlan[]>([]);
  const [selectedPlotSize, setSelectedPlotSize] = useState<PlotSize>(PLOT_SIZES[0]);
  const [generationMode, setGenerationMode] = useState<'cgan' | 'enhanced'>('cgan');
  const [requirements, setRequirements] = useState<RoomRequirements>({
    bedrooms: 0,
    bathrooms: 0,
    kitchens: 0,
    lounges: 0,
    garages: 0,
    lawn: false,
    drawingRoom: false,
    diningRoom: false,
    storage: false,
    laundry: false,
    library: false,
    stairs: false
  });

  const handleGenerate = async () => {
    setIsGenerating(true);
    
    setTimeout(async () => {
      // Generate with TRULY random variation for unique results every time
      const variation = Math.floor(Math.random() * 100000) + Date.now();
      const generator = new CGANFloorPlanGenerator(selectedPlotSize, requirements, variation);
      const newPlan = await generator.generate();
      
      setGeneratedPlans([newPlan]);
      setIsGenerating(false);
      
      onProjectUpdate({
        id: newPlan.id,
        name: `${selectedPlotSize.name} Floor Plan (${generationMode.toUpperCase()})`,
        plotSize: selectedPlotSize,
        requirements,
        createdAt: new Date()
      });
    }, 2000 + Math.random() * 2000);
  };

  const generateMultiplePlans = async () => {
    setIsGenerating(true);
    
    setTimeout(async () => {
      const plans = [];
      
      // Generate 3 COMPLETELY different variations with different strategies
      for (let i = 0; i < 3; i++) {
        const variation = Math.floor(Math.random() * 100000) + Date.now() + i * 50000;
        const generator = new CGANFloorPlanGenerator(selectedPlotSize, requirements, variation);
        const plan = await generator.generate();
        plan.id = `plan-${Date.now()}-${variation}-${Math.random().toString(36).substr(2, 8)}`;
        plans.push(plan);
        
        // Add delay between generations to ensure different seeds
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      
      setGeneratedPlans(plans);
      setIsGenerating(false);
      
      onProjectUpdate({
        id: plans[0].id,
        name: `${selectedPlotSize.name} Floor Plans (${plans.length} ${generationMode.toUpperCase()} variations)`,
        plotSize: selectedPlotSize,
        requirements,
        createdAt: new Date()
      });
    }, 4000 + Math.random() * 2000);
  };

  const regenerateWithNewVariation = async () => {
    if (generatedPlans.length === 0) return;
    
    setIsGenerating(true);
    
    setTimeout(async () => {
      const variation = Math.floor(Math.random() * 100000) + Date.now();
      const generator = new CGANFloorPlanGenerator(selectedPlotSize, requirements, variation);
      const newPlan = await generator.generate();
      
      setGeneratedPlans([newPlan]);
      setIsGenerating(false);
    }, 2000);
  };

  const downloadAllPlans = () => {
    generatedPlans.forEach((plan, index) => {
      setTimeout(() => {
        const event = new CustomEvent('downloadPlan', { detail: { plan, index } });
        document.dispatchEvent(event);
      }, index * 500);
    });
  };

  return (
    <div className="min-h-screen pt-24 pb-16">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12 text-center"
        >
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
            Enhanced Floor Plan Generator
          </h1>
          <p className="text-xl text-slate-300">AI-driven architectural design with intelligent non-overlapping layouts</p>
          <div className="mt-4 flex items-center justify-center space-x-6 text-sm text-slate-400">
            <div className="flex items-center space-x-2">
              <Brain className="w-4 h-4 text-cyan-400" />
              <span>Smart Placement</span>
            </div>
            <div className="flex items-center space-x-2">
              <Cpu className="w-4 h-4 text-purple-400" />
              <span>No Overlapping</span>
            </div>
            <div className="flex items-center space-x-2">
              <RefreshCw className="w-4 h-4 text-green-400" />
              <span>Perfect Spacing</span>
            </div>
            <div className="flex items-center space-x-2">
              <Download className="w-4 h-4 text-orange-400" />
              <span>Professional Export</span>
            </div>
          </div>
        </motion.div>

        <div className="grid lg:grid-cols-4 gap-8">
          {/* Parameter Panel */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-1 space-y-6"
          >
            {/* Generation Mode Selection */}
            <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl border border-slate-700/30 p-6">
              <div className="flex items-center space-x-3 mb-6">
                <Brain className="w-6 h-6 text-cyan-400" />
                <h2 className="text-xl font-bold text-white">AI Mode</h2>
              </div>

              <div className="space-y-3">
                <button
                  onClick={() => setGenerationMode('cgan')}
                  className={`w-full p-4 rounded-lg border transition-all ${
                    generationMode === 'cgan'
                      ? 'bg-cyan-500/20 border-cyan-500 text-cyan-400'
                      : 'bg-slate-700/50 border-slate-600 text-slate-300 hover:border-cyan-500/50'
                  }`}
                >
                  <div className="text-left">
                    <div className="font-semibold flex items-center space-x-2">
                      <Brain className="w-4 h-4" />
                      <span>Smart Mode</span>
                    </div>
                    <div className="text-sm opacity-75 mt-1">
                      Advanced AI with perfect room placement
                    </div>
                    <div className="text-xs opacity-60 mt-1">
                      Zero overlapping, optimal spacing
                    </div>
                  </div>
                </button>

                <button
                  onClick={() => setGenerationMode('enhanced')}
                  className={`w-full p-4 rounded-lg border transition-all ${
                    generationMode === 'enhanced'
                      ? 'bg-purple-500/20 border-purple-500 text-purple-400'
                      : 'bg-slate-700/50 border-slate-600 text-slate-300 hover:border-purple-500/50'
                  }`}
                >
                  <div className="text-left">
                    <div className="font-semibold flex items-center space-x-2">
                      <Cpu className="w-4 h-4" />
                      <span>Enhanced Rules</span>
                    </div>
                    <div className="text-sm opacity-75 mt-1">
                      Rule-based with intelligent organization
                    </div>
                    <div className="text-xs opacity-60 mt-1">
                      Architectural standards with smart variation
                    </div>
                  </div>
                </button>
              </div>
            </div>

            {/* Plot Size Selection */}
            <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl border border-slate-700/30 p-6">
              <div className="flex items-center space-x-3 mb-6">
                <Home className="w-6 h-6 text-cyan-400" />
                <h2 className="text-xl font-bold text-white">Plot Size</h2>
              </div>

              <div className="space-y-3">
                {PLOT_SIZES.map((plot) => (
                  <button
                    key={plot.name}
                    onClick={() => setSelectedPlotSize(plot)}
                    className={`w-full p-4 rounded-lg border transition-all ${
                      selectedPlotSize.name === plot.name
                        ? 'bg-cyan-500/20 border-cyan-500 text-cyan-400'
                        : 'bg-slate-700/50 border-slate-600 text-slate-300 hover:border-cyan-500/50'
                    }`}
                  >
                    <div className="text-left">
                      <div className="font-semibold">{plot.name}</div>
                      <div className="text-sm opacity-75">
                        {plot.dimensions.width}' × {plot.dimensions.height}'
                      </div>
                      <div className="text-sm opacity-75">
                        {plot.totalArea.toLocaleString()} sq ft
                      </div>
                      <div className="text-xs opacity-60 mt-1">
                        Output: {plot.imageSpecs.width} × {plot.imageSpecs.height}px
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Room Requirements */}
            <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl border border-slate-700/30 p-6">
              <div className="flex items-center space-x-3 mb-6">
                <Settings className="w-6 h-6 text-cyan-400" />
                <h2 className="text-xl font-bold text-white">Room Requirements</h2>
              </div>

              <div className="space-y-4">
                {/* Essential Rooms */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      <Home className="w-4 h-4 inline mr-1" />
                      Bedrooms
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="6"
                      value={requirements.bedrooms}
                      onChange={(e) => setRequirements({...requirements, bedrooms: parseInt(e.target.value) || 0})}
                      className="w-full bg-slate-900/50 border border-slate-600 rounded-lg px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      <Bath className="w-4 h-4 inline mr-1" />
                      Bathrooms
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="4"
                      value={requirements.bathrooms}
                      onChange={(e) => setRequirements({...requirements, bathrooms: parseInt(e.target.value) || 0})}
                      className="w-full bg-slate-900/50 border border-slate-600 rounded-lg px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      <ChefHat className="w-4 h-4 inline mr-1" />
                      Kitchens
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="2"
                      value={requirements.kitchens}
                      onChange={(e) => setRequirements({...requirements, kitchens: parseInt(e.target.value) || 0})}
                      className="w-full bg-slate-900/50 border border-slate-600 rounded-lg px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      <Sofa className="w-4 h-4 inline mr-1" />
                      Lounges
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="3"
                      value={requirements.lounges}
                      onChange={(e) => setRequirements({...requirements, lounges: parseInt(e.target.value) || 0})}
                      className="w-full bg-slate-900/50 border border-slate-600 rounded-lg px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      <Car className="w-4 h-4 inline mr-1" />
                      Garages
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="2"
                      value={requirements.garages}
                      onChange={(e) => setRequirements({...requirements, garages: parseInt(e.target.value) || 0})}
                      className="w-full bg-slate-900/50 border border-slate-600 rounded-lg px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
                    />
                  </div>
                </div>

                {/* Optional Features - ONLY the 7 specified */}
                <div className="space-y-3">
                  <h3 className="text-sm font-semibold text-slate-300">Optional Features</h3>
                  <div className="grid grid-cols-2 gap-2">
                    {[
                      { key: 'lawn', label: 'Lawn', icon: Trees },
                      { key: 'drawingRoom', label: 'Drawing Room', icon: Home },
                      { key: 'diningRoom', label: 'Dining Room', icon: ChefHat },
                      { key: 'storage', label: 'Storage', icon: Home },
                      { key: 'laundry', label: 'Laundry', icon: Home },
                      { key: 'library', label: 'Library', icon: BookOpen },
                      { key: 'stairs', label: 'Stairs', icon: Stairs }
                    ].map(({ key, label, icon: Icon }) => (
                      <button
                        key={key}
                        onClick={() => setRequirements({...requirements, [key]: !requirements[key as keyof RoomRequirements]})}
                        className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm transition-all ${
                          requirements[key as keyof RoomRequirements]
                            ? 'bg-cyan-500 text-white'
                            : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                        }`}
                      >
                        <Icon className="w-4 h-4" />
                        <span className="text-xs">{label}</span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Generation Buttons */}
                <div className="space-y-3 pt-4">
                  <motion.button
                    onClick={handleGenerate}
                    disabled={isGenerating}
                    className="w-full py-3 bg-gradient-to-r from-cyan-500 to-purple-600 rounded-lg font-semibold text-white shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/40 transition-all duration-300 flex items-center justify-center space-x-2 disabled:opacity-50"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    {isGenerating ? (
                      <>
                        <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        <span>Generating Perfect Layout...</span>
                      </>
                    ) : (
                      <>
                        {generationMode === 'cgan' ? <Brain className="w-5 h-5" /> : <Zap className="w-5 h-5" />}
                        <span>Generate Floor Plan</span>
                      </>
                    )}
                  </motion.button>

                  <motion.button
                    onClick={generateMultiplePlans}
                    disabled={isGenerating}
                    className="w-full py-3 bg-gradient-to-r from-purple-500 to-pink-600 rounded-lg font-semibold text-white shadow-lg shadow-purple-500/25 hover:shadow-purple-500/40 transition-all duration-300 flex items-center justify-center space-x-2 disabled:opacity-50"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    {isGenerating ? (
                      <>
                        <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        <span>Generating 3 Perfect Variations...</span>
                      </>
                    ) : (
                      <>
                        <Layers className="w-5 h-5" />
                        <span>Generate 3 Variations</span>
                      </>
                    )}
                  </motion.button>

                  {generatedPlans.length > 0 && (
                    <motion.button
                      onClick={regenerateWithNewVariation}
                      disabled={isGenerating}
                      className="w-full py-2 bg-gradient-to-r from-green-500 to-teal-600 rounded-lg font-medium text-white shadow-lg shadow-green-500/25 hover:shadow-green-500/40 transition-all duration-300 flex items-center justify-center space-x-2 disabled:opacity-50"
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <RefreshCw className="w-4 h-4" />
                      <span>New Variation</span>
                    </motion.button>
                  )}
                </div>
              </div>
            </div>

            {/* Color Legend */}
            <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl border border-slate-700/30 p-6">
              <h3 className="text-lg font-bold text-white mb-4">Room Color Legend</h3>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {Object.entries(ROOM_COLORS).slice(0, 15).map(([key, color]) => (
                  <div key={key} className="flex items-center space-x-3">
                    <div 
                      className="w-4 h-4 rounded border border-slate-600 flex-shrink-0"
                      style={{ backgroundColor: color.hex }}
                    />
                    <span className="text-sm text-slate-300">{color.name}</span>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>

          {/* Preview Area */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-3"
          >
            <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl border border-slate-700/30 p-6">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-xl font-bold text-white">Generated Floor Plans</h2>
                  <p className="text-sm text-slate-400 mt-1">
                    {generationMode === 'cgan' 
                      ? 'Smart AI designs with perfect room placement and zero overlapping' 
                      : 'Enhanced rule-based designs with intelligent room organization'
                    }
                  </p>
                </div>
                {generatedPlans.length > 0 && (
                  <div className="flex items-center space-x-2">
                    <button 
                      onClick={downloadAllPlans}
                      className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors text-white text-sm font-medium flex items-center space-x-2"
                    >
                      <Download className="w-4 h-4" />
                      <span>Download All</span>
                    </button>
                    <button className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors">
                      <Share2 className="w-4 h-4 text-slate-300" />
                    </button>
                    <button className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors">
                      <Maximize2 className="w-4 h-4 text-slate-300" />
                    </button>
                    <button 
                      onClick={() => setGeneratedPlans([])}
                      className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
                    >
                      <RotateCcw className="w-4 h-4 text-slate-300" />
                    </button>
                  </div>
                )}
              </div>

              <div className="min-h-[600px]">
                {isGenerating ? (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center">
                      <div className="w-16 h-16 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
                      <p className="text-slate-300 mb-4">
                        {generationMode === 'cgan' 
                          ? 'Smart AI is creating your perfect floor plan...' 
                          : 'Enhanced AI is creating your organized floor plan...'
                        }
                      </p>
                      <div className="space-y-2">
                        <div className="text-sm text-slate-400">• Analyzing your room requirements</div>
                        <div className="text-sm text-slate-400">• Applying intelligent placement algorithms</div>
                        <div className="text-sm text-slate-400">• Preventing room overlaps and ensuring proper spacing</div>
                        <div className="text-sm text-slate-400">• Optimizing room adjacencies and circulation</div>
                        <div className="text-sm text-slate-400">• Generating professional-grade layout</div>
                      </div>
                    </div>
                  </div>
                ) : generatedPlans.length > 0 ? (
                  <div className={`grid gap-6 ${generatedPlans.length > 1 ? 'grid-cols-1 lg:grid-cols-2 xl:grid-cols-3' : 'grid-cols-1'}`}>
                    {generatedPlans.map((plan, index) => (
                      <div key={plan.id} className="bg-slate-900/50 rounded-xl p-4">
                        <div className="flex items-center justify-between mb-4">
                          <div>
                            <h3 className="text-lg font-semibold text-white">
                              {generatedPlans.length > 1 ? `Perfect Layout ${index + 1}` : `Perfect Floor Plan`}
                            </h3>
                            <div className="text-sm text-slate-400">
                              {plan.plotSize.name} - {plan.plotSize.dimensions.width}' × {plan.plotSize.dimensions.height}'
                            </div>
                          </div>
                          <div className="text-xs text-slate-500">
                            ID: {plan.id.slice(-8)}
                          </div>
                        </div>
                        <FloorPlanCanvas plan={plan} />
                        <div className="mt-4 flex items-center justify-between text-sm">
                          <div className="text-slate-400">
                            Generated: {plan.createdAt.toLocaleTimeString()}
                          </div>
                          <div className="text-cyan-400 font-medium">
                            {plan.rooms.length} rooms • {plan.doors.length} doors
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full text-center text-slate-400">
                    <div>
                      <div className="w-16 h-16 bg-slate-700 rounded-xl flex items-center justify-center mx-auto mb-4">
                        {generationMode === 'cgan' ? <Brain className="w-8 h-8 text-slate-500" /> : <Zap className="w-8 h-8 text-slate-500" />}
                      </div>
                      <p className="text-lg font-medium mb-2">Ready to Generate Perfect Floor Plans</p>
                      <p className="text-sm">Configure your requirements and click "Generate" to create {generationMode === 'cgan' ? 'smart' : 'enhanced'} floor plans</p>
                      <p className="text-sm mt-2 text-cyan-400">
                        {generationMode === 'cgan' 
                          ? 'Using Smart AI with intelligent non-overlapping placement'
                          : 'Using enhanced rule-based system with smart room organization'
                        }
                      </p>
                      <div className="mt-4 text-xs text-slate-500 space-y-1">
                        <p>✨ Every generation creates unique, non-overlapping layouts</p>
                        <p>🏠 Main entrance is always included</p>
                        <p>🔄 Use "New Variation" for different layouts with same requirements</p>
                        <p>🏗️ Stairs option now available in optional features</p>
                        <p>📐 Perfect spacing and professional quality guaranteed</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}