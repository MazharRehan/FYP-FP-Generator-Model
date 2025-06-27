import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Plus, Search, Filter, Grid, List, Eye, Download, Share2, Trash2, Calendar, Folder } from 'lucide-react';

interface ProjectManagerProps {
  onProjectSelect: (project: any) => void;
}

const ProjectManager: React.FC<ProjectManagerProps> = ({ onProjectSelect }) => {
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');

  const mockProjects = [
    {
      id: 1,
      name: 'Modern Villa',
      type: 'Residential',
      status: 'Completed',
      date: '2024-01-15',
      thumbnail: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      rooms: 5,
      sqft: 3200
    },
    {
      id: 2,
      name: 'Urban Loft',
      type: 'Residential',
      status: 'In Progress',
      date: '2024-01-10',
      thumbnail: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
      rooms: 3,
      sqft: 1800
    },
    {
      id: 3,
      name: 'Office Complex',
      type: 'Commercial',
      status: 'Draft',
      date: '2024-01-08',
      thumbnail: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
      rooms: 12,
      sqft: 8500
    },
    {
      id: 4,
      name: 'Retail Store',
      type: 'Commercial',
      status: 'Completed',
      date: '2024-01-05',
      thumbnail: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
      rooms: 8,
      sqft: 2400
    }
  ];

  const statusColors = {
    'Completed': 'text-green-400 bg-green-500/20',
    'In Progress': 'text-yellow-400 bg-yellow-500/20',
    'Draft': 'text-slate-400 bg-slate-500/20'
  };

  const filteredProjects = mockProjects.filter(project => {
    const matchesSearch = project.name.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterType === 'all' || project.type.toLowerCase() === filterType;
    return matchesSearch && matchesFilter;
  });

  return (
    <div className="min-h-screen pt-24 pb-16">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-8">
            <div>
              <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                Project Manager
              </h1>
              <p className="text-xl text-slate-300">Organize and manage your architectural projects</p>
            </div>
            
            <motion.button
              className="mt-4 md:mt-0 px-6 py-3 bg-gradient-to-r from-cyan-500 to-purple-600 rounded-xl font-semibold text-white shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/40 transition-all duration-300 flex items-center space-x-2"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Plus className="w-5 h-5" />
              <span>New Project</span>
            </motion.button>
          </div>

          {/* Controls */}
          <div className="flex flex-col md:flex-row md:items-center md:justify-between space-y-4 md:space-y-0 mb-8">
            {/* Search */}
            <div className="relative flex-1 md:max-w-md">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-5 h-5" />
              <input
                type="text"
                placeholder="Search projects..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-3 bg-slate-800/50 border border-slate-700/30 rounded-xl text-white placeholder-slate-400 focus:border-cyan-500 focus:outline-none backdrop-blur-xl"
              />
            </div>

            <div className="flex items-center space-x-4">
              {/* Filter */}
              <div className="relative">
                <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-4 h-4" />
                <select
                  value={filterType}
                  onChange={(e) => setFilterType(e.target.value)}
                  className="pl-10 pr-8 py-3 bg-slate-800/50 border border-slate-700/30 rounded-xl text-white focus:border-cyan-500 focus:outline-none backdrop-blur-xl appearance-none"
                >
                  <option value="all">All Types</option>
                  <option value="residential">Residential</option>
                  <option value="commercial">Commercial</option>
                </select>
              </div>

              {/* View Mode Toggle */}
              <div className="flex items-center bg-slate-800/50 border border-slate-700/30 rounded-xl p-1 backdrop-blur-xl">
                <button
                  onClick={() => setViewMode('grid')}
                  className={`p-2 rounded-lg transition-all ${
                    viewMode === 'grid' ? 'bg-cyan-500 text-white' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <Grid className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  className={`p-2 rounded-lg transition-all ${
                    viewMode === 'list' ? 'bg-cyan-500 text-white' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <List className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Projects Grid/List */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          {viewMode === 'grid' ? (
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredProjects.map((project, index) => (
                <motion.div
                  key={project.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="group bg-slate-800/50 backdrop-blur-xl rounded-2xl border border-slate-700/30 overflow-hidden hover:border-cyan-500/30 transition-all duration-300"
                >
                  {/* Thumbnail */}
                  <div 
                    className="h-48 relative overflow-hidden"
                    style={{ background: project.thumbnail }}
                  >
                    <div className="absolute inset-0 bg-black/20 group-hover:bg-black/10 transition-all duration-300" />
                    <div className="absolute top-4 right-4">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${statusColors[project.status as keyof typeof statusColors]}`}>
                        {project.status}
                      </span>
                    </div>
                    <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                      <button
                        onClick={() => onProjectSelect(project)}
                        className="p-2 bg-cyan-500 hover:bg-cyan-600 rounded-lg text-white transition-colors"
                      >
                        <Eye className="w-4 h-4" />
                      </button>
                      <div className="flex items-center space-x-2">
                        <button className="p-2 bg-slate-700/80 hover:bg-slate-600 rounded-lg text-slate-300 transition-colors">
                          <Download className="w-4 h-4" />
                        </button>
                        <button className="p-2 bg-slate-700/80 hover:bg-slate-600 rounded-lg text-slate-300 transition-colors">
                          <Share2 className="w-4 h-4" />
                        </button>
                        <button className="p-2 bg-red-500/80 hover:bg-red-600 rounded-lg text-white transition-colors">
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* Project Info */}
                  <div className="p-6">
                    <h3 className="text-xl font-bold text-white mb-2 group-hover:text-cyan-400 transition-colors">
                      {project.name}
                    </h3>
                    <p className="text-slate-400 mb-4">{project.type}</p>
                    
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div className="flex items-center space-x-2 text-slate-300">
                        <Folder className="w-4 h-4 text-cyan-400" />
                        <span>{project.rooms} rooms</span>
                      </div>
                      <div className="flex items-center space-x-2 text-slate-300">
                        <Calendar className="w-4 h-4 text-purple-400" />
                        <span>{project.date}</span>
                      </div>
                    </div>
                    
                    <div className="mt-4 pt-4 border-t border-slate-700/30">
                      <div className="text-2xl font-bold text-cyan-400">
                        {project.sqft.toLocaleString()} sq ft
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          ) : (
            <div className="space-y-4">
              {filteredProjects.map((project, index) => (
                <motion.div
                  key={project.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-slate-800/50 backdrop-blur-xl rounded-2xl border border-slate-700/30 p-6 hover:border-cyan-500/30 transition-all duration-300"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-6">
                      <div 
                        className="w-16 h-16 rounded-lg"
                        style={{ background: project.thumbnail }}
                      />
                      <div>
                        <h3 className="text-xl font-bold text-white mb-1">{project.name}</h3>
                        <p className="text-slate-400">{project.type}</p>
                        <div className="flex items-center space-x-4 mt-2 text-sm text-slate-300">
                          <span>{project.rooms} rooms</span>
                          <span>•</span>
                          <span>{project.sqft.toLocaleString()} sq ft</span>
                          <span>•</span>
                          <span>{project.date}</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-4">
                      <span className={`px-3 py-1 rounded-full text-sm font-medium ${statusColors[project.status as keyof typeof statusColors]}`}>
                        {project.status}
                      </span>
                      <div className="flex items-center space-x-2">
                        <button
                          onClick={() => onProjectSelect(project)}
                          className="p-2 bg-cyan-500 hover:bg-cyan-600 rounded-lg text-white transition-colors"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        <button className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-slate-300 transition-colors">
                          <Download className="w-4 h-4" />
                        </button>
                        <button className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-slate-300 transition-colors">
                          <Share2 className="w-4 h-4" />
                        </button>
                        <button className="p-2 bg-red-500/80 hover:bg-red-600 rounded-lg text-white transition-colors">
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </motion.div>

        {filteredProjects.length === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-16"
          >
            <div className="w-16 h-16 bg-slate-700 rounded-xl flex items-center justify-center mx-auto mb-4">
              <Folder className="w-8 h-8 text-slate-500" />
            </div>
            <h3 className="text-xl font-bold text-slate-300 mb-2">No projects found</h3>
            <p className="text-slate-400">Try adjusting your search or filter criteria</p>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default ProjectManager;