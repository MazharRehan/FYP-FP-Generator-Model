import React from 'react';
import { motion } from 'framer-motion';
import { Zap, Github, Twitter, Linkedin, Mail } from 'lucide-react';

const Footer: React.FC = () => {
  const socialLinks = [
    { icon: Github, label: 'GitHub', href: '#' },
    { icon: Twitter, label: 'Twitter', href: '#' },
    { icon: Linkedin, label: 'LinkedIn', href: '#' },
    { icon: Mail, label: 'Email', href: '#' }
  ];

  const footerLinks = {
    'Product': ['Features', 'Pricing', 'API', 'Documentation'],
    'Company': ['About', 'Blog', 'Careers', 'Press'],
    'Resources': ['Help Center', 'Community', 'Tutorials', 'Examples'],
    'Legal': ['Privacy', 'Terms', 'Security', 'Cookies']
  };

  return (
    <footer className="relative mt-20 border-t border-slate-800/50">
      <div className="max-w-7xl mx-auto px-6 py-16">
        {/* Main Footer Content */}
        <div className="grid lg:grid-cols-5 gap-12 mb-12">
          {/* Brand Column */}
          <div className="lg:col-span-2">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-6"
            >
              <div className="flex items-center space-x-3 mb-4">
                <div className="relative">
                  <div className="w-12 h-12 bg-gradient-to-r from-cyan-400 to-purple-600 rounded-xl flex items-center justify-center">
                    <Zap className="w-7 h-7 text-white" />
                  </div>
                  <div className="absolute inset-0 bg-gradient-to-r from-cyan-400 to-purple-600 rounded-xl blur-lg opacity-30" />
                </div>
                <div>
                  <h3 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                    GenArch
                  </h3>
                  <p className="text-sm text-slate-400">Future of Design</p>
                </div>
              </div>
              <p className="text-slate-300 leading-relaxed mb-6">
                Revolutionizing architectural design with AI-powered tools. Create stunning floor plans, 
                visualize in 3D, and bring your architectural vision to life.
              </p>
              
              {/* Social Links */}
              <div className="flex items-center space-x-4">
                {socialLinks.map((social, index) => {
                  const Icon = social.icon;
                  return (
                    <motion.a
                      key={index}
                      href={social.href}
                      className="w-10 h-10 bg-slate-800/50 rounded-lg flex items-center justify-center text-slate-400 hover:text-cyan-400 hover:bg-slate-700/50 transition-all duration-200"
                      whileHover={{ scale: 1.1, y: -2 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      <Icon className="w-5 h-5" />
                    </motion.a>
                  );
                })}
              </div>
            </motion.div>
          </div>

          {/* Links Columns */}
          {Object.entries(footerLinks).map(([category, links], index) => (
            <motion.div
              key={category}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <h4 className="text-white font-semibold mb-4">{category}</h4>
              <ul className="space-y-3">
                {links.map((link) => (
                  <li key={link}>
                    <a
                      href="#"
                      className="text-slate-400 hover:text-cyan-400 transition-colors duration-200"
                    >
                      {link}
                    </a>
                  </li>
                ))}
              </ul>
            </motion.div>
          ))}
        </div>

        {/* Newsletter Signup */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-gradient-to-r from-slate-800/40 to-slate-700/40 backdrop-blur-xl rounded-2xl border border-slate-700/30 p-8 mb-12"
        >
          <div className="max-w-2xl">
            <h3 className="text-2xl font-bold text-white mb-2">Stay Updated</h3>
            <p className="text-slate-300 mb-6">
              Get the latest updates on new features, tutorials, and architectural design trends.
            </p>
            <div className="flex flex-col sm:flex-row gap-4">
              <input
                type="email"
                placeholder="Enter your email address"
                className="flex-1 px-4 py-3 bg-slate-900/50 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:border-cyan-500 focus:outline-none"
              />
              <motion.button
                className="px-6 py-3 bg-gradient-to-r from-cyan-500 to-purple-600 rounded-lg font-semibold text-white shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/40 transition-all duration-300"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                Subscribe
              </motion.button>
            </div>
          </div>
        </motion.div>

        {/* Bottom Bar */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="flex flex-col md:flex-row md:items-center md:justify-between pt-8 border-t border-slate-800/50"
        >
          <p className="text-slate-400 text-sm">
            © 2024 GenArch. All rights reserved.
          </p>
          <div className="flex items-center space-x-6 mt-4 md:mt-0">
            <span className="text-slate-400 text-sm">Made with AI-powered design</span>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              <span className="text-green-400 text-sm">System Online</span>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Background Effects */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -bottom-32 -left-32 w-64 h-64 bg-cyan-500/10 rounded-full blur-3xl" />
        <div className="absolute -bottom-32 -right-32 w-64 h-64 bg-purple-500/10 rounded-full blur-3xl" />
      </div>
    </footer>
  );
};

export default Footer;