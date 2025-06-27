# GenArch Floor Plan Generator - Local Setup Guide

## Prerequisites

Before running this application locally, ensure you have the following installed:

1. **Node.js** (version 18 or higher)
   - Download from: https://nodejs.org/
   - Verify installation: `node --version`

2. **npm** (comes with Node.js)
   - Verify installation: `npm --version`

3. **Git** (optional, for cloning)
   - Download from: https://git-scm.com/

## Installation Steps

### Option 1: Download Project Files

1. Download all project files to your local machine
2. Extract to a folder (e.g., `GenArch-floor-planner`)

### Option 2: Clone Repository (if using Git)

```bash
git clone <repository-url>
cd GenArch-floor-planner
```

## Setup and Run

1. **Navigate to project directory**
   ```bash
   cd GenArch-floor-planner
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```

4. **Open in browser**
   - The application will automatically open at: `http://localhost:5173`
   - If it doesn't open automatically, manually navigate to the URL

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build locally
- `npm run lint` - Run ESLint for code quality

## Project Structure

```
GenArch-floor-planner/
├── src/
│   ├── components/          # React components
│   │   ├── FloorPlanGenerator.tsx
│   │   ├── FloorPlanCanvas.tsx
│   │   ├── Header.tsx
│   │   ├── Hero.tsx
│   │   ├── ProjectManager.tsx
│   │   └── Footer.tsx
│   ├── types/              # TypeScript type definitions
│   │   └── floorplan.ts
│   ├── utils/              # Utility functions
│   │   └── floorPlanGenerator.ts
│   ├── App.tsx             # Main application component
│   ├── main.tsx            # Application entry point
│   └── index.css           # Global styles
├── public/                 # Static assets
├── package.json            # Dependencies and scripts
├── vite.config.ts          # Vite configuration
├── tailwind.config.js      # Tailwind CSS configuration
└── tsconfig.json           # TypeScript configuration
```

## Features

### Floor Plan Generation
- **5 Layout Strategies**: Traditional, Modern, Compact, Open-plan, Courtyard
- **Real Architectural Standards**: Proper room dimensions and spacing
- **Connected Layouts**: Rooms are logically connected with proper circulation
- **Randomized Variations**: Each generation creates unique layouts

### Export Formats
- **PNG**: High-resolution raster images
- **SVG**: Scalable vector graphics
- **DXF**: CAD-compatible format for AutoCAD

### Room Types
- Bedrooms (with master bedroom)
- Bathrooms (including master bathroom)
- Kitchen with optional pantry
- Drawing room and dining room
- Family lounge
- Study room and library
- Storage and laundry
- Garage
- Outdoor spaces (lawn, terrace, balcony)
- Walk-in closet

## Customization

### Adding New Room Types

1. **Update types** in `src/types/floorplan.ts`:
   ```typescript
   // Add to RoomRequirements interface
   newRoomType: boolean;
   
   // Add to ROOM_COLORS
   newRoomType: { name: 'New Room', rgb: 'rgb(r,g,b)', hex: '#RRGGBB' }
   
   // Add to ARCHITECTURAL_STANDARDS.minRoomSizes
   newRoomType: { width: 10, height: 12 }
   ```

2. **Update generator logic** in `src/utils/floorPlanGenerator.ts`

3. **Update UI** in `src/components/FloorPlanGenerator.tsx`

### Modifying Layout Algorithms

Edit the layout generation methods in `src/utils/floorPlanGenerator.ts`:
- `generateTraditionalLayout()`
- `generateModernLayout()`
- `generateCompactLayout()`
- `generateOpenPlanLayout()`
- `generateCourtyardLayout()`

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Kill process on port 5173
   npx kill-port 5173
   # Or use different port
   npm run dev -- --port 3000
   ```

2. **Dependencies not installing**
   ```bash
   # Clear npm cache
   npm cache clean --force
   # Delete node_modules and reinstall
   rm -rf node_modules package-lock.json
   npm install
   ```

3. **TypeScript errors**
   ```bash
   # Check TypeScript configuration
   npx tsc --noEmit
   ```

### Performance Optimization

For better performance in development:

1. **Disable source maps** (in `vite.config.ts`):
   ```typescript
   export default defineConfig({
     build: {
       sourcemap: false
     }
   });
   ```

2. **Enable hot reload** (already configured)

## Production Deployment

1. **Build for production**
   ```bash
   npm run build
   ```

2. **Preview production build**
   ```bash
   npm run preview
   ```

3. **Deploy** the `dist` folder to your hosting service

### Hosting Options
- **Netlify**: Drag and drop `dist` folder
- **Vercel**: Connect GitHub repository
- **GitHub Pages**: Use GitHub Actions
- **Traditional hosting**: Upload `dist` folder contents

## Technical Details

- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **Animations**: Framer Motion
- **Canvas Rendering**: HTML5 Canvas API
- **Export Generation**: Custom algorithms for PNG/SVG/DXF

## Support

For issues or questions:
1. Check the console for error messages
2. Verify all dependencies are installed
3. Ensure Node.js version compatibility
4. Check network connectivity for CDN resources

The application runs entirely in the browser with no backend dependencies required.