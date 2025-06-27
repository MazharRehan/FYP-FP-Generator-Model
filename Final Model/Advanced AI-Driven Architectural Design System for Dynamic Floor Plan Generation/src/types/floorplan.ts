export interface PlotSize {
  name: string;
  dimensions: {
    width: number;
    height: number;
  };
  totalArea: number;
  imageSpecs: {
    width: number;
    height: number;
    dpi: number;
    aspectRatio: number;
  };
}

export interface RoomRequirements {
  bedrooms: number;
  bathrooms: number;
  kitchens: number;
  lounges: number;
  garages: number;
  lawn: boolean;
  drawingRoom: boolean;
  diningRoom: boolean;
  storage: boolean;
  laundry: boolean;
  library: boolean;
  stairs: boolean; // New stairs option
}

export interface RoomColor {
  name: string;
  rgb: string;
  hex: string;
}

export const PLOT_SIZES: PlotSize[] = [
  {
    name: '5 Marla',
    dimensions: { width: 25, height: 45 },
    totalArea: 1125,
    imageSpecs: { width: 608, height: 1088, dpi: 96, aspectRatio: 0.559 }
  },
  {
    name: '10 Marla',
    dimensions: { width: 35, height: 65 },
    totalArea: 2250,
    imageSpecs: { width: 849, height: 1570, dpi: 96, aspectRatio: 0.541 }
  },
  {
    name: '20 Marla',
    dimensions: { width: 50, height: 90 },
    totalArea: 4500,
    imageSpecs: { width: 1209, height: 2170, dpi: 96, aspectRatio: 0.557 }
  }
];

export const ROOM_COLORS: Record<string, RoomColor> = {
  bedroom: { name: 'Bedroom', rgb: 'rgb(255, 0, 0)', hex: '#FF0000' },
  bathroom: { name: 'Bathroom', rgb: 'rgb(0, 0, 255)', hex: '#0000FF' },
  kitchen: { name: 'Kitchen', rgb: 'rgb(255, 165, 0)', hex: '#FFA500' },
  drawingRoom: { name: 'Drawing Room', rgb: 'rgb(0, 128, 0)', hex: '#008000' },
  garage: { name: 'Garage', rgb: 'rgb(165, 42, 42)', hex: '#A52A2A' },
  lounge: { name: 'Lounge (Sitting Area)', rgb: 'rgb(255, 255, 0)', hex: '#FFFF00' },
  backyard: { name: 'Backyard', rgb: 'rgb(50, 205, 50)', hex: '#32CD32' },
  stairs: { name: 'Stairs', rgb: 'rgb(0, 128, 128)', hex: '#008080' },
  storage: { name: 'Store (Storage Room)', rgb: 'rgb(128, 0, 128)', hex: '#800080' },
  openSpace: { name: 'Open Space', rgb: 'rgb(0, 255, 255)', hex: '#00FFFF' },
  staircase: { name: 'Staircase', rgb: 'rgb(153, 51, 255)', hex: '#9933FF' },
  lobby: { name: 'Main Entrance', rgb: 'rgb(255, 0, 255)', hex: '#FF00FF' },
  lawn: { name: 'Lawn', rgb: 'rgb(64, 224, 208)', hex: '#40E0D0' },
  dining: { name: 'Dining', rgb: 'rgb(225, 192, 203)', hex: '#E1C0CB' },
  passage: { name: 'Passage', rgb: 'rgb(128, 128, 0)', hex: '#808000' },
  laundry: { name: 'Laundry', rgb: 'rgb(230, 230, 250)', hex: '#E6E6FA' },
  dressingArea: { name: 'Dressing Area', rgb: 'rgb(255, 127, 80)', hex: '#FF7F50' },
  sideGarden: { name: 'Side Garden', rgb: 'rgb(255, 215, 0)', hex: '#FFD700' },
  library: { name: 'Library', rgb: 'rgb(255, 191, 0)', hex: '#FFBF00' },
  study: { name: 'Study Room', rgb: 'rgb(139, 69, 19)', hex: '#8B4513' },
  balcony: { name: 'Balcony', rgb: 'rgb(176, 196, 222)', hex: '#B0C4DE' },
  terrace: { name: 'Terrace', rgb: 'rgb(210, 180, 140)', hex: '#D2B48C' },
  walkInCloset: { name: 'Walk-in Closet', rgb: 'rgb(221, 160, 221)', hex: '#DDA0DD' },
  pantry: { name: 'Pantry', rgb: 'rgb(255, 228, 181)', hex: '#FFE4B5' },
  walls: { name: 'Walls', rgb: 'rgb(0, 0, 0)', hex: '#000000' },
  door: { name: 'Door', rgb: 'rgb(128, 0, 0)', hex: '#800000' }
};

// Real-life architectural standards for room dimensions and adjacencies
export const ARCHITECTURAL_STANDARDS = {
  // Minimum room dimensions in feet
  minRoomSizes: {
    bedroom: { width: 10, height: 10 },
    masterBedroom: { width: 12, height: 14 },
    bathroom: { width: 5, height: 8 },
    kitchen: { width: 8, height: 10 },
    drawingRoom: { width: 12, height: 16 },
    diningRoom: { width: 10, height: 12 },
    lounge: { width: 14, height: 18 },
    garage: { width: 10, height: 20 },
    storage: { width: 4, height: 6 },
    laundry: { width: 6, height: 8 },
    study: { width: 8, height: 10 },
    walkInCloset: { width: 6, height: 8 },
    lobby: { width: 8, height: 6 },
    stairs: { width: 4, height: 8 },
    library: { width: 10, height: 12 },
    lawn: { width: 8, height: 8 }
  },
  
  // Room adjacency preferences (which rooms should be connected)
  adjacencyRules: {
    masterBedroom: ['bathroom', 'walkInCloset', 'balcony'],
    bedroom: ['bathroom'],
    kitchen: ['dining', 'pantry', 'laundry'],
    drawingRoom: ['dining', 'lobby'],
    lounge: ['dining', 'kitchen'],
    garage: ['storage', 'laundry'],
    study: ['library', 'bedroom'],
    bathroom: ['bedroom', 'laundry'],
    stairs: ['lobby', 'lounge', 'drawingRoom']
  },
  
  // Circulation requirements
  circulation: {
    corridorWidth: 4,
    doorWidth: 3,
    minRoomAccess: 3
  }
};

export interface GeneratedFloorPlan {
  id: string;
  plotSize: PlotSize;
  requirements: RoomRequirements;
  rooms: Array<{
    type: string;
    color: string;
    position: { x: number; y: number };
    size: { width: number; height: number };
    label: string;
  }>;
  walls: Array<{
    start: { x: number; y: number };
    end: { x: number; y: number };
    internal?: boolean;
  }>;
  doors: Array<{
    position: { x: number; y: number };
    orientation: 'horizontal' | 'vertical';
    width?: number;
  }>;
  createdAt: Date;
}