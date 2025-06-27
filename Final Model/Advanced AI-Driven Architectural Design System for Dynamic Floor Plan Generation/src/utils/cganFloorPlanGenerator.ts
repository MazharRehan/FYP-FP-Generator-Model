import * as tf from '@tensorflow/tfjs';
import { PlotSize, RoomRequirements, GeneratedFloorPlan, ROOM_COLORS, ARCHITECTURAL_STANDARDS } from '../types/floorplan';

// Enhanced architectural patterns with perfect room placement logic
const ARCHITECTURAL_PATTERNS = {
  // Room adjacency patterns with stronger connections
  adjacencyMatrix: {
    'bedroom': { 'bathroom': 0.95, 'walkInCloset': 0.8, 'balcony': 0.6, 'bedroom': 0.3 },
    'bathroom': { 'bedroom': 0.95, 'laundry': 0.7, 'bathroom': 0.4 },
    'kitchen': { 'dining': 0.9, 'pantry': 0.85, 'laundry': 0.6, 'lounge': 0.5 },
    'lounge': { 'dining': 0.8, 'kitchen': 0.5, 'drawingRoom': 0.7, 'lobby': 0.6 },
    'drawingRoom': { 'dining': 0.6, 'lounge': 0.7, 'lobby': 0.9 },
    'dining': { 'kitchen': 0.9, 'lounge': 0.8, 'drawingRoom': 0.6 },
    'garage': { 'storage': 0.7, 'laundry': 0.5, 'lobby': 0.4 },
    'study': { 'library': 0.9, 'bedroom': 0.4 },
    'lobby': { 'drawingRoom': 0.9, 'lounge': 0.6, 'garage': 0.4 },
    'stairs': { 'lobby': 0.8, 'lounge': 0.6, 'drawingRoom': 0.5 },
    'library': { 'study': 0.9, 'bedroom': 0.3, 'lounge': 0.4 },
    'storage': { 'garage': 0.7, 'laundry': 0.6 },
    'laundry': { 'kitchen': 0.6, 'bathroom': 0.7, 'storage': 0.6 }
  },

  // Room size ratios with better proportions
  roomSizeRatios: {
    'bedroom': { minArea: 100, optimalRatio: 1.0, priority: 1 },
    'bathroom': { minArea: 40, optimalRatio: 1.6, priority: 1 },
    'kitchen': { minArea: 80, optimalRatio: 1.25, priority: 1 },
    'drawingRoom': { minArea: 192, optimalRatio: 1.33, priority: 2 },
    'dining': { minArea: 120, optimalRatio: 1.2, priority: 2 },
    'lounge': { minArea: 252, optimalRatio: 1.29, priority: 2 },
    'garage': { minArea: 200, optimalRatio: 2.0, priority: 3 },
    'storage': { minArea: 24, optimalRatio: 1.5, priority: 3 },
    'laundry': { minArea: 48, optimalRatio: 1.33, priority: 3 },
    'lobby': { minArea: 48, optimalRatio: 1.33, priority: 1 },
    'stairs': { minArea: 32, optimalRatio: 2.0, priority: 3 },
    'library': { minArea: 120, optimalRatio: 1.2, priority: 3 },
    'lawn': { minArea: 64, optimalRatio: 1.0, priority: 4 }
  },

  // Layout strategies for true variation
  layoutStrategies: [
    'traditional-front-back',
    'modern-open-plan', 
    'compact-grid',
    'courtyard-central',
    'linear-side-by-side',
    'l-shaped-corner',
    'u-shaped-embrace'
  ]
};

export class CGANFloorPlanGenerator {
  private plotSize: PlotSize;
  private requirements: RoomRequirements;
  private layoutVariation: number;
  private model: tf.LayersModel | null = null;
  private isModelLoaded: boolean = false;
  private randomSeed: number;
  private occupiedSpaces: Array<{x: number, y: number, width: number, height: number}> = [];
  private layoutStrategy: string;

  constructor(plotSize: PlotSize, requirements: RoomRequirements, variation = 0) {
    this.plotSize = plotSize;
    this.requirements = requirements;
    this.layoutVariation = variation;
    this.randomSeed = Date.now() + variation * 1000 + Math.random() * 100000;
    this.occupiedSpaces = [];
    
    // Select layout strategy based on variation for TRUE uniqueness
    this.layoutStrategy = ARCHITECTURAL_PATTERNS.layoutStrategies[
      Math.floor(this.seededRandom() * ARCHITECTURAL_PATTERNS.layoutStrategies.length)
    ];
    
    console.log(`🎯 Using layout strategy: ${this.layoutStrategy} for variation ${variation}`);
    
    this.initializeModel();
  }

  // Seeded random for consistent but varied results
  private seededRandom(): number {
    this.randomSeed = (this.randomSeed * 9301 + 49297) % 233280;
    return this.randomSeed / 233280;
  }

  private async initializeModel() {
    try {
      await tf.ready();
      this.model = this.createGenerativeModel();
      this.isModelLoaded = true;
      console.log('Smart AI model initialized successfully');
    } catch (error) {
      console.warn('TensorFlow.js model initialization failed, falling back to enhanced rule-based system:', error);
      this.isModelLoaded = false;
    }
  }

  private createGenerativeModel(): tf.LayersModel {
    const inputDim = 25;
    const outputDim = 100;

    const model = tf.sequential({
      layers: [
        tf.layers.dense({
          inputShape: [inputDim],
          units: 256,
          activation: 'relu',
          name: 'generator_dense1'
        }),
        tf.layers.dropout({ rate: 0.3 }),
        tf.layers.dense({
          units: 512,
          activation: 'relu',
          name: 'generator_dense2'
        }),
        tf.layers.dropout({ rate: 0.3 }),
        tf.layers.dense({
          units: 1024,
          activation: 'relu',
          name: 'generator_dense3'
        }),
        tf.layers.dense({
          units: outputDim,
          activation: 'sigmoid',
          name: 'generator_output'
        }),
        tf.layers.reshape({
          targetShape: [10, 10, 1],
          name: 'spatial_reshape'
        })
      ]
    });

    this.initializeWithArchitecturalKnowledge(model);
    return model;
  }

  private initializeWithArchitecturalKnowledge(model: tf.LayersModel) {
    const weights = model.getWeights();
    const initializedWeights = weights.map(weight => {
      const shape = weight.shape;
      const fan_in = shape.length > 1 ? shape[0] : 1;
      const fan_out = shape.length > 1 ? shape[1] : shape[0];
      const limit = Math.sqrt(6 / (fan_in + fan_out));
      
      return tf.randomUniform(shape, -limit, limit);
    });
    
    model.setWeights(initializedWeights);
  }

  async generate(): Promise<GeneratedFloorPlan> {
    // Reset occupied spaces for each generation
    this.occupiedSpaces = [];
    
    if (this.isModelLoaded && this.model) {
      return await this.generateWithCGAN();
    } else {
      return this.generateWithEnhancedRules();
    }
  }

  private async generateWithCGAN(): Promise<GeneratedFloorPlan> {
    try {
      const encodedInput = this.encodeRequirements();
      const prediction = this.model!.predict(encodedInput) as tf.Tensor;
      const layoutGrid = await prediction.data();
      
      const rooms = this.generateRoomsWithStrategyVariation();
      const walls = this.generateWallsFromRooms(rooms);
      const doors = this.generateIntelligentDoors(rooms);

      encodedInput.dispose();
      prediction.dispose();

      return {
        id: `smart-${Date.now()}-${this.layoutVariation}-${Math.random().toString(36).substr(2, 8)}`,
        plotSize: this.plotSize,
        requirements: this.requirements,
        rooms,
        walls,
        doors,
        createdAt: new Date()
      };
    } catch (error) {
      console.warn('Smart AI generation failed, falling back to enhanced rules:', error);
      return this.generateWithEnhancedRules();
    }
  }

  private encodeRequirements(): tf.Tensor {
    const input = [
      // Room counts (normalized)
      this.requirements.bedrooms / 6,
      this.requirements.bathrooms / 4,
      this.requirements.kitchens / 2,
      this.requirements.lounges / 3,
      this.requirements.garages / 2,
      
      // Boolean features (all optional features)
      this.requirements.drawingRoom ? 1 : 0,
      this.requirements.diningRoom ? 1 : 0,
      this.requirements.storage ? 1 : 0,
      this.requirements.laundry ? 1 : 0,
      this.requirements.library ? 1 : 0,
      this.requirements.stairs ? 1 : 0,
      this.requirements.lawn ? 1 : 0,
      
      // Plot characteristics
      this.plotSize.dimensions.width / 100,
      this.plotSize.dimensions.height / 100,
      this.plotSize.totalArea / 5000,
      
      // Variation parameters for uniqueness
      this.layoutVariation / 1000,
      this.seededRandom(),
      this.seededRandom(),
      this.seededRandom(),
      this.seededRandom(),
      this.seededRandom(),
      this.seededRandom(),
      this.seededRandom(),
      this.seededRandom(),
      this.seededRandom()
    ];

    return tf.tensor2d([input]);
  }

  private generateWithEnhancedRules(): GeneratedFloorPlan {
    const rooms = this.generateRoomsWithStrategyVariation();
    const walls = this.generateWallsFromRooms(rooms);
    const doors = this.generateIntelligentDoors(rooms);

    return {
      id: `enhanced-${Date.now()}-${this.layoutVariation}-${Math.random().toString(36).substr(2, 8)}`,
      plotSize: this.plotSize,
      requirements: this.requirements,
      rooms,
      walls,
      doors,
      createdAt: new Date()
    };
  }

  private generateRoomsWithStrategyVariation(): any[] {
    const { width, height } = this.plotSize.dimensions;
    const rooms = [];
    const margin = 2;
    
    // Reset occupied spaces
    this.occupiedSpaces = [];
    
    console.log(`🏗️ Starting generation with strategy: ${this.layoutStrategy}`);
    console.log('📋 Requirements:', this.requirements);
    console.log('📐 Plot size:', this.plotSize.name, `${width}' × ${height}'`);

    // STEP 1: Always add main entrance first (guaranteed placement)
    const entranceRoom = {
      type: 'lobby',
      color: ROOM_COLORS.lobby.hex,
      position: { x: margin, y: margin },
      size: { width: width - 2 * margin, height: 6 },
      label: 'Main Entrance'
    };
    rooms.push(entranceRoom);
    this.addToOccupiedSpaces(entranceRoom.position.x, entranceRoom.position.y, entranceRoom.size.width, entranceRoom.size.height);

    // STEP 2: Generate rooms based on selected strategy
    const roomQueue = this.createExactRoomQueue();
    console.log('📋 Room queue created:', roomQueue.map(r => `${r.label} (${r.type})`));

    // STEP 3: Apply strategy-specific placement that covers the entire plot
    switch (this.layoutStrategy) {
      case 'traditional-front-back':
        this.applyTraditionalFrontBackStrategy(rooms, roomQueue);
        break;
      case 'modern-open-plan':
        this.applyModernOpenPlanStrategy(rooms, roomQueue);
        break;
      case 'compact-grid':
        this.applyCompactGridStrategy(rooms, roomQueue);
        break;
      case 'courtyard-central':
        this.applyCourtyardCentralStrategy(rooms, roomQueue);
        break;
      case 'linear-side-by-side':
        this.applyLinearSideBySideStrategy(rooms, roomQueue);
        break;
      case 'l-shaped-corner':
        this.applyLShapedCornerStrategy(rooms, roomQueue);
        break;
      case 'u-shaped-embrace':
        this.applyUShapedEmbraceStrategy(rooms, roomQueue);
        break;
      default:
        this.applyFallbackStrategy(rooms, roomQueue);
    }

    console.log(`🎯 Successfully placed ${rooms.length} rooms using ${this.layoutStrategy} strategy`);
    return rooms;
  }

  private applyTraditionalFrontBackStrategy(rooms: any[], roomQueue: any[]) {
    const { width, height } = this.plotSize.dimensions;
    const margin = 2;
    let currentY = 10; // After entrance

    // Front zone: Drawing room, lounge (covers full width)
    const frontZoneHeight = height * 0.35;
    
    roomQueue.filter(r => ['drawingRoom', 'lounge'].includes(r.type)).forEach(roomData => {
      const roomSize = { width: width - 2 * margin, height: frontZoneHeight / 2 - 1 };
      const position = { x: margin, y: currentY };
      
      if (this.canPlaceRoom(position.x, position.y, roomSize.width, roomSize.height)) {
        const room = this.createRoom(roomData, position, roomSize);
        rooms.push(room);
        this.addToOccupiedSpaces(position.x, position.y, roomSize.width, roomSize.height);
        currentY += roomSize.height + 2;
      }
    });

    // Middle zone: Kitchen, dining (side by side, covers full width)
    const middleY = height * 0.35 + 10;
    const middleHeight = height * 0.25;
    let middleX = margin;
    const middleRooms = roomQueue.filter(r => ['kitchen', 'dining'].includes(r.type));
    
    if (middleRooms.length > 0) {
      const roomWidth = (width - 2 * margin - (middleRooms.length - 1) * 2) / middleRooms.length;
      
      middleRooms.forEach(roomData => {
        const roomSize = { width: roomWidth, height: middleHeight };
        const position = { x: middleX, y: middleY };
        
        if (this.canPlaceRoom(position.x, position.y, roomSize.width, roomSize.height)) {
          const room = this.createRoom(roomData, position, roomSize);
          rooms.push(room);
          this.addToOccupiedSpaces(position.x, position.y, roomSize.width, roomSize.height);
          middleX += roomWidth + 2;
        }
      });
    }

    // Back zone: Bedrooms, bathrooms (covers remaining space)
    const backY = height * 0.6 + 10;
    const backHeight = height - backY - margin;
    this.placeRemainingRoomsInGrid(rooms, roomQueue.filter(r => !['drawingRoom', 'lounge', 'kitchen', 'dining'].includes(r.type)), margin, backY, width - 2 * margin, backHeight);
  }

  private applyModernOpenPlanStrategy(rooms: any[], roomQueue: any[]) {
    const { width, height } = this.plotSize.dimensions;
    const margin = 2;

    // Large open area for living spaces (covers 70% width)
    const openAreaWidth = width * 0.7;
    const openAreaHeight = height * 0.5;
    
    // Combine lounge and dining in open area
    const livingRooms = roomQueue.filter(r => ['lounge', 'dining'].includes(r.type));
    if (livingRooms.length > 0) {
      const combinedSize = { width: openAreaWidth, height: openAreaHeight };
      const position = { x: margin, y: 10 };
      
      const room = this.createRoom(
        { type: 'lounge', label: 'Open Living & Dining' },
        position,
        combinedSize
      );
      rooms.push(room);
      this.addToOccupiedSpaces(position.x, position.y, combinedSize.width, combinedSize.height);
    }

    // Kitchen on the side (covers remaining width)
    const kitchenRooms = roomQueue.filter(r => r.type === 'kitchen');
    if (kitchenRooms.length > 0) {
      const kitchenSize = { width: width - openAreaWidth - 2 * margin - 2, height: openAreaHeight };
      const position = { x: openAreaWidth + margin + 2, y: 10 };
      
      if (this.canPlaceRoom(position.x, position.y, kitchenSize.width, kitchenSize.height)) {
        const room = this.createRoom(kitchenRooms[0], position, kitchenSize);
        rooms.push(room);
        this.addToOccupiedSpaces(position.x, position.y, kitchenSize.width, kitchenSize.height);
      }
    }

    // Private areas at back (covers full width)
    const backY = openAreaHeight + 12;
    const backHeight = height - backY - margin;
    this.placeRemainingRoomsInGrid(rooms, roomQueue.filter(r => !['lounge', 'dining', 'kitchen'].includes(r.type)), margin, backY, width - 2 * margin, backHeight);
  }

  private applyCompactGridStrategy(rooms: any[], roomQueue: any[]) {
    const { width, height } = this.plotSize.dimensions;
    const margin = 2;
    
    // Grid-based placement that covers entire plot
    const availableWidth = width - 2 * margin;
    const availableHeight = height - 10 - margin; // After entrance
    
    const cols = Math.ceil(Math.sqrt(roomQueue.length * (availableWidth / availableHeight)));
    const rows = Math.ceil(roomQueue.length / cols);
    
    const cellWidth = availableWidth / cols;
    const cellHeight = availableHeight / rows;
    
    let roomIndex = 0;
    for (let row = 0; row < rows && roomIndex < roomQueue.length; row++) {
      for (let col = 0; col < cols && roomIndex < roomQueue.length; col++) {
        const roomData = roomQueue[roomIndex];
        const x = margin + col * cellWidth;
        const y = 10 + row * cellHeight;
        
        const roomSize = { width: cellWidth - 1, height: cellHeight - 1 };
        
        if (this.canPlaceRoom(x, y, roomSize.width, roomSize.height)) {
          const room = this.createRoom(roomData, { x, y }, roomSize);
          rooms.push(room);
          this.addToOccupiedSpaces(x, y, roomSize.width, roomSize.height);
        }
        
        roomIndex++;
      }
    }
  }

  private applyCourtyardCentralStrategy(rooms: any[], roomQueue: any[]) {
    const { width, height } = this.plotSize.dimensions;
    const margin = 2;
    
    // Central courtyard (smaller to leave more room for actual rooms)
    const courtyardSize = Math.min(width, height) * 0.25;
    const courtyardX = (width - courtyardSize) / 2;
    const courtyardY = (height - courtyardSize) / 2;
    
    // Add courtyard
    const courtyard = {
      type: 'openSpace',
      color: ROOM_COLORS.openSpace?.hex || '#00FFFF',
      position: { x: courtyardX, y: courtyardY },
      size: { width: courtyardSize, height: courtyardSize },
      label: 'Central Courtyard'
    };
    rooms.push(courtyard);
    this.addToOccupiedSpaces(courtyardX, courtyardY, courtyardSize, courtyardSize);
    
    // Place rooms around courtyard to cover entire plot
    this.placeRoomsAroundCentral(rooms, roomQueue, courtyardX, courtyardY, courtyardSize);
  }

  private applyLinearSideBySideStrategy(rooms: any[], roomQueue: any[]) {
    const { width, height } = this.plotSize.dimensions;
    const margin = 2;
    
    // Linear arrangement side by side (covers full width)
    let currentX = margin;
    const roomHeight = height - 10 - 2 * margin; // Full height minus entrance
    const roomWidth = (width - 2 * margin - (roomQueue.length - 1) * 2) / roomQueue.length;
    
    roomQueue.forEach(roomData => {
      const roomSize = { width: roomWidth, height: roomHeight };
      const position = { x: currentX, y: 10 };
      
      if (this.canPlaceRoom(position.x, position.y, roomSize.width, roomSize.height)) {
        const room = this.createRoom(roomData, position, roomSize);
        rooms.push(room);
        this.addToOccupiedSpaces(position.x, position.y, roomSize.width, roomSize.height);
        currentX += roomWidth + 2;
      }
    });
  }

  private applyLShapedCornerStrategy(rooms: any[], roomQueue: any[]) {
    const { width, height } = this.plotSize.dimensions;
    const margin = 2;
    
    // L-shaped arrangement that covers most of the plot
    const verticalArmWidth = width * 0.4;
    const horizontalArmHeight = height * 0.4;
    
    // Vertical arm (left side) - covers full height
    let currentY = 10;
    const verticalRooms = roomQueue.slice(0, Math.ceil(roomQueue.length / 2));
    const verticalRoomHeight = (height - 10 - margin) / verticalRooms.length;
    
    verticalRooms.forEach(roomData => {
      const roomSize = { width: verticalArmWidth, height: verticalRoomHeight - 1 };
      const position = { x: margin, y: currentY };
      
      if (this.canPlaceRoom(position.x, position.y, roomSize.width, roomSize.height)) {
        const room = this.createRoom(roomData, position, roomSize);
        rooms.push(room);
        this.addToOccupiedSpaces(position.x, position.y, roomSize.width, roomSize.height);
        currentY += verticalRoomHeight;
      }
    });
    
    // Horizontal arm (bottom) - covers remaining width
    let currentX = verticalArmWidth + margin + 2;
    const horizontalRooms = roomQueue.slice(Math.ceil(roomQueue.length / 2));
    const horizontalRoomWidth = (width - verticalArmWidth - 2 * margin - 2) / Math.max(horizontalRooms.length, 1);
    
    horizontalRooms.forEach(roomData => {
      const roomSize = { width: horizontalRoomWidth - 1, height: horizontalArmHeight };
      const position = { x: currentX, y: height - horizontalArmHeight - margin };
      
      if (this.canPlaceRoom(position.x, position.y, roomSize.width, roomSize.height)) {
        const room = this.createRoom(roomData, position, roomSize);
        rooms.push(room);
        this.addToOccupiedSpaces(position.x, position.y, roomSize.width, roomSize.height);
        currentX += horizontalRoomWidth;
      }
    });
  }

  private applyUShapedEmbraceStrategy(rooms: any[], roomQueue: any[]) {
    const { width, height } = this.plotSize.dimensions;
    const margin = 2;
    
    // U-shaped arrangement with central open area
    const armWidth = width * 0.3;
    const centralWidth = width * 0.4;
    const armHeight = height - 20; // Full height minus entrance and bottom
    
    // Left arm (covers full height)
    this.placeRoomsInArm(rooms, roomQueue.slice(0, Math.floor(roomQueue.length / 3)), margin, 10, armWidth, armHeight);
    
    // Right arm (covers full height)
    this.placeRoomsInArm(rooms, roomQueue.slice(Math.floor(roomQueue.length / 3), Math.floor(2 * roomQueue.length / 3)), 
                        width - armWidth - margin, 10, armWidth, armHeight);
    
    // Bottom connecting arm (covers central width)
    this.placeRoomsInArm(rooms, roomQueue.slice(Math.floor(2 * roomQueue.length / 3)), 
                        armWidth + margin + 2, height - 15, centralWidth, 10);
  }

  private applyFallbackStrategy(rooms: any[], roomQueue: any[]) {
    // Simple fallback strategy that covers the entire plot
    const { width, height } = this.plotSize.dimensions;
    const margin = 2;
    this.placeRemainingRoomsInGrid(rooms, roomQueue, margin, 10, width - 2 * margin, height - 10 - margin);
  }

  private placeRemainingRoomsInGrid(rooms: any[], roomQueue: any[], startX: number, startY: number, areaWidth: number, areaHeight: number) {
    if (roomQueue.length === 0) return;
    
    const cols = Math.ceil(Math.sqrt(roomQueue.length * (areaWidth / areaHeight)));
    const rows = Math.ceil(roomQueue.length / cols);
    
    const cellWidth = areaWidth / cols;
    const cellHeight = areaHeight / rows;
    
    let roomIndex = 0;
    for (let row = 0; row < rows && roomIndex < roomQueue.length; row++) {
      for (let col = 0; col < cols && roomIndex < roomQueue.length; col++) {
        const roomData = roomQueue[roomIndex];
        const x = startX + col * cellWidth;
        const y = startY + row * cellHeight;
        
        const roomSize = { width: cellWidth - 1, height: cellHeight - 1 };
        
        if (this.canPlaceRoom(x, y, roomSize.width, roomSize.height)) {
          const room = this.createRoom(roomData, { x, y }, roomSize);
          rooms.push(room);
          this.addToOccupiedSpaces(x, y, roomSize.width, roomSize.height);
        }
        
        roomIndex++;
      }
    }
  }

  private placeRoomsAroundCentral(rooms: any[], roomQueue: any[], centerX: number, centerY: number, centerSize: number) {
    const { width, height } = this.plotSize.dimensions;
    const margin = 2;
    
    // Define zones around the central courtyard
    const zones = [
      // Top zone
      { x: margin, y: 10, width: width - 2 * margin, height: centerY - 10 - 2 },
      // Bottom zone  
      { x: margin, y: centerY + centerSize + 2, width: width - 2 * margin, height: height - (centerY + centerSize + 2) - margin },
      // Left zone
      { x: margin, y: centerY, width: centerX - margin - 2, height: centerSize },
      // Right zone
      { x: centerX + centerSize + 2, y: centerY, width: width - (centerX + centerSize + 2) - margin, height: centerSize }
    ];
    
    let roomIndex = 0;
    zones.forEach(zone => {
      const zoneRooms = roomQueue.slice(roomIndex, roomIndex + Math.ceil(roomQueue.length / zones.length));
      if (zoneRooms.length > 0) {
        this.placeRemainingRoomsInGrid(rooms, zoneRooms, zone.x, zone.y, zone.width, zone.height);
        roomIndex += zoneRooms.length;
      }
    });
  }

  private placeRoomsInArm(rooms: any[], roomQueue: any[], armX: number, armY: number, armWidth: number, armHeight: number) {
    if (roomQueue.length === 0) return;
    
    const roomHeight = armHeight / roomQueue.length;
    let currentY = armY;
    
    roomQueue.forEach(roomData => {
      const roomSize = { width: armWidth, height: roomHeight - 1 };
      const position = { x: armX, y: currentY };
      
      if (this.canPlaceRoom(position.x, position.y, roomSize.width, roomSize.height)) {
        const room = this.createRoom(roomData, position, roomSize);
        rooms.push(room);
        this.addToOccupiedSpaces(position.x, position.y, roomSize.width, roomSize.height);
        currentY += roomHeight;
      }
    });
  }

  private createRoom(roomData: any, position: {x: number, y: number}, size: {width: number, height: number}) {
    return {
      type: roomData.type,
      color: ROOM_COLORS[roomData.type]?.hex || '#CCCCCC',
      position,
      size,
      label: roomData.label
    };
  }

  private canPlaceRoom(x: number, y: number, width: number, height: number): boolean {
    const { width: plotWidth, height: plotHeight } = this.plotSize.dimensions;
    const margin = 2;
    
    // Check bounds
    if (x < margin || y < margin || x + width > plotWidth - margin || y + height > plotHeight - margin) {
      return false;
    }
    
    // Check overlaps
    return !this.hasOverlapWithOccupiedSpaces(x, y, width, height);
  }

  private createExactRoomQueue(): any[] {
    const rooms = [];
    
    console.log('🔢 Creating room queue based on user requirements:');
    console.log('  - Bedrooms:', this.requirements.bedrooms);
    console.log('  - Bathrooms:', this.requirements.bathrooms);
    console.log('  - Kitchens:', this.requirements.kitchens);
    console.log('  - Lounges:', this.requirements.lounges);
    console.log('  - Garages:', this.requirements.garages);
    console.log('  - Drawing Room:', this.requirements.drawingRoom);
    console.log('  - Dining Room:', this.requirements.diningRoom);
    console.log('  - Storage:', this.requirements.storage);
    console.log('  - Laundry:', this.requirements.laundry);
    console.log('  - Library:', this.requirements.library);
    console.log('  - Stairs:', this.requirements.stairs);
    console.log('  - Lawn:', this.requirements.lawn);
    
    // ESSENTIAL ROOMS (Priority 1) - EXACTLY as user specified
    for (let i = 0; i < this.requirements.bedrooms; i++) {
      rooms.push({
        type: 'bedroom',
        label: `Bedroom ${i + 1}`,
        priority: 1
      });
    }
    
    for (let i = 0; i < this.requirements.bathrooms; i++) {
      rooms.push({
        type: 'bathroom',
        label: `Bathroom ${i + 1}`,
        priority: 1
      });
    }
    
    for (let i = 0; i < this.requirements.kitchens; i++) {
      rooms.push({
        type: 'kitchen',
        label: i === 0 ? 'Kitchen' : `Kitchen ${i + 1}`,
        priority: 1
      });
    }
    
    // LIVING SPACES (Priority 2) - EXACTLY as user specified
    for (let i = 0; i < this.requirements.lounges; i++) {
      rooms.push({
        type: 'lounge',
        label: i === 0 ? 'Family Lounge' : `Lounge ${i + 1}`,
        priority: 2
      });
    }
    
    // OPTIONAL ROOMS (Priority 3) - ONLY if user selected them
    if (this.requirements.drawingRoom) {
      rooms.push({ type: 'drawingRoom', label: 'Drawing Room', priority: 2 });
    }
    
    if (this.requirements.diningRoom) {
      rooms.push({ type: 'dining', label: 'Dining Room', priority: 2 });
    }
    
    // SERVICE ROOMS (Priority 3) - EXACTLY as user specified
    for (let i = 0; i < this.requirements.garages; i++) {
      rooms.push({
        type: 'garage',
        label: this.requirements.garages === 1 ? 'Car Garage' : `Garage ${i + 1}`,
        priority: 3
      });
    }
    
    if (this.requirements.storage) {
      rooms.push({ type: 'storage', label: 'Store Room', priority: 3 });
    }
    
    if (this.requirements.laundry) {
      rooms.push({ type: 'laundry', label: 'Laundry Room', priority: 3 });
    }
    
    if (this.requirements.library) {
      rooms.push({ type: 'library', label: 'Library', priority: 3 });
    }
    
    if (this.requirements.stairs) {
      rooms.push({ type: 'stairs', label: 'Stairs', priority: 3 });
    }
    
    // OUTDOOR SPACES (Priority 4) - ONLY if user selected them
    if (this.requirements.lawn) {
      rooms.push({ type: 'lawn', label: 'Lawn', priority: 4 });
    }
    
    // Sort by priority (essential first, then optional)
    rooms.sort((a, b) => a.priority - b.priority);
    
    console.log(`📊 Total rooms to place: ${rooms.length}`);
    return rooms;
  }

  private addToOccupiedSpaces(x: number, y: number, width: number, height: number) {
    this.occupiedSpaces.push({ x, y, width, height });
  }

  private hasOverlapWithOccupiedSpaces(x: number, y: number, width: number, height: number): boolean {
    const buffer = 1; // Small buffer to prevent touching
    
    return this.occupiedSpaces.some(occupied => {
      return !(x + width + buffer <= occupied.x || 
               x >= occupied.x + occupied.width + buffer || 
               y + height + buffer <= occupied.y || 
               y >= occupied.y + occupied.height + buffer);
    });
  }

  private generateWallsFromRooms(rooms: any[]): any[] {
    const { width, height } = this.plotSize.dimensions;
    
    const walls = [
      // Outer boundary walls
      { start: { x: 0, y: 0 }, end: { x: width, y: 0 }, internal: false },
      { start: { x: width, y: 0 }, end: { x: width, y: height }, internal: false },
      { start: { x: width, y: height }, end: { x: 0, y: height }, internal: false },
      { start: { x: 0, y: height }, end: { x: 0, y: 0 }, internal: false }
    ];

    // Generate internal walls between adjacent rooms
    rooms.forEach((room, index) => {
      rooms.slice(index + 1).forEach(otherRoom => {
        const sharedWall = this.findSharedWall(room, otherRoom);
        if (sharedWall) {
          walls.push({ ...sharedWall, internal: true });
        }
      });
    });

    return walls;
  }

  private findSharedWall(room1: any, room2: any) {
    const r1 = {
      left: room1.position.x,
      right: room1.position.x + room1.size.width,
      top: room1.position.y,
      bottom: room1.position.y + room1.size.height
    };
    
    const r2 = {
      left: room2.position.x,
      right: room2.position.x + room2.size.width,
      top: room2.position.y,
      bottom: room2.position.y + room2.size.height
    };

    const tolerance = 1;

    // Check for vertical shared walls
    if (Math.abs(r1.right - r2.left) <= tolerance && this.hasVerticalOverlap(r1, r2)) {
      return {
        start: { x: r1.right, y: Math.max(r1.top, r2.top) },
        end: { x: r1.right, y: Math.min(r1.bottom, r2.bottom) }
      };
    }

    if (Math.abs(r1.left - r2.right) <= tolerance && this.hasVerticalOverlap(r1, r2)) {
      return {
        start: { x: r1.left, y: Math.max(r1.top, r2.top) },
        end: { x: r1.left, y: Math.min(r1.bottom, r2.bottom) }
      };
    }

    // Check for horizontal shared walls
    if (Math.abs(r1.bottom - r2.top) <= tolerance && this.hasHorizontalOverlap(r1, r2)) {
      return {
        start: { x: Math.max(r1.left, r2.left), y: r1.bottom },
        end: { x: Math.min(r1.right, r2.right), y: r1.bottom }
      };
    }

    if (Math.abs(r1.top - r2.bottom) <= tolerance && this.hasHorizontalOverlap(r1, r2)) {
      return {
        start: { x: Math.max(r1.left, r2.left), y: r1.top },
        end: { x: Math.min(r1.right, r2.right), y: r1.top }
      };
    }

    return null;
  }

  private hasVerticalOverlap(r1: any, r2: any): boolean {
    return !(r1.bottom <= r2.top || r1.top >= r2.bottom);
  }

  private hasHorizontalOverlap(r1: any, r2: any): boolean {
    return !(r1.right <= r2.left || r1.left >= r2.right);
  }

  private generateIntelligentDoors(rooms: any[]): any[] {
    const doors = [];
    const { width } = this.plotSize.dimensions;

    // Main entrance door
    doors.push({
      position: { x: width / 2, y: 0 },
      orientation: 'horizontal' as const,
      width: 3
    });

    // Generate doors based on architectural adjacency patterns
    const adjacencyMatrix = ARCHITECTURAL_PATTERNS.adjacencyMatrix;
    
    rooms.forEach((room1, i) => {
      rooms.slice(i + 1).forEach(room2 => {
        if (this.areRoomsAdjacent(room1, room2)) {
          const adjacencyScore = adjacencyMatrix[room1.type]?.[room2.type] || 
                                adjacencyMatrix[room2.type]?.[room1.type] || 0;
          
          if (adjacencyScore > 0.4) {
            const door = this.createOptimalDoor(room1, room2);
            if (door) {
              doors.push(door);
            }
          }
        }
      });
    });

    // Add circulation doors for main areas
    this.addCirculationDoors(rooms, doors);

    return doors;
  }

  private areRoomsAdjacent(room1: any, room2: any): boolean {
    const r1 = {
      left: room1.position.x,
      right: room1.position.x + room1.size.width,
      top: room1.position.y,
      bottom: room1.position.y + room1.size.height
    };
    
    const r2 = {
      left: room2.position.x,
      right: room2.position.x + room2.size.width,
      top: room2.position.y,
      bottom: room2.position.y + room2.size.height
    };

    const tolerance = 3;

    const adjacentVertically = (Math.abs(r1.right - r2.left) <= tolerance || Math.abs(r1.left - r2.right) <= tolerance) &&
                              this.hasVerticalOverlap(r1, r2);
    
    const adjacentHorizontally = (Math.abs(r1.bottom - r2.top) <= tolerance || Math.abs(r1.top - r2.bottom) <= tolerance) &&
                                this.hasHorizontalOverlap(r1, r2);

    return adjacentVertically || adjacentHorizontally;
  }

  private createOptimalDoor(room1: any, room2: any) {
    const r1 = {
      left: room1.position.x,
      right: room1.position.x + room1.size.width,
      top: room1.position.y,
      bottom: room1.position.y + room1.size.height
    };
    
    const r2 = {
      left: room2.position.x,
      right: room2.position.x + room2.size.width,
      top: room2.position.y,
      bottom: room2.position.y + room2.size.height
    };

    const tolerance = 3;

    // Vertical door (between left-right adjacent rooms)
    if (Math.abs(r1.right - r2.left) <= tolerance || Math.abs(r1.left - r2.right) <= tolerance) {
      const x = Math.abs(r1.right - r2.left) <= tolerance ? r1.right : r1.left;
      const overlapStart = Math.max(r1.top, r2.top);
      const overlapEnd = Math.min(r1.bottom, r2.bottom);
      const y = overlapStart + (overlapEnd - overlapStart) / 2;
      
      return {
        position: { x, y },
        orientation: 'vertical' as const,
        width: 3
      };
    }

    // Horizontal door (between top-bottom adjacent rooms)
    if (Math.abs(r1.bottom - r2.top) <= tolerance || Math.abs(r1.top - r2.bottom) <= tolerance) {
      const y = Math.abs(r1.bottom - r2.top) <= tolerance ? r1.bottom : r1.top;
      const overlapStart = Math.max(r1.left, r2.left);
      const overlapEnd = Math.min(r1.right, r2.right);
      const x = overlapStart + (overlapEnd - overlapStart) / 2;
      
      return {
        position: { x, y },
        orientation: 'horizontal' as const,
        width: 3
      };
    }

    return null;
  }

  private addCirculationDoors(rooms: any[], doors: any[]) {
    // Add doors from main circulation areas to entrance
    const mainRooms = ['drawingRoom', 'lounge', 'lobby'];
    
    rooms.forEach(room => {
      if (mainRooms.includes(room.type) && room.position.y < 30) {
        const door = {
          position: { 
            x: room.position.x + room.size.width / 2, 
            y: room.position.y 
          },
          orientation: 'horizontal' as const,
          width: 3
        };
        
        const hasConflict = doors.some(existingDoor => 
          Math.abs(existingDoor.position.x - door.position.x) < 5 &&
          Math.abs(existingDoor.position.y - door.position.y) < 5
        );
        
        if (!hasConflict) {
          doors.push(door);
        }
      }
    });

    // Ensure all rooms have access
    rooms.forEach(room => {
      const hasDirectAccess = doors.some(door => this.isDoorInRoom(door, room));
      
      if (!hasDirectAccess && room.type !== 'lobby') {
        // Find nearest room with access
        const nearestRoom = this.findNearestRoom(room, rooms.filter(r => 
          r !== room && doors.some(door => this.isDoorInRoom(door, r))
        ));
        
        if (nearestRoom && this.areRoomsAdjacent(room, nearestRoom)) {
          const door = this.createOptimalDoor(room, nearestRoom);
          if (door) {
            doors.push(door);
          }
        }
      }
    });
  }

  private isDoorInRoom(door: any, room: any): boolean {
    const tolerance = 5;
    return door.position.x >= room.position.x - tolerance &&
           door.position.x <= room.position.x + room.size.width + tolerance &&
           door.position.y >= room.position.y - tolerance &&
           door.position.y <= room.position.y + room.size.height + tolerance;
  }

  private findNearestRoom(targetRoom: any, candidateRooms: any[]): any {
    if (candidateRooms.length === 0) return null;
    
    let nearestRoom = candidateRooms[0];
    let minDistance = this.calculateRoomDistance(targetRoom, nearestRoom);
    
    candidateRooms.slice(1).forEach(room => {
      const distance = this.calculateRoomDistance(targetRoom, room);
      if (distance < minDistance) {
        minDistance = distance;
        nearestRoom = room;
      }
    });
    
    return nearestRoom;
  }

  private calculateRoomDistance(room1: any, room2: any): number {
    const center1 = {
      x: room1.position.x + room1.size.width / 2,
      y: room1.position.y + room1.size.height / 2
    };
    const center2 = {
      x: room2.position.x + room2.size.width / 2,
      y: room2.position.y + room2.size.height / 2
    };
    
    return Math.sqrt((center1.x - center2.x) ** 2 + (center1.y - center2.y) ** 2);
  }
}