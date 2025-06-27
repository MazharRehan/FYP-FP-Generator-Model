import { PlotSize, RoomRequirements, GeneratedFloorPlan, ROOM_COLORS, ARCHITECTURAL_STANDARDS } from '../types/floorplan';

export class FloorPlanGeneratorLogic {
  private plotSize: PlotSize;
  private requirements: RoomRequirements;
  private layoutVariation: number;
  private randomSeed: number;

  constructor(plotSize: PlotSize, requirements: RoomRequirements, variation = 0) {
    this.plotSize = plotSize;
    this.requirements = requirements;
    this.layoutVariation = variation;
    // Create truly unique seed for each generation
    this.randomSeed = Date.now() + variation * 10000 + Math.random() * 100000;
  }

  // Seeded random number generator for consistent but varied results
  private seededRandom(): number {
    this.randomSeed = (this.randomSeed * 9301 + 49297) % 233280;
    return this.randomSeed / 233280;
  }

  generate(): GeneratedFloorPlan {
    const rooms = this.generateArchitecturallySound();
    const walls = this.generateProperWalls(rooms);
    const doors = this.generateConnectedDoors(rooms);

    return {
      id: `plan-${Date.now()}-${this.layoutVariation}-${Math.random().toString(36).substr(2, 9)}`,
      plotSize: this.plotSize,
      requirements: this.requirements,
      rooms,
      walls,
      doors,
      createdAt: new Date()
    };
  }

  private generateArchitecturallySound() {
    const { width, height } = this.plotSize.dimensions;
    const rooms = [];

    // Choose layout strategy based on variation and plot size
    const strategies = ['traditional', 'modern', 'compact', 'open-plan', 'courtyard'];
    const strategy = strategies[this.layoutVariation % strategies.length];

    console.log(`Generating ${strategy} layout for ${this.plotSize.name}`);

    switch (strategy) {
      case 'traditional':
        return this.generateTraditionalLayout();
      case 'modern':
        return this.generateModernLayout();
      case 'compact':
        return this.generateCompactLayout();
      case 'open-plan':
        return this.generateOpenPlanLayout();
      case 'courtyard':
        return this.generateCourtyardLayout();
      default:
        return this.generateTraditionalLayout();
    }
  }

  private generateTraditionalLayout() {
    const rooms = [];
    const { width, height } = this.plotSize.dimensions;
    
    // Traditional Pakistani house layout with proper circulation
    let currentY = 2;
    const margin = 2;

    // 1. Main entrance and lobby (front)
    const entranceHeight = 8;
    rooms.push({
      type: 'lobby',
      color: ROOM_COLORS.lobby.hex,
      position: { x: margin, y: currentY },
      size: { width: width - 2 * margin, height: entranceHeight },
      label: 'Main Entrance'
    });
    currentY += entranceHeight + 2;

    // 2. Drawing room (formal reception)
    if (this.requirements.drawingRoom) {
      const drawingHeight = Math.max(12, height * 0.15);
      rooms.push({
        type: 'drawingRoom',
        color: ROOM_COLORS.drawingRoom.hex,
        position: { x: margin, y: currentY },
        size: { width: width - 2 * margin, height: drawingHeight },
        label: 'Drawing Room'
      });
      currentY += drawingHeight + 2;
    }

    // 3. Living and dining area (side by side)
    const livingDiningHeight = Math.max(14, height * 0.2);
    const livingWidth = (width - 3 * margin) * (0.55 + this.seededRandom() * 0.1);
    
    if (this.requirements.lounges > 0) {
      rooms.push({
        type: 'lounge',
        color: ROOM_COLORS.lounge.hex,
        position: { x: margin, y: currentY },
        size: { width: livingWidth, height: livingDiningHeight },
        label: 'Family Lounge'
      });
    }

    if (this.requirements.diningRoom) {
      rooms.push({
        type: 'dining',
        color: ROOM_COLORS.dining.hex,
        position: { x: livingWidth + margin + 2, y: currentY },
        size: { width: width - livingWidth - 3 * margin - 2, height: livingDiningHeight },
        label: 'Dining Room'
      });
    }
    currentY += livingDiningHeight + 2;

    // 4. Kitchen area with pantry
    if (this.requirements.kitchens > 0) {
      const kitchenHeight = Math.max(10, height * 0.12);
      let kitchenWidth = width - 2 * margin;
      
      if (this.requirements.pantry) {
        kitchenWidth = (width - 3 * margin) * 0.7;
        rooms.push({
          type: 'kitchen',
          color: ROOM_COLORS.kitchen.hex,
          position: { x: margin, y: currentY },
          size: { width: kitchenWidth, height: kitchenHeight },
          label: 'Kitchen'
        });

        rooms.push({
          type: 'pantry',
          color: ROOM_COLORS.pantry.hex,
          position: { x: kitchenWidth + margin + 2, y: currentY },
          size: { width: width - kitchenWidth - 3 * margin - 2, height: kitchenHeight },
          label: 'Pantry'
        });
      } else {
        rooms.push({
          type: 'kitchen',
          color: ROOM_COLORS.kitchen.hex,
          position: { x: margin, y: currentY },
          size: { width: kitchenWidth, height: kitchenHeight },
          label: 'Kitchen'
        });
      }
      currentY += kitchenHeight + 2;
    }

    // 5. Bedroom area (upper floor simulation)
    this.addBedroomZone(rooms, margin, currentY, width - 2 * margin, height - currentY - margin);

    // 6. Add service areas
    this.addServiceAreas(rooms, width, height);

    return rooms;
  }

  private generateModernLayout() {
    const rooms = [];
    const { width, height } = this.plotSize.dimensions;
    
    // Modern open-plan layout with zones
    const margin = 2;

    // 1. Open entrance with living space
    const frontZoneHeight = height * 0.4;
    
    if (this.requirements.lounges > 0) {
      const livingWidth = width * (0.6 + this.seededRandom() * 0.2);
      rooms.push({
        type: 'lounge',
        color: ROOM_COLORS.lounge.hex,
        position: { x: margin, y: margin },
        size: { width: livingWidth - margin, height: frontZoneHeight },
        label: 'Open Living Area'
      });

      // Modern kitchen island concept
      if (this.requirements.kitchens > 0) {
        rooms.push({
          type: 'kitchen',
          color: ROOM_COLORS.kitchen.hex,
          position: { x: livingWidth + 2, y: margin },
          size: { width: width - livingWidth - margin - 2, height: frontZoneHeight * 0.6 },
          label: 'Modern Kitchen'
        });

        if (this.requirements.diningRoom) {
          rooms.push({
            type: 'dining',
            color: ROOM_COLORS.dining.hex,
            position: { x: livingWidth + 2, y: margin + frontZoneHeight * 0.6 + 2 },
            size: { width: width - livingWidth - margin - 2, height: frontZoneHeight * 0.4 - 2 },
            label: 'Dining Area'
          });
        }
      }
    }

    // 2. Private zone (bedrooms)
    const privateZoneY = frontZoneHeight + margin + 2;
    const privateZoneHeight = height - privateZoneY - margin;
    
    this.addModernBedroomLayout(rooms, margin, privateZoneY, width - 2 * margin, privateZoneHeight);

    return rooms;
  }

  private generateCompactLayout() {
    const rooms = [];
    const { width, height } = this.plotSize.dimensions;
    
    // Efficient space utilization for smaller plots
    const margin = 1;
    const totalRooms = this.countRequiredRooms();
    
    // Calculate optimal grid
    const cols = Math.ceil(Math.sqrt(totalRooms * (width / height)));
    const rows = Math.ceil(totalRooms / cols);
    
    const cellWidth = (width - margin * (cols + 1)) / cols;
    const cellHeight = (height - margin * (rows + 1)) / rows;

    let roomIndex = 0;
    const roomQueue = this.createPriorityRoomQueue();

    for (let row = 0; row < rows && roomIndex < roomQueue.length; row++) {
      for (let col = 0; col < cols && roomIndex < roomQueue.length; col++) {
        const roomData = roomQueue[roomIndex];
        const x = margin + col * (cellWidth + margin);
        const y = margin + row * (cellHeight + margin);

        // Adjust room size based on type and importance
        const sizeMultiplier = this.getRoomSizeMultiplier(roomData.type);
        const actualWidth = Math.min(cellWidth * sizeMultiplier.width, width - x - margin);
        const actualHeight = Math.min(cellHeight * sizeMultiplier.height, height - y - margin);

        rooms.push({
          type: roomData.type,
          color: ROOM_COLORS[roomData.type]?.hex || '#CCCCCC',
          position: { x, y },
          size: { width: actualWidth, height: actualHeight },
          label: roomData.label
        });

        roomIndex++;
      }
    }

    return rooms;
  }

  private generateOpenPlanLayout() {
    const rooms = [];
    const { width, height } = this.plotSize.dimensions;
    
    // Open plan with minimal walls
    const margin = 2;

    // 1. Large open living space
    const openAreaHeight = height * 0.5;
    const kitchenWidth = width * 0.3;

    if (this.requirements.lounges > 0 && this.requirements.diningRoom) {
      // Combined living-dining space
      rooms.push({
        type: 'lounge',
        color: ROOM_COLORS.lounge.hex,
        position: { x: margin, y: margin },
        size: { width: width - kitchenWidth - 3 * margin, height: openAreaHeight },
        label: 'Open Living & Dining'
      });
    }

    if (this.requirements.kitchens > 0) {
      rooms.push({
        type: 'kitchen',
        color: ROOM_COLORS.kitchen.hex,
        position: { x: width - kitchenWidth - margin, y: margin },
        size: { width: kitchenWidth, height: openAreaHeight * 0.7 },
        label: 'Open Kitchen'
      });
    }

    // 2. Private bedroom wing
    const bedroomZoneY = openAreaHeight + margin + 2;
    this.addBedroomZone(rooms, margin, bedroomZoneY, width - 2 * margin, height - bedroomZoneY - margin);

    return rooms;
  }

  private generateCourtyardLayout() {
    const rooms = [];
    const { width, height } = this.plotSize.dimensions;
    
    // Traditional courtyard design
    const margin = 2;
    const courtyardSize = Math.min(width, height) * (0.25 + this.seededRandom() * 0.1);
    const courtyardX = (width - courtyardSize) / 2;
    const courtyardY = (height - courtyardSize) / 2;

    // Central courtyard
    rooms.push({
      type: 'openSpace',
      color: ROOM_COLORS.openSpace.hex,
      position: { x: courtyardX, y: courtyardY },
      size: { width: courtyardSize, height: courtyardSize },
      label: 'Central Courtyard'
    });

    // Rooms around courtyard
    // Front (drawing room)
    if (this.requirements.drawingRoom) {
      rooms.push({
        type: 'drawingRoom',
        color: ROOM_COLORS.drawingRoom.hex,
        position: { x: margin, y: margin },
        size: { width: width - 2 * margin, height: courtyardY - margin - 2 },
        label: 'Drawing Room'
      });
    }

    // Left side (kitchen and dining)
    const leftWidth = courtyardX - margin - 2;
    if (leftWidth > 8 && this.requirements.kitchens > 0) {
      const kitchenHeight = courtyardSize * 0.6;
      rooms.push({
        type: 'kitchen',
        color: ROOM_COLORS.kitchen.hex,
        position: { x: margin, y: courtyardY },
        size: { width: leftWidth, height: kitchenHeight },
        label: 'Kitchen'
      });

      if (this.requirements.diningRoom) {
        rooms.push({
          type: 'dining',
          color: ROOM_COLORS.dining.hex,
          position: { x: margin, y: courtyardY + kitchenHeight + 2 },
          size: { width: leftWidth, height: courtyardSize - kitchenHeight - 2 },
          label: 'Dining Room'
        });
      }
    }

    // Right side (bedrooms)
    const rightX = courtyardX + courtyardSize + 2;
    const rightWidth = width - rightX - margin;
    if (rightWidth > 10) {
      this.addBedroomZone(rooms, rightX, courtyardY, rightWidth, courtyardSize);
    }

    // Back area (service rooms)
    const backY = courtyardY + courtyardSize + 2;
    const backHeight = height - backY - margin;
    if (backHeight > 8) {
      this.addServiceAreas(rooms, width, height, margin, backY, width - 2 * margin, backHeight);
    }

    return rooms;
  }

  private addBedroomZone(rooms: any[], x: number, y: number, zoneWidth: number, zoneHeight: number) {
    const bedroomCount = this.requirements.bedrooms;
    const bathroomCount = this.requirements.bathrooms;
    
    if (bedroomCount === 0) return;

    // Master bedroom gets priority and larger size
    const masterWidth = Math.min(zoneWidth * 0.6, zoneWidth - 8); // Leave space for bathroom
    const masterHeight = Math.max(14, zoneHeight * 0.4);

    rooms.push({
      type: 'bedroom',
      color: ROOM_COLORS.bedroom.hex,
      position: { x, y },
      size: { width: masterWidth, height: masterHeight },
      label: 'Master Bedroom'
    });

    // Master bathroom (attached)
    if (bathroomCount > 0) {
      rooms.push({
        type: 'bathroom',
        color: ROOM_COLORS.bathroom.hex,
        position: { x: x + masterWidth + 2, y },
        size: { width: zoneWidth - masterWidth - 2, height: Math.min(masterHeight, 10) },
        label: 'Master Bathroom'
      });
    }

    // Additional bedrooms
    let currentY = y + masterHeight + 2;
    const remainingHeight = zoneHeight - masterHeight - 2;
    const additionalBedrooms = bedroomCount - 1;

    if (additionalBedrooms > 0 && remainingHeight > 10) {
      const bedroomHeight = remainingHeight / additionalBedrooms;
      
      for (let i = 0; i < additionalBedrooms; i++) {
        const bedroomWidth = zoneWidth * (0.7 + this.seededRandom() * 0.2);
        
        rooms.push({
          type: 'bedroom',
          color: ROOM_COLORS.bedroom.hex,
          position: { x, y: currentY },
          size: { width: bedroomWidth, height: bedroomHeight - 2 },
          label: `Bedroom ${i + 2}`
        });

        // Additional bathroom if needed
        if (i < bathroomCount - 1) {
          rooms.push({
            type: 'bathroom',
            color: ROOM_COLORS.bathroom.hex,
            position: { x: x + bedroomWidth + 2, y: currentY },
            size: { width: zoneWidth - bedroomWidth - 2, height: Math.min(bedroomHeight - 2, 8) },
            label: `Bathroom ${i + 2}`
          });
        }

        currentY += bedroomHeight;
      }
    }
  }

  private addModernBedroomLayout(rooms: any[], x: number, y: number, zoneWidth: number, zoneHeight: number) {
    const bedroomCount = this.requirements.bedrooms;
    
    if (bedroomCount === 0) return;

    // Modern layout with walk-in closets and en-suite bathrooms
    const bedroomWidth = zoneWidth / Math.min(bedroomCount, 2); // Max 2 bedrooms per row
    const bedroomHeight = zoneHeight / Math.ceil(bedroomCount / 2);

    for (let i = 0; i < bedroomCount; i++) {
      const col = i % 2;
      const row = Math.floor(i / 2);
      
      const roomX = x + col * bedroomWidth;
      const roomY = y + row * bedroomHeight;
      
      const actualBedroomWidth = bedroomWidth * 0.7; // Leave space for bathroom
      
      rooms.push({
        type: 'bedroom',
        color: ROOM_COLORS.bedroom.hex,
        position: { x: roomX, y: roomY },
        size: { width: actualBedroomWidth, height: bedroomHeight - 2 },
        label: i === 0 ? 'Master Bedroom' : `Bedroom ${i + 1}`
      });

      // En-suite bathroom
      if (i < this.requirements.bathrooms) {
        rooms.push({
          type: 'bathroom',
          color: ROOM_COLORS.bathroom.hex,
          position: { x: roomX + actualBedroomWidth + 2, y: roomY },
          size: { width: bedroomWidth - actualBedroomWidth - 2, height: Math.min(bedroomHeight - 2, 8) },
          label: i === 0 ? 'Master Bathroom' : `Bathroom ${i + 1}`
        });
      }

      // Walk-in closet for master bedroom
      if (i === 0 && this.requirements.walkInCloset) {
        rooms.push({
          type: 'walkInCloset',
          color: ROOM_COLORS.walkInCloset.hex,
          position: { x: roomX, y: roomY + bedroomHeight - 8 },
          size: { width: actualBedroomWidth * 0.4, height: 6 },
          label: 'Walk-in Closet'
        });
      }
    }
  }

  private addServiceAreas(rooms: any[], totalWidth: number, totalHeight: number, x?: number, y?: number, width?: number, height?: number) {
    const serviceX = x || 2;
    const serviceY = y || totalHeight - 12;
    const serviceWidth = width || totalWidth - 4;
    const serviceHeight = height || 10;

    const serviceRooms = [];
    
    if (this.requirements.garages > 0) {
      serviceRooms.push({ type: 'garage', label: 'Car Garage', priority: 1 });
    }
    if (this.requirements.laundry) {
      serviceRooms.push({ type: 'laundry', label: 'Laundry Room', priority: 2 });
    }
    if (this.requirements.storage) {
      serviceRooms.push({ type: 'storage', label: 'Store Room', priority: 3 });
    }
    if (this.requirements.study) {
      serviceRooms.push({ type: 'study', label: 'Study Room', priority: 1 });
    }
    if (this.requirements.library) {
      serviceRooms.push({ type: 'library', label: 'Library', priority: 2 });
    }

    if (serviceRooms.length === 0) return;

    // Sort by priority
    serviceRooms.sort((a, b) => a.priority - b.priority);

    const roomWidth = serviceWidth / serviceRooms.length;
    let currentX = serviceX;

    serviceRooms.forEach(roomData => {
      rooms.push({
        type: roomData.type,
        color: ROOM_COLORS[roomData.type]?.hex || '#CCCCCC',
        position: { x: currentX, y: serviceY },
        size: { width: roomWidth - 2, height: serviceHeight },
        label: roomData.label
      });
      currentX += roomWidth;
    });

    // Add outdoor spaces
    if (this.requirements.lawn) {
      rooms.push({
        type: 'lawn',
        color: ROOM_COLORS.lawn.hex,
        position: { x: totalWidth - 15, y: totalHeight - 15 },
        size: { width: 12, height: 12 },
        label: 'Lawn'
      });
    }

    if (this.requirements.terrace) {
      rooms.push({
        type: 'terrace',
        color: ROOM_COLORS.terrace.hex,
        position: { x: 2, y: 2 },
        size: { width: 12, height: 8 },
        label: 'Terrace'
      });
    }

    if (this.requirements.balcony) {
      rooms.push({
        type: 'balcony',
        color: ROOM_COLORS.balcony.hex,
        position: { x: totalWidth - 10, y: 20 },
        size: { width: 8, height: 6 },
        label: 'Balcony'
      });
    }
  }

  private countRequiredRooms(): number {
    let count = this.requirements.bedrooms + this.requirements.bathrooms + this.requirements.kitchens + this.requirements.lounges;
    
    if (this.requirements.drawingRoom) count++;
    if (this.requirements.diningRoom) count++;
    if (this.requirements.garages > 0) count++;
    if (this.requirements.storage) count++;
    if (this.requirements.laundry) count++;
    if (this.requirements.study) count++;
    if (this.requirements.library) count++;
    if (this.requirements.walkInCloset) count++;
    if (this.requirements.pantry) count++;
    
    return Math.max(count, 4);
  }

  private createPriorityRoomQueue() {
    const rooms = [];
    
    // Priority 1: Essential rooms
    for (let i = 0; i < this.requirements.bedrooms; i++) {
      rooms.push({
        type: 'bedroom',
        label: i === 0 ? 'Master Bedroom' : `Bedroom ${i + 1}`,
        priority: 1
      });
    }
    
    for (let i = 0; i < this.requirements.bathrooms; i++) {
      rooms.push({
        type: 'bathroom',
        label: i === 0 ? 'Master Bathroom' : `Bathroom ${i + 1}`,
        priority: 1
      });
    }
    
    if (this.requirements.kitchens > 0) {
      rooms.push({ type: 'kitchen', label: 'Kitchen', priority: 1 });
    }
    
    // Priority 2: Living spaces
    if (this.requirements.lounges > 0) {
      rooms.push({ type: 'lounge', label: 'Family Lounge', priority: 2 });
    }
    
    if (this.requirements.drawingRoom) {
      rooms.push({ type: 'drawingRoom', label: 'Drawing Room', priority: 2 });
    }
    
    if (this.requirements.diningRoom) {
      rooms.push({ type: 'dining', label: 'Dining Room', priority: 2 });
    }
    
    // Priority 3: Service and optional rooms
    if (this.requirements.garages > 0) {
      rooms.push({ type: 'garage', label: 'Car Garage', priority: 3 });
    }
    
    if (this.requirements.laundry) {
      rooms.push({ type: 'laundry', label: 'Laundry Room', priority: 3 });
    }
    
    if (this.requirements.storage) {
      rooms.push({ type: 'storage', label: 'Store Room', priority: 3 });
    }
    
    if (this.requirements.study) {
      rooms.push({ type: 'study', label: 'Study Room', priority: 3 });
    }
    
    if (this.requirements.library) {
      rooms.push({ type: 'library', label: 'Library', priority: 3 });
    }
    
    if (this.requirements.walkInCloset) {
      rooms.push({ type: 'walkInCloset', label: 'Walk-in Closet', priority: 3 });
    }
    
    if (this.requirements.pantry) {
      rooms.push({ type: 'pantry', label: 'Pantry', priority: 3 });
    }
    
    // Sort by priority
    rooms.sort((a, b) => a.priority - b.priority);
    
    return rooms;
  }

  private getRoomSizeMultiplier(roomType: string) {
    const multipliers = {
      'bedroom': { width: 1.2, height: 1.2 },
      'bathroom': { width: 0.8, height: 0.8 },
      'kitchen': { width: 1.1, height: 1.0 },
      'lounge': { width: 1.3, height: 1.3 },
      'drawingRoom': { width: 1.4, height: 1.2 },
      'dining': { width: 1.1, height: 1.1 },
      'garage': { width: 1.2, height: 1.5 },
      'storage': { width: 0.7, height: 0.7 },
      'laundry': { width: 0.8, height: 0.8 },
      'study': { width: 1.0, height: 1.0 },
      'library': { width: 1.1, height: 1.1 },
      'walkInCloset': { width: 0.6, height: 0.8 },
      'pantry': { width: 0.7, height: 0.9 }
    };
    
    return multipliers[roomType] || { width: 1.0, height: 1.0 };
  }

  private generateProperWalls(rooms: any[]) {
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

  private generateConnectedDoors(rooms: any[]) {
    const doors = [];
    const { width } = this.plotSize.dimensions;

    // Main entrance door
    doors.push({
      position: { x: width / 2, y: 0 },
      orientation: 'horizontal' as const,
      width: 3
    });

    // Define logical room connections based on architectural standards
    const connectionRules = [
      // Bedroom-bathroom connections
      { from: 'bedroom', to: 'bathroom', priority: 1 },
      // Kitchen connections
      { from: 'kitchen', to: 'dining', priority: 1 },
      { from: 'kitchen', to: 'pantry', priority: 1 },
      // Living area connections
      { from: 'lounge', to: 'dining', priority: 2 },
      { from: 'drawingRoom', to: 'dining', priority: 2 },
      { from: 'drawingRoom', to: 'lounge', priority: 2 },
      // Service connections
      { from: 'laundry', to: 'kitchen', priority: 3 },
      { from: 'storage', to: 'garage', priority: 3 },
      { from: 'study', to: 'library', priority: 3 },
      { from: 'walkInCloset', to: 'bedroom', priority: 1 }
    ];

    // Create doors based on adjacency and connection rules
    rooms.forEach((room1, i) => {
      rooms.slice(i + 1).forEach(room2 => {
        if (this.areRoomsAdjacent(room1, room2)) {
          const shouldConnect = connectionRules.some(rule => 
            (rule.from === room1.type && rule.to === room2.type) ||
            (rule.from === room2.type && rule.to === room1.type)
          );

          if (shouldConnect) {
            const door = this.createDoorBetweenRooms(room1, room2);
            if (door) {
              doors.push(door);
            }
          }
        }
      });
    });

    // Add doors to main circulation areas
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

  private createDoorBetweenRooms(room1: any, room2: any) {
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
      if (mainRooms.includes(room.type) && room.position.y < 30) { // Near entrance
        const door = {
          position: { 
            x: room.position.x + room.size.width / 2, 
            y: room.position.y 
          },
          orientation: 'horizontal' as const,
          width: 3
        };
        
        // Check if door doesn't conflict with existing doors
        const hasConflict = doors.some(existingDoor => 
          Math.abs(existingDoor.position.x - door.position.x) < 5 &&
          Math.abs(existingDoor.position.y - door.position.y) < 5
        );
        
        if (!hasConflict) {
          doors.push(door);
        }
      }
    });
  }
}