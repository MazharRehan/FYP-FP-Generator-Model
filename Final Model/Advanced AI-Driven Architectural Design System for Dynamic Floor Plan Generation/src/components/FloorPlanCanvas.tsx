import React, { useRef, useEffect } from 'react';
import { GeneratedFloorPlan } from '../types/floorplan';

interface FloorPlanCanvasProps {
  plan: GeneratedFloorPlan;
}

const FloorPlanCanvas: React.FC<FloorPlanCanvasProps> = ({ plan }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size based on plot image specifications
    const { width: imgWidth, height: imgHeight } = plan.plotSize.imageSpecs;
    canvas.width = imgWidth;
    canvas.height = imgHeight;

    // Clear canvas with white background
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, imgWidth, imgHeight);

    // Calculate scale factor
    const scaleX = imgWidth / plan.plotSize.dimensions.width;
    const scaleY = imgHeight / plan.plotSize.dimensions.height;

    // Draw outer boundary (plot boundary) with thick black border
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 8;
    ctx.strokeRect(0, 0, imgWidth, imgHeight);

    // Draw rooms with exact colors and proper styling
    plan.rooms.forEach(room => {
      const x = room.position.x * scaleX;
      const y = room.position.y * scaleY;
      const width = room.size.width * scaleX;
      const height = room.size.height * scaleY;

      // Fill room with specified color
      ctx.fillStyle = room.color;
      ctx.fillRect(x, y, width, height);

      // Draw room borders (internal walls) with proper thickness
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, width, height);
    });

    // Draw internal walls between rooms with proper styling
    plan.walls.forEach(wall => {
      if (wall.internal) {
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 5;
        ctx.beginPath();
        ctx.moveTo(wall.start.x * scaleX, wall.start.y * scaleY);
        ctx.lineTo(wall.end.x * scaleX, wall.end.y * scaleY);
        ctx.stroke();
      }
    });

    // Draw doors with enhanced styling like reference images
    plan.doors.forEach(door => {
      const x = door.position.x * scaleX;
      const y = door.position.y * scaleY;
      
      ctx.fillStyle = '#FFFFFF'; // White door opening
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 3;
      
      if (door.orientation === 'horizontal') {
        const doorWidth = (door.width || 3) * scaleX;
        const doorHeight = 12; // Increased door opening height
        
        // Create door opening (white gap in wall)
        ctx.fillRect(x - doorWidth/2, y - doorHeight/2, doorWidth, doorHeight);
        
        // Draw door swing arc (like in reference images)
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x - doorWidth/2, y, doorWidth * 0.8, 0, Math.PI/2);
        ctx.stroke();
        
        // Draw door frame with white fill
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(x - doorWidth/2, y - doorHeight/2, doorWidth, doorHeight);
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 3;
        ctx.strokeRect(x - doorWidth/2, y - doorHeight/2, doorWidth, doorHeight);
        
      } else {
        const doorWidth = 12; // Increased door opening width
        const doorHeight = (door.width || 3) * scaleY;
        
        // Create door opening (white gap in wall)
        ctx.fillRect(x - doorWidth/2, y - doorHeight/2, doorWidth, doorHeight);
        
        // Draw door swing arc
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y - doorHeight/2, doorHeight * 0.8, 0, Math.PI/2);
        ctx.stroke();
        
        // Draw door frame with white fill
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(x - doorWidth/2, y - doorHeight/2, doorWidth, doorHeight);
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 3;
        ctx.strokeRect(x - doorWidth/2, y - doorHeight/2, doorWidth, doorHeight);
      }
    });

    // Draw room labels with better styling and positioning
    plan.rooms.forEach(room => {
      const x = room.position.x * scaleX;
      const y = room.position.y * scaleY;
      const width = room.size.width * scaleX;
      const height = room.size.height * scaleY;

      // Only draw labels for rooms large enough
      if (width > 60 && height > 40) {
        const fontSize = Math.min(width / 6, height / 3, 18);
        ctx.font = `bold ${fontSize}px Arial, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        const textX = x + width / 2;
        const textY = y + height / 2;
        
        // Add semi-transparent white background for better readability
        const textMetrics = ctx.measureText(room.label);
        const padding = 8;
        ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
        ctx.fillRect(
          textX - textMetrics.width / 2 - padding,
          textY - fontSize / 2 - padding/2,
          textMetrics.width + padding * 2,
          fontSize + padding
        );
        
        // Add border to text background
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.4)';
        ctx.lineWidth = 1;
        ctx.strokeRect(
          textX - textMetrics.width / 2 - padding,
          textY - fontSize / 2 - padding/2,
          textMetrics.width + padding * 2,
          fontSize + padding
        );
        
        // Draw text with better contrast
        ctx.fillStyle = '#000000';
        ctx.fillText(room.label, textX, textY);
      }
    });

    // Draw dimensions and annotations with professional styling
    ctx.fillStyle = '#000000';
    ctx.font = 'bold 18px Arial, sans-serif';
    ctx.textAlign = 'center';
    
    // Plot size label at bottom with enhanced background
    const plotLabel = `${plan.plotSize.name} - ${plan.plotSize.dimensions.width}' × ${plan.plotSize.dimensions.height}'`;
    const labelY = imgHeight - 20;
    const labelMetrics = ctx.measureText(plotLabel);
    
    // Enhanced background for label
    ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
    ctx.fillRect(
      imgWidth / 2 - labelMetrics.width / 2 - 12,
      labelY - 16,
      labelMetrics.width + 24,
      28
    );
    
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 2;
    ctx.strokeRect(
      imgWidth / 2 - labelMetrics.width / 2 - 12,
      labelY - 16,
      labelMetrics.width + 24,
      28
    );
    
    ctx.fillStyle = '#000000';
    ctx.fillText(plotLabel, imgWidth / 2, labelY);

    // Add enhanced scale indicator
    const scaleLength = 60; // 5 feet in pixels
    const scaleX_pos = 25;
    const scaleY_pos = imgHeight - 50;
    
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(scaleX_pos, scaleY_pos);
    ctx.lineTo(scaleX_pos + scaleLength, scaleY_pos);
    ctx.stroke();
    
    // Scale markers with enhanced styling
    ctx.beginPath();
    ctx.moveTo(scaleX_pos, scaleY_pos - 8);
    ctx.lineTo(scaleX_pos, scaleY_pos + 8);
    ctx.moveTo(scaleX_pos + scaleLength, scaleY_pos - 8);
    ctx.lineTo(scaleX_pos + scaleLength, scaleY_pos + 8);
    ctx.stroke();
    
    ctx.font = 'bold 14px Arial, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('5 ft', scaleX_pos + scaleLength/2, scaleY_pos + 25);

    // Add generation timestamp
    const timestamp = plan.createdAt.toLocaleString();
    ctx.font = '12px Arial, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    ctx.fillText(`Generated: ${timestamp}`, imgWidth - 10, 20);

  }, [plan]);

  const downloadImage = (format: 'png' | 'svg' | 'dxf' = 'png') => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const fileName = `${plan.plotSize.name.replace(' ', '')}_GF_FP_${String(Math.floor(Math.random() * 1000)).padStart(3, '0')}_V${String(plan.id.slice(-2)).padStart(2, '0')}`;
    
    if (format === 'png') {
      const link = document.createElement('a');
      link.download = `${fileName}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
    } else if (format === 'svg') {
      const svgContent = generateSVG(plan);
      const blob = new Blob([svgContent], { type: 'image/svg+xml' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.download = `${fileName}.svg`;
      link.href = url;
      link.click();
      URL.revokeObjectURL(url);
    } else if (format === 'dxf') {
      const dxfContent = generateDXF(plan);
      const blob = new Blob([dxfContent], { type: 'application/dxf' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.download = `${fileName}.dxf`;
      link.href = url;
      link.click();
      URL.revokeObjectURL(url);
    }
  };

  const generateSVG = (plan: GeneratedFloorPlan): string => {
    const { width: imgWidth, height: imgHeight } = plan.plotSize.imageSpecs;
    const scaleX = imgWidth / plan.plotSize.dimensions.width;
    const scaleY = imgHeight / plan.plotSize.dimensions.height;

    let svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg width="${imgWidth}" height="${imgHeight}" xmlns="http://www.w3.org/2000/svg">
  <rect width="${imgWidth}" height="${imgHeight}" fill="white" stroke="black" stroke-width="8"/>
`;

    // Add rooms
    plan.rooms.forEach(room => {
      const x = room.position.x * scaleX;
      const y = room.position.y * scaleY;
      const width = room.size.width * scaleX;
      const height = room.size.height * scaleY;

      svg += `  <rect x="${x}" y="${y}" width="${width}" height="${height}" fill="${room.color}" stroke="black" stroke-width="4"/>
`;
      
      // Add room label
      if (width > 60 && height > 40) {
        const textX = x + width / 2;
        const textY = y + height / 2;
        svg += `  <text x="${textX}" y="${textY}" text-anchor="middle" dominant-baseline="middle" font-family="Arial" font-weight="bold" font-size="16" fill="black">${room.label}</text>
`;
      }
    });

    // Add doors
    plan.doors.forEach(door => {
      const x = door.position.x * scaleX;
      const y = door.position.y * scaleY;
      
      if (door.orientation === 'horizontal') {
        const doorWidth = (door.width || 3) * scaleX;
        svg += `  <rect x="${x - doorWidth/2}" y="${y - 6}" width="${doorWidth}" height="12" fill="white" stroke="black" stroke-width="3"/>
`;
      } else {
        const doorHeight = (door.width || 3) * scaleY;
        svg += `  <rect x="${x - 6}" y="${y - doorHeight/2}" width="12" height="${doorHeight}" fill="white" stroke="black" stroke-width="3"/>
`;
      }
    });

    svg += '</svg>';
    return svg;
  };

  const generateDXF = (plan: GeneratedFloorPlan): string => {
    let dxf = `0
SECTION
2
HEADER
9
$ACADVER
1
AC1015
0
ENDSEC
0
SECTION
2
ENTITIES
`;

    // Add rooms as polylines
    plan.rooms.forEach((room, index) => {
      const x = room.position.x;
      const y = room.position.y;
      const width = room.size.width;
      const height = room.size.height;

      dxf += `0
LWPOLYLINE
5
${(100 + index).toString(16)}
100
AcDbEntity
8
ROOMS
100
AcDbPolyline
90
4
70
1
10
${x}
20
${y}
10
${x + width}
20
${y}
10
${x + width}
20
${y + height}
10
${x}
20
${y + height}
`;
    });

    // Add doors as lines
    plan.doors.forEach((door, index) => {
      const x = door.position.x;
      const y = door.position.y;
      
      dxf += `0
LINE
5
${(200 + index).toString(16)}
100
AcDbEntity
8
DOORS
100
AcDbLine
10
${x}
20
${y}
11
${x}
21
${y}
`;
    });

    dxf += `0
ENDSEC
0
EOF`;

    return dxf;
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-center items-center bg-white rounded-lg p-4 overflow-auto">
        <canvas
          ref={canvasRef}
          className="border border-gray-300 shadow-lg max-w-full"
          style={{ 
            maxWidth: '100%',
            height: 'auto',
            imageRendering: 'crisp-edges'
          }}
        />
      </div>
      
      <div className="flex justify-center space-x-2">
        <button
          onClick={() => downloadImage('png')}
          className="px-4 py-2 bg-cyan-500 hover:bg-cyan-600 text-white rounded-lg font-medium transition-colors flex items-center space-x-2"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <span>PNG</span>
        </button>
        
        <button
          onClick={() => downloadImage('svg')}
          className="px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg font-medium transition-colors flex items-center space-x-2"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
          </svg>
          <span>SVG</span>
        </button>
        
        <button
          onClick={() => downloadImage('dxf')}
          className="px-4 py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg font-medium transition-colors flex items-center space-x-2"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <span>DXF</span>
        </button>
      </div>
    </div>
  );
};

export default FloorPlanCanvas;