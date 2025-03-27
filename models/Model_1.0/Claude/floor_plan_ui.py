import os
import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from floor_plan_generator import FloorPlanGenerator, PLOT_TYPES

class FloorPlanUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Floor Plan Generator")
        self.root.geometry("1200x800")
        
        # Load the generator model
        self.load_model()
        
        # Create UI components
        self.create_ui()
    
    def load_model(self):
        """Load the trained generator model."""
        self.generator = FloorPlanGenerator('dataset')
        
        try:
            self.generator.generator = tf.keras.models.load_model('generator_model.h5')
            print("Loaded trained generator model.")
        except:
            print("WARNING: Trained model not found. Please train the model first.")
            # We'll continue anyway for UI demo purposes
    
    def create_ui(self):
        """Create the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for controls
        left_panel = ttk.Frame(main_frame, padding="10", width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        # Right panel for displaying generated floor plan
        right_panel = ttk.Frame(main_frame, padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Controls
        ttk.Label(left_panel, text="Floor Plan Generator", font=("Arial", 16)).pack(pady=10)
        
        # Plot type selection
        ttk.Label(left_panel, text="Plot Type:").pack(anchor="w", pady=5)
        self.plot_type = tk.StringVar(value=PLOT_TYPES[0])
        plot_type_combo = ttk.Combobox(left_panel, textvariable=self.plot_type, values=PLOT_TYPES)
        plot_type_combo.pack(fill=tk.X, pady=5)
        
        # Regenerate button
        ttk.Button(left_panel, text="Generate Floor Plan", command=self.generate_floor_plan).pack(pady=10)
        
        # Add separator
        ttk.Separator(left_panel, orient="horizontal").pack(fill=tk.X, pady=10)
        
        # Room requirements
        ttk.Label(left_panel, text="Room Requirements:", font=("Arial", 12)).pack(anchor="w", pady=5)
        
        # Create room requirement options
        self.room_vars = {}
        rooms = ["Bedroom", "Bathroom", "Kitchen", "Dining", "Drawing Room", "Lounge", "Garage", "Store", "Lawn"]
        
        for room in rooms:
            frame = ttk.Frame(left_panel)
            frame.pack(fill=tk.X, pady=2)
            
            label = ttk.Label(frame, text=room, width=10)
            label.pack(side=tk.LEFT)
            
            var = tk.IntVar(value=1 if room in ["Bedroom", "Bathroom", "Kitchen"] else 0)
            self.room_vars[room] = var
            
            spinbox = ttk.Spinbox(frame, from_=0, to=5, width=5, textvariable=var)
            spinbox.pack(side=tk.RIGHT)
        
        # Add separator
        ttk.Separator(left_panel, orient="horizontal").pack(fill=tk.X, pady=10)
        
        # Export options
        ttk.Label(left_panel, text="Export Options:", font=("Arial", 12)).pack(anchor="w", pady=5)
        
        # Export format
        ttk.Label(left_panel, text="Format:").pack(anchor="w", pady=5)
        self.export_format = tk.StringVar(value="PNG")
        format_combo = ttk.Combobox(left_panel, textvariable=self.export_format, values=["PNG", "SVG", "DXF"])
        format_combo.pack(fill=tk.X, pady=5)
        
        # Export button
        ttk.Button(left_panel, text="Export Floor Plan", command=self.export_floor_plan).pack(pady=10)
        
        # Canvas for displaying the floor plan
        self.canvas_frame = ttk.Frame(right_panel)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Initially generate a floor plan
        self.generate_floor_plan()
    
    def generate_floor_plan(self):
        """Generate and display a floor plan."""
        plot_type = self.plot_type.get()
        
        try:
            # Generate floor plan
            floor_plan = self.generator.generate_floor_plan(plot_type)
            
            # Convert to PIL Image and display
            img = Image.fromarray(floor_plan)
            self.display_image(img)
            
            # Store the current floor plan
            self.current_floor_plan = floor_plan
            
        except Exception as e:
            print(f"Error generating floor plan: {str(e)}")
            # Display error message on canvas
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                text="Error generating floor plan.\nPlease ensure the model is trained.",
                fill="red",
                font=("Arial", 14)
            )
    
    def display_image(self, img):
        """Display an image on the canvas."""
        # Resize to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Ensure canvas has size
        if canvas_width < 10:
            canvas_width = 800
        if canvas_height < 10:
            canvas_height = 600
        
        # Resize image maintaining aspect ratio
        img_ratio = img.width / img.height
        canvas_ratio = canvas_width / canvas_height
        
        if img_ratio > canvas_ratio:
            # Image wider than canvas
            new_width = canvas_width
            new_height = int(canvas_width / img_ratio)
        else:
            # Image taller than canvas
            new_height = canvas_height
            new_width = int(canvas_height * img_ratio)
        
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(resized_img)
        
        # Display on canvas
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2, canvas_height // 2,
            image=self.photo, anchor=tk.CENTER
        )
    
    def export_floor_plan(self):
        """Export the floor plan to a file."""
        if not hasattr(self, 'current_floor_plan'):
            return
        
        # Get export format
        format_str = self.export_format.get().lower()
        
        # Ask for save location
        file_types = [
            ('PNG Image', '*.png'),
            ('SVG Image', '*.svg'),
            ('DXF File', '*.dxf')
        ]
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=f".{format_str}",
            filetypes=file_types,
            title="Save Floor Plan As"
        )
        
        if file_path:
            try:
                self.generator.export_floor_plan(
                    self.current_floor_plan, file_path, format=format_str
                )
                
                # Show confirmation
                tk.messagebox.showinfo(
                    "Export Successful",
                    f"Floor plan exported successfully to {file_path}"
                )
                
            except Exception as e:
                tk.messagebox.showerror(
                    "Export Error",
                    f"Error exporting floor plan: {str(e)}"
                )

if __name__ == "__main__":
    root = tk.Tk()
    app = FloorPlanUI(root)
    root.mainloop()