# ComfyUI Image Resizing   
*Precision dimension control nodes for aspect ratio adjustments and resizing workflows*

---

## Core Features

### 📐 Aspect Ratio Calculations
- `calculate_aspect_ratio_extension`   
  `►` Computes required pixel extension to match target proportions   
  `►` Outputs integer value for direct pipeline integration

### 🔢 Scaling Mathematics
- `calculate_upscale_factor`   
  `►` Determines optimal scaling ratio between source/target dimensions   
- `calculate_upscale_rounds`   
  `►` Calculates iteration count for multi-stage upscaling (requires IF node)

### 🖌️ Canvas Manipulation
- `extend_canvas_by_percentage`   
  `►` Expand canvas using % or absolute pixels (`use_pixels` toggle)   
- `image_crop_by_percentage`   
  `►` Precision cropping with %/pixel unit switching   
- `mask_crop_by_percentage`   
  `►` Synchronized mask cropping with identical controls

---

## Key Advantages
- **Pixel-Perfect Precision**   
  `▸` Mathematical dimension calculations   
  `▸` Rounding control for production pipelines   
    
- **Dual Measurement Modes**   
  `▸` Seamless percentage ↔ pixel conversion   
  `▸` Unit toggling without breaking workflows   
    
- **Non-Destructive Workflow**   
  `▸` All operations preserve source data   
  `▸` Chainable outputs for complex transforms   

---

## Typical Use Cases
