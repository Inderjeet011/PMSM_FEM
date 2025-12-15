# ParaView Visualization Guide for 3D PMSM Mesh

## Quick Start - View Motor Components Only

The Air domain (26,939 cells) covers the motor, making it hard to see. Use this method:

### Step-by-Step:

1. **Open the enhanced XDMF file:**
   - File → Open → Select `meshes/3d/pmesh3D_ipm_enhanced.xdmf`
   - Click "Apply"

2. **Filter out Air to see motor components:**
   - Filters → Threshold
   - In Threshold properties:
     - Scalar: Select **"MotorComponents"**
     - Lower threshold: **0.5**
     - Upper threshold: **1.0**
   - Click "Apply"

3. **Color by component:**
   - In the Coloring dropdown, select **"CellTags"**
   - Click "Edit Color Map" (gear icon)
   - Set "Interpolate" to **"Discrete"**
   - Click "Apply"

4. **You should now see:**
   - Rotor (tag 5) - Iron gray
   - Stator (tag 6) - Iron gray  
   - Permanent Magnets (tags 13-22) - Different colors
   - Copper Windings (tags 7-12) - Different colors
   - Air Gap (tags 2-3) - Light blue

## Alternative: View with Material Types

1. Open `pmesh3D_ipm_enhanced.xdmf`
2. Click "Apply"
3. In Coloring, select **"MaterialType"**
4. Set color map to **"Discrete"**
5. To see through Air:
   - Edit Color Map → Opacity
   - For MaterialType = 0 (Air), set opacity to **0.1**

## View Specific Components

### View Only Rotor:
- Add Threshold filter
- Scalar: **MaterialType**
- Lower: **3**, Upper: **3**
- Apply

### View Only Stator:
- Add Threshold filter  
- Scalar: **MaterialType**
- Lower: **4**, Upper: **4**
- Apply

### View Only Permanent Magnets:
- Add Threshold filter
- Scalar: **MaterialType**
- Lower: **6**, Upper: **6**
- Apply

### View Only Copper Windings:
- Add Threshold filter
- Scalar: **MaterialType**
- Lower: **5**, Upper: **5**
- Apply

## Material Type Mapping

- **0** = Air
- **1** = Air Gap
- **2** = Aluminum (Shaft)
- **3** = Rotor
- **4** = Stator
- **5** = Copper Windings
- **6** = Permanent Magnets

## Troubleshooting

**If you still can't see components:**
1. Make sure you're using `pmesh3D_ipm_enhanced.xdmf` (not the original)
2. Try rotating the view (middle mouse button)
3. Zoom in (scroll wheel)
4. Try "Representation" → "Surface" instead of "Surface with Edges"

**To see the full mesh including Air:**
- Don't use Threshold filter
- Use "MaterialType" for coloring
- Set Air opacity to 0.05-0.1

