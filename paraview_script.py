#!/usr/bin/env python
"""
ParaView Python Script for Motor Visualization
Run this in ParaView's Python Console (View → Python Shell)
"""

# Clear any existing sources
from paraview.simple import *

# Clear the view
Disconnect()
Connect()

# ============================================================================
# STEP 1: Load the XDMF file
# ============================================================================
print("Loading XDMF file...")
xdmf_file = "/root/FEniCS/results/3d/av_solver.xdmf"
reader = XDMFReader(FileName=[xdmf_file])
reader.PointArrayStatus = ['A', 'B', 'B_Direction', 'B_Magnitude', 'B_vis', 'B_vis_mag', 'V']
reader.CellArrayStatus = ['B_dg', 'CellTags', 'MaterialID']

# Update to load data
UpdatePipeline()

# Get the first timestep data
reader.UpdatePipelineInformation()
timesteps = reader.TimestepValues
if timesteps:
    reader.TimestepValues = [timesteps[0]]  # Use first timestep

# ============================================================================
# STEP 2: Create view and show mesh with MaterialID coloring
# ============================================================================
print("Setting up view with MaterialID coloring...")

# Get active view or create one
view = GetActiveView()
if not view:
    view = CreateRenderView()
    view.ViewSize = [1200, 800]

# Show the reader with MaterialID coloring
meshDisplay = Show(reader, view)
meshDisplay.Representation = 'Surface'
meshDisplay.ColorArrayName = ['CELLS', 'MaterialID']
meshDisplay.LookupTable = GetColorTransferFunction('MaterialID')

# Set discrete color map for MaterialID
materialIDLUT = GetColorTransferFunction('MaterialID')
materialIDLUT.InterpretValuesAsCategories = 1
materialIDLUT.Annotations = ['1', 'Air', '2', 'AirGap', '3', 'Rotor', 
                             '4', 'Stator', '5', 'Coils', '6', 'Magnets', '7', 'Aluminium']
materialIDLUT.IndexedColors = [1.0, 0.0, 0.0,    # Red - Air
                               0.0, 1.0, 1.0,    # Cyan - AirGap
                               0.5, 0.5, 0.5,    # Gray - Rotor
                               0.8, 0.8, 0.8,    # Light Gray - Stator
                               1.0, 0.65, 0.0,   # Orange - Coils
                               0.0, 0.0, 1.0,    # Blue - Magnets
                               0.5, 0.0, 0.5]    # Purple - Aluminium

# Update view
view.ResetCamera()
Render()

print("✓ Mesh displayed with MaterialID coloring")

# ============================================================================
# STEP 3: Add Glyph filter to show B-field direction
# ============================================================================
print("Adding Glyph filter for B-field vectors...")

# Create Glyph filter
glyph = Glyph(Input=reader)
glyph.OrientationArray = ['POINTS', 'B']
glyph.ScaleArray = ['POINTS', 'B_Magnitude']
glyph.VectorScaleMode = 'Scale by Magnitude'
glyph.ScaleFactor = 0.1
glyph.GlyphType = 'Arrow'
glyph.GlyphMode = 'All Points'

# Reduce number of glyphs for better visualization
glyph.MaximumNumberOfSamplePoints = 2000

# Show glyphs
glyphDisplay = Show(glyph, view)
glyphDisplay.Representation = 'Surface'
glyphDisplay.ColorArrayName = ['POINTS', 'B_Magnitude']
glyphDisplay.LookupTable = GetColorTransferFunction('B_Magnitude')

# Make glyphs semi-transparent to see through them
glyphDisplay.Opacity = 0.7

# Update view
Render()

print("✓ Glyph filter added - arrows show B-field direction")

# ============================================================================
# STEP 4: Create a second view showing B-field magnitude
# ============================================================================
print("Creating second view for B-field magnitude...")

# Create second view
view2 = CreateRenderView()
view2.ViewSize = [1200, 800]

# Show mesh with B-field magnitude
meshDisplay2 = Show(reader, view2)
meshDisplay2.Representation = 'Surface'
meshDisplay2.ColorArrayName = ['POINTS', 'B_Magnitude']
meshDisplay2.LookupTable = GetColorTransferFunction('B_Magnitude')

# Set color map for B-field (blue to red)
bMagLUT = GetColorTransferFunction('B_Magnitude')
bMagLUT.ApplyPreset('Cool to Warm', True)

view2.ResetCamera()
Render(view2)

print("✓ Second view created showing B-field magnitude")

# ============================================================================
# STEP 5: Print summary
# ============================================================================
print("\n" + "="*70)
print("VISUALIZATION SETUP COMPLETE")
print("="*70)
print("\nAvailable Fields:")
print("  - MaterialID: Region identification (1=Air, 2=AirGap, 3=Rotor, etc.)")
print("  - B: Magnetic field vector (for Glyph orientation)")
print("  - B_Magnitude: Field strength")
print("  - B_Direction: Normalized direction vectors")
print("\nViews Created:")
print("  - View 1: MaterialID coloring + Glyph arrows")
print("  - View 2: B-field magnitude")
print("\nTo adjust Glyph arrows:")
print("  - Change Scale Factor: glyph.ScaleFactor = 0.2")
print("  - Change number of arrows: glyph.MaximumNumberOfSamplePoints = 5000")
print("="*70)

