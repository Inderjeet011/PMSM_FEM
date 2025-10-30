#!/bin/bash
# Launch ParaView with TEAM 30 results

cd /root/FEniCS/team_30

echo "=================================================================="
echo " Opening ParaView for TEAM 30 Visualization"
echo "=================================================================="
echo ""
echo "📁 Results location: TEAM30_0.0_three/"
echo "   - B.bp  (Magnetic flux density)"
echo "   - Az.bp (Magnetic vector potential)"
echo "   - V.bp  (Electric scalar potential)"
echo ""
echo "=================================================================="
echo " Instructions:"
echo "=================================================================="
echo "1. In ParaView: File → Open"
echo "2. Navigate to TEAM30_0.0_three/"
echo "3. Open the B.bp FOLDER (select the directory)"
echo "4. In Properties panel, click 'Apply'"
echo "5. Change 'Coloring' to show the magnetic field"
echo "6. Use Play button for animation through time"
echo "=================================================================="
echo ""

# Try to launch ParaView
if command -v paraview &> /dev/null; then
    echo "🚀 Launching ParaView..."
    paraview TEAM30_0.0_three/B.bp &
else
    echo "❌ ParaView not found. Install with: sudo apt install paraview"
fi

