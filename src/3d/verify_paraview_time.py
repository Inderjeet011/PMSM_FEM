#!/usr/bin/env python3
"""
Verify XDMF file for ParaView time animation.
This script checks if the time series is properly structured.
"""

import h5py
import xml.etree.ElementTree as ET
from pathlib import Path

xdmf_path = Path(__file__).parents[2] / 'results' / '3d' / 'av_solver.xdmf'
h5_path = xdmf_path.with_suffix('.h5')

print("="*70)
print("ParaView Time Series Verification")
print("="*70)

# Check HDF5 file
print("\n[1] HDF5 File Check:")
with h5py.File(h5_path, 'r') as h5:
    if 'Function' in h5:
        funcs = list(h5['Function'].keys())
        print(f"   Functions: {funcs}")
        
        for func_name in funcs[:3]:  # Check first 3
            if func_name in h5['Function']:
                timesteps = sorted(h5['Function'][func_name].keys())
                print(f"\n   {func_name}:")
                print(f"     Time steps: {len(timesteps)}")
                print(f"     First: {timesteps[0]}")
                print(f"     Last: {timesteps[-1]}")

# Check XDMF structure
print("\n[2] XDMF Structure Check:")
tree = ET.parse(xdmf_path)
root = tree.getroot()

# Find temporal collections
collections = root.findall('.//Grid[@CollectionType="Temporal"]')
print(f"   Temporal collections: {len(collections)}")

for coll in collections:
    name = coll.get('Name', 'Unknown')
    grids = coll.findall('Grid')
    times = []
    for grid in grids:
        time_elem = grid.find('Time')
        if time_elem is not None:
            times.append(float(time_elem.get('Value', '0')))
    
    if len(times) > 0:
        print(f"\n   {name}:")
        print(f"     Time steps: {len(times)}")
        print(f"     Time range: {min(times):.6f} to {max(times):.6f} s")
        print(f"     Time step: {(max(times)-min(times))/(len(times)-1):.6f} s")

print("\n" + "="*70)
print("ParaView Instructions:")
print("="*70)
print("""
1. Open ParaView → File → Open → Select av_solver.xdmf
2. Click "Apply"
3. Enable Time Animation:
   - Look for time slider at the TOP of ParaView window
   - If not visible: View → Animation View (or Ctrl+Shift+A)
4. Select a field to visualize:
   - In "Coloring" dropdown, select "B_Magnitude" or "B_Magnitude_cell"
   - For volume rendering, use "_cell" versions
5. Animate:
   - Click Play button (▶) in Animation View
   - Or use time slider to step through manually
6. If still static:
   - Check that time slider shows 0-16.667 ms range
   - Try: View → Time Inspector to see available time steps
   - Make sure you're viewing the correct field (not a static one)
""")

