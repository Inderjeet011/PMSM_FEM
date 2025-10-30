# ParaView Visualization Instructions for TEAM 30 Results

## Files Location
- Results are in: `TEAM30_0.0_three/`
- Az.bp - Magnetic vector potential
- B.bp - Magnetic flux density  
- V.bp - Electric scalar potential

## Method 1: Open BP files directly in ParaView

### Start ParaView:
```bash
cd /root/FEniCS/team_30
paraview &
```

### In ParaView GUI:
1. **File → Open**
2. Navigate to `TEAM30_0.0_three/B.bp/`
3. Select the `B.bp` folder (it's a directory)
4. Click **Apply** in the Properties panel
5. In the **Display** dropdown, select **Surface** or **Surface With Edges**
6. Click the **Play** button to animate through timesteps

### To visualize the magnetic field magnitude:
1. After loading B.bp
2. In the Properties panel, select **B** from the **Coloring** dropdown
3. Click **Apply**
4. Adjust the color map range in the **Color Map Editor**

## Method 2: Use converted VTK files (easier)

If BP files don't load properly, convert them first:
```bash
cd /root/FEniCS/team_30
python3 convert_to_vtk.py
```

Then open the `.vtu` files in ParaView.

## Tips:
- Use **Filters → Warp By Scalar** to create 3D height maps
- Use **Filters → Glyph** to show vector fields  
- Use **View → Animation View** to control timesteps
- Export animations via **File → Save Animation**

