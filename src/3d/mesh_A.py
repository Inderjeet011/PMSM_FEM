# Generate TEAM24 mesh with only coils and air (without rotor or stator)
import numpy as np
import gmsh
import meshio

# Physical tags — same scheme as mesh_3D.py / load_mesh.py (3D volumes + 2D facets)
PHYS_AIR = 1
PHYS_ROTOR = 5
PHYS_STATOR = 6
PHYS_COILS = (7, 8, 9, 10, 11, 12)  # six copper volumes
PHYS_MAGNETS = tuple(range(13, 23))  # ten PM volumes (13–22)
FACET_EXTERIOR = 1
FACET_MIDAIR = 2

def coil(r, dx, dy, dz):
    b1 = gmsh.model.occ.addBox(0.0, r, 0.0, 2 * r + dx, dy, dz)
    b2 = gmsh.model.occ.addBox(r, 0.0, 0.0, dx, 2 * r + dy, dz)
    c1 = gmsh.model.occ.addCylinder(r, r, 0.0, 0.0, 0.0, dz, r)
    c2 = gmsh.model.occ.addCylinder(r, r + dy, 0.0, 0.0, 0.0, dz, r)
    c3 = gmsh.model.occ.addCylinder(r + dx, r, 0.0, 0.0, 0.0, dz, r)
    c4 = gmsh.model.occ.addCylinder(r + dx, r + dy, 0.0, 0.0, 0.0, dz, r)
    cmb1 = gmsh.model.occ.fuse([(3, b1)], [(3, b2), (3, c1), (3, c2), (3, c3), (3, c4)])
    cmb2 = gmsh.model.occ.addBox(r, r, 0.0, dx, dy, dz)
    cmb3 = gmsh.model.occ.cut([cmb1[0][0]], [(3, cmb2)])

    return cmb3


def _add_permanent_magnets_2d(angle: float, center: int, r6: float, r7: float, z_plane: float) -> int:
    """
    Same IPM wedge as mesh_3D._add_permanent_magnets: annular sector [r6, r7],
    30° wide (dphi = pi/12), lying in plane z = z_plane.
    """
    dphi = np.pi / 12.0  # 15° half-width => 30° magnet
    p_i0 = gmsh.model.occ.addPoint(r6 * np.cos(angle - dphi), r6 * np.sin(angle - dphi), z_plane)
    p_i1 = gmsh.model.occ.addPoint(r6 * np.cos(angle + dphi), r6 * np.sin(angle + dphi), z_plane)
    p_o0 = gmsh.model.occ.addPoint(r7 * np.cos(angle - dphi), r7 * np.sin(angle - dphi), z_plane)
    p_o1 = gmsh.model.occ.addPoint(r7 * np.cos(angle + dphi), r7 * np.sin(angle + dphi), z_plane)
    arc_inner = gmsh.model.occ.addCircleArc(p_i0, center, p_i1)
    arc_outer = gmsh.model.occ.addCircleArc(p_o0, center, p_o1)
    side1 = gmsh.model.occ.addLine(p_i0, p_o0)
    side2 = gmsh.model.occ.addLine(p_i1, p_o1)
    loop = gmsh.model.occ.addCurveLoop([arc_inner, side2, arc_outer, side1])
    pm_surf = gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.synchronize()
    return pm_surf


def _extrude_2d_to_volume(dim2_tag: int, dz: float) -> tuple[int, int]:
    """Extrude a 2D surface along +Z; return the created 3D volume dimTag."""
    out = gmsh.model.occ.extrude([(2, dim2_tag)], 0.0, 0.0, dz)
    gmsh.model.occ.synchronize()
    for d, t in out:
        if d == 3:
            return (3, int(t))
    raise RuntimeError("extrude produced no 3D volume")


def _stator_coil_has_3d_overlap(stator_w: tuple[int, int], coil_w: tuple[int, int]) -> bool:
    """True if stator iron and coil share a positive 3D volume (not just surface contact)."""
    out, _ = gmsh.model.occ.intersect(
        [stator_w],
        [coil_w],
        removeObject=False,
        removeTool=False,
    )
    has_vol = any(d == 3 for d, t in out)
    if out:
        gmsh.model.occ.remove(out, recursive=True)
    gmsh.model.occ.synchronize()
    return has_vol


def _stator_touches_coils_no_iron_overlap(
    stator_w: tuple[int, int], coil_wrappers: list[tuple[int, int]]
) -> bool:
    """No 3D overlap between stator iron and any coil (touching on a face is OK)."""
    for w in coil_wrappers:
        if _stator_coil_has_3d_overlap(stator_w, w):
            return False
    return True


def _min_stator_coil_clearance(stator_w: tuple[int, int], coil_wrappers: list[tuple[int, int]]) -> float:
    d_min = float("inf")
    for w in coil_wrappers:
        d, *_ = gmsh.model.occ.getDistance(stator_w[0], stator_w[1], w[0], w[1])
        d_min = min(d_min, float(d))
    return d_min


def _add_stator_annulus_extruded(z_plane: float, r_inner: float, r_outer: float, height: float) -> tuple[int, int]:
    """
    Stator iron ring [r_inner, r_outer] in XY, same construction as mesh_3D stator_surf, extruded in +Z.
    """
    c_in = gmsh.model.occ.addCircle(0.0, 0.0, z_plane, r_inner)
    c_out = gmsh.model.occ.addCircle(0.0, 0.0, z_plane, r_outer)
    loop_in = gmsh.model.occ.addCurveLoop([c_in])
    loop_out = gmsh.model.occ.addCurveLoop([c_out])
    stator_surf = gmsh.model.occ.addPlaneSurface([loop_out, loop_in])
    gmsh.model.occ.synchronize()
    return _extrude_2d_to_volume(stator_surf, height)


# Remove rotor and stator parts
gmsh.initialize()
gmsh.model.add("coils_only")

# Coil parameters
H = 25.4
r_champ = 5.0

# Coils
# Keep the exact geometry you approved, but place 6 coils with uniform
# angular spacing so each neighboring coil is equidistant.
# Inner rectangle (dx×dy) unchanged. "Flatter" coil: reduce dz = extrusion depth of the profile
# (the other in-plane axis vs dy); this thins the coil in that direction without widening the hole.
coil_r = 6.0
coil_dx = 12.0
coil_dy = 48.0
coil_dz = 8.0  # local extrusion depth; lower = flatter (dx, dy unchanged; 2*r+dy = 60 mm axial)
coil_outer_h = 2.0 * coil_r + coil_dy  # 60 mm
D = 63 - coil_dz / 2.0

# --- Air gap: target 2–4 mm (nominal 3 mm); larger rotor; coil ring tuned with OCC distance ---
TARGET_AIR_GAP_MM = 3.0
AIR_GAP_MIN_MM = 2.0
AIR_GAP_MAX_MM = 4.0
rotor_radius = 40.0  # mm (was 28); scales PM radii below
# Initial coil COM ring; fine-tuned below with getDistance(rotor, coil) so gap ≈ TARGET_AIR_GAP_MM
ring_radius = rotor_radius + 22.0

coil_wrappers: list[tuple[int, int]] = []
for coil_id in range(6):
    angle = coil_id * 2.0 * np.pi / 6.0
    c = coil(coil_r, coil_dx, coil_dy, coil_dz)
    w = c[0][0]

    # Alternate orientation to preserve opposite winding directions.
    if coil_id % 2 == 0:
        gmsh.model.occ.rotate([w], 0, 0, 0, 1, 0, 0, np.pi / 2.0)
        gmsh.model.occ.translate([w], -41.0, -D, -27.0)
    else:
        gmsh.model.occ.rotate([w], 0, 0, 0, 1, 0, 0, -np.pi / 2.0)
        gmsh.model.occ.translate([w], -41.0, D, -27.0 + coil_outer_h)

    # Force uniform spacing: move each coil center to a common ring radius.
    com = gmsh.model.occ.getCenterOfMass(*w)
    tx = ring_radius * np.cos(angle)
    ty = ring_radius * np.sin(angle)
    gmsh.model.occ.translate([w], tx - com[0], ty - com[1], 0.0)

    # Make each coil face the center (radially inward) instead of all sharing
    # one global orientation.
    com2 = gmsh.model.occ.getCenterOfMass(*w)
    face_angle = angle + np.pi / 2.0
    gmsh.model.occ.rotate([w], com2[0], com2[1], com2[2], 0, 0, 1, face_angle)

    coil_wrappers.append(w)

# IPM rotor + embedded magnets (same idea as mesh_3D.py: PM annulus [r6,r7], cut from solid rotor)
z0 = -27.0
axial_h = coil_outer_h  # 60 mm
# mesh_3D (m): r3=0.042 rotor outer, r6=0.036, r7=0.038 — scale to mm ring matching our rotor_radius
r3_ref = 0.042
r6_ref = 0.036
r7_ref = 0.038
scale = rotor_radius / r3_ref
r6 = r6_ref * scale
r7 = r7_ref * scale

center_point = gmsh.model.occ.addPoint(0.0, 0.0, z0)
gmsh.model.occ.synchronize()

pm_count = 10
pm_spacing = 2.0 * np.pi / pm_count
pm_angles = [i * pm_spacing for i in range(pm_count)]

pm_2d_tags: list[int] = []
for ang in pm_angles:
    pm_2d_tags.append(_add_permanent_magnets_2d(ang, center_point, r6, r7, z0))

magnet_wrappers: list[tuple[int, int]] = []
for s2 in pm_2d_tags:
    magnet_wrappers.append(_extrude_2d_to_volume(s2, axial_h))

# Solid rotor cylinder, then subtract PM pockets so magnets sit inside rotor (IPM)
rotor_cyl_tag = gmsh.model.occ.addCylinder(0.0, 0.0, z0, 0.0, 0.0, axial_h, rotor_radius)
gmsh.model.occ.synchronize()
rotor_cut, _ = gmsh.model.occ.cut(
    [(3, rotor_cyl_tag)],
    magnet_wrappers,
    removeObject=True,
    removeTool=False,
)
gmsh.model.occ.synchronize()
rotor_vols = [w for w in rotor_cut if w[0] == 3]
if not rotor_vols:
    raise RuntimeError("Rotor cut produced no 3D volumes")

# Fuse iron fragments into one rotor volume for simpler tagging (optional but stable)
if len(rotor_vols) == 1:
    rotor_w = rotor_vols[0]
else:
    fused = gmsh.model.occ.fuse([rotor_vols[0]], rotor_vols[1:])
    rotor_w = fused[0][0]
    gmsh.model.occ.synchronize()

# Move all coils radially so the smallest rotor–coil clearance matches TARGET_AIR_GAP_MM (smooth, symmetric).
gmsh.model.occ.synchronize()
_air_final = TARGET_AIR_GAP_MM
for _it in range(30):
    gaps = []
    for w in coil_wrappers:
        d, *_ = gmsh.model.occ.getDistance(rotor_w[0], rotor_w[1], w[0], w[1])
        gaps.append(float(d))
    d_min = min(gaps)
    err = d_min - TARGET_AIR_GAP_MM
    if abs(err) < 0.06:
        _air_final = d_min
        break
    for w in coil_wrappers:
        com = gmsh.model.occ.getCenterOfMass(*w)
        r = float(np.hypot(com[0], com[1]))
        if r < 1e-9:
            continue
        gmsh.model.occ.translate([w], -err * com[0] / r, -err * com[1] / r, 0.0)
else:
    gaps = []
    for w in coil_wrappers:
        d, *_ = gmsh.model.occ.getDistance(rotor_w[0], rotor_w[1], w[0], w[1])
        gaps.append(float(d))
    _air_final = min(gaps)

gmsh.model.occ.synchronize()
if _air_final < 0.0:
    raise RuntimeError(
        f"Rotor and coils overlap (min clearance {_air_final:.3f} mm). Increase ring_radius (currently {ring_radius})."
    )
print(
    f"Air gap (rotor–coil, OCC min over 6 coils): {_air_final:.2f} mm "
    f"(target {TARGET_AIR_GAP_MM} mm, band [{AIR_GAP_MIN_MM}, {AIR_GAP_MAX_MM}] mm)"
)

# Stator annulus: inner radius = smallest r such that stator iron does **not** overlap any coil
# volume (touching on the cylindrical bore face is OK). AABB corners lie **outside** the coil
# solid, so bbox-based r_inner leaves a large gap; bisection on OCC volume intersection fixes that.
STATOR_RADIAL_THICKNESS_MM = 70.0 - 62.0  # mesh_3D (mm)
R_bbox_max = 0.0
for w in coil_wrappers:
    bb = gmsh.model.getBoundingBox(*w)
    for x in (bb[0], bb[3]):
        for y in (bb[1], bb[4]):
            R_bbox_max = max(R_bbox_max, float(np.hypot(x, y)))


def _stator_at_r_inner(r_inner: float) -> tuple[int, int]:
    return _add_stator_annulus_extruded(
        z0, r_inner, r_inner + STATOR_RADIAL_THICKNESS_MM, axial_h
    )


# Bracket: [r_lo, r_hi] with volume overlap at r_lo and none at r_hi (coil sits in bore at r_hi).
r_lo = rotor_radius + 0.5
r_hi = R_bbox_max
st_tmp = _stator_at_r_inner(r_lo)
if _stator_touches_coils_no_iron_overlap(st_tmp, coil_wrappers):
    gmsh.model.occ.remove([st_tmp], recursive=False)
    gmsh.model.occ.synchronize()
    raise RuntimeError(
        f"Stator sizing: r_lo={r_lo:.2f} mm has no iron–coil overlap; lower r_lo (rotor_radius?)."
    )
gmsh.model.occ.remove([st_tmp], recursive=False)
gmsh.model.occ.synchronize()

st_tmp = _stator_at_r_inner(r_hi)
if not _stator_touches_coils_no_iron_overlap(st_tmp, coil_wrappers):
    gmsh.model.occ.remove([st_tmp], recursive=False)
    gmsh.model.occ.synchronize()
    raise RuntimeError(
        f"Stator sizing: r_hi={r_hi:.2f} mm still overlaps coils; increase r_hi (bbox?)."
    )
gmsh.model.occ.remove([st_tmp], recursive=False)
gmsh.model.occ.synchronize()

stator_w: tuple[int, int] | None = None
for _ in range(56):
    if r_hi - r_lo < 1e-3:
        break
    mid = 0.5 * (r_lo + r_hi)
    if stator_w is not None:
        gmsh.model.occ.remove([stator_w], recursive=False)
        gmsh.model.occ.synchronize()
    stator_w = _stator_at_r_inner(mid)
    if _stator_touches_coils_no_iron_overlap(stator_w, coil_wrappers):
        r_hi = mid
    else:
        r_lo = mid

if stator_w is not None:
    gmsh.model.occ.remove([stator_w], recursive=False)
    gmsh.model.occ.synchronize()
# Bisection lands on the no-overlap limit (≈ touching). Extra bore clearance avoids sliver air
# regions / PLC failures in 3D Delaunay.
# Empirically: ε < ~0.07 mm → PLC Error on mesh.generate(3) for this geometry; 0.07 mm is the
# smallest ε that completed successfully here. The only *intentional* radial coil–stator gap is ε.
STATOR_MESH_EPSILON_MM = 0.07
stator_r4 = r_hi + STATOR_MESH_EPSILON_MM
stator_r5 = stator_r4 + STATOR_RADIAL_THICKNESS_MM
if stator_r4 <= rotor_radius + 0.5:
    raise RuntimeError(
        f"Stator inner r={stator_r4:.2f} mm is not outside rotor r={rotor_radius:.2f} mm; check coils."
    )
stator_tagged = _stator_at_r_inner(stator_r4)
gmsh.model.occ.synchronize()

_min_d = _min_stator_coil_clearance(stator_tagged, coil_wrappers)
print(
    f"Stator ring: r_inner={stator_r4:.4f} mm (touching limit r≈{r_hi:.4f} mm + {STATOR_MESH_EPSILON_MM} mm mesh ε), "
    f"bbox bound {R_bbox_max:.2f} mm, r_outer={stator_r5:.2f} mm — min OCC distance stator–coil: {_min_d:.4f} mm"
)

# Air region (box around coils)
M = 35.0
outer_box = gmsh.model.occ.addBox(-120, -120, -M, 240, 240, H + 2 * M)

# Build air explicitly as (outer_box - rotor - magnets - coils - stator), keep solids unchanged.
tool_wrappers = [rotor_w] + magnet_wrappers + coil_wrappers + [stator_tagged]
air_cut, _ = gmsh.model.occ.cut([(3, outer_box)], tool_wrappers, removeObject=True, removeTool=False)

gmsh.model.occ.synchronize()
air_parts = [w for w in air_cut if w[0] == 3]
if not air_parts:
    raise RuntimeError("Air cut returned no 3D air volume")

# --- 3D physical groups: fixed IDs (mesh_3D / load_mesh compatible) ---
all_air_vols = sorted({int(w[1]) for w in air_parts})

gmsh.model.addPhysicalGroup(3, all_air_vols, PHYS_AIR)
gmsh.model.setPhysicalName(3, PHYS_AIR, "Air")

gmsh.model.addPhysicalGroup(3, [rotor_w[1]], PHYS_ROTOR)
gmsh.model.setPhysicalName(3, PHYS_ROTOR, "Rotor")

for i, w in enumerate(coil_wrappers):
    tag = PHYS_COILS[i]
    gmsh.model.addPhysicalGroup(3, [w[1]], tag)
    gmsh.model.setPhysicalName(3, tag, f"Cu_{i + 1}")

for i, w in enumerate(magnet_wrappers):
    tag = PHYS_MAGNETS[i]
    gmsh.model.addPhysicalGroup(3, [w[1]], tag)
    gmsh.model.setPhysicalName(3, tag, f"PM_{i + 1}")

gmsh.model.addPhysicalGroup(3, [stator_tagged[1]], PHYS_STATOR)
gmsh.model.setPhysicalName(3, PHYS_STATOR, "Stator")

# --- 2D physical groups: boundary faces via OCC (not bbox scan), then Exterior / MidAir like mesh_3D ---
all_vol_dimtags: list[tuple[int, int]] = [(3, v) for v in all_air_vols]
all_vol_dimtags.append(rotor_w)
all_vol_dimtags.extend(coil_wrappers)
all_vol_dimtags.extend(magnet_wrappers)
all_vol_dimtags.append(stator_tagged)

bnd = gmsh.model.getBoundary(all_vol_dimtags, combined=False, oriented=False)
surf_tags = sorted({s[1] for s in bnd})

air_box_x_min = -120.0
air_box_y_min = -120.0
air_box_z_min = -M
air_box_x_max = air_box_x_min + 240.0
air_box_y_max = air_box_y_min + 240.0
air_box_z_max = air_box_z_min + (H + 2.0 * M)

# Cylindrical band between rotor OD and stator bore (mesh_3D-style MidAir split)
r_mid_air = 0.5 * (rotor_radius + stator_r4)
motor_z_start = z0
motor_z_end = z0 + axial_h
tol = 1e-3 * max(float(stator_r5), 1.0)
tol_mid = 0.02 * max(r_mid_air, 1.0)
eps_box = 1e-6 * max(120.0, abs(air_box_z_max), 1.0)

midair_faces: list[int] = []
exterior_faces: list[int] = []
for s_tag in surf_tags:
    x, y, z = gmsh.model.occ.getCenterOfMass(2, s_tag)
    r_surf = float(np.hypot(x, y))
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, s_tag)

    if abs(r_surf - r_mid_air) < tol_mid and (motor_z_start - tol) <= z <= (motor_z_end + tol):
        midair_faces.append(s_tag)

    on_x_min = abs(xmin - air_box_x_min) < eps_box
    on_x_max = abs(xmax - air_box_x_max) < eps_box
    on_y_min = abs(ymin - air_box_y_min) < eps_box
    on_y_max = abs(ymax - air_box_y_max) < eps_box
    on_z_min = abs(zmin - air_box_z_min) < eps_box
    on_z_max = abs(zmax - air_box_z_max) < eps_box
    if on_x_min or on_x_max or on_y_min or on_y_max or on_z_min or on_z_max:
        exterior_faces.append(s_tag)

if midair_faces:
    gmsh.model.addPhysicalGroup(2, midair_faces, FACET_MIDAIR)
    gmsh.model.setPhysicalName(2, FACET_MIDAIR, "MidAir")
if exterior_faces:
    gmsh.model.addPhysicalGroup(2, exterior_faces, FACET_EXTERIOR)
    gmsh.model.setPhysicalName(2, FACET_EXTERIOR, "Exterior")

gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 5)

#%%
gmsh.model.mesh.generate(3)
gmsh.write("coils.msh")

gmsh.finalize()

# Robust conversion to XDMF/H5 (avoids dolfinx read_from_msh MPI rank crashes).
msh = meshio.read("coils.msh")
pts = msh.points

tet_blocks = []
tri_blocks = []
phys_data = msh.cell_data_dict.get("gmsh:physical", {})

for cblock in msh.cells:
    ctype = cblock.type
    if ctype == "tetra":
        tet_blocks.append(cblock.data)
    elif ctype == "triangle":
        tri_blocks.append(cblock.data)

if not tet_blocks:
    raise RuntimeError("No tetra cells found in coils.msh")

tet_cells = np.vstack(tet_blocks)
tet_tags = np.asarray(phys_data.get("tetra", np.zeros(tet_cells.shape[0], dtype=np.int32)), dtype=np.int32)

vol_mesh = meshio.Mesh(
    points=pts,
    cells=[("tetra", tet_cells)],
    cell_data={"Cell_markers": [tet_tags]},
)
meshio.write("coils.xdmf", vol_mesh)

if tri_blocks:
    tri_cells = np.vstack(tri_blocks)
    tri_tags = np.asarray(phys_data.get("triangle", np.zeros(tri_cells.shape[0], dtype=np.int32)), dtype=np.int32)
    facet_mesh = meshio.Mesh(
        points=pts,
        cells=[("triangle", tri_cells)],
        cell_data={"Facet_markers": [tri_tags]},
    )
    meshio.write("coils_facets.xdmf", facet_mesh)