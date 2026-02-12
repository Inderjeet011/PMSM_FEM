# Automatic parent–submesh coupling in DOLFINx with `entity_maps`

This document explains how to couple a parent mesh and a submesh using UFL `entity_maps` **without** manual quadrature or DOF mapping. Target: DOLFINx 0.7+, MPI parallel.

---

## 1. Minimal pattern (block form + entity_maps)

- **Parent mesh** `domain`: full geometry.
- **Submesh** `submesh_copper`: created with `create_submesh(domain, tdim, conductor_cells)`; same geometry as the conductor region on the parent.
- **Spaces**: e.g. `V` = Nédélec on `domain`, `V1` = Lagrange on `submesh_copper`.
- **Entity map**: `submesh_copper, subdomain_copper_to_domain = create_submesh(...)[:2]` → `entity_maps = [subdomain_copper_to_domain]`.
- **Measure for coupling**: use a measure on the **parent** restricted to the conductor (e.g. `dx = Measure("dx", domain=domain, subdomain_data=ct)` then `dx(vol_ids["copper"])` or `measure_over(dx, conductor_markers)`).
- **Block form** (bilinear and linear):

```python
# Coupling and V–V terms must use the PARENT measure over conductor cells
a01 = dt * ufl.inner(sigma * ufl.grad(u1), v) * dx(vol_ids["copper"])
a10 = ufl.inner(sigma * ufl.grad(v1), u) * dx(vol_ids["copper"])
a11 = dt * ufl.inner(sigma * ufl.grad(u1), ufl.grad(v1)) * dx(vol_ids["copper"])

a = fem.form([[a00, a01], [a10, a11]], entity_maps=entity_maps)
L = fem.form([L0, L1], entity_maps=entity_maps)
```

- **Assembly**: `A_mat = assemble_matrix(a, bcs=bc)`, `b = assemble_vector(L)` (then lifting and `set_bc`). Solution vector is monolithic; split by DOF offset for `uh` and `uh1`.

---

## 2. How trial/test on different meshes are coupled

- **Principle**: All integrals that involve both parent and submesh functions use a **single integration domain**: the conductor region on the **parent** mesh (e.g. `dx(copper)` with `subdomain_data=ct`).
- **Role of `entity_maps`**: The entity map is **submesh → parent** (submesh cell index → parent cell index). For each parent conductor cell, the assembler knows the corresponding submesh cell. So:
  - **A00**: only parent; integrate with `dx(whole)` or similar on parent.
  - **A01, A10, A11**: integrands involve V1 (submesh) and/or V (parent). The measure is **parent** `dx(copper)`. The assembler loops over parent conductor cells; for each such cell it gets the submesh cell via the **inverse** of the entity map and uses the V1 dofs on that submesh cell. Parent (V) dofs are taken from the same parent cell. So trial/test on different meshes are coupled by “same physical cell, two meshes” via the map.
- **No cross-mesh measure**: Do **not** use `dx(domain=submesh_copper)` for a01/a10/a11 when using `entity_maps`. Use the parent measure restricted to conductor (e.g. `dx(copper)` on `domain`). The submesh is only used to define V1; the integration domain for coupling is always the parent conductor region.

---

## 3. How quadrature points are mapped internally

- Submesh and parent conductor cells are the **same cells** (same geometry): submesh is a view over a subset of parent cells. So:
  - Quadrature points in **parent** conductor cell `K` are the same as in the **submesh** cell `K'` with `entity_map(K') = K`.
  - The assembler runs quadrature on the parent conductor cells. For each quadrature point it evaluates:
    - Parent basis (e.g. Nédélec) using the parent cell geometry.
    - Submesh basis (e.g. Lagrange) using the submesh cell that maps to that parent cell; geometry is the same, so the same physical point is used.
  - No explicit “pull-back” of points is needed: the entity map ensures that the right submesh cell and dofs are used for each parent cell. Coefficients (e.g. `sigma`) should be defined on the **parent** (e.g. DG0 on `domain`) and used in all conductor terms so that cell-wise values match.

---

## 4. Common pitfalls

| Pitfall | What goes wrong | Fix |
|--------|------------------|-----|
| **Wrong measure for coupling** | Using `dx(domain=submesh)` for a01/a10/a11 | Use parent measure restricted to conductor, e.g. `dx(copper)` on `domain`. |
| **Missing or wrong entity_maps** | Segfault or wrong coupling | Pass the **cell** entity map from `create_submesh` as `entity_maps=[subdomain_to_domain]`. |
| **Multiple submeshes** | One entity map per submesh | `entity_maps = [map1, map2, ...]` in the order expected by the form (see docs for your DOLFINx version). |
| **BCs and block layout** | `apply_lifting` / `set_bc` on wrong blocks | Use `bcs_by_block(extract_function_spaces(a), bc)` (and same for `L`) so BCs apply to the correct block. |
| **Solution vector layout** | Wrong split of monolithic `sol` | Split by `offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs`, then `uh` from `sol[:offset]`, `uh1` from `sol[offset:]` (with correct local/global handling). |
| **Ghost cells** | Missing ghosts on submesh or parent | Use the same `ghost_mode` as in the reference (e.g. `GhostMode.shared_facet`); ensure `create_submesh` and assembly run with consistent communicator. |
| **Preconditioner block form** | Using full block form in PC | Often use `a_p = form([[a00, None], [None, a11]], entity_maps=entity_maps)` and `P = assemble_matrix(a_p, bcs=bc)` so the PC ignores coupling. |

---

## 5. Assumptions under which this approach is safe

- **Submesh from parent**: The submesh is created with `create_submesh(parent, tdim, cells)` so that submesh cells are in 1–1 correspondence with the given parent cells. No refinement or different geometry.
- **Same geometry**: Submesh and parent conductor region have the same cell geometry (same vertices, same numbering per cell). So quadrature points coincide.
- **Single submesh**: One submesh (e.g. one conductor region). For multiple submeshes, the exact `entity_maps` API and order must be checked in the DOLFINx version you use.
- **Compatible DOLFINx**: `fem.form(..., entity_maps=...)` and `assemble_matrix` / `assemble_vector` supporting block forms with `entity_maps` (DOLFINx 0.7+).
- **MPI**: Same communicator for parent and submesh; assembly and solvers are collective where required.
- **Coefficients on parent**: For coupling and V–V terms, use coefficients (e.g. `sigma`) defined on the parent mesh so that cell-wise values are consistent with the parent conductor cells.

---

## 6. Reference code layout (summary)

- Load mesh and tags; get conductor cell indices; `submesh_copper, subdomain_copper_to_domain = create_submesh(domain, tdim, copper_cells)[:2]`.
- `entity_maps = [subdomain_copper_to_domain]`.
- Define V on `domain`, V1 on `submesh_copper`; trial/test u,v (V) and u1,v1 (V1).
- All conductor integrals use `dx(vol_ids["copper"])` (or equivalent) on the **parent**.
- `a = fem.form([[a00, a01], [a10, a11]], entity_maps=entity_maps)`, `L = fem.form([L0, L1], entity_maps=entity_maps)`.
- `A_mat = assemble_matrix(a, bcs=bc)`, assemble `b` from `L`, apply lifting and `set_bc`.
- Solve with monolithic `sol`; extract `uh` and `uh1` by DOF offset.
- For visualization of parent field on submesh: `u_n_submesh.interpolate(u_n, cells0=parent_cells, cells1=smsh_cells)` with `parent_cells = subdomain_copper_to_domain.sub_topology_to_topology(smsh_cells, inverse=False)` (submesh cell indices → parent cell indices; pass a numpy array of submesh cell indices).
