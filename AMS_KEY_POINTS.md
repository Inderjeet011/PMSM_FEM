# Key Points from Hypre AMS Documentation

## Critical Requirements:
1. **Discrete gradient G**: Must include ALL edges and vertices (no boundary conditions)
   - ✅ We're doing this correctly

2. **Edge constant vectors**: For lowest order, need representations of (1,0,0), (0,1,0), (0,0,1) in edge basis
   - ✅ We're interpolating constant vector fields
   - ⚠️ Need to verify the vectors are in the correct format

3. **Setup order is CRITICAL**:
   - Must call HYPRE_AMSSetDiscreteGradient BEFORE setup
   - Must call HYPRE_AMSSetEdgeConstantVectors BEFORE setup
   - Must NOT call setup before these are set
   - ✅ We're doing this correctly

4. **Matrix format**: AMS expects ParCSR format
   - ⚠️ Need to verify our matrices are in correct format

5. **No boundary conditions on G**: The discrete gradient should include boundary edges
   - ✅ We removed strong BCs on A, so this should be fine

## Potential Issues:
1. The edge constant vectors might need to be computed differently
2. The matrix might need to be converted to ParCSR format
3. Fieldsplit might be interfering with AMS setup

## Waiting for more info from user...
