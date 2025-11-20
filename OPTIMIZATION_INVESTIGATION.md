# Investigation: "ABNORMAL" Optimization Termination

## Problem Summary

When running `python -m clay examples/architecture.clay -s`, the stats output shows:
```json
{
  "success": false,
  "iterations": 3,
  "message": "ABNORMAL: "
}
```

## Root Causes

### 1. Message Truncation
The message `"ABNORMAL: "` is incomplete. Scipy's L-BFGS-B optimizer returns messages as strings, but this one appears truncated. The full message is likely:
- `"ABNORMAL_TERMINATION_IN_LNSRCH"`

This indicates the **line search algorithm failed** to find an acceptable step after multiple evaluations.

### 2. Optimization Failure
The optimizer only completed **3 iterations** (vs typical 30-70+ for successful runs) with **840 function evaluations** before failing.

The final energy breakdown shows:
- Edge length penalty: 1,516,240 (very high - nodes are far apart)
- Straightness penalty: 175,629 (high)
- Area penalty: 72,274
- **Total: 1,744,200** (compare to ~30,000 for successful simple.clay)

### 3. Why Line Search Fails

According to scipy documentation and Stack Overflow investigations:

**Primary causes:**
1. **Error in function or gradient evaluation** - Function returns NaN/Inf
2. **Rounding errors dominate computation** - Numerical precision issues
3. **Variable scaling problems** - Variables with very different magnitudes
4. **Initial position issues** - Starting point causes numerical problems

**In our case:**
- We use numerical gradients (scipy auto-computes), so gradient bugs unlikely
- Energy values are VERY large (~1.7M) compared to coordinate scales (0-800)
- The initial grid layout may put nodes in difficult configurations
- Complex graphs (7+ nodes, multiple edges) more prone to failure

## Analysis of Examples

| Example | Nodes | Edges | Success | Iterations | Final Energy | Message |
|---------|-------|-------|---------|------------|--------------|---------|
| simple.clay | 3 | 2 | ✓ | 49 | 30,207 | CONVERGENCE |
| architecture.clay | 7 | 7 | ✗ | 3 | 1,744,200 | ABNORMAL: |
| workflow.clay | 8 | 11 | ✗ | 87 | 2,258,182 | ABNORMAL: |

The workflow example managed 87 iterations before failing, suggesting it was "trying harder" but still couldn't converge.

## Solutions

### Immediate Fixes (Low hanging fruit)

1. **Improve optimizer options**
   - Increase `maxls` (max line search steps) from default 20 to 50+
   - Adjust `ftol` (function tolerance) to be less strict
   - Add `gtol` (gradient tolerance) settings
   - Increase `maxiter` beyond 2000 if needed

2. **Better initial positions**
   - Current: Simple grid layout
   - Improvement: Use spring/force-directed layout for initial guess
   - Alternative: Random with jitter to avoid symmetry issues

3. **Energy function scaling**
   - Normalize energy terms to similar magnitudes
   - Current weights cause huge energy values (1M+)
   - Consider scaling coordinates or normalizing penalties

### Medium-term Improvements

4. **Diagnostic output**
   - Add warning messages when optimizer fails
   - Print energy evolution during optimization (via callback)
   - Show which penalty dominates at failure

5. **Graceful degradation**
   - Accept partial solutions even when optimizer fails
   - Current: We render the failed result anyway
   - Improvement: Warn user but don't treat as fatal error

6. **Alternative optimizers**
   - Try `method='trust-constr'` for difficult cases
   - Try `method='SLSQP'` (Sequential Least Squares)
   - Implement multi-start: try multiple random initial positions

### Long-term Architecture Changes

7. **Two-phase optimization**
   - Phase 1: Rough layout with relaxed constraints
   - Phase 2: Refinement with full constraints

8. **Hierarchical approach**
   - Cluster nodes and optimize clusters first
   - Then optimize within clusters

9. **Constraint reformulation**
   - Replace penalty-based constraints with true constraints
   - Use constrained optimization methods properly

## Recommended Implementation Order

1. **Quick fix (15 min)**: Update `layout_graph()` options
   ```python
   options={
       'maxiter': 2000,
       'maxls': 50,      # NEW: more line search attempts
       'ftol': 1e-5,     # CHANGED: less strict (was 1e-6)
       'gtol': 1e-4      # NEW: gradient tolerance
   }
   ```

2. **Better messaging (10 min)**: Detect truncated messages and expand
   ```python
   if result.message == "ABNORMAL: ":
       result.message = "ABNORMAL_TERMINATION_IN_LNSRCH: Line search failed (see OPTIMIZATION_INVESTIGATION.md)"
   ```

3. **Warnings (15 min)**: Print helpful diagnostics on failure
   ```python
   if not result.success:
       print(f"⚠️  Warning: Optimization did not converge ({result.message})")
       print(f"   Generated layout may be suboptimal")
   ```

4. **Testing (30 min)**: Re-run all examples and verify improvements

## References

- [Stack Overflow: ABNORMAL_TERMINATION_IN_LNSRCH](https://stackoverflow.com/questions/66036929/)
- [Scipy L-BFGS-B docs](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
- [L-BFGS-B Fortran source](https://github.com/scipy/scipy/blob/main/scipy/optimize/lbfgsb_src/)

## Current Status

✓ Investigation complete
✓ Root cause identified
⧗ Solutions designed
☐ Implementation pending
☐ Testing pending
