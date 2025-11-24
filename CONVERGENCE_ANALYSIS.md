# Convergence Analysis Report

**Date**: 2025-11-24
**Branch**: optimization-monitoring
**Tool**: monitor_layout.py with analyze_convergence.py

## Executive Summary

Analysis of all 9 example `.clay` files reveals **significant convergence issues** affecting the reliability and reproducibility of the layout engine:

- **0 out of 9 examples** show consistent convergence behavior across multiple runs
- **All 9 examples** exhibit non-deterministic behavior (varying between "converged" and "ABNORMAL termination")
- **5 examples** have high average energy (> 1.0), indicating suboptimal layouts
- **5 examples** show high energy variance (> 0.5), indicating initialization sensitivity

## Detailed Results

### Per-Example Analysis

| Example | Nodes | Edges | Converged/3 | Terminated/3 | Avg Energy | Energy Variance | Avg Iterations |
|---------|-------|-------|-------------|--------------|------------|-----------------|----------------|
| architecture | 7 | 7 | 2 | 1 | 5.37 | 6.59 | 6.3 |
| chain | 4 | 3 | 2 | 1 | 0.17 | 0.05 | 10.0 |
| cycle | 3 | 3 | 1 | 2 | 0.16 | 0.19 | 8.7 |
| diamond | 5 | 5 | 2 | 1 | 2.49 | 6.45 | 15.0 |
| **flowchart** | 7 | 7 | 1 | 2 | **268.38** | **803.50** | 13.7 |
| simple | 3 | 2 | 1 | 2 | 0.18 | 0.17 | 9.3 |
| state_machine | 6 | 8 | 2 | 1 | 2.00 | 1.37 | 10.3 |
| workflow | 8 | 8 | 1 | 2 | 9.65 | 9.89 | 19.7 |
| yshape | 4 | 3 | 2 | 1 | 0.29 | 0.34 | 10.0 |

**Most problematic**: `flowchart.clay` shows extreme instability with energy variance of 803.50 and average energy of 268.38 (vs typical 0.1-10 range).

### Convergence Status Distribution

**None of the examples consistently converge or terminate.** Each example alternates randomly between:
- ✅ **"CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH"** (success=True)
- ⚠️ **"ABNORMAL: "** (success=False)

### High Energy Examples (> 1.0)

These examples consistently find layouts with high residual energy, indicating poor layout quality:

1. **flowchart**: 268.38 (EXTREME)
2. **workflow**: 9.65
3. **architecture**: 5.37
4. **diamond**: 2.49
5. **state_machine**: 2.00

### High Variance Examples (> 0.5)

These examples show the most sensitivity to initialization:

1. **flowchart**: 803.50 (EXTREME)
2. **workflow**: 9.89
3. **architecture**: 6.59
4. **diamond**: 6.45
5. **state_machine**: 1.37

## Root Cause Analysis

### 1. **ABNORMAL Termination (Primary Issue)**

**Observation**: scipy's L-BFGS-B optimizer frequently returns `success=False` with message `"ABNORMAL: "`.

**Likely Causes**:
- **Numerical instability**: The overlap penalty function returns extremely large values (~65 billion) when nodes overlap significantly
- **Line search failure**: L-BFGS-B's line search cannot find a descent direction due to the extreme energy landscape
- **Roundoff errors**: The combination of penalties with vastly different magnitudes (overlap: 1000×, area: 1×) causes numerical precision issues

**Evidence**:
```python
# All nodes at same position
Energy at (0,0): 64,984,703,684.17

# Even "successful" terminations from ABNORMAL state show low penalties:
Seed 789: ABNORMAL, energy=0.85
  Penalties: overlap=0.00, edge_len=0.07, straight=0.01
```

The optimizer terminates abnormally but actually finds good layouts (no overlap, low energy).

### 2. **Non-Determinism (Secondary Issue)**

**Observation**: Without setting a random seed, results vary dramatically between runs.

**Cause**:
- The `simple_spring_layout()` initialization uses random starting positions when `seed=None`
- Different initial positions lead to different local minima
- Some initial positions cause ABNORMAL termination, others converge normally

**Impact**:
- Same `.clay` file produces different results on each run
- Energy can vary by orders of magnitude (e.g., flowchart: 0.54 to 804.04)
- Users cannot reproduce layouts

### 3. **Energy Function Scale Issues**

**Problem**: The overlap penalty dominates other penalties by several orders of magnitude:

```python
# Penalty weights in energy_function():
W_OVERLAP = 1000       # Hard constraint
W_EDGE_LENGTH = 10     # Medium
W_STRAIGHTNESS = 5     # Medium
W_EDGE_NODE = 200      # High
W_BBOX = 100           # High
W_AREA = 1             # Low
```

**Consequence**:
- When nodes overlap, energy explodes to billions
- Gradient becomes nearly singular
- L-BFGS-B struggles with the ill-conditioned Hessian approximation

### 4. **Interpretation of "Terminated"**

**Important Discovery**: scipy's `success` flag doesn't always mean failure.

Testing shows:
- `ABNORMAL` termination often produces **excellent** layouts (overlap=0, low energy)
- The optimizer may be stopping early because it cannot improve further, not because it failed
- This is actually a **false alarm** - the layouts are valid

**Hypothesis**: The discrete nature of non-overlapping constraints creates a discontinuous energy landscape that L-BFGS-B (designed for smooth functions) struggles with, even though it finds valid solutions.

## Impact Assessment

### Severity: **HIGH**

1. **User Experience**:
   - ⚠️ Users see "ABNORMAL termination" warnings even for good layouts
   - ⚠️ Non-reproducible results without explicit seeds
   - ⚠️ Confusing mix of "converged" and "terminated" messages

2. **Layout Quality**:
   - ✅ Most layouts are actually good (overlap=0, reasonable energy)
   - ⚠️ `flowchart.clay` sometimes produces very poor layouts (energy > 800)
   - ⚠️ High variance means unstable/unpredictable output quality

3. **Reliability**:
   - ❌ 0% consistent convergence across examples
   - ❌ Cannot guarantee reproducibility without seeds
   - ⚠️ May find poor local minima depending on initialization

## Recommendations

### Immediate (High Priority)

1. **Add default random seed**:
   ```python
   # In layout_graph(), change default:
   seed: Optional[int] = None  # Current
   seed: Optional[int] = 42     # Recommended
   ```
   This ensures reproducible results by default.

2. **Suppress misleading warnings**:
   - scipy's `success=False` doesn't mean layout failed
   - Check `overlap_penalty == 0` to determine actual success
   - Redefine "converged" based on layout quality, not optimizer status

3. **Investigate flowchart.clay**:
   - This example shows extreme instability
   - May need better initialization or different penalty weights
   - Consider multi-start optimization for complex graphs

### Medium Priority

4. **Improve energy function conditioning**:
   - Consider logarithmic scaling for overlap penalty
   - Better normalization across penalty types
   - Smoother transitions in penalty functions

5. **Add convergence quality metrics**:
   - Report overlap violations explicitly
   - Show penalty breakdown in summary
   - Define "acceptable layout" criteria independently of optimizer status

6. **Multi-start optimization**:
   - Try 3-5 different seeds automatically
   - Pick best result by energy and overlap
   - Improves robustness for complex graphs

### Long-term

7. **Alternative optimizers**:
   - Try SLSQP (handles constraints better)
   - Try trust-region methods (better for ill-conditioned problems)
   - Consider specialized graph layout algorithms for initialization

8. **Better initialization**:
   - Smarter spring layout with edge-weight awareness
   - Topological sort for DAGs
   - Hierarchical clustering for disconnected components

## Conclusions

The monitoring feature successfully revealed critical issues:

1. **All examples show non-deterministic behavior** due to random initialization
2. **Optimizer "ABNORMAL" termination is common but misleading** - layouts are often valid
3. **Energy function scale issues** cause numerical instability
4. **Flowchart example is severely problematic** with 800+ variance

**The good news**: Most layouts are actually visually acceptable despite "ABNORMAL" status. The issue is primarily about optimizer convergence criteria, not layout quality.

**The action item**: Add default seed for reproducibility and redefine "success" based on layout validity (no overlaps) rather than optimizer status.

## Test Commands

To reproduce this analysis:

```bash
# Run convergence analysis
python analyze_convergence.py

# Test specific example with monitoring
python monitor_layout.py examples/flowchart.clay -o output/test.png

# Test with explicit seed for reproducibility
python -c "
from clay import layout_from_file
result = layout_from_file('examples/flowchart.clay', seed=42)
"
```

## Related Files

- `monitor_layout.py` - Monitoring script with OptimizationMonitor class
- `analyze_convergence.py` - Automated convergence testing across all examples
- `clay/layout.py:660` - `layout_graph()` function with callback support
- `clay/layout.py:228` - `simple_spring_layout()` initialization function
- `clay/layout.py:549` - `energy_function()` with penalty weights
