# Analysis: Why is Total Energy So Huge?

## Question
When `architecture.clay` fails to optimize, the total energy is **1,744,200** compared to **30,207** for successful `simple.clay`. What causes this 58× difference?

## Answer: It's NOT the weights - it's the layout quality!

### The Math

The edge length penalty function is:
```python
def edge_length_penalty(positions, edges):
    total = 0
    for (u, v) in edges:
        dist = np.linalg.norm(positions[v] - positions[u])
        total += dist ** 2  # Sum of squared distances
    return total
```

This is multiplied by weight = 10 in the energy function.

### Comparison

| Example | Nodes | Edges | Success | Raw Edge Length | Weight | Weighted | Per-Edge Raw |
|---------|-------|-------|---------|-----------------|--------|----------|--------------|
| simple.clay | 3 | 2 | ✓ | 2,607 | 10 | 26,069 | **1,303** |
| architecture.clay | 7 | 7 | ✗ | 151,624 | 10 | 1,516,240 | **21,661** |

### Key Insight

**Per-edge raw penalty:**
- Simple (successful): 1,303 → sqrt(1,303) ≈ **36 units** average edge length
- Architecture (failed): 21,661 → sqrt(21,661) ≈ **147 units** average edge length

The failed case has edges **4× longer** on average!

With a bounding box of 800×600, having average edge lengths of 147 units means nodes are spread out across the entire canvas.

## Root Cause

The huge energy is **not caused by the weight** - it's caused by:

1. **Bad initial layout**: Grid initialization spaces nodes far apart (especially with 7 nodes in 800×600 bbox)
2. **Optimizer fails immediately**: Only 3 iterations before "ABNORMAL_TERMINATION_IN_LNSRCH"
3. **Stuck at bad position**: Nodes never get moved closer together
4. **We measure the failure**: The final energy reflects the awful initial layout

### Energy Breakdown for Failed Case

```
edge_length:  151,624 × 10   = 1,516,240  (86.9%)  ← HUGE because nodes far apart
straightness:  35,126 × 5    =   175,629  (10.1%)
area:          72,274 × 1    =    72,274  ( 4.1%)
overlap:            0 × 1000 =         0  ( 0.0%)
edge_node:          0 × 200  =         0  ( 0.0%)
bbox:               0 × 100  =         0  ( 0.0%)
                                ─────────
                   TOTAL:     1,744,200
```

## Are the Weights Wrong?

**No.** The weights are appropriately scaled:

- `overlap`: 1000 (hard constraint - must be 0)
- `edge_length`: 10 (soft constraint - prefer shorter)
- `straightness`: 5 (soft constraint - prefer straight)
- `edge_node`: 200 (hard-ish constraint - avoid crossings)
- `bbox`: 100 (hard constraint - must fit)
- `area`: 1 (soft objective - minimize waste)

The problem is:
- **When optimization succeeds**: Nodes move close together, edge_length raw penalty is small (~2,600)
- **When optimization fails**: Nodes stay far apart, edge_length raw penalty is huge (~151,000)

### Proof: Check a Successful Larger Example

Let's check `chain.clay` (4 nodes, 3 edges, **successful**):

```
Success: True
Iterations: 68
Total Energy: 47,201
edge_length: 3,904 raw × 10 = 39,039 weighted
Per edge: 3,904 / 3 = 1,301 raw
```

Per-edge raw penalty is **1,301** - almost identical to simple.clay (1,303)!

## Conclusion

The energy is huge because **the optimizer failed to optimize**, leaving nodes in their initial far-apart positions. The weights are fine - we need to fix the optimizer failure (see `OPTIMIZATION_INVESTIGATION.md`).

### What This Means

1. ✓ **Weights are correctly tuned** - successful runs have similar per-edge penalties
2. ✗ **Optimizer is failing** - it can't even take a few steps to improve layout
3. ✗ **Initial positions matter** - grid layout works for small graphs, fails for larger ones
4. ✗ **Line search is brittle** - numerical issues prevent gradient descent

### Solutions (Priority Order)

1. **Fix optimizer settings** (maxls, ftol, gtol) - see OPTIMIZATION_INVESTIGATION.md
2. **Better initialization** - smarter initial positions
3. **Multi-start optimization** - try several random starts
4. **Energy scaling/normalization** - might help numerical stability (lower priority)

## Experiments to Try

### 1. Force simple.clay to fail by messing with initial positions
```python
# Use very spread-out initial positions
x0 = np.array([[0, 0], [800, 0], [400, 600]]).flatten()
```
Prediction: Energy will be huge even though it's only 3 nodes.

### 2. Hand-craft good initial positions for architecture.clay
```python
# Place nodes close together initially
x0 = np.random.uniform(200, 400, size=(7, 2)).flatten()
```
Prediction: Optimizer more likely to succeed, energy will be reasonable.

### 3. Vary the weights
```python
# Try edge_length weight = 1 instead of 10
```
Prediction: Won't help - optimizer still fails, final energy just 10× smaller but still "failed" quality.
