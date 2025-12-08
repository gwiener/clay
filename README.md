# CLay - Claude's Layout algorithm

`clay` is an automatic graph layout engine designed for achieving the most aesthetically-pleasing results
for small-to-medium graphs fitting into a limited space.

## Background: *why am I doing this*?
Can coding assistants, like Claude Code, help in designing and implementing *algorithms*,
not just boilerplate application code?
This repository is an experiment to show that they can.
After all, AI model trainers have access to NumPy docs and SciPy tutorials as readily as to front-end crash-course.
The experience and best practices for tweaking formulas and adjusting coefficients somewhat differ from choosing colors and optimizing responsiveness – but surprisingly, not by much.

My name is Guy Wiener. I've been a Data Scientist for the last 10 years or so, but I'm a Software Engineer at heart.
As one, my favorite way of explaining complex workflows, to myself or to others, is drawing a diagram.
Nothing too fancy - just a handful of boxes and ellipses connected by arrows.
However, I *hate* arranging them by hand to get a nice layout. Can't my computer do that for me?

Sadly, existing automatic graph layout engines, like GraphViz or D2, fail to hit the sweet-spot I am looking for.
Real-world workflows are rarely a well-balanced tree; more often, they include long sequences or wide branches.
On one hand, when laid out by a rank-based algorithm, the result is often too large to fit to a single page or a slide.
On the other hand, spring-based or force-directed layouts have little regard to the order within the flow itself.

After years of frustration from existing tools, I presented Claude with the following requirements:

1. Nodes must not intersect, and should be placed at a minimal distance apart.
2. Linked nodes should be placed close to each other (but not closer than the minimal distance).
3. Edges should not cross nodes.
4. Edges should not cross each other.

Up to this point, these are roughly the same requirements as from any layout engine.
My following preferences are the ones making the problem more challenging:

5. All the nodes must be placed within a canvas with given dimensions.
6. Chains of nodes should be as colinear as possible.
That is, given the edges `A->B` and `B->C`, the node `B` should be placed near the line connecting `A` and `C`.
(Later, I relaxed this requirement to hold only if these edges are marked a a part of a *flow*,
to allow for nodes appearing aside the main sequences.)
7. The overall result should be as compact as possible.

The last two requirements are contradicting, making it a minimax problem.
This is probably why arranging realistic flow diagrams manually is annoying.
It's also probably the reason why off-the-shelf tools are not too happy supporting this full set of demands.

Then, I asked the AI "can you design and implement a tool that aims at achieving these goals?"
The surprising answer was "yes".
Apparently, function optimizers can find solutions balancing between contradicting guidelines,
once they are phrased as penalty functions.
The only missing piece was that, despite my background in Python, I had no idea how to write such geometrical metrics.

The problem was not with the vibe-coding; it was *vibe-algo-ing*. 
Can I guide a Generative AI agent *to write the formulas for me*?

I set out find out.
This repository is both the result and the report of this experiment.