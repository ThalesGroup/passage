---
title: "Notebooks"
icon: material/notebook-multiple
---

# Dataset / solver notebooks

Demo and analysis notebooks for the data-generation side. Each page links to the source notebook, which you can open on the Hugging Face Hub or download and run locally.

[Notebook rendering and licenses](../../licenses/notebook-exports.md)

<div class="grid cards" markdown>

- **Getting started**

    ---

    The shortest path from the dataset to a training example: load one released scene, decode its tensor, draw a reference route and turn it into a segmentation target.

    [Get started with PASSAGE](demo.md)

- **Obstacles**

    ---

    Obstacle masks from least to most covered, drawn coverage against its target, and the detour that obstacles force on the optimal route over a whole validation split.

    [Explore obstacle coverage](obstacles.md)

- **The three objectives**

    ---

    What the `elevation`, `slope_uphill` and `energy` objectives reward on one scene, their elevation profiles, and the extra cost of a route optimised for the wrong objective.

    [Compare the three objectives](costmodel.md)

- **Exact re-solve**

    ---

    Solves stored labels again with the exact float64 A* reference, compares costs and routes, shows why the search budget matters, and repeats the check on a sample of validation scenes.

    [Check the reference routes](solve.md)

- **Timing a planner**

    ---

    Times the canonical A* planner of `passage-eval` with the practices that make a timing meaningful: warm-up, repetitions, percentiles and a stated machine.

    [Time a planner](astar-inference-timing.md)


</div>
