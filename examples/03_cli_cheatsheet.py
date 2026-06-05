"""Example 3 — CLI quick reference.

For full-featured runs (ray-tracing, trajectory, ablation, HTML viz),
use the CLI:

    # One-command interactive 3D visualization (4 scenes)
    radiorange --mode interactive

    # Single experiment
    radiorange --scene tdl_a --radios uwb --algo threshold --trials 200

    # Ray-tracing with cache
    radiorange --scene box --radios uwb --dump-truths cache/rt/my_box
    radiorange --from-cache cache/rt/my_box --algo leading_edge --trials 500

    # Single-scene 3D visualization (from existing cache)
    radiorange --mode rt-viz --from-cache cache/rt/box

    # Multi-algorithm comparison
    radiorange --mode compare-algos --scene tdl_a --radios all --trials 100

    # Material comparison
    radiorange --mode compare-materials --radios all --trials 50

    # Measure — ranging along a walking path
    radiorange --mode measure --trajectory-scene corridor --radios uwb --impairments full

    # Impairment ablation
    radiorange --mode ablation --ablation-mode ablation --trials 100

See README.md for the full documentation.
"""
print(__doc__)
