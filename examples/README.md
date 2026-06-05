# Examples

Quick-start scripts demonstrating RadioRange-Sim usage.

## CLI (recommended)

```bash
# One-command interactive 3D visualization (4 scenes)
radiorange --mode interactive

# Basic ranging
radiorange --scene tdl_a --radios uwb --algo threshold --trials 200

# Multi-algorithm comparison
radiorange --mode compare-algos --scene tdl_a --radios all --trials 100

# Material sweep
radiorange --mode compare-materials --radios all --trials 50

# Ray-tracing with cache
radiorange --scene box --radios uwb --dump-truths cache/rt/my_box
radiorange --from-cache cache/rt/my_box --algo leading_edge --trials 500

# Single-scene 3D visualization (from cache)
radiorange --mode rt-viz --from-cache cache/rt/box

# Trajectory-based ranging simulation
radiorange --mode measure --trajectory-scene corridor --radios uwb --impairments full

# Impairment ablation
radiorange --mode ablation --ablation-mode ablation --trials 100
```

## Python API

| File | Description |
|------|-------------|
| `01_basic_uwb_tdl.py` | Minimal example: UWB + TDL-A + ThresholdLDE |
| `02_compare_algorithms.py` | Compare all 5 LDE algorithms programmatically |
| `03_cli_cheatsheet.py` | CLI reference |

### Run

```bash
python3 examples/01_basic_uwb_tdl.py
python3 examples/02_compare_algorithms.py
```
