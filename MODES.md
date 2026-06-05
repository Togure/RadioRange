# Configuration & Experiment Modes

This guide covers how to configure RadioRange-Sim and use all eight experiment modes. For a quick introduction, see [README.md](README.md).

---

## Step-by-Step Configuration

### 1. Choose Protocols (`--radios`)

```bash
# Single protocol
radiorange --radios uwb ...

# Multiple protocols (comma-separated)
radiorange --radios uwb,wifi ...

# All three
radiorange --radios all ...
```

**Customizing radio parameters:** Each protocol has a default profile in `config/radio_profiles/default_radios.yaml`. Copy it and edit:

```yaml
# my_radios.yaml
radios:
  uwb:
    enabled: true
    carrier_frequency_hz: 7987200000.0   # UWB Ch9, 7.99 GHz
    bandwidth_hz: 499200000.0            # 499.2 MHz
    cir_bins: 1024
    accumulation_count: 128              # Processing gain
    interpolation_factor: 10
    window: hamming                      # CIR sidelobe suppression
    adc_bits: 6                          # DW1000-class ADC
```

```bash
radiorange --config my_radios.yaml --scene tdl_a --algo threshold --trials 200
```

| Protocol | Bandwidth | Rayleigh Resolution | Effective Resolution | Key Limitation |
|----------|-----------|---------------------|----------------------|----------------|
| **UWB** | 499.2 MHz | 2.0 ns (0.60 m) | 0.06 m (10× interp) | — |
| **WiFi** | 160 MHz | 6.3 ns (1.88 m) | 0.19 m (10× interp) | Narrower BW merges close multipath |
| **5G NR** | 122.88 MHz | 8.1 ns (2.44 m) | 0.24 m (10× interp) | Narrowest BW, but denser pilot grid |

### 2. Choose a Scene (`--scene`)

```bash
# List all available scenes
radiorange --list-scenes
```

```
Statistical (3GPP TR 38.901, no Sionna required):
  tdl_a    TDL-A  (NLOS, 23 taps)
  tdl_b    TDL-B  (NLOS, 23 taps)
  tdl_c    TDL-C  (NLOS, 24 taps)
  tdl_d    TDL-D  (LOS, 13 taps + K-factor)
  tdl_e    TDL-E  (LOS, 14 taps + K-factor)
  two_path LOS + ground reflection

Sionna built-in scenes (requires sionna):
  box               Simple enclosed box
  box_knife         Box with knife-edge obstacle
  box_knife_concrete Box + knife-edge, concrete walls
  etoile            Etoile / Arc de Triomphe
  florence          Florence city scene
  munich            Munich city centre
  simple_room       4×4×3m rectangular room
```

```bash
# Statistical channel (fast, no GPU)
radiorange --scene tdl_a --radios uwb --algo threshold --trials 200

# Sionna ray-tracing (needs GPU / sionna package)
radiorange --scene box --radios uwb --algo leading_edge --trials 100
```

**Custom floorplan:** Use any PNG image as a 3D scene — walls are extracted by color:

```bash
radiorange --floorplan-image floorplans/complex_building.png \
           --tx 5 10 1.5 --rx 32 20 1.5 \
           --radios uwb --dump-truths cache/rt/my_building
```

Colors map to materials via `environments/materials.py`. Adjust detection tolerance with `--floorplan-tolerance 40`.

**Override TX/RX positions** for any built-in scene:

```bash
radiorange --scene box --tx -2 0 1.5 --rx 3 1 1.5 ...
```

### 3. Choose Impairments (`--impairments`)

Two presets, plus full control via YAML:

```bash
# Clean channel — ideal hardware (default)
radiorange --impairments none ...

# All 11 impairments enabled
radiorange --impairments full ...
```

**Custom impairment set:** Copy `config/impairments/full.yaml` and toggle individually:

```yaml
# my_impairments.yaml
impairments:
  enable_sfo: true                # Sampling Frequency Offset (20 ppm)
  enable_cfo: true                # Carrier Frequency Offset (200–500 Hz)
  enable_adc_quantization: true   # ADC quantization (4–12 bits)
  enable_iq_imbalance: true       # I/Q gain/phase mismatch
  enable_agc: false               # Automatic Gain Control
  enable_antenna_offset: false    # Antenna Phase Center Variation
```

```bash
radiorange --config my_impairments.yaml --scene tdl_a ...
```

| Impairment | Effect on Ranging | Key Finding |
|------------|-------------------|-------------|
| **ADC quantization** | UWB: −23% RMSE at 4–6 bits (stochastic resonance) | Counter-intuitive: lower bits can improve UWB |
| **I/Q imbalance** | WiFi: +18% RMSE (mirror peak confuses LDE) | UWB nearly immune (real-valued CIR) |
| **ADC timing jitter** | WiFi: +15% RMSE in complex-LOS | Reorders closely-spaced CIR peaks |
| **SFO, CFO, AGC, antenna PCV** | < 1% each | Modern DSP suppresses these below multipath noise floor |

### 4. Choose Algorithm (`--algo`)

```bash
radiorange --algo leading_edge ...
```

| Algorithm | Principle | Key Parameter | Best For |
|-----------|-----------|---------------|----------|
| `max_peak` | argmax of \|CIR\| | None | LOS, strongest path is correct |
| `threshold` | First bin > α · max(\|CIR\|) | α = 0.18 | Fast, NLOS-robust |
| `leading_edge` | Adaptive μ + nσ from noise floor | n_sigma = 4.0 | Best overall, SNR-adaptive |
| `search_back` | From peak, leftward to α · peak_amp | α = 0.18 | Peak-biased scenes |
| `chip_lde` | Noise floor + 10 dB, min_run = 3 | threshold = 10 dB | Mimics DW1000/DW3000 hardware |

```bash
# Custom algorithm parameters via config
radiorange --config config/runs/tdl_a.yaml --algo leading_edge
```

### 5. Set Trials & Seed

```bash
radiorange --trials 500 --seed 42 ...
```

- **`--trials N`** — number of Monte Carlo trials (default: from config, usually 100)
- **`--seed N`** — random seed for reproducibility (default: 42)

---

## Experiment Modes

```bash
radiorange --mode <MODE> [options]
```

### single

The default mode. Runs one configuration and produces CIR comparison + error distribution plots:

```bash
radiorange --scene munich --radios uwb --algo leading_edge --impairments full --trials 200
```

Output: `outputs/clean_cir_comparison.png`, `outputs/snr_based_cir_comparison.png`, `outputs/ranging_error_comparison.png`

### interactive

One-command 4-scene interactive 3D HTML reports. Generates RT caches if needed:

```bash
radiorange --mode interactive [--trials 100] [--radios uwb,wifi,fiveg]
```

Scenes: `box_knife` (NLOS), `box` (LOS), `etoile` (urban LOS), `complex_building` (custom floorplan).

Each HTML report contains a 3D ray-path viewer, CIR + multipath identification panels, and a ranging error CDF — see Fig. 2–5 in the README for examples of the visual output.

### compare-algos

Runs all 5 LDE algorithms on the same scene and prints a comparison table:

```bash
radiorange --mode compare-algos --scene tdl_a --radios all --trials 100
```

```
Algorithm              UWB RMSE   WiFi RMSE   5G RMSE
---------------------------------------------------------
MaxPeak                 0.423      0.891       1.245
Threshold(0.18)         0.287      0.624       0.893
LeadingEdge(4σ)         0.164      0.351       0.663
SearchBack(0.18)        0.352      0.782       1.051
ChipLDE(10dB)           0.178      0.389       0.704
```

### compare-materials

Sweeps wall materials in a procedural room to measure material sensitivity:

```bash
radiorange --mode compare-materials --radios all --trials 50
```

Output: `outputs/material_comparison/`

### rt-viz

Interactive 3D HTML report for a **single** scene from a pre-computed RT cache:

```bash
# Step 1: generate cache
radiorange --scene box --radios uwb --dump-truths cache/rt/my_box

# Step 2: visualize
radiorange --mode rt-viz --from-cache cache/rt/my_box --radios uwb --trials 100
```

Output: `outputs/interactive/<scene>_chip_sim.html`

> **`rt-viz` vs `interactive`:** `--mode interactive` is the all-in-one quick start (generates caches + builds HTML for 4 scenes). `--mode rt-viz` works with a single pre-existing cache — useful when you've already run `--dump-truths` and just want to visualize.

### measure

Ranging simulation along a trajectory — two ways to use it:

**Built-in demo** (no files needed):

```bash
radiorange --mode measure --trajectory-scene corridor --radios uwb --speed 0.5
radiorange --mode measure --trajectory-scene t_junction --radios uwb --impairments full
```

| Option | Default | Description |
|--------|---------|-------------|
| `--trajectory-scene` | `corridor` | `corridor` (25m straight) or `t_junction` (T-shaped) |
| `--speed` | 0.5 m/s | Walking speed |
| `--waypoints` | 40 | Number of waypoints (adjust for density) |

**Custom floorplan + trajectory** (user-provided PNG + CSV):

```bash
radiorange --mode measure \
           --floorplan-image my_office.png \
           --waypoints-file trajectory.csv \
           --floorplan-width-m 15.0 \
           --tx 2 1.5 1.5 \
           --radios uwb --impairments full
```

Example waypoint CSVs are provided alongside the built-in floorplans:

```
floorplans/
├── corridor_waypoints.csv    # 201 points along 25m corridor
└── t_junction_waypoints.csv  # 152 points through T-junction
```

Output: `outputs/measure/<scene>_<ts>/` (CSV + error_map.png + error_map_grid.png + error_vs_distance.png + error_cdf.png + rmse_summary.txt)

### fingerprint

Generate a WiFi RSSI fingerprint radio map on a floorplan grid. Ray-tracing is run at every grid point × AP pair, producing per-AP RSSI + range estimate + range error panels:

```bash
# Default — 2 auto-placed APs, 2 m grid
radiorange --mode fingerprint \
           --floorplan-image floorplans/complex_building.png \
           --floorplan-width-m 42
```

| Option | Default | Description |
|--------|---------|-------------|
| `--floorplan-image` | *(required)* | Path to floorplan PNG |
| `--floorplan-width-m` | *(required)* | Physical width of floorplan in meters |
| `--aps` | `auto` | Path to AP positions CSV (`ap_id,x,y,z`); auto-generates 2 APs if omitted |
| `--grid-spacing` | `2.0` | Grid point spacing in meters |
| `--tx-power` | `20.0` | WiFi TX power in dBm |
| `--algo` | `leading_edge` | LDE algorithm for range estimation |
| `--impairments` | `none` | `none` or `full` |
| `--trials` | `1` | Number of trials per measurement |
| `--seed` | `42` | Random seed |
| `--output` | `auto` | Output directory (auto-generated if omitted) |

**AP positions file format** (`ap_id,x,y,z`):

```
ap_id,x,y,z
ap1,10.0,5.0,2.5
ap2,30.0,5.0,2.5
```

When `--aps` is not provided, 2 APs are auto-placed at 1/3 and 2/3 of the room's width and depth.

**Output:**

```
outputs/fingerprint/<name>_<ts>/
├── fingerprint_db.csv       # per grid-point × AP results
├── radiomap_ap1.png          # RSSI + Estimated Range + Range Error panels
├── radiomap_ap2.png
├── metadata.json
└── rt_cache/                 # per-measurement RT caches (reusable)
```

Each `radiomap_<ap_id>.png` is a 3-panel figure:
1. **RSSI (dBm)** — interpolated heatmap of received signal strength
2. **Estimated Range (m)** — LDE range estimate at each grid point
3. **Range Error (m)** — |estimated − true range| error map

Red ★ star markers indicate AP positions on all panels.

> **GPU acceleration:** Sionna RT can use GPU if TensorFlow GPU is installed. No extra configuration is needed — the simulator auto-detects available GPUs.

### ablation

Measures how each impairment individually affects ranging RMSE:

```bash
radiorange --mode ablation --ablation-mode ablation --trials 100
```

| Sub-mode | Description |
|----------|-------------|
| `ablation` | Baseline vs. each impairment toggled on alone |
| `sweeps` | Continuous parameter sweeps (I/Q gain, SNR, ADC bits) |
| `all` | Both ablation + sweeps |

Output: `outputs/impairment_ablation_final/<ts>/`

---

## Ray-Tracing Cache

For Sionna scenes, ray-tracing is the slowest step. Use caching to run it once and replay many times:

```bash
# Generate cache
radiorange --scene munich --radios uwb,wifi,fiveg --dump-truths cache/rt/munich_run1

# Replay with different algorithms (no re-tracing)
radiorange --from-cache cache/rt/munich_run1 --algo threshold --impairments none --trials 500
radiorange --from-cache cache/rt/munich_run1 --algo leading_edge --impairments full --trials 500
radiorange --from-cache cache/rt/munich_run1 --algo chip_lde --impairments none --trials 1000
```

The cache stores per-protocol `ChannelTruth` objects (NPZ format) so RT is done only once per carrier frequency. Caches live under `cache/rt/<name>/`.

---

## Configuration System

RadioRange-Sim uses a layered YAML config that composes small, reusable fragments:

```yaml
# config/runs/quick_start.yaml
include:
  - ../defaults.yaml                       # Seed, timing
  - ../radio_profiles/default_radios.yaml   # UWB/WiFi/5G parameters
  - ../observation/snr.yaml                # SNR-based observation model
  - ../channels/quick_start_tdl_a.yaml     # TDL-A channel
  - ../experiments/ranging/ranging.yaml     # Experiment definition
```

**Layers** (each in `config/`):

| Layer | Directory | Purpose |
|-------|-----------|---------|
| Defaults | `defaults.yaml` | Global seed, timing mode |
| Radio profiles | `radio_profiles/` | Protocol-specific parameters per radio |
| Observation model | `observation/` | SNR-based or explicit device observation |
| Impairments | `impairments/` | Hardware impairment toggles (none / full) |
| Channels | `channels/` | Environment definitions (TDL, ray-tracing, rooms) |
| Experiments | `experiments/` | Experiment mode config |
| Runs | `runs/` | Composed configs that include one of each layer |

Create custom runs by composing layers:

```bash
# Use a composed run file
radiorange --experiment config/runs/tdl_a.yaml

# Override individual settings from CLI
radiorange --experiment config/runs/tdl_a.yaml --algo leading_edge --trials 500
```

CLI arguments always take precedence over config file values.

---

## Output Structure

All results land under `outputs/`:

```
outputs/
├── clean_cir_comparison.png           # single mode: CIR plots
├── ranging_error_comparison.png       # single mode: error distribution
├── material_comparison/               # compare-materials mode
├── impairment_ablation_final/         # ablation mode
├── measure/
│   ├── corridor_<ts>/                 # error_map.png + CSV + CDF
│   └── t_junction_<ts>/
├── fingerprint/
│   └── complex_building_<ts>/         # radiomap_*.png + fingerprint_db.csv
└── interactive/
    ├── box_knife_chip_sim.html        # interactive 3D report
    ├── box_chip_sim.html
    ├── etoile_chip_sim.html
    └── complex_building_chip_sim.html
```

---

## Development

```bash
pip install -e ".[dev]"

# Run tests
pytest tests/ -v                         # all 154 tests
pytest tests/ --ignore=tests/test_sionna_builtin.py  # skip Sionna tests
```
