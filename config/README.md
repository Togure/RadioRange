# Configuration Layout

The recommended entry point is a run config under `config/runs/`:

```bash
.venv/bin/python main.py --experiment config/runs/tdl_a.yaml
```

Run configs compose smaller config fragments through `include`.
Later files override earlier files.

## Layers

- `defaults.yaml`: global defaults (seed, hardware flags).
- `radio_profiles/`: UWB, WiFi, and 5G hardware/profile parameters.
- `observation/`: SNR-based (`snr.yaml`) or explicit device-observation (`explicit.yaml`) models.
- `impairments/`: Hardware impairment toggles — `none.yaml` (all off) or `full.yaml` (all on).
- `channels/`: Environment definitions (TDL models, manual paths, ray-tracing rooms).
- `experiments/`: Experiment mode definitions (currently `ranging/ranging.yaml`).
- `runs/`: Composed runnable configs that include one of each layer above.
