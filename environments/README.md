# Environment Inputs

`main.py` selects the channel-truth generator through `environment.type`.

## `standard_tdl`

Uses Sionna's 3GPP TDL model and outputs physical path coefficients `a_paths`
and delays `tau_paths_s`. The config field `environment.model` can be set to
`TDL-A`, `TDL-B`, `TDL-C`, `TDL-D`, or `TDL-E`.

Example:

```bash
.venv/bin/python main.py --config config/runs/tdl_d.yaml
```

## `manual_paths`

Uses user-defined multipath entries. Each path can define `tau_s`, `amplitude`,
`phase_rad`, `azimuth_deg`, and `elevation_deg`. Only `tau_s` is required for
the current ranging pipeline.

Example:

```bash
.venv/bin/python main.py --config config/runs/two_path.yaml
```

## `simple_room`

Supports two early engines:

- `image_method`: deterministic 2D/3D rectangular-room image method.
- `sionna_box`: Sionna RT built-in 3D `box` scene with `PathSolver`.

Examples:

```bash
.venv/bin/python main.py --config config/runs/simple_room_2d.yaml
.venv/bin/python main.py --config config/runs/simple_room_3d.yaml
```

## Reserved

`sionna_builtin_scene` and `custom_scene` are reserved in the registry for the
next phase. They will cover Sionna's larger built-in scenes and user-provided
scene files.
