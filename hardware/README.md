# Hardware vs Observation Models

The project is a ranging/positioning simulator, not a full communication
receiver simulator. Hardware effects are therefore modeled at the lowest layer
that is needed to change the CIR/CSI seen by the ranging algorithm.

## Hardware Layer

Files in `hardware/` modify the physical channel truth, receiver timing state,
or CIR-domain observation before the ranging algorithm runs.

Use this layer for effects that should apply consistently across radios or
should change the physical path interpretation:

- `clock.py`: first-order truth-level SFO/CFO approximations.
- `antenna.py`: positioning-level antenna phase-center variation (PCV).
- `adc.py`: CIR-envelope quantization approximation.
- `agc.py`: CIR-envelope gain/clip approximation.
- `impairments.py`: orchestration for truth-level impairments.

`clock.py`, `antenna.py`, `adc.py`, and `agc.py` all have active integration
points. ADC/AGC are intentionally CIR-domain approximations, not full I/Q
front-end models.

## Error Layers

| Error | Current layer | Affects | Why modeled there |
| --- | --- | --- | --- |
| `sync_bias_s`, `clock_bias_s` | timing/system | all path delays | Ranging directly sees a ToF bias |
| antenna PCV | truth/device | per-path delays | PCV is an equivalent path-length error |
| truth-level SFO | truth/device | path delays | First-order clock scale error |
| truth-level CFO | truth/device | path phases | First-order approximation of uncompensated carrier offset |
| CSI noise/amplitude/phase | observation | observed CSI | Device reports imperfect CSI/CIR |
| residual SFO / sampling phase | observation | observed CSI phase slope | Represents post-compensation residuals |
| subcarrier mask / pilots | observation | observed OFDM CSI | Protocol/device reports only part of H(f) |
| ADC quantization | CIR approximation | observed discrete CIR | We care about weak first-path visibility |
| AGC/clipping | CIR approximation | observed discrete CIR | We care about dynamic range and threshold distortion |

## Observation Model

The radio `observation_model` in `core/transceivers/base_radio.py` modifies the
device-observed CSI/CIR after clean CSI has been generated.

Use this layer for effects that describe how a device reports CSI/CIR:

- SNR-based additive CSI noise.
- Explicit CSI amplitude/phase errors.
- Residual sampling phase offset.
- Residual SFO approximation at the observation layer.
- OFDM subcarrier mask.
- Sparse pilot sampling and interpolation.

In short:

- `hardware/` changes the device/channel state.
- `observation_model` changes the measured CSI/CIR returned by the radio.

Some real-world effects can be modeled at either layer. The rule of thumb is:
if it changes the path truth for all later processing, put it in `hardware/`;
if it changes only what the radio reports, put it in `observation_model`.

Avoid enabling the same physical effect in both layers unless this is the
intended experiment. For example, truth-level `enable_sfo` models a device clock
scale error, while `explicit_impairments.sfo_ppm` models residual observed CSI
phase slope after synchronization/tracking.
