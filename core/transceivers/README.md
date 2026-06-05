# Radio Observation Models

Each radio converts physical multipath truth into clean CSI/CIR and observed
CSI/CIR.

## `observation_model: "snr"`

Compact model. It adds one equivalent noise term controlled by `snr_db`.

This is useful for fast sweeps and fair baseline comparisons.

## `observation_model: "explicit"`

Device-observation model. It separates several effects that can be swept
independently:

- `csi_noise_snr_db`: additive CSI noise level
- `csi_amplitude_std`: per-bin multiplicative amplitude error
- `csi_phase_std_rad`: per-bin phase error
- `common_phase_offset_rad`: deterministic common phase offset
- `common_phase_std_rad`: random common phase offset per observation
- `sampling_phase_offset_s`: deterministic residual timing offset
- `random_sampling_phase_std_s`: random residual timing offset
- `sfo_ppm`: residual sampling-frequency offset approximation
- `dc_null`: null the DC subcarrier for OFDM radios
- `active_subcarrier_fraction`: keep only the central occupied bandwidth
- `pilot_spacing_subcarriers`: sparse pilot sampling plus linear interpolation

UWB currently uses the amplitude/phase/timing/noise terms. WiFi and 5G also use
the OFDM-specific mask and pilot interpolation terms.
