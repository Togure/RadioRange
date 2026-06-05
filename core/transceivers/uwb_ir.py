"""UWB Impulse Radio — DW3000 / QM33xxx compatible device model.

═══════════════════════════════════════════════════════════════════════════════
PHYSICS: How a UWB CIR is obtained
═══════════════════════════════════════════════════════════════════════════════

The transmitter sends a preamble composed of M identical symbols.  Each symbol
is a known ternary Ipatov sequence (length 31 for PRF64, 127 for PRF16) of
pulse chips.  The receiver cross-correlates the received signal with the known
preamble code — the result of each cross-correlation IS a CIR estimate.

All M symbols are coherently accumulated in a hardware accumulator before
any CIR is read out:

    CIR_final = (1/M) * Σₘ CIRₘ    (coherent, phase-aligned)

Because the signal component adds linearly (×M) while thermal noise adds in
power (×√M), the SNR gain is:

    SNR_gain = 10 × log₁₀(M)

This happens entirely inside the silicon — the host never sees individual
symbol-level CIRs, only the accumulated result.

═══════════════════════════════════════════════════════════════════════════════
SIMULATION CHAIN
═══════════════════════════════════════════════════════════════════════════════

  H(f) = Σ a_i · exp(-j 2πf τ_i)          ← physics (path_response)
    ↓
  H(f) × W(f)                               ← windowing (Hamming/Hann/Blackman)
    ↓
  + per-subcarrier noise                     ← explicit / SNR observation model
  + common phase / residual SFO             ← see observe_frequency_response
  + ADC quantization (6-bit, freq domain)  ← see hardware.adc
    ↓
  Ĥ(f) → IFFT → |Ĥ(t)|                      ← CIR (discrete + upsampled continuous)
    ↓
  + AGC + clipping                          ← CIR-domain impairments

═══════════════════════════════════════════════════════════════════════════════
DEVICE CONFIGURATION PARAMETERS — full register map of DW3000
═══════════════════════════════════════════════════════════════════════════════

── INCLUDED (affect the simulated CIR) ─────────────────────────────────────

  Parameter              SDK Enum / Value        Used In Sim    Why
  ─────────────────────────────────────────────────────────────────────────
  accumulation_count     DWT_PLEN_*              uwb_ir.py      Coherent SNR gain = 10×log₁₀(M).
                        (64,128,256,512,         config         Default 128 (SDK default).
                         1024,1536,2048,4096)                   SDK note: PLEN_4096 for
                        Default: 128                             dual-antenna CIR capture.
                        Per-symbol duration
                        ≈ 1 μs (PRF64).

  cir_bins               DWT_CIR_LEN_IP_PRF64    uwb_ir.py      Time-domain CIR sampling points.
                        = 1016 (PRF64)           config         Determines IFFT size →
                        DWT_CIR_LEN_IP_PRF16                    delay bin spacing
                        = 992  (PRF16)                          = 1 / 499.2 MHz ≈ 2.004 ns.
                        Default: PRF64 = 1016                   We default to 1024 (power of 2
                        STS CIR: 512 bins                       for efficient FFT while matching
                                                                DW3000's 1016 bins).
                                                                Dual-antenna: 2×512 bins
                                                                (placeholder for PDOA mode).

  bandwidth_hz           499.2 MHz (chip        config          Sampling rate = chip frequency.
                        frequency)                             FFT bin spacing ≈ 499.2 MHz / N.
                        Fixed in hardware.

  carrier_frequency_hz   Ch5 = 6489.6 MHz       config          Affects: path loss, Fresnel Γ,
                        Ch9 = 7987.2 MHz                       antenna PCV.
                        Default: Ch9                            5G coexistence: Ch5 better.

  adc_bits               6-bit (DW3000 fixed)   config →        Quantization step Δ.
                                                hardware/adc    SNR_quant ≈ 37.9 dB (Nyquist)
                                                                      ≈ 68 dB (after 128× accum).

  window                 "hamming", "hann",     config          Spectral window applied to H(f)
                        "blackman", "none"                      before IFFT.  Matches the
                                                                implicit shaping of the matched
                                                                filter in a real receiver.

── NOT INCLUDED (negligible or orthogonal to ranging CIR) ─────────────────

  Parameter              SDK Value              Why Excluded
  ─────────────────────────────────────────────────────────────────────────
  PAC                    4, 8, 16, 32           Only affects preamble *detection*
                        Default: 8              threshold / false-alarm rate.
                                                Once detected, ALL preamble
                                                symbols still accumulate.  The
                                                CIR is identical regardless of
                                                PAC.  Pure MAC-layer concern.

  PRF                    16 MHz, 64 MHz,        Drives cir_bins (included via
                        SCP (~100 MHz)          that parameter).  The PRF
                        Default: 64 MHz          choice itself only affects
                                                CIR bin count and symbol
                                                duration; bin count is already
                                                configurable via cir_bins.

  SFD type               IEEE 8-bit (0)         Frame delimiter — marks the
                        DW 8-bit (1)            boundary between preamble and
                        DW 16-bit (2)           PHR.  No effect on CIR content,
                        4z 8-bit (3)            which is captured entirely from
                        Default: 4z 8-bit       the preamble before SFD.

  Data rate              850 kbps, 6.8 Mbps     Only affects the payload/data
                        Default: 6.8 Mbps       portion of the packet — CIR is
                                                already locked in the accumulator
                                                before the data phase begins.

  Preamble code           9–24 (PRF64)          The specific ternary sequence
                          1–8  (PRF16)          used for correlation.  Affects
                        Default: 9              sidelobe shape but not the
                                                fundamental precision limit.
                                                All codes have the same aperiodic
                                                autocorrelation properties.

  PHR mode / rate        STD / EXT              Packet header format.  Not
                                                related to channel estimation.

  STS mode / length      OFF, Mode1/2/ND/SDC    Secure ranging (scrambled
                        Length: 64,128,256,...  timestamp sequence).  Adds a
                        Default: OFF for        2nd CIR for authentication, not
                        simple-ranging apps     for the primary range estimate.
                                                Can be enabled later for
                                                secure-ranging scenarios.

  GPIO / LED config      8 pins configurable    Board-level I/O.  Not in signal
                                                chain.

  SPI CRC / speed        Fast/Slow rate         Host communication bus.  Not
                                                in signal chain.

  Sleep / Wakeup         Several modes          Power management.  Not in
                                                signal chain.

  CIR read mode          Full 48-bit /          Post-accumulation readout
                        LO/MID/HI 32-bit        precision.  All modes have
                                                >> 100 dB dynamic range,
                                                far below thermal noise floor.

  TX power               Configurable           Affects RSSI / SNR but not
                        (per segment:           the relative CIR shape.
                        SHR/PHR/DATA/STS)        Raw SNR is captured by the
                                                link budget, not per-segment
                                                power registers.

  PDOA mode              M0 (off) /             Phase-difference-of-arrival
                        M1 / M3                 for angle estimation.  Requires
                                                dual-antenna hardware.  M3
                                                would give 2 × 512-bin CIRs
                                                instead of 1 × 1016.  Placeholder
                                                exists for future work.

  WiFi coexistence       COEX GPIO              Shared-medium arbitration.
                                                Not a CIR effect.

── PARAMETERS THAT EXIST IN CODE BUT ARE MARKED "NOT CONSIDERED" IN COMMENTS ─

  sfo_ppm (20.0)          — In default_radios.yaml for UWB.  The doc states
                             modern receivers compensate CFO/SFO in preamble
                             processing, leaving only *residual* terms.  The
                             residual SFO is in explicit.yaml, not here.
  cfo_hz (500.0)          — Same rationale as SFO: compensated by preamble.
                             Residual CFO is handled per-protocol in L2/L3.

═══════════════════════════════════════════════════════════════════════════════
REFERENCES
═══════════════════════════════════════════════════════════════════════════════
  - DW3XXX_Software_API_Guide_4p12.pdf
  - DW3000 SDK: config_options.h / config_options.c
  - DW3000 SDK: deca_device_api.h (register definitions)
  - IEEE 802.15.4z-2020 (HRP UWB PHY)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.models import ChannelTruth, RadioObservation
from core.transceivers.base_radio import BaseRadio, apply_cir_impairments, normalize_observation, normalize_with_reference, observe_frequency_response
from core.transceivers.common import continuous_uwb, path_response

_WINDOW_BUILDERS = {
    "hann": np.hanning,
    "hamming": np.hamming,
    "blackman": np.blackman,
}


class UwbImpulseRadio(BaseRadio):
    protocol = "uwb"

    def observe(self, truth: ChannelTruth, rng: np.random.Generator) -> RadioObservation:
        radio_cfg = self.config["radios"]["uwb"]

        # ── CIR bins ────────────────────────────────────────────────────────
        # cir_bins (new, preferred) → num_bins (legacy fallback) → 1024 default.
        # DW3000 hardware: 1016 bins (PRF64) or 992 bins (PRF16).
        # We default to 1024 — the nearest power of 2 to 1016, efficient for FFT.
        num_bins = int(radio_cfg.get("cir_bins") or radio_cfg.get("num_bins", 1024))
        bandwidth_hz = float(radio_cfg["bandwidth_hz"])
        factor = int(radio_cfg.get("interpolation_factor", 10))
        window_name = str(radio_cfg.get("window", "none")).lower()

        # ── Coherent accumulation ────────────────────────────────────────────
        # Number of preamble symbols coherently accumulated in the hardware
        # accumulator before CIR readout.  Each symbol is one full Ipatov
        # sequence (31 chips for PRF64).  SNR gain = 10 × log₁₀(M).
        accumulation_count = int(radio_cfg.get("accumulation_count", 128))
        accumulation_gain_db = 10.0 * np.log10(float(accumulation_count))

        frequency_hz = np.fft.fftfreq(num_bins, d=1.0 / bandwidth_hz)
        h_clean = path_response(frequency_hz, truth.a_paths, truth.tau_paths_s)

        if window_name in _WINDOW_BUILDERS:
            h_clean = h_clean * np.fft.ifftshift(_WINDOW_BUILDERS[window_name](num_bins))
        elif window_name != "none":
            raise ValueError(f"Unknown UWB window: {window_name}")

        h_observed, observation_meta = observe_frequency_response(
            h_clean,
            frequency_hz,
            radio_cfg,
            rng,
            enable_mask_and_pilots=False,
            impair_cfg=self.config.get("impairments", {}),
        )

        iq_meta: dict[str, Any] = {}
        if bool(radio_cfg.get("enable_iq_imbalance", False)
                or self.config.get("impairments", {}).get("enable_iq_imbalance", False)):
            from hardware.iq_imbalance import apply_iq_imbalance

            sigma_eps = float(radio_cfg.get("iq_gain_imbalance", 0.03))
            sigma_phi = float(radio_cfg.get("iq_phase_imbalance_deg", 3.0))
            epsilon = float(np.abs(rng.normal(0.0, sigma_eps)))
            phi_deg = float(rng.normal(0.0, sigma_phi))
            h_observed = apply_iq_imbalance(
                h_observed,
                epsilon=epsilon,
                phi_deg=phi_deg,
            )
            iq_meta["iq_gain_imbalance"] = epsilon
            iq_meta["iq_phase_imbalance_deg"] = phi_deg

        # ── ADC sampling phase offset (±0.5 bin uniform random) ──────────
        # Simulates the asynchronous nature of the ADC sampling clock relative
        # to the received signal waveform.  Even when the ADC samples at
        # exactly the Nyquist rate, the sampling *grid* is randomly shifted
        # relative to the continuous CIR — the sample may land before, on,
        # or after the true peak by up to half a bin.
        #
        # In the frequency domain this is a linear phase ramp:
        #   H'(f) = H(f) × exp(-j·2π·f·Δ)
        #
        # This is distinct from the absolute timing bias in
        # hardware/impairments.py (sync_bias, clock_bias, adc_phase_offset)
        # which shifts the entire time-axis.  Here we model the *relative*
        # misalignment between the discrete sampling grid and the analog
        # signal waveform — both effects coexist in real receivers.
        bin_period_s = 1.0 / bandwidth_hz
        sampling_offset_s = rng.uniform(-0.5 * bin_period_s, 0.5 * bin_period_s)
        if sampling_offset_s != 0.0:
            h_observed = h_observed * np.exp(-1j * 2.0 * np.pi * frequency_hz * sampling_offset_s)

        cir_clean_discrete_raw = np.abs(np.fft.ifft(h_clean))
        cir_observed_discrete_raw = np.abs(np.fft.ifft(h_observed))

        cir_observed_discrete_raw, cir_impair_meta = apply_cir_impairments(
            cir_observed_discrete_raw, self.config, rng, radio_cfg=radio_cfg,
        )

        cir_clean_cont_raw = np.abs(continuous_uwb(h_clean, factor))
        cir_observed_cont_raw = np.abs(continuous_uwb(h_observed, factor))
        # AGC gain and clipping are analog-domain — apply to continuous CIR too.
        agc_gain = float(cir_impair_meta.get("agc_gain_linear", 1.0))
        if agc_gain != 1.0:
            cir_observed_cont_raw = cir_observed_cont_raw * agc_gain
            if cir_impair_meta.get("agc_clip_enable", False):
                cir_observed_cont_raw = np.clip(cir_observed_cont_raw, 0.0, 1.0)
        cir_clean_discrete = normalize_observation(cir_clean_discrete_raw)
        cir_observed_discrete = normalize_with_reference(cir_observed_discrete_raw, cir_clean_discrete_raw)
        cir_clean_cont = normalize_with_reference(cir_clean_cont_raw, cir_clean_discrete_raw)
        cir_observed_cont = normalize_with_reference(cir_observed_cont_raw, cir_clean_discrete_raw)
        t_discrete_s = np.arange(num_bins, dtype=float) / bandwidth_hz
        t_cont_s = np.arange(num_bins * factor, dtype=float) / (bandwidth_hz * factor)

        return RadioObservation(
            protocol=self.protocol,
            t_discrete_s=t_discrete_s,
            t_cont_s=t_cont_s,
            frequency_hz=frequency_hz,
            h_clean=h_clean,
            h_observed=h_observed,
            cir_clean_discrete=cir_clean_discrete,
            cir_observed_discrete=cir_observed_discrete,
            cir_clean_cont=cir_clean_cont,
            cir_observed_cont=cir_observed_cont,
            metadata={
                "bandwidth_hz": bandwidth_hz,
                "num_bins": num_bins,
                "cir_bins": num_bins,
                "accumulation_count": accumulation_count,
                "accumulation_gain_db": round(accumulation_gain_db, 2),
                "window": window_name,
                "bin_period_s": bin_period_s,
                "sampling_bin_offset_s": sampling_offset_s,
                **observation_meta,
                **cir_impair_meta,
                **iq_meta,
            },
        )
