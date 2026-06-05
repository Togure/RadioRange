from __future__ import annotations

from typing import Any

import numpy as np

from core.models import ChannelTruth, RadioObservation
from core.transceivers.base_radio import BaseRadio, apply_cir_impairments, normalize_observation, normalize_with_reference, observe_frequency_response
from core.transceivers.common import continuous_ofdm, path_response


def _sionna_ofdm_channel(
    fft_size: int,
    subcarrier_spacing_hz: float,
    a_paths: np.ndarray,
    tau_paths_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """3GPP-standard OFDM subcarrier grid + frequency response via Sionna.

    Sionna returns ``H(f)`` in *natural* order (most negative frequency
    first, DC in the middle).  We convert to NumPy's ``fftfreq`` order
    (DC first, then positive, then negative) so that ``np.fft.ifft`` and
    ``continuous_ofdm`` receive the array layout they expect.

    Falls back to manual ``fftfreq + path_response`` when Sionna is not
    installed or the call fails.
    """
    try:
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import torch

        from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies

        n_paths = len(a_paths)
        freqs = subcarrier_frequencies(fft_size, subcarrier_spacing_hz)

        a_tensor = torch.reshape(
            torch.tensor(a_paths, dtype=torch.complex64),
            [1, 1, 1, 1, 1, n_paths, 1],
        )
        tau_tensor = torch.reshape(
            torch.tensor(tau_paths_s, dtype=torch.float32),
            [1, 1, 1, n_paths],
        )

        h_tensor = cir_to_ofdm_channel(freqs, a_tensor, tau_tensor)

        # cir_to_ofdm_channel returns [batch, rx, rx_ant, tx, tx_ant, time, subcarriers].
        # Extract the single-link response and convert to fftfreq order.
        h_natural = h_tensor[0, 0, 0, 0, 0, 0, :].numpy()
        frequency_hz = np.fft.ifftshift(freqs.numpy()).astype(np.float64)
        h_frequency = np.fft.ifftshift(h_natural)
        return frequency_hz, h_frequency

    except Exception:
        sampling_frequency_hz = fft_size * subcarrier_spacing_hz
        frequency_hz = np.fft.fftfreq(fft_size, d=1.0 / sampling_frequency_hz)
        h_frequency = path_response(frequency_hz, a_paths, tau_paths_s)
        return frequency_hz, h_frequency


class OfdmRadio(BaseRadio):
    protocol = "ofdm"
    config_key = ""

    def observe(self, truth: ChannelTruth, rng: np.random.Generator) -> RadioObservation:
        radio_cfg = self.config["radios"][self.config_key]

        fft_size = int(radio_cfg["fft_size"])
        subcarrier_spacing_hz = float(radio_cfg["subcarrier_spacing_hz"])
        sampling_frequency_hz = fft_size * subcarrier_spacing_hz
        factor = int(radio_cfg.get("interpolation_factor", 10))

        frequency_hz, h_clean = _sionna_ofdm_channel(
            fft_size, subcarrier_spacing_hz,
            truth.a_paths, truth.tau_paths_s,
        )

        h_observed, observation_meta = observe_frequency_response(
            h_clean,
            frequency_hz,
            radio_cfg,
            rng,
            enable_mask_and_pilots=True,
            impair_cfg=self.config.get("impairments", {}),
        )

        iq_meta: dict[str, Any] = {}
        if bool(radio_cfg.get("enable_iq_imbalance", False)
                or self.config.get("impairments", {}).get("enable_iq_imbalance", False)):
            from hardware.iq_imbalance import apply_iq_imbalance

            sigma_eps = float(radio_cfg.get("iq_gain_imbalance", 0.015))
            sigma_phi = float(radio_cfg.get("iq_phase_imbalance_deg", 1.5))
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
        # exactly the sampling rate, the sampling *grid* is randomly shifted
        # relative to the continuous signal — the sample may land before, on,
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
        bin_period_s = 1.0 / sampling_frequency_hz
        sampling_offset_s = rng.uniform(-0.5 * bin_period_s, 0.5 * bin_period_s)
        if sampling_offset_s != 0.0:
            h_observed = h_observed * np.exp(-1j * 2.0 * np.pi * frequency_hz * sampling_offset_s)

        cir_clean_discrete_raw = np.abs(np.fft.ifft(h_clean))
        cir_observed_discrete_raw = np.abs(np.fft.ifft(h_observed))

        cir_observed_discrete_raw, cir_impair_meta = apply_cir_impairments(
            cir_observed_discrete_raw, self.config, rng, radio_cfg=radio_cfg,
        )

        cir_clean_cont_raw = np.abs(continuous_ofdm(h_clean, factor))
        cir_observed_cont_raw = np.abs(continuous_ofdm(h_observed, factor))
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
        t_discrete_s = np.arange(fft_size, dtype=float) / sampling_frequency_hz
        t_cont_s = np.arange(fft_size * factor, dtype=float) / (sampling_frequency_hz * factor)

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
                "fft_size": fft_size,
                "subcarrier_spacing_hz": subcarrier_spacing_hz,
                "sampling_frequency_hz": sampling_frequency_hz,
                "bin_period_s": bin_period_s,
                "sampling_bin_offset_s": sampling_offset_s,
                **observation_meta,
                **cir_impair_meta,
                **iq_meta,
            },
        )


class WifiOfdmRadio(OfdmRadio):
    protocol = "wifi"
    config_key = "wifi"


class FiveGNrRadio(OfdmRadio):
    protocol = "fiveg"
    config_key = "fiveg"
