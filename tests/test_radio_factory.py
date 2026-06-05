"""Tests for utils.radio_factory — radio creation and factory functions."""

import numpy as np
import pytest

from utils.radio_factory import (
    RADIO_CLASSES,
    RADIO_DEFAULTS,
    DEFAULT_CARRIER_FREQ_HZ,
    create_radio,
    create_all_radios,
    build_radios_from_config,
)
from core.transceivers import UwbImpulseRadio, WifiOfdmRadio, FiveGNrRadio


class TestCreateRadio:
    """Tests for create_radio() — the single-protocol factory."""

    def test_uwb_defaults(self):
        radio = create_radio("uwb")
        assert isinstance(radio, UwbImpulseRadio)
        assert radio.protocol == "uwb"

    def test_wifi_defaults(self):
        radio = create_radio("wifi")
        assert isinstance(radio, WifiOfdmRadio)
        assert radio.protocol == "wifi"

    def test_fiveg_defaults(self):
        radio = create_radio("fiveg")
        assert isinstance(radio, FiveGNrRadio)
        assert radio.protocol == "fiveg"

    def test_unknown_protocol_raises(self):
        with pytest.raises(ValueError, match="Unknown protocol"):
            create_radio("bluetooth")

    def test_custom_carrier_frequency(self):
        radio = create_radio("uwb", carrier_frequency_hz=6.5e9)
        assert radio.config["radios"]["uwb"]["carrier_frequency_hz"] == 6.5e9

    def test_override_bandwidth(self):
        radio = create_radio("wifi", overrides={"bandwidth_hz": 80e6})
        assert radio.config["radios"]["wifi"]["bandwidth_hz"] == 80e6

    def test_legacy_num_bins_key_normalized(self):
        """Old scripts use 'num_bins' — it should map to 'cir_bins'."""
        radio = create_radio("uwb", overrides={"num_bins": 512})
        assert radio.config["radios"]["uwb"]["cir_bins"] == 512

    def test_legacy_num_bins_does_not_override_cir_bins(self):
        """When both keys provided, cir_bins takes precedence (updated later)."""
        radio = create_radio("uwb", overrides={"num_bins": 256, "cir_bins": 768})
        assert radio.config["radios"]["uwb"]["cir_bins"] == 768

    def test_impairments_passed_to_config(self):
        imp = {"enable_agc": True, "enable_sfo": False}
        radio = create_radio("uwb", impairments=imp)
        assert radio.config["impairments"]["enable_agc"] is True

    def test_no_impairments_defaults_to_empty_dict(self):
        radio = create_radio("uwb")
        assert radio.config["impairments"] == {}


class TestCreateAllRadios:
    """Tests for create_all_radios() — multi-protocol factory."""

    def test_all_three_by_default(self):
        radios = create_all_radios()
        assert set(radios.keys()) == {"uwb", "wifi", "fiveg"}
        assert isinstance(radios["uwb"], UwbImpulseRadio)
        assert isinstance(radios["wifi"], WifiOfdmRadio)
        assert isinstance(radios["fiveg"], FiveGNrRadio)

    def test_subset_of_protocols(self):
        radios = create_all_radios(protocols=["uwb"])
        assert list(radios.keys()) == ["uwb"]

    def test_custom_carrier_freqs(self):
        radios = create_all_radios(
            carrier_freqs={"uwb": 7.0e9, "wifi": 5.5e9}
        )
        assert radios["uwb"].config["radios"]["uwb"]["carrier_frequency_hz"] == 7.0e9
        assert radios["wifi"].config["radios"]["wifi"]["carrier_frequency_hz"] == 5.5e9

    def test_per_protocol_overrides(self):
        radios = create_all_radios(
            overrides={"uwb": {"snr_db": 40}, "wifi": {"snr_db": 35}}
        )
        assert radios["uwb"].config["radios"]["uwb"]["snr_db"] == 40
        assert radios["wifi"].config["radios"]["wifi"]["snr_db"] == 35

    def test_impairments_shared_across_all(self):
        imp = {"enable_cfo": True}
        radios = create_all_radios(impairments=imp)
        for proto in ["uwb", "wifi", "fiveg"]:
            assert radios[proto].config["impairments"]["enable_cfo"] is True


class TestBuildRadiosFromConfig:
    """Tests for build_radios_from_config() — config-driven multi-radio builder."""

    def test_builds_enabled_radios(self):
        config = {
            "radios": {
                "uwb": {"enabled": True, "carrier_frequency_hz": 8e9},
                "wifi": {"enabled": True, "carrier_frequency_hz": 5.2e9},
                "fiveg": {"enabled": False},
            }
        }
        radios = build_radios_from_config(config)
        protocols = [r.protocol for r in radios]
        assert "uwb" in protocols
        assert "wifi" in protocols
        assert "fiveg" not in protocols

    def test_impairments_from_config(self):
        config = {
            "radios": {
                "uwb": {"enabled": True},
            },
            "impairments": {"enable_agc": True},
        }
        radios = build_radios_from_config(config, protocols=["uwb"])
        assert radios[0].config["impairments"]["enable_agc"] is True

    def test_observation_model_override(self):
        config = {
            "radios": {
                "uwb": {"enabled": True, "observation_model": "explicit"},
            },
        }
        radios = build_radios_from_config(
            config, protocols=["uwb"], observation_model="snr"
        )
        # The override should force observation_model to "snr"
        assert radios[0].config["radios"]["uwb"]["observation_model"] == "snr"


class TestDefaultsConsistency:
    """Verify default tables are internally consistent."""

    def test_all_protocols_have_radio_class(self):
        for proto in RADIO_DEFAULTS:
            assert proto in RADIO_CLASSES, f"{proto} missing from RADIO_CLASSES"

    def test_all_protocols_have_carrier_frequency(self):
        for proto in RADIO_CLASSES:
            assert proto in DEFAULT_CARRIER_FREQ_HZ, f"{proto} missing from DEFAULT_CARRIER_FREQ_HZ"

    def test_all_radios_have_minimal_defaults(self):
        required = {"interpolation_factor", "adc_bits", "observation_model"}
        # Note: UWB has explicit bandwidth_hz; WiFi/5G infer it from
        # subcarrier_spacing_hz * fft_size, so bandwidth_hz is not required.
        for proto, defaults in RADIO_DEFAULTS.items():
            missing = required - set(defaults.keys())
            assert not missing, f"{proto} missing defaults: {missing}"
