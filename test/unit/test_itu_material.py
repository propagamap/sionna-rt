#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import numpy as np

import drjit as dr
import mitsuba as mi
from sionna import rt
from sionna.rt import load_scene, ITURadioMaterial
from sionna.rt.radio_materials.itu import itu_material, ITU_MATERIALS_PROPERTIES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scene_with_material(mat):
    """Load a minimal scene and attach *mat* to the reflector object."""
    scene = load_scene(rt.scene.simple_reflector, merge_shapes=False)
    scene.objects["reflector"].radio_material = mat
    scene.remove("reflector-mat")
    return scene


# ---------------------------------------------------------------------------
# itu_material() function
# ---------------------------------------------------------------------------

class TestItuMaterialFunction:

    def test_in_range_returns_correct_values(self):
        """Concrete at 3 GHz should match the 1-100 GHz parameters exactly."""
        f_hz = 3e9
        f_ghz = 3.0
        eta_r, sigma = itu_material("concrete", mi.Float(f_hz))

        a, b, c, d = 5.24, 0.0, 0.0462, 0.7822
        expected_eta_r = a * (f_ghz ** b)
        expected_sigma = c * (f_ghz ** d)

        assert dr.allclose(eta_r, mi.Float(expected_eta_r), rtol=1e-5)
        assert dr.allclose(sigma, mi.Float(expected_sigma), rtol=1e-5)

    def test_metal_in_range(self):
        """Metal conductivity is 1e7 independent of frequency."""
        eta_r, sigma = itu_material("metal", mi.Float(5e9))
        assert dr.allclose(eta_r, mi.Float(1.0))
        assert dr.allclose(sigma, mi.Float(1e7))

    def test_all_materials_instantiate_in_range(self):
        """All ITU materials evaluate without error at a representative in-range frequency."""
        representative_freqs = {
            "vacuum":            0.5e9,
            "concrete":          5e9,
            "brick":             5e9,
            "plasterboard":      5e9,
            "wood":              5e9,
            "glass":             5e9,
            "clear_acrylic":     200e9,
            "ceiling_board":     5e9,
            "chipboard":         5e9,
            "plywood":           5e9,
            "marble":            5e9,
            "floorboard":        75e9,
            "vinyl_tile":        5e9,
            "carpet_tile":       5e9,
            "asphalt_concrete":  5e9,
            "metal":             5e9,
            "very_dry_ground":   5e9,
            "medium_dry_ground": 5e9,
            "wet_ground":        5e9,
        }
        for name, f_hz in representative_freqs.items():
            eta_r, sigma = itu_material(name, mi.Float(f_hz))
            assert float(eta_r[0]) > 0.0, f"{name}: eta_r must be positive"
            assert float(sigma[0]) >= 0.0, f"{name}: sigma must be non-negative"

    def test_invalid_material_raises(self):
        """Requesting an unknown material name must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown ITU material"):
            itu_material("unobtanium", mi.Float(5e9))

    # --- Out-of-range behaviour -------------------------------------------

    def test_out_of_range_nonstrict_falls_back_to_nearest(self):
        """For non-strict materials, frequencies outside all defined ranges
        should silently use the parameters of the closest range."""
        # vinyl_tile is only defined for 1-40 GHz.  At 50 GHz (above range)
        # the function must not raise and must return the 1-40 GHz params.
        f_hz = 50e9
        f_ghz = 50.0
        eta_r, sigma = itu_material("vinyl_tile", mi.Float(f_hz))

        a, b, c, d = ITU_MATERIALS_PROPERTIES["vinyl_tile"][(1.0, 40.0)]
        expected_eta_r = a * (f_ghz ** b)
        expected_sigma = c * (f_ghz ** d)

        assert dr.allclose(eta_r, mi.Float(expected_eta_r), rtol=1e-5)
        assert dr.allclose(sigma, mi.Float(expected_sigma), rtol=1e-5)

    def test_out_of_range_nonstrict_below_range(self):
        """Below the defined range also falls back to the nearest range."""
        # carpet_tile: 1-40 GHz.  At 0.1 GHz the nearest boundary is 1 GHz.
        f_hz = 0.1e9
        f_ghz = 0.1
        eta_r, sigma = itu_material("carpet_tile", mi.Float(f_hz))

        a, b, c, d = ITU_MATERIALS_PROPERTIES["carpet_tile"][(1.0, 40.0)]
        expected_eta_r = a * (f_ghz ** b)
        expected_sigma = c * (f_ghz ** d)

        assert dr.allclose(eta_r, mi.Float(expected_eta_r), rtol=1e-5)
        assert dr.allclose(sigma, mi.Float(expected_sigma), rtol=1e-5)

    @pytest.mark.parametrize("name", ["very_dry_ground", "medium_dry_ground", "wet_ground"])
    def test_strict_range_materials_raise_out_of_range(self, name):
        """Ground materials are strict: frequencies outside 1-10 GHz must raise ValueError."""
        with pytest.raises(ValueError, match="not defined for this frequency"):
            itu_material(name, mi.Float(20e9))  # 20 GHz > 10 GHz

    @pytest.mark.parametrize("name", ["very_dry_ground", "medium_dry_ground", "wet_ground"])
    def test_strict_range_materials_ok_in_range(self, name):
        """Ground materials do not raise when the frequency is inside 1-10 GHz."""
        eta_r, sigma = itu_material(name, mi.Float(5e9))
        assert float(eta_r[0]) > 0.0
        assert float(sigma[0]) >= 0.0

    # --- forced_freq_range ---------------------------------------------------

    def test_forced_freq_range_selects_correct_params(self):
        """With forced_freq_range the specified range's params are used when
        the frequency falls inside that range."""
        # Plasterboard has three ranges; force (110, 330) at 150 GHz
        f_hz = 150e9
        f_ghz = 150.0
        forced = (110., 330.)
        eta_r, sigma = itu_material("plasterboard", mi.Float(f_hz), forced_freq_range=forced)

        a, b, c, d = ITU_MATERIALS_PROPERTIES["plasterboard"][forced]
        expected_eta_r = a * (f_ghz ** b)
        expected_sigma = c * (f_ghz ** d)

        assert dr.allclose(eta_r, mi.Float(expected_eta_r), rtol=1e-5)
        assert dr.allclose(sigma, mi.Float(expected_sigma), rtol=1e-5)

    def test_forced_freq_range_ignored_when_freq_outside_forced_range(self):
        """When the current frequency is outside the forced range the function
        falls through to the standard selection logic."""
        # Force (110, 330) for plasterboard but evaluate at 5 GHz (1-100 range).
        f_hz = 5e9
        f_ghz = 5.0
        forced = (110., 330.)

        eta_r_forced, sigma_forced = itu_material(
            "plasterboard", mi.Float(f_hz), forced_freq_range=forced)
        eta_r_auto, sigma_auto = itu_material(
            "plasterboard", mi.Float(f_hz))

        # Both should match the 1-100 GHz parameters.
        assert dr.allclose(eta_r_forced, eta_r_auto, rtol=1e-5)
        assert dr.allclose(sigma_forced, sigma_auto, rtol=1e-5)

    def test_forced_freq_range_nonexistent_key_ignored(self):
        """A forced range that is not a key in the material's props is silently
        ignored and the standard selection logic is used."""
        f_hz = 5e9
        nonexistent = (999., 1000.)
        eta_r_forced, sigma_forced = itu_material(
            "concrete", mi.Float(f_hz), forced_freq_range=nonexistent)
        eta_r_auto, sigma_auto = itu_material("concrete", mi.Float(f_hz))

        assert dr.allclose(eta_r_forced, eta_r_auto, rtol=1e-5)
        assert dr.allclose(sigma_forced, sigma_auto, rtol=1e-5)


# ---------------------------------------------------------------------------
# ITURadioMaterial class
# ---------------------------------------------------------------------------

class TestITURadioMaterial:

    def test_itu_type_property(self):
        mat = ITURadioMaterial("mat-wood", "wood", 0.02)
        assert mat.itu_type == "wood"

    def test_invalid_itu_type_raises(self):
        with pytest.raises(ValueError, match="Invalid ITU material type"):
            ITURadioMaterial("bad-mat", "not_a_material", 0.01)

    def test_all_itu_types_instantiate(self):
        """Every material listed in ITU_MATERIALS_PROPERTIES can be instantiated."""
        for i, name in enumerate(ITU_MATERIALS_PROPERTIES):
            mat = ITURadioMaterial(f"mat-{i}", name, 0.01)
            assert mat.itu_type == name

    def test_default_color_matches_itu_material_colors(self):
        """When no color is given, the material uses ITU_MATERIAL_COLORS."""
        for name, expected_color in ITURadioMaterial.ITU_MATERIAL_COLORS.items():
            mat = ITURadioMaterial(f"c-{name}", name, 0.01)
            assert dr.allclose(mat.color, mi.ScalarColor3f(expected_color), rtol=1e-5), \
                f"Color mismatch for {name}"

    def test_custom_color_overrides_default(self):
        custom = (0.1, 0.2, 0.3)
        mat = ITURadioMaterial("mat-custom-color", "concrete", 0.1, color=custom)
        assert dr.allclose(mat.color, mi.ScalarColor3f(custom), rtol=1e-5)

    def test_forced_range_property_getter(self):
        """forced_range getter returns the value set at construction."""
        forced = (110., 330.)
        mat = ITURadioMaterial("mat-forced", "concrete", 0.1,
                               forced_freq_range=forced)
        assert mat.forced_range == forced

    def test_forced_range_property_none_by_default(self):
        mat = ITURadioMaterial("mat-no-forced", "concrete", 0.1)
        assert mat.forced_range is None

    def test_forced_range_setter_triggers_frequency_update(self):
        """Setting forced_range on a material attached to a scene should update
        eta_r and sigma to reflect the newly forced parameters."""
        # Plasterboard at 150 GHz has two overlapping ranges: (110,330) and (100,400).
        scene = load_scene(rt.scene.simple_reflector, merge_shapes=False)
        scene.frequency = 150e9

        mat = ITURadioMaterial("mat-plaster", "plasterboard", 0.1)
        scene.add(mat)
        scene.objects["reflector"].radio_material = mat
        scene.remove("reflector-mat")

        # Force (110, 330) range and capture resulting values
        mat.forced_range = (110., 330.)
        eta_r_forced = float(mat.relative_permittivity[0])
        sigma_forced = float(mat.conductivity[0])

        # Compute expected values from the forced range
        f_ghz = 150.0
        a, b, c, d = ITU_MATERIALS_PROPERTIES["plasterboard"][(110., 330.)]
        assert abs(eta_r_forced - a * (f_ghz ** b)) < 1e-4
        assert abs(sigma_forced - c * (f_ghz ** d)) < 1e-4

    def test_material_evaluates_on_scene_frequency_change(self):
        """Changing scene.frequency should update the ITU material properties."""
        scene = load_scene(rt.scene.simple_reflector, merge_shapes=False)

        mat = ITURadioMaterial("mat-concrete", "concrete", 0.1)
        scene.add(mat)
        scene.objects["reflector"].radio_material = mat
        scene.remove("reflector-mat")

        scene.frequency = 2e9
        eta_r_2ghz = float(mat.relative_permittivity[0])
        sigma_2ghz = float(mat.conductivity[0])

        scene.frequency = 10e9
        eta_r_10ghz = float(mat.relative_permittivity[0])
        sigma_10ghz = float(mat.conductivity[0])

        # Concrete: b=0 so eta_r is constant, but sigma changes with frequency.
        assert abs(eta_r_2ghz - 5.24) < 1e-4
        assert abs(eta_r_10ghz - 5.24) < 1e-4
        assert sigma_2ghz != sigma_10ghz

    def test_to_string(self):
        """to_string() must not raise and must mention the ITU type."""
        mat = ITURadioMaterial("mat-str", "brick", 0.05)
        s = mat.to_string()
        assert "brick" in s
