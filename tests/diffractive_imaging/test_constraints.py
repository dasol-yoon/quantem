"""Tests for the ptychography constraint dataclass API."""

import warnings

import numpy as np
import pytest
import torch

from quantem.core.datastructures import Dataset4dstem
from quantem.diffractive_imaging import (
    DetectorPixelated,
    ObjectPixelated,
    ProbePixelated,
    PtychoDatasetConstraintParams,
    Ptychography,
    PtychographyDatasetRaster,
    PtychoObjConstraintParams,
    PtychoProbeConstraintParams,
)

N_SCAN = 8
N_DET = 16
PROBE_ENERGY = 80e3
PROBE_SEMIANGLE = 20
PROBE_DEFOCUS = 100


@pytest.fixture
def ptycho():
    rng = np.random.default_rng(42)
    array = rng.random((N_SCAN, N_SCAN, N_DET, N_DET)).astype(np.float32)
    dset = Dataset4dstem.from_array(
        array,
        name="test",
        sampling=[1.0, 1.0, 0.05, 0.05],
        units=["A", "A", "A^-1", "A^-1"],
    )
    pdset = PtychographyDatasetRaster.from_dataset4dstem(dset)
    pdset.preprocess(com_fit_function="constant", plot_rotation=False, plot_com=False)
    obj = ObjectPixelated.from_uniform(obj_type="pure_phase", num_slices=1)
    probe = ProbePixelated.from_params(
        probe_params={
            "energy": PROBE_ENERGY,
            "defocus": PROBE_DEFOCUS,
            "semiangle_cutoff": PROBE_SEMIANGLE,
        }
    )
    p = Ptychography.from_models(
        dset=pdset,
        obj_model=obj,
        probe_model=probe,
        detector_model=DetectorPixelated(),
        verbose=False,
        rng=42,
    )
    p.preprocess(obj_padding_px=(4, 4))
    return p


# --- parse_dict tests ---------------------------------------------------------


class TestParseDict:
    def test_object_raster_by_name(self):
        c = PtychoObjConstraintParams.parse_dict({"name": "raster", "tv_weight_z": 5.0})
        assert isinstance(c, PtychoObjConstraintParams.Raster)
        assert c.tv_weight_z == 5.0
        assert c.positivity is True  # default preserved

    def test_object_inr_by_type(self):
        c = PtychoObjConstraintParams.parse_dict({"type": "inr"})
        assert isinstance(c, PtychoObjConstraintParams.INR)

    def test_object_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown object constraint type"):
            PtychoObjConstraintParams.parse_dict({"name": "nope"})

    def test_object_missing_name_raises(self):
        with pytest.raises(ValueError, match="Must provide either 'name' or 'type'"):
            PtychoObjConstraintParams.parse_dict({"tv_weight_z": 5.0})

    def test_probe_raster_with_fields(self):
        c = PtychoProbeConstraintParams.parse_dict(
            {"name": "raster", "center_probe": True, "tv_weight": 0.1}
        )
        assert isinstance(c, PtychoProbeConstraintParams.Raster)
        assert c.center_probe is True
        assert c.tv_weight == 0.1

    def test_dataset_raster_default(self):
        c = PtychoDatasetConstraintParams.parse_dict({"name": "raster"})
        assert isinstance(c, PtychoDatasetConstraintParams.Raster)
        assert c.clip_scan_positions is True  # default preserved


# --- Constraint typo catching -------------------------------------------------


class TestTypoCatching:
    def test_setting_unknown_field_via_dict_raises(self, ptycho):
        with pytest.raises(KeyError, match="Invalid constraint key"):
            ptycho.obj_model.constraints = {"not_a_real_field": True}

    def test_add_constraint_unknown_key_raises(self, ptycho):
        with pytest.raises(KeyError, match="Invalid constraint key"):
            ptycho.obj_model.add_constraint("not_a_real_field", True)


# --- Round-trip: pass dataclass via reconstruct(), read back through getter ---


class TestRoundtrip:
    def test_obj_constraints_dataclass(self, ptycho):
        obj_c = PtychoObjConstraintParams.Raster(tv_weight_z=2.5, identical_slices=True)
        ptycho.constraints = {"object": obj_c}
        assert ptycho.obj_model.constraints is obj_c
        assert ptycho.obj_model.constraints.tv_weight_z == 2.5
        assert ptycho.obj_model.constraints.identical_slices is True

    def test_probe_constraints_dataclass(self, ptycho):
        probe_c = PtychoProbeConstraintParams.Raster(center_probe=True, tv_weight=0.05)
        ptycho.constraints = {"probe": probe_c}
        assert ptycho.probe_model.constraints is probe_c

    def test_dataset_constraints_dataclass(self, ptycho):
        dset_c = PtychoDatasetConstraintParams.Raster(descan_tv_weight=0.01)
        ptycho.constraints = {"dataset": dset_c}
        assert ptycho.dset.constraints is dset_c

    def test_dict_form_still_works(self, ptycho):
        """Backward compatibility: nested-dict form sets individual fields."""
        ptycho.constraints = {
            "object": {"tv_weight_z": 3.0, "positivity": False},
            "probe": {"tv_weight": 0.02},
        }
        assert ptycho.obj_model.constraints.tv_weight_z == 3.0
        assert ptycho.obj_model.constraints.positivity is False
        assert ptycho.probe_model.constraints.tv_weight == 0.02


# --- Reconstruct() with constraints= ------------------------------------------


class TestReconstructKwargs:
    def test_dataclass_leaf_applied(self, ptycho):
        from quantem.core.ml import OptimizerParams

        obj_c = PtychoObjConstraintParams.Raster(tv_weight_z=1.5)
        ptycho.reconstruct(
            num_iters=1,
            reset=True,
            optimizer_params={"object": OptimizerParams.Adam(lr=1e-2)},
            constraints={"object": obj_c},
            batch_size=4,
            device="cpu",
        )
        assert ptycho.obj_model.constraints.tv_weight_z == 1.5

    def test_dict_leaf_partial_update(self, ptycho):
        from quantem.core.ml import OptimizerParams

        ptycho.reconstruct(
            num_iters=1,
            reset=True,
            optimizer_params={"object": OptimizerParams.Adam(lr=1e-2)},
            constraints={"object": {"surface_zero_weight": 0.7}},
            batch_size=4,
            device="cpu",
        )
        assert ptycho.obj_model.constraints.surface_zero_weight == 0.7
        # other fields keep their defaults
        assert ptycho.obj_model.constraints.positivity is True

    def test_mixed_dataclass_and_dict_leaves(self, ptycho):
        from quantem.core.ml import OptimizerParams

        ptycho.reconstruct(
            num_iters=1,
            reset=True,
            optimizer_params={"object": OptimizerParams.Adam(lr=1e-2)},
            constraints={
                "object": PtychoObjConstraintParams.Raster(tv_weight_xy=0.4),
                "probe": {"center_probe": True},
            },
            batch_size=4,
            device="cpu",
        )
        assert ptycho.obj_model.constraints.tv_weight_xy == 0.4
        assert ptycho.probe_model.constraints.center_probe is True


# --- Real-valued pure_phase representation -----------------------------------


class TestPurePhaseRealValued:
    def test_pure_phase_pixelated_obj_is_real(self):
        obj = ObjectPixelated.from_uniform(obj_type="pure_phase", num_slices=1)
        obj._initialize_obj((1, 16, 16), sampling=(0.1, 0.1))
        assert not obj._obj.is_complex(), f"pure_phase _obj should be real, got {obj._obj.dtype}"

    def test_complex_pixelated_obj_is_complex(self):
        obj = ObjectPixelated.from_uniform(obj_type="complex", num_slices=1)
        obj._initialize_obj((1, 16, 16), sampling=(0.1, 0.1))
        assert obj._obj.is_complex()

    def test_potential_pixelated_obj_is_real(self):
        obj = ObjectPixelated.from_uniform(obj_type="potential", num_slices=1)
        obj._initialize_obj((1, 16, 16), sampling=(0.1, 0.1))
        assert not obj._obj.is_complex()

    def test_pure_phase_tv_emits_no_phase_warning(self):
        obj = ObjectPixelated.from_uniform(obj_type="pure_phase", num_slices=1)
        obj._initialize_obj((1, 16, 16), sampling=(0.1, 0.1))
        obj.constraints.tv_weight_xy = 0.1
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            obj.get_tv_loss(obj._obj)
        phase_warnings = [w for w in caught if "phase wrapping" in str(w.message)]
        assert not phase_warnings, (
            f"pure_phase should not emit phase-wrap warning, got {phase_warnings}"
        )

    def test_complex_tv_still_emits_phase_warning(self):
        obj = ObjectPixelated.from_uniform(obj_type="complex", num_slices=1)
        obj._initialize_obj((1, 16, 16), sampling=(0.1, 0.1))
        obj.constraints.tv_weight_xy = 0.1
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            obj.get_tv_loss(obj._obj)
        assert any("phase wrapping" in str(w.message) for w in caught), (
            "complex obj_type should still emit phase-wrap warning"
        )

    def test_pure_phase_apply_hard_constraints_stays_real(self):
        obj = ObjectPixelated.from_uniform(obj_type="pure_phase", num_slices=1)
        obj._initialize_obj((1, 16, 16), sampling=(0.1, 0.1))
        out = obj.apply_hard_constraints(obj._obj)
        assert not out.is_complex()


# --- Depth / lateral regularizer filters (kz_filter, kr_filter, gaussian_blur_z) ---


class TestDepthLateralFilters:
    def _make_obj(self, obj_type, num_slices=6) -> ObjectPixelated:
        obj = ObjectPixelated.from_uniform(
            obj_type=obj_type, num_slices=num_slices, slice_thicknesses=2.0
        )
        obj._initialize_obj((num_slices, 16, 16), sampling=(0.1, 0.1))
        return obj

    def _random_tensor(self, obj_type, num_slices, rng):
        if obj_type == "complex":
            amp = 0.95 + 0.1 * torch.as_tensor(
                rng.random((num_slices, 16, 16)), dtype=torch.float32
            )
            phase = 0.1 * torch.as_tensor(
                rng.standard_normal((num_slices, 16, 16)), dtype=torch.float32
            )
            return amp * torch.exp(1j * phase)
        return 0.1 * torch.as_tensor(rng.standard_normal((num_slices, 16, 16)), dtype=torch.float32)

    @pytest.mark.parametrize("obj_type", ["potential", "pure_phase", "complex"])
    @pytest.mark.parametrize(
        "method,kwargs",
        [
            ("kz_filter", {"beta": 0.5, "alpha": 1.0}),
            ("kr_filter", {"radius": 0.6, "width": 0.05}),
            ("gaussian_blur_z", {"sigma": 1.0}),
        ],
    )
    def test_filter_preserves_shape_and_dtype(self, obj_type, method, kwargs):
        rng = np.random.default_rng(0)
        obj = self._make_obj(obj_type)
        raw = self._random_tensor(obj_type, 6, rng)
        out = getattr(obj, method)(raw, **kwargs)
        assert out.shape == raw.shape
        assert out.is_complex() == raw.is_complex()

    @pytest.mark.parametrize("method,kwargs", [
        ("kz_filter", {"beta": 0.5, "alpha": 1.0}),
        ("gaussian_blur_z", {"sigma": 1.0}),
    ])
    def test_depth_filters_smooth_the_slice_axis(self, method, kwargs):
        # Oscillating z-profile, uniform laterally, on a real (potential) object.
        obj = self._make_obj("potential", num_slices=8)
        z = torch.arange(8, dtype=torch.float32)
        profile = torch.sin(z * 2.0)
        raw = profile[:, None, None].expand(8, 16, 16).clone()
        out = getattr(obj, method)(raw, **kwargs)
        assert out.var(dim=0).mean() < raw.var(dim=0).mean()

    def test_gaussian_blur_z_leaves_lateral_dims_untouched(self):
        obj = self._make_obj("potential", num_slices=8)
        z = torch.arange(8, dtype=torch.float32)
        profile = torch.sin(z * 2.0)
        raw = profile[:, None, None].expand(8, 16, 16).clone()
        out = obj.gaussian_blur_z(raw, sigma=1.0)
        assert torch.allclose(out[0], out[0, 0, 0] * torch.ones(16, 16))

    def test_kz_filter_and_z_blur_wired_as_alternatives(self):
        # Exercise the apply_hard_constraints wiring end-to-end (num_slices > 1).
        obj = self._make_obj("potential", num_slices=6)
        obj.constraints.kz_filter_beta = 0.5
        out = obj.apply_hard_constraints(obj._obj)
        assert out.shape == obj._obj.shape
        obj.constraints.kz_filter_beta = None
        obj.constraints.z_blur_sigma = 1.0
        out2 = obj.apply_hard_constraints(obj._obj)
        assert out2.shape == obj._obj.shape

    def test_kr_filter_via_constraints_does_not_require_multislice(self):
        obj = self._make_obj("pure_phase", num_slices=1)
        obj.constraints.kr_filter_radius = 0.6
        out = obj.apply_hard_constraints(obj._obj)
        assert out.shape == obj._obj.shape

    def test_depth_filters_combine_with_existing_lateral_filters(self):
        # kz_filter_beta + q_lowpass shouldn't crash when both active.
        obj = self._make_obj("potential", num_slices=6)
        obj.constraints.kz_filter_beta = 0.5
        obj.constraints.q_lowpass = 0.3
        out = obj.apply_hard_constraints(obj._obj)
        assert out.shape == obj._obj.shape

    def test_kz_filter_matches_ptyrad_transfer_function(self):
        """kz_filter must reproduce fold_slice/ptyrad's arctan filter exactly.

        The transfer function is recovered by filtering a delta at the origin: since
        fftn(delta) == 1 everywhere, fftn(kz_filter(delta)) == Wa. Compared against an
        independent scalar reference with the 1e-3 regularizer INSIDE the sqrt, as in
        fold_slice's regulation_multilayers.m and ptyrad's constraints.kz_filter.
        """
        import math

        beta, alpha, n = 0.5, 1.0, 8
        obj = self._make_obj("potential", num_slices=n)
        delta = torch.zeros(n, n, n)
        delta[0, 0, 0] = 1.0
        wa = torch.fft.fftn(obj.kz_filter(delta, beta=beta, alpha=alpha))
        # Wa is real and even, so the filtered delta transforms back to a real spectrum.
        assert wa.imag.abs().max() < 1e-6
        wa = wa.real

        def reference(kz, kr2):
            arg = (beta * abs(kz) / math.sqrt(kr2 + 1e-3)) ** 2
            return (1.0 - math.atan(arg) / (math.pi / 2)) * math.exp(-alpha * kr2)

        freq = torch.fft.fftfreq(n).tolist()
        for iz, iy, ix in [(0, 0, 0), (1, 0, 0), (2, 1, 0), (4, 0, 0), (0, 2, 2), (3, 1, 1)]:
            kr2 = freq[ix] ** 2 + freq[iy] ** 2
            assert wa[iz, iy, ix].item() == pytest.approx(reference(freq[iz], kr2), abs=1e-5)

        # The kr = 0 line is the lateral mean of each slice, i.e. the object's depth
        # profile. Regression guard: with the epsilon misplaced outside the sqrt this
        # value collapses to 1.6e-4 and the depth profile is forced flat.
        assert wa[1, 0, 0].item() == pytest.approx(0.159548, abs=1e-5)
        assert wa[0, 0, 0].item() == pytest.approx(1.0, abs=1e-6)  # DC preserved


# --- Constraint default isolation ---------------------------------------------


class TestConstraintDefaultsIsolation:
    def test_reset_recon_does_not_mutate_class_defaults(self, ptycho):
        """``reset_recon`` must hand out a *copy* of ``DEFAULT_CONSTRAINTS``.

        The constraints setter binds a ``Constraints`` instance by reference, so if
        ``reset_recon`` assigned the class-level singleton directly, the next partial-dict
        update would ``setattr`` onto that singleton and silently change the defaults for
        every subsequent model in the process -- e.g. a parameter sweep's later "baseline"
        runs would inherit an earlier run's constraints.
        """
        from quantem.diffractive_imaging.object_models import ObjectConstraints

        pristine = PtychoObjConstraintParams.Raster()
        assert ObjectConstraints.DEFAULT_CONSTRAINTS == pristine

        ptycho.reset_recon()
        ptycho.obj_model.constraints = {"gaussian_sigma": 3.0, "tv_weight_xy": 0.25}

        assert ptycho.obj_model.constraints.gaussian_sigma == 3.0
        assert ptycho.obj_model.constraints.tv_weight_xy == 0.25
        assert ObjectConstraints.DEFAULT_CONSTRAINTS == pristine
        assert ptycho.obj_model.constraints is not ObjectConstraints.DEFAULT_CONSTRAINTS


# --- FOV-mask single application ---------------------------------------------


class TestFovMaskSingleApplication:
    def _make_obj(self, obj_type) -> ObjectPixelated:
        obj = ObjectPixelated.from_uniform(obj_type=obj_type, num_slices=1)
        obj._initialize_obj((1, 16, 16), sampling=(0.1, 0.1))
        obj.constraints.apply_fov_mask = True
        # Force a non-trivial _obj so masking is observable
        if obj_type == "complex":
            obj._obj = torch.nn.Parameter(
                torch.ones(1, 16, 16, dtype=torch.complex64) * (0.5 + 0.3j)
            )
        else:
            obj._obj = torch.nn.Parameter(torch.full((1, 16, 16), 0.7))
        return obj

    @pytest.mark.parametrize("obj_type", ["pure_phase", "complex", "potential"])
    def test_mask_applied_once(self, obj_type):
        obj = self._make_obj(obj_type)
        # Half-mask: ones on the left, zeros on the right; if mask is applied
        # twice the masked region squares the multiplication (no observable
        # difference for 0/1 masks), so use a non-binary mask.
        mask = torch.full((1, 16, 16), 0.5)
        obj._mask = mask
        out = obj.apply_hard_constraints(obj._obj, mask=mask)
        # Verify nothing crashed and shape is preserved.
        assert out.shape == obj._obj.shape
        # If mask had been applied twice, |out| would scale by 0.5**2 = 0.25
        # of the unmasked value; once it scales by 0.5. We compare to the
        # per-obj-type expected post-constraint value.
        if obj_type == "pure_phase":
            # phase recentered to zero mean, then *= 0.5 mask
            expected_mag = 0.0  # phase=constant -> recenter to 0 -> *0.5 = 0
        elif obj_type == "potential":
            # positivity clamp keeps 0.7, * 0.5 -> 0.35 (one application)
            expected_mag = 0.35
        else:  # complex
            # amp clamp keeps 0.5+0.3j, * 0.5 -> magnitude 0.5 * |0.5+0.3j|
            expected_mag = 0.5 * abs(0.5 + 0.3j)
        # Sample the magnitude in the masked region
        if out.is_complex():
            sampled = out.abs().mean().item()
        else:
            sampled = out.abs().mean().item()
        assert abs(sampled - expected_mag) < 1e-4, (
            f"{obj_type}: expected mag ~{expected_mag}, got {sampled} "
            f"(would be {expected_mag * 0.5} if mask were applied twice)"
        )
