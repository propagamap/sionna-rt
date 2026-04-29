#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""ITU radio materials"""

import mitsuba as mi
from typing import Tuple, Callable

from .itu import itu_material, ITU_MATERIALS_PROPERTIES
from .radio_material import RadioMaterial


class ITURadioMaterial(RadioMaterial):
    # pylint: disable=line-too-long
    r"""
    Class implementing the materials defined in the ITU-R P.2040-4 recommendation :cite:p:`ITURP20404`

    This class inherits from :class:`~sionna.rt.RadioMaterial`.

    The models from the ITU-R P.2040-4 recommendation are based on curve fitting
    to measurement results and assume non-ionized and non-magnetic materials (:math:`\mu_r = 1`).
    Frequency dependence is modeled by

    .. math::

        \begin{aligned}
            \varepsilon_r &= a f_{\text{GHz}}^b\\
            \sigma &= c f_{\text{GHz}}^d
        \end{aligned}

    where :math:`f_{\text{GHz}}` is the frequency in GHz, and the constants
    :math:`a`, :math:`b`, :math:`c`, and :math:`d` characterize the material.

    Note that the relative permittivity :math:`\varepsilon_r` and
    conductivity :math:`\sigma` of all materials are updated automatically when
    the frequency is set through the scene's property :class:`~.rt.Scene.frequency`.

    In addition to the following inputs, additional keyword arguments can be
    provided that will be passed to the scattering pattern as keyword
    arguments.

    :param name: Unique name of the material. Ignored if ``props`` is provided.
    :param itu_type: Type the ITU material. The available materials are listed in :ref:`the corresponding table <provided-materials>`. Ignored if ``props`` is provided.
    :param thickness: Thickness of the material [m]. Ignored if ``props`` is provided.
    :param scattering_coefficient: Scattering coefficient :math:`S\in[0,1]` as defined in :eq:`scattering_coefficient`. Ignored if ``props`` is provided.
    :param xpd_coefficient:  Cross-polarization discrimination coefficient :math:`K_x\in[0,1]` as defined in :eq:`xpd`. Only relevant if ``scattering_coefficient`` is not equal to zero. Ignored if ``props`` is provided.
    :param scattering_pattern: Scattering pattern to use for diffuse reflection. Only relevant if ``scattering_coefficient`` is not equal to zero. Ignored if ``props`` is provided. Defaults to :func:`~sionna.rt.lambertian_pattern`.
    :param color: RGB (red, green, blue) color for the radio material as displayed in the previewer and renderer. Each RGB component must have a value within the range :math:`[0,1]`. If set to :py:class:`None`, then a random color is used.
    :param forced_freq_range: Forces the use of the parameters corresponding to the given frequency range (in GHz) when evaluating the material properties. This can be used to evaluate the material properties at frequencies for which there is more than one defined range with different properties.
    :param props: Mitsuba container storing the material properties, and used when loading a scene to initialize the radio material.
    """

    # ITU material colors
    ITU_MATERIAL_COLORS = {
        "vacuum": (0.75, 0.75, 0.75),
        "concrete": (0.539, 0.539, 0.539),
        "brick": (0.402, 0.112, 0.087),
        "plasterboard": (0.051, 0.539, 0.133),
        "wood": (0.266, 0.109, 0.060),
        "glass": (0.168, 0.139, 0.509),
        "clear_acrylic": (0.70, 0.85, 0.95),
        "ceiling_board": (0.376, 0.539, 0.117),
        "chipboard": (0.509, 0.159, 0.323),
        "plywood": (0.136, 0.076, 0.539),
        "marble": (0.701, 0.644, 0.485),
        "floorboard": (0.539, 0.386, 0.025),
        "vinyl_tile": (0.20, 0.20, 0.20),
        "carpet_tile": (0.45, 0.10, 0.10),
        "asphalt_concrete": (0.25, 0.25, 0.25),
        "metal": (0.220, 0.220, 0.254),
        "very_dry_ground": (0.539, 0.319, 0.223),
        "medium_dry_ground": (0.539, 0.181, 0.076),
        "wet_ground": (0.539, 0.027, 0.147),
    }

    # pylint: disable=line-too-long
    def __init__(
        self,
        name: str | None = None,
        itu_type: str | None = None,
        thickness: float | mi.Float | None = None,
        scattering_coefficient: float | mi.Float = 0.0,
        xpd_coefficient: float | mi.Float = 0.0,
        scattering_pattern: Callable[[mi.Vector3f, mi.Vector3f, ...], mi.Float] | None = None,
        color: Tuple[float, float, float] | None = None,
        forced_freq_range: Tuple[float, float] | None = None,
        props: mi.Properties | None = None,
        **kwargs):

        has_props = props is not None
        if has_props:
            direct_args_none = (
                (name is None) and (itu_type is None) and (thickness is None)
                and (scattering_coefficient == 0.0) and (xpd_coefficient == 0.0)
            )
            if not direct_args_none:
                raise ValueError(
                    "When providing a `props` dictionary, not argument other"
                    " than `scattering_pattern` and `color` should be provided."
                )
            if 'type' not in props:
                raise ValueError(
                    "Missing property \"type\" (string) to select the ITU"
                    " material type."
                )
            itu_type = props['type']
            del props['type']

        if itu_type not in ITU_MATERIALS_PROPERTIES:
            raise ValueError(f"Invalid ITU material type \"{itu_type}\"")
        self._itu_type = itu_type
        self.forced_freq_range = forced_freq_range

        # Order of priority to set the visual color of this ITU material:
        # 1. `color` keyword argument
        # 2. `color`, `reflectance` or `base_color` property specified in the
        #    props (scene dictionary or XML file).
        # 3. Default color from `ITU_MATERIAL_COLORS`.
        if color is None:
            color = ITURadioMaterial.ITU_MATERIAL_COLORS[itu_type]
            if has_props:
                for pname in ("color", "reflectance", "base_color"):
                    if pname in props:
                        color = tuple(props[pname])
                        del props[pname]
                props["color"] = mi.ScalarColor3f(color)

        # Frequency update callback
        def cb(f: float):
            return itu_material(itu_type, f, self.forced_freq_range)

        if has_props:
            super().__init__(scattering_pattern=scattering_pattern,
                             frequency_update_callback=cb,
                             props=props,
                             **kwargs)
        else:
            super().__init__(name=name,
                             thickness=thickness,
                             scattering_coefficient=scattering_coefficient,
                             xpd_coefficient=xpd_coefficient,
                             scattering_pattern=scattering_pattern,
                             frequency_update_callback=cb,
                             color=color,
                             **kwargs)

    @property
    def itu_type(self):
        r"""
        Get the ITU type

        :type: :py:class:`str`
        """
        return self._itu_type

    @property
    def forced_range(self):
        r"""
        Get/set the frequency range (min_GHz, max_GHz) to force when evaluating
        ITU material parameters, or :py:class:`None` to use automatic selection.
        Setting this property immediately re-evaluates the material parameters
        at the current scene frequency.

        :type: :py:class:`tuple` [:py:class:`float`, :py:class:`float`] | :py:class:`None`
        """
        return self.forced_freq_range

    @forced_range.setter
    def forced_range(self, value):
        self.forced_freq_range = value
        self.frequency_update()

    def to_string(self) -> str:
        r"""
        Returns a string describing the object
        """
        s = f"ITURadioMaterial type={self._itu_type}\n"\
            f"                 eta_r={self._eta_r[0]:.3f}\n"\
            f"                 sigma={self._sigma[0]:.3f}\n"\
            f"                 thickness={self._d[0]:.3f}\n"\
            f"                 scattering_coefficient={self._s[0]:.3f}\n"\
            f"                 xpd_coefficient={self._kx[0]:.3f}"
        return s


mi.register_bsdf("itu-radio-material",
                 lambda props: ITURadioMaterial(props=props))
