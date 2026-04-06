# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class M0GravityRequest(BaseModel):
    mass1: float = Field(..., description="Mass of the first object. (positive real number)")
    mass2: float = Field(..., description="Mass of the second object. (positive real number)")
    distance: float = Field(..., description="Distance between the two objects. (positive real number)")

    # for simple_system and complex_system
    initial_velocity: Optional[float] = Field(None, description="Initial velocity of the object. (optional)")
    duration: Optional[float] = Field(None, description="Duration of the simulation. (optional)")
    time_step: Optional[float] = Field(None, description="Time step for the simulation. (optional)")


class M1CoulombForceRequest(BaseModel):
    q1: float = Field(..., description="Magnitude of the charge of the first object. (positive real number)")
    q2: float = Field(..., description="Magnitude of the charge of the second object. (positive real number)")
    distance: float = Field(..., description="Distance between the two objects. (positive real number)")

    # for simple_system and complex_system
    m1: Optional[float] = Field(None, description="Mass of the first object. (optional)")
    m2: Optional[float] = Field(None, description="Mass of the second object. (optional)")
    duration: Optional[float] = Field(None, description="Duration of the simulation. (optional)")
    time_step: Optional[float] = Field(None, description="Time step for the simulation. (optional)")


class M2MagneticForceRequest(BaseModel):
    current1: float = Field(..., description="Current in the first wire. (positive real number)")
    current2: float = Field(..., description="Current in the second wire. (positive real number)")
    distance: float = Field(..., description="Distance between the two wires. (positive real number)")

    # for simple_system and complex_system
    mass_wire: Optional[float] = Field(
        None, description="Mass per unit length of the moving wire (the second wire). (optional)"
    )
    initial_velocity: Optional[float] = Field(None, description="Initial velocity of the second wire. (optional)")
    duration: Optional[float] = Field(None, description="Duration of the simulation. (optional)")
    time_step: Optional[float] = Field(None, description="Time step for the simulation. (optional)")


class M3FourierLawRequest(BaseModel):
    k: float = Field(..., description="K constant (a physical constant). (positive real number)")
    A: float = Field(..., description="Cross-sectional area. (positive real number)")
    delta_T: float = Field(..., description="Temperature difference. (positive real number)")
    d: float = Field(..., description="Thickness of the material. (positive real number)")

    # for simple_system and complex_system
    num_points: Optional[int] = Field(None, description="Number of sampling points for simulation. (optional)")


class M4SnellLawRequest(BaseModel):
    refractive_index_1: Optional[float] = Field(
        None, description="Refractive index of the first medium. (positive real number, typically >= 1) (optional)"
    )
    refractive_index_2: Optional[float] = Field(
        None, description="Refractive index of the second medium. (positive real number, typically >= 1) (optional)"
    )
    incidence_angle: float = Field(
        ..., description="Angle of incidence in degrees. (real number, typically between 0 and 90)"
    )

    # for simple_system and complex_system
    refractive_index_3: Optional[float] = Field(None, description="Refractive index of the third medium. (optional)")
    speed_medium1: Optional[float] = Field(None, description="Speed of light in medium 1. (optional)")
    speed_medium2: Optional[float] = Field(None, description="Speed of light in medium 2. (optional)")


class M5RadioactiveDecayRequest(BaseModel):
    N0: Optional[float] = Field(None, description="Initial number of atoms. (positive real number) (optional)")
    lambda_constant: Optional[float] = Field(
        None, description="Lambda constant (a physical constant). (positive real number) (optional)"
    )
    t: float = Field(..., description="Time elapsed. (positive real number)")

    # for simple_system and complex_system
    N0a: Optional[float] = Field(None, description="Initial number of atoms for Isotope A. (optional)")
    N0b: Optional[float] = Field(None, description="Initial number of atoms for Isotope B. (optional)")
    lambda_a: Optional[float] = Field(None, description="Lambda constant for Isotope A. (optional)")
    lambda_b: Optional[float] = Field(None, description="Lambda constant for Isotope B. (optional)")
    num_points: Optional[int] = Field(None, description="Number of sampling points for simulation. (optional)")


class M6UnderdampedHarmonicRequest(BaseModel):
    k_constant: float = Field(..., description="K constant (a physical constant). (positive real number)")
    mass: float = Field(..., description="Mass of the object. (positive real number)")
    b_constant: float = Field(..., description="B constant (a physical constant). (positive real number)")

    # for simple_system and complex_system
    initial_amplitude: Optional[float] = Field(None, description="Initial amplitude. (optional)")


class M7MalusLawRequest(BaseModel):
    I_0: float = Field(..., description="Initial intensity of the light. (positive real number)")
    theta: float = Field(
        ...,
        description="Angle between the polarization axis of the polarizer and the polarization direction of the incident light in radians. (real number between 0 and π/2)",
    )


class M8SoundSpeedRequest(BaseModel):
    adiabatic_index: float = Field(..., description="Adiabatic index of the gas. (positive real number)")
    temperature: float = Field(..., description="Temperature of the medium in Kelvin. (positive real number)")
    molar_mass: float = Field(..., description="Molar mass of the medium. (positive real number)")

    # for simple_system and complex_system
    distance: Optional[float] = Field(None, description="Distance to the wall. (optional)")
    driving_frequency: Optional[float] = Field(
        None, description="Driving frequency of the sound wave from the source. (optional)"
    )
    tube_diameter: Optional[float] = Field(None, description="Internal diameter of the resonance tube. (optional)")


class M9HookeLawRequest(BaseModel):
    x: float = Field(..., description="Displacement from the equilibrium position. (positive real number)")

    # for simple_system and complex_system
    m: Optional[float] = Field(None, description="Mass attached to the spring. (optional)")


class M10BEDistributionRequest(BaseModel):
    omega: Optional[float] = Field(None, description="Angular frequency. (positive real number) (optional)")
    temperature: float = Field(..., description="Temperature. (positive real number)")

    # for simple_system and complex_system
    probe_frequency: Optional[float] = Field(
        None, description="Specific angular frequency at which to measure the radiation. (optional)"
    )
    center_frequency: Optional[float] = Field(
        None, description="Central angular frequency that the filter is tuned to. (optional)"
    )
    bandwidth: Optional[float] = Field(
        None, description="Width of the frequency band that the filter allows to pass. (optional)"
    )


class M11HeatTransferRequest(BaseModel):
    m: float = Field(..., description="Mass of the substance. (positive real number)")
    c: float = Field(..., description="C constant (a physical constant). (positive real number)")
    delta_T: float = Field(..., description="Temperature change. (positive real number)")


def get_tool_schema(module_name: str) -> dict:
    """
    Generates an OpenAI-compatible tool definition for a specific NewtonBench module.
    """
    model_cls = MODULE_REQUEST_CLASSES_MAPPING.get(module_name)
    if not model_cls:
        raise ValueError(f"No request class found for module: {module_name}")

    schema = model_cls.model_json_schema()

    properties = {}
    required = []

    for prop_name, prop_info in schema.get("properties", {}).items():
        prop_type = prop_info.get("type", "number")

        if prop_name not in schema.get("required", []):
            properties[prop_name] = {
                "type": [prop_type, "null"],
                "description": prop_info.get("description", f"Parameter {prop_name}"),
            }
        else:
            properties[prop_name] = {
                "type": prop_type,
                "description": prop_info.get("description", f"Parameter {prop_name}"),
            }
        required.append(prop_name)

    return {
        "type": "function",
        "name": f"run_experiment_{module_name}",
        "description": f"Run a physics experiment for the {module_name} module to gather data.",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        "strict": True,
    }


MODULE_REQUEST_CLASSES_MAPPING: Dict[str, Any] = {
    "m0_gravity": M0GravityRequest,
    "m1_coulomb_force": M1CoulombForceRequest,
    "m2_magnetic_force": M2MagneticForceRequest,
    "m3_fourier_law": M3FourierLawRequest,
    "m4_snell_law": M4SnellLawRequest,
    "m5_radioactive_decay": M5RadioactiveDecayRequest,
    "m6_underdamped_harmonic": M6UnderdampedHarmonicRequest,
    "m7_malus_law": M7MalusLawRequest,
    "m8_sound_speed": M8SoundSpeedRequest,
    "m9_hooke_law": M9HookeLawRequest,
    "m10_be_distribution": M10BEDistributionRequest,
    "m11_heat_transfer": M11HeatTransferRequest,
}
