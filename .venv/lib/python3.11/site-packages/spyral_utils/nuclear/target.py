"""Module containing AT-TPC representation of target materials and their associated energy loss calculations.

Target materials can be defined as either the corresponding CAtima name or a compound array.

Classes should never be directly instantiated, rather use the load_target function to do the work for you.

Classes
-------
TargetData
    Raw target dataclass, should never be instantiated directly by user
GasTarget
    AT-TPC representation of a gas target
SolidTarget
    AT-TPC representation of a gas target

Functions
---------
deserialize_target_data(target_path: Path, nuclear_map: NuclearDataMap) -> TargetData | None
    Deserialize raw target data from JSON. Should never be used directly.
serialize_target_data(target_path: Path, target_data: TargetData)
    Serialize raw target data to JSON. Should never be used directly.
load_target(target_path: Path, nuclear_map: NuclearDataMap) -> GasTarget | SolidTarget | None
    Load a target from JSON. This is what should be used to load a target.
save_target(target_path: Path, target: GasTarget | SolidTarget)
    Save a targe to JSON. This is what should be used to save a target.
"""

from .nuclear_map import NuclearDataMap, NucleusData
from ..constants import AMU_2_MEV, GAS_CONSTANT, ROOM_TEMPERATURE

import pycatima as catima
from pathlib import Path
from json import load, dumps
import numpy as np


class TargetError(Exception):
    pass


class GasTarget:
    """An AT-TPC gas target

    Gas target for which energy loss can be calculated using pycatima. Can perform several types of
    energy loss calculations (straggling, dEdx, energy lost, etc.)

    Attributes
    ----------
    data: TargetData
        The raw target data from a JSON file
    ugly_string: str
        A string representation without rich formatting
    pretty_string: str
        A string representation with rich formatting
    material: catima.Material
        The catima representation of the target
    density: float
        The target density in g/cm^3

    Methods
    -------
    get_dedx(projectile_data: NucleusData, projectile_energy: float) -> float
        get the stopping power (dEdx) for a projectile in this target
    get_angular_straggling(projectile_data: NucleusData, projectile_energy: float) -> float:
        get the angular straggling for a projectile in this target
    get_energy_loss(projectile_data: NucleusData, projectile_energy: float, distances: ndarray) -> ndarray:
        get the energy loss values for a projectile travelling distances through the target
    """

    def __init__(
        self,
        compound: list[tuple[int, int, int]],
        pressure: float,
        nuclear_data: NuclearDataMap,
    ):
        self.compound = compound
        self.pressure = pressure  # Torr
        molar_mass: float = 0.0
        for z, a, s in self.compound:
            molar_mass += a * s
        self.density: float = (
            molar_mass * self.pressure / (GAS_CONSTANT * ROOM_TEMPERATURE)
        )
        self.pretty_string: str = "(Gas)" + "".join(
            [
                f"{nuclear_data.get_data(z, a).pretty_iso_symbol}<sub>{s}</sub>"
                for (z, a, s) in self.compound
            ]
        )
        self.ugly_string: str = "(Gas)" + "".join(
            [
                f"{nuclear_data.get_data(z, a).isotopic_symbol}{s}"
                for (z, a, s) in self.compound
            ]
        )

        # Construct the target material
        self.material = catima.Material()
        for (
            z,
            a,
            s,
        ) in self.compound:
            self.material.add_element(
                nuclear_data.get_data(z, a).atomic_mass, z, float(s)
            )
        self.material.density(self.density)

    def __str__(self) -> str:
        return self.pretty_string

    def get_dedx(self, projectile_data: NucleusData, projectile_energy: float) -> float:
        """Calculate the stopping power of the target for a projectile

        Parameters
        ----------
        projectile_data: NucleusData
            the projectile type
        projectile_energy: float
            the projectile kinetic energy in MeV

        Returns
        -------
        float
            dEdx in MeV/g/cm^2
        """
        mass_u = projectile_data.mass / AMU_2_MEV  # convert to u
        projectile = catima.Projectile(mass_u, projectile_data.Z)
        projectile.T(projectile_energy / mass_u)
        return catima.dedx(projectile, self.material)

    def get_angular_straggling(
        self, projectile_data: NucleusData, projectile_energy: float
    ) -> float:
        """Calculate the angular straggling for a projectile

        Parameters
        ----------
        projectile_data: NucleusData
            the projectile type
        projectile_energy: float
            the projectile kinetic energy in MeV

        Returns
        -------
        float
            angular straggling in radians
        """
        mass_u = projectile_data.mass / AMU_2_MEV  # convert to u
        projectile = catima.Projectile(
            mass_u, projectile_data.Z, T=projectile_energy / mass_u
        )
        return catima.calculate(projectile, self.material).get_dict()["sigma_a"]

    def get_energy_loss(
        self,
        projectile_data: NucleusData,
        projectile_energy: float,
        distances: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the energy loss of a projectile traveling over a set of distances

        Parameters
        ----------
        projectile_data: NucleusData
            the projectile type
        projectile_energy: float
            the projectile kinetic energy in MeV
        distances: ndarray
            a set of distances in meters over which to calculate the energy loss

        Returns
        -------
        ndarray
            set of energy losses
        """
        mass_u = projectile_data.mass / AMU_2_MEV  # convert to u
        projectile = catima.Projectile(
            mass_u, projectile_data.Z, T=projectile_energy / mass_u
        )
        eloss = np.zeros(len(distances))
        for idx, distance in enumerate(distances):
            self.material.thickness_cm(distance * 100.0)
            projectile.T(projectile_energy / mass_u)
            eloss[idx] = catima.calculate(projectile, self.material).get_dict()["Eloss"]
        return eloss

    def get_range(
        self, projectile_data: NucleusData, projectile_energy: float
    ) -> float:
        """Calculate the range of a projectile in the target

        Parameters
        ----------
        projectile_data: NucleusData
            the projectile type
        projectile_energy: float
            the projectile kinetic energy in MeV

        Returns
        -------
        float
            The range in m
        """
        mass_u = projectile_data.mass / AMU_2_MEV  # convert to u
        projectile = catima.Projectile(
            mass_u, projectile_data.Z, T=projectile_energy / mass_u
        )
        range_gcm2 = catima.calculate(projectile, self.material).get_dict()["range"]
        range_m = range_gcm2 / self.material.density() * 0.01
        return range_m

    def get_number_density(self) -> float:
        """Get the number density of gas molecules

        Returns
        -------
        Number density of gas molecules in molecules/cm^3
        """
        return self.material.number_density()


def deserialize_gas_target_data(
    target_path: Path, nuclear_map: NuclearDataMap
) -> GasTarget | None:
    """Deserialize gas target data from a JSON file

    Parameters
    ----------
    target_path: Path
        Path to JSON file containing gas target data

    Returns
    -------
    GasTarget | None
        If the data loaded successfully, returns a GasTarget. Otherwise returns None.
    """
    with open(target_path, "r") as target_file:
        json_data = load(target_file)
        if "compound" not in json_data or "pressure(Torr)" not in json_data:
            return None
        else:
            return GasTarget(
                json_data["compound"], json_data["pressure(Torr)"], nuclear_map
            )


def serialize_gas_target_data(target_path: Path, data: GasTarget) -> None:
    """Serialize gas target data to JSON file

    Parameters
    ----------
    target_path: Path
        Path to JSON file to write target data
    data: GasTarget
        data to be written
    """
    with open(target_path, "w") as target_file:
        json_str = dumps(
            data,
            default=lambda data: {
                "compound": data.compound,
                "pressure(Torr)": data.pressure,
            },
        )
        target_file.write(json_str)


class SolidTarget:
    """An AT-TPC gas target

    Gas target for which energy loss can be calculated using pycatima. Can perform several types of
    energy loss calculations (straggling, dEdx, energy lost, etc.)

    Attributes
    ----------
    data: TargetData
        The raw target data from a JSON file
    ugly_string: str
        A string representation without rich formatting
    pretty_string: str
        A string representation with rich formatting
    material: catima.Material
        The catima representation of the target

    Methods
    -------
    get_dedx(projectile_data: NucleusData, projectile_energy: float) -> float
        get the stopping power (dEdx) for a projectile in this target
    get_angular_straggling(projectile_data: NucleusData, projectile_energy: float) -> float:
        get the angular straggling for a projectile in this target
    get_energy_loss(projectile_data: NucleusData, projectile_energy: float, distances: ndarray) -> ndarray:
        get the energy loss values for a projectile travelling distances through the target
    """

    UG2G: float = 1.0e-6  # convert ug to g

    def __init__(
        self,
        compound: list[tuple[int, int, int]],
        thickness: float,
        nuclear_data: NuclearDataMap,
    ):
        self.compound = compound
        self.thickness = thickness

        self.pretty_string: str = "(Solid)" + "".join(
            [
                f"{nuclear_data.get_data(z, a).pretty_iso_symbol}<sub>{s}</sub>"
                for (z, a, s) in self.compound
            ]
        )
        self.ugly_string: str = "(Solid)" + "".join(
            [
                f"{nuclear_data.get_data(z, a).isotopic_symbol}{s}"
                for (z, a, s) in self.compound
            ]
        )

        # Construct the target material
        self.material = catima.Material()
        for (
            z,
            a,
            s,
        ) in self.compound:
            self.material.add_element(
                nuclear_data.get_data(z, a).atomic_mass, z, float(s)
            )
        self.material.thickness(self.thickness * self.UG2G)  # Convert ug/cm^2 to g/cm^2

    def get_dedx(self, projectile_data: NucleusData, projectile_energy: float) -> float:
        """Calculate the stopping power of the target for a projectile

        Parameters
        ----------
        projectile_data: NucleusData
            the projectile type
        projectile_energy: float
            the projectile kinetic energy in MeV

        Returns
        -------
        float
            dEdx in MeV/g/cm^2
        """
        mass_u = projectile_data.mass / AMU_2_MEV  # convert to u
        projectile = catima.Projectile(mass_u, projectile_data.Z)
        projectile.T(projectile_energy / mass_u)
        return catima.dedx(projectile, self.material)

    def get_angular_straggling(
        self, projectile_data: NucleusData, projectile_energy: float
    ) -> float:
        """Calculate the angular straggling for a projectile

        Parameters
        ----------
        projectile_data: NucleusData
            the projectile type
        projectile_energy: float
            the projectile kinetic energy in MeV

        Returns
        -------
        float
            angular straggling in radians
        """
        mass_u = projectile_data.mass / AMU_2_MEV  # convert to u
        projectile = catima.Projectile(
            mass_u, projectile_data.Z, T=projectile_energy / mass_u
        )
        return catima.calculate(projectile, self.material).get_dict()["sigma_a"]

    def get_energy_loss(
        self,
        projectile_data: NucleusData,
        projectile_energy: float,
        incident_angles: np.ndarray,
    ) -> np.ndarray:
        """Calculate the energy loss of a projectile traveling through the solid target for a given set of incident angles

        Assumes travelling through the entire target at an incident angle.

        Parameters
        ----------
        projectile_data: NucleusData
            the projectile type
        projectile_energy: float
            the projectile kinetic energy in MeV
        incident_angles: ndarray
            a set of incident angles, describing the angle between the particle trajectory and the normal of the target surface

        Returns
        -------
        ndarray
            set of energy losses
        """
        mass_u = projectile_data.mass / AMU_2_MEV  # convert to u
        projectile = catima.Projectile(
            mass_u, projectile_data.Z, T=projectile_energy / mass_u
        )
        eloss = np.zeros(len(incident_angles))
        nominal_thickness = self.material.thickness()
        for idx, angle in enumerate(incident_angles):
            self.material.thickness(nominal_thickness / abs(np.cos(angle)))
            projectile.T(projectile_energy / mass_u)
            eloss[idx] = catima.calculate(projectile, self.material).get_dict()["Eloss"]
        self.material.thickness(nominal_thickness)
        return eloss

    def get_range(
        self, projectile_data: NucleusData, projectile_energy: float
    ) -> float:
        """Calculate the range of a projectile in the target in g/cm^2

        Parameters
        ----------
        projectile_data: NucleusData
            the projectile type
        projectile_energy: float
            the projectile kinetic energy in MeV

        Returns
        -------
        float
            The range in g/cm^2
        """
        mass_u = projectile_data.mass / AMU_2_MEV  # convert to u
        projectile = catima.Projectile(
            mass_u, projectile_data.Z, T=projectile_energy / mass_u
        )

        return catima.calculate(projectile, self.material).get_dict()["range"]


def deserialize_solid_target_data(
    target_path: Path, nuclear_map: NuclearDataMap
) -> SolidTarget | None:
    """Deserialize solid target data from a JSON file

    Parameters
    ----------
    target_path: Path
        Path to JSON file containing target data

    Returns
    -------
    SolidTarget | None
        If the data loaded successfully, returns a SolidTarget. Otherwise returns None.
    """
    with open(target_path, "r") as target_file:
        json_data = load(target_file)
        if "compound" not in json_data or "thickness(ug/cm^2)" not in json_data:
            return None
        else:
            return SolidTarget(
                json_data["compound"], json_data["thickness(ug/cm^2)"], nuclear_map
            )


def serialize_solid_target_data(target_path: Path, data: SolidTarget) -> None:
    """Serialize solid target data to JSON file

    Parameters
    ----------
    target_path: Path
        Path to JSON file to write target data
    data: SolidTarget
        data to be written
    """
    with open(target_path, "w") as target_file:
        json_str = dumps(
            data,
            default=lambda data: {
                "compound": data.compound,
                "thickness(ug/cm^2)": data.thickness,
            },
        )
        target_file.write(json_str)


class GasMixtureTarget:
    def __init__(
        self,
        components: list[list[tuple[int, int, int]]],
        volume_fractions: list[float],
        pressure: float,
        nuclear_map: NuclearDataMap,
    ):
        self.components = components
        self.volume_fractions = volume_fractions
        self.pressure = pressure
        self.equivalent_compound: list[tuple[int, int, int]] = []
        self.average_molar_mass = 0.0
        self.density = 0.0
        self.pretty_string: str = "(GasMix)"
        self.ugly_string: str = "(GasMix)"
        for fraction, component in zip(self.volume_fractions, self.components):
            self.pretty_string += f"[{fraction * 100.0:.0}%-"
            self.ugly_string += f"[{fraction * 100.0:.0}%-"
            for z, a, s in component:
                self.pretty_string += (
                    f"{nuclear_map.get_data(z, a).pretty_iso_symbol}<sub>{s}</sub>"
                )
                self.ugly_string += f"{nuclear_map.get_data(z, a).isotopic_symbol}{s}"
            self.pretty_string += "]"
            self.ugly_string += "]"

        for compound, fraction in zip(self.components, self.volume_fractions):
            scale = int(100.0 * fraction)
            for element_z, element_a, element_s in compound:
                self.equivalent_compound.append(
                    (element_z, element_a, element_s * scale)
                )
                self.average_molar_mass += element_a * element_s * fraction

        self.density = (
            self.average_molar_mass * self.pressure / (GAS_CONSTANT * ROOM_TEMPERATURE)
        )
        # Construct the target material
        self.material = catima.Material()
        for (
            z,
            a,
            s,
        ) in self.equivalent_compound:
            self.material.add_element(
                nuclear_map.get_data(z, a).atomic_mass, z, float(s)
            )
        self.material.density(self.density)

    def get_dedx(self, projectile_data: NucleusData, projectile_energy: float) -> float:
        """Calculate the stopping power of the target for a projectile

        Parameters
        ----------
        projectile_data: NucleusData
            the projectile type
        projectile_energy: float
            the projectile kinetic energy in MeV

        Returns
        -------
        float
            dEdx in MeV/g/cm^2
        """
        mass_u = projectile_data.mass / AMU_2_MEV  # convert to u
        projectile = catima.Projectile(mass_u, projectile_data.Z)
        projectile.T(projectile_energy / mass_u)
        return catima.dedx(projectile, self.material)

    def get_angular_straggling(
        self, projectile_data: NucleusData, projectile_energy: float
    ) -> float:
        """Calculate the angular straggling for a projectile

        Parameters
        ----------
        projectile_data: NucleusData
            the projectile type
        projectile_energy: float
            the projectile kinetic energy in MeV

        Returns
        -------
        float
            angular straggling in radians
        """
        mass_u = projectile_data.mass / AMU_2_MEV  # convert to u
        projectile = catima.Projectile(
            mass_u, projectile_data.Z, T=projectile_energy / mass_u
        )
        return catima.calculate(projectile, self.material).get_dict()["sigma_a"]

    def get_energy_loss(
        self,
        projectile_data: NucleusData,
        projectile_energy: float,
        distances: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the energy loss of a projectile traveling over a set of distances

        Parameters
        ----------
        projectile_data: NucleusData
            the projectile type
        projectile_energy: float
            the projectile kinetic energy in MeV
        distances: ndarray
            a set of distances in meters over which to calculate the energy loss

        Returns
        -------
        ndarray
            set of energy losses
        """
        mass_u = projectile_data.mass / AMU_2_MEV  # convert to u
        projectile = catima.Projectile(
            mass_u, projectile_data.Z, T=projectile_energy / mass_u
        )
        eloss = np.zeros(len(distances))
        for idx, distance in enumerate(distances):
            self.material.thickness_cm(distance * 100.0)
            projectile.T(projectile_energy / mass_u)
            eloss[idx] = catima.calculate(projectile, self.material).get_dict()["Eloss"]
        return eloss

    def get_range(
        self, projectile_data: NucleusData, projectile_energy: float
    ) -> float:
        """Calculate the range of a projectile in the target

        Parameters
        ----------
        projectile_data: NucleusData
            the projectile type
        projectile_energy: float
            the projectile kinetic energy in MeV

        Returns
        -------
        float
            The range in m
        """
        mass_u = projectile_data.mass / AMU_2_MEV  # convert to u
        projectile = catima.Projectile(
            mass_u, projectile_data.Z, T=projectile_energy / mass_u
        )
        range_gcm2 = catima.calculate(projectile, self.material).get_dict()["range"]
        range_m = range_gcm2 / self.material.density() * 0.01
        return range_m

    def get_number_density(self) -> float:
        """Get the number density of gas molecules

        Returns
        -------
        Number density of gas molecules in molecules/cm^3
        """
        return self.material.number_density()


def deserialize_mixture_data(
    path: Path, nuclear_map: NuclearDataMap
) -> GasMixtureTarget | None:
    """Deserialize gas mixture target data from a JSON file

    Parameters
    ----------
    path: Path
        Path to JSON file containing target data

    Returns
    -------
    GasMixtureTarget | None
        If the data loaded successfully, returns a GasMixtureTarget. Otherwise returns None.
    """
    with open(path, "r") as mix_file:
        json_data = load(mix_file)
        if (
            "components" not in json_data
            or "volume_fractions" not in json_data
            or "pressure(Torr)" not in json_data
        ):
            return None
        return GasMixtureTarget(
            components=json_data["components"],
            volume_fractions=json_data["volume_fractions"],
            pressure=json_data["pressure(Torr)"],
            nuclear_map=nuclear_map,
        )


def serialize_mixture_data(target_path: Path, data: GasMixtureTarget) -> None:
    """Serialize gas mixture target data to JSON file

    Parameters
    ----------
    target_path: Path
        Path to JSON file to write target data
    data: GasMixtureTarget
        data to be written
    """
    with open(target_path, "w") as target_file:
        json_str = dumps(
            data,
            default=lambda data: {
                "components": data.components,
                "volume_fractions": data.volume_fractions,
                "pressure(Torr)": data.pressure,
            },
        )
        target_file.write(json_str)


def load_target(
    target_path: Path, nuclear_map: NuclearDataMap
) -> GasTarget | SolidTarget | GasMixtureTarget:
    """Load a target from a JSON file

    Read the JSON data, and construct the appropriate target type.

    Parameters
    ----------
    target_path: Path
        Path to JSON data
    nuclear_map: NuclearDataMap
        Nucleus mass data

    Returns
    -------
    GasTarget | SolidTarget | GasMixtureTarget
        Return a GasTarget, GasMixtureTarget, or SolidTarget where appropriate. Raises a
        TargetError on failure.
    """
    gas = deserialize_gas_target_data(target_path, nuclear_map)
    if gas is not None:
        return gas
    solid = deserialize_solid_target_data(target_path, nuclear_map)
    if solid is not None:
        return solid
    mix = deserialize_mixture_data(target_path, nuclear_map)
    if mix is not None:
        return mix
    raise TargetError(f"Invalid format for target data at {target_path}")


def save_target(target_path: Path, target: GasTarget | SolidTarget | GasMixtureTarget):
    """Write a target to a JSON file

    Create the JSON data, and write to disk.

    Parameters
    ----------
    target_path: Path
        Path to JSON data file
    target: GasTarget | SolidTarget | GasMixtureTarget
        Target to be written
    """
    if isinstance(target, GasTarget):
        serialize_gas_target_data(target_path, target)
    elif isinstance(target, SolidTarget):
        serialize_solid_target_data(target_path, target)
    elif isinstance(target, GasMixtureTarget):
        serialize_mixture_data(target_path, target)
    else:
        raise TargetError(
            f"Object {target} does not match any known target type and could not be saved"
        )
