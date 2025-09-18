"""Module for loading and accessing nuclear mass data

Reads in the bundled AMDC AME mass data (amdc_2020.txt) and loads into a dictionary.

Classes
-------
NucleusData
    dataclass representing AME mass data
NuclearDataMap
    interface for accessing mass data

Functions
---------
generate_nucleus_id(z: int, a: int) -> int:
    get a unqiue nucleus id
"""

from ..constants import AMU_2_MEV, ELECTRON_MASS_U

from dataclasses import dataclass
from pathlib import Path
from importlib.resources import files, as_file

DATA_PATH: Path = Path(__file__).parent.resolve() / "amdc_2020.txt"


@dataclass
class NucleusData:
    """Nucleus data from AME dataclass

    Contains information on nuclear masses and symbols

    Attributes
    ----------
    mass: float
        mass of the nucleus in MeV
    atomic_mass: float
        atomic mass in amu
    element_symbol: str
        atomic symbol (H, He, Li, etc.)
    isotopic_symbol: str
        isotopic symbol without formatting (1H, 2H, 3H, 4He, etc.)
    pretty_iso_symbol: str
        isotopic symbol with rich text formatting (<sup>1</sup>H, etc.)
    Z: int
        atomic number
    A: int
        mass number

    Methods
    -------
    __str__()
        string representation, isotopic_symbol
    get_latex_rep()
        get a LaTeX style represenation, for use with matplotlib etc.

    """

    mass: float = 0.0  # nuclear mass, MeV
    atomic_mass: float = 0.0  # atomic mass (includes electrons), amu
    element_symbol: str = ""  # Element symbol (H, He, Li, etc.)
    isotopic_symbol: str = ""  # Isotopic symbol w/o formating (1H, 2H, 3H, 4He, etc.)
    pretty_iso_symbol: str = (
        ""  # Isotopic symbol w/ rich text formating (<sup>1</sup>H, etc.)
    )
    Z: int = 0
    A: int = 0

    def __str__(self) -> str:
        return self.isotopic_symbol

    def get_latex_rep(self) -> str:
        """Get the LaTeX representation of the isotopic symbol

        Returns
        -------
        str
            a string of the isotopic symbol in LaTeX format
        """

        return "$^{" + str(self.A) + "}$" + self.element_symbol


# Szudzik pairing function, requires use of unsigned integers
def generate_nucleus_id(z: int, a: int) -> int:
    """get a unqiue nucleus id

    Use the Szudzik pairing function to generate a unique nucleus ID

    Parameters
    ----------
    z: int
        Nucleus atomic number
    a: int
        Nucleus mass number

    Returns
    -------
    int
        Unique identifier for this nucleus

    """
    return z * z + z + a if z == max(z, a) else a * a + z


class NuclearDataMap:
    """Maps nuclear numbers (Z,A) to mass information

    Reads in AME 2020 mass evaluation and creates a map of nucleus numbers (Z, A)
    to nuclear mass information

    Attributes
    ----------
    map: dict[int, NucleusData]
        dictionary mapping nucleus id's to associated data

    Methods
    -------
    get_data(z: int, a: int) -> NucleusData
        retrieve the mass data for a given nucleus
    """

    def __init__(self):
        self.map = {}

        data_handle = files("spyral_utils.nuclear").joinpath("amdc_2020.txt")
        with as_file(data_handle) as data_path:
            data_file = open(data_path, "r")
            data_file.readline()  # Header
            for line in data_file:
                entries = line.split()
                data = NucleusData()
                data.Z = int(entries[0])  # Column 1: Z
                data.A = int(entries[1])  # Column 2: A
                data.element_symbol = entries[2]  # Column 3: Element
                data.atomic_mass = float(entries[3])
                data.mass = (
                    (float(entries[3]) - float(data.Z) * ELECTRON_MASS_U) * AMU_2_MEV
                )  # Remove electron masses to obtain nuclear masses, Column 4
                data.isotopic_symbol = f"{data.A}{entries[2]}"
                data.pretty_iso_symbol = f"<sup>{data.A}</sup>{entries[2]}"
                self.map[generate_nucleus_id(data.Z, data.A)] = data
            data_file.close()

    def get_data(self, z: int, a: int) -> NucleusData:
        """retrieve the mass data for a given nucleus

        Parameters
        ----------
        z: int
            atomic number
        a: int
            mass number

        Returns
        -------
        NucleusData
            associated mass data
        """
        return self.map[generate_nucleus_id(z, a)]
