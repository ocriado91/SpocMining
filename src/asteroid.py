# -*- coding: utf-8 -*-

"""
Asteroid class definition file
"""
#######################################################################
# Imports area
#######################################################################

# Generic / Built-in


# Other Libs
import pykep as pk

# Own Libs
from src import constants


#######################################################################


class Asteroid:
    """
    Asteroid class

    Attributes
    ----------
    asteroidData : list
        Asteroid data

    Methods
    -------
    parse_asteroid_data
    get_coordinates

    """

    def __init__(self,
                 asteroidData: str):
        """
        Initialize asteroid object
        """

        # Initialize Asteroid attributes
        self.asteroidId = None
        self.keplerianElements = []
        self.planetObject = None
        self.normalizedMass = None
        self.materialType = None

        self.parse_asteroid_data(asteroidData)

    def parse_asteroid_data(self,
                            asteroidData: list):
        """
        Parse candidate asteroid data

        Parameters
        ----------
        self: Asteroid
            Asteroid object
        asteroidData : list
            Asteroid data

        """
        self.asteroidId = asteroidData[0]
        self.inclination = asteroidData[4]
        self.keplerianElements = [asteroidData[1],
                                  asteroidData[2],
                                  asteroidData[3],
                                  asteroidData[4],
                                  asteroidData[5],
                                  asteroidData[6]]
        self.planetObject = pk.planet.keplerian(
            pk.epoch_from_iso_string(constants.ISO_T_START),
            (
                asteroidData[1],
                asteroidData[2],
                asteroidData[3],
                asteroidData[4],
                asteroidData[5],
                asteroidData[6],
            ),
            constants.MU_TRAPPIST,
            constants.G * asteroidData[7],
            1,
            1.1,
            "Asteroid " + str(int(asteroidData[0])),
        )

        # And asteroids' masses and material type
        self.normalizedMass = asteroidData[-2]
        self.materialType, self.materialColor =\
            set_material_type(int(asteroidData[-1]))

    def get_coordinates(self):
        """
        Get coordinates of asteroid
        """
        r, _ = self.planetObject.eph(
            pk.epoch_from_iso_string(constants.ISO_T_START))
        x, y, z = r
        return x, y, z

def set_material_type(materialType: int)->tuple:
    """
    Transform material type to string and color

    Parameters
    ----------
    materialType : int
        Material type integer to be transformed

    Returns
    -------
    materialType : str
        Material type string
    color : str
        Material type color

    """
    if materialType == 0:
        return "Gold", "r"
    if materialType == 1:
        return "Platinum", "g"
    if materialType == 2:
        return "Nickel", "b"
    return "Propellant" , "c"
