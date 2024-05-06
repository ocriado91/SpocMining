#!/usr/bin/env python3

import polars as pl
import pykep as pk

import constants

def convert_to_planet(planet_data: pl.DataFrame) -> pk.planet:
    '''
    Convert planet data to pykep planet object

    Parameters:
        - planet_data (pl.DataFrame): Polars DataFrame of asteroid to be
        converted to pykep planet object
    '''

    for row in planet_data.iter_rows(named=True):
        return pk.planet.keplerian(
            pk.epoch_from_iso_string(constants.ISO_T_START),
            (
                row["Semi-major axis [m]"],
                row["Eccentricity"],
                row["Inclination [rad]"],
                row["Ascending Node [rad]"],
                row["Argument of Periapsis [rad]"],
                row["True Anomaly [rad]"]
            ),
            constants.MU_TRAPPIST,
            constants.G*row['Mass [0 to 1]'],
            1,
            1.1,
            "Asteroid " + str(row["ID"])
            )
