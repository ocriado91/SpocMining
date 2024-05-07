#!/usr/bin/env python3
'''
Rover class to
'''

import logging

import numpy as np
import polars as pl
import pykep as pk

import constants
import utils

import plotting


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO)
logger = logging.getLogger(__name__)


# Start and end epochs
T_START = pk.epoch_from_iso_string(constants.ISO_T_START)
T_END = pk.epoch_from_iso_string(constants.ISO_T_END)

class Rover:

    def __init__(self,
                 datafile: str,
                 mission_window: int = constants.TIME_OF_MISSION,
                 first_asteroid=0):
        self.fuel = 1
        self.tank = [0] * 3
        self.score = 0
        self.visited_asteroids = []

        # Append first visited asteroid to list
        self.visited_asteroids.append(first_asteroid)

        self.mission_window = mission_window

        self._read_datafile(datafile)

    def _read_datafile(self,
                      datafile: str):
        '''
        Read asteroid candidates file and store into rover data attribute

        Parameters:
            - datafile (str): Asteroid candidates file
        '''

        logger.info("Reading datafile %s", datafile)

        # Set custom headers
        HEADER = ["ID",
                "Semi-major axis [m]",
                "Eccentricity",
                "Inclination [rad]",
                "Ascending Node [rad]",
                "Argument of Periapsis [rad]",
                "True Anomaly [rad]",
                "Mass [0 to 1]",
                "Material Type"]

        # Read asteroid candidates file
        data = pl.read_csv(datafile,
                           separator=" ",
                           has_header=False,
                           new_columns=HEADER)

        # Cast columns
        self.data = data.cast({"ID": pl.UInt16,
                "Material Type": pl.UInt8})

    def compute_journey(self,
                        asteroids: list,
                        time_of_arrival: list,
                        time_mining: list) -> float:
        '''
        Compute journey between asteroids using evaluation code
        method passing list of asteroids, time of arrival and time
        mining into each asteroid.

        Parameters:
            - asteroids (list): List of asteroids into the rover's journey.
            - time_of_arrival (list): Time of arrival to each asteroid.
            - time_mining (list): Time mining in each asteroid.

        '''

        # Check length of asteroid list and time lists
        assert len(asteroids) == len(time_of_arrival)
        assert len(asteroids) == len(time_mining)


        for idx in range(1, len(asteroids)):
            # Get indices of current asteroid and previous one
            current_asteroid_id = asteroids[idx]
            previous_asteroid_id = asteroids[idx - 1]

            # Extract data of asteroids
            current_asteroid_data = self.data.filter(
                pl.col("ID") == current_asteroid_id
            )

            previous_asteroid_data = self.data.filter(
                pl.col("ID") == previous_asteroid_id
            )

            # Convert asteroids data to planet objects
            current_planet_obj = utils.convert_to_planet(
                current_asteroid_data
            )
            previous_planet_obj = utils.convert_to_planet(
                previous_asteroid_data
            )

            logger.info("Computing journey between %s and %s",
                        previous_asteroid_id, current_asteroid_id)
            # Check if mission window is reached
            if time_of_arrival[idx] - time_of_arrival[0] > self.mission_window:
                logger.error("Mission window exceeded")
                break

            # Compute time of flight
            time_of_flight = time_of_arrival[idx] - \
                time_of_arrival[idx - 1] - \
                time_mining[idx - 1]

            # Compute the ephemeris of the previous asteroid at departing time
            time_of_departure = time_of_arrival[idx - 1] + time_mining[idx - 1]

            r1, v1 = previous_planet_obj.eph(
                T_START.mjd2000 + time_of_departure
            )

            # Compute the ephemeris of the current asteroid at time of arrival
            r2, v2 = current_planet_obj.eph(
                T_START.mjd2000 + time_of_arrival[idx]
            )

            # Solve the Lambert problem
            lambert = pk.lambert_problem(
                r1=r1,
                r2=r2,
                tof=time_of_flight * pk.DAY2SEC,
                mu=constants.MU_TRAPPIST,
                cw=False,
                max_revs=0
            )

            # Compute the delta-V necessary to go to current asteroid
            # from previous asteroid and match its velocity
            DV1 = [a - b for a, b in zip(v1, lambert.get_v1()[0])]
            DV2 = [a - b for a, b in zip(v2, lambert.get_v2()[0])]
            DV = np.linalg.norm(DV1) + np.linalg.norm(DV2)

            # Compute fuel consumption
            fuel_consumption = DV / constants.DV_per_fuel
            self.fuel -= fuel_consumption

            if self.fuel <= 0:
                logger.error("Out of fuel")
                break

            # Get material type of visited asteroid
            material_type = int(current_asteroid_data["Material Type"].item())

            # Material collected is the minimum between
            # mine the entire asteroid or just mine the asteroid partially
            material_collected = min(
                current_asteroid_data["Mass [0 to 1]"].item(),
                time_mining[idx] / constants.TIME_TO_MINE_FULLY
            )

            # Propellant asteroid
            if material_type == 3:
                self.fuel += material_collected
            else:
                self.tank[material_type] += material_collected

            # Update the score
            self.score = min(self.tank)

            # Report status
            logger.info("Travelling from %s to %s with a DV = %s. Fuel = %s. Tank = %s. Score = %s",
                         previous_asteroid_id,
                         current_asteroid_id,
                         DV,
                         self.fuel,
                         self.tank,
                         self.score)


    def compute_knn(self,
                    time: float,
                    target_asteroid_id: int,
                    k: int = 20) -> list:

        '''
        Compute K-Nearest Neighbors asteroid using pykep's KNN phasing method:
        https://esa.github.io/pykep/documentation/phasing.html?highlight=knn#pykep.phasing.knn

        Parameters:
            - time (float): Epoch to compute KNN
            - target_asteroid_id (int): Reference asteroid ID to
            compute its neighborhood
            - k (int): Compute the K-th nearest neighbors. (Default: 20)
        '''

        # Convert asteroids data to pykep planet objects
        asteroids_planet = []
        for idx in range(self.data.height):
            asteroid = self.data[idx]
            asteroids_planet.append(utils.convert_to_planet(asteroid))

        # Since version 1.24 numpy np.object is deprecated. Rename it as
        # a dummy variable (https://stackoverflow.com/questions/75069062/module-numpy-has-no-attribute-object)
        np.object = object
        knn = pk.phasing.knn(asteroids_planet,
                             T_START.mjd2000 + time,
                             T=180)
        _, ids, _ = knn.find_neighbours(asteroids_planet[target_asteroid_id],
                                        query_type="knn",
                                        k=k)

        # Filter out previous visited asteroids
        ids = [x for x in ids if x not in self.visited_asteroids]
        return ids

if __name__ == '__main__':

    datafile = "data/candidates.txt"
    rover = Rover(datafile)
    ids = rover.compute_knn(time=0,
                            target_asteroid_id=0,
                            k=20)
    logger.info("Extracted nearest asteroids: %s", ids)
