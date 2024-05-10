#!/usr/bin/env python3
'''
Rover class to
'''

import logging

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pykep as pk

import constants
import utils

import plotting


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s",
    level=logging.INFO)
logger = logging.getLogger(__name__)


# Start and end epochs
T_START = pk.epoch_from_iso_string(constants.ISO_T_START)
T_END = pk.epoch_from_iso_string(constants.ISO_T_END)

class Rover:

    def __init__(self,
                 datafile: str,
                 mission_window: int = constants.TIME_OF_MISSION):
        self.fuel = 1
        self.tank = [0] * 3
        self.score = 0
        self.visited_asteroids = []

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

            # Compute the ephemeris of the previous asteroid at departing time
            time_of_departure = time_of_arrival[idx - 1] + time_mining[idx - 1]

            # Compute time of flight
            time_of_flight = time_of_arrival[idx] - time_of_departure

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
            self.update_fuel(DV)

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

    def compute_optimal_journey(self,
                                source_asteroid_id: int,
                                destination_asteroid_id: int,
                                time_of_arrival: float,
                                N: int = 100,
                                step: float = 0.25):
        '''
        Compute the journey between two asteroids optimizing the fuel
        consumption

        Parameters:
            - source_asteroid_id(int): Source asteroid ID where the journey
            begins.
            - destination_asteroid_id(int): Destination asteroid ID where the
            journey ends.
            - time_of_arrival(float): Time of arrival to source asteroid,
            - N(int): Maximum value of time of flight into iterative process
            (Default 100)
            - step(float): Step value of time of flight into iterative process
            (Default 0.25)

        Returns:
            - None
        '''

        logger.info("Starting optimal journey between %s and %s",
                    source_asteroid_id,
                    destination_asteroid_id)
        # Extract asteroids data and transform to pykep planet objects
        source_asteroid_data = self.data.filter(
            pl.col("ID") == source_asteroid_id
        )
        source_asteroid = utils.convert_to_planet(
            source_asteroid_data
        )

        destination_asteroid_data = self.data.filter(
            pl.col("ID") == destination_asteroid_id
        )
        destination_asteroid = utils.convert_to_planet(
            destination_asteroid_data
        )

        # Mine the entire asteroid
        time_mining = self.compute_time_mining(source_asteroid_id)
        if time_of_arrival == 0:
            time_mining = 0

        # Compute time of departure from source to destination
        time_of_departure = time_of_arrival + time_mining

        # Compute the ephemeris of source asteroid at time
        r1, v1 = source_asteroid.eph(
            T_START.mjd2000 + time_of_departure
        )

        DV = []
        time_of_flights = np.arange(1, N, step)
        for time_of_flight in time_of_flights:
            _time_of_arrival = time_of_flight + time_of_departure

            # Compute the ephemeris of destination source at new
            # time of arrival
            r2, v2 = destination_asteroid.eph(
                T_START.mjd2000 + _time_of_arrival
            )

            # Compute Lambert for current ephemeris
            lambert = pk.lambert_problem(r1=r1,
                                         r2=r2,
                                         tof=time_of_flight * pk.DAY2SEC,
                                         mu=constants.MU_TRAPPIST,
                                         cw=False,
                                         max_revs=0)

            # Compute Delta-V
            DV1 = [a - b for a, b in zip(v1, lambert.get_v1()[0])]
            DV2 = [a - b for a, b in zip(v2, lambert.get_v2()[0])]
            DV.append(np.linalg.norm(DV1) + np.linalg.norm(DV2))

        # Extract the min DV and its related time of flight
        min_DV = min(DV)
        min_time_of_flight = time_of_flights[DV.index(min_DV)]

        # Compute new time of arrival
        min_time_of_arrival = min_time_of_flight + time_of_departure

        # Update tank values
        self.update_tank(source_asteroid_data)
        self.update_fuel(min_DV)

        # Update score
        self.score = min(self.tank)

        logger.info("Optimal journey between %s and %s: DV = %s, tof=%s, tarr=%s, tm=%s",
                    source_asteroid_id,
                    destination_asteroid_id,
                    min_DV,
                    min_time_of_flight,
                    min_time_of_arrival,
                    time_mining)
        return min_DV, min_time_of_arrival, time_mining

    def compute_time_mining(self,
                            asteroid_id: int) -> float:
        '''
        Compute time spent mining in an asteroid.

        Parameters:
            - asteroid_id(int): ID of asteroid.

        Returns:
            - float: Time spent mining.
        '''

        asteroid_data = self.data.filter(
            pl.col("ID") == asteroid_id
        )

        return asteroid_data["Mass [0 to 1]"].item() *\
            constants.TIME_TO_MINE_FULLY


    def update_fuel(self,
                    delta_v: float):
        '''
        Update fuel according to DV.

        Parameters:
            - DV (float): DV consumption into the journey
        '''

        fuel_consumption = delta_v / constants.DV_per_fuel
        logger.info("Fuel consumption = %s", fuel_consumption)
        self.fuel -= fuel_consumption
        logger.info("Fuel level = %s", self.fuel)
        if self.fuel <= 0:
            logger.error("OUT OF FUEL!!")
            raise Exception("OUT OF FUEL!")

    def update_tank(self,
                    asteroid_data: pl.DataFrame):
        '''
        Update tank after mine the entire asteroid
        '''

        # Get material type of asteroid
        material_type = asteroid_data["Material Type"].item()

        # Compute material collected
        material_collected = asteroid_data["Mass [0 to 1]"].item()

        if material_type == 3:
            self.fuel += material_collected
        else:
            self.tank[material_type] += material_collected



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
        print(self.visited_asteroids)
        ids = [x for x in ids if x not in self.visited_asteroids]
        return ids

def main():
    '''
    Main function
    '''

    datafile = "data/candidates.txt"

    scores = []
    for first_asteroid_id in range(0, 1):
        # Initalize a rover starting its journey into the first asteroid
        rover = Rover(datafile)
        logger.error("STARTING ASTEROID %s", first_asteroid_id)
        rover.visited_asteroids.append(first_asteroid_id)

        # Initialize time of arrival and mining of current jorney
        times_of_arrival = [0]
        times_mining = []

        # Init the journey starting from the first asteroid previously
        # selected.
        for asteroid in rover.visited_asteroids:

            # Extract the neighborhood of current asteroid
            neigh_ids = rover.compute_knn(time=0,
                                          target_asteroid_id=asteroid,
                                          k=50)

            logger.info("Neighbors of %s:%s",
                        asteroid,
                        neigh_ids)
            # Compute the optimal journey to the nearest asteroid
            try:
                DV, time_of_arrival, time_mining = rover.compute_optimal_journey(
                    source_asteroid_id=asteroid,
                    destination_asteroid_id=neigh_ids[0],
                    time_of_arrival=times_of_arrival[-1],
                )
                times_mining.append(time_mining)
                # Append nearest asteroid to list of asteroids and update
                # the rover visited asteroids list
                rover.visited_asteroids.append(neigh_ids[0])
                times_of_arrival.append(time_of_arrival)
            except Exception as error:
                # If an exception occurs is because of the journey between
                # asteroids is not possible, but the rover mine the source
                # asteroid anyway.
                time_mining = rover.compute_time_mining(asteroid)
                if time_of_arrival == 0: # Avoid mine the first asteroid
                    time_mining = 0
                times_mining.append(time_mining)
                break

        logger.info("Asteroids: %s", rover.visited_asteroids)
        logger.info("Time of arrival: %s", times_of_arrival)
        logger.info("Time mining: %s", times_mining)
        logger.info("Score = %s", rover.score)

        # Use another rover to evalute the computed journey
        evaluation_rover = Rover(datafile)
        evaluation_rover.compute_journey(rover.visited_asteroids,
                            times_of_arrival,
                            times_mining)

if __name__ == '__main__':
    main()


