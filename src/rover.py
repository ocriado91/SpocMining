#!/usr/bin/env python3
'''
Implementation of a Rover class to manage journey between asteroids. This class
allows to compute different methods to optimize the journey between asteroids.
'''

import logging

import argparse
import numpy as np
import polars as pl
import pykep as pk

import constants
import utils

np.seterr(divide='ignore', invalid='ignore')


# Start and end epochs
T_START = pk.epoch_from_iso_string(constants.ISO_T_START)
T_END = pk.epoch_from_iso_string(constants.ISO_T_END)

# Generate a exception to handle out of fuel error
class OutOfFuelException(Exception):
    ''' Raises when out-of-fuel error'''

class Rover:
    '''
    Rover class to travel between asteroids.
    '''
    def __init__(self,
                 datafile: str,
                 mission_window: int = constants.TIME_OF_MISSION):
        self.fuel = 1
        self.tank = [0] * 3
        self.score = 0
        self.visited_asteroids = []

        self.mission_window = mission_window

        self.datafile = datafile
        self._read_datafile()

    def _read_datafile(self):
        '''
        Read asteroid candidates file and store into rover data attribute

        Parameters:
            - datafile (str): Asteroid candidates file
        '''

        logging.info("Reading datafile %s", self.datafile)

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
        data = pl.read_csv(self.datafile,
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

            logging.info("Computing journey between %s and %s",
                        previous_asteroid_id, current_asteroid_id)
            # Check if mission window is reached
            if time_of_arrival[idx] - time_of_arrival[0] > self.mission_window:
                logging.error("Mission window exceeded")
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
            delta_v1 = [a - b for a, b in zip(v1, lambert.get_v1()[0])]
            delta_v2 = [a - b for a, b in zip(v2, lambert.get_v2()[0])]
            delta_v = np.linalg.norm(delta_v1) + np.linalg.norm(delta_v2)

            # Compute fuel consumption
            self.update_fuel(delta_v)

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
            logging.info("Traveling from %s to %s with a delta_v = %s. Fuel = %s. Tank = %s. Score = %s",
                         previous_asteroid_id,
                         current_asteroid_id,
                         delta_v,
                         self.fuel,
                         self.tank,
                         self.score)

    def compute_optimal_journey(self,
                                source_asteroid_id: int,
                                destination_asteroid_id: int,
                                time_of_arrival: float,
                                N: int = 100,
                                step: float = 0.25) -> (float, float, float):
        '''
        Compute the journey between two asteroids optimizing the fuel
        consumption. This method is based into determine the time of flight
        iteratively between 1 and N, extracting the minimum DV required to
        travel within asteroids and its related time of arrival to the
        destination asteroid.

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

        logging.info("Starting optimal journey between %s and %s",
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

        # Mine the entire asteroid. Avoid to mine the first asteroid
        # (detected as time_of_arrival = 0 according to challenge
        # description)
        time_mining = self.compute_time_mining(source_asteroid_id)
        if time_of_arrival == 0:
            time_mining = 0

        # Compute time of departure from source to destination
        time_of_departure = time_of_arrival + time_mining

        # Compute the ephemeris of source asteroid at time
        r1, v1 = source_asteroid.eph(
            T_START.mjd2000 + time_of_departure
        )

        delta_v = []
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
            delta_v1 = [a - b for a, b in zip(v1, lambert.get_v1()[0])]
            delta_v2 = [a - b for a, b in zip(v2, lambert.get_v2()[0])]
            delta_v.append(np.linalg.norm(delta_v1) + np.linalg.norm(delta_v2))

        # Extract the min DV and its related time of flight
        min_delta_v = min(delta_v)
        min_time_of_flight = time_of_flights[delta_v.index(min_delta_v)]

        # Compute new time of arrival
        min_time_of_arrival = min_time_of_flight + time_of_departure

        # Reduce fuel level according to consumed fuel
        self.update_fuel(min_delta_v)

        # Update tank. Avoid to mine the first asteroid.
        if time_of_arrival != 0:
            self.update_tank(source_asteroid_data)

        # Update score
        self.score = min(self.tank)

        logging.debug("Optimal journey between %s and %s: delta_v = %s, tof=%s, tarr=%s, tm=%s",
                    source_asteroid_id,
                    destination_asteroid_id,
                    min_delta_v,
                    min_time_of_flight,
                    min_time_of_arrival,
                    time_mining)
        logging.info("Traveling from %s to %s with a delta_v = %s. Fuel = %s. Tank = %s. Score = %s",
                         source_asteroid_id,
                         destination_asteroid_id,
                         min_delta_v,
                         self.fuel,
                         self.tank,
                         self.score)
        return min_delta_v, min_time_of_arrival, time_mining

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
            - delta_v (float): delta_v consumption into the journey
        '''

        fuel_consumption = delta_v / constants.DV_per_fuel
        logging.info("Fuel consumption = %s", fuel_consumption)
        self.fuel -= fuel_consumption
        logging.info("Fuel level = %s", self.fuel)
        if self.fuel <= 0:
            logging.error("OUT OF FUEL!!")
            raise OutOfFuelException

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
            if self.fuel > 1:
                self.fuel = 1
            logging.info("Updated fuel: %s", self.fuel)
        else:
            self.tank[material_type] += material_collected
            logging.info("Updated tank: %s", self.tank)


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

    def rate_candidates(self,
                        asteroid_candidate: int):
        '''
        Rate asteroid candidate in based on its material
        '''

        material_rates = np.sum(self.tank) / self.tank

        # Normalize material rates
        material_rates /= np.sum(material_rates)

        # Convert NaN values (incoming from 0 into tank mass)
        if np.count_nonzero(np.isnan(material_rates)):
            material_rates = np.where(np.isnan(material_rates),
                                    1/np.count_nonzero(np.isnan(material_rates)),
                                    material_rates)

        # Extract material type of candidate asteroid
        material_type = self.data.filter(
            pl.col("ID") == asteroid_candidate
        )["Material Type"].item()


        if material_type == 3: # Propellant
            return 1

        logging.debug("Returning material rate %s for asteroid %s of type %s",
                      material_rates[material_type],
                      asteroid_candidate,
                      material_type)
        return material_rates[material_type]

    def compute_aco(self,
                    rovers: int = 100,
                    iterations: int = 1000,
                    first_asteroid_id: int = 0):
        '''
        Compute the Ant Colony Optimization algorithm

        Parameters:
            - rovers (int): Number of exploratory rovers (ants) (Default: 100)
            - iterations (int): Number of ACO iterations (Default: 1000)
            - first_asteroid_id (int): Asteroid ID to become the algorithm (Default: 0)
        '''

        # Initialize the pheromone matrix
        pheromone = np.ones((self.data.height,
                            self.data.height))
        # Initialize best score
        best_score = 0
        # Start iteration
        for iteration in range(iterations):
            logging.info("Starting ACO iteration %s", iteration)
            # Each rover becomes its journey
            for rover_id in range(rovers):

                # Initialize a new rover object
                rover = Rover(datafile=self.datafile)

                # Initialize the visited asteroid list
                visited = [False] * rover.data.height

                # Start the journey into the first asteroid
                current_asteroid = first_asteroid_id

                # Add current asteroid to list of visited asteroids
                visited[current_asteroid] = True

                # Initialize time of arrival list
                time_of_arrival = [0]
                time_mining = []
                asteroids = [current_asteroid]

                logging.info("Starting rover %s journey...", rover_id)
                # Try to visit all asteroids
                while False in visited and \
                    time_of_arrival[-1] <= constants.TIME_OF_MISSION:

                    # Extract the neighborhood of current asteroid
                    neigh_ids = rover.compute_knn(time=time_of_arrival[-1],
                                                  target_asteroid_id=current_asteroid,
                                                  k=5)
                    # Extract the list of unvisited asteroids
                    unvisited = [idx for idx, x in enumerate(visited) if not x]

                    # Extract unvisited neighbors
                    unvisited_neigh = [x for x in neigh_ids if x in unvisited]
                    logging.debug("Unvisited neighbors of %s: %s",
                                  current_asteroid,
                                  unvisited_neigh)

                    # Compute the probability of go to another unvisited
                    # asteroid from current asteroid based on its potential
                    # resources
                    probabilities = [0] * len(unvisited_neigh)

                    for idx, unvisited_asteroid in enumerate(unvisited_neigh):
                        unvisited_rate = rover.rate_candidates(
                            unvisited_asteroid
                        )
                        probability = pheromone[current_asteroid,
                                                unvisited_asteroid] * unvisited_rate
                        probabilities[idx] = probability

                    # If all probabilities are 0 (any neighbor asteroid found
                    # with desired material), assign the same value to
                    # probabilities array
                    if probabilities.count(0) == len(probabilities):
                        probabilities = [1] * len(probabilities)

                    # Normalize probabilities
                    probabilities /= np.sum(probabilities)

                    # Select next asteroid based on probabilities
                    next_asteroid = np.random.choice(unvisited_neigh,
                                                     p=probabilities)
                    logging.info("Selected next asteroid: %s", next_asteroid)

                    # Move to the next point
                    try:
                        # Compute the optimal journey between current asteroid
                        # and next one.
                        DV, _time_of_arrival, _time_mining = \
                            rover.compute_optimal_journey(
                                source_asteroid_id=current_asteroid,
                                destination_asteroid_id=next_asteroid,
                                time_of_arrival=time_of_arrival[-1]
                            )
                        # Add next asteroids and visited asteroids lists
                        asteroids.append(next_asteroid)
                        visited[next_asteroid] = True

                        # Append time of arrival and mining to lists
                        time_of_arrival.append(_time_of_arrival)
                        time_mining.append(_time_mining)

                        # Update pheromone
                        pheromone[current_asteroid, next_asteroid] += rover.rate_candidates(next_asteroid)
                        logging.info("Updated pheromone between %s and %s: %s",
                                     current_asteroid,
                                     next_asteroid,
                                     pheromone[current_asteroid, next_asteroid])
                        current_asteroid = next_asteroid
                    except OutOfFuelException:
                        time_mining.append(rover.compute_time_mining(current_asteroid))
                        logging.info("SCORE = %s", rover.score)
                        if rover.score > best_score:
                            best_score = rover.score
                            logging.info("New best score (%s) with: %s, %s, %s",
                                        best_score,
                                        asteroids,
                                        time_of_arrival,
                                        time_mining)
                        break
        logging.info("BEST SCORE = %s", best_score)
        return asteroids, time_of_arrival, time_mining

def configure_logging(args: argparse.ArgumentParser):
    '''
    Setup logging object
    '''

    if args.log_file:
        logging.basicConfig(
            filename=args.log_file,
            format="%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s",
            level=args.log_level)
    else:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s",
            level=args.log_level)


def argument_parser():
    '''
    Function to retrieve CLI arguments
    '''

    args = argparse.ArgumentParser()
    args.add_argument("--iterations",
                      help="Number of iterations to be executed",
                      type=int,
                      default=100)
    args.add_argument("--rovers",
                      help="Number of rovers (ants)",
                      type=int,
                      default=10)
    args.add_argument("--log_level",
                      help="Level of log",
                      choices=["CRITICAL",
                               "ERROR",
                               "WARNING",
                               "INFO",
                               "DEBUG"],
                      default="INFO")
    args.add_argument("--log_file",
                      help="Filename of log")

    return args.parse_args()

def main():
    '''
    Main function
    '''
    # Read CLI arguments
    args = argument_parser()

    # Setup logging
    configure_logging(args)

    # Initialize rover object
    datafile = "data/candidates.txt"
    rover = Rover(datafile=datafile)

    # Execute ACO algorithm
    asteroids, time_of_arrival, time_mining = \
        rover.compute_aco(iterations=args.iterations,
                          rovers=args.rovers)

    # Show results
    logging.info("Asteroids: %s", asteroids)
    logging.info("Time of arrival: %s", time_of_arrival)
    logging.info("Time mining: %s", time_mining)

if __name__ == '__main__':
    main()

