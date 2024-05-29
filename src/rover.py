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


import sys

import constants
import utils

np.seterr(divide='ignore', invalid='ignore')


# Start and end epochs
T_START = pk.epoch_from_iso_string(constants.ISO_T_START)
T_END = pk.epoch_from_iso_string(constants.ISO_T_END)

# Generate a exception to handle out of fuel error
class OutOfFuelException(Exception):
    ''' Raises when out-of-fuel error'''

class ReachedEndOfMission(Exception):
    ''' Reached end of mission window time '''

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

        logging.debug("Reading datafile %s", self.datafile)

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
                        time_mining: list) -> None:
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

            logging.debug("Computing journey between %s and %s",
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
                if self.fuel > 1:
                    self.fuel = 1
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
                                step: float = 0.25,
                                verbose: bool = True) -> (float, float, float):
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
        if verbose:
            logging.debug("Starting optimal journey between %s and %s",
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
        if min_time_of_arrival > constants.TIME_OF_MISSION:
            raise ReachedEndOfMission

        # Reduce fuel level according to consumed fuel
        self.update_fuel(min_delta_v)

        # Update tank with destination resources
        self.update_tank(destination_asteroid_data)

        # Update score
        self.score = min(self.tank)

        if verbose:
            logging.debug("Optimal journey between %s and %s: delta_v = %s, tof=%s, tarr=%s, tm=%s",
                        source_asteroid_id,
                        destination_asteroid_id,
                        min_delta_v,
                        min_time_of_flight,
                        min_time_of_arrival,
                        time_mining)
            logging.debug("Traveling from %s to %s with a delta_v = %s. Fuel = %s. Tank = %s. Score = %s",
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
        logging.debug("Fuel consumption = %s", fuel_consumption)
        self.fuel -= fuel_consumption
        logging.debug("Fuel level = %s", self.fuel)
        if self.fuel <= 0:
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

        asteroid_id = asteroid_data["ID"].item()
        logging.debug("Mining asteroid %s of type %s with collected mass = %s",
                     asteroid_id,
                     material_type,
                     material_collected)
        if material_type == 3:
            self.fuel += material_collected
            if self.fuel > 1:
                self.fuel = 1
            logging.debug("Updated fuel: %s", self.fuel)
        else:
            self.tank[material_type] += material_collected
            logging.debug("Updated tank: %s", self.tank)


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

    def material_rates(self) -> list:
        '''
        Extract the rate related with current tank status
        '''

        material_rates = [1/len(self.tank)] * len(self.tank)
        if np.count_nonzero(self.tank):
            for idx, _tank in enumerate(self.tank):
                if _tank != 0:
                    material_rates[idx] = np.sum(self.tank) / _tank
                else:
                    # We want to prioritize the minimum mass collected. If
                    # there is any tank value with 0, assign a rate of a high
                    # value (eg. 100) to prioritize the collect of this material.
                    material_rates[idx] = 100
        # Normalize material rates
        material_rates /= np.sum(material_rates)

        return material_rates

    def rate_candidates(self,
                        current_asteroid: int,
                        asteroid_candidate: int,
                        time_of_arrival: float):
        '''
        Rate asteroid candidate in based on its material
        '''

        logging.debug("Checking rate between %s and %s",
                     current_asteroid,
                     asteroid_candidate)

        # Calculate the material rates pending on current tank status.
        material_rates = self.material_rates()

        _time_of_arrival = 0
        rover = Rover(self.datafile)
        try:
            DV, _time_of_arrival, _time_mining = rover.compute_optimal_journey(
                source_asteroid_id=current_asteroid,
                destination_asteroid_id=asteroid_candidate,
                time_of_arrival=time_of_arrival,
                verbose=False
            )
        except OutOfFuelException:
            logging.debug("Out-of-fuel traveling to asteroid %s rate = 0",
                            asteroid_candidate)
            return 0
        except ReachedEndOfMission:
            logging.debug("Time of arrival to %s exceeds mission window %s",
                            asteroid_candidate,
                            constants.TIME_OF_MISSION)
            return 0

        fuel_consumption = DV / constants.DV_per_fuel
        fuel_rate = self.fuel / fuel_consumption

        # Extract material type of candidate asteroid
        material_type = self.data.filter(
            pl.col("ID") == asteroid_candidate
        )["Material Type"].item()

        if material_type == 3: # Propellant
            if self.fuel != 1:
                return fuel_rate
            return 0

        # Set total rate
        total_rate = material_rates[material_type] * fuel_rate
        logging.debug("Material rate %s for asteroid %s of type %s | Fuel rate %s | Total rate %s",
                      material_rates[material_type],
                      asteroid_candidate,
                      material_type,
                      fuel_rate,
                      total_rate)
        return total_rate

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
        # Initialize best score variable.
        # Score defined as minimum material mass collected of the three types.
        best_score = 0

        # Initialize the best solution lists
        best_asteroids = []
        best_time_of_arrival = []
        best_time_mining = []

        # Start iteration
        for iteration in range(iterations):
            logging.info("Starting ACO iteration %s", iteration)

            # Each rover becomes its journey into the current iteration
            for rover_id in range(rovers):

                # Initialize a new rover object
                rover = Rover(datafile=self.datafile)

                # Initialize the visited asteroid list
                visited = [False] * rover.data.height

                # Start the journey with the first asteroid
                current_asteroid = first_asteroid_id

                # Add current asteroid (first asteroid)
                # to list of visited asteroids
                visited[current_asteroid] = True

                # Initialize time of arrival, time mining and asteroid lists
                time_of_arrival = [0]
                time_mining = []
                asteroids = [current_asteroid]

                logging.info("Starting rover %s journey...", rover_id)

                # Try to visit all asteroids within time of mission window
                while False in visited and \
                    time_of_arrival[-1] <= constants.TIME_OF_MISSION:

                    logging.debug("Current tank %s and fuel %s at %s",
                                 rover.tank,
                                 rover.fuel,
                                 time_of_arrival[-1])

                    # Extract the neighborhood of current asteroid
                    neigh_ids = rover.compute_knn(
                        time=time_of_arrival[-1],
                        target_asteroid_id=current_asteroid,
                        k=50
                    )

                    # Extract the list of unvisited asteroids
                    unvisited = [idx for idx, x in enumerate(visited) if not x]

                    # Extract unvisited neighbors
                    unvisited_neigh = [x for x in neigh_ids if x in unvisited]

                    # Compute the probability of go to another unvisited
                    # asteroid from current asteroid based on its potential
                    # resources computed as rates
                    probabilities = [0] * len(unvisited_neigh)

                    for idx, unvisited_asteroid in enumerate(unvisited_neigh):
                        unvisited_rate = rover.rate_candidates(
                            current_asteroid=current_asteroid,
                            asteroid_candidate=unvisited_asteroid,
                            time_of_arrival=time_of_arrival[-1]
                        )
                        probability = pheromone[current_asteroid,
                                                unvisited_asteroid] * unvisited_rate
                        probabilities[idx] = probability

                    # If all probabilities are 0 (none neighbor asteroid found
                    # with desired material), assign the same value to
                    # probabilities array
                    if probabilities.count(0) == len(probabilities):
                        probabilities = [1] * len(probabilities)

                    # Normalize probabilities
                    probabilities /= np.sum(probabilities)

                    # Select next asteroid based on probabilities
                    next_asteroid = np.random.choice(unvisited_neigh,
                                                     p=probabilities)
                    logging.debug("Probabilities = %s of unvisited = %s and selected = %s",
                                 probabilities,
                                 unvisited_neigh,
                                 next_asteroid)

                    # Compute the rate of next asteroid to update
                    next_asteroid_rate = rover.rate_candidates(
                        current_asteroid=current_asteroid,
                        asteroid_candidate=next_asteroid,
                        time_of_arrival=time_of_arrival[-1])

                    # Update pheromone as asteroid rate
                    # pheromone[current_asteroid, next_asteroid] +=  next_asteroid_rate
                    pheromone[current_asteroid, next_asteroid] += 1
                    logging.debug("Updated pheromone between %s and %s with a rate of %s: %s",
                                    current_asteroid,
                                    next_asteroid,
                                    next_asteroid_rate,
                                    pheromone[current_asteroid, next_asteroid])

                    # Travel to next asteroid
                    try:
                        # Compute the optimal journey between current asteroid
                        # and next one.
                        _, _time_of_arrival, _time_mining = \
                            rover.compute_optimal_journey(
                                source_asteroid_id=current_asteroid,
                                destination_asteroid_id=next_asteroid,
                                time_of_arrival=time_of_arrival[-1]
                            )

                        if _time_of_arrival >= constants.TIME_OF_MISSION:
                            logging.error("Reached end of mission!!")
                            break

                        # Add next asteroid to total asteroid and
                        # visited asteroids lists
                        asteroids.append(next_asteroid)
                        visited[next_asteroid] = True

                        # Append time of arrival and mining to lists
                        time_of_arrival.append(_time_of_arrival)
                        time_mining.append(_time_mining)

                        # Arrival at next asteroid
                        current_asteroid = next_asteroid

                    except OutOfFuelException:
                        # Current rover is out of fuel. Break the while loop
                        # to move to the next rover
                        break
                    except ReachedEndOfMission:
                        # Current journey exceeds the window time of mission.
                        # Break the while loop to move to the next rover
                        break

                # End of while loop (time mission reached or rover out of fuel)
                time_mining.append(rover.compute_time_mining(current_asteroid))
                logging.debug("Asteroids: %s", asteroids)
                logging.debug("Score = %s", rover.score)
                if rover.score > best_score:
                    best_score = rover.score
                    best_asteroids = asteroids
                    best_time_of_arrival = time_of_arrival
                    best_time_mining = time_mining
                    logging.info("New best score (%s) with: \n%s, \n%s, \n%s",
                                best_score,
                                best_time_of_arrival,
                                best_time_mining,
                                best_asteroids)
        logging.info("BEST SCORE = %s", best_score)
        return best_asteroids, best_time_of_arrival, best_time_mining

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

    # Evaluation rover
    eval_rover = Rover(datafile=datafile)
    eval_rover.compute_journey(asteroids=asteroids,
                               time_of_arrival=time_of_arrival,
                               time_mining=time_mining)

if __name__ == '__main__':
    main()

