#!/usr/bin/env python3

'''
Rover class
'''
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pykep as pk

# Set logging level and format
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Start and end epochs
ISO_T_START = "30190302T000000"
ISO_T_END = "30240302T000000"

# Cavendish constant (m^3/s^2/kg)
G = 6.67430e-11

# Sun_mass (kg)
SM = 1.989e30

# Mass and Mu of the Trappist-1 star
MS = 8.98266512e-2 * SM
MU_TRAPPIST = G * MS

# DV per propellant [m/s]
DV_PER_PROPELLANT = 10000

# Maximum time to fully mine an asteroid
TIME_TO_MINE_FULLY = 30

# Eccentric anomaly constants
MAX_ITER_E = 15
EPSILON_E = 1e-10

T_START = pk.epoch_from_iso_string(ISO_T_START)
T_END = pk.epoch_from_iso_string(ISO_T_END)


def convert_to_planet_object(asteroid: pd.DataFrame) -> pk.planet:
    '''
    Convert asteroid pandas Dataframe to pykep planet object
    '''

    return pk.planet.keplerian(
        pk.epoch_from_iso_string(ISO_T_START),
        (
            asteroid['Semi-major axis [m]'].values[0],
            asteroid['Eccentricity'].values[0],
            asteroid['Inclination [rad]'].values[0],
            asteroid['Ascending Node [rad]'].values[0],
            asteroid['Argument of Periapsis [rad]'].values[0],
            asteroid['True Anomaly [rad]'].values[0]
        ),
        MU_TRAPPIST,
        G*asteroid['Mass [0 to 1]'].values[0],
        1,
        1.1,
        "Asteroid " + str(asteroid.index.values[0])
    )


def compute_time_of_flight(target_arrival_time: float,
                           source_arrival_time: float,
                           source_mining_time: float) -> float:
    '''
    Time of Flight = arrival timestamp - departure timestamp
    begin departure timestamp = arrival at source asteroid timetamp
    # - mining spent at source asteroid
    '''

    return target_arrival_time - source_arrival_time - source_mining_time


class Rover:
    '''
    Rover class
    '''

    def __init__(self,
                 asteroids: pd.DataFrame):

        self.asteroids = asteroids

        self.source_ast = None
        self.target_ast = None

        self.prepared_mass1 = 0
        self.prepared_mass2 = 0
        self.prepared_mass3 = 0

        self.propellant = 1

    def evaluate_journey(self,
                         asteroids_id: list,
                         arrival_time: list,
                         mining_time: list,
                         verbose: bool = True):
        '''
        Method to evalute journey between asteroids
        '''

        ast_indexes = self.asteroids.index
        for idx in range(len(asteroids_id)-1):
            source_index = asteroids_id[idx]
            target_index = asteroids_id[idx+1]
            source_ast_df = self.asteroids[ast_indexes.isin([source_index])]
            target_ast_df = self.asteroids[ast_indexes.isin([target_index])]

            # Convert to planet object
            self.source_ast = convert_to_planet_object(source_ast_df)
            self.target_ast = convert_to_planet_object(target_ast_df)

            # Compute remaining propellant
            delta_v = self.compute_delta_v(arrival_time[idx+1],
                                           arrival_time[idx],
                                           mining_time[idx])
            logger.info('Computing delta-V between %d and %d = %f',
                        source_index,
                        target_index,
                        delta_v)

            self.propellant = self.propellant -\
                delta_v / DV_PER_PROPELLANT

            logger.info("Remaining propellant = %f", self.propellant)

            # Journey is possible, extract material info
            material_type = source_ast_df['Material Type'].values[0]
            extracted_mass = mining_time[idx] / TIME_TO_MINE_FULLY

            logger.info("Extracted %f from asteroid (%d) type %d",
                        extracted_mass,
                        source_index,
                        material_type)

            # Add mass to propellant
            if material_type == 3:
                self.propellant += extracted_mass
                logger.info('Detected propellant resource. Current total = %f',
                            self.propellant)

            elif material_type == 2:
                self.prepared_mass3 += extracted_mass

            elif material_type == 1:
                self.prepared_mass2 += extracted_mass

            elif material_type == 0:
                self.prepared_mass1 += extracted_mass

            logger.info("Current prepared mass = %f | %f | %f",
                        self.prepared_mass1,
                        self.prepared_mass2,
                        self.prepared_mass3)

            # Update score
            score = self.compute_score()
            if verbose:
                print(f'{target_index}\t'
                      f'{arrival_time[idx]:<4.2f}\t'
                      f'{self.propellant:<14.2f}'
                      f'{delta_v:<8.2f}\t'
                      f'{material_type}\t'
                      f'{self.prepared_mass1:<8.2f}'
                      f'{self.prepared_mass2:<8.2f}'
                      f'{self.prepared_mass3:<8.2f}'
                      f'{score:<.2f}')

            if self.propellant < 0:
                logger.error("Out of propellant")
                break

    def compute_score(self) -> float:
        '''
        Compute score as minimum of mass of the 3 types
        '''

        return min(self.prepared_mass1,
                   self.prepared_mass2,
                   self.prepared_mass3)

    def compute_delta_v(self,
                        target_arrival_time: float,
                        source_arrival_time: float,
                        source_mining_time: float):
        '''
        Compute Delta-V necessary to go there and match its velocity
        '''

        tof = compute_time_of_flight(target_arrival_time,
                                     source_arrival_time,
                                     source_mining_time)

        # Compute ephemeris of source asteroid
        t_1 = T_START.mjd2000 + source_arrival_time + source_mining_time
        r_1, v_1 = self.source_ast.eph(t_1)

        # Compute ephemeris of target asteroid
        t_2 = T_START.mjd2000 + target_arrival_time
        r_2, v_2 = self.target_ast.eph(t_2)

        lambert_solution = pk.lambert_problem(
            r1=r_1,
            r2=r_2,
            tof=tof * pk.DAY2SEC,
            mu=MU_TRAPPIST,
            cw=False,
            max_revs=0)

        delta_v1 = [a - b for a, b in zip(v_1, lambert_solution.get_v1()[0])]
        delta_v2 = [a - b for a, b in zip(v_2, lambert_solution.get_v2()[0])]
        return np.linalg.norm(delta_v1) + np.linalg.norm(delta_v2)

    def plot_asteroids(self,
                       start_time: float = 0.0,
                       end_time: float = 250.0,
                       figurename: str = 'asteroids.png') -> None:
        '''
        Method for generate orbit plot between source asteroid
        and target asteroid attributes
        '''
        _, axes = plt.subplots(subplot_kw={'projection': '3d'})
        pk.orbit_plots.plot_planet(self.source_ast,
                                   axes=axes,
                                   t0=start_time,
                                   tf=end_time,
                                   legend=(False, True),
                                   color='b')
        pk.orbit_plots.plot_planet(self.target_ast,
                                   axes=axes,
                                   t0=start_time,
                                   tf=end_time,
                                   legend=(False, True),
                                   color='c')
        plt.savefig(figurename)
