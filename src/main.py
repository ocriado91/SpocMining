#!/usr/bin/env python3

'''
Main method of SpOC Mining Challenge
'''

import logging
import sys

import pandas as pd

from rover import Rover

# Set logging level and format
logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HEADER = ["ID",
          "Semi-major axis [m]",
          "Eccentricity",
          "Inclination [rad]",
          "Ascending Node [rad]",
          "Argument of Periapsis [rad]",
          "True Anomaly [rad]",
          "Mass [0 to 1]",
          "Material Type"]

def example(asteroids: list):
    ''' Example data from SpOC Mining Challenge'''

    target_asteroids = [0,
                1446,
                5131,
                4449,
                8091,
                1516,
                151,
                4905,
                8490]

    arrival_time = [0,
                    11.0,
                    45.98091676982585,
                    98.86574387748259,
                    144.3421379448264,
                    178.78720680368133,
                    198.49061810149578,
                    236.39180345018394,
                    268.4772894184571]

    mining_time = [0,
                   18.980916769828053,
                   22.88482710766111,
                   29.47639406736512,
                   17.445068858837555,
                   18.703411297804774,
                   19.901185348707877,
                   24.085485968277332,
                   17.543366859589646]

    rover = Rover(asteroids)

    rover.evaluate_journey(target_asteroids,
                           arrival_time,
                           mining_time)

def main():
    '''
    Main method
    '''

    if len(sys.argv) != 2:
        logger.error('Please provide an asteroids candidates file')
        sys.exit(1)

    candidates_file = sys.argv[1]
    asteroids = pd.read_csv(candidates_file,
        sep=' ',
        names=HEADER,
        index_col=0)

    example(asteroids)

if __name__ == '__main__':
    main()
