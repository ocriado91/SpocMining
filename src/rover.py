# -*- coding: utf-8 -*-

"""
Rover class definition file
"""
#######################################################################
# Imports area
#######################################################################

# Generic / Built-in
import numpy as np
import logging

# Other Libs
import pykep as pk

# Own Libs
from src import constants
from src import asteroid

#######################################################################

# Set logging level and format and save to file
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('logs/rover.log'))

T_START = pk.epoch_from_iso_string(constants.ISO_T_START)
T_END = pk.epoch_from_iso_string(constants.ISO_T_END)

class Rover:
    """
    Rover class

    TBC

    """
    def __init__(self,
                 asteroids: list,
                 missionWindow: int = 1827):

        self.asteroids = asteroids
        self.propellant = 1
        self.goldMass = 0
        self.platinumMass = 0
        self.nickelMass = 0
        self.missionWindow = missionWindow
        self.missionTime = 0

    def evaluate_journey(self,
                         sourceAsteroidId: int = 0,
                         sourceArrivalTime: float = 0.0,
                         sourceMiningTime: float = 0.0,
                         targetAsteroidId: int = 0,
                         targetArrivalTime: float = 0.0,)->bool:
        '''
        Evaluate rover journey between
        source and target asteroid through
        Lambert solution
        '''
        if self.missionTime > self.missionWindow:
            logger.error('Reach mission window time {}'\
                .format(self.missionWindow))
            return False

        self.sourceAsteroid = self.asteroids[sourceAsteroidId]
        self.targetAsteroid = self.asteroids[targetAsteroidId]

        logger.info('Init Rover journey between asteroids {} and {}'\
            .format(self.sourceAsteroid.asteroidId,
                    self.targetAsteroid.asteroidId))

        # tof (Time Of Flight) = arrival timestamp - departure timestamp
        # being departure timestamp = arrival at source asteroid timestamp
        #  - mining spent at source asteroid
        tof = targetArrivalTime - sourceArrivalTime - sourceMiningTime

        # Compute ephemeris of source asteroid
        t1 = T_START.mjd2000 + sourceArrivalTime  + sourceMiningTime
        r1, v1 = self.sourceAsteroid.planetObject.eph(t1)

        # Compute ephemeris of target asteroid
        t2 = T_START.mjd2000 + targetArrivalTime
        r2, v2 = self.targetAsteroid.planetObject.eph(t2)

        logger.info('tof = {} '.format(tof))

        l = pk.lambert_problem(
                r1=r1,
                r2=r2,
                tof=tof * pk.DAY2SEC,
                mu=constants.MU_TRAPPIST,
                cw=False,
                max_revs=0)

        # Compute the delta-v necessary to go there and match its velocity
        DV1 = [a - b for a, b in zip(v1, l.get_v1()[0])]
        DV2 = [a - b for a, b in zip(v2, l.get_v2()[0])]
        DV = np.linalg.norm(DV1) + np.linalg.norm(DV2)
        logger.info('Computing Delta-V equal to {}'.format(DV))

        # Compute propellant used for this transfer and update ship
        # propellant level
        self.propellant = self.propellant - DV / constants.DV_PER_PROPELLANT
        if self.propellant < 0:
            logger.error("Out of propellant!")
            return False

        # This journey is possible, extract the material of source asteroid
        # and accumulate to the rover
        materialType = self.targetAsteroid.materialType

        # Compute spent time to mining the entire target asteroid
        targetMiningTime = self.targetAsteroid.normalizedMass *\
            constants.TIME_TO_MINE_FULLY
        logger.info('Target mining time {}'.format(targetMiningTime))

        # Compute prepared material and add its related material counter
        preparedMaterial = targetMiningTime / constants.TIME_TO_MINE_FULLY

        if materialType == 'Gold':
            logger.info('Extracting Gold material: {}'\
                .format(preparedMaterial))
            self.goldMass += preparedMaterial

        elif materialType == 'Platinum':
            logger.info('Extracting Platinum material: {}'\
                .format(preparedMaterial))
            self.platinumMass += preparedMaterial

        elif materialType == 'Nickel':
            logger.info('Extracting Nickel material: {}'\
                .format(preparedMaterial))
            self.nickelMass += preparedMaterial

        elif materialType == 'Propellant':
            logger.info('Extracting Propellant material: {}'\
                .format(preparedMaterial))
            self.propellant += preparedMaterial

        else:
            logger.warning('Detected asteroid with unknow type = '\
                .format(materialType))

        logger.info('--------SUMMARY--------')
        logger.info('Remaining propellant = {} '.format(self.propellant))
        logger.info('Gold mass = {}; Platinum mass = {}; Nickel mass {}'\
            .format(self.goldMass, self.platinumMass, self.nickelMass))
        self.missionTime += targetArrivalTime
        logger.info('Mission time = {}'.format(self.missionTime))
        return True


def main():
    ''' Main function '''

    data = np.loadtxt('data/candidates.txt')
    asteroids = [asteroid.Asteroid(line) for line in data]

    t_arr = [0,
            11.0,
            45.98091676982585,
            98.86574387748259,
            144.3421379448264,
            178.78720680368133,
            198.49061810149578,
            236.39180345018394,
            268.4772894184571]
    a = [0,
            1446,
            5131,
            4449,
            8091,
            1516,
            151,
            4905,
            8490]

    # Init Rover
    rover = Rover(asteroids)

    for idx in range(len(a)-1):
        logger.info('###############################')
        logger.info('Starting Journey # {}'.format(idx))
        logger.info('###############################')
        flag = rover.evaluate_journey(sourceAsteroidId=a[idx],
                                      sourceArrivalTime=t_arr[idx],
                                      targetAsteroidId=a[idx+1],
                                      targetArrivalTime=t_arr[idx+1])
        if not flag:
            logger.info('Finish!')
            break

if __name__ == '__main__':
    main()


