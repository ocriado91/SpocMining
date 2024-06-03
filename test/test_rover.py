#!/usr/bin/env python3

from rover import Rover

def test_compute_journey():
    '''
    Check rover journey computation using challenge data file
    '''

    datafile = "data/candidates.txt"
    rover = Rover(datafile)

    # Values extracted from challenge evaluation code
    t_arr = [0,
             11.0,
             45.98091676982585,
             98.86574387748259,
             144.3421379448264,
             178.78720680368133,
             198.49061810149578,
             236.39180345018394,
             268.4772894184571]
    t_m = [0,
           18.980916769828053,
           22.88482710766111,
           29.47639406736512,
           17.445068858837555,
           18.703411297804774,
           19.901185348707877,
           24.085485968277332,
           17.543366859589646]
    a = [0,
         1446,
         5131,
         4449,
         8091,
         1516,
         151,
         4905,
         8490]

    # Expected value from challenge evaluation code
    expected_score = 0.5847788953196549
    rover.compute_journey(a, t_arr, t_m)
    assert expected_score == rover.score

def test_compute_knn():
    '''
    Test compute of K-Nearest Neighbors
    '''

    datafile = "data/candidates.txt"
    rover = Rover(datafile)

    # Add first asteroid to list of visited asteroids
    rover.visited_asteroids.append(0)

    expected_ids = [7183, 7181, 6576, 6340]
    ids = rover.compute_knn(time=0,
                            target_asteroid_id=0,
                            k=5)
    assert expected_ids == ids

def test_material_rate_empty_tank():
    """Check material rates when rover tank is empty."""

    datafile = "data/candidates.txt"
    rover = Rover(datafile)

    expected_material_rate = 1/3
    material_rate = rover.material_rate(7183)
    assert expected_material_rate == material_rate

def test_material_rate_non_empty_tank():
    """Check material rates when rover tank is not empty."""

    datafile = "data/candidates.txt"
    rover = Rover(datafile)

    rover.tank = [0.5, 1.5, 0.25]
    expected_material_rate = 0.6
    material_rate = rover.material_rate(7183)
    assert material_rate == expected_material_rate

def test_material_rate_lack_one_material():
    """Check material rates when there is lack of one material."""

    datafile = "data/candidates.txt"
    rover = Rover(datafile)

    rover.tank = [0.5, 0, 1]
    expected_material_rate = 0.014354066985645933
    material_rate = rover.material_rate(7183)
    assert material_rate == expected_material_rate

def test_material_rate_lack_two_materials():
    """Check material rates when there is lack of two different materials."""

    datafile = "data/candidates.txt"
    rover = Rover(datafile)

    rover.tank = [0, 0, 200]
    expected_material_rate = 0.004975124378109453
    material_rate = rover.material_rate(7183)
    assert material_rate == expected_material_rate
