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

def test_material_rates():
     '''
     Testing material rates in the following cases:
     - Case 1: Rover tank empty.
     - Case 2: Rover tank with three types of material.
     - Case 3: Rover tank with lack of one type of material.
     - Case 4: Rover tank with lack of two types of materials.
     '''

     datafile = "data/candidates.txt"
     rover = Rover(datafile)

     # Case 1: Rover tank with three types of material
     expected_material_rates = [1/3] * 3
     material_rates = list(rover.material_rates())
     assert material_rates == expected_material_rates

     # Case 2: Rover tank with three types of material
     rover.tank = [0.5, 1.5, 0.25]
     expected_material_rates = [0.3, 0.1, 0.6]
     material_rates = list(rover.material_rates())
     assert material_rates == expected_material_rates

     # Case 3: Rover tank with lack of one type of material.
     rover.tank = [0.5, 0, 1]
     expected_material_rates = [0.028708133971291867,
                                0.9569377990430622,
                                0.014354066985645933]
     material_rates = list(rover.material_rates())
     assert material_rates == expected_material_rates

     # Case 4: Rover tank with lack of two types of material.
     rover.tank = [0, 0, 200]
     expected_material_rates = [0.4975124378109453,
                                0.4975124378109453,
                                0.004975124378109453]
     material_rates = list(rover.material_rates())
     print(material_rates)
     assert material_rates == expected_material_rates





