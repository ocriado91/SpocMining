import numpy as np
import pykep as pk

import constants

# Start and end epochs
T_START = pk.epoch_from_iso_string(constants.ISO_T_START)
T_END = pk.epoch_from_iso_string(constants.ISO_T_END)

# Loading the asteroid data
data = np.loadtxt("data/candidates.txt")
asteroids = []
for line in data:
    p = pk.planet.keplerian(
        T_START,
        (
            line[1],
            line[2],
            line[3],
            line[4],
            line[5],
            line[6],
        ),
        constants.MU_TRAPPIST,
        constants.G * line[7],  # mass in planet is not used in UDP, instead separate array below
        1,  # these variable are not relevant for this problem
        1.1,  # these variable are not relevant for this problem
        "Asteroid " + str(int(line[0])),
    )
    asteroids.append(p)

# And asteroids' masses and material type
asteroid_masses = data[:, -2]
asteroid_materials = data[:, -1].astype(int)


def convert_to_chromosome(solution, check_correctness=True):
    """Convert a solution to a fixed-length chromosome to be compatible with the optimize and pygmo framework.

    Args:
        solution (list or np.array): The solution to be converted. Has to have format [t_arrival_0, ..., t_arrival_n, t_mining_0, ..., t_mining_n, ast_id_0, ..., ast_id_n]
        check_correctness (bool): If True, the function will check if the solution fulfills some validity checks (unique asteroids, solution length).

    Returns:
        np.array: The chromosome as required by optimize / pygmo.
    """
    N = len(solution) // 3  # number of asteroids in chromosome
    ast_ids = solution[2 * N :]  # ids of the visited asteroids

    # Check if the solution is valid
    if check_correctness:
        assert (
            len(solution) % 3 == 0
        ), "Solution must be a multiple of 3 containing asteroid id, arrival time and time to mine. for each visit."

        assert (
            len(set(ast_ids)) - len(ast_ids) == 0
        ), "Asteroid IDs must be unique, can only visit each asteroid once."

    # The final chromosome will need to contain all asteroids, so we need to
    # add the asteroids to the chromosome that are not in the solution
    chromosome = np.zeros(30000, dtype=np.float64)

    # Set placeholder values for mining times and arrival times for irrelevant chromosome entries
    chromosome[N:10000] = 0
    chromosome[10000 + N : 20000] = 0

    # Add time of arrivals and mining times
    chromosome[:N] = solution[:N]
    chromosome[10000 : 10000 + N] = solution[N : 2 * N]

    # Add the asteroids that are in the solution
    chromosome[20000 : 20000 + N] = ast_ids

    # Add the asteroids that are not in the solution.
    # There is the potential of a very rare edgecase where by conincidence the next
    # asteroid added this way could still be visited but this is excessively unlikely
    ast_not_in_solution = set(np.arange(10000)).symmetric_difference(set(ast_ids))
    chromosome[20000 + N :] = np.array(list(ast_not_in_solution))

    return chromosome


class belt_mining_udp:
    """
    pygmo User Defined Problem (UDP) describing the optimization problem.
    https://esa.github.io/pygmo2/tutorials/coding_udp_simple.html explains what more details on UDPs.
    """

    def __init__(self, mission_window):
        """Initialize the UDP.

        Args:
            mission_window (list [float, float]): Bounds on the overall mission in days.
        """
        self.asteroids = asteroids
        self.asteroid_masses = asteroid_masses
        self.asteroid_materials_types = asteroid_materials
        self.mission_window = mission_window
        self.n = len(self.asteroids)
        self.MU = constants.MU_TRAPPIST

    def get_bounds(self):
        """Get bounds for the decision variables.

        Returns:
            Tuple of lists: bounds for the decision variables.
        """
        lb = [self.mission_window[0]] * self.n + [0] * self.n + [0] * self.n
        ub = [self.mission_window[1]] * self.n + [60] * self.n + [self.n - 1] * self.n
        return (lb, ub)

    def get_nix(self):
        """Get number of integer variables.

        Returns:
            int: number of integer variables.
        """
        return self.n

    def get_nic(self):
        """Get number of inequality constraints.

        Returns:
            int: number of inequality constraints.
        """
        # Inequality constraints are only set to all visiting epochs (except the first)
        return self.n - 1

    def get_nec(self):
        """Get number of equality constraints.

        Returns:
            int: number of equality constraints.
        """
        # The only equality constraint is that each asteroid must be in the list exactly once
        return 1

    def fitness(self, x, verbose=False):
        """Evaluate the fitness of the decision variables.

        Args:
            x (numpy.array): Chromosome for the decision variables.
            verbose (bool): If True, print some info.

        Returns:
            float: Fitness of the chromosome.
        """
        fuel = 1  # fuel level of the ship, cannot go below 0 or we abort
        visited = 0  # viable number of visited asteroids, will be computed
        n = len(x) // 3  # number of asteroids in chromosome
        time_at_arrival = x[:n]  # time at arrival of each asteroid in days
        time_spent_mining = x[n : 2 * n]  # how many days spent mining each asteroid
        material_collected = [0] * 4  # last is fuel and will be disregarded for score
        ast_ids = x[2 * n :]  # ids of the visited asteroids
        if verbose:
            print(f"ID\tt0\tFuel \tDV \t  Material ID\t Prepared \t \tScore")

        # Lets compute the fitness
        for i in range(1, n):

            # Get indices of currently visited asteroid
            # and the previous one
            current_ast_id = int(ast_ids[i])
            previous_ast_id = int(ast_ids[i - 1])

            ###################### Step 1 #######################
            # Validate the transfer from asteroid i to i+1      #
            #####################################################

            # Break as soon as we exceed mission window
            if time_at_arrival[i] - time_at_arrival[0] > self.mission_window[1]:
                if verbose:
                    print("Mission window exceeded")
                break

            # Also break if the time of flight is too short (avoids singular lambert solutions)
            tof = time_at_arrival[i] - time_at_arrival[i - 1] - time_spent_mining[i - 1]
            if tof < 0.1:
                if verbose:
                    print("Time of flight too short or reached of chain.")
                break

            # Compute the ephemeris of the asteroid we are departing
            r1, v1 = self.asteroids[previous_ast_id].eph(
                T_START.mjd2000 + time_at_arrival[i - 1] + time_spent_mining[i - 1]
            )

            # Compute the ephemeris of the next target asteroid
            r2, v2 = self.asteroids[current_ast_id].eph(
                T_START.mjd2000 + time_at_arrival[i]
            )

            # Solve the lambert problem for this flight
            l = pk.lambert_problem(
                r1=r1, r2=r2, tof=tof * pk.DAY2SEC, mu=self.MU, cw=False, max_revs=0
            )

            # Compute the delta-v necessary to go there and match its velocity
            DV1 = [a - b for a, b in zip(v1, l.get_v1()[0])]
            DV2 = [a - b for a, b in zip(v2, l.get_v2()[0])]
            DV = np.linalg.norm(DV1) + np.linalg.norm(DV2)

            # Compute fuel used for this transfer and update ship fuel level
            fuel = fuel - DV / constants.DV_per_fuel

            # Break if we ran out of fuel during this transfer
            if fuel < 0:
                if verbose:
                    print("Out of fuel")
                break

            ###################### Step 2 #######################
            # If we are here, this asteroid-to-asteroid         #
            # jump is possible and we accumulate the mining     #
            # resource to the objective function.               #
            #####################################################

            # Get material of the asteroid we are visiting
            mat_idx = self.asteroid_materials_types[current_ast_id]

            # Collect as much material as is there or we have time to
            material_collected[mat_idx] += np.minimum(
                self.asteroid_masses[current_ast_id],
                time_spent_mining[i] / constants.TIME_TO_MINE_FULLY,
            )

            # If this is a fuel asteroid, we add it to the fuel
            if mat_idx == 3:
                fuel_found = np.minimum(
                    self.asteroid_masses[current_ast_id],
                    time_spent_mining[i] / constants.TIME_TO_MINE_FULLY,
                )
                fuel = np.minimum(1.0, fuel + fuel_found)

            if verbose:
                tank = f"{material_collected[0]:.2f}|{material_collected[1]:.2f}|{material_collected[2]:.2f}"
                score = np.min(material_collected[:3])
                print(
                    f"{current_ast_id}\t{time_at_arrival[i]:<4.2f}\t{fuel:<7.2f}\t{DV:<8.2f}\t{mat_idx}\t {tank}\t\t{score:.2f}"
                )

            visited = visited + 1

        # The object function in the end is the minimum
        # collected mass of the three non-fuel material types.
        obj = np.min(material_collected[:3])

        # Now the constraints
        # The visited asteroid ids must all be different (equality)
        ec = len(set(ast_ids[:visited])) - len(ast_ids[:visited])
        # The visiting epoch must be after the previous visiting epoch plus the mining time (inequalities)
        ic = [0] * (n - 1)
        for i in range(1, visited):
            ic[i] = (
                time_at_arrival[i - 1] + time_spent_mining[i - 1] - time_at_arrival[i]
            )
        return [-obj] + [ec] + ic

    def pretty(self, x):
        """Pretty print the chromosome.

        Args:
            x (numpy.array): Chromosome for the decision variables.

        Returns:
            str: Pretty print of the chromosome.
        """
        self.fitness(x, True)

    def example(self):
        """Returns an example solution."""
        # (disable automatic formatting for this)
        # fmt: off
        t_arr = [0, 11.0, 45.98091676982585, 98.86574387748259, 144.3421379448264, 178.78720680368133, 198.49061810149578, 236.39180345018394, 268.4772894184571]
        t_m = [0, 18.980916769828053, 22.88482710766111, 29.47639406736512, 17.445068858837555, 18.703411297804774, 19.901185348707877, 24.085485968277332, 17.543366859589646]
        a = [0, 1446, 5131, 4449, 8091, 1516, 151, 4905, 8490]
        # fmt: on
        return convert_to_chromosome(t_arr + t_m + a)


udp = belt_mining_udp([0, constants.TIME_OF_MISSION])
chromosome = udp.example()
udp.pretty(chromosome)