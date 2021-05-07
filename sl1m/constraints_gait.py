import numpy as np


class Constraints:
    """
    Implementation of the class constraints, which implements the constraints used by the generic planner

    The problem is a LP with these variables for each phase:
    [ com_x, com_y, com_z_1, com_z_2, {p_i_x, p_i_y, p_i_z}, {a_ij}                                                  ]
    [ 1    , 1    , 1      , 1      , 3 * phase_n_moving   , (0 if n_surfaces == 1, else n_surfaces) * phase_n_moving]

    Under the following constraints :
    - fixed_foot_com: Ensures the COM is 'above' the fixed feet
    - foot_relative_distance: Ensures the moving feet are close enough to the other feet
    - surface: Each foot belongs to one surface
    - slack_positivity: The slack variables are positive
    - com_weighted_equality: Fix the horizontal position of the COm at the barycenter of the contact points (fixed feet)
    """

    def __init__(self, n_effectors):
        self.n_effectors = n_effectors

        self.WEIGHTS = [1./float(n_effectors - 2)] * n_effectors

    def _default_n_variables(self, phase):
        """
        @param phase phase concerned
        @return the number of non slack variables in phase
        """
        return 3 * len(phase.moving)

    def _expression_matrix(self, size, _default_n_variables, j):
        """
        Generate a selection matrix for a given variable 
        @param size number of rows of the variable
        @param _default_n_variables number of non slack variables in the phase
        @param x position of the variable in the phase variables
        @return a (size, number of default variables (without slacks)) matrix with identity at column j
        """
        M = np.zeros((size, _default_n_variables))
        M[:, j:j + size] = np.identity(size)
        return M

    def foot(self, phase, foot=None, id=None):
        """
        Generate a selection matrix for a given foot 
        @param phase phase data
        @param foot foot to select
        @param id id of the foot in the moving feet list
        @return a (3, number of variables (without slacks)) selection matrix
        """
        if foot is not None:
            id = np.argmax(phase.moving == foot)
        elif id is None:
            print("Error in foot selection matrix: you must specify either foot or id")
        return self._expression_matrix(3, self._default_n_variables(phase), 3 * id)

    def foot_xy(self, phase, foot=None, id=None):
        """
        Generate a selection matrix for a given foot x and y coordinates
        @param phase phase data
        @param foot foot to select
        @param id id of the foot in the moving feet list
        @return a (3, number of variables (without slacks)) selection matrix
        """
        if foot is not None:
            id = np.argmax(phase.moving == foot)
        elif id is None:
            print("Error in foot selection matrix: you must specify either foot or id")
        return self._expression_matrix(2, self._default_n_variables(phase), 3 * id)

    def foot_relative_distance(self, pb, phase, G, h, i_start, js, feet_phase):
        """
        The distance between the moving effector and the other stance feet is limited
        For i in moving_foot, For j !=i, Ki (pj - pi) <= ki
        @param pb          The problem specific data
        @param phase       The phase specific data
        @param G           The inequality constraint matrix
        @param h           The inequality constraint vector
        @param i_start     Initial row to use
        @param js          List of column corresponding to the start of each phase
        @param feet_phase List of the feet las moving phase, -1 if they haven't moved
        @return i_start + the number of rows used by the constraint
        """
        i = i_start
        j = js[-1]
        constraints = phase.allRelativeK[phase.moving[0]]
        for (foot, (K, k)) in constraints:
            if foot in phase.stance:
                l = k.shape[0]
                G[i:i + l, j:j + self._default_n_variables(phase)] = -K.dot(self.foot(phase, id=0))
                h[i:i + l] = k
                if foot in phase.moving:
                    G[i:i + l, j:j + self._default_n_variables(phase)] += K.dot(self.foot(phase, foot))
                elif feet_phase[foot] != -1:
                    j_f = js[feet_phase[foot]]
                    phase_f = pb.phaseData[feet_phase[foot]]
                    G[i:i + l, j_f:j_f + self._default_n_variables(phase_f)] = K.dot(self.foot(phase_f, foot))
                else:
                    foot_pose = pb.p0[foot]
                    h[i:i + l] -= K.dot(foot_pose)
                print("Foot relative constraints")
                print("Feet : ", foot)
                print("Initial foot", phase.moving[0])
                print(G[i:i + l, j:j + self._default_n_variables(phase)])
                i += l
        return i

    def slack_positivity(self, phase, G, h, i_start, j):
        """
        The slack variables (alpha) should be positive
        Sl for each surface s, -alpha_s <= 0
        @param phase       The phase specific data
        @param G           The inequality constraint matrix
        @param h           The inequality constraint vector
        @param i_start     Initial row to use
        @param j           Column corresponding to this phase variables
        @return i_start + the number of rows used by the constraint
        """
        i = i_start
        j_alpha = j + self._default_n_variables(phase)
        for n_surface in phase.n_surfaces:
            if n_surface > 1:
                G[i:i + n_surface, j_alpha:j_alpha + n_surface] = -np.identity(n_surface)
                j_alpha += n_surface
                i += n_surface
        print("Slack positivity constraint")
        print(G[i_start:i, j:j + self._default_n_variables(phase) + 4])
        return i

    def surface_inequality(self, phase, G, h, i_start, j):
        """
        Each moving foot must belong to one surface
        For each surface l: Sl pi - alphal <= sl
        @param phase       The phase specific data
        @param G           The inequality constraint matrix
        @param h           The inequality constraint vector
        @param i_start     Initial row to use
        @param j           Column corresponding to this phase variables
        @return i_start + the number of rows used by the constraint
        """
        i = i_start
        j_alpha = self._default_n_variables(phase)
        for id, surfaces in enumerate(phase.S):
            for S, s in surfaces:
                l = S.shape[0]
                G[i:i + l, j:j + self._default_n_variables(phase)] = S.dot(self.foot(phase, id=id))
                h[i:i + l] = s
                if phase.n_surfaces[id] > 1:
                    G[i:i + l, j + j_alpha] = -1000. * np.ones(l)
                    j_alpha += 1
                print("Surface inequality constraints")
                print("Id = ", id)
                print("S = ", S)
                print("s = ", s)
                print(G[i:i + l, j:j + self._default_n_variables(phase)+4])
                i += l
        print(i-i_start)
        return i

    def slack_equality(self, phase, C, d, i_start, j):
        """
        The slack variables (alpha) sum should be equal to the number of surfaces -1 
        Sl for each moving foot, sum(alpha_s) = n_surfaces - 1
        @param phase       The phase specific data
        @param C           The equality constraint matrix
        @param d           The equality constraint vector
        @param i_start     Initial row to use
        @param j           Column corresponding to this phase variables
        @return i_start + the number of rows used by the constraint
        """
        i = i_start
        j_alpha = j + self._default_n_variables(phase)
        for n_surface in phase.n_surfaces:
            if n_surface > 1:
                C[i, j_alpha:j_alpha + n_surface] = np.ones(n_surface)
                d[i] = n_surface - 1
                j_alpha += n_surface
                i += 1
        print("slack_equality constraints")
        print("C = ", C)
        print("d = ", d)
        return i