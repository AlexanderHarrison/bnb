import math
import heapq
import bisect

import numpy as np
import highspy
import scipy.sparse

# Integer linear programming solver
# Solved for k best solution
# Minimizes costs
#
# TODO 
#
# sharpen each node's upper bounds if other problems with 
# an unsolved node's upper bound is calculated extremely naively
#
# Better sorting heuristic
# The current solves waaaay too breadth first (the upper bound is incredibly generous)
# The more sparse A is, the more this is an issue.
#
# naive upper bound is cheap to compute. Do something with this?


# Instances of this class will sit in the bnb queue.
# The Node with the best upper bound will be solved, 
# and then it's children will be added back into the queue
#
# If a solved node's solution is integral, it will be added to the solution list.
# When we have k integral solutions, 
# and there are no unsolved nodes with upper bounds greater than the worst integral solution,
# we are done!
class UnsolvedNode:
    def __init__(self, ties, c):
        self.ties = ties
        self.upper_bound = np.dot(c, ties == 1)
        self.lower_bound = np.dot(c, np.abs(ties))

    def __lt__(self, other):
        return self.lower_bound < other.lower_bound

    def solve(self, lpsolver):
        x, cost = lpsolver.solve(self.ties)
        return SolvedNode(x, cost, self.ties)


class SolvedNode:
    def __init__(self, x, cost, ties):
        self.x = x
        self.cost = cost
        self.ties = ties

    def integral(self):
        return np.all(np.trunc(self.x) == self.x)

    def __lt__(self, other):
        return self.cost < other.cost

    # generator yielding child UnsolvedNodes
    def branch(self, A, c):
        tie_count = np.sum(self.ties != -1)

        # Guaranteed to find k best solutions at most k nodes deep
        # NEW: Can't use number of ties as nodes deep anymore...
        #if tie_count == k:
        #    return []

        # if all vars are tied then there are no children
        if tie_count == len(self.ties):
            return []

        # if any vars are nonintegral, create two subproblems tying first nonintegral to 0 and 1
        nonintegral_vars = np.trunc(self.x) != self.x

        if np.any(nonintegral_vars):
            first_nonintegral_index = np.where(nonintegral_vars)[0][0]

            child_0_ties = np.copy(self.ties)
            child_1_ties = np.copy(self.ties)
            child_0_ties = self.ties[first_nonintegral_index] = 0
            child_1_ties = self.ties[first_nonintegral_index] = 1

            if check_valid(A, child_0_ties):
                yield UnsolvedNode(child_0_ties, c)
            if check_valid(A, child_1_ties):
                yield UnsolvedNode(child_1_ties, c)
        else:
            # follow murty-like branching method.
            # i.e.
            # ties = [-1, -1, -1, 1, -1]
            # x = [0, 1, 1, 0, 1]
            # child_ties = [
            #     [1, -1, -1, 1, -1],
            #     [0, 0, -1, 1, -1],
            #     [0, 1, 0, 1, -1],
            #     [0, 1, 0, 1, 0] (pruned)
            # ]

            num_vars = self.x.size
            untied_mask = self.ties == -1

            # exercize for the reader
            tie_mask_matrix = np.identity(num_vars, dtype=np.uint8)[untied_mask]
            invert_mask_matrix = np.tri(num_vars, k=-1, dtype=np.uint8)[untied_mask] & untied_mask
            child_ties = invert_mask_matrix*(1+self.x) + tie_mask_matrix*(2-self.x) + self.ties

            child_ties = filter_invalid(A, child_ties)

            # TODO verify children
            for t in child_ties:
                yield UnsolvedNode(t, c)


class LPSolver:
    def __init__(self, c, A):
        self.lp_solver = highspy.Highs()
        self.lp = highspy.HighsLp()
        self.c = c
        self.lp_solver.setOptionValue("log_to_console", False)

        num_constraints, num_vars = A.shape
        inf = highspy.kHighsInf

        self.lp.num_col_ = num_vars;
        self.lp.num_row_ = num_constraints;
        self.lp.col_cost_ = c

        self.lp.row_lower_ = np.repeat(-inf, num_constraints)
        self.lp.row_upper_ = np.repeat(1, num_constraints)

        A_csc = scipy.sparse.csc_matrix(A)
        self.lp.a_matrix_.start_ = A_csc.indptr
        self.lp.a_matrix_.index_ = A_csc.indices
        self.lp.a_matrix_.value_ = A_csc.data

    # returns (x, cost)
    def solve(self, ties):
        self.lp.col_lower_ = ties == 1
        self.lp.col_upper_ = ties != 0

        self.lp_solver.passModel(self.lp)
        self.lp_solver.run()
        x = np.abs(np.array(self.lp_solver.getSolution().col_value))

        return x, x.dot(self.c)

    
def main():
    from data import A, c

    #A = np.array([
    #    [1,0,1,0,0],
    #    [0,1,1,0,1],
    #    [1,0,1,1,0],
    #])
    #c = np.array([-5,-4,-3,-5,-3])
    
    import cProfile
    #cProfile.runctx('sols = branch_and_bound(c, A, 3)', globals(), locals(), sort=True, filename="data.txt")
    sols = branch_and_bound(c, A, 4)
    for sol in sols:
        print(sol.x, sol.cost)


class ExitLoop(Exception):
    pass


def branch_and_bound(c, A, k):
    A = A[np.sum(A, axis=1) > 1]

    num_constraints, num_vars = A.shape

    if not isinstance(c, np.ndarray) or c.shape != (num_vars,):
        raise ValueError(f"c must be a numpy array of shape ({num_vars},) but found shape {c.shape}")

    if not np.all((A == 0) | (A == 1)):
        raise ValueError("A must only contain 0 or 1")

    lpsolver = LPSolver(c, A)
    root_ties = np.repeat(-1, c.size).astype(np.int8)
    root_node = UnsolvedNode(root_ties, c)
    
    best_sols = []
    test_sol_heap = [root_node]

    worst_sol = 0.0

    i = 0

    # Python can't break out of nested loops, use an exception hack
    try:
        while len(test_sol_heap) > 0:
            test_sol = heapq.heappop(test_sol_heap)

            if test_sol.lower_bound > worst_sol:
                raise ExitLoop()

            #print("\tTesting:\n\t\t" + str(test_sol.ties))
            #print("\tUnsolved: ")
            #for sol in test_sol_heap:
            #    print('\t\t' + str(sol.ties) + ": " + str(sol.lower_bound))

            #print("\tSolved: ")
            #for sol in best_sols:
            #    print('\t\t' + str(sol.x) + ": " + str(sol.cost))
            i += 1

            #if i == 1000:
            #    raise ExitLoop()

            solved_node = test_sol.solve(lpsolver)

            if solved_node.integral():
                worst_sol = max(worst_sol, solved_node.cost)
                if len(best_sols) == k:
                    if best_sols[-1].cost > solved_node.cost:
                        best_sols.pop()
                        bisect.insort(best_sols, solved_node)
                else:
                    bisect.insort(best_sols, solved_node)

            for child in solved_node.branch(A, c):
                heapq.heappush(test_sol_heap, child)
    except ExitLoop:
        pass

    print(f"Iterations: {i}")
    return best_sols


def check_valid(A, t):
    return np.all(np.dot(A, np.maximum(t, 0)) <= 1) and np.any(t == -1)

def filter_invalid(A, potential_ties):
    mask = np.all(np.matmul(np.maximum(potential_ties, 0), A.T) <= 1, axis=1)
    return potential_ties[mask]

def solved_already(solved, test_sol):
    # A solution is solved already if the tied variables
    # are equal to the variables in the solution
    #
    # i.e. [-1,1,-1,0] would be solved already if [1,1,0,0] was in test_sol
    # I am not sure that this is valid

    mask = test_sol.ties != -1;
    tied_vars = test_sol.ties[mask]

    return np.any(np.all(solved[:, mask] == tied_vars, axis=1))


if __name__ == "__main__":
    main()
