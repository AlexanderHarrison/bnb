import time
import math
import heapq
import bisect

import numpy as np
import highspy
import scipy.sparse

# TODO
# merge with MCMCLinker
# subspace merging?
# subspace separating

# Integer linear programming solver
# Solved for k best solution
# Minimizes costs
#
# sharpen each node's upper bounds if other problems with 
# an unsolved node's upper bound is calculated extremely naively
#
# Better sorting heuristic
# The current solves waaaay too breadth first (the upper bound is incredibly generous)
# The more sparse A is, the more this is an issue.


# If a solved node's solution is integral, it will be added to the solution list.
# When we have k integral solutions, 
# and there are no unsolved nodes with upper bounds greater than the worst integral solution,
# we are done!

class Node:
    def __init__(self, x, cost, ties, depth):
        self.x = x
        self.ties = ties
        self.cost = cost
        self.depth = depth

    def integral(self):
        return np.allclose(np.trunc(self.x), self.x)
        #return np.all(np.allclose(self.x, 0.0) or

    def __lt__(self, other):
        return self.cost < other.cost

    def reorder(self, idxs):
        self.ties = self.ties[idxs]
        self.x = self.x[idxs]

    # generator yielding child UnsolvedNodes
    def branch(self, A, c, k, lpsolver):
        # Guaranteed to find k best solutions at most k nodes deep
        if self.depth+1 == k:
            return []

        tie_count = np.sum(self.ties != -1)
        # if all vars are tied then there are no children
        if tie_count == len(self.ties):
            return []

        # follow murty-like branching method.
        # i.e.
        # ties = [-1, -1, -1, 0, -1]
        # x = [0, 1, 0.2, 0, 1]
        # child_ties = [
        #     [1, -1, -1, 0, -1],
        #     [0, 0, -1, 0, -1],
        #     [0, 1, 0, 0, -1],
        #     [0, 1, 1, 0, 0]
        # ]
        #
        # nonintegral vars are set to 0 when 'flipped', 1 otherwise 

kashdkjahskdj
aksjjdhkasj
sahdksjhdk
        x_trunc = np.ceil(self.x)
    
        num_vars = x_trunc.size
        untied_mask = self.ties == -1

        # exercise for the reader
        tie_mask_matrix = np.identity(num_vars, dtype=np.uint8)[untied_mask]
        invert_mask_matrix = np.tri(num_vars, k=-1, dtype=np.uint8)[untied_mask] & untied_mask
        child_ties = invert_mask_matrix*(1+x_trunc) + tie_mask_matrix*(2-x_trunc) + self.ties

        # fix edge case where space isn't completely partitioned if last untied element is nonintegral
        last_nontied_idx = np.where(untied_mask)[0][-1]
        last_nontied_element = self.x[last_nontied_idx]
        if last_nontied_element != 0.0 and last_nontied_element != 1.0:
            child_ties[-1, last_nontied_idx] = -1

        child_ties = filter_invalid(A, child_ties)

        for t in child_ties:
            x, cost = lpsolver.solve(t)
            yield Node(x, cost, t, self.depth) 


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

        self.lp.col_lower_ = np.repeat(0, num_vars)
        self.lp.col_upper_ = np.repeat(1, num_vars)

        A_csc = scipy.sparse.csc_matrix(A)
        self.lp.a_matrix_.start_ = A_csc.indptr
        self.lp.a_matrix_.index_ = A_csc.indices
        self.lp.a_matrix_.value_ = A_csc.data

        self.lp_solver.passModel(self.lp)

    # returns (x, cost)
    def solve(self, ties):
        self.lp.col_lower_ = ties == 1
        self.lp.col_upper_ = ties != 0

        self.lp_solver.passModel(self.lp)
        #numvars = len(ties)
        #self.lp_solver.changeColsBounds(
        #    numvars,
        #    np.arange(numvars), 
        #    (ties == 1).astype(np.double), 
        #    (ties != 0).astype(np.double)
        #)
        self.lp_solver.run()
        x = np.abs(np.array(self.lp_solver.getSolution().col_value))

        return x, x.dot(self.c)

    
def main():
    import cProfile
    from data import A, c

    for data in A:
        if np.any(np.all(data == 0, axis=0)):
            print("zero col!")

    #A = (np.random.random((411, 189)) < 0.037).astype(np.uint8)
    #A = (np.random.random((411, 189)) < 0.037).astype(np.uint8)

    #A = np.array([
    #    [0,1,0,1,1],
    #    [0,1,1,0,1],
    #    [1,0,0,1,0],
    #])
    #c = np.array([-5.4,-5,-5,-5,-4])
    #for sol in branch_and_bound(c, A, 10):
    #    print(sol.x, sol.cost)
    #exit()

    #cProfile.runctx('sols = branch_and_bound(c, A, 3)', globals(), locals(), sort=True, filename="data.txt")

    #t = time.time()
    #branch_and_bound(c, A, 5)
    #print("time: " + str(time.time() - t))

    cProfile.runctx('for A1, c1 in zip(A, c): branch_and_bound(c1, A1, 2)', globals(), locals(), sort=True, filename="data.txt")
    #t = time.time()
    #for A1, c1 in zip(A, c): branch_and_bound(c1, A1, 5)
    #print(str(time.time() - t))

    #sols = branch_and_bound(c, A, 10)
    #for sol in sols:
    #    print(sol.x, sol.cost)


def branch_and_bound(c, A, k):
    A = A[np.sum(A, axis=1) > 1] # remove constraints with 1 or less nonzero elements
    A = np.unique(A, axis=0) # make constraints unique

    # slightly slower total
    #subsets = []
    #for i, r1 in enumerate(A):
    #    for j, r2 in enumerate(A):
    #        if not i == j and np.all(r1 <= r2):
    #            subsets.append(i)
    #A = np.delete(A, subsets, axis=0)

    # sort costs from smallest to largest
    # murty constrains the first columns more than the last, so sorting this way tends to constrain better solutions 
    sort_idx = np.argsort(c)
    A = A[:, sort_idx]
    c = c[sort_idx]

    num_constraints, num_vars = A.shape

    if not isinstance(c, np.ndarray) or c.shape != (num_vars,):
        raise ValueError(f"c must be a numpy array of shape ({num_vars},) but found shape {c.shape}")

    if not np.all((A == 0) | (A == 1)):
        raise ValueError("A must only contain 0 or 1")

    lpsolver = LPSolver(c, A)
    root_ties = np.repeat(-1, c.size).astype(np.int8)

    root_x, root_cost = lpsolver.solve(root_ties)
    root_node = Node(root_x, root_cost, root_ties, 0)
    
    best_sols = []
    test_sol_heap = [root_node]
    worst_sol = 0.0

    i = 0

    while len(test_sol_heap) > 0:
        solved_node = heapq.heappop(test_sol_heap)
        #print(solved_node.cost, np.sum(solved_node.x == np.trunc(solved_node.x)))

        best_sols_full = len(best_sols) == k
        if best_sols_full and solved_node.cost > worst_sol:
            break

        #print("\tTesting:\n\t\t" + str(solved_node.x) + str(solved_node.ties), solved_node.cost)
        #print("\tUnchecked: ")
        #for sol in test_sol_heap:
        #    print('\t\t' + str(sol.x) + str(sol.ties), str(sol.cost))

        #print("\tSolved: ")
        #for sol in best_sols:
        #    print('\t\t' + str(sol.x) + str(sol.ties), str(sol.cost))
        #print()

        if solved_node.integral():
            if best_sols_full:
                if solved_node.cost < worst_sol:
                    best_sols.pop()
                    bisect.insort(best_sols, solved_node)
                    worst_sol = best_sols[-1].cost
            else:
                bisect.insort(best_sols, solved_node)
                worst_sol = best_sols[-1].cost

        best_sols_full = len(best_sols) == k

        #if solved_node.cost < best_sols[-1].cost or not best_sols_full:
        for child in solved_node.branch(A, c, k, lpsolver):
            i += 1
            if not best_sols_full or child.cost < worst_sol:
                heapq.heappush(test_sol_heap, child)

    unsort_idx = np.argsort(sort_idx)
    best_sols = [s.reorder(unsort_idx) for s in best_sols] # undo ordering of variables

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
