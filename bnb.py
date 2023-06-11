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
# Fix duplicate solutions
# Why are there duplicates?
#
# sharpen each node's upper bounds if other problems with 
# an unsolved node's upper bound is calculated extremely naively
#
# Better sorting heuristic
# The current solves waaaay too breadth first (the upper bound is incredibly generous)
# The more sparse A is, the more this is an issue.
#
# naive lower bound is cheap to compute. Do something with this?



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
        #self.lower_bound = np.sum(c[ties == 1])
        self.upper_bound = np.sum(c[ties != 0])

    def __lt__(self, other):
        return self.upper_bound < other.upper_bound

    def solve(self, lpsolver):
        x, cost = lpsolver.solve(self.ties)
        return SolvedNode(x, cost, self.ties)


class SolvedNode:
    def __init__(self, x, cost, ties):
        self.x = x
        self.cost = cost
        self.ties = ties

    def __lt__(self, other):
        return self.cost < other.cost

    def integral(self):
        return np.all(np.trunc(self.x) == self.x)

    # generator yielding child UnsolvedNodes
    def branch(self, A, c, sol_):
        # if all vars are tied then there are no children
        if np.all(self.ties != -1):
            return []

        # for each nonintegral var, create two subproblems for tying to 0 and 1
        # then prune by checking if valid.
        nonintegral_vars = np.trunc(self.x) != self.x

        if np.any(nonintegral_vars):
            # e.x.:
            # nonintegral_vars = [True, False, True, True] 
            # tie_mask_matrix = [[1, 0, 0, 0],
            #                    [0, 0, 1, 0],
            #                    [0, 0, 0, 1]]
            tie_mask_matrix = np.identity(self.x.size, dtype=np.uint8)[nonintegral_vars]

            # each row in these matrices is a set of new ties for each UnsolvedNode child.
            # There might be a better way of doing this operation
            tie_0_children = tie_mask_matrix + self.ties    # turn -1 to 0
            tie_1_children = 2*tie_mask_matrix + self.ties  # turn -1 to 1

            # we only have to verify the children that are tied to 1, 
            # as tying to zero won't ever induce a bounds error
            #print(tie_1_children == 1)
            #print(np.all(np.matmul(tie_1_children == 1, A.transpose()) <= 1, axis=1))
            valid_children_mask = np.all(np.matmul(tie_1_children == 1, A.transpose()) <= 1, axis=1)
            tie_1_children = tie_1_children[valid_children_mask]

            for child_ties in tie_1_children:
                yield UnsolvedNode(child_ties, c)

            for child_ties in tie_0_children:
                yield UnsolvedNode(child_ties, c)
        else:
            # if there are no nonintegral variables, 
            # we have to tie the untied but integral values to the opposite value
            tie_mask_matrix = np.identity(self.x.size, dtype=np.uint8)[self.ties == -1]

            # forgive my cleverness...
            # I will not explain this, sorry
            tie_children = (np.abs(tie_mask_matrix-self.x)+1) * tie_mask_matrix + self.ties

            # TODO verify children
            for t in tie_children:
                yield UnsolvedNode(t, c)

            #return 


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
        x = np.array(self.lp_solver.getSolution().col_value)

        assert self.lp_solver.getInfo().valid

        return x, x.dot(self.c)

    
def main():
    A = np.array([
        [1,0,1,0,0],
        [0,1,1,0,1],
        [1,0,1,1,0],
    ])
    c = np.array([-5,-4,-3,-5,-3])

    sols = branch_and_bound(c, A, 3)
    for sol in sols:
        print(sol.x, sol.cost)


def branch_and_bound(c, A, k):
    num_constraints, num_vars = A.shape

    if not isinstance(c, np.ndarray) or c.shape != (num_vars,):
        raise ValueError(f"c must be a numpy array of shape ({num_vars},) but found shape {c.shape}")

    if not np.all((A == 0) | (A == 1)):
        raise ValueError("A must only contain 0 or 1")

    lpsolver = LPSolver(c, A)
    root_ties = np.repeat(-1, c.size).astype(np.int8)
    root_node = UnsolvedNode(root_ties, c)
    
    valid_sols = []
    test_sol_heap = [root_node]

    worst_sol = highspy.kHighsInf

    i = 0
    while len(test_sol_heap) > 0:
        print(f"Iteration: {i}")
        print("\tUnsolved: ")
        for sol in test_sol_heap:
            print('\t\t' + str(sol.ties) + ": " + str(sol.upper_bound))

        print("\tSolved: ")
        for sol in valid_sols:
            print('\t\t' + str(sol.x) + ": " + str(sol.cost))
        i += 1

        test_sol = heapq.heappop(test_sol_heap)
        solved_node = test_sol.solve(lpsolver)

        if solved_node.integral():
            if len(valid_sols) < k:
                worst_sol = max(worst_sol, solved_node.cost)
                bisect.insort(valid_sols, solved_node)
            elif test_sol_heap[0].upper_bound < worst_sol:
                if worst_sol > solved_node.cost:
                    worst_sol = solved_node.cost
                    valid_sols.pop()
                    bisect.insort(valid_sols, solved_node)
            else:
                break

        for child in solved_node.branch(A, c):
            heapq.heappush(test_sol_heap, child)

    return valid_sols

if __name__ == "__main__":
    main()
