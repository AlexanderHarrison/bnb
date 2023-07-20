import time
import math
import heapq
import bisect
import multiprocessing
import queue

import numpy as np
import highspy
import scipy.sparse

class Node:
    def __init__(self, x, cost, ties):
        self.x = x
        self.ties = ties
        self.cost = cost

    def integral(self):
        # seems that np.allclose isn't necessary
        return np.allclose(np.trunc(self.x), self.x)

    def __lt__(self, other):
        return self.cost < other.cost

    def reorder(self, idxs):
        self.ties = self.ties[idxs]
        self.x = self.x[idxs]

    # generator yielding child UnsolvedNodes
    def branch(self, A, c, k):
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

        x_ceil = np.ceil(self.x)
        num_vars = self.x.size

        untied_nonzero_mask = (self.ties == -1) & ~np.isclose(self.x, 0.0)
        print("branch count: " + str(np.sum(untied_nonzero_mask)))

        # exercise for the reader
        tie_mask_matrix = np.identity(num_vars, dtype=np.uint8)[untied_nonzero_mask]
        invert_mask_matrix = np.tri(num_vars, k=-1, dtype=np.uint8)[untied_nonzero_mask] & untied_nonzero_mask
        child_ties = invert_mask_matrix*(1+x_ceil) + tie_mask_matrix*(2-x_ceil) + self.ties

        child_ties = filter_invalid(A, child_ties)

        return child_ties


class LPSolver:
    def __init__(self, c, A_csc):
        self.lp_solver = highspy.Highs()
        self.lp = highspy.HighsLp()
        self.c = c
        self.lp_solver.setOptionValue("log_to_console", False)

        num_constraints, num_vars = A_csc.shape
        inf = highspy.kHighsInf

        self.lp.num_col_ = num_vars;
        self.lp.num_row_ = num_constraints;
        self.lp.col_cost_ = c

        self.lp.row_lower_ = np.repeat(-inf, num_constraints)
        self.lp.row_upper_ = np.repeat(1, num_constraints)

        self.lp.col_lower_ = np.repeat(0, num_vars)
        self.lp.col_upper_ = np.repeat(1, num_vars)

        self.lp.a_matrix_.start_ = A_csc.indptr
        self.lp.a_matrix_.index_ = A_csc.indices
        self.lp.a_matrix_.value_ = A_csc.data

        self.lp_solver.passModel(self.lp)

    # returns (x, cost)
    def solve(self, ties):
        self.lp.col_lower_ = ties == 1
        self.lp.col_upper_ = ties != 0
        self.lp.integrality_ = np.repeat(highspy.HighsVarType.kInteger, ties.size)

        self.lp_solver.passModel(self.lp)
        self.lp_solver.run()
        x = np.abs(np.array(self.lp_solver.getSolution().col_value))

        return x, x.dot(self.c)

    # returns (x, cost)
    def lower_bound(self, ties):
        self.lp.col_lower_ = ties == 1
        self.lp.col_upper_ = ties != 0
        self.lp.integrality_ = np.repeat(highspy.HighsVarType.kContinuous, ties.size)

        self.lp_solver.passModel(self.lp)
        self.lp_solver.run()
        sol = self.lp_solver.getSolution()
        x = np.abs(np.array(sol.col_value))
        col_dual = np.array(sol.col_dual)
        return x, x.dot(self.c), col_dual

    
def main():
    #import cProfile
    from data import A, c
    A = A[0]
    c = c[0]

    #mats = np.load("large_mats.npy.npz")
    #A = mats["A"]
    #c = mats["c"]

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

    t = time.time()
    sols = branch_and_bound(c, A, 20)
    print("time: " + str(time.time() - t))
    for sol in sols:
        print(sol.cost)

    #cProfile.runctx('for A1, c1 in zip(A, c): branch_and_bound(c1, A1, 2)', globals(), locals(), sort=True, filename="data.txt")
    #t = time.time()
    #for A1, c1 in zip(A, c): branch_and_bound(c1, A1, 5)
    #print(str(time.time() - t))

    #sols = branch_and_bound(c, A, 10)
    #for sol in sols:
    #    print(sol.x, sol.cost)

def init_solver(c, A_csc):
    global lp
    lp = LPSolver(c, A_csc)

def solve_lp(t):
    _, lb, _ = lp.lower_bound(t)
    return lb

def solve_milp(tie_q, child_q, c, A_csc):
    lp = LPSolver(c, A_csc)
    while True:
        t = tie_q.get()
        x, cost = lp.solve(t)
        node = Node(x, cost, t) 
        child_q.put(node)

def root_solve(q, t, c, A_csc):
    lp = LPSolver(c, A_csc)
    x, cost = lp.solve(t)
    node = Node(x, cost, t) 
    q.put(node)

def branch_and_bound(c, A, k):
    multiprocessing.set_start_method("spawn")
    A = A[np.sum(A, axis=1) > 1] # remove constraints with 1 or less nonzero elements
    # sort costs from smallest to largest
    # murty constrains the first columns more than the last, so sorting this way tends to constrain better solutions 
    sort_idx = np.argsort(c)
    A = A[:, sort_idx]
    c = c[sort_idx]

    num_constraints, num_vars = A.shape
    A_csr = scipy.sparse.csr_matrix(A)

    def remove_duplicate_rows(data): # np.unique for sparse matrices
        unique_row_indices, unique_columns = [], []
        for row_idx, row in enumerate(data):
            indices = row.indices.tolist()
            if indices not in unique_columns:
                unique_columns.append(indices)
                unique_row_indices.append(row_idx)
        return data[unique_row_indices]
    A_csr = remove_duplicate_rows(A_csr)
    A_csc = scipy.sparse.csc_matrix(A_csr)

    if not isinstance(c, np.ndarray) or c.shape != (num_vars,):
        raise ValueError(f"c must be a numpy array of shape ({num_vars},) but found shape {c.shape}")

    if not np.all((A == 0) | (A == 1)):
        raise ValueError("A must only contain 0 or 1")

    lpsolver = LPSolver(c, A_csc)
    root_ties = np.repeat(-1, c.size).astype(np.int8)
    root_x, root_cost = lpsolver.solve(root_ties)
    root_node = Node(root_x, root_cost, root_ties)

    # for some reason milp deadlocks unless this is solved on another thread...
    # probably some weirdness in HiGHS
    #q = multiprocessing.Queue()
    #t = multiprocessing.Process(target=root_solve, args=(q, root_ties, c, A_csc))
    #t.start()
    #t.join()
    #root_node = q.get()
    #q.close()
    #q.join_thread()

    if root_node.integral():
        best_sols = [root_node]
        worst_sol = root_node.cost
    else:
        best_sols = []
        worst_sol = 1.0
    test_sol_heap = [root_node]
    i = 1

    tie_q = multiprocessing.Queue()
    child_q = multiprocessing.Queue()
    threads = [
        multiprocessing.Process(target=solve_milp, args=(tie_q, child_q, c, A_csc)) 
        for _ in range(multiprocessing.cpu_count())
    ]

    lb_pool = multiprocessing.Pool(initializer=init_solver, initargs=(c, A_csc))

    for t in threads:
        t.start()
    try: 
        while len(test_sol_heap) > 0:
            solved_node = heapq.heappop(test_sol_heap)

            if len(best_sols) == k and solved_node.cost >= worst_sol:
                break

            #print("\tTesting:\n\t\t" + str(solved_node.x) + str(solved_node.ties), solved_node.cost)
            #print("\tUnchecked: ")
            #for sol in test_sol_heap:
            #    print('\t\t' + str(sol.x) + str(sol.ties), str(sol.cost))

            #print("\tSolved: ")
            #for sol in best_sols:
            #    print('\t\t' + str(sol.x) + str(sol.ties), str(sol.cost))
            #print()

            ties = solved_node.branch(A_csr, c, k)

            print("getting lower bounds for {0} children".format(len(ties)))
            #lb, col_duals = zip(*lb_pool.map(solve_lp, ties))
            lb = np.array(lb_pool.map(solve_lp, ties))

            lb_reorder = np.argsort(lb)
            lb = lb[lb_reorder]
            ties = ties[lb_reorder]

            print("solving children")
            lb_iter = zip(ties, lb)

            children_left = 0
            pushed = 0
            # fill tie queue
            for _ in range(multiprocessing.cpu_count()):
                try:
                    t, _ = next(lb_iter)
                    tie_q.put(t)
                    children_left += 1
                    pushed += 1
                    print("{0} / {1}".format(pushed, len(ties)), end="\r")
                except StopIteration:
                    break
            
            finished = False
            while children_left != 0 or not finished:
                child = child_q.get()
                children_left -= 1

                try:
                    while not tie_q.full():
                        t, lb = next(lb_iter)
                        pushed += 1
                        print("{0} / {1}".format(pushed, len(ties)), end="\r")
                        if len(best_sols) != k or lb < worst_sol:
                            tie_q.put(t)
                            children_left += 1
                            break
                except StopIteration:
                    finished = True
                    pass

                i += 1
                if len(best_sols) != k or child.cost < worst_sol:
                    if len(best_sols) == k:
                        best_sols.pop()
                    bisect.insort(best_sols, child)
                    worst_sol = best_sols[-1].cost
                    heapq.heappush(test_sol_heap, child)
    except KeyboardInterrupt:
        for t in threads:
            t.terminate()
        lb_pool.terminate()
        raise KeyboardInterrupt
    for t in threads:
        t.terminate()
        t.join()
    lb_pool.terminate()
    lb_pool.join()


    # undo ordering of variables
    unsort_idx = np.argsort(sort_idx)
    for s in best_sols:
        s.reorder(unsort_idx)

    print(f"Iterations: {i}")
    return best_sols


def filter_invalid(A, potential_ties):
    mask = np.all(A.dot(np.maximum(potential_ties, 0).T) <= 1, axis=0)
    return potential_ties[mask]

if __name__ == "__main__":
    main()
