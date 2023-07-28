import time
import heapq
import bisect
import multiprocessing
import queue

import numpy as np
import highspy
import scipy.sparse

def solve(c, A, k):
    """Solve for the `k` best integer solutions to the linear optimization problem of the form:

    minimize dot(c, x)
    subject to A @ x <= 1.0
    and x >= 0
 
    `c` -- const vector as a 1D numpy array.
    `A` -- constraint matrix as a 2D numpy matrix or a sparse scipy matrix. Width must be the same as c.
    `k` -- number of solutions to find.

    Requires process start method of 'spawn'. May deadlock on 'fork'.
    Use `multiprocessing.set_start_method("spawn")` to set.

    Returns a list of `Node`s.
    """
    if c.size < 800:
        return solve_small(c, A, k)
    else:
        return solve_large(c, A, k)


class Node:
    """Represents a valid solution (`x`) to some problem."""

    def __init__(self, x, cost, ties):
        self.x = x
        self.ties = ties
        self.cost = cost

    def solution(self):
        return self.x

    def solution_cost(self):
        return self.cost

    def integral(self):
        return np.allclose(np.trunc(self.x), self.x)


    def __lt__(self, other):
        return self.cost < other.cost

    def reorder(self, idxs):
        self.ties = self.ties[idxs]
        self.x = self.x[idxs]

    # returns a np array of ties
    def branch(self, A):
        x_ceil = np.ceil(self.x)
        num_vars = self.x.size

        zero_mask = np.isclose(self.x, 0.0)
        set_mask = np.isclose(self.x, 1.0)
        untied_mask = self.ties == -1
        to_tie_mask = untied_mask & ~zero_mask

        # exercise for the reader
        tie_mask_matrix = np.identity(num_vars, dtype=np.uint8)[to_tie_mask]
        invert_mask_matrix = np.tri(num_vars, k=-1, dtype=np.uint8)[to_tie_mask] & to_tie_mask
        child_ties = invert_mask_matrix*(1+x_ceil) + tie_mask_matrix*(2-x_ceil) + self.ties

        child_ties = filter_invalid(A, child_ties)
        return child_ties


class LPSolver:
    """Helper interface to HiGHS solver"""

    def __init__(self, c, A_csc):
        self.lp_solver = highspy.Highs()
        self.lp = highspy.HighsLp()
        self.c = c
        self.lp_solver.setOptionValue("log_to_console", False)
        #self.lp_solver.setOptionValue("solver", "simplex")
        #self.lp_solver.setOptionValue("run_crossover", "off")
        #self.lp_solver.setOptionValue("parallel", "off")
        #self.lp_solver.setOptionValue("threads", "1")
        #self.lp_solver.setOptionValue("presolve", "on")

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
    # x values are always integer
    def solve_int(self, ties):
        self.lp.col_lower_ = ties == 1
        self.lp.col_upper_ = ties != 0
        self.lp.integrality_ = np.repeat(highspy.HighsVarType.kInteger, ties.size)

        self.lp_solver.passModel(self.lp)
        self.lp_solver.run()
        x = np.abs(np.array(self.lp_solver.getSolution().col_value))

        return x, x.dot(self.c)

    # returns (x, cost)
    # x values may be between 0 and 1
    # faster than integral version
    def solve(self, ties):
        self.lp.col_lower_ = ties == 1
        self.lp.col_upper_ = ties != 0
        self.lp.integrality_ = np.repeat(highspy.HighsVarType.kContinuous, ties.size)

        self.lp_solver.passModel(self.lp)
        self.lp_solver.run()
        sol = self.lp_solver.getSolution()
        x = np.abs(np.array(sol.col_value))
        return x, x.dot(self.c)


def init_solver(c, A_csc):
    global lp
    lp = LPSolver(c, A_csc)


def solve_lp(t):
    _, lb = lp.solve(t)
    return lb


def solve_milp(tie_q, child_q, c, A_csc):
    lp = LPSolver(c, A_csc)
    while True:
        t = tie_q.get()
        x, cost = lp.solve_int(t)
        node = Node(x, cost, t) 
        child_q.put(node)


# lp and ilp, no thread spawning
def solve_small(c, A, k):
    """Solves without creating multiple processes"""

    (sort_idx, c, A_csr, A_csc) = reduce_constraints(c, A)
    lpsolver = LPSolver(c, A_csc)
    root_ties = np.repeat(-1, c.size).astype(np.int8)
    root_x, root_cost = lpsolver.solve_int(root_ties)
    root_node = Node(root_x, root_cost, root_ties)

    best_sols = [root_node]
    worst_sol = root_node.cost
    test_sol_heap = [root_node]

    while len(test_sol_heap) > 0:
        solved_node = heapq.heappop(test_sol_heap)

        if len(best_sols) == k and solved_node.cost >= worst_sol:
            break

        ties = solved_node.branch(A_csr)
        if len(ties) == 0:
            continue;

        for t in ties:
            if len(best_sols) != k:
                x, cost = lpsolver.solve_int(t)
                child = Node(x, cost, t)
                bisect.insort(best_sols, child)
                worst_sol = best_sols[-1].cost
                heapq.heappush(test_sol_heap, child)
            else:
                x, lower_bound = lpsolver.solve(t)
                if lower_bound >= worst_sol:
                    continue

                lower_bound_child = Node(x, lower_bound, t)
                if lower_bound_child.integral():
                    child = lower_bound_child
                else:
                    x, cost = lpsolver.solve_int(t)
                    child = Node(x, cost, t)

                if child.cost >= worst_sol:
                    continue

                best_sols.pop()
                bisect.insort(best_sols, child)
                worst_sol = best_sols[-1].cost
                heapq.heappush(test_sol_heap, child)

    # undo ordering of variables
    unsort_idx = np.argsort(sort_idx)
    for s in best_sols:
        s.reorder(unsort_idx)

    return best_sols


def init_sync_objects(c, A_csc):
    tie_q = multiprocessing.Queue()
    child_q = multiprocessing.Queue()
    threads = [
        multiprocessing.Process(target=solve_milp, args=(tie_q, child_q, c, A_csc)) 
        for _ in range(multiprocessing.cpu_count())
    ]
    lb_pool = multiprocessing.Pool(initializer=init_solver, initargs=(c, A_csc))

    for t in threads:
        t.start()

    return (tie_q, child_q, threads, lb_pool)


def reduce_constraints(c, A):
    """Simplify and reorder columns on `c` and `A`."""
    # remove constraints with 1 or less nonzero elements
    # weird behavior on sparse matrices
    A = A[np.array((A.sum(axis=1) > 1).flat)]
    # sort costs from smallest to largest
    # murty constrains the first columns more than the last, 
    # so sorting this way tends to find solutions faster
    sort_idx = np.argsort(-c)
    A = A[:, sort_idx]
    c = c[sort_idx]

    num_constraints, num_vars = A.shape
    
    A_csr = scipy.sparse.csr_matrix(A)

    # np.unique for sparse matrices
    unique_row_indices, unique_columns = [], []
    for row_idx, row in enumerate(A_csr):
        indices = row.indices.tolist()
        if indices not in unique_columns:
            unique_columns.append(indices)
            unique_row_indices.append(row_idx)
    A_csr = A_csr[unique_row_indices]
    A_csc = scipy.sparse.csc_matrix(A_csr)

    if not isinstance(c, np.ndarray) or c.shape != (num_vars,):
        raise ValueError(f"c must be a numpy array of shape ({num_vars},) but found shape {c.shape}")

    return (sort_idx, c, A_csr, A_csc)


def solve_large(c, A, k):
    """Requires thread start method of 'spawn'. May deadlock on 'fork'.
    Use `multiprocessing.set_start_method("spawn")` to set.

    Uses multiple processes to speed up computation.
    """

    assert multiprocessing.get_start_method() == "spawn"
    
    (sort_idx, c, A_csr, A_csc) = reduce_constraints(c, A)
    lpsolver = LPSolver(c, A_csc)
    root_ties = np.repeat(-1, c.size).astype(np.int8)
    root_x, root_cost = lpsolver.solve_int(root_ties)
    root_node = Node(root_x, root_cost, root_ties)

    best_sols = [root_node]
    worst_sol = root_node.cost
    test_sol_heap = [root_node]

    (tie_q, child_q, threads, lb_pool) = init_sync_objects(c, A_csc)

    try: 
        while len(test_sol_heap) > 0:
            solved_node = heapq.heappop(test_sol_heap)

            if len(best_sols) == k and solved_node.cost >= worst_sol:
                break

            ties = solved_node.branch(A_csr)
            if len(ties) == 0:
                continue;

            lb = np.array(lb_pool.map(solve_lp, ties))

            lb_reorder = np.argsort(lb)
            lb = lb[lb_reorder]
            ties = ties[lb_reorder]

            lb_iter = zip(ties, lb)

            # the number of children currently being processed
            # only zero initially and when there are no more children to process
            children_left = 0

            # fill queue
            try:
                for _ in range(multiprocessing.cpu_count()):
                    t, lb = next(lb_iter)
                    while len(best_sols) == k and lb >= worst_sol:
                        t, lb = next(lb_iter)
                    tie_q.put(t)
                    children_left += 1
            except StopIteration:
                pass
                
            # push a new child for each solved child popped
            while children_left != 0:
                child = child_q.get()
                children_left -= 1

                try:
                    t, lb = next(lb_iter)
                    while len(best_sols) == k and lb >= worst_sol:
                        t, lb = next(lb_iter)
                    tie_q.put(t)
                    children_left += 1
                except StopIteration:
                    pass

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

    return best_sols


def filter_invalid(A, potential_ties):
    mask = np.all(A.dot(np.maximum(potential_ties, 0).T) <= 1, axis=0)
    return potential_ties[mask]


def main():
    multiprocessing.set_start_method("spawn")
    import cProfile
    #from data import A, c
    #A = A[0]
    #c = c[0]

    #x = np.array([1,1,0,0,1,0.5, 0.5, 1, 0, 0.5])
    #cost = 0
    #ties = np.array([-1, 1, -1, -1, 1, -1, -1, 1, 0, -1])
    #n = Node(x, cost, ties)
    #A  = np.repeat(0, len(x))[None]
    #print(x)
    #print(n.branch(A, nonints_first = True))
    #return

    #for A1, c1 in zip(A, c):
    #    sols1 = branch_and_bound_lp(c1, A1, 3, nonint_first=False)
    #    sols2 = branch_and_bound_lp(c1, A1, 3, nonint_first=True)
    #    for s1, s2 in zip(sols1, sols2):
    #        print(s1.cost == s2.cost)

    ##x = np.array([1,1,0,0,1,0.5, 0.5, 1, 0, 0.5])
    #x = np.array([1,0,0,0.5, 0.5])
    ##x = np.array([1,0.5])
    #cost = 0
    #ties = np.repeat(-1, len(x))
    #n = Node(x, cost, ties)
    #A = np.repeat(0, len(x))[None]
    #print(n.branch(A))
    #return

    mats = np.load("large_mats.npy.npz")
    A = mats["A"]
    c = mats["c"]

    As = scipy.sparse.csr_matrix(A)

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

    #cProfile.runctx('sols = solve(c, A, 3)', globals(), locals(), sort=True, filename="data.txt")

    t = time.time()
    sols = solve(c, A, 100)
    for s in sols:
        print(s.solution_cost())
    print(str(time.time() - t))


if __name__ == "__main__":
    main()
