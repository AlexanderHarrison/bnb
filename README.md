# KILP

Solve for the `k` best integer solutions to the linear optimization problem of the form:

```
minimize dot(c, x)
subject to A @ x <= 1
and 0 <= x <= 1
```

- `c` cost vector as a 1D numpy array.
- `A` constraint matrix as a 2D numpy matrix or a sparse scipy matrix. Width must be the same as c.
- `k` number of solutions to find.

## Example

```python
import numpy as np
import kilp

A = np.array([
    [0,1,0,1,1],
    [0,1,1,0,1],
    [1,0,0,1,0],
])
c = np.array([-5.4,-5,-5,-5,-4])
sols = kilp.solve(c, A, 10)
for s in sols:
    print(s.x, s.cost)
# [1. 0. 1. 0. 0.] -10.4
# [1. 1. 0. 0. 0.] -10.4
# [0. 0. 1. 1. 0.] -10.0
# [1. 0. 0. 0. 1.] -9.4
# [1. 0. 0. 0. 0.] -5.4
# [0. 1. 0. 0. 0.] -5.0
# [0. 0. 1. 0. 0.] -5.0
# [0. 0. 0. 1. 0.] -5.0
# [0. 0. 0. 0. 1.] -4.0
# [0. 0. 0. 0. 0.] 0.0
```
