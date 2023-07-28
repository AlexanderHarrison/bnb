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
