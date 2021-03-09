# Critics
A critical geometry searcher

*critics* can search for:
* minimum
* saddle point
* minimum energy crossing

For now saddle point search is only a local search by minimizing the norm of gradient

## Link to user routines
cmake -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_PREFIX_PATH=~/Software/Mine/diabatz/tools/v0/critics/libadiabatz ~/Software/Mine/critics/

## Theory behind minimum energy crossing search
We minimize the total energy of the crossed states
```
E = E1 + E2
```
subject to
```
C = (E2 - E1)^2
```
This definition of `E` and `C` avoids the discontinuity arisen from the interchange between `1` and `2`

We carry out this constrained optimization by augmented Lagrangian method

We minimize the augmented Lagrangian by a line search method, e.g. the BFGS quasi-Newton method in our implementation

Once we reach the degenerate seam, we may should remove the projections of the search direction on g and h