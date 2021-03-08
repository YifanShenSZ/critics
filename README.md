# Critics
A critical geometry searcher

*critics* can search for:
* minimum
* saddle point
* minimum energy crossing

For now saddle point search is only a local search by minimizing the norm of gradient

## Link to user routines
cmake -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_PREFIX_PATH=~/Software/Mine/diabatz/tools/v0/critics/libadiabatz ~/Software/Mine/critics/