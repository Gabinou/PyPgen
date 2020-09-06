# PyPgen
## Point Process Generation

## Notes on HPP generation
The functions presented here are convenient for cases where the spase is rectangular.


Convenient functions for HPP generation are given, but points can be generated using only *numpy*. With the HPP parameter *\lambda*, the total number of counts in the space *N(A) ~ P(\lambda)*, and can be generated using *np.random.poisson* if need be. Then, you only need to generate *N(A)* samples uniformly distributed in the space using *np.random.uniform* for each dimension. If the data space is not recangular, the points can be generate in any convenient shape, then only the points which lie in *A* are retained.

## MISC

Prerequisites: *scipy*, *numpy*.

Keywords: Poisson Process, Point process, Spatial Point patterns


## Glossary

- *A*: Data space
- *&lambda*: Process intensity (function for non-homogeneous)
- PP: Point Process
- HPP: Homogeneous Poisson Process
- NHPP: Non-Homogeneous Poisson Process
- MPP: Mixed Poisson Process
- MaPP: Markov Point Process / Finite Gibbs Point Process
	- As far as I know, those are the same. Refer to [1-3].

# References
[1] Diggle, Peter J. Statistical analysis of spatial and spatio-temporal point patterns. CRC press, 2013.

[2] Illian, Janine, et al. Statistical analysis and modelling of spatial point patterns. Vol. 70. John Wiley & Sons, 2008.
    # n: current total of points

[3] Jensen, Eva B. Vedel, and Linda Stougaard Nielsen. "Inhomogeneous Markov point processes by transformation." Bernoulli 6.5 (2000): 761-782.