# PyPgen

## Point Process Generation

PyPgen is a collection of functions used to generate randomly distributed points in space or time.
Such collection of points are known in the statistic literature as *stochastic processes*, and specifically *point processes*.
These can be used to model many different real-life events: the position of trees in a forest, the arrival position of rain drops on a street, the arrival-time of calls in a customer call center.
Many different type of point processes exist, starting from the simplest, completely random, spatially-homogeneous Poisson Process (HPP).
Some of the HPP's properties can be relaxed, leading to many kinds of point processes: Non-Homogeneous Poisson Process (NHPP), Mixed Poisson Process (MPP), Cox Process (CP), etc.

## Notes on HPP generation

Points of an HPP can be generated using only *numpy*: they are presented for convenience, and apply only to rectangular space.
With the HPP parameter *\lambda*, the total number of counts in the space *N(A) ~ Poisson(|A|\lambda)*, and can be generated using *np.random.poisson* if need be.
Then, you only need to generate *N(A)* samples uniformly distributed in *A* using *np.random.uniform* for each dimension.
If the data space is not recangular, the points can be generated in any convenient shape and only the points which lie in *A* need be retained.
Other algorithms exist, some implementations of which are given here.

## MISC

Prerequisites: *scipy*, *numpy*.

Keywords: Poisson Process, Point process, Spatial Point patterns

# Glossary

- *A*: Data space
- *\lambda*: Process intensity (function for non-homogeneous)
- PP: Point Process
- HPP: Homogeneous Poisson Process
- NHPP: Non-Homogeneous Poisson Process
- MPP: Mixed Poisson Process
- CP: Cox Process/Doubly Stochastic Point Process
- MaPP: Markov Point Process / Finite Gibbs Point Process
	- As far as I know, those are the same. Refer to [1-3].

# References
[1] Diggle, Peter J. Statistical analysis of spatial and spatio-temporal point patterns. CRC press, 2013.

[2] Illian, Janine, et al. Statistical analysis and modelling of spatial point patterns. Vol. 70. John Wiley & Sons, 2008.

[3] Jensen, Eva B. Vedel, and Linda Stougaard Nielsen. "Inhomogeneous Markov point processes by transformation." Bernoulli 6.5 (2000): 761-782.

[4] Snyder, Donald L., and Michael I. Miller. Random point processes in time and space. Springer Science & Business Media, 2012.

[5] Resnick, Sidney I. Adventures in stochastic processes. Springer Science & Business Media, 1992.

[6] Grandell, Jan. Mixed poisson processes. Vol. 77. CRC Press, 1997.
