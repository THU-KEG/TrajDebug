# Data

`unified/` contains the trajectory representation consumed by TRAJDEBUG.
TRAJERRBENCH consists of 400 failed trajectories derived from τ²-Bench and 86
failed trajectories derived from SWE-Bench Pro. The remaining folders reproduce
the paper's evaluations on WhoAndWhen and AgentDebugBench.

The repository's MIT license applies to TRAJDEBUG code and original annotations.
It does not relicense task text, trajectories, environments, repositories, or
other material originating from the upstream benchmarks. Those components remain
subject to their respective licenses and terms:

- τ²-Bench
- SWE-Bench Pro and the repositories represented by its tasks
- WhoAndWhen
- AgentDebugBench (ALFWorld, GAIA, and WebShop subsets)

Before redistributing or using the data beyond research reproduction, review the
current upstream terms. The converters in `data_processing/` can rebuild the
unified representation from separately obtained source datasets.
