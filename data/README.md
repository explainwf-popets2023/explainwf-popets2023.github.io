# Datasets

In this file we describe the datasets that are made available in this artifact.

## Fidelity

For the fidelity experiments in Section 3 of the paper, we have data from live
Tor crawls and data from the Shadow experiments. In both cases, the
[urls.txt](urls.txt) file contains a list of the URLs from the wikipedia mirror
that were fetched during the experiment.

### Live Tor

The web pages were crawled using TBB and wget2. The traces resulting from these
crawls is located in the [fidelity/livetor](https://github.com/explainwf-popets2023/explainwf-popets2023.github.io/tree/main/data/fidelity/livetor) subdirectory.

### Shadow

We ran Shadow simulations fetching the URLs using wget in Shadow. We repeated
the simulation six times in order to gather enough traces for analysis. The
traces resulting from these crawls is located in the
[fidelity/tornet-net_0.25-load_2.0](https://github.com/explainwf-popets2023/explainwf-popets2023.github.io/tree/main/data/fidelity/tornet-net_0.25-load_2.0)
subdirectory. The data from these experiments can be combined into a single file
for analysis purposes using the
[fidelity/tornet-net_0.25-load_2.0/concat_all.sh](fidelity/tornet-net_0.25-load_2.0/concat_all.sh)
script.

## Sensitivity and Robustness

For the sensitivity and robustness experiments in Sections 4 and 5 in the paper,
we have a total of nine unique Shadow networks. We ran two simulations in each
network to obtain additional traces for analysis. The traces resulting from
these simulations are located in subdirectories of the
[sensitivity_robustness](https://github.com/explainwf-popets2023/explainwf-popets2023.github.io/tree/main/data/sensitivity_robustness) subdirectory. The data from
these experiments can be organized for analysis purposes using the
[sensitivity_robustness/concat_traces.sh](sensitivity_robustness/concat_traces.sh)
script.
