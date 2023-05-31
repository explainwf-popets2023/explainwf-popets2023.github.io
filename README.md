### Overview

This page describes research artifacts for the following research publication:

**Data-Explainable Website Fingerprinting with Network Simulation**  
_[Proceedings on Privacy Enhancing Technologies (PoPETs), 2023](https://petsymposium.org/2023/index.php)_  
by [Rob Jansen](https://www.robgjansen.com) and [Ryan Wails](https://ryanwails.com/)  
<!--\[[Conference version](https://www.robgjansen.com/publications/explainwf-popets2023.pdf)\]-->

If you reference this paper or use any of the data or code provided on this
site, please cite the paper. Here is a bibtex entry for latex users:

```
@article{explainwf-popets2023,
  author = {Rob Jansen and Ryan Wails},
  title = {Data-Explainable Website Fingerprinting with Network Simulation},
  journal = {Proceedings on Privacy Enhancing Technologies},
  year = {2023},
  volume = {2023},
  number = {4}
  note = {See also \url{https://explainwf-popets2023.github.io}},
}
```

### Contents

Our artifact contains two primary contributions: (1) Datasets of cell traces
gathered from crawling the live Tor network and produced during the execution of
Shadow simulations, and (2) Analysis scripts and code used to run our ML
experiments. We also provide a Dockerfile and the Shadow configuration files
that we used to run our Shadow experiments. However, due to the resource demands
of running the Shadow simulations, we do not expect these to be used directly by
future researchers and therefore we do not provide precise documentation for
them. Finally, we provide some details about our patched version of wget2.

We have broken our artifact into multiple parts, each of which is described on
its own subpage of this site:

  1. [The data page](/data) describes the datasets we make available
  2. [The ML page](/ml) describes our machine learning analyses
  3. [The wget2 page](/wget2) describes our patched version of wget2
  4. [The Shadow page](/shadow) describes our Shadow config files
