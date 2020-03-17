# Plan-Space State Embedding

**_This is research code!_**
*Researchers should use this code to understand the details of our algorithms. It is provided for expository pruposes without any warranty, expressed or implied.*

This code implements the algorithms for plan-space state embedding described in a forthcoming paper. The `psse` directory contains code relevant to training the embedding space.  The `rl-benchmark` directory contains code we used for evaluating the impact of our algorithm on some reinforcement learning algorithms.  Please contact us if you would like clarification on anything you see here, we could provide a copy of our currently in-submission paper, or add to the documentation on this page.  

If you would like to use this code in another project please contact us so we can provide guidance on the suitability of this code base, and also establish an appropriate license, as this is currently all rights reserved.

Contact: Max Pflueger `pflueger` at `usc.edu`

## Dependencies
- Tensorflow 1.15
- yaml
- garage

## Setup with Anaconda
- Create an anaconda env with python 3.6 (or later)
- `pip install garage` (this will pick up tensorflow and most other dependencies we need)
- `git clone [this project]`
- `cd plan-space-state-embedding/psse`
- `pip install -e .`
