# Second-Order MPC-Based Distributed Q-Learning

[![Source Code License](https://img.shields.io/badge/license-GPL-blueviolet)](https://github.com/SamuelMallick/dhs-ensemble/blob/main/LICENSE)
![Python 3.11](https://img.shields.io/badge/python-3.13-green.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


This repository contains the source code used to produce the results obtained in [Second-Order MPC-Based Distributed Q-Learning](https://arxiv.org/abs/2511.16424) submitted to [23rd IFAC World Congress](https://ifac2026.org/fairDash.do).

In this work we extend distributed MPC-based Q-learning to use second-order learning updates.

If you find the paper or this repository helpful in your publications, please consider citing it.

```bibtex
@article{mallick2025second,
  title={Second-order MPC-based distributed Q-learning},
  author={Mallick, Samuel and Airaldi, Filippo and Dabiri, Azita and De Schutter, Bart},
  journal={arXiv preprint arXiv:2511.16424},
  year={2025}
}
```

---

## Installation

The code was created with `Python 3.13`. To access it, clone the repository

```bash
git clone https://github.com/SamuelMallick/dmpcrl_so
cd dmpcrl_so
```

and then install the required packages by, e.g., running

```bash
pip install -r requirements.txt
```

### Structure

The repository code is structured in the following way

- **`learn.py`** is the main script for running simulations.
- **`agent.py`** contains the class for the learning agents, implementing the consensus algorithms required for second-order updates.
- **`env.py`** contains the environment which the agent interacts with.
- **`model.py`** contains the functions for the LTI system models.
- **`optimizers.py`** contains the classes for the second-order optimizers.
- **`mpc.py`** contains the class for the MPC controller.
- **`plot.py`** is a script for generating the figures in the paper: Second-Order MPC-Based Distributed Q-Learning.
- **`tikz.py`** is an auxillary file for saving images in tikz format.
- **`results`** contains all data, as pickles, for the results in the paper: Second-Order MPC-Based Distributed Q-Learning.
## License

The repository is provided under the GNU General Public License. See the [LICENSE](https://github.com/SamuelMallick/dmpcrl_so/blob/main/LICENSE) file included with this repository.

---

## Author
[Samuel Mallick](https://www.tudelft.nl/staff/s.h.mallick/), PhD Candidate [s.mallick@tudelft.nl | sam.mallick.97@gmail.com]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/) in [Delft University of Technology](https://www.tudelft.nl/en/)

> This research is part of a project that has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme ([Grant agreement No. 101018826 - CLariNet](https://cordis.europa.eu/project/id/101018826)).

Copyright (c) 2025 Samuel Mallick.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest in the program “dmpcrl_so” (Second-Order MPC-Based Distributed Q-Learning) written by the Author(s). Prof. Dr. Ir. Fred van Keulen, Dean of 3mE.