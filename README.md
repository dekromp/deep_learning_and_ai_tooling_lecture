# Code Examples from Deep Learnig Tooling Lecture
This repository contains the code examples shown in the *Tooling* Lecture of the cource Deep Learning & AI held at the LMU Munich 06.11.2019.

Slides are available [here](http://www.dbs.ifi.lmu.de/cms/studium_lehre/lehre_master/deep1920/index.html), but require a login to the universitie's system. If you cannot access the slides this way, please don't hesitate to contact me.

## Setup
I recommend using [Anaconda/Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) for installation:

    ```bash
    $ conda env create -f ./environment.yml
    $ conda activate tooling_lecture_2019
    ```

For some code snippet it is important to add this project to the PYTHONPATH. You can use `conda develop` for this (if you used `conda` in the first place):

    ```bash
    $ conda install -y conda-build
    $ conda develop <path>/tooling_lecture
    ```

Also, for running tensorflow serving [Docker](https://www.docker.com/) is required. Install instructions can be found [here](https://docs.docker.com/).
* [Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
* [Mac](https://docs.docker.com/docker-for-mac/install/)
* [Windows](https://docs.docker.com/docker-for-windows/install/)

## Contributions
If you see any bugs, please file an issue and let me know.
