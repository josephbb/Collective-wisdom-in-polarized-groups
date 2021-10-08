# [Collective Wisdom in Polarized Groups](https://github.com/josephbb/polarized-collective-wisdom)
Joseph B. Bak-Coleman(1,2), Christopher K. Tokita(3), Dylan H. Morris(4),
Daniel I. Rubenstein(3), Iain D. Couzin(5,6,7)

1. Center for an Informed Public, University of Washington, Seattle, WA  98195,
2. eScience Institute, University of Washington, Seattle, WA  98195,
3. Ecology and Evolutionary Biology, Princeton University, Princeton NJ, 08540,
4. Institute of the Environment and Sustainability, University of California Los Angeles, Los Angeles, CA 90095,
5. Department of Collective Behaviour, Max Planck Institute for Animal Behavior, D-78547 Konstanz, Germany,
6. Center for the Advanced Study of Collective Behaviour, University of Konstanz, D-78547 Konstanz, Germany,
7. Department of Biology, University of Konstanz, D-78547, Konstanz, Germany}


## Repository information
This repository provides all code and data necessary to generates results, tables, and figures found in the article ["Collective Wisdom in Polarized Groups"](https://github.com/josephbb/) (J. Bak-Coleman et al. 2021).

## Article abstract
The potential for groups to outperform the cognitive capabilities of even highly skilled individuals, known as the "wisdom of the crowd", is crucial to the functioning of democratic institutions. In recent years, increasing polarization has led to concern about its effects on the accuracy of electorates, juries, courts, and congress. While there is empirical evidence of collective wisdom in partisan crowds, a general theory has remained elusive.  Central to the challenge is the difficulty of disentangling the effect of limited interaction between opposing groups (homophily) from their tendency to hold opposing viewpoints (partisanship). To overcome this challenge, we develop an agent-based model of collective wisdom parameterized by the experimentally-measured behaviour of participants across the political spectrum. In doing so, we reveal that differences across the political spectrum in how individuals express and respond to knowledge interact with the structure of the network to either promote or undermine wisdom. We verify these findings experimentally and construct a more general theoretical framework. Finally, we provide evidence that incidental, context-specific differences across the political spectrum likely determine the impact of polarization. Overall, our results show that whether polarized groups benefit from collective wisdom is generally predictable but highly context-specific.

## License and citation information
If you plan on using this code for any purpose, please see the [license](https://github.com/josephbb/Collective-wisdom-in-polarized-groups/blob/main/LICENSE.txt) and please cite our work as below:

Citation and BiBTeX record to come.
## Directories
- ``polarization-analysis.ipynb`` : Primary analysis file as an ipython notebook.
- ``analysis.py``: Python script (``.py``) export of polarization-analysis.ipynb.
- ``src``: The Bayesian Models (``*.stan``), code used to clean the raw data, code for generating figures, and utilities used in the primary analysis.  
    - ``*figures.py`` functions to generate figures
    - ``*.stan`` Stan model code
    - ``demographics.py`` Demographic plots.
    - ``clean.py`` Functions for converting raw data to cleaned data
    - ``simulation_study.py`` Simulation code
    - ``stan_data_format.py`` Code for formatting pandas data frames into a Stan-friendly format
    - ``utils.py`` Miscellaneous utilities.
- ``dat``: data files in comma-separated values (``.csv``) formats
    - ``dat/raw``: raw data files
    - ``dat/cleaned``: data files processed and prepared for analyis
- ``out``: output files
    - ``out/chains``: Markov Chain Monte Carlo (MCMC) output, as pickled python objects (``.p``)
    - ``out/figures``: Figures generated from results
    - ``out/models``: Diagnostic tests for MCMC convergence.
    - ``out/tables``: Diagnostic tests for MCMC convergence.

## Reproducing analysis

You can reproduce the analysis, including all figures and tables by following the guide below. Please note that minor, non-qualitative differences may exist due to difference in pseudorandom number generation.

### Getting the code
First download this repository. Either download directly or open a command line and type:

    git clone https://github.com/josephbb/polarized-collective-wisdom

## Dependency installation guide
You will an [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) or python installation and command-line interface. The simplest way to install the requirements is to navigate to the directory and type ``pip install -r requirements.txt``. You may, however, wish to install these in a [virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to avoid conflicts with your currently installed python packages. Note that installing these packages, particularly Stan and Pystan can take time and require compilation on your local machine.

### Running the analysis

The simplest approach is to navigate to the directory and simply type:

    jupyter nbconvert --execute ./polarization-analysis.ipynb --ExecutePreprocessor.timeout -1
This will generate a rendered output of the notebook(``.HTML``) that you can open in your browswer, along with all figures and tables on your local machine. Please note that this code can take a long time (perhaps hours) to run, necessitating  timeout being set to -1 in the command above.  . You may prefer simply to open and review the notebook using

    jupyter notebook


## Project structure when complete

Once the full analysis has been run, figures can be found in ``out/figures``, tables in ``out/tables``, compiled stan models in ``out/models`` and MCMC chains in ``out/chains``.

#System Specifications

Beyond what is in requirements.txt, this analysis was run on a machine with the following configuration.

- CPU: AMD Ryzen 9 3900X 12-Core Processor
- Memory: 64 GiB
- GPU (not used): NVIDIA 1080 Ti
- OS: Ubuntu 20.04.1 LtS
- Python: 3.8.5
- Anaconda: 4.10.3
- Pystan 2.19.1.1 (Pystan 3 will not work)
- clang 10.0.0.0-4ubuntu1
