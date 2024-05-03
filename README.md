# pygformula: a python implementation of the parametric g-formula

[![PyPI version](https://badge.fury.io/py/pygformula.svg)](https://pypi.org/project/pygformula)
[![Documentation Status](https://readthedocs.org/projects/pygformula/badge/?version=latest)](https://pygformula.readthedocs.io)
[![Downloads](https://static.pepy.tech/badge/pygformula)](https://pepy.tech/project/pygformula)

**Authors: Jing Li, Sophia Rein, Sean McGrath, Roger Logan, Ryan O’Dea, Miguel Hernán**


## Overview
The pygformula package implements the non-iterative conditional expectation (NICE) algorithm of the g-formula algorithm
(Robins, 1986). The g-formula can estimate an outcome’s counterfactual mean or risk under hypothetical treatment strategies
(interventions) when there is sufficient information on time-varying treatments and confounders.


### Features

* Treatments: discrete or continuous time-varying treatments.
* Outcomes: failure time outcomes or continuous/binary end of follow-up outcomes.
* Interventions: interventions on a single treatment or joint interventions on multiple treatments.
* Random measurement/visit process.
* Incorporation of a priori knowledge of the data structure.
* Censoring events.
* Competing events.


## Requirements

The package requires python 3.8+ and these necessary dependencies:

- cmprsk
- joblib
- lifelines
- matplotlib
- numpy
- pandas
- prettytable
- pytruncreg
- scipy
- seaborn
- statsmodels
- tqdm


## Documentation

The online documentation is available at [pygformula documentation](https://pygformula.readthedocs.io).

## Issues

If you have any issues, please open an [issue](https://github.com/CausalInference/pygformula/issues) on github, we will 
regularly check the questions. For any additional questions or comments, please email jing_li@hsph.harvard.edu.