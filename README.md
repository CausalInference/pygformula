# pygformula: a python implementation of the parametric g-formula

**Authors: Jing Li, Sophia Rein, Sean McGrath, Roger Logan, Ryan O’Dea, Miguel Hernán**


## Overview
The pygformula package implements the non-iterative conditional expectation (NICE) algorithm of the g-formula algorithm
(Robins, 1986). The g-formula can estimate an outcome’s counterfactual mean or risk under hypothetical treatment strategies
(interventions) when there is sufficient information on time-varying treatments and confounders.


### Features: 

* Treatments: discrete or continuous time-varying treatments.
* Outcomes: failure time outcomes or continuous/binary end of follow-up outcomes.
* Interventions: interventions on a single treatment or joint interventions on multiple treatments.
* Random measurement/visit process.
* Incorporation of a priori knowledge of the data structure.
* Censoring events.
* Competing events.


## Requirements

The package requires python ≥ 3.8 and these necessary dependencies:

- joblib
- lifelines
- matplotlib
- numpy
- pandas
- prettytable
- scipy
- statsmodels
- tqdm

and an optional dependency:

- cmprsk


## Issues:

If you have any issues, please open an [issue](https://github.com/CausalInference/pygformula/issues) on github, we will 
regularly check the questions. For any additional questions or comments, please email jing_li@hsph.harvard.edu.