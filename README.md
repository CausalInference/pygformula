# pygformula: a python implementation of the parametric g-formula

**Authors: Jing Li, Sophia Rein, Sean McGrath, Roger Logan, Ryan O’Dea, Miguel Hernán**


## Overview
The pygformula implements the parametric g-formula (Robins 1986) to estimate the risk or mean of an outcome under 
hypothetical sustained treatment strategies or interventions from longitudinal data with time-varying measurements of 
treatments and confounders.


### Features: 

* Covariates: binary, normal, categorical, bounded normal, zero-inflated normal, truncated normal, absorbing and time variable. 
* Treatments: binary or continuous/multi-level time-varying treatments; multiple treatments.
* Interventions: natural course intervention, static interventions, dynamic interventions, and threshold interventions.
* Outcomes: survival outcomes, fixed binary end of follow-up outcomes, and fixed continuous end of follow-up outcomes.
* Censoring events.
* Competing events.
* Visit process.
* Priori knowledge incorporation.


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

If you have any issues, please open an [issue](https://github.com/CausalInference/pygformula/issues) on github. 
We will regularly check the questions. For any additional questions or comments, please 
email jing_li@hsph.harvard.edu.