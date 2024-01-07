# pygformula: a python implementation of the parametric g-formula


## Overview
The pygformula is a python package that implements the parametric g-formula to estimate the effects of sustained treatment strategies.
Specifically, it implements the noniterative conditional expectation estimator of parametric g-formula which accommodates 
different types of covariates, different types of outcomes, different types of interventions, as well as more complicated scenarios such as competing events
, censoring events and visit process etc.

### Features: 

* Covariates: binary, normal, categorical, bounded normal, zero-inflated normal, truncated normal, absorbing and time variable. 
* Outcomes: survival outcomes, fixed binary end of follow-up outcomes, and fixed continuous end of follow-up outcomes.
* Interventions: natural course intervention, static interventions, dynamic interventions, and threshold interventions.
* Competing events.
* Censoring events.
* Visit process.


## Requirements

The package supports python ≥ 3.8. It requires these necessary dependencies:

- joblib
- lifelines
- matplotlib
- numpy
- pandas
- prettytable
- scipy
- statsmodels
- tqdm

and these optional dependencies:

- cmprsk
- rpy2