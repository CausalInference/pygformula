
Welcome to pygformula's documentation!
======================================
The `pygformula <https://github.com/CausalInference/pygformula>`_ is a python package that implements the parametric g-formula method
to estimate the effects of sustained treatment strategies from observational data with time-varying treatments, confounders, and outcomes [1]_ [2]_.

Specifically, this package implements the noniterative conditional expectation (NICE) estimator of parametric
g-formula which accommodates different types of covariates, different types of outcomes, different types of interventions, as well as more complicated scenarios
such as competing event, censoring event and visit process.

The goal of this package is to give an easy and convenient tool for users to perform data analysis based on the parametric g-formula method in python.
You can have a quick understanding of how to use the pygformula by going through a simple example in
:doc:`Get Started`.
If you want to perform a more specific analysis, please see :doc:`Specifications/index` in which you
will learn how to specify each module in the usage of the package.


.. toctree::
   :maxdepth: 2

   Installation
   Get Started

.. toctree::
   :maxdepth: 4

   Specifications/index

.. toctree::
   :maxdepth: 2

   Datasets
   Contact


.. [1] Robins JM. A new approach to causal inference in mortality studies with a sustained exposure period—application to control of the healthy worker survivor effect. Mathematical modelling 1986;7(9-12):1393-512.
.. [2] Hernán MA, Robins JM (2020). Causal Inference: What If. Boca Raton: Chapman & Hall/CRC.




