
Welcome to pygformula's documentation!
======================================

The `pygformula <https://github.com/CausalInference/pygformula>`_ package implements the parametric g-formula
algorithm [1]_ [2]_. The g-formula can be used to estimate the causal effects of hypothetical time-varying
treatment interventions on the mean or risk of an outcome from longitudinal data with time-varying confounding.
This package allows: 1) binary or continuous/multi-level time-varying treatments; 2) different types of outcomes
(survival or continuous/binary end of follow-up); 3) data with competing events or truncation by death and loss
to follow-up and other types of censoring events; 4) different options for handling competing events in the case
of survival outcomes; 5) a random measurement/visit process; 6) joint interventions on multiple treatments; and
7) general incorporation of a priori knowledge of the data structure.

For a quick review of how to use the pygformula, see a simple example in :doc:`Get Started`.
For a detailed list of options, see :doc:`Specifications/index`.


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


.. [1] Robins JM. A new approach to causal inference in mortality studies with a sustained exposure period:
       application to the healthy worker survivor effect. Mathematical Modelling. 1986;7:1393–1512. [Errata (1987)
       in Computers and Mathematics with Applications 14, 917-921. Addendum (1987) in Computers and Mathematics
       with Applications 14, 923-945. Errata (1987) to addendum in Computers and Mathematics with Applications
       18, 477.
.. [2] Hernán, M.A., and Robins, J. (2020). Causal Inference: What If (Chapman & Hall/CRC).




