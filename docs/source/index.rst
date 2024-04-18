
Welcome to pygformula's documentation!
======================================

The `pygformula <https://github.com/CausalInference/pygformula>`_ package implements the non-iterative
conditional expectation (NICE) algorithm of the g-formula algorithm [1]_ :sup:`,` [2]_. The g-formula can estimate an
outcome’s counterfactual mean or risk under hypothetical treatment strategies (interventions) when there
is sufficient information on time-varying treatments and confounders.

This package can be used for discrete or continuous time-varying treatments and for failure time outcomes or
continuous/binary end of follow-up outcomes. The package can handle a random measurement/visit process and a
priori knowledge of the data structure, as well as censoring (e.g., by loss to follow-up) and two options for
handling competing events for failure time outcomes. Interventions can be flexibly specified, both as
interventions on a single treatment or as joint interventions on multiple treatments.

For a quick overview of how to use the pygformula, see a simple example in :doc:`Get Started`.
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




