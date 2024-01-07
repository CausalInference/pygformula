Specifications
===================

The ‘‘Specifications’’ section gives detailed instructions about how to specify the required or optional
arguments in different modules of pygformula to construct a specific analysis. To use the g-formula method in the package,
the first step is to make sure that the input data meets the requirement of
:doc:`Input data`.
Then, users need to specify their parametric covariate model in :doc:`Covariate model`,
parametric outcome model in :doc:`Outcome model`
, as well as the treatment strategy of interest in :doc:`Intervention`.
Once these required modules are well-defined, the g-formula in pygformula can be called and output the results of the method.

Additionally, if there is censoring event, users need to specify corresponding arguments of the censoring event in
:doc:`Censoring event`.
If there is competing event, users need to specify corresponding arguments of the competing event in
:doc:`Competing event`.
The package also provides option for calculating the hazard ratio of any two interventions of interest in
:doc:`Hazard ratio`.
If the data structure contains visit process, users can also perform g-formula analysis for this setting in
:doc:`Visit process`.
If there is deterministic knowledge about the relationship between the variables, it can be incorporated into the estimation
of g-formula by applying restrictions, please see :doc:`Deterministic knowledge`.

The capabilities implemented in the modules of pygformula are basically the same as gfoRmula [1]_ and GFORMULA SAS macro [2]_.



**Contents**:

.. toctree::
    :maxdepth: 2

    Input data
    Intervention
    Covariate model
    Outcome model
    Censoring event
    Competing event
    Hazard ratio
    Visit process
    Deterministic knowledge
    Output


.. [1] McGrath S, Lin V, Zhang Z, Petito LC, Logan RW, Hernán MA, Young JG. gfoRmula: An R Package for Estimating the Effects of Sustained Treatment Strategies via the Parametric g-formula. Patterns (N Y). 2020;1(3):100008. `gfoRmula <https://github.com/CausalInference/gfoRmula>`_.
.. [2] Roger W. Logan, Jessica G. Young, Sarah Taubman, Yu-Han Chiu, Sara Lodi, Sally Picciotto, Goodarz Danaei, Miguel A. Hernán. `GFORMULA SAS <https://github.com/CausalInference/GFORMULA-SAS>`_.






