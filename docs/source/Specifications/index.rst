Specifications
===================



The ‘‘Specifications’’ section gives detailed instructions about how to specify the required or optional
arguments in different modules of pygformula to construct a specific analysis. To use the g-formula method in the package,
the first step is to make sure that the input data meets the requirement of
:doc:`Input data`.
Then, users need to specify their parametric covariate model (see :doc:`Covariate models`),
parametric outcome model (see :doc:`Outcome model`)
, as well as the intervention of interest (see :doc:`Interventions`).
Once these required modules are well-defined, the g-formula in pygformula can be called and output the results of the method.

Additionally, if there are censoring events, the package provides the option to obtain inverse probability weighted estimates
for comparison with the g-formula estimates,
see :doc:`Censoring event`.
If there are competing events, the package provides two options for handling competing events in the case of survival outcomes, see
:doc:`Competing event`.
The package also provides option for calculating the hazard ratio of any two interventions of interest in
:doc:`Hazard ratio`.
If the data structure contains visit process, users can also perform g-formula analysis for this setting in
:doc:`Visit process`.
If there is deterministic knowledge about the relationship between the variables, it can be incorporated into the estimation
of g-formula by applying restrictions, see :doc:`Deterministic knowledge`.




**Contents**:

.. toctree::
    :maxdepth: 2

    Input data
    Interventions
    Covariate models
    Outcome model
    Censoring event
    Competing event
    Hazard ratio
    Visit process
    Deterministic knowledge
    Output







