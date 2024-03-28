.. _Censoring event:

Censoring event
===================

In the presence of censoring, the name of the censoring variable in the input data should be specified,
users also need to specify a censor model.
Here the censor model is used for nonparametric estimate of natural course which applies inverse probability weighting method [1]_.
Meanwhile, the compared parametric g-formula involves a hypothetical intervention to abolish censoring.

The arguments for censoring events:

.. list-table::
    :header-rows: 1

    * - Arguments
      - Description
    * - censor_name
      - (Optional) A string specifying the name of the censoring variable in obs_data.
    * - censor_model
      - (Optional) A string specifying the model statement for the censoring variable.
    * - ipw_cutoff_quantile
      - (Optional) A percentile by which to truncate inverse probability weights.
    * - ipw_cutoff_value
      - (Optional) A cutoff value by which to truncate inverse probability weights.

Users may specify a cutoff value (in the argument ‘‘ipw_cutoff_quantile’’) or a cutoff quantile
(in the argument ‘‘ipw_cutoff_value’’) to truncate inverse probability weight.


**Sample syntax**:

.. code-block::

       censor_name = 'C'
       censor_model = 'C ~ A + L'

       g = ParametricGformula(..., censor_name = censor_name, censor_model = censor_model, ...)

.. note::

   When there are categorical covariates (which are assigned a 'C' symbol) in the model statement of censoring variable,
   please name the censoring variable any name except 'C' to avoild name confusion.


**Running example** `[code] <https://github.com/CausalInference/pygformula/blob/main/running_examples/test_censor.py>`_:

.. code-block::

        import numpy as np
        import pygformula
        from pygformula import ParametricGformula
        from pygformula.parametric_gformula.interventions import static
        from pygformula.data import load_censor_data

        obs_data = load_censor_data()
        time_name = 't0'
        id_name = 'id'

        covnames = ['L', 'A']
        covtypes = ['binary', 'normal']

        covmodels = ['L ~ lag1_L + t0',
                     'A ~ lag1_A + L + t0']

        outcome_name = 'Y'
        outcome_model = 'Y ~ A + L'

        censor_name = 'C'
        censor_model = 'C ~ A + L'

        time_points = np.max(np.unique(obs_data[time_name])) + 1
        int_descript = ['Never treat', 'Always treat']


        g = ParametricGformula(obs_data = obs_data, id_name = id_name, time_name=time_name,
            time_points = time_points,
            int_descript=int_descript,
            Intervention1_A = [static, np.zeros(time_points)],
            Intervention2_A = [static, np.ones(time_points)],
            censor_name= censor_name, censor_model=censor_model,
            covnames = covnames, covtypes = covtypes, covmodels = covmodels,
            outcome_name=outcome_name, outcome_model=outcome_model, outcome_type='survival')
        g.fit()


**Output**:

    .. image:: ../media/censor_example_output.png
         :align: center

.. [1] Yu-Han Chiu, Lan Wen, Sean McGrath, Roger Logan, Issa J Dahabreh, and Miguel A Hernán. 2022. Evaluating model specification when using the parametric g-formula in the presence of censoring. American Journal of Epidemiology.