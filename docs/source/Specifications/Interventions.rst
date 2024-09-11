.. _Interventions:


Interventions
=========================

The package supports natural course intervention, static and dynamic interventions, as well as
threshold interventions [1]_ :sup:`,` [2]_ . It provides pre-coded intervention functions and options for users
to define other custom interventions that are beyond the interventions
provided.

    * **Natural course**: no intervention on any treatment variables.
    * **Static intervention**:  intervention wherein the rule for assigning treatment at each time point does not depend on past covariates.
    * **Dynamic intervention**: intervention wherein the rule for assigning treatment depends on past covariates.
    * **Threshold intervention**: intervention wherein the rule for assigning treatment at each time point depends on the natural value of treatment at the time point.

The following are the arguments for specifying the intervention of interest. If not specified, the package will
return the result without intervention, i.e., natural course result. This section introduces how to
specify different types of intervention with these arguments.


.. list-table::
    :header-rows: 1

    * - Arguments
      - Description
    * - int_descript
      - (Required) A list of strings, each of which describes a user-specified intervention.
    * - interventions
      - (Required) A dictionary whose key is the treatment name in the intervention with the format Intervention{id}_{treatment_name},
        value is a list that contains the intervention function, values required by the function, and a list of time
        points in which the intervention is applied.

The package uses keyword arguments to implement the intervention and allows any number of interventions.
When users specify each intervention, they should specify the intervention id (means the id-th intervention in all interventions,
and the id should start from 1) and treatment name in the argument name.

For static interventions, the value of the argument name is a list where the first element is the static intervention function, the second element
is the intervened values at all time points, the third element is a list specifying the time points to apply the intervention
(if not specified, the default is intervening on all time points);

An example of intervening on ‘‘A’’ in the first three time points with the ‘‘Never treat’’ intervention:

.. code-block::

      Intervention1_A = [static, np.zeros(time_points), [0, 1, 2]]

For dynamic interventions, the value of the argument name is a list where the first element is the dynamic intervention function,
the second element is a list specifying the time points to apply the intervention (if not specified, the default is intervening on all time points).

An example of intervening on ‘‘A’’ in the first three time points with a dynamic intervention:

.. code-block::

      Intervention1_A = [dynamic, [0, 1, 2]]

For threshold interventions, the value of the argument name is a list where the first element is the threshold intervention function, the second element
is the threshold values, the third element is a list specifying the time points to apply the intervention.

An example of intervening on ‘‘A’’ in the first three time points with threshold intervention:

.. code-block::

      Intervention1_A = [threshold, [0.5, float('inf')], [0, 1, 2]]

If users want to specify multiple interventions, they should use different IDs for different interventions.

An example of intervening on ‘‘A’’ with 3 different interventions:

.. code-block::

      Intervention1_A = [static, np.zeros(time_points)]
      Intervention2_A = [dynamic]
      Intervention3_A = [threshold, [0.5, float('inf')]]

If users want to specify joint intervention where there are multiple treatment variables,
they should specify different treatment name with the same intervention id.

An example of intervening on ‘‘A1’’ and ‘‘A2’’ within a static intervention:

.. code-block::

      Intervention1_A1 = [static, np.zeros(time_points)]
      Intervention1_A2 = [static, np.ones(time_points)]


Natural course intervention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If no intervention is specified, the package will return the result of natural course, containing
parametric and nonparametric natural course risk/mean outcome of g-formula. Users may assess model misspecification in
the parametric g-formula by comparing the two estimates [3]_.



**Running example** `[code] <https://github.com/CausalInference/pygformula/blob/main/running_examples/test_natural_course.py>`_:

.. code-block::

        import numpy as np
        from pygformula import ParametricGformula
        from pygformula.data import load_basicdata_nocomp

        obs_data = load_basicdata_nocomp()
        time_name = 't0'
        id = 'id'

        covnames = ['L1', 'L2', 'A']
        covtypes = ['binary', 'bounded normal', 'binary']
        covmodels = ['L1 ~ lag1_A + lag2_A + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                   'L2 ~ lag1_A + L1 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                   'A ~ lag1_A + L1 + L2 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0']

        basecovs = ['L3']

        outcome_name = 'Y'
        ymodel = 'Y ~ L1 + L2 + L3 + A + lag1_A + lag1_L1 + lag1_L2 + t0'
        outcome_type = 'survival'

        time_points = np.max(np.unique(obs_data[time_name])) + 1

        g = ParametricGformula(obs_data = obs_data, id = id, time_name=time_name,
                 time_points = time_points, covnames=covnames, covtypes=covtypes,
                 covmodels=covmodels, basecovs=basecovs, outcome_name=outcome_name,
                 ymodel=ymodel, outcome_type=outcome_type)
        g.fit()


**Output**:

    .. image:: ../media/natural_course_output.png
         :align: center



Static interventions
~~~~~~~~~~~~~~~~~~~~~~~~~~

The package has pre-coded static intervention.

.. automodule:: pygformula.interventions
.. autosummary:: static
.. autofunction:: static

which can be called by:

    .. code-block::

        from pygformula.interventions import static

When specifying the static intervention for one treatment,
the treatment value at each time step k will be replaced by the kth value in the list of intervened values.


**Running example of static intervention on one treatment variable** `[code] <https://github.com/CausalInference/pygformula/blob/main/running_examples/test_static_one_treatment.py>`_:

.. code-block::

    import numpy as np
    from pygformula import ParametricGformula
    from pygformula.interventions import static
    from pygformula.data import load_basicdata_nocomp

    obs_data = load_basicdata_nocomp()
    time_name = 't0'
    id = 'id'

    covnames = ['L1', 'L2', 'A']
    covtypes = ['binary', 'bounded normal', 'binary']
    covmodels = ['L1 ~ lag1_A + lag2_A + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                 'L2 ~ lag1_A + L1 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                 'A ~ lag1_A + L1 + L2 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0']

    basecovs = ['L3']

    time_points = np.max(np.unique(obs_data[time_name])) + 1
    int_descript = ['Always treat']

    outcome_name = 'Y'
    ymodel = 'Y ~ L1 + L2 + L3 + A + lag1_A + lag1_L1 + lag1_L2 + t0'

    g = ParametricGformula(obs_data = obs_data, id = id, time_name=time_name,
                  time_points = time_points, covnames=covnames, covtypes=covtypes,
                  covmodels=covmodels, basecovs=basecovs, int_descript = int_descript,
                  Intervention1_A = [static, np.ones(time_points), [0, 1, 4]],
                  outcome_name=outcome_name, ymodel=ymodel, outcome_type='survival')
    g.fit()


**Output**:

    .. image:: ../media/static_example_one_treatment_output.png
         :align: center


**Running example of a static intervention on multiple treatment variables** `[code] <https://github.com/CausalInference/pygformula/blob/main/running_examples/test_static_multiple_treatments.py>`_:

.. code-block::

    import numpy as np
    from pygformula import ParametricGformula
    from pygformula.interventions import static
    from pygformula.data import load_multiple_treatments_data

    obs_data = load_multiple_treatments_data()
    time_name = 't0'
    id = 'id'

    covnames = ['L1', 'L2', 'A1', 'A2']
    covtypes = ['binary', 'bounded normal', 'binary', 'binary']
    covmodels = ['L1 ~ lag1_L1',
                 'L2 ~ lag1_L1 + lag1_L2 + lag1_A2 + L1',
                 'A1 ~ lag1_L1 + lag1_L2',
                 'A2 ~ lag1_A1']

    time_points = np.max(np.unique(obs_data[time_name])) + 1
    int_descript = ['Always treat on A1 & A2']


    outcome_name = 'Y'
    ymodel = 'Y ~ L1 + L2 + A1 + A2'

    g = ParametricGformula(obs_data = obs_data, id = id, time_name=time_name,
                 time_points = time_points, covnames=covnames, covtypes=covtypes,
                 covmodels=covmodels, int_descript = int_descript,
                 Intervention1_A1 = [static, np.ones(time_points)],
                 Intervention1_A2 = [static, np.ones(time_points)],
                 outcome_name=outcome_name, ymodel=ymodel, outcome_type='survival')
    g.fit()


**Output**:

    .. image:: ../media/static_example_two_treatments.png
         :align: center


**Running example of multiple static interventions** `[code] <https://github.com/CausalInference/pygformula/blob/main/running_examples/get_started_example.py>`_:

.. code-block::

        import numpy as np
        from pygformula import ParametricGformula
        from pygformula.interventions import static
        from pygformula.data import load_basicdata_nocomp

        obs_data = load_basicdata_nocomp()
        time_name = 't0'
        id = 'id'

        covnames = ['L1', 'L2', 'A']
        covtypes = ['binary', 'bounded normal', 'binary']
        covmodels = ['L1 ~ lag1_A + lag2_A + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                   'L2 ~ lag1_A + L1 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                   'A ~ lag1_A + L1 + L2 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0']

        basecovs = ['L3']

        outcome_name = 'Y'
        ymodel = 'Y ~ L1 + L2 + L3 + A + lag1_A + lag1_L1 + lag1_L2 + t0'
        outcome_type = 'survival'

        time_points = np.max(np.unique(obs_data[time_name])) + 1
        int_descript = ['Never treat', 'Always treat']


        g = ParametricGformula(obs_data = obs_data, id = id, time_name=time_name,
                     time_points = time_points, int_descript = int_descript,
                     covnames=covnames, covtypes=covtypes,
                     covmodels=covmodels, basecovs=basecovs,
                     outcome_name=outcome_name, ymodel=ymodel, outcome_type=outcome_type,
                     Intervention1_A = [static, np.zeros(time_points)],
                     Intervention2_A = [static, np.ones(time_points)])
        g.fit()


**Output**:

    .. image:: ../media/static_multiple_interventions.png
         :align: center



Dynamic interventions
~~~~~~~~~~~~~~~~~~~~~~~~~~

For dynamic intervention, users need to define a dynamic function which encodes the dynamic treatment strategy for
one treatment variable and then pass it into the g-formula method by the intervention argument.

Example dynamic intervention: treatment is assgined (A = 1) for individuals where the covariate L2 is above a certain threshold 0.75.
Otherwise, the treatment is assigned 0.


**Sample syntax of a dynamic function example**:

.. code-block::

      def dynamic_intervention(new_df, pool, int_var, time_name, t):
          new_df.loc[new_df[time_name] == t, int_var] = 0
          new_df.loc[new_df['L2'] > 0.75, int_var] = 1

      int_descript = ['Dynamic intervention']
      Intervention1_A = [dynamic_intervention]

      g = ParametricGformula(..., int_descript = int_descript,
                            Intervention1_A = [dynamic_intervention], ...)


The dynamic intervention function should contain the following input parameters (these parameters do not all need to be specified in the function).
The function should modify the data table ‘‘new_df’’ in place, no output is returned.

+ new_df: data table of the simulated data at current time t.
+ pool: data table of the simulated data up to current time t.
+ int_var: name of the treatment variable to be intervened.
+ time_name: name of the time variable.
+ t: current time index.


**Running example** `[code] <https://github.com/CausalInference/pygformula/blob/main/running_examples/test_dynamic_intervention.py>`_:

.. code-block::

        from pygformula import ParametricGformula
        from pygformula.data import load_basicdata_nocomp

        obs_data = load_basicdata_nocomp()
        time_name = 't0'
        id = 'id'

        covnames = ['L1', 'L2', 'A']
        covtypes = ['binary', 'bounded normal', 'binary']
        covmodels = ['L1 ~ lag1_A + lag2_A + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                     'L2 ~ lag1_A + L1 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                     'A ~ lag1_A + L1 + L2 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0']

        basecovs = ['L3']

        time_points = np.max(np.unique(obs_data[time_name])) + 1

        def dynamic_intervention(new_df, pool, int_var, time_name, t):
            new_df.loc[new_df[time_name] == t, int_var] = 0
            new_df.loc[new_df['L2'] > 0.75, int_var] = 1

        int_descript = ['Dynamic intervention']

        outcome_name = 'Y'
        ymodel = 'Y ~ L1 + L2 + L3 + A + lag1_A + lag1_L1 + lag1_L2 + t0'

        g = ParametricGformula(obs_data = obs_data, id = id, time_name=time_name,
                     time_points = time_points, covnames=covnames, covtypes=covtypes,
                     covmodels=covmodels, basecovs=basecovs, int_descript = int_descript,
                     Intervention1_A = [dynamic_intervention],
                     outcome_name=outcome_name, ymodel=ymodel, outcome_type='survival')
        g.fit()


**Output**:

    .. image:: ../media/dynamic_example_output.png
         :align: center


The package also provides two pre-coded dynamic interventions with grace period: natural grace period intervention
and uniform grace period intervention. When specifying an intervention with a grace period, the list of the intervention
argument should contain the grace period intervention function in the first element, a two-element list with the
duration of grace period and conditions in the second element. (If users want to intervene on particular
time points, the third element should be specified.)


**Natural grace period intervention**: once a covariate meets a threshold level, the treatment
is initiated within a duration of the grace period. During the grace period, the treatment takes its natural value.

.. automodule:: pygformula.interventions
.. autosummary:: natural_grace_period
.. autofunction:: natural_grace_period

which can be called by:

.. code-block::

        from pygformula.interventions import natural_grace_period

**Sample syntax of an example**:

When the covariate ‘‘L1’’ equals 1, start a treatment initiation in a grace period with duration 3. The ‘‘natural_grace_period’’
specifies the type of the grace period intervention, the two-element list specifies the duration of the grace period
in the first entry and the condition of the covariate in the second entry.

.. code-block::

      from pygformula.interventions import natural_grace_period

      int_descript = ['natural grace period intervention']

      g = ParametricGformula(..., int_descript = int_descript,
          Intervention1_A = [natural_grace_period, [3, {'L1': lambda x: x == 1}]], ...)

An example of grace period intervention where the treatment is initiated when multiple conditions
(the covariate ‘‘L1’’ equals 1, and the covariate ‘‘L2’’ is greater than 2) are met:

.. code-block::

      from pygformula.interventions import natural_grace_period

      int_descript = ['natural grace period intervention']

      g = ParametricGformula(..., int_descript = int_descript,
                  Intervention1_A = [natural_grace_period, [natural_grace_period, [3, {'L1': lambda x: x == 1, 'L2': lambda x: x >= 2}]], ...)



**Running example** `[code] <https://github.com/CausalInference/pygformula/blob/main/running_examples/test_natural_grace_period.py>`_:

.. code-block::

        import numpy as np
        from pygformula import ParametricGformula
        from pygformula.interventions import natural_grace_period
        from pygformula.data import load_basicdata_nocomp

        obs_data = load_basicdata_nocomp()
        time_name = 't0'
        id = 'id'

        covnames = ['L1', 'L2', 'A']
        covtypes = ['binary', 'bounded normal', 'binary']
        covmodels = ['L1 ~ lag1_A + lag2_A + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                     'L2 ~ lag1_A + L1 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                     'A ~ lag1_A + L1 + L2 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0']

        basecovs = ['L3']

        time_points = np.max(np.unique(obs_data[time_name])) + 1

        int_descript = ['natural grace period intervention']

        outcome_name = 'Y'
        ymodel = 'Y ~ L1 + L2 + L3 + A + lag1_A + lag1_L1 + lag1_L2 + t0'

        g = ParametricGformula(obs_data = obs_data, id = id, time_name=time_name,
                       time_points = time_points,covnames=covnames, covtypes=covtypes,
                       covmodels=covmodels, basecovs=basecovs, int_descript = int_descript,
                       Intervention1_A = [natural_grace_period, [3, {'L1': lambda x: x == 1}]],
                       outcome_name=outcome_name, ymodel=ymodel, outcome_type='survival')
        g.fit()


**Output**:

    .. image:: ../media/natural_grace_period.png
         :align: center



**Uniform grace period intervention**: once a covariate meets a threshold level, the treatment
is initiated within a duration of the grace period. During grace period, treatment initiation is
randomly allocated with a uniform probability of starting treatment in each time interval of the grace period.

.. automodule:: pygformula.interventions
.. autosummary:: uniform_grace_period
.. autofunction:: uniform_grace_period

which can be called by:

.. code-block::

        from pygformula.interventions import uniform_grace_period

**Sample syntax of an example**:

When the covariate ‘‘L1’’ equals 1, start a treatment initiation in a grace period with duration 3. The ‘‘uniform_grace_period’’
specifies the type of the grace period intervention, the two-element list specifies the duration of the grace period
in the first entry and the condition of the covariate in the second entry.

.. code-block::

      from pygformula.interventions import uniform_grace_period

      int_descript = ['uniform grace period intervention']

      g = ParametricGformula(..., int_descript = int_descript,
          Intervention1_A = [uniform_grace_period, [3, {'L1': lambda x: x == 1}]], ...)


**Running example** `[code] <https://github.com/CausalInference/pygformula/blob/main/running_examples/test_uniform_grace_period.py>`_:

.. code-block::

        import numpy as np
        from pygformula import ParametricGformula
        from pygformula.interventions import uniform_grace_period
        from pygformula.data import load_basicdata_nocomp

        obs_data = load_basicdata_nocomp()
        time_name = 't0'
        id = 'id'

        covnames = ['L1', 'L2', 'A']
        covtypes = ['binary', 'bounded normal', 'binary']
        covmodels = ['L1 ~ lag1_A + lag2_A + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                     'L2 ~ lag1_A + L1 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                     'A ~ lag1_A + L1 + L2 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0']

        basecovs = ['L3']

        time_points = np.max(np.unique(obs_data[time_name])) + 1

        int_descript = ['uniform grace period intervention']

        outcome_name = 'Y'
        ymodel = 'Y ~ L1 + L2 + L3 + A + lag1_A + lag1_L1 + lag1_L2 + t0'

        g = ParametricGformula(obs_data = obs_data, id = id, time_name=time_name,
                    time_points = time_points, covnames=covnames, covtypes=covtypes,
                    covmodels=covmodels, basecovs=basecovs,int_descript = int_descript,
                    Intervention1_A = [uniform_grace_period, [3, {'L1': lambda x: x == 1}]],
                    outcome_name=outcome_name, ymodel=ymodel, outcome_type='survival')
        g.fit()


**Output**:

    .. image:: ../media/uniform_grace_period.png
         :align: center


Threshold interventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The threshold interventions in the package implement interventions that depend on the natural value of treatment.
In a threshold intervention, if a subject’s natural value of treatment at time k is below/above a
particular threshold, then set treatment to this threshold value. Otherwise, do not intervene on this subject at k.
The natural value of treatment at time k is the value of treatment that would have been observed at
time k were the intervention discontinued right before k.

The package provides pre-coded threshold function.

.. automodule:: pygformula.interventions
.. autosummary:: threshold
.. autofunction:: threshold

which can be called by

.. code-block::

    from pygformula.interventions import threshold

Users should specify a two-element list (containing minimum and maximum values) of threshold values after the threshold function
in the argument.

Example threshold intervention: if the subject’s natural value of treatment L2 falls outside the interval [0.5, inf],
set the treatment the threshold value.


**Sample syntax of example threshold intervention**:

.. code-block::

       int_descript = ['Threshold intervention']

       g = ParametricGformula(..., int_descript = int_descript,
           Intervention1_A = [threshold, [0.5, float('inf')]], ...)


**Running example** `[code] <https://github.com/CausalInference/pygformula/blob/main/running_examples/test_threshold_intervention.py>`_:

.. code-block::

        import numpy as np
        from pygformula import ParametricGformula
        from pygformula.interventions import threshold
        from pygformula.data import load_threshold_data

        obs_data = load_threshold_data()
        time_name = 't0'
        id = 'id'

        covnames = ['L1', 'L2', 'A']
        covtypes = ['binary', 'bounded normal', 'normal']
        covmodels = ['L1 ~ lag1_L1',
                     'L2 ~ lag1_L1 + lag1_L2 + L1',
                     'A ~ L1 + L2']

        time_points = np.max(np.unique(obs_data[time_name])) + 1

        int_descript = ['Threshold intervention']

        outcome_name = 'Y'
        ymodel = 'Y ~ L1 + L2 + A'

        g = ParametricGformula(obs_data = obs_data, id = id, time_name=time_name,
                      time_points = time_points, covnames=covnames, covtypes=covtypes,
                      covmodels=covmodels, int_descript = int_descript,
                      Intervention1_A = [threshold, [0.5, float('inf')]],
                      outcome_name=outcome_name, ymodel=ymodel, outcome_type='survival')
        g.fit()


**Output**:

    .. image:: ../media/threshold_example_output.png
         :align: center


.. [1] Taubman SL, Robins JM, Mittleman MA, Hernán MA. Intervening on risk factors for coronary heart disease: an
       application of the parametric g-formula. Int J Epidemiol 2009; 38(6):1599-611.
.. [2] Young JG, Hernán MA, Robins JM. Identification, estimation and approximation of risk under interventions that
       depend on the natural value of treatment using observational data. Epidemiologic Methods 2014; 3(1):1-19.
.. [3] Yu-Han Chiu, Lan Wen, Sean McGrath, Roger Logan, Issa J Dahabreh, Miguel A Hernán, Evaluating Model Specification
       When Using the Parametric G-Formula in the Presence of Censoring, American Journal of Epidemiology, Volume 192, Issue 11, November 2023, Pages 1887–1895



