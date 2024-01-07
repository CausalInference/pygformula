.. _Intervention:


Intervention
=========================

The package supports natural course intervention, static and dynamic interventions, as well as
threshold interventions [1]_ [2]_ . The package provides pre-coded intervention functions and options for users
to define other custom interventions that are beyond the interventions
provided.

    * **Natural course**: no intervention on any treatment variables..
    * **Static intervention**:  intervention wherein the rule for assigning treatment at each time point does not depend on past covariates.
    * **Dynamic intervention**: intervention wherein the rule for assigning treatment depends on past covariates.
    * **Threshold intervention**: intervention wherein the rule for assigning treatment at each time point depends on the natural value of treatment at the time point.

The following is the arguments for specifying the intervention of interest. If not specified, the package will
return the result without intervention, i.e., natural course result. This section introduces how to
specify different types of intervention with these arguments.


.. list-table::
    :header-rows: 1

    * - Arguments
      - Default
      - Description
    * - int_descripts
      - None
      - (Required) A list of strings, each of which describes a user-specified intervention.
    * - intvars
      - None
      - (Required) A list, each element is a list of strings. The kth element in intvars specifies the name(s)
        of the variable(s) to be intervened on under the kth intervention in interventions.
    * - interventions
      - None
      - (Required) A list whose elements are lists of lists. Each list in interventions specifies a unique intervention on
        the relevant variable(s) in intvars. Each inner list contains a function implementing a particular intervention
        on a single variable, and required values for the specific treatment strategy. (e.g., intervention values for static strategy; threshold values for threshold strategy.)
    * - int_times
      - None
      - (Optional) A list, each element is a list. The kth list in int_times corresponds to the kth intervention in interventions.
        Each inner list specifies the time points in which the relevant intervention is applied on the corresponding
        variable in intvars. By default, this argument is set so that all interventions are applied in all the time points.



Natural course intervention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If no intervention is specified, the package will return the result of natural course, containing
parametric and nonparametric natural course risk/mean outcome of g-formula. The user may assess model misspecification in
the parametric g-formula by comparing the two estimates.



**Running example**:

.. code-block::

      import numpy as np
      import pygformula
      from pygformula import ParametricGformula
      from pygformula.data import load_basicdata_nocomp

      obs_data = load_basicdata_nocomp()
      time_name = 't0'
      id_name = 'id'

      covnames = ['L1', 'L2', 'A']
      covtypes = ['binary', 'bounded normal', 'binary']
      covmodels = ['L1 ~ lag1_A + lag2_A + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                 'L2 ~ lag1_A + L1 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                 'A ~ lag1_A + L1 + L2 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0']

      basecovs = ['L3']

      outcome_name = 'Y'
      outcome_model = 'Y ~ L1 + L2 + L3 + A + lag1_A + lag1_L1 + lag1_L2 + t0'
      outcome_type = 'survival'

      time_points = np.max(np.unique(obs_data[time_name])) + 1

      g = ParametricGformula(obs_data = obs_data, id_name = id_name, time_name=time_name, time_points = time_points,
                 covnames=covnames, covtypes=covtypes, covmodels=covmodels, basecovs=basecovs,
                 outcome_name=outcome_name, outcome_model=outcome_model, outcome_type=outcome_type
                 )
      g.fit()


**Output**:

    .. image:: ../media/natural_course_output.png
         :align: center



Static interventions
~~~~~~~~~~~~~~~~~~~~~~~~~~

Users can specify multiple interventions as well as multiple treatment variables for each intervention. For each
intervention, users should specify its name or description in ‘‘int_descripts’’. For each treatment variable,
users should specify the treatment name in ‘‘intvars’’, and specify corresponding
intervention function and intervened values in ‘interventions’’. If the intervention type is defined as static,
the treatment value at each time step k will be replaced by the kth value in the intervened values list.


**Sample syntax of static intervention on one treatment variable**:

.. code-block::

      int_descripts = ['Always treat']
      interventions = [[[static, np.ones(time_points)]]]
      intvars = [['A']]
      int_times = [[0, 1, 4]]

      g = ParametricGformula(..., int_descripts = int_descripts, interventions = interventions,
                        intvars = intvars, int_times = int_times, ...)

For ‘‘int_times’’ argument, the corresponding intervention is not applied for the unspecified time points, the simulated natural course
value is used in this case.


**Running example**:

.. code-block::

    import numpy as np
    import pygformula
    from pygformula import ParametricGformula
    from pygformula.data import load_basicdata_nocomp

    obs_data = load_basicdata_nocomp()
    time_name = 't0'
    id_name = 'id'

    covnames = ['L1', 'L2', 'A']
    covtypes = ['binary', 'bounded normal', 'binary']
    covmodels = ['L1 ~ lag1_A + lag2_A + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                 'L2 ~ lag1_A + L1 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                 'A ~ lag1_A + L1 + L2 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0']

    basecovs = ['L3']

    time_points = np.max(np.unique(obs_data[time_name])) + 1
    int_descripts = ['Always treat']
    interventions = [[[static, np.ones(time_points)]]]
    intvars = [['A']]
    int_times = [[0, 1, 4]]

    outcome_name = 'Y'
    outcome_model = 'Y ~ L1 + L2 + L3 + A + lag1_A + lag1_L1 + lag1_L2 + t0'

    g = ParametricGformula(obs_data = obs_data, id_name = id_name, time_name=time_name, time_points = time_points,
                 covnames=covnames,  covtypes=covtypes, covmodels=covmodels, basecovs=basecovs,
                 int_descripts = int_descripts, interventions = interventions, intvars = intvars, int_times = int_times,
                 outcome_name=outcome_name, outcome_model=outcome_model, outcome_type='survival')
    g.fit()


**Output**:

    .. image:: ../media/static_example_one_treatment_output.png
         :align: center


**Sample syntax of a static intervention on multiple treatment variables**:

.. code-block::

       int_descripts = ['Always treat']
       interventions = [[[static, np.ones(time_points)], [static, np.ones(time_points)]]]
       intvars = [['A1', 'A2']]

       g = ParametricGformula(..., int_descripts = int_descripts, interventions = interventions,
                        intvars = intvars, ...)


**Running example**:

.. code-block::

    import numpy as np
    import pygformula
    from pygformula import ParametricGformula
    from pygformula.parametric_gformula.interventions import static
    from pygformula.data import load_multiple_treatments_data

    obs_data = load_multiple_treatments_data()
    time_name = 't0'
    id_name = 'id'

    covnames = ['L1', 'L2', 'A1', 'A2']
    covtypes = ['binary', 'bounded normal', 'binary', 'binary']
    covmodels = ['L1 ~ lag1_L1',
                 'L2 ~ lag1_L1 + lag1_L2 + lag1_A2 + L1',
                 'A1 ~ lag1_L1 + lag1_L2',
                 'A2 ~ lag1_A1']

    time_points = np.max(np.unique(obs_data[time_name])) + 1
    int_descripts = ['Always treat on A1 & A2']
    interventions = [[[static, np.ones(time_points)], [static, np.ones(time_points)]]]
    intvars = [['A1', 'A2']]

    outcome_name = 'Y'
    outcome_model = 'Y ~ L1 + L2 + A1 + A2'

    g = ParametricGformula(obs_data = obs_data, id_name = id_name, time_name=time_name, time_points = time_points,
                 covnames=covnames,  covtypes=covtypes, covmodels=covmodels,
                 int_descripts = int_descripts, interventions = interventions, intvars = intvars,
                 outcome_name=outcome_name, outcome_model=outcome_model, outcome_type='survival')
    g.fit()


**Output**:

    .. image:: ../media/static_example_two_treatments.png
         :align: center


**Sample syntax of multiple interventions**:

.. code-block::

        time_points = np.max(np.unique(obs_data[time_name])) + 1
        intervention_names = ['Never treat', 'Always treat']
        interventions = [[[static, np.zeros(time_points)]], [[static, np.ones(time_points)]]]
        intvars = [['A'], ['A']]


**Running example**:

.. code-block::

      import numpy as np
      import pygformula
      from pygformula import ParametricGformula
      from pygformula.parametric_gformula.interventions import static
      from pygformula.data import load_basicdata_nocomp

      obs_data = load_basicdata_nocomp()
      time_name = 't0'
      id_name = 'id'

      covnames = ['L1', 'L2', 'A']
      covtypes = ['binary', 'bounded normal', 'binary']
      covmodels = ['L1 ~ lag1_A + lag2_A + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                   'L2 ~ lag1_A + L1 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                   'A ~ lag1_A + L1 + L2 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0']

      basecovs = ['L3']

      outcome_name = 'Y'
      outcome_model = 'Y ~ L1 + L2 + L3 + A + lag1_A + lag1_L1 + lag1_L2 + t0'
      outcome_type = 'survival'

      time_points = np.max(np.unique(obs_data[time_name])) + 1
      int_descripts = ['Never treat', 'Always treat']
      interventions = [[[static, np.zeros(time_points)]], [[static, np.ones(time_points)]]]
      intvars = [['A'], ['A']]

      g = ParametricGformula(obs_data = obs_data, id_name = id_name, time_name=time_name, time_points = time_points,
                     interventions=interventions, int_descripts = int_descripts, intvars=intvars,
                     covnames=covnames, covtypes=covtypes, covmodels=covmodels, basecovs=basecovs,
                     outcome_name=outcome_name, outcome_model=outcome_model, outcome_type=outcome_type)
      g.fit()


**Output**:

    .. image:: ../media/static_multiple_interventions.png
         :align: center



Dynamic interventions
~~~~~~~~~~~~~~~~~~~~~~~~~~

For dynamic intervention, users need to define a custom function which encodes the dynamic strategy for
one treatment variable and then pass it into the g-formula method by the ‘‘interventions’’ argument.

Example dynamic intervention: treatment is assgined (A = 1) for individuals where the covariate L2 is above a certain threshold 0.75.
Otherwise, the treatment is assigned 0.


**Sample syntax of a dynamic intervention example**:

.. code-block::

      def dynamic_intervention(new_df, pool, intvar, int_type, int_values, time_name, t):
          new_df.loc[new_df[time_name] == t, intvar] = 0
          new_df.loc[new_df['L2'] > 0.75, intvar] = 1

      intervention_names = ['Dynamic intervention']
      interventions = [[[dynamic_intervention, None]]]
      intvars = [['A']]

      g = ParametricGformula(..., intervention_names = intervention_names, interventions = interventions,
                        intvars = intvars, ...)

The dynamic intervention function should contain the following input parameters (these parameters are not all need to be specified in the function).
The function should modify the data table ‘‘new_df’’ in place, no output is required.

+ new_df: data table of the simulated data at current time t.
+ pool: data table of the simulated data from time 0 to current time t.
+ intvar: the name of treatment variable to be intervened.
+ int_type: the distribution type of the treatment variable to be intervened.
+ int_values: user-specified threshold values for an intervention (only applicable for Threshold interventions).
+ time_name: the name of time variable.
+ t: current time index.

Note that in the argument ‘‘interventions’’, the list that contains the custom dynamic function should also contain a
‘‘None’’ value although no specific value is specified here.

**Running example**:

.. code-block::

        import pygformula
        from pygformula import ParametricGformula
        from pygformula.parametric_gformula.interventions import static
        from pygformula.data import load_basicdata_nocomp

        obs_data = load_basicdata_nocomp()
        time_name = 't0'
        id_name = 'id'

        covnames = ['L1', 'L2', 'A']
        covtypes = ['binary', 'bounded normal', 'binary']
        covmodels = ['L1 ~ lag1_A + lag2_A + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                     'L2 ~ lag1_A + L1 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                     'A ~ lag1_A + L1 + L2 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0']

        basecovs = ['L3']

        time_points = np.max(np.unique(obs_data[time_name])) + 1

        def dynamic_intervention(new_df, pool, intvar, int_values, time_name, t):
            new_df.loc[new_df[time_name] == t, intvar] = 0
            new_df.loc[new_df['L2'] > 0.75, intvar] = 1

        int_descripts = ['Dynamic intervention']
        interventions = [[[dynamic_intervention, None]]]
        intvars = [['A']]

        outcome_name = 'Y'
        outcome_model = 'Y ~ L1 + L2 + L3 + A + lag1_A + lag1_L1 + lag1_L2 + t0'

        g = ParametricGformula(obs_data = obs_data, id_name = id_name, time_name=time_name, time_points = time_points,
                     covnames=covnames,  covtypes=covtypes, covmodels=covmodels, basecovs=basecovs,
                     int_descripts = int_descripts, interventions = interventions, intvars = intvars,
                     outcome_name=outcome_name, outcome_model=outcome_model, outcome_type='survival')
        g.fit()


**Output**:

    .. image:: ../media/dynamic_example_output.png
         :align: center

The package also provides two pre-coded dynamic interventions with grace period: natural grace period intervention
and uniform grace period intervention.

**Natural grace period intervention**:  once a conditional covariate meets a threshold level, the treatment
is initiated within a duration of the grace period. During the grace period, the treatment takes its natural value.


**Sample syntax of an example**:

When the covariate ‘‘L1’’ equals 1, start a treatment initiation within 3 time points. The ‘‘natural_grace_period’’
specifies the type of the grace period intervention, the two-element list specifies the duration of the grace period
in the first entry and the condition of the covariate in the second entry.

.. code-block::

      from pygformula.parametric_gformula.interventions import natural_grace_period

      int_descripts = ['natural grace period intervention']
      conditions = {'L1': lambda x: x == 1}
      interventions = [[[natural_grace_period, [3, conditions]]]]
      intvars = [['A']]

      g = ParametricGformula(..., int_descripts = int_descripts, interventions = interventions,
                        intvars = intvars, ...)

**Running example**:

.. code-block::

        import numpy as np
        import pygformula
        from pygformula import ParametricGformula
        from pygformula.parametric_gformula.interventions import natural_grace_period
        from pygformula.data import load_basicdata_nocomp

        obs_data = load_basicdata_nocomp()
        time_name = 't0'
        id_name = 'id'

        covnames = ['L1', 'L2', 'A']
        covtypes = ['binary', 'bounded normal', 'binary']
        covmodels = ['L1 ~ lag1_A + lag2_A + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                     'L2 ~ lag1_A + L1 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                     'A ~ lag1_A + L1 + L2 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0']

        basecovs = ['L3']

        time_points = np.max(np.unique(obs_data[time_name])) + 1

        int_descripts = ['natural grace period intervention']
        conditions = {'L1': lambda x: x == 1}
        interventions = [[[natural_grace_period, [3, conditions]]]]
        intvars = [['A']]

        outcome_name = 'Y'
        outcome_model = 'Y ~ L1 + L2 + L3 + A + lag1_A + lag1_L1 + lag1_L2 + t0'

        g = ParametricGformula(obs_data = obs_data, id_name = id_name, time_name=time_name, time_points = time_points,
                     covnames=covnames,  covtypes=covtypes, covmodels=covmodels, basecovs=basecovs,
                     int_descripts = int_descripts, interventions = interventions, intvars = intvars,
                     outcome_name=outcome_name, outcome_model=outcome_model, outcome_type='survival')
        g.fit()


**Uniform grace period intervention**  Once a conditional covariate meets a threshold level, the treatment
is initiated within a duration of the grace period. During grace period, treatment initiation is
randomly allocated with a uniform probability of starting treatment in each time interval of the grace period.

**Sample syntax of an example**:

When the covariate ‘‘L1’’ equals 1, start a treatment initiation within 3 time points. The ‘‘uniform_grace_period’’
specifies the type of the grace period intervention, the two-element list specifies the duration of the grace period
in the first entry and the condition of the covariate in the second entry.

.. code-block::

      from pygformula.parametric_gformula.interventions import uniform_grace_period

      int_descripts = ['uniform grace period intervention']
      conditions = {'L1': lambda x: x == 1}
      interventions = [[[uniform_grace_period, [3, conditions]]]]
      intvars = [['A']]

      g = ParametricGformula(..., int_descripts = int_descripts, interventions = interventions,
                        intvars = intvars, ...)


**Running example**:

.. code-block::

        import numpy as np
        import pygformula
        from pygformula import ParametricGformula
        from pygformula.parametric_gformula.interventions import uniform_grace_period
        from pygformula.data import load_basicdata_nocomp

        obs_data = load_basicdata_nocomp()
        time_name = 't0'
        id_name = 'id'

        covnames = ['L1', 'L2', 'A']
        covtypes = ['binary', 'bounded normal', 'binary']
        covmodels = ['L1 ~ lag1_A + lag2_A + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                     'L2 ~ lag1_A + L1 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                     'A ~ lag1_A + L1 + L2 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0']

        basecovs = ['L3']

        time_points = np.max(np.unique(obs_data[time_name])) + 1

        int_descripts = ['uniform grace period intervention']
        conditions = {'L1': lambda x: x == 1}
        interventions = [[[uniform_grace_period, [3, conditions]]]]
        intvars = [['A']]

        outcome_name = 'Y'
        outcome_model = 'Y ~ L1 + L2 + L3 + A + lag1_A + lag1_L1 + lag1_L2 + t0'

        g = ParametricGformula(obs_data = obs_data, id_name = id_name, time_name=time_name, time_points = time_points,
                     covnames=covnames,  covtypes=covtypes, covmodels=covmodels, basecovs=basecovs,
                     int_descripts = int_descripts, interventions = interventions, intvars = intvars,
                     outcome_name=outcome_name, outcome_model=outcome_model, outcome_type='survival')
        g.fit()

Threshold interventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The threshold interventions in the package implement interventions that depend on the natural value of treatment.
In a threshold intervention, if a subject’s natural value of treatment at time k is below/above a particular threshold
, then set treatment to this threshold value. Otherwise, do not intervene on this subject at k.
The natural value of treatment at time k is the value of treatment that would have been observed at
time k were the intervention discontinued right before k.

Example threshold intervention: if the subject’s natural value of treatment L2 falls outside the interval [0.5, inf],
set the treatment the threshold value.


**Sample syntax of example threshold intervention**:

.. code-block::

       int_descripts = ['Threshold intervention']
       interventions = [[[threshold, [0.5, float('inf')]]]]
       intvars = [['A']]

       g = ParametricGformula(..., int_descripts = int_descripts, interventions = interventions,
                        intvars = intvars, ...)

The user should specify a two-element list (containing minimum and maximum values) of int_values after the threshold function
in the argument ‘‘interventions’’.


**Running example**:

.. code-block::

        import numpy as np
        import pygformula
        from pygformula import ParametricGformula
        from pygformula.parametric_gformula.interventions import threshold
        from pygformula.data import load_threshold_data

        obs_data = load_threshold_data()
        time_name = 't0'
        id_name = 'id'

        covnames = ['L1', 'L2', 'A']
        covtypes = ['binary', 'bounded normal', 'normal']
        covmodels = ['L1 ~ lag1_L1',
                     'L2 ~ lag1_L1 + lag1_L2 + L1',
                     'A ~ L1 + L2']

        time_points = np.max(np.unique(obs_data[time_name])) + 1

        int_descripts = ['Threshold intervention']
        interventions = [[[threshold, [0.5, float('inf')]]]]
        intvars = [['A']]

        outcome_name = 'Y'
        outcome_model = 'Y ~ L1 + L2 + A'

        g = ParametricGformula(obs_data = obs_data, id_name = id_name, time_name=time_name, time_points = time_points,
                     covnames=covnames,  covtypes=covtypes, covmodels=covmodels,
                     int_descripts = int_descripts, interventions = interventions, intvars = intvars,
                     outcome_name=outcome_name, outcome_model=outcome_model, outcome_type='survival')
        g.fit()


**Output**:

    .. image:: ../media/threshold_example_output.png
         :align: center


.. [1] Taubman SL, Robins JM, Mittleman MA, Hernán MA. Intervening on risk factors for coronary heart disease: an
       application of the parametric g-formula. Int J Epidemiol 2009; 38(6):1599-611.
.. [2] Young JG, Hernán MA, Robins JM. Identification, estimation and approximation of risk under interventions that
       depend on the natural value of treatment using observational data. Epidemiologic Methods 2014; 3(1):1-19.




