Get Started
''''''''''''''''''''

===================
Algorithm outline
===================


The parametric g-formula estimator of the noniterative conditional expectation (NICE) requires
the specification of models for the joint density of the confounders, treatments, and outcomes over time.
The algorithm has three steps: (1) Parametric estimation, (2) Monte Carlo simulation
, and (3) Calculation of risk/mean under each intervention.

+  **Parametric estimation**: (a) estimate the conditional densities of each covariate given past covariate history
   by fitting user-specified regression models, (b) estimate the discrete hazard (for survival outcome) or mean
   (for binary/continuous end of follow-up) of the outcome conditional on past covariate history by fitting a user-specified
   regression model, (c) if the event of interest is subject to competing events and competing events are not treated as censoring events, estimate the conditional probability of the competing event
   conditional on past covariate history by fitting user-specified regression model for the competing event.

+  **Monte Carlo simulation**: (a) generate a new dataset which is usually larger than original dataset, for each covariate,
   generate simulated values at each time step using the estimated covariate models from step (1), (b) for the
   covariates that are to undergo intervention, their values are assigned according to the user-specified intervention rule,
   (c) obtain the discrete hazard / mean of the outcome based on the estimated outcome model from step (1),
   (d) if the event of interest is subject to competing events and competing events are not treated as censoring events,
   obtain the discrete hazard of the competing event based on the estimated competing model from step (1).

+  **Calculation of risk/mean under each intervention**: for binary/continuous end of follow-up, the final estimate is the mean of
   the estimated outcome of all individuals in the new dataset computed from Step (2). For survival outcome,
   the final estimate is obtained by calculating the mean of cumulative risks for all individuals using the discrete hazards computed from step (2).



Arguments:

.. automodule:: pygformula.parametric_gformula
.. autosummary:: ParametricGformula
.. autoclass:: ParametricGformula



===================
Example
===================
The observational dataset
`example_data_basicdata_nocomp <https://github.com/CausalInference/pygformula/blob/master/datasets/example_data_basicdata_nocomp.csv>`_ consists of 13,170 observations on 2,500 individuals with a maximum of 7 follow-up
times. The dataset contains the following variables:

 - id: Unique identifier for each individual.
 - t0: Time index.
 - L1: Binary time-varying covariate.
 - L2: Continuous time-varying covariate.
 - L3: Categorical baseline covariate.
 - A: Binary treatment variable.
 - Y: Outcome of interest; time-varying indicator of failure.

We are interested in the risk by the end of follow-up under the static interventions ‘‘Never treat’’ (set treatment
to 0 at all times) and ‘‘Always treat’’ (set treatment to 1 at all times).

- First, import the g-formula package:

  .. code-block::

      import pygformula
      from pygformula import ParametricGformula

- Then, load the data (here is an example of loading simulated `data <https://github.com/CausalInference/pygformula/blob/main/datasets/example_data_basicdata_nocomp.csv>`_ in the package,
  users can also load their own data) as required pandas DataFrame type

  .. code::

      from pygformula.data import load_basicdata_nocomp
      obs_data = load_basicdata_nocomp()

- Specify the name of the time variable, and the name of the individual identifier in the input data

  .. code-block::

      time_name = 't0'
      id = 'id'

- Specify covariate names, covariate types, and corresponding model statements

  .. code-block::

      covnames = ['L1', 'L2', 'A']
      covtypes = ['binary', 'bounded normal', 'binary']
      covmodels = ['L1 ~ lag1_A + lag2_A + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                   'L2 ~ lag1_A + L1 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
                   'A ~ lag1_A + L1 + L2 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0']

  If there are baseline covariates (i.e., covariate with same value at all times) in the model statement, specify them in the
  ‘‘basecovs’’ argument:

  .. code::

      basecovs = ['L3']


- Specify the static interventions of interest:

  .. code-block::

      from pygformula.parametric_gformula.interventions import static

      time_points = np.max(np.unique(obs_data[time_name])) + 1
      int_descript = ['Never treat', 'Always treat']

      Intervention1_A = [static, np.zeros(time_points)],
      Intervention2_A = [static, np.ones(time_points)],

- Specify the outcome name, outcome model statement, and the outcome type

  .. code-block::

      outcome_name = 'Y'
      ymodel = 'Y ~ L1 + L2 + L3 + A + lag1_A + lag1_L1 + lag1_L2 + t0'
      outcome_type = 'survival'

- Speficy all the arguments in the "ParametricGformula" class and call its "fit" function:

  .. code-block::

      g = ParametricGformula(obs_data = obs_data, id = id, time_name=time_name,
          covnames=covnames, covtypes=covtypes,
          covmodels=covmodels, basecovs=basecovs,
          time_points=time_points,
          Intervention1_A = [static, np.zeros(time_points)],
          Intervention2_A = [static, np.ones(time_points)],
          outcome_name=outcome_name, ymodel=ymodel,
          outcome_type = outcome_type)

      g.fit()

- Finally, get the output:

  .. image:: media/get_started_example.png
     :align: center


  - "Intervention": the name of natural course intervention and user-specified interventions.
  - "NP-risk": the nonparametric estimates of the natural course risk.
  - "g-formula risk": the parametric g-formula estimates of each interventions.
  - "Risk Ratio (RR)": the risk ratio comparing each intervention and reference intervention.
  - "Risk Difference (RD)": the risk difference comparing each intervention and reference intervention.

In the output table, the g-formula risk results under the specified interventions are shown, as well as the natural course.
Furthermore, the nonparametric risk under the natural course is provided, which can be used to assess model misspecification of parametric
g-formula. The risk ratio and risk difference comparing the specific intervention and the reference
intervention (set to natural course by default) are also calculated.

Users can also get the standard errors and 95% confidence intervals of the g-formula estimates by specifying the ‘‘nsamples’’ argument.
For example, specifying ‘‘nsamples’’ as 20 with parallel processing using 8 cores:

  .. code-block::

      g = ParametricGformula(obs_data = obs_data, id = id, time_name=time_name,
          time_points = time_points,
          Intervention1_A = [static, np.zeros(time_points)],
          Intervention2_A = [static, np.ones(time_points)],
          covnames=covnames, covtypes=covtypes,
          covmodels=covmodels, basecovs=basecovs,
          outcome_name=outcome_name, ymodel=ymodel, outcome_type=outcome_type,
          nsamples=20, parallel=True, ncores=8)
      g.fit()

The package will return following results:

  .. image:: media/get_started_example_bootstrap.jpg
     :align: center
     :width: 8.5in
     :height: 2in

The result table contains 95% lower bound and upper bound for the risk, risk difference and risk ratio for all interventions.

The pygformula also provides plots for risk curves of interventions, which can be called by:

  .. code::

     g.plot_interventions()

It will return the g-formula risk (with 95% confidence intervals if using bootstrap samples) at all follow-up times under each intervention:

  .. image:: media/get_started_example_intervention_curve.jpg
     :align: center
     :width: 5in
     :height: 4in

User can also get the plots of parametric and nonparametric estimates of
the risks and covariate means under natural course by:

  .. code::

     g.plot_natural_course()


  .. image:: media/get_started_example_all.jpg
     :align: center



**Running example** `[code] <https://github.com/CausalInference/pygformula/blob/main/running_examples/get_started_example.py>`_:

   .. code-block::

    import numpy as np
    from pygformula import ParametricGformula
    from pygformula.parametric_gformula.interventions import static
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
                 Intervention2_A = [static, np.ones(time_points)],
                 nsamples=20, parallel=True, ncores=8)
    g.fit()
    g.plot_natural_course()
    g.plot_interventions()











