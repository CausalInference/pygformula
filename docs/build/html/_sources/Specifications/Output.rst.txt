.. _Output:


Output
=================


Numerical results
~~~~~~~~~~~~~~~~~~~~~~~~~~

The package provides the following outputs:

+ **Data table of g-formula estimates**: The result table of g-formula estimates is returned by the fit function, containing (1) the nonparametric estimates
  of the natural course risk/mean outcome, (2) the parametric g-formula estimates of the risk/mean outcome under each user-specified intervention,
  (3) the risk ratio between each intervention and the reference intervention (natural course by default, can be specified in the argument ‘‘ref_int’’),
  (4) the risk difference between each intervention and the reference intervention.


+ **Simulated data table for interventions**: The package gives the simulated data table in the simulation step under
  each specified intervention, which can be obtained by:

     .. code::

        sim_data = g.summary_dict['sim_data']

  To get the simulated data under a particular intervention:

     .. code::

        sim_data = g.summary_dict['sim_data'][intervention_name]


+ **The IP weights**: To get the inverse probability weights when there is censoring event:

     .. code::

        ip_weights = g.summary_dict['IP_weights']



+ **The fitted models**: The package gives the fitted model for each covariate, outcome, competing event (if applicable), censoring event (if applicable).
  First the argument ‘‘model_fits’’ should be set to True, then the fitted models can be obtained by:

     .. code::

        fitted_models = g.summary_dict['model_fits_summary']

  To get the fitted model for a particular variable:

     .. code::

        fitted_model = g.summary_dict['model_fits_summary'][variable_name]

+ **The coefficients**: The package gives the parameter estimates of all the fitted models, which can be obtained by:

     .. code::

        model_coeffs = g.summary_dict['model_coeffs']

  To get the coefficients of the fitted model for a particular variable, please use:

     .. code::

        model_coeffs = g.summary_dict['model_coeffs'][variable_name]


+ **The standard errors**: The package gives the standard errors of the parameter estimates of all the fitted models, which can be obtained by:

     .. code::

        model_stderrs = g.summary_dict['model_stderrs']

  To get the standard errors of the fitted model for a particular variable, please use:

     .. code::

        model_stderrs = g.summary_dict['model_stderrs'][variable_name]

+ **The variance-covariance matrices**: The package gives the variance-covariance matrices of the parameter estimates of all the fitted models,
  which can be obtained by:

     .. code::

        model_vcovs = g.summary_dict['model_vcovs']

  To get the variance-covariance matrix of the parameter estimates of the fitted model for a particular variable, please use:

     .. code::

        model_vcovs = g.summary_dict['model_vcovs'][variable_name]


+ **The root mean square error**: The package gives the RMSE values of the fitted models, which can be obtained by:

     .. code::

        rmses = g.summary_dict['rmses']

  To get the RMSE of the fitted model for a particular variable, please use:

     .. code::

        rmses = g.summary_dict['rmses'][variable_name]

+ **Nonparametric estimates at each time point**: The package gives the nonparametric estimates of all covariates and risk at each time point for survival outcomes, which can be obtained by:

     .. code::

        obs_estimates = g.summary_dict['obs_plot']

  To get the nonparametric estimates of a particular variable, e.g., risk, please use:

     .. code::

        obs_estimates = g.summary_dict['obs_plot']['risk']

+ **Parametric estimates at each time point**: The package gives the parametric estimates of all covariates and risk at each time point for survival outcomes, which can be obtained by:

     .. code::

        est_estimates = g.summary_dict['est_plot']

  To get the parametric estimates of a particular variable, e.g., risk, please use:

     .. code::

        est_estimates = g.summary_dict['est_plot']['risk']


+ **Hazard ratio**: The package gives hazard ratio value for the two interventions specified, which can be obtained by:

     .. code::

        hazard_ratio = g.summary_dict['hazard_ratio']

The package also implement nonparametric bootstrapping to obtain 95% confidence intervals for risk/mean estimates
by repeating the algorithm for many bootstrap samples. Users can choose the argument ‘‘nsamples’’ to specify the number of new generated bootstrap samples.
Users may choose the argument ‘‘parallel’’ to parallelize bootstrapping and simulation step under each intervention to
make the algorithm run faster. The argument ‘‘ncores’’ can be used to specify the desired number of CPU cores
in parallarization.

The package provides two ways for calculating the confidence intervals
in argument ‘‘ci_method’’, ‘‘percentile’’ means using percentile bootstrap method which takes the 2.5th and 97.5th percentiles of the bootstrap estimates to get the 95% confidence interval,
"normal" means using the normal bootstrap method which uses the the original estimate and
the standard deviation of the bootstrap estimates to get the normal approximation 95% confidence interval.

+ **The g-formula estimates of bootstrap samples**: The package gives the parametric g-formula estimates of all
  bootstrap samples, which can be obtained by:

     .. code-block::

        g = ParametricGformula(..., nsamples = 20, parallel=True, n_core=10, ci_method = 'percentile', ...)
        g.fit()
        bootests = g.summary_dict['bootests']

  To get the parametric g-formula estimates of a particular bootstrap sample, please use:

     .. code::

        g.summary_dict['bootests']['sample_{id}_estimates']

  where id is the sample id which should be an integer between 0 and ‘‘nsamples’’ - 1.


+ **The coefficients of bootstrap samples**: The package gives the parameter estimates of all the fitted models for all generated
  bootstrap samples, which can be obtained by:

     .. code-block::

        g = ParametricGformula(..., nsamples = 20, parallel=True, n_core=10, ci_method = 'percentile', boot_diag=True, ...)
        g.fit()
        bootcoeffs = g.summary_dict['bootcoeffs']

Note that the ‘‘boot_diag’’ should be set to true when to obtain the coefficients, standard errors or variance-covariance matrices
of bootstrap samples.

  To get the coefficients of a particular bootstrap sample, please use:

     .. code::

        g.summary_dict['bootcoeffs']['sample_{id}_coeffs']

+ **The standard errors of bootstrap samples**: The package gives the standard errors of the parameter estimates of all the fitted models for all generated
  bootstrap samples, which can be obtained by:

     .. code-block::

        g = ParametricGformula(..., nsamples = 20, parallel=True, n_core=10, ci_method = 'percentile', boot_diag=True, ...)
        g.fit()
        bootstderrs = g.summary_dict['bootstderrs']

  To get the standard errors of a particular bootstrap sample, please use:

     .. code::

        g.summary_dict['bootstderrs']['sample_{id}_stderrs']


+ **The variance-covariance matrices of bootstrap samples**: The package gives the variance-covariance matrices of the parameter estimates of all the fitted models for all generated
  bootstrap samples, which can be obtained by:

     .. code-block::

        g = ParametricGformula(..., nsamples = 20, parallel=True, n_core=10, ci_method = 'percentile', boot_diag=True, ...)
        g.fit()
        bootvcovs = g.summary_dict['bootvcovs']

  To get the variance-covariance matrices of a particular bootstrap sample, please use:

     .. code::

        g.summary_dict['bootvcovs']['sample_{id}_vcovs']


Note that to get bootstrap results of coefficients, standard errors, and variance-covariance matrices, the argument
‘‘boot_diag’’ must be set to True.

All the output results above can be saved by the argument ‘‘save_results’’, once it is set to True,
results will be saved locally by creating a folder automatically. Users can also specify the folder path by the
argument ‘‘save_path’’:

     .. code-block::

        g = ParametricGformula(..., save_results = True, save_path = 'user-specified path', ...)
        g.fit()


**Arguments**:

.. list-table::
    :header-rows: 1

    * - Arguments
      - Description
    * - n_simul
      - (Optional) An integer indicating the number of subjects for whom to simulate data. The default is the number of subjects in the input data.
    * - ref_int
      - (Optional) An integer indicating the intervention to be used as the reference for calculating the end-of-follow-up mean
        ratio and mean difference. 0 denotes the natural course, while subsequent integers denote user-specified
        interventions in the order that they are named in interventions.
    * - nsamples
      - (Optional) An integer specifying the number of bootstrap samples to generate.
    * - parallel
      - (Optional) A boolean value indicating whether to parallelize simulations of different interventions to multiple cores.
    * - ncores
      - (Optional) An integer indicating the number of cores used in parallelization. It is set to 1 if not specified by users.
    * - model_fits
      - (Optional) A boolean value indicating whether to return the fitted models.
    * - ci_method
      - (Optional) A string specifying the method for calculating the bootstrap 95% confidence intervals, if applicable.
        The options are "percentile" and "normal".
    * - boot_diag
      - (Optional) A boolean value indicating whether to return the parametric g-formula estimates as well as the coefficients,
        standard errors, and variance-covariance matrices of the parameters of the fitted models in the bootstrap samples.
    * - save_results
      - (Optional) A boolean value indicating whether to save all the returned results to the save_path.
    * - save_path
      - (Optional) A path to save all the returned results. A folder will be created automatically in the current working directory
        if the save_path is not specified by users.
    * - seed
      - (Optional) An integar indicating the starting seed for simulations and bootstrapping.





Graphical results
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pygformula.plot


The package also provides two plotting functions: "plot_natural_course" and "plot_interventions".
The plot_natural_course function plots the curves of each covariate mean (for all types of outcome) and risk (for survival outcomes only) under g-formula parametric and
non-parametric estimation.

.. autosummary:: plot_natural_course
.. autofunction:: plot_natural_course


The plot_interventions fucntion plots the curves of risk under interventions of interest (for survival outcomes only).


.. autosummary:: plot_interventions
.. autofunction:: plot_interventions


Arguments for plotting:

.. list-table::
    :header-rows: 1

    * - Arguments
      - Description
    * - plot_name
      - A string specifying the name for plotting, which is set to "all", "risk" or one specific covariate name. Only
        applicable for the plot_natural_course function. The default is "all".
    * - colors
      - For plot_natural_course function, it is a list wth two elements, specifying the non-parametric estimate curve and parametric curve respectively.
        Users can choose colors from `matplotlib colors <https://matplotlib.org/stable/gallery/color/named_colors.html>`_.
        For plot_interventions function, it is a list wth m elements with m the number of interventions plus 1,
        specifying all intervention curves. If not specified, the function will use default colors.
    * - marker
      - A string used to customize the appearance of points in plotting. Users can also choose markers from
        `matplotlib markers <https://matplotlib.org/stable/api/markers_api.html>`_ library.
    * - markersize
      - An integer specifies the size of the markers in plotting.
    * - linewidth
      - A number that specifies the width of the line in plotting.
    * - save_figure
      - A boolean value indicating whether to save the figure or not.


Users can call the 'plot_natural_course' function by:

     .. code-block::

        g.plot_natural_course()

Users can call the 'plot_interventions' function by:

     .. code-block::

        g.plot_interventions()


Note that the plotting functions can only be applied after calling the 'g.fit' function.

The figures can be saved by the argument ‘‘save_figure’’, once it is set to True,
results will be saved locally by creating a folder automatically. If the argument ‘‘save_path’’ is specified, the figure will be saved to the corresponding folder.


**Sample syntax**:

.. code-block::

        g.plot_natural_course(plot_name='L1', colors=['blue', 'red'], markersize=5, linewidth=1, marker='v', save_figure=True)
        g.plot_interventions(colors =['green', 'red', 'yellow'], markersize=5, linewidth=1, marker='v', save_figure=True)

.. note::

   We recommend setting the ‘‘save_figure’’ as True if users want to access the figure
   when running the package on Linux system.