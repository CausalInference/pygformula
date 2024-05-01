import numpy as np
from functools import reduce
import operator


def natural(new_df, pool, int_var, time_name, t):
    """
    This is an internal function used by natural course which does nothing on the new_df data.

    Parameters
    ----------
    new_df: DataFrame
        A DataFrame that contains the observed or simulated data at time t.

    pool: DataFrame
        A DataFrame that contains the observed or simulated data up to time t.

    int_var: List
        A list containing strings of treatment names to be intervened in a particular intervention.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    t: Int
        An integer indicating the current time index to be intervened.

    Returns
    -------
    None

    """
    pass


def static(new_df, pool, int_var, int_values, time_name, t):
    """
    This is an internal function to perform a static intervention.

    Parameters
    ----------
    new_df: DataFrame
        A DataFrame that contains the observed or simulated data at time t.

    pool: DataFrame
        A DataFrame that contains the observed or simulated data up to time t.

    int_var: List
        A list containing strings of treatment names to be intervened in a particular intervention.

    int_values: List
        A list containing the value needed when performing a particular intervention function.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    t: Int
        An integer indicating the current time index to be intervened.

    Returns
    -------
    Nothing is returned, the new_df is changed under a particular intervention.

    """
    new_df.loc[new_df[time_name] == t, int_var] = int_values[t]


def threshold(new_df, pool, int_var, threshold_values, time_name, t):
    """
    This is an internal function to perform a threshold intervention.

    Parameters
    ----------
    new_df: DataFrame
        A DataFrame that contains the observed or simulated data at time t.

    pool: DataFrame
        A DataFrame that contains the observed or simulated data up to time t.

    int_var: List
        A list containing strings of treatment names to be intervened in a particular intervention.

    threshold_values: List
        A list containing the threshold values needed when performing a threshold intervention function.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    t: Int
        An integer indicating the current time index to be intervened.

    Returns
    -------
    Nothing is returned, the new_df is changed under a particular intervention.

    """
    new_df.loc[new_df[time_name] == t, int_var] = new_df[int_var].where(new_df[int_var] > threshold_values[0], threshold_values[0])
    new_df.loc[new_df[time_name] == t, int_var] = new_df[int_var].where(new_df[int_var] < threshold_values[1], threshold_values[1])


def natural_grace_period(new_df, pool, int_var, nperiod, conditions, time_name, t):
    """
    This is a pre-coded function to perform a natural grace period intervention. Once a covariate
    meets a threshold level, the treatment (int_var) is initiated within m (nperiod) time intervals which is the duration
    of the grace period. During grace period, the treatment takes its natural value.

    Parameters
    ----------
    new_df: DataFrame
        A DataFrame that contains the observed or simulated data at time t.

    pool: DataFrame
        A DataFrame that contains the observed or simulated data up to time t.

    int_var: Str
        A string specifying the treatment variable to be intervened.

    nperiod: Int
        An integer indicating the duration of the grace period.

    conditions: Dict
        A dictionary that contains the covariate and its coditions for initiating the treatment.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    t: Int
        An integer indicating the current time index to be intervened.

    Returns
    -------
    Nothing is returned, the new_df is changed under a particular intervention.

    """

    # if condition is True, start initiation of the treatment with grace period
    masks = []
    for cond_var, condition in conditions.items():
        mask = new_df[cond_var].apply(condition)
        masks.append(mask)
    restrict_mask = reduce(operator.and_, masks)
    new_df[int_var] = np.where(restrict_mask, new_df[int_var], 0)

    # treatment is initiated by the end of the grace period
    if t >= nperiod:
        pool_data = pool[pool[time_name] == t - nperiod]
        masks = []
        for cond_var, condition in conditions.items():
            mask = pool_data[cond_var].apply(condition)
            masks.append(mask)
        restrict_mask = reduce(operator.and_, masks)
        new_df[int_var] = np.where(restrict_mask, 1, new_df[int_var])

    # treatment is set to 1 once it is initiated
    if t > 0:
        new_df[int_var] = np.where(pool.loc[pool[time_name] == t - 1, int_var] == 1, 1, new_df[int_var]).tolist()


def uniform_grace_period(new_df, pool, int_var, nperiod, conditions, time_name, t):
    """
    This is a pre-coded function to perform a uniform grace period intervention. Once a covariate
    meets a threshold level, the treatment (int_var) is initiated within m (nperiod) time intervals which is the duration
    of the grace period. During grace period, treatment initiation is randomly allocated with a uniform probability of
    starting treatment in each time interval of the grace period.

    Parameters
    ----------
    new_df: DataFrame
        A DataFrame that contains the observed or simulated data at time t.

    pool: DataFrame
        A DataFrame that contains the observed or simulated data up to time t.

    int_var: Str
        A string specifying the treatment variable to be intervened.

    nperiod: Int
        An integer indicating the duration of the grace period.

    conditions: Dict
        A dictionary that contains the covariate and its coditions for initiating the treatment.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    t: Int
        An integer indicating the current time index to be intervened.

    Returns
    -------
    Nothing is returned, the new_df is changed under a particular intervention.

    """

    def sample(prob):
        treatment = np.random.binomial(1, prob)
        return treatment

    masks = []
    for cond_var, condition in conditions.items():
        mask = new_df[cond_var].apply(condition)
        masks.append(mask)
    cond_initiation = reduce(operator.and_, masks)

    if t == 0:
        # initialize counts: the number of consecutive intervals up to t that an individual failed to receive treatment
        new_df['counts'] = 0

        # if condition is True, start initiation of the treatment according to a uniform distribution with grace period
        new_df['uni_prob'] = np.where(cond_initiation, 1 / (nperiod + 1 - new_df['counts']), 0)
        new_df[int_var] = np.where(cond_initiation, new_df['uni_prob'].apply(sample), 0)

        # update counts according to current treatment value
        new_df['counts'] = np.where(cond_initiation & (new_df[int_var] == 0), 1, 0)
        pool.loc[pool[time_name] == t, 'counts'] = new_df['counts']

    else:
        # calculate the uniform probability for initiation when 1) the grace period has started in previous step, or 2) the grace period started at current step
        new_df['uni_prob'] = np.where(pool.loc[pool[time_name] == t - 1, 'counts'] > 0, 1 / (nperiod + 1 - pool.loc[pool[time_name] == t - 1, 'counts']),
                                      np.where(cond_initiation, 1 / (nperiod + 1 - pool.loc[pool[time_name] == t - 1, 'counts']), 0))

        # get the teatment value according to the uniform probability
        new_df[int_var] = np.where((pool.loc[pool[time_name] == t - 1, 'counts'] > 0) | cond_initiation, new_df['uni_prob'].apply(sample), 0)

        # treatment is initiated by the end of the grace period
        if t >= nperiod:
            previous_pool_data = pool[pool[time_name] == t - nperiod]
            masks = []
            for cond_var, condition in conditions.items():
                mask = previous_pool_data[cond_var].apply(condition)
                masks.append(mask)
            pre_cond_initiation = reduce(operator.and_, masks)
            new_df[int_var] = np.where(pre_cond_initiation, 1, new_df[int_var])

        # treatment is set to 1 once it is initiated
        new_df[int_var] = np.where(pool.loc[pool[time_name] == t - 1, int_var] == 1, 1, new_df[int_var])

        # update current counts according to current treatment value
        new_df['counts'] = np.where((pool.loc[pool[time_name] == t - 1, 'counts'] > 0) & (new_df[int_var] == 0), pool.loc[pool[time_name] == t - 1, 'counts'] + 1,
                               np.where(cond_initiation & (new_df[int_var] == 0), pool.loc[pool[time_name] == t - 1, 'counts'] + 1, 0))


def intervention_func(new_df, pool, intervention, time_name, t):

    """
    This is an internal function which applies user-specified interventions on the data during simulation.

    Parameters
    ----------
    new_df: DataFrame
        A DataFrame that contains the observed or simulated data at time t.

    pool: DataFrame
        A DataFrame that contains the observed or simulated data up to time t.

    intervention: List
        List of lists. The k-th list contains the intervention list on k-th treatment name in the intervention.
        The intervention list contains a function implementing a particular intervention on the treatment variable,
        required values for the intervention function and a list of time points in which the intervention
        is applied.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    t: Int
        An integer indicating the current time index to be intervened.

    Returns
    -------
    Nothing is returned.

    """

    if intervention == natural:
        pass
    else:
        for i in range(len(intervention)):
            int_var = intervention[i][0]
            int_func = intervention[i][1]

            if int_func ==  static:
                int_values = intervention[i][2]
                if len(intervention[i]) == 3:  # no int_times specified, intervene on all times
                    int_func(new_df, pool, int_var, int_values, time_name, t)
                else:  # intervene on specified int_times
                    int_times = intervention[i][3]
                    if t in int_times:
                        int_func(new_df, pool, int_var, int_values, time_name, t)

            elif int_func ==  threshold:
                threshold_values = intervention[i][2]
                if len(intervention[i]) == 3:
                    int_func(new_df, pool, int_var, threshold_values, time_name, t)
                else:
                    int_times = intervention[i][3]
                    if t in int_times:
                        int_func(new_df, pool, int_var, threshold_values, time_name, t)

            elif int_func == natural_grace_period or int_func == uniform_grace_period:
                nperiod = intervention[i][2][0]
                conditions = intervention[i][2][1]
                if len(intervention[i]) == 3:
                    int_func(new_df, pool, int_var, nperiod, conditions, time_name, t)
                else:
                    int_times = intervention[i][3]
                    if t in int_times:
                        int_func(new_df, pool, int_var, nperiod, conditions, time_name, t)

            else:  # dynamic or custom intervention
                if len(intervention[i]) == 2:
                    int_func(new_df, pool, int_var, time_name, t)
                else:
                    int_times = intervention[i][2]
                    if t in int_times:
                        int_func(new_df, pool, int_var, time_name, t)




