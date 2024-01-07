import numpy as np
from functools import reduce
import operator


def natural(new_df, pool, intvar, intvals, time_name, t):
    """
    This is an internal function used by natural course which does nothing on the new_df data.

    Parameters
    ----------
    new_df: DataFrame
        A DataFrame that contains data frame at time t.

    pool: DataFrame
        A DataFrame that contains data frame up to time t.

    intvar: List
        A list contains strings of treatment names to be intervened in a particular intervention.

    intvals: List
        A list contains the value needed when performing a particular intervention function.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    t: Int
        An integar indicating the current time index to be intervened.

    Returns
    -------
    None

    """
    return None


def static(new_df, pool, intvar, intvals, time_name, t):
    """
    This is an internal function to perform static intervention.

    Parameters
    ----------
    new_df: DataFrame
        A DataFrame that contains data frame at time t.

    pool: DataFrame
        A DataFrame that contains data frame up to time t.

    intvar: List
        A list contains strings of treatment names to be intervened in a particular intervention.

    intvals: List
        A list contains the value needed when performing a particular intervention function.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    t: Int
        An integer indicating the current time index to be intervened.

    Returns
    -------
    Nothing is returned, the new_df is changed under a particular intervention.

    """
    new_df.loc[new_df[time_name] == t, intvar] = intvals[t]


def threshold(new_df, pool, intvar, intvals, time_name, t):
    """
    This is an internal function to perform threshold intervention.

    Parameters
    ----------
    new_df: DataFrame
        A DataFrame that contains data frame at time t.

    pool: DataFrame
        A DataFrame that contains data frame up to time t.

    intvar: List
        A list contains strings of treatment names to be intervened in a particular intervention.

    intvals: List
        A list contains the threshold values needed when performing a threshold intervention function.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    t: Int
        An integer indicating the current time index to be intervened.

    Returns
    -------
    Nothing is returned, the new_df is changed under a particular intervention.

    """
    new_df.loc[new_df[time_name] == t, intvar] = new_df[intvar].where(new_df[intvar] > intvals[0], intvals[0])
    new_df.loc[new_df[time_name] == t, intvar] = new_df[intvar].where(new_df[intvar] < intvals[1], intvals[1])


def natural_grace_period(new_df, pool, intvar, nperiod, conditions, time_name, t):
    """
    This is a pre-coded function to perform a natural grace period intervention. Once a conditional covariate (cond_var)
    meets a threshold level, the treatment (intvar) is initiated within m (nperiod) time intervals which is the duration
    of the grace period. During grace period, the treatment takes its natural value.

    Parameters
    ----------
    new_df: DataFrame
        A DataFrame that contains data frame at time t.

    pool: DataFrame
        A DataFrame that contains data frame up to time t.

    intvar: List
        A list contains strings of treatment names to be intervened in a particular intervention.

    nperiod: Int
        An integar indicating the duration of the grace period.

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
    new_df[intvar] = np.where(restrict_mask, new_df[intvar], 0)

    # treatment is initiated by the end of the grace period
    if t >= nperiod:
        pool_data = pool[pool[time_name] == t - nperiod]
        masks = []
        for cond_var, condition in conditions.items():
            mask = pool_data[cond_var].apply(condition)
            masks.append(mask)
        restrict_mask = reduce(operator.and_, masks)
        new_df[intvar] = np.where(restrict_mask, 1, new_df[intvar])

    # treatment is set to 1 once it is initiated
    if t > 0:
        new_df[intvar] = np.where(pool.loc[pool[time_name] == t - 1, intvar] == 1, 1, new_df[intvar]).tolist()


def uniform_grace_period(new_df, pool, intvar, nperiod, conditions, time_name, t):
    """
    This is a pre-coded function to perform a uniform grace period intervention. Once a conditional covariate (cond_var)
    meets a threshold level, the treatment (intvar) is initiated within m (nperiod) time intervals which is the duration
    of the grace period. During grace period, treatment initiation is randomly allocated with a uniform probability of
    starting treatment in each time interval of the grace period.

    Parameters
    ----------
    new_df: DataFrame
        A DataFrame that contains data frame at time t.

    pool: DataFrame
        A DataFrame that contains data frame up to time t.

    intvar: List
        A list contains strings of treatment names to be intervened in a particular intervention.

    nperiod: Int
        An integar indicating the duration of the grace period.

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
        new_df[intvar] = np.where(cond_initiation, new_df['uni_prob'].apply(sample), 0)

        # update counts according to current treatment value
        new_df['counts'] = np.where(cond_initiation & (new_df[intvar] == 0), 1, 0)
        pool.loc[pool[time_name] == t, 'counts'] = new_df['counts']

    else:
        # calculate the uniform probability for initiation when 1) the grace period has started in previous step, or 2) the grace period started at current step
        new_df['uni_prob'] = np.where(pool.loc[pool[time_name] == t - 1, 'counts'] > 0, 1 / (nperiod + 1 - pool.loc[pool[time_name] == t - 1, 'counts']),
                                      np.where(cond_initiation, 1 / (nperiod + 1 - pool.loc[pool[time_name] == t - 1, 'counts']), 0))

        # get the teatment value according to the uniform probability
        new_df[intvar] = np.where((pool.loc[pool[time_name] == t - 1, 'counts'] > 0) | cond_initiation, new_df['uni_prob'].apply(sample), 0)

        # treatment is initiated by the end of the grace period
        if t >= nperiod:
            previous_pool_data = pool[pool[time_name] == t - nperiod]
            masks = []
            for cond_var, condition in conditions.items():
                mask = previous_pool_data[cond_var].apply(condition)
                masks.append(mask)
            pre_cond_initiation = reduce(operator.and_, masks)
            new_df[intvar] = np.where(pre_cond_initiation, 1, new_df[intvar])

        # treatment is set to 1 once it is initiated
        new_df[intvar] = np.where(pool.loc[pool[time_name] == t - 1, intvar] == 1, 1, new_df[intvar])

        # update current counts according to current treatment value
        new_df['counts'] = np.where((pool.loc[pool[time_name] == t - 1, 'counts'] > 0) & (new_df[intvar] == 0), pool.loc[pool[time_name] == t - 1, 'counts'] + 1,
                               np.where(cond_initiation & (new_df[intvar] == 0), pool.loc[pool[time_name] == t - 1, 'counts'] + 1, 0))


def intervention_func(new_df, pool, intervention, intvar, int_time, time_name, t):
    """
    This is an internal function which applies user-specified interventions on the data during simulation.

    Parameters
    ----------
    new_df: DataFrame
        A DataFrame that contains data frame at time t.

    pool: DataFrame
        A DataFrame that contains data frame up to time t.

    intervention: List
        A list of lists, the inner list contains a function implementing a particular intervention on a single variable,
        and required values for specific treatment strategy.

    intvar: List
        A list contains strings of treatment names to be intervened in a particular intervention.

    int_time: List
        A list specifies the time points in which the relevant intervention is applied on the corresponding variable in intvar.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    t: Int
        An integer indicating the current time index to be intervened.

    Returns
    -------
    Nothing is returned.

    """
    if t in int_time and intvar != None:
        for i, var in enumerate(intvar):
            if intervention[i][0] == natural_grace_period or intervention[i][0] == uniform_grace_period:
                nperiod = intervention[i][1][0]
                conditions = intervention[i][1][1]
                intervention[i][0](new_df, pool, intvar[i], nperiod, conditions, time_name, t)
            else:
                intervention[i][0](new_df, pool, intvar[i], intervention[i][1], time_name, t)



