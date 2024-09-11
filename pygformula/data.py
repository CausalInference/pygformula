import pandas as pd


def load_basicdata():
    """
    Data description: a survival dataset that contains 11,332 observations on 2,500 individuals over 7 time points.
    Each row in the dataset corresponds to the record of one individual at one time point.

    id: Unique identifier for each individual.
    t0: Time index.
    L1: Binary time-varying covariate.
    L2: Continuous time-varying covariate.
    L3: Categorical baseline covariate. For each individual, the baseline values are repeated at each time point.
    A: Binary treatment variable.
    D: Competing event; time-varying indicator of failure.
    Y: Outcome of interest; time-varying indicator of failure.

    Returns
    -------
    A pandas dataframe
    """
    data_url = 'https://raw.githubusercontent.com/CausalInference/pygformula/main/datasets/example_data_basicdata.csv'
    data = pd.read_csv(data_url)
    return data

def load_basicdata_nocomp():
    """
    Data description: a survival dataset that contains 13,170 observations on 2,500 individuals over 7 time points.
    Each row in the dataset corresponds to the record of one individual at one time point.

    id: Unique identifier for each individual.
    t0: Time index.
    L1: Binary time-varying covariate.
    L2: Continuous time-varying covariate.
    L3: Categorical baseline covariate. For each individual, the baseline values are repeated at each time point.
    A: Binary treatment variable.
    Y: Outcome of interest; time-varying indicator of failure.

    This is a survival dataset that contains 2500 individuals with maximum 7 time points. There are
    one binary covariate L1, one normal covariate L2, one baseline covariate L3, one binary treatment variable A, and a
    binary outcome Y.

    Returns
    -------
    A pandas dataframe
    """
    data_url = 'https://raw.githubusercontent.com/CausalInference/pygformula/main/datasets/example_data_basicdata_nocomp.csv'
    data = pd.read_csv(data_url)
    return data

def load_absorbing_data():
    """
    Data description: a survival dataset that contains 6,033 observations, 1,000 individuals over 10 time points.
    Each row in the dataset corresponds to the record of one individual at one time point.

    id: Unique identifier for each individual.
    t0: Time index.
    L: Binary time-varying covariate with absorbing type, once it takes value 1, it keeps 1 at subsequent time points.
    A: Binary treatment variable.
    Y: Outcome of interest; time-varying indicator of failure.

    Returns
    -------
    A pandas dataframe
    """
    data_url = 'https://raw.githubusercontent.com/CausalInference/pygformula/main/datasets/example_data_absorbing.csv'
    data = pd.read_csv(data_url)
    return data

def load_binary_eof():
    """
    Data description: a dataset that contains 17,500 observations on 2,500 individuals over 7 time points.
    Each row in the dataset corresponds to the record of one individual at one time point.

    id: Unique identifier for each individual.
    t0: Time index.
    L1: Binary time-varying covariate.
    L2: Continuous time-varying covariate.
    L3: Categorical baseline covariate. For each individual, the baseline values are repeated at each time point.
    A: Continuous treatment variable.
    Y: Binary outcome of interest. Because this outcome is only defined at the end of follow-up, values of NA are given
    in all other time points.

    Returns
    -------
    A pandas dataframe
    """
    data_url = 'https://raw.githubusercontent.com/CausalInference/pygformula/main/datasets/example_data_binary_eof.csv'
    data = pd.read_csv(data_url)
    return data

def load_categorical():
    """
    Data description: a survival dataset that contains 7,822 observations, 1,000 individuals over 10 time points.
    Each row in the dataset corresponds to the record of one individual at one time point.

    id: Unique identifier for each individual.
    t0: Time index.
    L: Categorical covariate with 5 categories.
    A: Binary treatment variable.
    Y: Outcome of interest; time-varying indicator of failure.

    Returns
    -------
    A pandas dataframe
    """
    data_url = 'https://raw.githubusercontent.com/CausalInference/pygformula/main/datasets/example_data_categorical.csv'
    data = pd.read_csv(data_url)
    return data

def load_censor_data():
    """
    Data description: a survival dataset with censoring event that contains 118,725 observations, 10,000 individuals
    over 10 time points. Each row in the dataset corresponds to the record of one individual at one time point.
    Individuals who are censored at time k+1 only have a total of k+1 records, which correspond to time indices 0,..., k.

    id: Unique identifier for each individual.
    t0: Time index.
    L: Binary time-varying covariate.
    A: Continuous treatment variable.
    C: Censoring indicator.
    Y: Outcome of interest; time-varying indicator of failure.

    Returns
    -------
    A pandas dataframe
    """
    data_url = 'https://raw.githubusercontent.com/CausalInference/pygformula/main/datasets/example_data_censor.csv'
    data = pd.read_csv(data_url)
    return data

def load_continuous_eof():
    """
    Data description: a dataset that contains 17,500 observations on 2,500 individuals over 7 time points.
    Each row in the dataset corresponds to the record of one individual at one time point.

    id: Unique identifier for each individual.
    t0: Time index.
    L1: Categorical time-varying covariate with 3 categories.
    L2: Continuous time-varying covariate.
    L3: Categorical baseline covariate. For each individual, the baseline values are repeated at each time point.
    A: Binary treatment variable.
    Y: Continuous outcome of interest. Because this outcome is only defined at the end of follow-up, values of NA are
    given in all other time points.

    Returns
    -------
    A pandas dataframe
    """
    data_url = 'https://raw.githubusercontent.com/CausalInference/pygformula/main/datasets/example_data_continuous_eof.csv'
    data = pd.read_csv(data_url)
    return data

def load_visit_process():
    """
    Data description: a survival dataset with visit process that contains 1,739 observations on 200 individuals over 37 time points.
    Each row in the dataset corresponds to the record of one individual at one time point.

    id: Unique identifier for each individual.
    month: Time index.
    sex: Binary baseline covariate. For each individual, the baseline values are repeated at each time point.
    age: Continuous baseline covariate.
    race: Categorical baseline covariate.
    cd4_v: Continuous time-varying covariate.
    visit_cd4: Indicator of whether there is a cd4 visit/measurement.
    rna_v: Continuous time-varying covariate.
    visit_rna: Indicator of whether there is a rna visit/measurement.
    everhaart: Binary treatment variable.
    event: Outcome of interest; time-varying indicator of failure.

    Returns
    -------
    A pandas dataframe
    """
    data_url = 'https://raw.githubusercontent.com/CausalInference/pygformula/main/datasets/example_data_visit_process.csv'
    data = pd.read_csv(data_url)
    return data

def load_truncated_normal():
    """
    Data description: a survival dataset with visit process that contains 7,855 observations on 1,000 individuals over 10 time points.
    Each row in the dataset corresponds to the record of one individual at one time point.

    id: Unique identifier for each individual.
    t0: Time index.
    L: Continuous time-varying covariate with truncated normal distribution.
    A: Binary treatment variable.
    Y: Outcome of interest; time-varying indicator of failure.

    Returns
    -------
    A pandas dataframe
    """
    data_url = 'https://raw.githubusercontent.com/CausalInference/pygformula/main/datasets/example_data_truncated_normal.csv'
    data = pd.read_csv(data_url)
    return data

def load_zero_inflated_normal():
    """
    Data description: a survival dataset with visit process that contains 7,678 observations on 1,000 individuals over 10 time points.
    Each row in the dataset corresponds to the record of one individual at one time point.

    id: Unique identifier for each individual.
    t0: Time index.
    L: Continuous time-varying covariate with zero-inflated normal distribution.
    A: Binary treatment variable.
    Y: Outcome of interest; time-varying indicator of failure.

    Returns
    -------
    A pandas dataframe
    """
    data_url = 'https://raw.githubusercontent.com/CausalInference/pygformula/main/datasets/example_data_zero_inflated_normal.csv'
    data = pd.read_csv(data_url)
    return data


def load_multiple_treatments_data():
    """
    Data description: a survival dataset that contains 3,416 observations on 1,000 individuals over 5 time points.
    Each row in the dataset corresponds to the record of one individual at one time point.

    id: Unique identifier for each individual.
    t0: Time index.
    L1: Binary time-varying covariate.
    L2: Continuous time-varying covariate.
    A1: Binary treatment variable.
    A2: Binary treatment variable.
    Y: Outcome of interest; time-varying indicator of failure.

    Returns
    -------
    A pandas dataframe
    """
    data_url = 'https://raw.githubusercontent.com/CausalInference/pygformula/main/datasets/example_data_multiple_treatments.csv'
    data = pd.read_csv(data_url)
    return data


def load_threshold_data():
    """
    Data description: a survival dataset that contains 1,853 observations on 1,000 individuals over 5 time points.
    Each row in the dataset corresponds to the record of one individual at one time point.

    id: Unique identifier for each individual.
    t0: Time index.
    L1: Binary time-varying covariate.
    L2: Continuous time-varying covariate.
    A: Continuous treatment variable.
    Y: Outcome of interest; time-varying indicator of failure.

    Returns
    -------
    A pandas dataframe
    """
    data_url = 'https://raw.githubusercontent.com/CausalInference/pygformula/main/datasets/example_threshold_data.csv'
    data = pd.read_csv(data_url)
    return data