import numpy as np
import pandas as pd
import seaborn as sns
import math
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_categorical(plot_name, cov_name, categorical_obs_mean, categorical_est_mean, obs_data, time_points, time_name,
                     colors, save_path, save_figure):
    """
    This is an internal function to plot the nonparametric and parametric comparison for categorical covariates.

    Parameters
    ----------
    plot_name: Str
        A string specifying the name for plotting, which is set to "all", "risk" or one specific covariate name.

    cov_name: Str
        A string specifying the name of the categorical covariate for plotting.

    categorical_obs_mean: List
        List of lists, the length of outer list equals to the number of categories, and the inner list contains the observed
        probabilities of the current category at each time point.

    categorical_est_mean: List
        List of lists, the length of outer list equals to the number of categories, and the inner list contains the estimated
        probabilities of the current category at each time point.

    obs_data: DataFrame
        A data frame containing the observed data.

    time_points: Int
        An integer indicating the number of time points to simulate. It is set equal to the maximum number of records (K)
        that obs_data contains for any individual plus 1, if not specified by users.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    colors: List
        A list that contains two strings, the first specifies the color for plotting nonparametric estimates, the second
        specifies the color for plotting the parametric estimates.

    save_path: Path
        A path to save all the figure results. A folder will be created automatically in the current working directory
        if the save_path is not specified by users.

    save_figure: Bool
        A boolean value indicating whether to save the figure or not.

    Returns
    -------
    Nothing is returned, the figure will be shown.

    """

    if colors is None:
        obs_color = 'xkcd:pale orange'
        est_color = 'xkcd:green/blue'
    else:
        obs_color = colors[0]
        est_color = colors[1]

    all_levels = np.unique(obs_data[cov_name])
    total_width = 0.8
    width = total_width / 2
    fig = plt.figure(figsize=(10, 5))
    x = np.arange(len(all_levels))
    for t in range(time_points):
        obs_mean_t = np.array(categorical_obs_mean)[:, t]
        est_mean_t = np.array(categorical_est_mean)[:, t]
        x_obs = x - (total_width - width) / 2
        x_est = x + (total_width - width) / 2
        ax = plt.subplot(1, time_points, t + 1)
        ax.grid(linestyle="--")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.get_major_ticks()[1].gridline.set_visible(False)
        ax.yaxis.get_major_ticks()[2].gridline.set_visible(False)
        ax.yaxis.get_major_ticks()[3].gridline.set_visible(False)
        ax.yaxis.get_major_ticks()[4].gridline.set_visible(False)
        ax.yaxis.get_major_ticks()[5].gridline.set_visible(False)
        ax.bar(x_obs, obs_mean_t, width=width, color=obs_color, label='nonparametric estimates')
        ax.bar(x_est, est_mean_t, width=width, color=est_color, label='parametric g-formula estimates (NICE)')
        if t == int(time_points / 2):
            ax.set_xlabel(time_name)
        if t == 0:
            ax.set_ylabel(cov_name)
        if t != 0:
            plt.yticks([])
        plt.ylim(0, 1)
        plt.xticks(range(len(all_levels)), labels=all_levels)
        ax.set_title(t)
    handles, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', frameon=False, ncol=2)
    if save_figure:
        figure_path = os.path.join(save_path, 'figures')
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        if plot_name != 'all':
            figure_name = os.path.join(figure_path, str(plot_name) + '.jpg')
            plt.savefig(figure_name)
        else:
            figure_name = os.path.join(figure_path, str(cov_name) + '.jpg')
            plt.savefig(figure_name)
    plt.show()


def plot_natural_course(time_points, covnames, covtypes, time_name, obs_data, obs_means, est_means, censor, outcome_type, plot_name,
                       marker, markersize, linewidth, colors, save_path, save_figure, boot_table):

    """
    This is an internal function that plots the results comparison of covariate means and risks between non-parametric
    estimates and g-formula parametric estimates.

    Parameters
    ----------
    time_points: Int
        An integer indicating the number of time points to simulate. It is set equal to the maximum number of records (K)
        that obs_data contains for any individual plus 1, if not specified by users.

    covnames: List
         A list of strings specifying the names of the time-varying covariates in obs_data.

    covtypes: List
        A list of strings specifying the “type” of each time-varying covariate included in covnames.
        The supported types: "binary", "normal", "categorical", "bounded normal", "zero-inflated normal",
        "truncated normal", "absorbing", "categorical time", "square time" and "custom". The list must be the same length
        as covnames and in the same order.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    obs_data: DataFrame
        A data frame containing the observed data.

    obs_means: Dict
        A dictionary, where the key is the covariate / risk name and the value is its observational mean at all the time points.

    est_means: Dict
        A dictionary, where the key is the covariate / risk name and the value is its parametric mean at all the time points.

    censor: Bool
        A boolean value indicating the if there is a censoring event.

    outcome_type: Str
        A string specifying the "type" of outcome. The possible "types" are: "survival", "continuous_eof", and "binary_eof".

    plot_name: Str
        A string specifying the name for plotting, which is set to "all", "risk" or one specific covariate name.

    marker: Str
        A string used to customize the appearance of points in plotting.

    markersize: Int
        An integer specifies the size of the markers in plotting.

    linewidth: Float
        A number that specifies the width of the line in plotting.

    colors: List
        A list that contains two strings, the first specifies the color for plotting nonparametric estimates, the second
        specifies the color for plotting the parametric estimates.

    save_path: Path
        A path to save all the figure results. A folder will be created automatically in the current working directory
        if the save_path is not specified by users.

    save_figure: Bool
        A boolean value indicating whether to save the figure or not.

    boot_table: DataFrame
        A DataFrame with nonparametric risk and parametric risks of all interventions.

    Returns
    -------
    Nothing is returned, the figure will be shown.

    """
    if time_points == 1:
        raise ValueError('Plotting is not available when time_points = 1.')
    if covnames is not None:
        covnames = covnames.copy()

    obs_means = obs_means.copy()
    est_means = est_means.copy()

    if colors is None:
        obs_color = 'xkcd:pale orange'
        est_color = 'xkcd:green/blue'
    else:
        obs_color = colors[0]
        est_color = colors[1]

    if plot_name != 'all':
        if plot_name == 'risk':
            plt.grid(linestyle="--")
            ax = plt.gca()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            obs_mean = obs_means[plot_name]
            est_mean = est_means[plot_name]
            obs_mean = [0] + obs_mean
            est_mean = [0] + est_mean
            plt.plot(range(time_points + 1), obs_mean, color=obs_color, marker=marker, markersize=markersize,
                    linewidth=linewidth, label='IP weighted estimates' if censor else 'nonparametric estimates')
            label = 'parametric g-formula estimates (NICE) with 95% CI' if boot_table is not None else \
                'parametric g-formula estimates (NICE)'
            plt.plot(range(time_points + 1), est_mean, color=est_color, marker=marker, markersize=markersize,
                    linewidth=linewidth, label=label)
            plt.xlabel(time_name)
            plt.ylabel(plot_name)
            plt.legend()
            if boot_table is not None:
                boot_table_copy = boot_table.copy(deep=True)
                new_row = {time_name: 0, 'Intervention': 'Natural course', 'g-form risk (NICE-parametric)': 0,
                           'Risk 95% lower bound': 0, 'Risk 95% upper bound': 0}
                boot_table_copy[time_name] = boot_table_copy[time_name] + 1
                new_boot_table = pd.concat([pd.DataFrame([new_row]), boot_table_copy], ignore_index=True)
                int_df = new_boot_table[new_boot_table['Intervention'] == 'Natural course']
                sns.lineplot(data=int_df, x=time_name, y='g-form risk (NICE-parametric)', marker='o',
                             color=est_color)
                plt.fill_between(int_df[time_name], int_df['Risk 95% lower bound'], int_df['Risk 95% upper bound'],
                                 alpha=0.2, color=est_color)
            if save_figure:
                if save_path is None:
                    save_path = os.path.join(os.getcwd(), 'results')
                figure_path = os.path.join(save_path, 'figures')
                if not os.path.exists(figure_path):
                    os.makedirs(figure_path)
                figure_name = os.path.join(figure_path, str(plot_name) + '.jpg')
                plt.savefig(figure_name)
            plt.show()
        else:
            plot_cov_type = covtypes[covnames.index(plot_name)]
            if plot_cov_type == 'categorical':
               obs_mean = obs_means[plot_name]
               est_mean = est_means[plot_name]
               cov_name = plot_name
               plot_categorical(plot_name, cov_name, obs_mean, est_mean, obs_data, time_points, time_name, colors, save_path, save_figure)
            else:
                plt.grid(linestyle="--")
                ax = plt.gca()
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                obs_mean = obs_means[plot_name]
                est_mean = est_means[plot_name]
                plt.plot(range(time_points), obs_mean, color=obs_color, marker=marker, markersize=markersize,
                         linewidth=linewidth, label='IP weighted estimates' if censor else 'nonparametric estimates')
                plt.plot(range(time_points), est_mean, color=est_color, marker=marker, markersize=markersize,
                         linewidth=linewidth, label='parametric g-formula estimates (NICE)')
                plt.xlabel(time_name)
                plt.ylabel(plot_name)
                plt.legend()
                if save_figure:
                    if save_path is None:
                        save_path = os.path.join(os.getcwd(), 'results')
                    figure_path = os.path.join(save_path, 'figures')
                    if not os.path.exists(figure_path):
                        os.makedirs(figure_path)
                    figure_name = os.path.join(figure_path, str(plot_name) + '.jpg')
                    plt.savefig(figure_name)
                plt.show()

    else:
        categorical_names = []
        for k, cov_name in enumerate(covnames):
            if covtypes[k] == 'categorical':
                categorical_names.append(cov_name)
                categorical_obs_mean = obs_means[cov_name]
                categorical_est_mean = est_means[cov_name]
                plot_categorical(plot_name, cov_name, categorical_obs_mean, categorical_est_mean, obs_data, time_points, time_name, colors,
                                 save_path, save_figure)
            if covtypes[k] == 'categorical time':
                categorical_names.append(cov_name)

        ### plotting the remaining covariates and risk
        for cov_name in categorical_names:
            covnames.remove(cov_name)
            del obs_means[cov_name]
            del est_means[cov_name]

        if outcome_type == 'survival':
            plot_names = covnames + ['risk']
        else:
            plot_names = covnames
        ncol = 2
        nrow = math.ceil(len(plot_names) / ncol)
        fig = plt.figure(figsize=(10, 4 * nrow))
        for k, name in enumerate(plot_names):
            ax = plt.subplot(nrow, ncol, k + 1)
            ax.grid(linestyle="--")
            ax.set_xlabel(time_name)
            ax.set_ylabel(name)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            obs_mean = obs_means[name]
            est_mean = est_means[name]
            if name == 'risk':
                obs_mean = [0] + obs_mean
                est_mean = [0] + est_mean
                ax.plot(range(time_points + 1), obs_mean, color=obs_color, marker=marker, markersize=markersize,
                        linewidth=linewidth, label='IP weighted estimates' if censor else 'nonparametric estimates')
                ax.plot(range(time_points + 1), est_mean, color=est_color, marker=marker, markersize=markersize,
                        linewidth=linewidth, label='parametric g-formula estimates (NICE)')
            else:
                ax.plot(range(time_points), obs_mean, color=obs_color, marker=marker, markersize=markersize,
                        linewidth=linewidth, label='IP weighted estimates' if censor else 'nonparametric estimates')
                ax.plot(range(time_points), est_mean, color=est_color, marker=marker, markersize=markersize,
                        linewidth=linewidth, label='parametric g-formula estimates (NICE)')

        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(loc='upper center', bbox_to_anchor=(0.50, 0.98), handles=lines, labels=labels,
                   frameon=False, ncol=2)
        if len(plot_names) <= 4:
            plt.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)
        else:
            plt.subplots_adjust(left=0.097, right=0.977, bottom=0.121, top=0.885, wspace=0.247, hspace=1)
        if save_figure:
            if save_path is None:
                save_path = os.path.join(os.getcwd(), 'results')
            figure_path = os.path.join(save_path, 'figures')
            if not os.path.exists(figure_path):
                os.makedirs(figure_path)
            figure_name = os.path.join(figure_path, str(plot_name) + '.jpg')
            plt.savefig(figure_name)
        plt.show()


def plot_interventions(time_points, time_name, risk_results, int_descript, outcome_type,
                       colors, marker, markersize, linewidth, save_path, save_figure, boot_table):
    """
    An internal function to plot the risk results comparison of all interventions and the natural course.

    Parameters
    ----------
    time_points: Int
        An integer indicating the number of time points to simulate. It is set equal to the maximum number of records (K)
        that obs_data contains for any individual plus 1, if not specified by users.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    risk_results: List
        A list that contains the risk estimates at all the time points of all interventions.

    int_descript: List
        A list of strings, each describing a user-specified intervention.

    outcome_type: Str
        A string specifying the "type" of outcome. The possible "types" are: "survival", "continuous_eof", and "binary_eof".

    colors: List
        A list that contains strings, each of which specifies the color for plotting the risk curve of the intervention.

    marker: Str
        A string used to customize the appearance of points in plotting.

    markersize: Int
        An integar specifies the size of the markers in plotting.

    linewidth: Float
        A number that specifies the width of the line in plotting.

    save_path: Path
        A path to save all the figure results. A folder will be created automatically in the current working directory
        if the save_path is not specified by users.

    save_figure: Bool
        A boolean value indicating whether to save the figure or not.

    boot_table: DataFrame
        A DataFrame with nonparametric risk and parametric risks of all interventions.

    Returns
    -------
    Nothing is returned, the figure will be shown.

    """
    if time_points == 1:
        raise ValueError('Plotting is not available when time_points = 1.')
    if outcome_type != 'survival':
        raise ValueError('The plot_interventions function is only applicable when the outcome_type is survival.')

    if colors is None:
        colors = list(mcolors.XKCD_COLORS)[:len(int_descript)]

    plt.grid(linestyle="--")
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    for index, intervention_name in enumerate(int_descript):
        risk_result = risk_results[index]
        risk_result = risk_result.copy()
        risk_result.insert(0, 0)
        label = intervention_name + ' with 95% CI' if boot_table is not None else intervention_name
        plt.plot(range(time_points + 1), risk_result, marker=marker, markersize=markersize,
                 color=colors[index], label=label, linewidth=linewidth)
        plt.xlabel(time_name, fontsize=15)
        plt.ylabel("risk", fontsize=15)
        plt.legend(loc=0, numpoints=1, frameon=False)
        leg = plt.gca().get_legend()
        ltext = leg.get_texts()
        plt.setp(ltext, fontsize=12)
        plt.xlabel(time_name, fontsize=15)
        plt.ylabel("risk", fontsize=15)
        plt.legend(loc=0, numpoints=1, frameon=False)
        leg = plt.gca().get_legend()
        ltext = leg.get_texts()
        plt.setp(ltext, fontsize=12)
        if boot_table is not None:
            boot_table_copy = boot_table.copy(deep=True)
            new_row = {time_name: 0, 'Intervention': intervention_name, 'g-form risk (NICE-parametric)': 0,
                       'Risk 95% lower bound': 0, 'Risk 95% upper bound': 0}
            boot_table_copy[time_name] = boot_table_copy[time_name] + 1
            new_boot_table = pd.concat([pd.DataFrame([new_row]), boot_table_copy], ignore_index=True)
            int_df = new_boot_table[new_boot_table['Intervention'] == intervention_name]
            sns.lineplot(data=int_df, x=time_name, y='g-form risk (NICE-parametric)', marker='o', color=colors[index])
            plt.fill_between(int_df[time_name], int_df['Risk 95% lower bound'], int_df['Risk 95% upper bound'],
                             alpha=0.2, color=colors[index])
    if save_figure:
        if save_path is None:
            save_path = os.path.join(os.getcwd(), 'results')
        figure_path = os.path.join(save_path, 'figures')
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        figure_name = os.path.join(figure_path, 'intervention_curve.jpg')
        plt.savefig(figure_name)
    plt.show()