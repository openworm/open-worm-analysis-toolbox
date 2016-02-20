# -*- coding: utf-8 -*-
"""
Create an in-memory object, statistics_manager, that contains the
statistics generated from comparing a set of 20 Feature .mat Files:
- 10 "Experiment" files and
- 10 "Control" files.

Notes
--------------
Formerly:
https://github.com/JimHokanson/SegwormMatlabClasses/
blob/master/%2Bseg_worm/%2Btesting/%2Bstats/t001_oldVsNewStats.m

"""
import sys
import os
import pickle
import matplotlib.pyplot as plt

# We must add .. to the path so that we can perform the
# import of movement_validation while running this as
# a top-level script (i.e. with __name__ = '__main__')
sys.path.append('..')
import open_worm_analysis_toolbox as mv


def main():
    base_path = os.path.abspath(mv.user_config.EXAMPLE_DATA_PATH)
    root_path = os.path.join(base_path, '30m_wait')

    exp_histogram_manager, ctl_histogram_manager = \
        obtain_histograms(root_path, "pickled_histograms.dat")

    # ctl_histogram_manager.plot_information()

    print("Done with Histogram generation.  Now let's calculate statistics.")

    statistics_manager = \
        mv.StatisticsManager(exp_histogram_manager, ctl_histogram_manager)

    print("Comparison p and q values are %.2f and %.2f, respectively." %
          (statistics_manager.min_p_wilcoxon,
           statistics_manager.min_q_wilcoxon))

    # statistics_manager.plot()
    #statistics_manager[0].plot(ax=plt.figure().gca(), use_alternate_plot=False)
    statistics_manager[0].plot(ax=None, use_alternate_plot=True)

    # plt.savefig('michael.png')

    """
    # Plot the p-values, ranked.
    # TODO: add a line at the 0.01 and 0.05 thresholds, with annotation for
    #       the intercept.
    plt.plot(np.sort(statistics_manager.p_wilcoxon_array),
             label="Sorted p-values by Wilcoxon's signed rank test")
    plt.plot(np.sort(statistics_manager.q_wilcoxon_array),
             label="Sorted q-values by Wilcoxon's signed rank test")
    plt.ylabel("Probability", fontsize=10)
    plt.xlabel("Feature", fontsize=10)
    plt.legend(loc='best', shadow=True)
    #plt.gca().set_axis_bgcolor('m')
    plt.show()
    """

    # TODO:
    # maybe compare to the segwormmatlabclasses-generated stats somehow?

    # Y-axis is features, labeled
    # X-axis is worm videos
    # then list the p and q values
    # List if the mean is vailable or golor red if not.


def obtain_histograms(root_path, pickle_file_path):
    """
    Compute histograms for 10 experiment and 10 control feature files.

    Uses Python's pickle module to save results to disk to save time
    on future times the function is run.

    Parameters
    ----------
    root_path: string
        A path that has two subfolders, L and R, containing some .mat files,
        for the experiment and control samples, respectively.
    pickle_file_path: string
        A relative path, to the pickle file that has serialized the
        histograms.  This is generally found in the examples folder
        if one wishes to delete it to rerun the code fresh.

    Returns
    -------
    exp_histogram_manager, ctl_histogram_manager
        Both instances of HistogramManager

    """
    if os.path.isfile(pickle_file_path):
        print("Found a pickled version of the histogram managers "
              "at:\n%s\n" % pickle_file_path + "Let's attempt to "
              "unpickle rather than re-calculate, to save time...")
        with open(pickle_file_path, "rb") as pickle_file:
            exp_histogram_manager = pickle.load(pickle_file)
            ctl_histogram_manager = pickle.load(pickle_file)
    else:
        print("Could not find a pickled version of the histogram "
              "managers so let's calculate from scratch and then pickle")

        experiment_path = os.path.join(root_path, 'L')
        control_path = os.path.join(root_path, 'R')

        experiment_files = mv.utils.get_files_of_a_type(
            experiment_path, '.mat')
        control_files = mv.utils.get_files_of_a_type(control_path, '.mat')

        # We need at least 10 files in each
        assert(len(experiment_files) >= 10)
        assert(len(control_files) >= 10)

        print('Loading features from disk: experiment_files')
        experiment_features = [
            mv.WormFeatures.from_disk(x) for x in experiment_files]

        print('Starting feature expansion')
        new_experiment_features = [
            mv.feature_manipulations.expand_mrc_features(x) for x in experiment_features]

        print('Starting histograms')
        exp_histogram_manager = mv.HistogramManager(new_experiment_features)

        print('Loading features from disk: experiment_files')
        control_features = [
            mv.WormFeatures.from_disk(x) for x in control_files]

        print('Starting feature expansion')
        new_control_features = [
            mv.feature_manipulations.expand_mrc_features(x) for x in control_features]

        print('Starting histograms')
        ctl_histogram_manager = mv.HistogramManager(new_control_features)

        # Store a pickle file in the same folder as this script
        # (i.e. movement_validation/examples/)
        with open(pickle_file_path, "wb") as pickle_file:
            pickle.dump(exp_histogram_manager, pickle_file)
            pickle.dump(ctl_histogram_manager, pickle_file)

    print("Experiment has a total of " +
          str(len(exp_histogram_manager.merged_histograms)) + " histograms")

    return exp_histogram_manager, ctl_histogram_manager


if __name__ == '__main__':
    main()
