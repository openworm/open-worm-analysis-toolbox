# -*- coding: utf-8 -*-
"""
manager.py

Classes
-----------------------
StatisticsManager
WormStatistics

Notes
-----------------------
A translation of Matlab code written by Jim Hokanson,
in the SegwormMatlabClasses GitHub repo.

"""
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import seaborn as sns
import pandas as pd

from .. import utils
from .histogram import Histogram

#%%


class StatisticsManager(object):
    """
    A class that encapsulates a statistical comparison between two
    arrays of histograms, experiment histograms and control histograms.

    This class stores WormStatistics objects for each of the 726 features,
    and some shared statistical properties.

    Attributes
    ---------------------------------------
    worm_statistics_objects: numpy array of WormStatistics objects
        one object for each of 726 features
    min_p_wilcoxon: float
        minimum p_wilcoxon from all objects in worm_statistics_objects
    min_q_wilcoxon: float
        minimum q_wilcoxon from all objects in worm_statistics_objects

    (HELPER ATTRIBUTES:)
    valid_p_studentst_array: numpy array
        each non-null p_studentst from worm_statistics_objects
    valid_p_wilcoxon_array: numpy array
        each non-null p_wilcoxon from worm_statistics_objects
    q_studentst_array: numpy array
        False Discovery Rate (FDR) (i.e. q-values) for p_studentst
    q_wilcoxon_array: numpy array
        False Discovery Rate (FDR) (i.e. q-values) for p_wilcoxon

    Methods
    ---------------------------------------
    __init__
        Initializer
    plot
        Plot the histograms against each other and display statistics

    Notes
    ---------------------------------------
    Formerly seg_worm.stats.manager

    """

    def __init__(self, exp_histogram_manager, ctl_histogram_manager):
        """
        Initializes the Manager class.

        Parameters
        ---------------------------------------
        exp_histogram_manager: HistogramManager object
            Experiment
        ctl_histogram_manager: HistogramManager object
            Control

        Notes
        ---------------------------------------
        Formerly seg_worm.stats.manager.initObject

        """
        assert(len(exp_histogram_manager) ==
               len(ctl_histogram_manager))
        num_features = len(exp_histogram_manager)

        # Initialize a WormStatistics object for each of 726 features,
        # comparing experiment and control.
        self.worm_statistics_objects = np.array([None] * num_features)
        for feature_index in range(num_features):
            self.worm_statistics_objects[feature_index] = WormStatistics(
                exp_histogram_manager[feature_index],
                ctl_histogram_manager[feature_index])

        # Q-values, as introduced by Storey et al. (2002), attempt to
        # account for the False Discovery Rate from multiple hypothesis
        # testing on the same subjects.  So we must calculate here across
        # all WormStatistics objects, then assign the values to the
        # individual WormStatistics objects.
        self.q_studentst_array = utils.compute_q_values(self.p_studentst_array)
        self.q_wilcoxon_array = utils.compute_q_values(self.p_wilcoxon_array)
        for feature_index in range(num_features):
            self.worm_statistics_objects[feature_index].q_studentst = \
                self.q_studentst_array[feature_index]
            self.worm_statistics_objects[feature_index].q_wilcoxon = \
                self.q_wilcoxon_array[feature_index]

    def __getitem__(self, index):
        return self.worm_statistics_objects[index]

    @property
    def p_studentst_array(self):
        return np.array([x.p_studentst for x in self.worm_statistics_objects])

    @property
    def p_wilcoxon_array(self):
        return np.array([x.p_wilcoxon for x in self.worm_statistics_objects])

    @property
    def valid_p_studentst_array(self):
        p_studentst_array = self.p_studentst_array

        # Filter the NaN entries
        return p_studentst_array[~np.isnan(p_studentst_array)]

    @property
    def valid_p_wilcoxon_array(self):
        p_wilcoxon_array = self.p_wilcoxon_array

        # Filter the NaN entries
        return p_wilcoxon_array[~np.isnan(p_wilcoxon_array)]

    @property
    def min_p_wilcoxon(self):
        return np.nanmin(self.p_wilcoxon_array)

    @property
    def min_q_wilcoxon(self):
        return np.nanmin(self.q_wilcoxon_array)

    def __repr__(self):
        return utils.print_object(self)

    def plot(self):
        # Set the font and enable Tex
        # mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        # for Palatino and other serif fonts use:
        # rc('font',**{'family':'serif','serif':['Palatino']})
        #mpl.rc('text', usetex=True)

        # Plot some histograms
        fig = plt.figure(figsize=(12, 9))
        fig.suptitle("Histogram Plots for all Features")
        fig.text(1, 1, s="SUBTITLE", fontdict={'weight': 'bold', 'size': 8},
                 horizontalalignment='center')
        rows = 5
        cols = 4
        # for i in range(0, 700, 100):
        for i in range(rows * cols):
            ax = plt.subplot2grid((rows, cols), (i // cols, i % cols))

            self.worm_statistics_objects[i].plot(ax)

        # From http://matplotlib.org/users/legend_guide.html#using-proxy-artist
        # I learned to make a figure legend:
        green_patch = mpatches.Patch(color='g', label='Experiment')
        grey_patch = mpatches.Patch(color='0.85', label='Control')

        plt.legend(handles=[green_patch, grey_patch],
                   loc='upper left',
                   fontsize=12, bbox_to_anchor=(0, -0.1, 1, 1),
                   bbox_transform=plt.gcf().transFigure)

        # plt.tight_layout()
        plt.subplots_adjust(
            left=0.125,
            right=0.9,
            bottom=0.1,
            top=0.9,
            wspace=0.8,
            hspace=0.6)  # blank space between plots


#%%
class WormStatistics(object):
    """
    WormStatistics class.  Statistical comparison of two MergedHistogram
    objects.

    Attributes
    --------------------
    exp_histogram  (the underlying histogram)
    ctl_histogram  (the underlying histogram)

    z_score_experiment: float
    exp_p_normal: float
        Probability of the experiment data given a normality assumption
            (Using Shapiro-Wilk)
    ctl_p_normal: float
        Same as exp_p_normal, but for the control.
    p_wilcoxon: float
        Probability of the data given a null hypothesis that all data are
        drawn from the same distribution (Using Wilcoxon signed-rank test)
    p_studentst: float
        Probability of the data given a null hypothesis that all data are
        drawn from the same distribution (Using Student's t test)

    specs
    histogram_type
    motion_type
    data_type

    Notes
    --------------------
    All attributes are set to np.NaN if one or both of ctl or exp are None.

    Formerly: seg_worm.stats

    """

    #%%
    def __init__(self, exp_histogram, ctl_histogram, USE_OLD_CODE=False):
        """
        Initializer for StatisticsManager

        Parameters
        ---------------------
        exp_histogram: MergedHistogram object
            "experiment"
        ctl_histogram: MergedHistogram object
            "control"
        USE_OLD_CODE: bool
            Use old code (i.e. Schafer Lab code)

        Notes
        ------------------
        Formerly:
        seg_worm.stats.initObject(obj,exp_hist,ctl_hist)
        worm2StatsInfo
        "Compute worm statistics information and save it to a file."
        See Also:
        seg_worm.stats.helpers.swtest

        """
        if exp_histogram is None or ctl_histogram is None:
            self._z_score_experiment = np.NaN
            self._p_wilcoxon = np.NaN
            self._p_studentst = np.NaN
            self._t_statistic = np.NaN
            self._fisher_p = np.NaN

            return

        # Ensure that we are comparing the same feature!
        assert(exp_histogram.specs.name ==
               ctl_histogram.specs.name)
        #assert(exp_histogram.histogram_type == ctl_histogram.histogram_type)
        #assert(exp_histogram.motion_type == ctl_histogram.motion_type)
        #assert(exp_histogram.data_type == ctl_histogram.data_type)

        self.exp_histogram = exp_histogram
        self.ctl_histogram = ctl_histogram
        self.USE_OLD_CODE = USE_OLD_CODE

    #%%
    @property
    def z_score_experiment(self):
        """
        Calculate the z-score experiment value.

        Returns
        ------------
        float
            the z_score_experiment value

        Notes
        ------------
        This definition is slightly different than the old version,
        but matches the textual description

        TODO: it does in code, but what about in published paper?

        From Nature Methods 2013 Supplemental Description:
        "Measurements exclusively found in the experimental group have
        a zScore of infinity and those found exclusively found in the
        control are -infinity."

        """
        try:
            return self._z_score_experiment
        except AttributeError:
            USE_OLD_CODE = self.USE_OLD_CODE

            if np.isnan(self.exp_histogram.mean):
                if ((USE_OLD_CODE and self.is_exclusive) or (
                        ~USE_OLD_CODE and self.ctl_histogram.num_valid_videos > 1)):
                    self._z_score_experiment = -np.Inf
                else:
                    self.z_score_experiment = np.NaN

            elif np.isnan(self.ctl_histogram.mean):
                if ((USE_OLD_CODE and self.is_exclusive) or (
                        ~USE_OLD_CODE and self.exp_histogram.num_valid_videos > 1)):
                    self._z_score_experiment = np.Inf
                else:
                    self._z_score_experiment = np.NaN

            else:
                # This might need to be means_per_video, not the mean ...
                # - @JimHokanson
                self._z_score_experiment = (
                    (self.exp_histogram.mean - self.ctl_histogram.mean) /
                    self.ctl_histogram.std)

            return self._z_score_experiment

    #%%
    @property
    def exp_p_normal(self):
        if self.exp_histogram is None:
            return np.NaN
        else:
            return self.exp_histogram.p_normal

    @property
    def ctl_p_normal(self):
        if self.ctl_histogram is None:
            return np.NaN
        else:
            return self.ctl_histogram.p_normal

    #%%
    @property
    def p_studentst(self):
        """
        p-value calculated using the Student's t-test.

        Rules:

        1. If no valid means exist in one, but all exist in the other:
            Use Fisher's exact test.

        2. Otherwise use Student's t-test

        Notes
        ----------
        mattest (bioinformatics toolbox)

        mattest([exp_histogram_manager.mean_per_video]',
                [ctl_histogram_manager.mean_per_video]')
        http://www.mathworks.com/help/bioinfo/ref/mattest.html
        perform an unpaired t-test for differential expression with
        a standard two-tailed and two-sample t-test on every gene in
        DataX and DataY and return a p-value for each gene.
        PValues = mattest(DataX, DataY)
        p_studentst_all is a 726x1 matrix with values between 0 and 1.

        (From SciPy docs:)
        Calculates the T-test for the means of TWO INDEPENDENT samples
        of scores.
        This is a two-sided test for the null hypothesis that 2
        independent samples have identical average (expected) values.
        This test assumes that the populations have identical variances.

        """
        try:
            return self._p_studentst
        except AttributeError:
            # Scenario 1
            if self.is_exclusive:
                return self.fisher_p

            # Scenario 2
            else:
                _, self._p_studentst = \
                    sp.stats.ttest_ind(self.exp_histogram.valid_mean_per_video,
                                       self.ctl_histogram.valid_mean_per_video)

            return self._p_studentst

    @property
    def p_wilcoxon(self):
        """
        p-value calculated using the Wilcoxon signed-rank test.

        Rules:

        1. If no valid means exist in one, but all exist in the other:
            Use Fisher's exact test.

        2. If at least one mean exists in both:
            Use the Wilcoxon signed-rank test.

        3. If no valid means exist in either:
            NaN.

        """
        try:
            return self._p_wilcoxon
        except AttributeError:
            # Scenario 1
            if self.is_exclusive:
                return self.fisher_p

            # Scenario 2
            elif not (self.exp_histogram.no_valid_videos or
                      self.ctl_histogram.no_valid_videos):
                _, self._p_wilcoxon = \
                    sp.stats.ranksums(self.exp_histogram.valid_mean_per_video,
                                      self.ctl_histogram.valid_mean_per_video)
            # Scenario 3
            else:
                self._p_wilcoxon = np.NaN

            return self._p_wilcoxon
    #%%

    @property
    def specs(self):
        assert(self.exp_histogram.specs.name ==
               self.ctl_histogram.specs.name)
        return self.exp_histogram.specs

    @property
    def histogram_type(self):
        return "histogram_type"

    @property
    def motion_type(self):
        return "motion_type"

    @property
    def data_type(self):
        return "data_type"

    #%%
    # Internal methods: not really intended for others to consume.

    @property
    def fisher_p(self):
        """
        Return Fisher's exact method

        Notes
        ---------------
        Original Matlab version
        self.p_wilcoxon = seg_worm.stats.helpers.fexact(*params)

        """
        try:
            return self._fisher_p
        except AttributeError:
            # This is a literal translation of the code (I think)
            # I'm a bit confused by it ...  - @JimHokanson
            num_exp_videos = self.exp_histogram.num_videos
            num_ctl_videos = self.ctl_histogram.num_videos
            num_videos = num_exp_videos + num_ctl_videos

            # This is a strange step, I don't know what it means...
            # Why this specific list of values, it's strange.
            # -@MichaelCurrie
            params = np.array([num_exp_videos, num_videos,
                               num_exp_videos, num_exp_videos])

            _, self._fisher_p = sp.stats.fisher_exact(params)

            return self._fisher_p

    @property
    def t_statistic(self):
        try:
            return self._t_statistic
        except AttributeError:
            self._t_statistic, _ = \
                sp.stats.ttest_ind(self.exp_histogram.valid_mean_per_video,
                                   self.ctl_histogram.valid_mean_per_video)

            return self._t_statistic

    @property
    def is_exclusive(self):
        """
        A flag indicating if either experiment has all valid means but
        control has none, or vice versa.

        """
        try:
            return self._is_exclusive
        except AttributeError:
            self._is_exclusive = ((self.exp_histogram.no_valid_videos and
                                   self.ctl_histogram.all_videos_valid) or
                                  (self.ctl_histogram.no_valid_videos and
                                   self.exp_histogram.all_videos_valid))

            return self._is_exclusive

    def __repr__(self):
        return utils.print_object(self)

    #%%
    @property
    def plot_title(self):
        """
        Return the title for the plot, information about control
        and experiment videos along with with p- and q- values.

        """
        exp_histogram = self.exp_histogram
        ctl_histogram = self.ctl_histogram

        # If data_type is just "all", don't bother showing it in the title
        if self.data_type == 'all':
            data_type_string = ''
        else:
            data_type_string = '- {0}'.format(self.data_type)

        title = ("{0} - {1}{2} {3}\n"
                 "WORMS = {4} [{5}] \u00A4 SAMPLES = {6:,} [{7:,}]\n").\
            format(self.specs.name.upper(),
                   self.motion_type,
                   data_type_string,
                   self.histogram_type,
                   exp_histogram.num_videos, ctl_histogram.num_videos,
                   exp_histogram.num_samples, ctl_histogram.num_samples)

        title += ("ALL = {0:.2f} +/- {1:.2f} <<< [{2:.2f} +/- {3:.2f}] "
                  "\u00A4 (p={4:.4f}, q={5:.4f})").format(
            exp_histogram.mean, exp_histogram.std,
            ctl_histogram.mean, ctl_histogram.std,
            self.p_wilcoxon, self.q_wilcoxon)

        # DEBUG: just use a short title for now:
        #title = (self.specs.name.upper())

        return title

    #%%
    def plot(self, ax, use_legend=False, use_alternate_plot=False):
        """
        Use matplotlib to plot the experiment histogram against the control.

        Note: You must still call plt.show() after calling this function.

        Parameters
        -----------
        ax: An matplotlib.axes.Axes object
            The place where the plot occurs.

        Usage example
        -----------------------
        import matplotlib.pyplot as plt

        fig = plt.figure(1)
        ax = fig.gca()
        worm_statistics_object.plot(ax)
        plt.show()

        # A more typical use would be this method being called by
        # a StatisticsManager object.

        Parameters
        -----------------------
        ax: A matplotlib.axes.Axes object
            This is the handle where we'll make the plot
        exp_hist: A Histogram object
            The "experiment"
        ctl_hist: A Histogram object
            The "control"

        """
        ctl_bins = self.ctl_histogram.bin_midpoints
        ctl_y_values = self.ctl_histogram.pdf

        exp_bins = self.exp_histogram.bin_midpoints
        exp_y_values = self.exp_histogram.pdf
        min_x = min([h[0] for h in [ctl_bins, exp_bins]])
        max_x = min([h[-1] for h in [ctl_bins, exp_bins]])

        # TODO: ADD a line for mean, and then another for std dev.
        # TODO: Do this for both experiment and control!
        # http://www.widecodes.com/CzVkXUqXPj/average-line-for-bar-chart-in-matplotlib.html

        # TODO: switch to a relative axis for x-axis
        # http://stackoverflow.com/questions/3677368

        # Decide on a background colour based on the statistical significance
        # of the particular feature.
        # The precise colour values were obtained MS Paint's eyedropper tool
        # on the background colours of the original Schafer worm PDFs
        if self.q_wilcoxon <= 0.0001:
            bgcolour = (229, 204, 255)  # 'm' # Magenta
        elif self.q_wilcoxon <= 0.001:
            bgcolour = (255, 204, 204)  # 'r' # Red
        elif self.q_wilcoxon <= 0.01:
            bgcolour = (255, 229, 178)  # 'darkorange' # Dark orange
        elif self.q_wilcoxon <= 0.05:
            bgcolour = (255, 255, 178)  # 'y' # Yellow
        else:
            bgcolour = (255, 255, 255)  # 'w' # White
        # Scale each of the R,G,and B entries to be between 0 and 1:
        bgcolour = np.array(bgcolour) / 255

        # Plot the Control histogram
        if use_alternate_plot:
            x = self.exp_histogram.data
            y = self.ctl_histogram.data

            truncated_length = min(len(x), len(y))

            df = pd.DataFrame(data={'Experiment': x[:truncated_length],
                                    'Control': y[:truncated_length]})
            # Seaborn hexbin plot
            g = sns.jointplot(x='Experiment', y='Control',
                              data=df, kind='hex', stat_func=sp.stats.wilcoxon,
                              color="#4CB391")
            g.fig.gca().set_title(self.plot_title, fontsize=10)

        else:
            plt.ticklabel_format(style='plain', useOffset=True)

            h1 = ax.fill_between(ctl_bins, ctl_y_values, alpha=1, color='0.85',
                                 label='Control')
            # Plot the Experiment histogram
            h2 = ax.fill_between(exp_bins, exp_y_values, alpha=0.5, color='g',
                                 label='Experiment')
            ax.set_axis_bgcolor(bgcolour)
            ax.set_xlabel(self.exp_histogram.specs.units, fontsize=10)
            ax.set_ylabel('Probability ($\sum P(x)=1$)', fontsize=10)
            ax.yaxis.set_ticklabels([])
            ax.yaxis.set_ticks([])
            ax.set_title(self.plot_title, fontsize=10)
            ax.set_xlim(min_x, max_x)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            # ticks only needed at bottom and right
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

        # If this is just one sub plot out of many, it's possible the caller
        # may want to make her own legend.  If not, this plot can display
        # its own legend.
        if use_legend:
            #handles, labels = ax.get_legend_handles_labels()
            # print("hi")
            ax.legend(handles=[h1, h2],
                      labels=['Control', 'Experiment'],
                      loc='upper left', fontsize=12)
