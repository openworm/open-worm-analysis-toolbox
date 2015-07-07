# -*- coding: utf-8 -*-
"""
manager.py

Classes
---------------------------------------    
StatisticsManager
WormStatistics

A translation of Matlab code written by Jim Hokanson,
in the SegwormMatlabClasses GitHub repo.

"""
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl

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
    p_t: numpy array
        each non-null p_t from worm_statistics_objects
    p_w: numpy array
        each non-null p_w from worm_statistics_objects
    q_t: numpy array
        False Discovery Rate (FDR) (i.e. q-values) for p_t
    q_w: numpy array
        False Discovery Rate (FDR) (i.e. q-values) for p_w
    p_worm: float
        minimum p_w
    q_worm: float
        minimum q_w

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
    

    @property
    def p_t_array(self):
        return np.array([x.p_t for x in self.worm_statistics_objects])

    @property
    def p_w_array(self):
        return np.array([x.p_w for x in self.worm_statistics_objects])

    @property
    def q_w_array(self):
        # TODO: THIS MAKES NO SENSE, SINCE Q-VALUES ARE CALCULATED NOT
        # PER HISTOGRAM BUT ACROSS ALL FEATURE HISTOGRAMS.
        return np.array([x.q_w for x in self.worm_statistics_objects])

    @property
    def valid_p_t_array(self):
        p_t_array = self.p_t_array

        # Filter the NaN entries 
        return p_t_array[~np.isnan(p_t_array)]

    @property        
    def valid_p_w_array(self):
        p_w_array = self.p_w_array

        # Filter the NaN entries 
        return p_w_array[~np.isnan(p_w_array)]
        
    @property
    def q_t(self):
        return utils.compute_q_values(self.p_t)

    @property
    def q_w(self):
        return utils.compute_q_values(self.p_w)
        
    @property
    def p_worm(self):
        return np.nanmin(self.p_w_array)
    
    @property
    def q_worm(self):

        return np.nanmin(self.q_w_array)

    def __repr__(self):
        return utils.print_object(self)

    def plot(self):
        # Set the font and enable Tex
        #mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        ## for Palatino and other serif fonts use:
        #rc('font',**{'family':'serif','serif':['Palatino']})
        #mpl.rc('text', usetex=True)
        
        
        # Plot some histograms
        fig = plt.figure(figsize=(12, 9))
        fig.suptitle("Histogram Plots for all Features")
        fig.text(1,1,s="SUBTITLE", fontdict={'weight':'bold','size':8}, 
                 horizontalalignment='center')
        rows = 5; cols = 4
        #for i in range(0, 700, 100):
        for i in range(rows * cols):
            exp_histogram = self.worm_statistics_objects[i].exp_histogram
            ctl_histogram = self.worm_statistics_objects[i].ctl_histogram
            
            ax = plt.subplot2grid((rows, cols), (i // cols, i % cols))
            Histogram.plot_versus(ax, exp_histogram, ctl_histogram)
    
        # From http://matplotlib.org/users/legend_guide.html#using-proxy-artist
        # I learned to make a figure legend:
        green_patch = mpatches.Patch(color='g', label='Experiment')
        grey_patch  = mpatches.Patch(color='0.85', label='Control')
        
        plt.legend(handles=[green_patch, grey_patch],
                   loc='upper left', 
                   fontsize=12, bbox_to_anchor = (0,-0.1,1,1),
                   bbox_transform = plt.gcf().transFigure)

        #plt.tight_layout()
        plt.subplots_adjust(left=0.125, right=0.9, 
                            bottom=0.1, top=0.9,
                            wspace=0.8, hspace=0.6)# blank space between plots



#%%
class WormStatistics(object):
    """
    WormStatistics class.  Statistical comparison of two MergedHistogram 
    objects.

    Attributes
    --------------------
    exp_histogram  (the underlying histogram)
    ctl_histogram  (the underlying histogram)  
    
    z_score_experiment
    exp_p_normal
        Probability of the data given a normality assumption
    ctl_p_normal
        Probability of the data given a normality assumption
    p_w
    p_t
    
    specs
    hist_type
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
            self._p_w = np.NaN
            self._p_t = np.NaN
            self._t_statistic = np.NaN
            self._fisher_p = np.NaN
            
            return

        # Ensure that we are comparing the same feature!
        assert(exp_histogram.specs.long_field == 
               ctl_histogram.specs.long_field)
        assert(exp_histogram.hist_type == ctl_histogram.hist_type)
        assert(exp_histogram.motion_type == ctl_histogram.motion_type)
        assert(exp_histogram.data_type == ctl_histogram.data_type)

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
                if ((USE_OLD_CODE and self.is_exclusive) or
                    (~USE_OLD_CODE and self.ctl_histogram.num_valid_videos>1)):
                    self._z_score_experiment = -np.Inf
                else:
                    self.z_score_experiment = np.NaN
    
            elif np.isnan(self.ctl_histogram.mean):
                if ((USE_OLD_CODE and self.is_exclusive) or
                    (~USE_OLD_CODE and self.exp_histogram.num_valid_videos>1)):
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
    def p_t(self):
        """
        p_t
        
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
        p_t_all is a 726x1 matrix with values between 0 and 1.

        (From SciPy docs:)
        Calculates the T-test for the means of TWO INDEPENDENT samples 
        of scores.
        This is a two-sided test for the null hypothesis that 2
        independent samples have identical average (expected) values. 
        This test assumes that the populations have identical variances.

        """
        try:
            return self._p_t
        except AttributeError:
            # Scenario 1
            if self.is_exclusive:
                return self.fisher_p
                
            # Scenario 2
            else:
                _, self._p_t = \
                    sp.stats.ttest_ind(self.exp_histogram.valid_mean_per_video,
                                       self.ctl_histogram.valid_mean_per_video)
            
            return self._p_t
    
    @property
    def p_w(self):
        """
        p_w.
        
        Rules:

        1. If no valid means exist in one, but all exist in the other:
            Use Fisher's exact test.
            
        2. If at least one mean exists in both:
            Use the Wilcoxon signed-rank test.
        
        3. If no valid means exist in either:
            NaN.

        """
        try:
            return self._p_w
        except AttributeError:
            # Scenario 1
            if self.is_exclusive:
                return self.fisher_p

            # Scenario 2    
            elif not (self.exp_histogram.no_valid_videos or 
                      self.ctl_histogram.no_valid_videos):
                _, self._p_w = \
                    sp.stats.ranksums(self.exp_histogram.valid_mean_per_video, 
                                      self.ctl_histogram.valid_mean_per_video)
            # Scenario 3              
            else: 
                self._p_w = np.NaN

            return self._p_w
    #%%     
    @property    
    def specs(self):
        assert(self.exp_histogram.specs == self.ctl_histogram.specs)
        return self.exp_histogram.specs

    @property    
    def hist_type(self):
        assert(self.exp_histogram.hist_type == self.ctl_histogram.hist_type)
        return self.exp_histogram.specs

    @property    
    def motion_type(self):
        assert(self.exp_histogram.motion_type == 
               self.ctl_histogram.motion_type)
        self.motion_type = self.exp_histogram.motion_type

    @property    
    def data_type(self):
        assert(self.exp_histogram.data_type == self.ctl_histogram.data_type)
        self.data_type = self.exp_histogram.data_type

    #%%
    # Internal methods: not really intended for others to consume.     

    @property
    def fisher_p(self):
        """
        Return Fisher's exact method

        Notes
        ---------------
        Original Matlab version
        self.p_w = seg_worm.stats.helpers.fexact(*params)
        
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

