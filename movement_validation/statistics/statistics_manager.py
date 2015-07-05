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

from .. import utils

#%%
class StatisticsManager(object):
    """

    Properties
    ---------------------------------------    
    stats
    p_worm
    q_worm

    Notes
    ---------------------------------------    
    Formerly seg_worm.stats.manager

    """

    def __init__(self, exp_hists, ctl_hists):
        """
        Initializes the Manager class.    

        Parameters
        ---------------------------------------    
        exp_hists
          An array of exp_hist entries
        ctl_hists
          An array of ctl_hist entries

        Notes
        ---------------------------------------    
        Formerly seg_worm.stats.manager.initObject

        """
        num_histograms = len(exp_hists.hists)

        # p_t Initialization
        #----------------------------------------------------------------------
        # @JimHokanson comments
        # :/ Sadly this needs to be done beforehand to be the same ...
        # It might be updated during object initialization ...
        #
        # TODO: This looks nicely vectorized, but it breaks the organization
        #       significantly ...
        #
        # How much of an impact do we get if we move this to being computed
        # for each object, instead of all of them at once?
        # Formerly: p_t_all =
        # mattest([exp_hists.mean_per_video]',[ctl_hists.mean_per_video]')
        # http://www.mathworks.com/help/bioinfo/ref/mattest.html
        # perform an unpaired t-test for differential expression with
        # a standard two-tailed and two-sample t-test on every gene in 
        # DataX and DataY and return a p-value for each gene.
        # PValues = mattest(DataX, DataY)
        # p_t_all is a 726x1 matrix with values between 0 and 1.

        # (From SciPy docs:)
        # Calculates the T-test for the means of TWO INDEPENDENT samples 
        # of scores.
        # This is a two-sided test for the null hypothesis that 2
        # independent samples have identical average (expected) values. 
        # This test assumes that the populations have identical variances.
        t_statistics, p_values = sp.stats.ttest_ind(exp_hists.valid_means_array,
                                                    ctl_hists.valid_means_array)
        # Removed this line: [stats_objs.p_t] = sl.struct.dealArray(p_t_all)

        # This is the main call to initialize each object
        #----------------------------------------------------------------------
        self.worm_statistics_objects = [None]*num_histograms
        for histogram_index in range(num_histograms):
            cur_worm_stats = WormStatistics(exp_hists.hists[histogram_index],
                                            ctl_hists.hists[histogram_index])
                #REMOVED:
               #, p_values) 
                # since len(p_values) == 10 but num_histograms == 726!
                                        

            self.worm_statistics_objects[histogram_index] = cur_worm_stats
    
        # Initialize properties that depend on the aggregate
        #----------------------------------------------------------------------
        self.p_t = np.array([x.p_t for x in self.worm_statistics_objects])
        self.p_w = np.array([x.p_w for x in self.worm_statistics_objects])
        
        # Filter the NaN entries 
        # I added this to get the code to work.  This puts the p's and q's 
        # out of order so it's not possible to know from what histogram they
        # are from after we do this, but it's necessary for compute_q_values
        # to work, since it needs a NaN-free array.
        # - @MichaelCurrie
        self.p_t = self.p_t[~np.isnan(self.p_t)]
        self.p_w = self.p_w[~np.isnan(self.p_w)]
    
        # TODO: DEBUG these four lines.
        #self.q_t = utils.compute_q_values(self.p_t)
        #self.q_w = utils.compute_q_values(self.p_w)

        #self.p_worm = min(self.p_w)
        #self.q_worm = min(self.q_w)


#%%
class WormStatistics(object):
    """
    WormStatistics class.

    Notes
    --------------------
    Formerly: seg_worm.stats

    Some of the statistics are aggegrate:
    - p_value
    - q_value

    List of exclusive features:
     properties

         #TODO: Move to object that both hist and stats display
         #
         #ALSO: We need two, one for experiment and one for controls
         #Definitions in: seg_worm.stats.hist
         field
         name
         short_name
         units
         feature_category
         hist_type
         motion_type
         data_type 
    
         #New properties
         #-------------------------------------------------------------------
         p_normal_experiment
         p_normal_control
         q_normal_experiment
         q_normal_control
         z_score_experiment
         #From documentation:
         #- no controls, this is empty
         #- absent in controls, but 2+ experiments, Inf
         #- present in 2+ controls, -Inf
         #Technically, this is incorrect
         #
    
         z_score_control    = 0 #By definition ...
    
         p_t  #Differential expression ...
         #    - function: mattest (bioinformatics toolbox)
         #    This doesn't seem like it is used ...
    
         p_w = NaN #NaN Default value, if all videos have a valid value 
         #then this is not set
    
         #NOTE: For the following, the corrections can be per strain or
         #across strains. I think the current implementation is per strain.
         #I'd like to go with the values used in the paper ...
    
         q_t
         #In the old code corrections were per strain or across all strains. 
    
         q_w
         #In the old code corrections were per strain or across all strains. 
         #Current implementation is per strain, not across strains ...
    
         p_significance
    
    
         #pTValue
         #pWValue
         #qTValue
         #qWValue


         #-------------------------------------------------------------------
  #        z_score   #not populated if no controls are provided ...
  #        mean      #mean of the mean hist values
  #        std       #std of the hist values
  #        n_samples ## of videos where the mean is not NaN
  #        p_normal = NaN  #probability of being a normal distribution
  #        #
  #        #    seg_worm.fex.swtest(data(i).dataMeans, 0.05, 0)
  #        q_normal  #
      """

    #%%
    def __init__(self, exp_histogram, ctl_histogram, USE_OLD_CODE=False):
        """
        I added p_t as a parameter because I think this is needed, but in the 
        code it seems not!  Then why were the p-values calculated in Manager 
        at all???  - @MichaelCurrie

        Parameters
        ---------------------
        exp_histogram: Histogram object
            "experiment"
        ctl_histogram: Histogram object
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
        """
        self.p_t = p_t
        del(p_t)
        """
        if exp_histogram is None or ctl_histogram is None:
            self.z_score_experiment = np.NaN
            self.p_normal_experiment = np.NaN
            self.p_normal_control = np.NaN
            self.p_w = np.NaN
            self.p_t = np.NaN
            return
            
        self.specs       = exp_histogram.specs
        self.hist_type   = exp_histogram.hist_type
        self.motion_type = exp_histogram.motion_type
        self.data_type   = exp_histogram.data_type

        # A flag indicating if either experiment has all valid means but
        # control has none, or vice versa.
        is_exclusive = (
            (exp_histogram.no_valid_means and ctl_histogram.all_means_valid) or
            (ctl_histogram.no_valid_means and exp_histogram.all_means_valid))
        
        # zscore
        #----------------------------------------------------------------------
        # This definition is slightly different than the old version, 
        # but matches the textual description
        # TODO: it does in code, but what about in published paper?

        # From Nature Methods 2013 Supplemental Description:
        #----------------------------------------------------------------------
        # Measurements exclusively found in the experimental group have 
        # a zScore of infinity and those found exclusively found in the 
        # control are -infinity.

        if np.isnan(exp_histogram.mean):
            if ((USE_OLD_CODE and is_exclusive) or
                (~USE_OLD_CODE and ctl_histogram.n_valid_measurements > 1)):
                self.z_score_experiment = -np.Inf
            else:
                self.z_score_experiment = np.NaN

        elif np.isnan(ctl_histogram.mean):
            if ((USE_OLD_CODE and is_exclusive) or
                (~USE_OLD_CODE and exp_histogram.n_valid_measurements > 1)):
                self.z_score_experiment = np.Inf
            else:
                self.z_score_experiment = np.NaN

        else:
            # This might need to be means_per_video, not the mean ...
            # - @JimHokanson
            self.z_score_experiment = (
                (exp_histogram.mean - ctl_histogram.mean) / ctl_histogram.std)

        self.p_normal_experiment = exp_histogram.p_normal
        self.p_normal_control    = ctl_histogram.p_normal

        # Rules are:
        # --------------------------------------
        # p_t
        #
        # - not in one, but all in the other - use fexact (Fisher's Exact)
        # - otherwise use mattest
        #
        # p_w
        # - not in one, but all in the other - use fexact
        # - partial in both - use Wilcoxon rank-sum test
        # - if in both, set to NaN

        if is_exclusive:
            # This is a literal translation of the code (I think)
            # I'm a bit confused by it ...  - @JimHokanson
            n_exp_videos = exp_histogram.n_videos
            n_videos = n_exp_videos + ctl_histogram.n_videos

            # This is a strange step, I don't know what it means...
            # Why this specific list of values, it's strange.
            # -@MichaelCurrie
            params = \
                np.array([n_exp_videos, n_videos, n_exp_videos, n_exp_videos])
            _, self.p_w = sp.stats.fisher_exact(params)
            # ORIGINAL MATLAB VERSION:
            #self.p_w = seg_worm.stats.helpers.fexact(*params)

            self.p_t = self.p_w

        # We need a few valid values from both ...
        elif not (exp_histogram.no_valid_means or 
                  ctl_histogram.no_valid_means):
            _, self.p_w = sp.stats.ranksums(exp_histogram.valid_means, 
                                            ctl_histogram.valid_means)
                           
            # I added this so p_t would be defined. - @MichaelCurrie
            self.p_t = np.NaN
        
        else: 
            # I added this so p_w and p_t would be defined. - @MichaelCurrie
            self.p_t = np.NaN
            self.p_w = np.NaN

        # NOTE: This code is for an individual object, the corrections
        #       are done in the manager which is aware of all objects ...

        # pWValues - these seem to be the real statistics used ...
        # - exclusive - fexact  seg_worm.stats.helpers.fexact
        # - ranksum