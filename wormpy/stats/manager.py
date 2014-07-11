# -*- coding: utf-8 -*-
"""
manager.py

@authors: @JimHokanson, @MichaelCurrie

Classes
---------------------------------------    
Manager
Stats

A translation of Matlab code written by Jim Hokanson,
in the SegwormMatlabClasses GitHub repo.  Original code paths:

                                                              # of lines
SegwormMatlabClasses / +seg_worm / @stats                        231 
SegwormMatlabClasses / +seg_worm / +stats / @manager              61

SegwormMatlabClasses / +seg_worm / +stats / @hist_specs           50 (unknown how used - @JimHokanson)
SegwormMatlabClasses / +seg_worm / +stats / @hist                339
SegwormMatlabClasses / +seg_worm / +stats / +helpers / fexact.m  405
SegwormMatlabClasses / +seg_worm / +stats / +helpers / swtest.m  260
SegwormMatlabClasses / +seg_worm / +stats / +hist / @manager     410
SegwormMatlabClasses / +seg_worm / +stats / +hist / docs         (several CSV files)
SegwormMatlabClasses / +seg_worm / +stats / event_specs.m        143
SegwormMatlabClasses / +seg_worm / +stats / movement_specs.m      81
SegwormMatlabClasses / +seg_worm / +stats / simple_specs.m        45
SegwormMatlabClasses / +seg_worm / +stats / specs.m               70

                                                          TOTAL 2095 lines + several CSV files
                                                               unclear so far if the CSV files
                                                               are being loaded and used by the
                                                               code or if they are just docs

DEBUG: understand why there is an @manager class in two different scopes!
there is one +stats and one in +stats/+hist !

"""
from scipy import stats 

# The original Matlab code was classdef manager < handle
# I don't know the significance of "< handle"
# - @MichaelCurrie
class Manager(object):
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

    Parameter
    ---------------------------------------    
    exp_hists
      An array of exp_hist entries
    ctl_hists
      An array of ctl_hist entries

    Notes
    ---------------------------------------    
    Formerly seg_worm.stats.manager.initObject

    """
    self.stats = None
    self.p_worm = None
    self.q_worm = None
    
    n_objs = len(exp_hists)
    
    #p_t Initialization
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
    # Formerly: p_t_all = mattest([exp_hists.mean_per_video]',[ctl_hists.mean_per_video]')
    t_statistics, p_values = stats.ttest_ind(exp_hists.mean_per_video, 
                                             ctl_hists.mean_per_video)
    # Removed this line: [stats_objs.p_t] = sl.struct.dealArray(p_t_all)
    
    #This is the main call to initialize each object
    #--------------------------------------------------------------------------
    self.stats = []
    for iObj in range(n_objs):
      #seg_worm.stats.initObject
      self.stats.append(Stats(exp_hists[iObj],
                              ctl_hists[iObj], 
                              p_values[iObj]))

    
    #Followup with properties that depend on the aggregate
    #--------------------------------------------------------------------------
    [~,q_t_all]      = mafdr([stats_objs.p_t])
    [stats_objs.q_t] = sl.struct.dealArray(q_t_all)
    
    [~,q_w_all]      = mafdr([stats_objs.p_w])
    [stats_objs.q_w] = sl.struct.dealArray(q_w_all)
    
    self.p_worm = min([stats_objs.p_w])
    self.q_worm = min([stats_objs.q_w])
    
  

class Stats(object):
  """

  Class:
  seg_worm.stats
  
  Important Files
  ---------------------------------------------------------
  seg_worm.w.stats.worm2histogram         TODO: Move this ...
  seg_worm.w.stats.addWormHistograms
  seg_worm.w.stats.worm2StatsInfo
  seg_worm.w.stats.wormStats2Matrix
  
  
  QUESTIONS:
  ================================================================
  1)
  
  Procesing Steps:
  ================================================================
  1) Create histogram files for "experiment" and "control" files
      - seg_worm.stats.hist
  2)
  
  Main initialization code in:
  --------------------------------------------------------------
   
  See Also:
  seg_worm.stats.manager.initObject
  seg_worm.stats.initObject
  seg_worm.stats.hist
  
  Some of the statistics are aggegrate:
  p_value
  q_value
  list of exclusive features
 
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
    

  def __init__(self, exp_hist, ctl_hist, p_t):
    """
    I added p_t as a parameter because I think this is needed, but in the 
    code it seems not!  Then why were the p-values calculated in Manager 
    at all???  - @MichaelCurrie
    
    
    Formerly seg_worm.stats.initObject(obj,exp_hist,ctl_hist)

    worm2StatsInfo  Compute worm statistics information and save it to a file.
    
       seg_worm.stats.initObject
    
       See Also:
       seg_worm.stats.helpers.swtest
    

    """    
    self.p_t = p_t
    del(p_t)
    
    ALPHA = 0.05 #Not really used since we don't examine H, just p
    TAIL  = 0
    
    # NOTE: This should be passed in instead of being changed here ...
    USE_OLD_CODE = False
    
    # TODO: I'm not a big fan of all of this copying. I'd rather just copy an
    # object (use name: seg_worm.stats.description?)  - @JimHokanson
    self.field              = exp_hist.field
    self.name               = exp_hist.name 
    self.short_name         = exp_hist.short_name
    self.units              = exp_hist.units
    self.feature_category   = exp_hist.feature_category
    self.hist_type          = exp_hist.hist_type
    self.motion_type        = exp_hist.motion_type
    self.data_type          = exp_hist.data_type
    
    
    is_exclusive = (exp_hist.none_valid && ctl_hist.all_valid) \
                || (exp_hist.all_valid  && ctl_hist.none_valid)
    
    #zscore
    #--------------------------------------------------------------------------
    #This definition is slightly different than the old version, but matches
    #the textual description (in code, what about in published paper?)
    
    #From Nature Methods 2013 Supplemental Description:
    #--------------------------------------------------------------------------
    # Measurements exclusively found in the experimental group have a zScore of
    # infinity and those found exclusively found in the control are -infinity.
    
    if np.isnan(exp_hist.mean):
      if (USE_OLD_CODE && is_exclusive) || 
        (~USE_OLD_CODE && ctl_hist.n_valid_measurements > 1):
        self.z_score_experiment = -np.Inf
      else:
        self.z_score_experiment = np.NaN
        
    elif np.isnan(ctl_hist.mean):
      if (USE_OLD_CODE && is_exclusive) || 
        (~USE_OLD_CODE && exp_hist.n_valid_measurements > 1):
        self.z_score_experiment = np.Inf
      else:
        self.z_score_experiment = np.NaN
        
    else:
      # This might need to be means_per_video, not the mean ...
      # - @JimHokanson
      self.z_score_experiment = (exp_hist.mean - ctl_hist.mean) / ctl_hist.std

    
    # TODO: Move this to the histogram, not here!  These are properties of how
    # normal the distributions of the histograms are, and have nothing to do
    # with the comparative statistics between the two groups
    # - @JimHokanson
    #------------------------------------------------------------------------
    hist_objs = {'p_normal_experiment': exp_hist,
                 'p_normal_control': ctl_hist}    
        
    for cur_p_field, cur_hist_obj in hist_objs:
      if cur_hist_obj.n_valid_measurements < 3:
        self.(cur_p_field) = np.NaN
      else:
        self.(cur_p_field) = seg_worm.stats.helpers.swtest(
                                              cur_hist_obj.mean_per_video,
                                              ALPHA,
                                              TAIL)

    # Rules are:
    # --------------------------------------
    # p_t
    #
    # - not in one, but all in the other - use fexact (Fishers Exact)
    # - otherwise use mattest
    #
    # p_w
    # - not in one, but all in the other - use fexact
    # - partial in both - use Wilcoxon rank-sum test
    # - if in both, set to NaN
    
    if is_exclusive:
      # This is a literal translation of the code (I think)
      # I'm a bit confused by it ...  - @JimHokanson
      n_expt  = exp_hist.n_videos
      n_total = n_expt + ctl_hist.n_videos
      self.p_w = seg_worm.stats.helpers.fexact(n_expt, n_total, n_expt, n_expt)
      self.p_t = self.p_w
    elif ~(exp_hist.none_valid || ctl_hist.none_valid):
      # We need a few valid values from both ...
      self.p_w = ranksum(exp_hist.valid_means, ctl_hist.valid_means)
    
    # NOTE: This code is for an individual object, the corrections 
    #       are done in the manager which is aware of all objects ...
    
    # pWValues - these seem to be the real statistics used ...
    # - exclusive - fexact  seg_worm.stats.helpers.fexact
    # - ranksum
  
