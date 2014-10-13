# -*- coding: utf-8 -*-
"""
This should replicate the behaviour of the compute "new" histograms code from
https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Btesting/%2Bstats/t001_oldVsNewStats.m

We are just trying to achieve the creation of an in-memory object that contains
the statistics generated from comparing a set of 20 Feature .mat Files:
- 10 "Experiment" files and
- 10 "Control" files.

The relevant Matlab script code, from the above file, is simply:

    feature_files = sl.dir.rdir([root_path '\**\*.mat']);
    
    %We'll take the first 10 for the "experiment" and the next 10 for the
    %"control"
    expt_files = {feature_files(1:10).name};
    ctl_files  = {feature_files(11:20).name};
    
    % STEP 2: Compute "new" Histograms - (just save in memory, don't 
    %                                    bother writing to disk)
    %----------------------------------------------------------------------    
    hist_man_exp = seg_worm.stats.hist.manager(expt_files);
    hist_man_ctl = seg_worm.stats.hist.manager(ctl_files);
    stats_manager = seg_worm.stats.manager(hist_man_exp,hist_man_ctl);
"""


import sys, os

# We must add .. to the path so that we can perform the 
# import of movement_validation while running this as 
# a top-level script (i.e. with __name__ = '__main__')
sys.path.append('..') 
import movement_validation as mv


def main():
    # TODO
    pass






if __name__ == '__main__':
    main()