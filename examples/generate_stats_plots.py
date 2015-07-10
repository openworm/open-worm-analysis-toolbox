# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 12:47:16 2015

@author: mcurrie
"""
import os, sys
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('..')
import movement_validation as mv

# Use pandas to load the features specification
feature_spec_path = os.path.join('..', 'documentation', 'database schema',
                                 'Features Specifications.xlsx')

feature_spec = pd.ExcelFile(feature_spec_path).parse('FeatureSpecifications')




# Calculate the number of histograms multiplier?  No...

def prepare_plots():


    # PAGE 0: First an introductory page
    # Table of contents, description of experiment and control, 
    # Heatmap of phenotype
    # a heatmap of available features for each of the worms
    # 
    page0 = 'Table of Contents."
    # Change this to map to a specific feature!
    # maps (row,col) to sub-extended feature ID
    page1 = {(0,0): 'legend', (1,0): 5, (2,0): 26, (3,0): 22,
             (0,1): 7, (1,1): 6, (2,1): 8, (3,1): 24,
             (0,2):12, (1,2):17, (2,2):50, (3,2): 53,
             (0,2):14, (1,2):19, (2,2):51, (3,2): 54,
             (0,2):16, (1,2):21, (2,2):52, (3,2): 55}
    page2 = 'similar to page1'
    page3 = 'similar to page1'
    
    page4 to page27: movement features...
    page28 = bends
    page29 = coils
    page30=  more movement features: 29 to 37 inclusive
    page  = forward motion ???
    page  = paused motion ???
    page  = backward motion ???
    page  = features 38 to 47 inclusive
    48 foraging amplitude
    50,51,52  crawling amplitude
    49 foraging speed
    53,54,55
    omegas
    upsilon
    path range (forward/paused/backward)
    dwelling: four-grid; worm, had, midbody, tail.
    path curvature
    exactly 30 pages of path and dwelling charts

    METHODS TABLE OF CONTENTS
    
    
    """
    
    for feature in master_feature_list:
        title = feature['title']
        #three legends:
            #experiment / control
            #backward / forward / paused
            #q-values
        if feature['is_time_series']:
            # Show 2 rows and 3 columns of plots, with:
            rows = 2
            cols = 3
            """
            (0,0) is motion_type = 'all'
                experiment is brown
            (1,0) is motion_type = 'forward'
                experiment is purple
            (1,1) is motion_type = 'paused'
                experiment is green
            (1,2) is motion_type = 'backward'
                experiment is blue
            ax = plt.subplot2grid((rows, cols), (i // cols, i % cols))
            """
    
    # Then link to feature descriptions
    # perhaps this could be stored in the FeatureInfo table
    
