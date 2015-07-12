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

    # Assigned numbers are the sub-extended feature ID.
    # any features that have feature_type = 'movement' use the standard
    # 6-figure movement type plot.

    # PAGE 0: First an introductory page
    # Table of contents, description of experiment and control, 
    # Heatmap of phenotype
    # a heatmap of available features for each of the worms
    # 
    page = [None]*94
    page[0] = 'Table of Contents.'

    # maps (row,col) to sub-extended feature ID
    page[1] = {(0,0): '', (0,1):  7, (0,2): 12, (0,3): 14, (0,4): 16,
               (1,0):  5, (1,1):  6, (1,2): 17, (1,3): 19, (0,4): 21,
               (2,0): 26, (2,1):  8, (2,2): 50, (2,3): 51, (0,4): 52,
               (3,0): 22, (3,1): 24, (3,2): 53, (3,3): 54, (0,4): 55}
    page[1][(0,0)] = 'legend'

    page[2] = {(0,0): 48, (0,1): 49, (0,2): 39, (0,3): 40, (0,4): 41,
               (1,0): 32, (1,1): 33, (1,2): 44, (1,3): 45, (0,4): 46,
               (2,0): 34, (2,1): 35, (2,2): 63, (2,3): 64, (0,4): 65,
               (3,0): 36, (3,1): 37, (3,2): 68, (3,3): 69, (0,4): 70}

    page[3] = {(0,0): 56, (0,1): 28, (0,2): 58, (0,3): 59, (0,4): 60,
               (1,0):  2, (1,1): 73, (1,2): 74, (1,3): 75, (0,4): 76,
               (2,0):  3, (2,1): 80, (2,2): 81, (2,3): 82, (0,4): 83,
               (3,0):  4, (3,1): 87, (3,2): 88, (3,3): 89, (0,4): 90}

    # THE SIX-FIGURE SECTIOIN    
    
    for i in range(4,27):
        # Movement features
        page[i] = i + 1

    page[27] = 'bend count'   #(features[58:63])
    page[28] = 'coil time'  #(features[58:63])

    for i in range(29,38):
        # More movement features
        page[i] = i

    page[38] = 'locomotion.motion_events.forward'
    page[39] = 'locomotion.motion_events.paused'
    page[40] = 'locomotion.motion_events.backward'

    for i in range(41,52):
        # More movement features
        page[i] = i - 3

    page[52] = 50 # Crawling amplitude
    page[53] = 51
    page[54] = 52

    page[55] = 49 # Foraging speed

    page[56] = 53 # Crawling frequency
    page[57] = 54
    page[58] = 55

    page[59] = 'Omega turns (just four plots)'
    page[60] = 'Upsilon turns (just four plots)'
    page[61] = 56   # ANOTHER 6-figure plot (Path range)
    page[62] = [1,2,3,4]  # Worm dwelling four-grid; worm, had, midbody, tail.
    page[63] = 67   # ANOTHER 6-figure plot (Path curvature)

    # PATH PLOTS, all annotated with omegas and coils:
    # i.e. exactly 30 pages of path and dwelling charts

    # 10 pages for Midbody/Head/Tail colors
    page[64] = 'two charts, 24 experiment, 24 control, blended'
    page[65] = ('two charts, 24 experiment, 24 control, split out into '
               '24 little plots, 5 columns, 6 rows')
    page[66] = '6 plots of experiment worms 0-6'
    page[67] = '6 plots of experiment worms 6-12'
    page[68] = '6 plots of experiment worms 12-18'
    page[69] = '6 plots of experiment worms 18-24'
    page[70] = '6 plots of control worms 0-6'
    page[71] = '6 plots of control worms 6-12'
    page[72] = '6 plots of control worms 12-18'
    page[73] = '6 plots of control worms 18-24'

    page[74:84] = 'SAME 10 pages AGAIN, BUT NOW FOR MIDBODY SPEED'
    page[84:94] = 'SAME 10 pages AGAIN, but now for FORAGING AMPLITUDE'

    # METHODS TABLE OF CONTENTS
    
    
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

            (0,0) is motion_type = 'all'
                experiment is brown
            (1,0) is motion_type = 'forward'
                experiment is purple
            (1,1) is motion_type = 'paused'
                experiment is green
            (1,2) is motion_type = 'backward'
                experiment is blue
            ax = plt.subplot2grid((rows, cols), (i // cols, i % cols))
    
    # Then link to feature descriptions
    # perhaps this could be stored in the FeatureInfo table
    
    """
    pass    
    