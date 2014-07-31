# -*- coding: utf-8 -*-
"""
  user_config.py: a module to specify installation-specific settings for
                  the movement_validation repository
  
  This file is in the .gitignore file for the movement_validation repo, so
  you can put configuration settings here that apply to just your computer.
  
  @authors: @JimHokanson, @MichaelCurrie
  
"""


DROPBOX_PATH = r"C:\Users\RNEL\Dropbox"

# An unc-8 (strong coiler) mutant worm
WORM_FILE_PATH = "worm_data\\example_feature_files\\" + \
                 "unc-8 (rev) on food " + \
                 "R_2010_03_19__09_14_57___2___2_features.mat"

NORMALIZED_WORM_PATH = "worm_data\\video\\testing_with_GUI\\.data\\" + \
                       "mec-4 (u253) off food " + \
                       "x_2010_04_21__17_19_20__1_seg\\normalized"

EIGENWORM_PATH = "worm_data"

# There is a step that is very slow that can be disabled
# if you are just debugging other parts of the code.
# For production, this should be True.
PERFORM_SLOW_ECCENTRICITY_CALC = False