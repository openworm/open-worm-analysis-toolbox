# -*- coding: utf-8 -*-
""" wormExample.py: illustrates the use of the wormpy.experiment_file module
    @author: mcurrie
    
"""
from wormpy import experiment_file
import os

# let's assume the worm file is in the same directory as this source file
worm_file_path1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               "unc-8 (rev) on food " +
                               "R_2010_03_19__09_14_57___2___2_features.mat")
  #u'F:/worm_data/segworm_data/features/798 JU258 on food R_2010_11_25__16_34_17___1___9_features.mat'

w = experiment_file.WormVideoFile(worm_file_path1)

w.create_animation()
w.save_to_mp4("worm_animation.mp4")
