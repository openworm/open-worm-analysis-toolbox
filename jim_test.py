# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:38:39 2014

@author: RNEL
"""

import wormpy_example as we
 
nw = we.example_nw()   

#from wormpy.WormFeatures import WormPath as wp #Temporary for directly accessing features  
#temp = wp(nw)

from wormpy.WormFeatures import WormPosture as wp
temp = wp(nw)

"""
#TODO: Change this to the real comparison file ...
#file_path = r'F:\worm_data\segworm_data\features\247 JU438 on food R_2010_11_25__12_18_40___1___5_features.mat'

file_path = r'C:\Users\RNEL\Dropbox\worm_data\video\testing_with_GUI\results\mec-4 (u253) off food x_2010_04_21__17_19_20__1_features.mat'
from wormpy.WormFeatures import WormFeatures as wf

worm = wf.from_disk(file_path)

print worm.morphology

#worm.path == temp
"""
