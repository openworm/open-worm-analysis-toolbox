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
from wormpy.WormFeatures import WormMorphology as wm

#temp = wp(nw)
temp = wm(nw)

#This file was created from the Matlab GUI
file_path = r'C:\Users\RNEL\Dropbox\worm_data\video\testing_with_GUI\results\mec-4 (u253) off food x_2010_04_21__17_19_20__1_features.mat'
from wormpy.WormFeatures import WormFeatures as wf

worm = wf.from_disk(file_path)

#print worm.morphology

print worm.morphology == temp

