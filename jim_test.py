# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:38:39 2014

@author: RNEL
"""

import wormpy_example as we  
 
nw = we.example_nw()   

from wormpy.WormFeatures import WormPath as wp #Temporary for directly accessing features  

temp = wp(nw)


"""
file_path = r'F:\worm_data\segworm_data\features\247 JU438 on food R_2010_11_25__12_18_40___1___5_features.mat'

from wormpy.WormFeatures import WormFeatures as wf

wf.from_disk(file_path)
"""