# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:38:39 2014

@author: RNEL
"""

import wormpy_example as we  
 
nw = we.example_nw()   

from wormpy.WormFeatures import WormPath as wp #Temporary for directly accessing features  

temp = wp(nw)
