# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 22:36:32 2013

@author: Michael
"""

# DEBUG: make this a subclass of WormExperimentFile perhaps?
class WormFeatures:
  morphology = None
  locomotion = None
  posture = None
  normalized_worm = None
  
  def __init__(self, nw):
    self.normalized_worm = nw
    self.getMorphologyFeatures()
    self.getLocomotionFeatures()
    self.getPostureFeatures()

  def getMorphologyFeatures(self):
    pass
  
  def getLocomotionFeatures(self):
    pass
  
  def getPostureFeatures(self):
    pass
  