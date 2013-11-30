# let's make it simple for users to do import wormpy
# and then get all the classes defined by files in the wormpy folder.
# so let's add some lines of the form:
# from [filename].py import [classname]

from wormpy.WormExperimentFile import WormExperimentFile
from wormpy.WormFeatures import WormFeatures
from wormpy.NormalizedWorm import NormalizedWorm

__all__ = ['WormExperimentFile', 'WormFeatures', 'NormalizedWorm']