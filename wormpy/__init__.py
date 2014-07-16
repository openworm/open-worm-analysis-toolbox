# let's make it simple for users to do import wormpy
# and then get all the classes defined by files in the wormpy folder.
# so let's add some lines of the form:
# from [filename].py import [classname]

from wormpy.SchaferExperimentFile import SchaferExperimentFile
from wormpy.WormFeatures import WormFeatures
from wormpy.feature_comparisons import fp_isequal
from wormpy.feature_comparisons import corr_value_high
from wormpy.NormalizedWorm import NormalizedWorm
from wormpy.WormPlotter import WormPlotter
from wormpy.WormPlotter import plot_frame_codes

__all__ = ['SchaferExperimentFile',
           'WormFeatures',
           'fp_isequal', 'corr_value_high',
           'NormalizedWorm',
           'WormPlotter',
           'plot_frame_codes']
