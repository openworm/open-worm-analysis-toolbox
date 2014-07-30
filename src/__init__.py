# let's make it simple for users to do import src
# and then get all the classes defined by files in the src folder.
# so let's add some lines of the form:
# from [filename].py import [classname]

from src.SchaferExperimentFile import SchaferExperimentFile
from src.features.WormFeatures import WormFeatures
from src.features.feature_comparisons import fp_isequal
from src.features.feature_comparisons import corr_value_high
from src.NormalizedWorm import NormalizedWorm
from src.WormPlotter import WormPlotter
from src.WormPlotter import plot_frame_codes

__all__ = ['SchaferExperimentFile',
           'WormFeatures',
           'fp_isequal', 'corr_value_high',
           'NormalizedWorm',
           'WormPlotter',
           'plot_frame_codes']
