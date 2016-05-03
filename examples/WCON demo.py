# -*- coding: utf-8 -*-
"""
Demonstrates loading and saving to a WCON file

"""
import sys
import os
import warnings

# We must add .. to the path so that we can perform the
# import of open_worm_analysis_toolbox while running this as
# a top-level script (i.e. with __name__ = '__main__')
sys.path.append('..')
import open_worm_analysis_toolbox as mv


sys.path.append('../../tracker-commons/src/Python')
import wcon
import pandas as pd
import numpy as np
from collections import OrderedDict
import pickle


def schafer_to_WCON(MAT_path):
    """
    Load a Schafer .mat skeleton file and return a WCONWorms
    object.

    """
    bw = mv.BasicWorm.from_schafer_file_factory(MAT_path)

    w = wcon.WCONWorms()
    w.units = {"t":"0.04*s","x":"um","y":"um"}
    for key in w.units:
        w.units[key] = wcon.MeasurementUnit.create(w.units[key])
    w.units['aspect_size'] = wcon.MeasurementUnit.create('')
    
    w.metadata = {"lab":{"location":"MRC Laboratory of Molecular Biology, "
                                    "Hills Road, Cambridge, CB2 0QH, "
                                    "United Kingdom","name":"Schafer Lab"}}

    num_frames = len(bw.h_dorsal_contour)
    worm_data_segment = {'id':0, 't':list(range(num_frames))}

    skel_prefixes = {'dorsal_contour': 'dc_',
                     'ventral_contour': 'vc_',
                     'skeleton': ''}

    skel_lists = {'dorsal_contour': bw.h_dorsal_contour,
                  'ventral_contour': bw.h_ventral_contour,
                  'skeleton': bw.h_dorsal_contour}  # DEBUG

    for k in skel_lists.keys():
        skel = np.array(skel_lists[k])

        # Replace None entries with empty articulations for both x and y dimensions
        empty_xy = np.empty((2,0))
        for frame_index in range(num_frames):
            if skel[frame_index] is None:
                skel[frame_index] = empty_xy

        for dimension_index, dimension in enumerate(['x', 'y']):
            worm_data_segment[skel_prefixes[k] + dimension] = \
                [list(skel[findex][dimension_index])
                 for findex in range(num_frames)]

    w._data = wcon.wcon_data.parse_data(worm_data_segment)

    return w


if __name__ == '__main__':
    warnings.filterwarnings('error')
    print('RUNNING TEST ' + os.path.split(__file__)[1] + ':')

    base_path = os.path.abspath(
        mv.user_config.EXAMPLE_DATA_PATH)
    schafer_bw_file_path = \
        os.path.join(base_path,
                     "example_contour_and_skeleton_info.mat")
  
    w = schafer_to_WCON(schafer_bw_file_path)
 
    start_time = mv.utils.timing_function()

    w.save_to_file('testfile_new.wcon.zip',
                   pretty_print=True, compress_file=True)

    print("Time elapsed: %.2f seconds" %
          (mv.utils.timing_function() - start_time))

