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

import wcon
import pandas as pd
import numpy as np

import pickle

class BasicWorm2(wcon.WCONWorms):
    pass

if __name__ == '__main__':
    warnings.filterwarnings('error')
    print('RUNNING TEST ' + os.path.split(__file__)[1] + ':')
    start_time = mv.utils.timing_function()

    base_path = os.path.abspath(
        mv.user_config.EXAMPLE_DATA_PATH)
    schafer_bw_file_path = \
        os.path.join(base_path,
                     "example_contour_and_skeleton_info.mat")
    bw = mv.BasicWorm.from_schafer_file_factory(
        schafer_bw_file_path)

    #with open('testfile.pickle', 'wb') as f:
    #    pickle.dump(bw, f)

    w = wcon.WCONWorms()
    w.units = {"t":"s","x":"mm","y":"mm"}
    for key in w.units:
        w.units[key] = wcon.MeasurementUnit.create(w.units[key])
    w.units['aspect_size'] = wcon.MeasurementUnit.create('')
    
    w.metadata = {"lab":{"location":"MRC Laboratory of Molecular Biology, "
                                    "Hills Road, Cambridge, CB2 0QH, "
                                    "United Kingdom","name":"Schafer Lab"}}

    num_frames = len(bw.h_dorsal_contour)
    worm_data_segment = {'id':0, 't':list(range(num_frames))}

    h_dorsal_contour = np.array(bw.h_dorsal_contour)
    empty_xy = np.empty((2,0))

    # Replace None entries with empty articulations for both x and y dimensions
    for frame_index in range(num_frames):
        if h_dorsal_contour[frame_index] is None:
            h_dorsal_contour[frame_index] = empty_xy

    #none_fixer = np.vectorize(lambda a: empty_xy if (a is None) else a)
    #h_dorsal_contour = none_fixer(h_dorsal_contour)
    
    for dimension_index, dimension in enumerate(['x', 'y']):
        worm_data_segment[dimension] = [list(h_dorsal_contour[findex][0]) 
                                        for findex in range(num_frames)]

    w.data = wcon.wcon_data.parse_data(worm_data_segment)

    w.save_to_file('testfile2.wcon', pretty_print=True)

    print("Time elapsed: %.2f seconds" %
          (mv.utils.timing_function() - start_time))   
    
    """
    with open('testfile.wcon', 'w') as f:
        f.write('{\n    ,\n    {"data":[{')
        num_timeframes = len(bw.h_dorsal_contour)
        f.write('"t":%s,' % str(list(range(num_timeframes))))
        for dimension_index, dimension in enumerate(['x', 'y']):
            f.write('\n"x":[')
            for frame_index in range(1): #num_timeframes):
                    f.write('%s,' % repr(list(bw.h_dorsal_contour[frame_index][dimension_index])))
            f.write(']')
        f.write('}]}}')
    """

    #for bw.h_skeleton

    #nw = mv.NormalizedWorm.from_BasicWorm_factory(bw)

    #wp = mv.NormalizedWormPlottable(nw, interactive=False)

    # wp.show()

    # TODO:
    # bw.save_to_wcon('test.wcon')
    #bw2 = mv.BasicWorm.load_from_wcon('test.wcon')

    #assert(bw == bw2)


def main2():
    base_path = os.path.abspath(
        mv.user_config.EXAMPLE_DATA_PATH)

    JSON_path = os.path.join(base_path, 'test.JSON')

    b = mv.BasicWorm()
    #b.contour[0] = 100.2
    #b.metadata['vulva'] = 'CCW'
    b.save_to_JSON(JSON_path)

    c = mv.BasicWorm()
    c.load_from_JSON(JSON_path)
    print(c.contour)

    # dat.save_to_JSON(JSON_path)
