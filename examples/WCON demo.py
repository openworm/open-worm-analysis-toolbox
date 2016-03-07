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

import pickle

if __name__ == '__main__000':
    warnings.filterwarnings('error')
    print('RUNNING TEST ' + os.path.split(__file__)[1] + ':')

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
    
    # Shrink the data to make it more manageable
    w.data = w.data.loc[:200,:]

    start_time = mv.utils.timing_function()

    w.save_to_file('testfile2.wcon', pretty_print=True)

    w = wcon.WCONWorms.load_from_file('testfile2.wcon')
    with open('w.pickle', 'wb') as f:
        pickle.dump(w, f)

    print("Time elapsed: %.2f seconds" %
          (mv.utils.timing_function() - start_time))   


if __name__ == '__main__333':    


    print("NOW, let's LOAD this file!")
    start_time = mv.utils.timing_function()

    from multiprocessing import Process, Queue

    def load_file(q, l):
        print("loading file %s" % l)
        w = wcon.WCONWorms.load_from_file(l)
        print("done loading file %s" % l)
        q.put(w)
        q.put("SURPRISE! %s" % l)

    q = Queue()
    q.put('hello')
    p1 = Process(target=load_file, args=(q, 'testfile2.wcon'))
    #p2 = Process(target=load_file, args=(q, 'testfile3.wcon'))

    p1.start()
    #p2.start()



    print("Do we get to after the start()?")
    w1 = q.get()
    print(str(w1))
    print("Past the first get")
    w2 = q.get()
    print(str(w2))
    print("Past the second get")
    p1.join()
    print("Past the first p1.join()")
    #p2.join()
    #print("Past the SECOND p2.join()")
    
    #w1 = q.get()
    #print("We have w1!")
    #w2 = q.get()
    #print("We even have w1 and w2!")

if __name__ == '__main__':
    print("try LOAD this file")
    start_time = mv.utils.timing_function()
    w2 = wcon.WCONWorms.load_from_file('testfile4.wcon')
    
    print("Time elapsed: %.2f seconds" %
          (mv.utils.timing_function() - start_time))

