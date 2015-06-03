# -*- coding: utf-8 -*-
"""
A utility to compare frame_codes and segementation_status to understand
how they are related, specifically, if they are redundant.

@author: @MichaelCurrie

"""
import numpy as np
import sys, os
import csv
import matplotlib.pyplot as plt

sys.path.append('..')

from movement_validation import user_config, NormalizedWorm


def main():
    base_path = os.path.abspath(user_config.EXAMPLE_DATA_PATH)
    schafer_nw_file_path = os.path.join(base_path, 
                                     "example_video_norm_worm.mat")
    nw = NormalizedWorm.from_schafer_file_factory(schafer_nw_file_path)

    frame_code = nw.frame_code
    segmentation_status = nw.segmentation_status
    
    print(frame_code)
    print(segmentation_status)

    # load and compare frame codes and segmentation status!
    frame_codes_path = r'C:\Users\mcurrie\Desktop\GitHub' + \
                       r'\movement_validation\documentation\frame_codes.csv'
    with open(frame_codes_path, 'r') as frame_codes_file:
        reader = csv.DictReader(frame_codes_file, delimiter=';', quotechar="'")

        for row in reader:
            print(row)

    # TODO: I want an incidence matrix!!

    plt.scatter(frame_code, [ord(x) for x in segmentation_status])
    plt.show()


    frame_codes_path2 = r'C:\Users\mcurrie\Desktop\GitHub' + \
                       r'\movement_validation\documentation\frame_codes2.csv'

    frame_codes_path3 = r'C:\Users\mcurrie\Desktop\GitHub' + \
                       r'\movement_validation\documentation\frame_codes3.csv'

    frame_codes_path4 = r'C:\Users\mcurrie\Desktop\GitHub' + \
                       r'\movement_validation\documentation\frame_codes4.csv'


    frame_code.tofile(frame_codes_path2, sep=',', format='%d')
    segmentation_status.tofile(frame_codes_path3, sep=',', format='%s')
    np.isnan(nw.skeleton[0,0,:]).tofile(frame_codes_path4, sep=',', format='%s')
    #with open(frame_codes_path2, 'w', newline='') as frame_output_file:
    #    writer = csv.writer(frame_output_file, delimiter=',')
    #    writer.writerows(frame_code.tolist())
    #    writer.writerows(segmentation_status.tolist())
        

    # a string of length n, showing, for each frame of the video:
    # s = segmented
    # f = segmentation failed
    # m = stage movement
    # d = dropped frame
    # n??? - there is reference in some old code to this
    # after loading this we convert it to a numpy array.
    #                'segmentation_status',
                

if __name__ == '__main__':
    main()
