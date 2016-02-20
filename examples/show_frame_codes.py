# -*- coding: utf-8 -*-
"""
A utility to compare frame_codes and segementation_status to understand
how they are related, specifically, if they are redundant.

We load the frame codes for an example video, example_video_norm_worm.mat

Then we load frame code and segementation status information

Then we look at which frames have distinct segementation status / frame codes

@author: @MichaelCurrie

"""
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sys.path.append('..')
    import open - worm - analysis - toolbox as mv
else:
    from open - worm - analysis - toolbox import open - worm - analysis - toolbox as mv


base_path = os.path.abspath(mv.user_config.EXAMPLE_DATA_PATH)
schafer_nw_file_path = os.path.join(base_path,
                                    "example_video_norm_worm.mat")
nw = mv.NormalizedWorm.from_schafer_file_factory(schafer_nw_file_path)

schafer_bw_file_path = os.path.join(base_path,
                                    "example_contour_and_skeleton_info.mat")
bw = mv.BasicWorm.from_schafer_file_factory(schafer_bw_file_path)


nw_frame_metadata = pd.DataFrame(
    {'Frame Code': nw.frame_code,
     'Segmentation Status': nw.segmentation_status,
     'is_nan_skeleton': np.isnan(nw.skeleton[0, 0, :]),
     'is_valid': bw.is_valid,
     'is_stage_movement': bw.is_stage_movement})

# obtain this computer's path to
# open-worm-analysis-toolbox\documentation\frame_codes.csv
package_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
frame_codes_path = os.path.join(package_path,
                                'documentation',
                                'frame_codes.csv')

# Load frame code information
frame_code_info = pd.read_csv(frame_codes_path, delimiter=';',
                              quotechar="'")
# Convert the 'Frame Codes' column, which is all int, to int.
frame_code_info = frame_code_info.convert_objects(convert_numeric=True)

segmentation_status_info = pd.DataFrame(
    {'Segmentation Status': ['s', 'f', 'm', 'd', 'n'],
     'Segmentation Description': ['Segmented',
                                  'Segmentation failed',
                                  'Stage movement',
                                  'Dropped frame',
                                  '??? There is reference '
                                  'in some old code to this']})
nw_frame_metadata = \
    nw_frame_metadata.merge(frame_code_info, how='left',
                            left_on='Frame Code',
                            right_on='Frame Code')

nw_frame_metadata = \
    nw_frame_metadata.merge(segmentation_status_info, how='left',
                            left_on='Segmentation Status',
                            right_on='Segmentation Status')

print(frame_code_info[['Frame Code', 'Code Name']].transpose().to_dict())


"""
CONCLUSIONS SO FAR:
- Only 7 unique rows
- is_stage_movement is 0 or 1 unpredictably when Segementation Status == 'd'
- is_stage_movement == 1 when Segementation Status == 'm'
- is_nan_skeleton == False iff Segementation Status == 's'
- Segementation Status == 's'
- other stuff too
"""


"""
# TODO: I want an incidence matrix!!

plt.scatter(frame_code, [ord(x) for x in segmentation_status])
plt.show()
"""
