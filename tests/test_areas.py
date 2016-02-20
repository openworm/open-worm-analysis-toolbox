# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 08:11:59 2015

@author: Avelino Javer

Validate that the "shoelace method" of calculating the worm area
is correct, by comparing it to the brute-force method of rasterizing the
worm and then counting the pixels bounded by the countour.

See https://github.com/openworm/open-worm-analysis-toolbox/issues/127

"""
import cv2
import sys
import os
import numpy as np
import matplotlib.pylab as plt

sys.path.append('..')
import open_worm_analysis_toolbox as mv


def compute_area_by_rasterization(contour):
    """
    Calculate the area by creating a binary image with a worm image

    Parameters
    --------------------
    contour: numpy array of shape (98,2,n)
        the contour, wrapped around either clockwise or counter-clockwise

    Returns
    --------------------
    A numpy array of shape (n,) giving the area in square microns at each
    frame

    """
    tot_cnt = contour.shape[2]
    area_filled = np.full(tot_cnt, np.nan)
    for ii in range(tot_cnt):
        worm_cnt = contour[:, :, ii]

        # Stop if the contour is not valid
        if np.any(np.isnan(worm_cnt)):
            continue

        # The minimum value of the contour will be the image upper-left corner
        corner = np.min(worm_cnt, axis=0)
        worm_cnt = worm_cnt - corner

        # After subracting the corner the maximum contour values will be the
        # other corner
        # (We have [::-1] because the coordinate system used by opencv is
        # different from numpy)
        im_size = np.max(worm_cnt, axis=0)[::-1]

        # TODO: it might be necessary to rescale the contour so the sampling
        #       is the same, in all cases
        im_dum = np.zeros(im_size)

        # Draw contours
        cv2.drawContours(im_dum, [worm_cnt.astype(np.int32)], 0, 1, -1)
        area_filled[ii] = np.sum(im_dum)

        # Example plot
        if ii == 4624:
            plt.figure()
            plt.imshow(im_dum)
            plt.plot(worm_cnt[:, 0], worm_cnt[:, 1], 'g')
            # plt.savefig('Filled_area_example.png')

    return area_filled


def test_areas():
    # Load from file some non-normalized contour data.
    base_path = os.path.abspath(mv.user_config.EXAMPLE_DATA_PATH)
    schafer_bw_file_path = os.path.join(
        base_path, "example_contour_and_skeleton_info.mat")
    bw = mv.BasicWorm.from_schafer_file_factory(schafer_bw_file_path)

    nw_calculated = mv.NormalizedWorm.from_BasicWorm_factory(bw)

    area_by_rasterization = compute_area_by_rasterization(
        nw_calculated.contour)

    #--------------------------------------------------------------------------
    # We perform comparison of three ways of calculating area:
    # nw.area == the old Schafer lab method
    # nw_calculated.area == the new Python method (uses the "Shoelace formula")
    # area_by_rasterization == a brute-force method that is slow but accurate

    # Compare (1) to (2) area calculations: uh oh, they are different!
    # but it turns out this is because the original method, which first
    # calculated the areas of small sub-regions and then added them up, may
    # have been somewhat off.  This is confirmed when later we compare
    # (2) to our brute-force method (3) and they are equal.
    """
    plt.title('Worm Area')
    area_loaded, = plt.plot(nw.area, 'r', label='Loaded')
    area_calculated, = plt.plot(nw_calculated.area, 'b', label='Calculated')
    area_ratio, = plt.plot(nw_calculated.area / nw.area, label='Ratio')
    area_diff, = plt.plot(nw_calculated.area - nw.area, label='Difference')
    plt.legend(handles=[area_loaded, area_calculated, area_ratio, area_diff])
    plt.xlabel("Frame #")
    plt.ylabel("Area")
    plt.show()

    print(np.nanmean(nw_calculated.area / nw.area))
    print(np.nanmin(nw_calculated.area / nw.area))
    print(np.nanmax(nw_calculated.area / nw.area))

    print(np.nanmean(nw_calculated.area - nw.area))
    print(np.nanmin(nw_calculated.area - nw.area))
    print(np.nanmax(nw_calculated.area - nw.area))
    """

    shoelace_equals_raster = mv.utils.correlation(nw_calculated.area,
                                                  area_by_rasterization,
                                                  'Areas calculation',
                                                  high_corr_value=0.997)
    print("Verify that shoelace (2) and raster (3) correlate closely: " +
          str(shoelace_equals_raster))
    assert(shoelace_equals_raster)

    # Bonus: Plot a comparison between (2) and (3): they should agree.
    plt.figure()
    plt.plot(nw_calculated.area, label='Shoelace method')
    plt.plot(area_by_rasterization, label='Rasterization method')
    plt.legend()
    plt.ylabel('Area')
    plt.xlabel('Frame Number')
    # plt.savefig('Shoelace_vs_Filled_area.png')


if __name__ == '__main__':
    print('RUNNING TEST ' + os.path.split(__file__)[1] + ':')
    start_time = mv.utils.timing_function()
    test_areas()
    print("Time elapsed: %.2fs" %
          (mv.utils.timing_function() - start_time))
