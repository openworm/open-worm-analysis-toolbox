# -*- coding: utf-8 -*-
"""
Posture features

"""


from __future__ import division

import scipy.ndimage.filters as filters
import numpy as np
import warnings
import os, inspect, h5py

import cv2

from .. import config, utils
from . import events

class Skeleton(object):
    def __init__(self, features_ref):
        
        nw  = features_ref.nw         
        
        self.x = nw.skeleton_x
        self.y = nw.skeleton_y
        
    @classmethod
    def from_disk(cls, skeleton_ref):
        self = cls.__new__(cls)      
        
        x_temp = utils._extract_time_from_disk(skeleton_ref, 'x',
                                               is_matrix=True)
        y_temp = utils._extract_time_from_disk(skeleton_ref, 'y',
                                               is_matrix=True)
        self.x = x_temp.transpose()
        self.y = y_temp.transpose()
        
        return self
     
    def __repr__(self):
        return utils.print_object(self) 
    
    
    def __eq__(self, other):
        eq_skeleton_x = utils.correlation(np.ravel(self.x), 
                                          np.ravel(other.x), 
                                          'posture.skeleton.x')
        eq_skeleton_y = utils.correlation(np.ravel(self.y), 
                                          np.ravel(other.y), 
                                          'posture.skeleton.y')

        return eq_skeleton_x and eq_skeleton_y

class Bends(object):

    """

    Posture Bends    
    
    Attributes
    ----------
    head : BendSection
    midbody : BendSection
    tail : BendSection
    hips : BendSection
    neck : BendSection
    
    """
    def __init__(self, features_ref):
        nw  = features_ref.nw

        p = nw.get_partition_subset('normal')

        for partition_key in p.keys():
            # Retrieve the part of the worm we are currently looking at:
            bend_angles = nw.get_partition(partition_key, 'angles')

            # shape = (n):

            # Suppress RuntimeWarning: Mean of empty slice for those frames 
            # that are ALL NaN.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)                

                # Throws warning on taking the mean of an empty slice
                temp_mean = np.nanmean(a=bend_angles, axis=0)
                
                # Throws warning when degrees of freedom <= 0 for slice
                temp_std = np.nanstd(a=bend_angles, axis=0)

                # Sign the standard deviation (to provide the bend's 
                # dorsal/ventral orientation)
                temp_std[temp_mean < 0] *= -1


            setattr(self, partition_key, 
                    BendSection(temp_mean, temp_std, partition_key))

    @classmethod
    def create(self,features_ref):
        options = features_ref.options
        if options.should_compute_feature('locomotion.bends',features_ref):
            return Bends(features_ref)
        else:
            return None

    def __repr__(self):
        return utils.print_object(self)


    def __eq__(self, other):
        pass
        """
        same_values = True
        for partition_key in self.posture_bend_keys:
            same_values = same_values and \
                (getattr(self,partition_key) == getattr(other,partition_key))
            
        return same_values    
	   """

    @classmethod
    def from_disk(cls, saved_bend_data):

        self = cls.__new__(cls)

        for partition_key in saved_bend_data.keys():
            setattr(self, partition_key, 
                    BendSection.from_disk(saved_bend_data[partition_key], 
                                          partition_key))

        return self


class BendSection(object):

    """
    Attributes
    ----------
    
    See Also
    --------
    Bends
    
    """
    def __init__(self, mean, std_dev, name):
        self.mean = mean
        self.std_dev = std_dev
        self.name = name

    @classmethod
    def from_disk(cls, saved_bend_data, name):

        self = cls.__new__(cls)

        self.mean = utils._extract_time_from_disk(saved_bend_data, 'mean')

        try:        
            self.std_dev = \
                utils._extract_time_from_disk(saved_bend_data, 'std_dev')
        except KeyError:
            self.std_dev = \
                utils._extract_time_from_disk(saved_bend_data, 'stdDev')

        self.name = name

        return self

    def __repr__(self):
        return utils.print_object(self)

    
    def __eq__(self, other):
        #TODO: Why is the head.std_dev so low???
        #Are we not mimicing some old error properly???
        return utils.correlation(self.mean,other.mean,
                                  'posture.bends.' + self.name + '.mean',
                                  high_corr_value=0.99) and \
               utils.correlation(self.std_dev,other.std_dev,
                                  'posture.bends.' + self.name + '.std_dev')


def get_eccentricity_and_orientation(features_ref):
    """
     Get the eccentricity and orientation of a contour using the moments

     http://en.wikipedia.org/wiki/Image_moment

     Calculated by opencv:moments (http://docs.opencv.org/modules/imgproc/
     doc/structural_analysis_and_shape_descriptors.html).
    """

    """
    OLD CODE FROM JIM DESCRIPTION:    

   get_eccentricity   
 
    [eccentricity, orientation] = \
        seg_worm.utils.posture.getEccentricity(xOutline, yOutline, gridSize)

    Given x and y coordinates of the outline of a region of interest, fill
    the outline with a grid of evenly spaced points and use these points in
    a center of mass calculation to calculate the eccentricity and
    orientation of the equivalent ellipse.

    Placing points in the contour is a well known computer science problem
    known as the Point-in-Polygon problem.

    http://en.wikipedia.org/wiki/Point_in_polygon

    This function became a lot more complicated in an attempt to make it 
    go much faster. The complication comes from the simplication that can
    be made when the worm doesn't bend back on itself at all.


    OldName: getEccentricity.m


    Inputs:
    =======================================================================
    xOutline : [96 x num_frames] The x coordinates of the contour. In 
                particular the contour
                starts at the head and goes to the tail and then back to
                the head (although no points are redundant)
    yOutline : [96 x num_frames]  The y coordinates of the contour "  "

    N_ECCENTRICITY (a constant from config.py):
               (scalar) The # of points to place in the long dimension. 
               More points gives a more accurate estimate of the ellipse 
               but increases the calculation time.

    Outputs: a namedtuple containing:
    =======================================================================
    eccentricity - [1 x num_frames]
        The eccentricity of the equivalent ellipse
    orientation  - [1 x num_frames]
        The orientation angle of the equivalent ellipse

    Nature Methods Description
    =======================================================================
    Eccentricity. 
    ------------------
    The eccentricity of the worm’s posture is measured using
    the eccentricity of an equivalent ellipse to the worm’s filled contour.
    The orientation of the major axis for the equivalent ellipse is used in
    computing the amplitude, wavelength, and track length (described
    below).

    Status
    =======================================================================
    The code below is finished although I want to break it up into smaller
    functions. I also need to submit a bug report for the inpoly FEX code.

    Translation of: SegwormMatlabClasses / 
    +seg_worm / +utils / +posture / getEccentricity.m
    """
    
    features_ref.timer.tic()
        
    contour = features_ref.nw.contour_without_redundant_points
    
    # OpenCV does not like float64, this actually make sense for image 
    # data where we do not require a large precition in the decimal part. 
    # This could save quite a lot of space
    contour = contour.astype(np.float32)
    tot = contour.shape[-1]

    
    eccentricity = np.full(tot, np.nan);
    orientation = np.full(tot, np.nan);
    for ii in range(tot):
        worm_cnt = contour[:,:,ii];
        if ~np.any(np.isnan(worm_cnt)):
            moments  = cv2.moments(worm_cnt)
        
            a1 = (moments['mu20'] + moments['mu02']) / 2
            a2 = np.sqrt(4*moments['mu11']**2 + 
                         (moments['mu20'] - moments['mu02'])**2) / 2
        
            minor_axis = a1 - a2
            major_axis = a1 + a2

            eccentricity[ii] = np.sqrt(1 - minor_axis/major_axis)
            orientation[ii] = \
                np.arctan2(2*moments['mu11'], 
                           (moments['mu20'] - moments['mu02'])) / 2
            # Convert from radians to degrees
            orientation[ii] *= 180 / np.pi
            
    features_ref.timer.toc('posture.eccentricity_and_orientation')
    
    return (eccentricity, orientation)


class AmplitudeAndWavelength(object):
    
    """
    Attributes
    ----------
    amplitude_max
    amplitude_ratio
    primary_wavelength
    secondary_wavelength
    track_length
    """
    def __init__(self,theta_d, features_ref):

        """
        Calculates amplitude of rotated worm (relies on orientation
        aka theta_d)
        
        Parameters
        ----------
        theta_d
        sx
        sy
        worm_lengths
        
        """
        #TODO: I think this would be better as a class
        
        
        if not features_ref.options.should_compute_feature(
                                'locomotion.amplitude_and_wavelength',
                                features_ref):
            self.amplitude_max = None
            self.amplitude_ratio = None
            self.primary_wavelength = None
            self.secondary_wavelength = None
            self.track_length = None

            return
        
        
        timer = features_ref.timer;
        timer.tic()
        
        options = features_ref.options    
        
        #TODO: Check if we should even compute this code    
        
        nw = features_ref.nw    
        sx = nw.skeleton_x
        sy = nw.skeleton_y
        worm_lengths = nw.length
    
        #TODO: Move these into posture options
    
        wave_options = features_ref.options.posture.wavelength
    
        # https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/
        # %2Bseg_worm/%2Bfeatures/%40posture/getAmplitudeAndWavelength.m
        N_POINTS_FFT = wave_options.n_points_fft
        HALF_N_FFT = int(N_POINTS_FFT / 2)
        MIN_DIST_PEAKS = wave_options.min_dist_peaks
        WAVELENGTH_PCT_MAX_CUTOFF = wave_options.pct_max_cutoff
        WAVELENGTH_PCT_CUTOFF = wave_options.pct_cutoff
    
        # TODO: Write in Python
        # assert(size(sx,1) <= N_POINTS_FFT,'# of points used in the FFT 
        # must be more than the # of points in the skeleton')
    
        #Rotate the worm so that it lies primarily along a single axis
        #-------------------------------------------------------------
        theta_r = theta_d * (np.pi / 180)
        wwx = sx * np.cos(theta_r) + sy * np.sin(theta_r)
        wwy = sx * -np.sin(theta_r) + sy * np.cos(theta_r)
    
        # Subtract mean
        #-----------------------------------------------------------------
        #??? - Why isn't this done before the rotation?
        wwx = wwx - np.mean(wwx, axis=0)
        wwy = wwy - np.mean(wwy, axis=0)
    
        # Calculate track amplitude
        #-----------------------------------------------------------------
        amp1 = np.amax(wwy, axis=0)
        amp2 = np.amin(wwy, axis=0)
        amplitude_max = amp1 - amp2
        amp2 = np.abs(amp2)
        
        #Ignore NaN division warnings
        with np.errstate(invalid='ignore'):
            amplitude_ratio = np.divide(np.minimum(amp1, amp2), 
                                        np.maximum(amp1, amp2))
    
        # Calculate track length
        #-----------------------------------------------------------------
        # This is the x distance after rotation, and is different from the
        # worm length which follows the skeleton. This will always be smaller 
        # than the worm length. If the worm were perfectly straight these 
        # values would be the same.
        track_length = np.amax(wwx, axis=0) - np.amin(wwx, axis=0)
    
        # Wavelength calculation
        #-----------------------------------------------------------------
        dwwx = np.diff(wwx, 1, axis=0)
    
        # Does the sign change? This is a check to make sure that the 
        # change in x is always going one way or the other. Is sign of all 
        # differences the same as the sign of the first, or rather, are any 
        # of the signs not the same as the first sign, indicating a "bad 
        # worm orientation".
        #
        # NOT: This means that within a frame, if the worm x direction 
        #      changes, then it is considered a bad worm and is not 
        #      evaluated for wavelength
        #
    
        with np.errstate(invalid='ignore'):
            bad_worm_orientation = np.any(
                np.not_equal(np.sign(dwwx), np.sign(dwwx[0, :])), axis=0)
    
        n_frames = bad_worm_orientation.size
    
        primary_wavelength = np.zeros(n_frames)
        primary_wavelength[:] = np.NaN
        secondary_wavelength = np.zeros(n_frames)
        secondary_wavelength[:] = np.NaN
    
        # NOTE: Right now this varies from worm to worm which means the 
        # spectral resolution varies as well from worm to worm
        spatial_sampling_frequency = (wwx.shape[0] - 1) / track_length
    
        ds = 1 / spatial_sampling_frequency
    
        frames_to_calculate = \
            (np.logical_not(bad_worm_orientation)).nonzero()[0]
    
        for cur_frame in frames_to_calculate:
    
            # Create an evenly sampled x-axis, note that ds varies
            x1 = wwx[0, cur_frame]
            x2 = wwx[-1, cur_frame]
            if x1 > x2:
                iwwx = utils.colon(x1, -ds[cur_frame], x2)
                iwwy = np.interp(iwwx, 
                                 wwx[::-1, cur_frame], 
                                 wwy[::-1, cur_frame])
                iwwy = iwwy[::-1]
            else:
                iwwx = utils.colon(x1, ds[cur_frame], x2)
                iwwy = np.interp(iwwx, 
                                 wwx[:, cur_frame], 
                                 wwy[:, cur_frame])
                iwwy = iwwy[::-1]
    
            temp = np.fft.fft(iwwy, N_POINTS_FFT)
    
            if options.mimic_old_behaviour:
                iY = temp[0:HALF_N_FFT]
                iY = iY * np.conjugate(iY) / N_POINTS_FFT
            else:
                iY = np.abs(temp[0:HALF_N_FFT])
    
            # Find peaks that are greater than the cutoff
            peaks, indx = utils.separated_peaks(iY, 
                                                MIN_DIST_PEAKS, 
                                                True, 
                                                (WAVELENGTH_PCT_MAX_CUTOFF * 
                                                 np.amax(iY)))
    
            # This is what the supplemental says, not what was done in 
            # the previous code. I'm not sure what was done for the actual 
            # paper, but I would guess they used power.
            #
            # This gets used when determining the secondary wavelength, as 
            # it must be greater than half the maximum to be considered a 
            # secondary wavelength.
    
            # NOTE: True Amplitude = 2*abs(fft)/
            #                    (length_real_data i.e. 48 or 49, not 512)
            #
            # i.e. for a sinusoid of a given amplitude, the above formula 
            # would give you the amplitude of the sinusoid
    
            # We sort the peaks so that the largest is at the first index 
            # and will be primary, this was not done in the previous 
            # version of the code
            I = np.argsort(-1 * peaks)
            indx = indx[I]
    
            frequency_values = (indx - 1) / N_POINTS_FFT * \
                spatial_sampling_frequency[cur_frame]
    
            all_wavelengths = 1 / frequency_values
    
            p_temp = all_wavelengths[0]
    
            if indx.size > 1:
                s_temp = all_wavelengths[1]
            else:
                s_temp = np.NaN
    
            worm_wavelength_max = (WAVELENGTH_PCT_CUTOFF * 
                                   worm_lengths[cur_frame])
    
            # Cap wavelengths ...
            if p_temp > worm_wavelength_max:
                p_temp = worm_wavelength_max
    
            # ??? Do we really want to keep this as well if p_temp == worm_2x?
            # i.e., should the secondary wavelength be valid if the primary is
            # also limited in this way ?????
            if s_temp > worm_wavelength_max:
                s_temp = worm_wavelength_max
    
            primary_wavelength[cur_frame] = p_temp
            secondary_wavelength[cur_frame] = s_temp
    
    
        if options.mimic_old_behaviour:
            # In the old code, the first peak (i.e. larger wavelength, 
            # lower frequency) was always the primary wavelength, where as 
            # the new definition is based on the amplitude of the peaks, 
            # not their position along the frequency axis
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mask = secondary_wavelength > primary_wavelength
    
            temp = secondary_wavelength[mask]
            secondary_wavelength[mask] = primary_wavelength[mask]
            primary_wavelength[mask] = temp

        self.amplitude_max = amplitude_max
        self.amplitude_ratio = amplitude_ratio
        self.primary_wavelength = primary_wavelength
        self.secondary_wavelength = secondary_wavelength
        self.track_length = track_length
    
        timer.toc('posture.amplitude_and_wavelength')

"""

Old Vs New Code:
  - power instead of magnitude is used for comparison
  - primary and secondary wavelength may be switched ...
  - error in maxPeaksDist for distance threshold, not sure where in code
        - see frame 880 for example
        - minus 1 just gives new problem - see 1794

"""


def get_worm_kinks(features_ref):
    """
    Parameters
    ----------
    features_ref : movement_validation.features.worm_features.WormFeatures
    
    Returns
    -------
    numpy.array

    """

    nw = features_ref.nw
    timer = features_ref.timer
    timer.tic()

    options = features_ref.options

    KINK_LENGTH_THRESHOLD_PCT = options.posture.kink_length_threshold_pct

    bend_angles = nw.angles

    # Determine the bend segment length threshold.
    n_angles = bend_angles.shape[0]
    length_threshold = np.round(n_angles * KINK_LENGTH_THRESHOLD_PCT)

    # Compute a gaussian filter for the angles.
    #--------------------------------------------------------------------------
    # JAH NOTE: This is a nice way of getting the appropriate odd value
    # unlike the other code with so many if statements ...
    #- see window code which tries to get an odd value ...
    #- I'd like to go back and fix that code ...
    half_length_thr = np.round(length_threshold / 2)
    gauss_filter = utils.gausswin(half_length_thr * 2 + 1) / half_length_thr

    # Compute the kinks for the worms.
    n_frames = bend_angles.shape[1]
    n_kinks_all = np.zeros(n_frames, dtype=float)
    n_kinks_all[:] = np.NaN

    #(np.any(np.logical_or(mask_pos,mask_neg),axis=0)).nonzero()[0]

    nan_mask = np.isnan(bend_angles)

    for iFrame in (~np.all(nan_mask, axis=0)).nonzero()[0]:
        smoothed_bend_angles = filters.convolve1d(
            bend_angles[:, iFrame], gauss_filter, cval=0, mode='constant')

        # This code is nearly identical in getForaging
        #-------------------------------------------------------
        n_frames = smoothed_bend_angles.shape[0]

        with np.errstate(invalid='ignore'):
            dataSign = np.sign(smoothed_bend_angles)

        if np.any(np.equal(dataSign, 0)):
            # I don't expect that we'll ever actually reach 0
            # The code for zero was a bit weird, it keeps counting if no sign
            # change i.e. + + + 0 + + + => all +
            #
            # but if counts for both if sign change
            # + + 0 - - - => 3 +s and 4 -s
            raise Exception("Unhandled code case")

        sign_change_I = (
            np.not_equal(dataSign[1:], dataSign[0:-1])).nonzero()[0]

        end_I = np.concatenate(
            (sign_change_I, 
             n_frames * np.ones(1, dtype=np.result_type(sign_change_I))))

        wtf1 = np.zeros(1, dtype=np.result_type(sign_change_I))
        wtf2 = sign_change_I + 1
        start_I = np.concatenate((wtf1, wtf2))  # +2? due to inclusion rules???

        # All NaN values are considered sign changes, remove these ...
        keep_mask = np.logical_not(np.isnan(smoothed_bend_angles[start_I]))

        start_I = start_I[keep_mask]
        end_I = end_I[keep_mask]

        # The old code had a provision for having NaN values in the middle
        # of the worm. I have not translated that feature to the newer code. I
        # don't think it will ever happen though for a valid frame, only on the
        # edges should you have NaN values.
        if start_I.size != 0 and \
           np.any(np.isnan(smoothed_bend_angles[start_I[0]:end_I[-1]])):
            raise Exception("Unhandled code case")

        #-------------------------------------------------------
        # End of identical code ...

        lengths = end_I - start_I + 1

        # Adjust lengths for first and last:
        # Basically we allow NaN values to count towards the length for the
        # first and last stretches
        if lengths.size != 0:
            if start_I[0] != 0:  # Due to leading NaNs
                lengths[0] = end_I[0] + 1
            if end_I[-1] != n_frames:  # Due to trailing NaNs
                lengths[-1] = n_frames - start_I[-1]

        n_kinks_all[iFrame] = np.sum(lengths >= length_threshold)

    timer.toc('posture.kinks')

    return n_kinks_all


def get_worm_coils(features_ref, midbody_distance):
    """
    Get the worm's posture.coils.
    
    Parameters
    ----------
    features_ref : movement_validation.features.worm_features.WormFeatures    
    
    This function is currently very reliant on the MRC processor.
    
    Translated From:
    https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/
    %2Bseg_worm/%2Bfeatures/%40posture/getCoils.m

    """
    options = features_ref.options
    posture_options = options.posture
    timer = features_ref.timer
    
    fps = features_ref.video_info.fps     
    
    timer.tic()
    
    frame_code = features_ref.video_info.frame_code
    

    COIL_FRAME_THRESHOLD = posture_options.coiling_frame_threshold(fps)
    
    # These are values that are specific to the MRC processor
    COIL_START_CODES = [105, 106]
    # Code that indicates a frame was successfully segmented
    FRAME_SEGMENTED = 1 
    
    # Algorithm: Whenever a new start is found, find the 
    # first segmented frame; that's the end.

    # Add on a frame to allow closing a coil at the end ...
    coil_start_mask = (frame_code == COIL_START_CODES[0]) | (
        frame_code == COIL_START_CODES[1])
    np_false = np.zeros((1,), dtype=bool)
    coil_start_mask = np.concatenate((coil_start_mask, np_false))

    # NOTE: These are not guaranteed ends, just possible ends ...
    end_coil_mask = frame_code == FRAME_SEGMENTED
    np_true = ~np_false
    end_coil_mask = np.concatenate((end_coil_mask, np_true))

    in_coil = False
    coil_frame_start = -1
    n_coils = 0
    n_frames_plus1 = len(frame_code) + 1

    starts = []
    ends = []

    for iFrame in range(n_frames_plus1):
        if in_coil:
            if end_coil_mask[iFrame]:
                n_coil_frames = iFrame - coil_frame_start
                if n_coil_frames >= COIL_FRAME_THRESHOLD:
                    n_coils += 1

                    starts.append(coil_frame_start)
                    ends.append(iFrame - 1)

                in_coil = False
        elif coil_start_mask[iFrame]:
            in_coil = True
            coil_frame_start = iFrame

    if options.mimic_old_behaviour:
        if (len(starts) > 0) and (ends[-1] == len(frame_code) - 1):
            ends[-1] += -1
            starts[-1] += -1

    temp = events.EventList(np.transpose(np.vstack((starts, ends))))

    timer.toc('posture.coils')  

    return events.EventListWithFeatures(fps, temp, midbody_distance)


class Directions(object):

    """

    Attributes
    ----------
    tail2head : numpy.array
    head : numpy.array
    tail : numpy.array

    """

    # These are the names of the final fields
    direction_keys = ['tail2head', 'head', 'tail']

    def __init__(self, features_ref):
        """

        Parameters
        ----------
        features_ref : movement_validation.features.worm_features.WormFeatures

        """

        timer = features_ref.timer
        timer.tic()

        nw = features_ref.nw
        
        sx = nw.skeleton_x
        sy = nw.skeleton_y
        wp = nw.worm_partitions

        # For each set of indices, compute the centroids of the tip and tail
        # then compute a direction vector between them (tip - tail)

        # I - "indices" - really a tuple of start,stop
        TIP_I = [wp['head'], wp['head_tip'], wp['tail_tip']]
        TAIL_I = [wp['tail'], wp['head_base'], wp['tail_base']]

        TIP_S = [slice(*x) for x in TIP_I]  # S - slice
        TAIL_S = [slice(*x) for x in TAIL_I]

        for iVector, attribute_name in enumerate(self.direction_keys):
            tip_x = np.mean(sx[TIP_S[iVector], :], axis=0)
            tip_y = np.mean(sy[TIP_S[iVector], :], axis=0)
            tail_x = np.mean(sx[TAIL_S[iVector], :], axis=0)
            tail_y = np.mean(sy[TAIL_S[iVector], :], axis=0)

            dir_value = 180 / np.pi * np.arctan2(tip_y - tail_y, 
                                                 tip_x - tail_x)
            setattr(self, attribute_name, dir_value)

        timer.toc('posture.directions')


    @classmethod
    def from_disk(cls, data):

        self = cls.__new__(cls)

        for key in self.direction_keys:            
            temp_value = utils._extract_time_from_disk(data, key)
            setattr(self, key, temp_value)

        return self

    def __repr__(self):
        return utils.print_object(self)

    def __eq__(self, other):
        
        same_values = True
        for partition_key in self.direction_keys:
            value1 = getattr(self,partition_key)
            value2 = getattr(self,partition_key)
            same_values = same_values and \
                utils.correlation(value1, value2, 
                                   'posture.directions.' + partition_key,
                                   high_corr_value=0.99)
            
        return same_values    


def load_eigen_worms():
    """ 
    Load the eigen_worms, which are stored in a Matlab data file

    The eigenworms were computed by the Schafer lab based on N2 worms

    Returns
    ----------
    eigen_worms: [7 x 48]

    From http://stackoverflow.com/questions/50499/

    """
    current_module_path = inspect.getfile(inspect.currentframe())
    package_path = os.path.dirname(os.path.abspath(current_module_path))
    
    repo_path        = os.path.split(package_path)[0]
    eigen_worm_file_path = os.path.join(repo_path,
                                        'features',
                                        config.EIGENWORM_FILE)

    h = h5py.File(eigen_worm_file_path,'r')
    eigen_worms = h['eigenWorms'].value
    
    return np.transpose(eigen_worms)


def get_eigenworms(features_ref):
    """

    Parameters
    ----------
    features_ref : movement_validation.features.worm_features.WormFeatures
    
    Returns
    -------
    eigen_projections: [N_EIGENWORMS_USE, n_frames]

    """
    
    nw = features_ref.nw
    sx = nw.skeleton_x
    sy = nw.skeleton_y
    posture_options = features_ref.options.posture
    N_EIGENWORMS_USE = posture_options.n_eigenworms_use    
    
    timer = features_ref.timer
    timer.toc    
    
    #eigen_worms: [7,48]  
    eigen_worms = load_eigen_worms()    
    
    #???? How does this differ from nw.angles???
    angles = np.arctan2(np.diff(sy, n=1, axis=0), np.diff(sx, n=1, axis=0))

    n_frames = sx.shape[1]

    # need to deal with cases where angle changes discontinuously from -pi
    # to pi and pi to -pi.  In these cases, subtract 2pi and add 2pi
    # respectively to all remaining points.  This effectively extends the
    # range outside the -pi to pi range.  Everything is re-centred later
    # when we subtract off the mean.
    false_row = np.zeros((1, n_frames), dtype=bool)

    # NOTE: By adding the row of falses, we shift the trues
    # to the next value, which allows indices to match. Otherwise after every
    # find statement we would need to add 1, I think this is a bit faster ...

    with np.errstate(invalid='ignore'):
        mask_pos = np.concatenate(
            (false_row, np.diff(angles, n=1, axis=0) > np.pi), axis=0)
        mask_neg = np.concatenate(
            (false_row, np.diff(angles, n=1, axis=0) < -np.pi), axis=0)

    # Only fix the frames we need to, in which there is a jump in going
    # from one segment to the next ...
    fix_frames_I = (
        np.any(np.logical_or(mask_pos, mask_neg), axis=0)).nonzero()[0]

    for cur_frame in fix_frames_I:

        positive_jump_I = (mask_pos[:, cur_frame]).nonzero()[0]
        negative_jump_I = (mask_neg[:, cur_frame]).nonzero()[0]

        # subtract 2pi from remainging data after positive jumps
        # Note that the jumps impact all subsequent frames
        for cur_pos_jump in positive_jump_I:
            angles[cur_pos_jump:, cur_frame] -= 2 * np.pi

        # add 2pi to remaining data after negative jumps
        for cur_neg_jump in negative_jump_I:
            angles[cur_neg_jump:, cur_frame] += 2 * np.pi

    angles = angles - np.mean(angles, axis=0)

    eigen_projections = np.dot(eigen_worms[0:N_EIGENWORMS_USE,:],angles)
    timer.toc('posture.eigenworms')

    return eigen_projections
