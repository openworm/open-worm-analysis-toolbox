# -*- coding: utf-8 -*-
"""
locomotion_bends.py

Calculate the "Bends" locomotion feature

Contains two classes:
  LocomotionCrawlingBends, which yields properties:
    .head
      .amplitude
      .frequency
    .mid
      .amplitude
      .frequency
    .tail
      .amplitude
      .frequency

  LocomotionForagingBends, which yields properties:
    .amplitude
    .angleSpeed

"""



class LocomotionCrawlingBends(object):
  """
  Locomotion Crawling Bends Feature.
  
  Properties
  ---------------------------------------    
  .head
    .amplitude
    .frequency
  .mid
    .amplitude
    .frequency
  .tail
    .amplitude
    .frequency

  Methods
  ---------------------------------------    
  __init__(self, bend_angles, is_paused, is_segmented_mask)
  calls...
  h__getBendData(self, avg_bend_angles, fps, options, is_paused)
  calls...
    h__getBoundingZeroIndices(self, avg_bend_angles, min_win_size)
    and 
    h__getBandwidth(self, data_win_length,fft_data,max_peak_I,INIT_MAX_I_FOR_BANDWIDTH)
    
    which calls seg_worm.util.maxPeaksDis


  Notes
  ---------------------------------------    
  Formerly +segworm/+features/@locomotion/getLocomotionBends

  Originally, part of wormBends.m


  Note from Ev Yemini on Setup Options
  ---------------------------------------    
  Empirically I've found the values below achieve good signal.
  
  Furthermore:
  The body bend frequency is much easier to see (than foraging). The N2
  signal is clearly centered around 1/3Hz in both the literature and
  through visual inspection.
  
  I chose a high-frequency threshold of 4 frames. With 4 frames a 3-frame
  tick, resulting from segmentation noise, will be diluted by the
  additional frame.


  Nature Methods Description
  ---------------------------------------    

  Worm crawling is expressed as both an amplitude and frequency
  (Supplementary Fig. 4e). We measure these features instantaneously at
  the head, midbody, and tail. The amplitude and frequency are signed
  negatively whenever the worm’s ventral side is contained within the
  concave portion of its instantaneous bend.
  
  Crawling is only measured during forward and backward motion states.
  The worm bend mean angles (described in the section on “Posture”) show
  a roughly periodic signal as the crawling wave travels along the worm’s
  body. This wave can be asymmetric due to differences in dorsal-ventral
  flexibility or simply because the worm is executing a turn. Moreover
  the wave dynamics can change abruptly to speed up or slow down.
  Therefore, the signal is only roughly periodic and we measure its
  instantaneous properties.
  
  Worm bends are linearly interpolated across unsegmented frames. The
  motion states criteria (described earlier in this section) guarantee
  that interpolation is no more than 1/4 of a second long. For each
  frame, we search both backwards and forwards for a zero crossing in the
  bend angle mean – the location where the measured body part (head,
  midbody, or tail) must have hit a flat posture (a supplementary bend
  angle of 0°). This guarantees that we are observing half a cycle for
  the waveform. Crawling is bounded between 1/30Hz (a very slow wave that
  would not resemble crawling) and 1Hz (an impossibly fast wave on agar).
  
  If the window between zero crossings is too small, the nearest zero
  crossing is assumed to be noise and we search for the next available
  zero crossing in its respective direction. If the window is too big,
  crawling is marked undefined at the frame.
  
  Once an appropriate window has been found, the window is extended in
  order to center the frame and measure instantaneous crawling by
  ensuring that the distance on either side to respective zero crossings
  is identical. If the distances are not identical, the distance of the
  larger side is used in place of the zero-crossing distance of the
  smaller side in order to expand the small side and achieve a symmetric
  window, centered at the frame of interest.
  
  We use a Fourier transform to measure the amplitude and frequency
  within the window described above. The largest peak within the
  transform is chosen for the crawling amplitude and frequency. If the
  troughs on either side of the peak exceed 1/2 its height, the peak is
  rejected for being unclear and crawling is marked as undefined at the
  frame. Similarly, if the integral between the troughs is less than half
  the total integral, the peak is rejected for being weak.


  """


  def __init__(self, bend_angles, is_paused, is_segmented_mask):
    """
    Compute the temporal bending frequency at the head, midbody, and tail.
    
    Parameters
    ---------------------------------------    
    bend_angles       : [49 x n_frames]
    is_paused         : [1 x n_frames]
    is_segmented_mask : [1 x n_frames]
    
    """
    pass
  
    """
    minBodyWinTime = 0.5
    minBodyWin     = round(minBodyWinTime * fps)
    maxBodyWinTime = 15
    maxBodyWin     = round(maxBodyWinTime * fps)
    options = struct( ...
        'minWin',   minBodyWin, ...
        ... 
        'maxWin',   maxBodyWin, ...
        ... 
        'res',      2^14, ... # the FFT is quantized below this resolution
        ... 
        'headI',    6:10, ... # centered at the head (1/6 the worm)
        ... 
        'midI',     23:27, ... # centered at the middle of the worm
        ... 
        'tailI',    40:44, ... # centered at the tail (1/6 the worm)
        ... 
        'minFreq',  1 / (4 * maxBodyWinTime), ... # require at least 50% of the wave
        ... 
        'maxFreq',  fps / 4, ... # with 4 frames we can resolve 75% of a wave ????
        'max_amp_pct_bandwidth', 0.5,...
        'peakEnergyThr',0.5)
    
    #max_amp_pct_bandwidth - when determining the bandwidth,
    #the minimums that are found can't exceed this percentage of the maximum.
    #Doing so invalidates the result,
    #
    #
    Why are the indices used that are used ????
    #
    #NOTE: These indices are not symettric. Should we use the skeleton indices
    #instead, or move these to the skeleton indices???
    #
    #SI = seg_worm.skeleton_indices
    
    # No worm data.
    #------------------------------------
    if ~any(is_segmented_mask)
        nanData = nan(1, length(is_segmented_mask))
        bend_struct = struct('frequency',nanData,'amplitude',nanData)
        bends = struct( ...
            'head',     bend_struct, ...
            'mid',      bend_struct, ...
            'tail',     bend_struct)
        return
    end
    
    #Set things up for the loop
    #------------------------------------
    section     = {'head'           'midbody'       'tail'}
    all_indices = {options.headI    options.midI    options.tailI}
    
    #Initialize interpolation - we'll interpolate over all frames but no extrapolation
    #--------------------------------------------------------------------------
    interpType  = 'linear'
    dataI       = find(is_segmented_mask)
    interpI     = find(~is_segmented_mask)
    
    bends = struct()
    for iBend = 1:3
        
        cur_indices     = all_indices{iBend}
        avg_bend_angles = mean(bend_angles(cur_indices,:), 1)
    
        if ~isempty(interpI) && length(dataI) > 1
            avg_bend_angles(interpI) = interp1(dataI, avg_bend_angles(dataI), interpI, interpType, NaN)
        end
        
        [amps,freqs] = self.h__getBendData(avg_bend_angles,fps,options,is_paused)
        
        bends.(section{iBend}).frequency = freqs
        bends.(section{iBend}).amplitude = amps
    end
        
    obj.bends = bends
  
    """
  
  
  def h__getBendData(self, avg_bend_angles, fps, options, is_paused):
    """
    Notes
    ----------------------
    Formerly [amps,freqs] = h__getBendData(avg_bend_angles, fps, options, is_paused)
    #
    #
    Compute the bend amplitude and frequency.
    #
    h__getBendData(avg_bend_angles, fps, options, is_paused)
    #
    Inputs
    =======================================================================
    avg_bend_angles : [1 x n_frames]
    fps             : (scalar)
    options         : (struct), this is defined in the calling function
    is_paused       : [1 x n_frames], whether or not the worm is considered
                      to be paused during the frame
    #
    
    INIT_MAX_I_FOR_BANDWIDTH = 2000 #TODO: Relate this to a real frequency ...
    #and pass it in from higher up. This number is NOT IMPORTANT TO THE OUTCOME
    #and is only to the speed in which the function runs. We try and find the
    #bandwidth within this number of samples. 
    
    #TODO: We need to check that the value above is less than the # of samples
    #in the FFT. We might also change this to being a percentage of the # of
    #points. Currently this is around 25% of the # of samples.
    
    #Options extraction
    #----------------------------------
    """
    pass
    """
    min_window = options.minWin
    max_window = options.maxWin
    max_freq   = options.maxFreq
    min_freq   = options.minFreq
    fft_n_samples = options.res
    max_amp_pct_bandwidth   = options.max_amp_pct_bandwidth
    peakEnergyThr = options.peakEnergyThr
    
    # TODO: This needs to be cleaned up ...
    [back_zeros_I,front_zeros_I] = self.h__getBoundingZeroIndices(avg_bend_angles,min_window)
    
    n_frames = length(avg_bend_angles)
    
    left_distances  = (1:n_frames) - back_zeros_I
    right_distances = front_zeros_I - (1:n_frames)
    half_distances  = max(left_distances,right_distances)
    
    
    left_bounds  = (1:n_frames) - half_distances
    right_bounds = (1:n_frames) + half_distances
    
    #Compute conditions by which we will ignore frames:
    #--------------------------------------------------------------------------
    #- frame is not bounded on both sides by a sign change
    #- avg_bend_angles is NaN, this will only happen on the edges because we
        interpolate over the other frames ... (we just don't extrapolate)
    #- the sign change region is too large
    #- the bounds we settle on exceed the data region
    #- mode segmentation determined the frame was a paused frame
    #
    #
    #??? - what about large NaN regions, are those paused regions???
    
    is_bad_mask  = back_zeros_I == 0 | front_zeros_I == 0 | isnan(avg_bend_angles) | half_distances > max_window
    is_bad_mask  = is_bad_mask | left_bounds < 1 | right_bounds > n_frames | is_paused
    
    
    # Compute the short-time Fourier transforms (STFT).
    fft_max_I   = fft_n_samples / 2 #Maximum index to keep for frequency analysis
    freq_scalar = (fps / 2) * 1/(fft_max_I - 1)
    
    n_frames = length(avg_bend_angles)
    
    amps  = NaN(1,n_frames)
    freqs = NaN(1,n_frames)
    for iFrame = find(~is_bad_mask)
        
        windowed_data   = avg_bend_angles(left_bounds(iFrame):right_bounds(iFrame))
        data_win_length = length(windowed_data)
        
        #fft frequency and bandwidth
        #----------------------------------------------------------------------
        # Compute the real part of the STFT.
        #These two steps take a lot of time ...
        fft_data = fft(windowed_data, fft_n_samples)
        fft_data = abs(fft_data(1:fft_max_I))
        
        # Find the peak frequency.
        [maxPeak,maxPeakI] = max(fft_data)
                
        #NOTE: If this is true, we'll never bound the peak on the left ...
        if maxPeakI == 1
            continue
        end
        
        #TODO: Not sure if this value is correct ...
        unsigned_freq = freq_scalar*(maxPeakI - 1)
        
        if unsigned_freq > max_freq || unsigned_freq < min_freq
           continue 
        end
        
        [peakStartI,peakEndI] = self.h__getBandwidth(data_win_length,fft_data,maxPeakI,INIT_MAX_I_FOR_BANDWIDTH)
        
        #Store data
        #----------------------------------------------------------------------
        if ~(   isempty(peakStartI)                           || ...
                isempty(peakEndI)                             || ...
                fft_data(peakStartI) > max_amp_pct_bandwidth*maxPeak    || ... #The minimums can't be too big
                fft_data(peakEndI)   > max_amp_pct_bandwidth*maxPeak    || ... 
                sum(fft_data(peakStartI:peakEndI) .^ 2) < peakEnergyThr * sum(fft_data .^ 2)) #Needs to have enough energy
    
            # Convert the peak to a time frequency.
            dataSign      = sign(mean(windowed_data)) # sign the data
            amps(iFrame)  = (2 * fft_data(maxPeakI) / data_win_length) * dataSign
            freqs(iFrame) = unsigned_freq * dataSign
        end
        
    end
    """
    
    
  
  
  
  
  
  
  
  
  def h__getBandwidth(self, data_win_length,fft_data,max_peak_I,INIT_MAX_I_FOR_BANDWIDTH):
    """
    
    Notes
    -----------------
    Formerly [peak_start_I,peak_end_I] = h__getBandwidth(data_win_length,fft_data,max_peak_I,INIT_MAX_I_FOR_BANDWIDTH)
    #
    # 
    The goal is to find minimum 'peaks' that border the maximal frequency
    response.
      
    Since this is a time intensive process, we try and start with a small
    range of frequencies, as execution time is proportional to the length
    of the input data. If this fails we use the full data set.
    
    Inputs
    =======================================================================
    data_win_length : length of real data (ignoring zero padding) that 
                      went into computing the FFT
    fft_data        : output of the fft function
    max_peak_I      : location (index) of the maximum of fft_data
    INIT_MAX_I_FOR_BANDWIDTH : see code
    #
    Outputs
    =======================================================================
    peak_start_I : (scalar)
    peak_end_I   : (scalar)
    #
    See Also:
    seg_worm.util.maxPeaksDist
    """
    pass
    """
    peakWinSize = round(sqrt(data_win_length))
  
    # Find the peak bandwidth.
    #----------------------------------------------------------------------
    
    
    if max_peak_I < INIT_MAX_I_FOR_BANDWIDTH
        
        #NOTE: It is incorrect to filter by the maximum here, as we want to
        #allow matching a peak that will later be judged invalid. If we
        #filter here we may find another smaller peak which will not be
        #judged invalid later on.
        [~, min_peaks_I] = seg_worm.util.maxPeaksDist(fft_data(1:INIT_MAX_I_FOR_BANDWIDTH), peakWinSize,false,Inf)
  
        peak_start_I = min_peaks_I(find(min_peaks_I < max_peak_I,1))
        peak_end_I   = min_peaks_I(find(min_peaks_I > max_peak_I,1))
    else
        peak_start_I = []
        peak_end_I   = []
    end
    
    #NOTE: Besides checking for an empty value, we also need to ensure that
    #the minimum didn't come too close to the data border, as more data
    #could invalidate the result we have.
    if isempty(peak_end_I) || peak_end_I + peakWinSize >= INIT_MAX_I_FOR_BANDWIDTH   
        [~, min_peaks_I] = seg_worm.util.maxPeaksDist(fft_data, peakWinSize,false,Inf)
  
        peak_start_I = min_peaks_I(find(min_peaks_I < max_peak_I,1))
        peak_end_I   = min_peaks_I(find(min_peaks_I > max_peak_I,1))      
    end
    """
  
  
  
  
  
  
  
  
  
  def h__getBoundingZeroIndices(self, avg_bend_angles, min_win_size):
    """
    Notes
    ----------------------
    Formerly [back_zeros_I,front_zeros_I] = h__getBoundingZeroIndices(avg_bend_angles,min_win_size)
    #
    #
    The goal of this function is to bound each index by 
    #
    Inputs
    =====================================================
    avg_bend_angles : [1 x n_frames]
    min_win_size    : (scalar), the minimum size of the data window
    #
    Outputs
    =======================================================================
    back_zeros_I    : [1 x n_frames] For each frame, this specifies a
                    proceeding frame in which a change in the bend angle
                    occurs. Invalid entries are indicated by 0.
    front_zeros_I   : [1 x n_frames]
    #  
    """
    pass
    """
    #Getting sign change indices ...
    #--------------------------------------------------------------------------
    #The old code found sign changes for every frame, even though the sign
    #changes never changed. Instead we find all sign changes, and then for each
    #frame know which frame to the left and right have sign changes. We do this
    #in such a way so that if we need to look further to the left or right, it
    #is really easy to get the next answer.
    
    sign_change_I  = find(sign(avg_bend_angles(2:end)) ~= sign(avg_bend_angles(1:end-1)))
    n_sign_changes = length(sign_change_I)
    
    #To get the correct frame numbers, we need to do the following depending on
    #whether or not the bound is the left (backward) bound or the right
    #(forward) bound. JAH NOTE: I haven't really thought through why this is, but
    #it mimics the old code.
    #
    #
    #for left bounds   - at sign changes - don't subtract or add
    #for right bounds  - we need to add 1
    
    
    #Let's say we have sign changes at indices 3  6  9
    #What we need ...
         1 2 3 4 5 6 7  9  10  Indices
    #Left  = 0 0 0 3 3 3 6  6  6   - at 4, the left sign change is at 3
    #Right = 4 4 4 7 7 7 10 10 0   - at 4, the right sign change is at 7
    #
    #NOTE: The values above are the final indices or values, but instead we
    #want to work with the indices, so we need:
    #
         1 2 3 4 5 6 7  9  10  Indices
    #Left  = 0 0 0 1 1 1 2  2  2 - left_sign_change_I
    #Right = 1 1 1 2 2 2 3  3  3 - right_sign_change_I
    #
    we also need:
    left_values  = [3 6 9]  #the sign change indices
    right_values = [4 7 10] #+1
    #
    So this says:
    left_sign_change_I(7) => 2
    left_values(2) => 6, our sign change is at 6
    #
    Let's say we need to expand further to the left, then we take 
    left_sign_change_I(7) - 1 => 1
    left_values(1) => 3, our new sign change is at 3
    #
    Further:
    left_sign_change_I(7) - 2 => 0
    We've gone too far, nothing at index 0, set to invalid
    
    
    # dSignChange = diff(sign_change_I)
    
    n_frames = length(avg_bend_angles)
    
    #For each element in the array, these values indicate which sign change
    #index to use ...
    left_sign_change_I  = zeros(1,n_frames)
    left_sign_change_I(sign_change_I + 1) = 1 #We increment at values to the 
    #right of the sign changes
    left_sign_change_I = cumsum(left_sign_change_I) #This is a little Matlab
    #trick in which something like:
    #1 0 0 1 0 0 1 0 0 
    becomes:
    #1 1 1 2 2 2 3 3 3 - so now at each frame, we get the index of the value
    #that is to the left
    
    right_sign_change_I    = zeros(1,n_frames)
    right_sign_change_I(sign_change_I(1:end-1)+1) = 1
    right_sign_change_I(1) = 1
    right_sign_change_I    = cumsum(right_sign_change_I)
    right_sign_change_I(sign_change_I(end)+1:end) = 0 #Nothing to the right of the last change ...
    
    #These are the actual indices that each sign crossing index points to
    #
    #We keep these separate as it makes it easier to go the next value, by
    #incrementing the pointer index, rather than doing a search
    left_values  = sign_change_I
    right_values = sign_change_I + 1 
    #--------------------------------------------------------------------------
    
    back_zeros_I  = zeros(1,n_frames)
    front_zeros_I = zeros(1,n_frames)
    
    for iFrame = 1:n_frames
        
        cur_left_index  = left_sign_change_I(iFrame)
        cur_right_index = right_sign_change_I(iFrame)
        if left_sign_change_I(iFrame) == 0 || right_sign_change_I(iFrame) == 0
            continue
        end
        
        back_zero_I  = left_values(cur_left_index)
        front_zero_I = right_values(cur_right_index)
        
        use_values = true
        
        # Expand the zero-crossing window.
        #----------------------------------------------------------------------
        #JAH NOTE: We center on the sample by using the larger of the two gaps.
        #This means the gap size is 2*larger_window, where the half window size
        #is:
        #left_window_size  = iFrame - back_zero_I
        #right_window_size = front_zero_I - iFrame
        #
        #so in reality we should use:
        #
        #front_zero_I - iFrame < min_win_size/2 && iFrame - back_zero_I < min_win_size/2
        #
        #By not doing this, we overshoot the minimum window size that we need
        #to use. Consider window sizes that are in terms of the minimum window
        #size.
        #
        #i.e. 0.5w means the left or right window is half min_win_size
        #
        #Consider we have:
        #0.5w left 
        #0.3w right
        #
        #If we stopped now, we would get a windows size of 2*0.5w or w, which
        #is what we want.
        #
        #Since we keep going, the right window will expand, let's say:
        #0.5w left
        #0.7w right
        #
        #Now in the calling function we will center on the frame and the total 
        #window distance will be 2*0.7w or 1.4w, which is larger than we
        #needed.
        
        while front_zero_I - back_zero_I + 1 < min_win_size
            
            #Expand the smaller of the two windows
            #----------------------------------------------------
            #  left_window_size       right_window_size
            if iFrame - back_zero_I < front_zero_I - iFrame
                #Then expand to the left
                cur_left_index = cur_left_index - 1
                if cur_left_index == 0
                    use_values = false
                    break
                end
                back_zero_I  = left_values(cur_left_index)
            else
                #Expand to the right 
                cur_right_index = cur_right_index + 1
                if cur_right_index > n_sign_changes
                    use_values = false
                    break
                end
                front_zero_I = right_values(cur_right_index)
            end
        end
        
        if use_values
            back_zeros_I(iFrame)  = back_zero_I
            front_zeros_I(iFrame) = front_zero_I
        end
    end
    """
  









class LocomotionForagingBends(object):
  """
  Locomotion Foraging Bends Feature.
  
  Properties
  ---------------------------------------    
  amplitude
  angleSpeed
  
  
  Methods
  ---------------------------------------    
  __init__
  h__computeNoseBends
  h__computeAvgAngles
  h__interpData
  h__getNoseInterpolationIndices
  h__foragingData
  h__getAmps

  Notes
  ---------------------------------------    
  Formerly +segworm/+features/@locomotion/getForaging

  Originally, part of wormBends.m


  Note from Ev Yemini on Setup Options
  ---------------------------------------    
  Note: empirically I've found the values below achieve good signal.
  
  Furthermore ...
  Huang et al. in 2006, measure foraging frequencies for several worms and
  find the signal centered at roughly 4Hz. For N2 worms, they see a second
  signal at 10Hz but I find this value too close to the background noise
  present in segmentation. Visually inspecting the foraging signal, as the
  bend between the nose and neck, corroborates a roughly 4Hz signal. But,
  foraging usually encompasses only half to a quarter cycle. In other
  words, the worm bends it nose sharply and sometimes bends it back but a
  full wave, akin to a body bend, occurs far less frequently. Therefore I
  chose to measure angular speed for foraging.  
  
  
  Nature Methods Description: Foraging
  ---------------------------------------    
  Foraging. Worm foraging is expressed as both an amplitude and an
  angular speed (Supplementary Fig. 4g). Foraging is signed negatively
  whenever it is oriented towards the ventral side. In other words, if
  the nose is bent ventrally, the amplitude is signed negatively.
  Similarly, if the nose is moving ventrally, the angular speed is signed
  negatively. As a result, the amplitude and angular speed share the same
  sign roughly only half the time. Foraging is an ambiguous term in
  previous literature, encompassing both fine movements of the nose as
  well as larger swings associated with the head. Empirically we have
  observed that the nose movements are aperiodic while the head swings
  have periodicity. Therefore, we measure the aperiodic nose movements
  and term these foraging whereas the head swings are referred to as
  measures of head crawling (described earlier in this section).
  
  Foraging movements can exceed 6Hz7 and, at 20-30fps, our video frame
  rates are just high enough to resolve the fastest movements. By
  contrast, the slowest foraging movements are simply a continuation of
  the crawling wave and present similar bounds on their dynamics.
  Therefore, we bound foraging between 1/30Hz (the lower bound used for
  crawling) and 10Hz.
  
  To measure foraging, we split the head in two (skeleton points 1-4 and
  5-8) and measure the angle between these sections. To do so, we measure
  the mean of the angle between subsequent skeleton points along each
  section, in the tail-to-head direction. The foraging angle is the
  difference between the mean of the angles of both sections. In other
  words, the foraging angle is simply the bend at the head.
  
  Missing frames are linearly interpolated, per each skeleton point, for
  fragments up to 0.2 seconds long (4-6 frames at 20-30fps – twice the
  upper foraging bound). When larger fragments are missing, foraging is
  marked undefined. Segmentation of the head at very small time scales
  can be noisy. Therefore, we smooth the foraging angles by convolving
  with a Gaussian filter 1/5 of a second long (for similar reasons to
  those mentioned in frame interpolation), with a width defined by the
  Matlab “gausswin” function’s default ? of 2.5 and normalized such that
  the filter integrates to 1.
  
  The foraging amplitude is defined as the largest foraging angle
  measured, prior to crossing 0°. In other words, the largest nose bend
  prior to returning to a straight, unbent position. Therefore, the
  foraging amplitude time series follows a discrete, stair-step pattern.
  The amplitude is signed negatively whenever the nose points towards the
  worm’s ventral side. The foraging angular speed is measured as the
  foraging angle difference between subsequent frames divided by the time
  between these frames. To center the foraging angular speed at the frame
  of interest and eliminate noise, each frame is assigned the mean of the
  angular speed computed between the previous frame and itself and
  between itself and the next frame. The angular speed is signed
  negatively whenever its vector points towards the worm’s ventral side.

  """

  def __init__(self, sx, sy, is_segmented_mask, ventral_mode):
    """
    Compute the temporal bending frequency at the head, midbody, and tail.
    
    Parameters
    ---------------------------------------    
    sx                 : [1 x n_frames]
    sy                 : [1 x n_frames]
    is_segmented_mask  : [1 x n_frames]
    ventral_mode       : [1 x n_frames]
    
    """
    pass

    """
    SI = seg_worm.skeleton_indices
    NOSE_I = SI.HEAD_TIP_INDICES(end:-1:1) %flip to maintain orientation for
    %angles and consistency with old code ...
    NECK_I = SI.HEAD_BASE_INDICES(end:-1:1)
    
    MIN_NOSE_WINDOW = round(0.1*FPS)
    MAX_NOSE_INTERP = 2*MIN_NOSE_WINDOW - 1
    
    nose_x = sx(NOSE_I,:)
    nose_y = sy(NOSE_I,:)
    neck_x = sx(NECK_I,:)
    neck_y = sy(NECK_I,:)
    
    # Step 1: Interpolation of skeleton indices
    #---------------------------------------    

    interp_nose_mask = h__getNoseInterpolationIndices(is_segmented_mask,MAX_NOSE_INTERP)
    
    dataI       = find(is_segmented_mask)
    noseInterpI = find(interp_nose_mask)
    
    nose_xi = h__interpData(nose_x,dataI,noseInterpI)
    nose_yi = h__interpData(nose_y,dataI,noseInterpI)
    neck_xi = h__interpData(neck_x,dataI,noseInterpI)
    neck_yi = h__interpData(neck_y,dataI,noseInterpI)
    
    # Step 2: Calculation of the bend angles
    #---------------------------------------    
    nose_bends = h__computeNoseBends(nose_xi,nose_yi,neck_xi,neck_yi)
    
    # Step 3: 
    #---------------------------------------    
    [nose_amps, nose_freqs] = h__foragingData(nose_bends, MIN_NOSE_WINDOW, FPS)
    
    if ventral_mode > 1:
        nose_amps  = -nose_amps
        nose_freqs = -nose_freqs
    
    self.amplitude  = nose_amps
    self.angleSpeed = nose_freqs
    
    """

  def h__computeNoseBends(nose_x, nose_y, neck_x, neck_y):
    """
    Compute the difference in angles between the nose and neck (really the
    head tip and head base).
    
    Formerly nose_bends_d = h__computeNoseBends(nose_x,nose_y,neck_x,neck_y)
    
    Inputs
    ======================================================
    nose_x: [4 x n_frames]
    nose_y: [4 x n_frames]
    neck_x: [4 x n_frames]
    neck_y: [4 x n_frames]

    Outputs
    ======================================================
    nose_bends_d

    """
    pass
    """

    noseAngles = h__computeAvgAngles(nose_x,nose_y)
    neckAngles = h__computeAvgAngles(neck_x,neck_y)
    
    %TODO: These three should be a method, calculating the difference
    %in angles and ensuring all results are within +/- 180
    nose_bends_d  = (noseAngles - neckAngles)'*180/pi
    
    nose_bends_d(nose_bends_d > 180)  = nose_bends_d(nose_bends_d > 180) - 360
    nose_bends_d(nose_bends_d < -180) = nose_bends_d(nose_bends_d < -180) + 360
    end
    """
    
  def h__computeAvgAngles(x, y):
    """
    
    
    Formerly angles = h__computeAvgAngles(x,y)
    
    
    Take average difference between successive x and y skeleton points, the
    compute the arc tangent from those averages.
    
    Simple helper for h__computeNoseBends
    """
    pass
    """
    avg_diff_x = mean(diff(x,1,1))
    avg_diff_y = mean(diff(y,1,1))
    
    angles = atan2(avg_diff_y,avg_diff_x)
    """

  def h__interpData(x_old, good_data_I, fix_data_I):
    """
    
    
    Formerly x_new = h__interpData(x_old,good_data_I,fix_data_I)
    
    TODO: move to: seg_worm.feature_helpers.interpolateNanData
    
    Inputs
    =======================================================================
    x_old       : [4 x n_frames]
    good_data_I : [1 x m]
    fix_data_I  : [1 x n]
    """
    pass
    """
    x_new = x_old
    
    #NOTE: This version is a bit weird because the size of y is not 1d
    for i1 = 1:size(x_old,1)
       x_new(i1,fix_data_I) = interp1(good_data_I,x_old(i1,good_data_I),fix_data_I,'linear',NaN) 
    end
    
    """


def h__getNoseInterpolationIndices(is_segmented_mask,max_nose_interp_sample_width):
  """
  
  
  Formerly interp_data_mask = h__getNoseInterpolationIndices(is_segmented_mask,max_nose_interp_sample_width)
  
  Interpolate data:
  - but don't extrapolate
  - don't interpolate if the gap is too large
  
  Inputs
  =======================================================================
  is_segmented_mask :
  max_nose_interp_sample_width : Maximum # of frames that 
  
  Outputs
  =======================================================================
  interp_data_mask : [1 x n_frames} whether or not to interpolate a data point
  
  Note from @JimHokason: 
  I'm considering replacing this and other interpolation code
  with a specific function for all feature processing.

  Note from @MichaelCurrie:
  I made this function, it's called feature_helpers.interpolate_with_threshold
  TODO: use it here, replacing this function.

  """
  pass
  
  """
  % Find the start and end indices for missing data chunks.
  isNotData        = ~is_segmented_mask
  interp_data_mask = isNotData
  diffIsNotData    = diff(isNotData)
  
  startNotDataI    = find(diffIsNotData == 1)
  endNotDataI      = find(diffIsNotData == -1) - 1
  
  % Don't interpolate missing data at the very start and end.
  %--------------------------------------------------------------------------
  if ~isempty(startNotDataI) && (isempty(endNotDataI) || startNotDataI(end) > endNotDataI(end))
      interp_data_mask(startNotDataI(end):end) = false
      startNotDataI(end) = []
  end
  
  if ~isempty(endNotDataI) && (isempty(startNotDataI) || startNotDataI(1) > endNotDataI(1))
      interp_data_mask(1:endNotDataI(1)) = false
      endNotDataI(1) = []
  end
  
  % Don't interpolate large missing chunks of data.
  for i = 1:length(startNotDataI)
      if endNotDataI(i) - startNotDataI(i) > max_nose_interp_sample_width
          interp_data_mask(startNotDataI(i):endNotDataI(i)) = false
      end
  end
  
  """



  def h__foragingData(nose_bend_angle_d, min_win_size, fps):
    """
    
    Formerly [amps,speeds] = h__foragingData(nose_bend_angle_d, min_win_size, fps)
    
    Compute the foraging amplitude and angular speed.
    
    
    Inputs
    =======================================================================
    nose_bend_angle_d : [n_frames x 1]
    min_win_size : (scalar)
    fps : (scalar)
    
    Outputs
    =======================================================================
    amps   : [1 x n_frames]
    speeds : [1 x n_frames]
    
    
    """
    pass
    
    """
    # Clean up the signal with a gaussian filter.
    if min_win_size > 0
        gaussFilter       = gausswin(2 * min_win_size + 1) / min_win_size
        nose_bend_angle_d = conv(nose_bend_angle_d, gaussFilter, 'same')
        
        %Remove partial data frames ...
        nose_bend_angle_d(1:min_win_size) = NaN
        nose_bend_angle_d((end - min_win_size + 1):end) = NaN
    end
    
    %Calculate amplitudes
    amps = h__getAmps(nose_bend_angle_d)
    
    
    %Calculate angular speed
    % Compute the speed centered between the back and front foraging movements.
    %
    %  1     2    3
     d1    d2     d1 = 2 - 1,   d2 = 3 - 2
         x        assign to x, avg of d1 and d2
    
    %???? - why multiply and not divide by fps????
    
    d_data = diff(nose_bend_angle_d) * fps
    speeds = NaN(size(amps))
    speeds(2:end-1) = (d_data(1:(end - 1)) + d_data(2:end)) / 2
    
    
    %Propagate NaN for speeds to amps
    amps(isnan(speeds)) = NaN
    
    """

  def h__getAmps(nose_bend_angle_d):
    """
    
    Formerly amps = h__getAmps(nose_bend_angle_d):
    
    
    In between all sign changes, get the maximum or minimum value and
    apply to all indices that have the same sign within the stretch
    
    i.e. 
    
    1 2 3 2 1 -1 -2 -1 1 2 2 5 becomes
    3 3 3 3 3 -2 -2 -2 5 5 5 5
    
    NOTE: This code is very similar to wormKinks
    """
    pass
    """
    n_frames = length(nose_bend_angle_d)
    
    dataSign      = sign(nose_bend_angle_d)
    sign_change_I = find(dataSign(2:end) ~= dataSign(1:end-1))
    
    end_I   = [sign_change_I; n_frames]
    start_I = [1; sign_change_I+1]
    
    %All Nan values are considered sign changes, remove these ...
    mask = isnan(nose_bend_angle_d(start_I))
    start_I(mask) = []
    end_I(mask)   = []
    
    amps = NaN(1,n_frames)
    
    %For each chunk, get max or min, depending on whether the data is positive
    %or negative ...
    for iChunk = 1:length(start_I)
       cur_start = start_I(iChunk)
       cur_end   = end_I(iChunk)
       
       if nose_bend_angle_d(cur_start) > 0
           amps(cur_start:cur_end) = max(nose_bend_angle_d(cur_start:cur_end))
       else
           amps(cur_start:cur_end) = min(nose_bend_angle_d(cur_start:cur_end))
       end
    end
    """



