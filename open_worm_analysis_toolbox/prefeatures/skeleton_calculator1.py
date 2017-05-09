# -*- coding: utf-8 -*-
"""
@JimHokason's translation of the Schafer Lab's skeletonization code.


"""
import numpy as np
import matplotlib.pyplot as plt

# If you are interested to know why the following line didn't work:
# import scipy.signal.savgol_filter as sgolay
# check out this: http://stackoverflow.com/questions/29324814/
# Instead we use the following statement:
from scipy.signal import savgol_filter as sgolay

from .. import utils
from .pre_features_helpers import WormParserHelpers

#%%


class SkeletonCalculatorType1(object):

    """
    The main method in this clas is compute_skeleton_and_widths. All other
    methods are just subfunctions of this main method.

    """
    #%%
    @staticmethod
    def compute_skeleton_and_widths(h_ventral_contour,
                                    h_dorsal_contour,
                                    frames_to_plot=[]):
        """
        Compute widths and a heterocardinal skeleton from a heterocardinal
        contour.

        Parameters
        -------------------------
        h_ventral_contour: list of numpy arrays.
            Each frame is an entry in the list.
        h_dorsal_contour:
        frames_to_plot: list of ints
            Optional list of frames to plot, to show exactly how the
            widths and skeleton were calculated.


        Returns
        -------------------------
        (h_widths, h_skeleton): tuple
            h_widths : the heterocardinal widths, frame by frame
            h_skeleton : the heterocardinal skeleton, frame by frame.


        Original algorithm notes:
        -------------------------
        Original code for this algorithm can be found at:
        https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/
        %2Bseg_worm/%2Bworm/%40skeleton/linearSkeleton.m
        Which calls an initial skeletonization algorithm:
        https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/
        %2Bseg_worm/%2Bcv/skeletonize.m
        Which then gets refined:
        https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/
        %2Bseg_worm/%2Bworm/%40skeleton/cleanSkeleton.m

        Widths are simply the distance between two "corresponding" sides of
        the contour. The question is how to get these two locations.

        From Ev's Thesis: 3.3.1.6 - page 126 (or 110 as labeled in document)
        -------------------------
        For each section, we begin at its center on both sides of the contour.
        We then walk, pixel by pixel, in either direction until we hit the end
        of the section on opposite sides, for both directions. The midpoint,
        between each opposing pixel pair, is considered the skeleton and the
        distance between these pixel pairs is considered the width for each
        skeleton point.

        Food tracks, noise, and other disturbances can form spikes on the
        worm contour. When no spikes are present, our walk attempts to
        minimize the width between opposing pairs of pixels. When a spike
        is present, this strategy may cause one side to get stuck in the
        spike while the opposing side walks.

        Therefore, when a spike is present, the spiked side walks while the
        other sideremains still.

        """
        FRACTION_WORM_SMOOTH = 1.0 / 12.0
        SMOOTHING_ORDER = 3
        PERCENT_BACK_SEARCH = 0.3
        PERCENT_FORWARD_SEARCH = 0.3
        END_S1_WALK_PCT = 0.15

        num_frames = len(h_ventral_contour)  # == len(h_dorsal_contour)

        h_skeleton = [None] * num_frames
        h_widths = [None] * num_frames

        profile_times = {'sgolay': 0,
                         'transpose': 0,
                         'h__getBounds': 0,
                         'compute_normal_vectors': 0,
                         'h__getMatches': 0,
                         'h__updateEndsByWalking': 0,
                         'hstack': 0,
                         'final calculations': 0}

        for frame_index, (s1, s2) in \
                enumerate(zip(h_ventral_contour, h_dorsal_contour)):
                    
            # If the frame has no contour values, assign no skeleton
            # or widths values
            if s1 is None:
                continue
            assert s1.shape[0] == 2 #x-y must be in the first dimension
            
            # Smoothing of the contour
            #------------------------------------------
            start = utils.timing_function()
            # Step 1: filter
            
            def _smooth_vec(sv):
                filter_width_sv = utils.round_to_odd(sv.shape[1] *
                                                     FRACTION_WORM_SMOOTH)
                try:
                    sv[0, :] = sgolay(sv[0, :], window_length=filter_width_sv,
                                      polyorder=SMOOTHING_ORDER)
                    sv[1, :] = sgolay(sv[1, :], window_length=filter_width_sv,
                                      polyorder=SMOOTHING_ORDER)
                except ValueError:
                    pass
                return sv
            
            s1 = _smooth_vec(s1)
            s2= _smooth_vec(s2)
            
            
            profile_times['sgolay'] += utils.timing_function() - start

            """
            # We require the contours to be coincident at the beginning
            # and end.  If they are not, add a middle point that joins them.
            if (s1[:,0] - s2[:,0] != 0).any():
                new_first_point = (s1[:,0] + s2[:,0]) / 2
                # We must have same dimensions to concatenate, so go from
                # shape (2,) to shape (2,1)
                new_first_point = new_first_point[:, None]
                # Now concatenate shape (2,1) with shape (2,k) along the
                # second axis, for a new array of shape (2,k+1)
                s1 = np.concatenate([new_first_point, s1], axis=1)
                s2 = np.concatenate([new_first_point, s2], axis=1)

            if (s1[:,-1] - s2[:,-1] != 0).any():
                new_last_point = (s1[:,-1] + s2[:,-1]) / 2
                # We must have same dimensions to concatenate, so go from
                # shape (2,) to shape (2,1)
                new_last_point = new_last_point[:, None]

                s1 = np.concatenate([s1, new_last_point], axis=1)
                s2 = np.concatenate([s2, new_last_point], axis=1)
            """

            # UP/DOWNSAMPLE if number of points is not betwen 49 and 250,
            # which seem like reasonable numbers.
            if s1.shape[1] < 49 or s1.shape[1] > 250:
                if s1.shape[1] < 49:
                    num_norm_points = 75
                else:
                    num_norm_points = 200
                # Upsample if we have too few points
                s1 = WormParserHelpers.normalize_all_frames_xy(
                    [s1], num_norm_points=num_norm_points)

                # There is only one frame so let's take that dimension out,
                # and transform s1 and s2 from having shape (k,2,n) to (k,2)
                s1 = s1[:, :, 0]

                # normalized_all_frames_xy rolls the axis so let's roll it back
                s1 = np.rollaxis(s1, 1)

            if s2.shape[1] < 49 or s2.shape[1] > 250:
                if s1.shape[1] < 49:
                    num_norm_points = 75
                else:
                    num_norm_points = 200
                # For documentation see the above for s1
                s2 = WormParserHelpers.normalize_all_frames_xy(
                    [s2], num_norm_points=num_norm_points)
                s2 = s2[:, :, 0]
                s2 = np.rollaxis(s2, 1)

            # Calculation of distances
            #-----------------------------------
            # Find the distance from each point in s1 to EVERY point in s2
            # Thus dx_across[0,5] gives the x-distance from point 0 on s1 to
            # point 5 on s2. The operation gives us an array of shape
            # (s1.shape[1],s2.shape[1])
            dx_across = np.transpose(s1[0, :][None, :]) - s2[0, :]
            dy_across = np.transpose(s1[1, :][None, :]) - s2[1, :]

            # d_across_partials has shape (ki, ji, 2)
            d_across_partials = np.dstack([dx_across, dy_across])
            d_across = np.linalg.norm(d_across_partials, axis=2)
            dx_across = dx_across / d_across
            dy_across = dy_across / d_across

            # Determine search bounds for possible "projection pairs"
            #------------------------------------------------
            start = utils.timing_function()
            left_indices, right_indices = \
                SkeletonCalculatorType1.h__getBounds(s1.shape[1],
                                                     s2.shape[1],
                                                     PERCENT_BACK_SEARCH,
                                                     PERCENT_FORWARD_SEARCH)

            profile_times['h__getBounds'] += utils.timing_function() - start
            start = utils.timing_function()

            # For each point on side 1, calculate normalized orthogonal values
            norm_x, norm_y = utils.compute_normal_vectors(s1)

            profile_times[
                'compute_normal_vectors'] += utils.timing_function() - start
            start = utils.timing_function()

            # For each point on side 1, find which side 2 the point pairs with
            match_I1 = SkeletonCalculatorType1.h__getMatches(s1, s2,
                                                             norm_x, norm_y,
                                                             dx_across,
                                                             dy_across,
                                                             d_across,
                                                             left_indices,
                                                             right_indices)

            profile_times['h__getMatches'] += utils.timing_function() - start
            start = utils.timing_function()

            # Pair off the points from one contour to the other
            I_1, I_2 = SkeletonCalculatorType1.h__updateEndsByWalking(
                d_across,
                match_I1,
                s1, s2,
                END_S1_WALK_PCT)

            profile_times[
                'h__updateEndsByWalking'] += utils.timing_function() - start

            # We're looking to the left and to the right to ensure that
            # things are ordered
            #                                    current is before next
            is_good = np.hstack((True, np.array((I_2[1:-1] <= I_2[2:]) & \
                                                # current after previous
                                                (I_2[1:-1] >= I_2[:-2])),
                                 True))
            # Filter out invalid entries
            I_1 = I_1[is_good]
            I_2 = I_2[is_good]

            # TODO: Allow smoothing on x & y

            # Create the skeleton sides
            s1 = s1[:, I_1]
            s1_p = s2[:, I_2]
            # The widths are simply the distance between the sides
            h_widths[frame_index] = np.linalg.norm(s1_p - s1, axis=0)
            # The skeleton is simply the midpoint between the sides
            h_skeleton[frame_index] = (s1 + s1_p) / 2

            # DEBUG
            # print("Final skeleton shape of frame %d: %s" %
            #      (frame_index, str(h_skeleton[frame_index].shape)))

            # DEBUG
            # Optional plotting code
            if frame_index in frames_to_plot:
                fig = plt.figure()
                # ARRANGE THE PLOTS AS:
                # AX1 AX1 AX2
                # AX1 AX1 AX3
                ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
                #ax2 = plt.subplot2grid((2,3), (0,2))
                ax3 = plt.subplot2grid((2, 3), (1, 2))
                ax1.set_title("Frame #%d of %d" % (frame_index,
                                                   len(h_ventral_contour)))

                # The points along one side of the worm
                ax1.scatter(s1[0, :], s1[1, :], marker='o',
                            edgecolors='r', facecolors='none')
                # The points along the other side
                ax1.scatter(s2[0, :], s2[1, :], marker='o',
                            edgecolors='b', facecolors='none')

                # To plot the widths, we need to run
                # plot([x1,x2],[y1,y2]), for each line segment
                for i in range(s1_p.shape[1]):
                    ax1.plot([s1_p[0, i], s1[0, i]], [s1_p[1, i], s1[1, i]],

                             # ax1.plot([s1_px[i], s1_x[i]], [s1_py[i],
                             # s1_y[i]],
                             color='g')

                skeleton = h_skeleton[frame_index]
                # The skeleton points
                ax1.scatter(skeleton[0, :], skeleton[1, :], marker='D',
                            edgecolors='b', facecolors='none')
                # The skeleton points, connected
                ax1.plot(skeleton[0, :], skeleton[1, :], color='navy')

                """
                # TODO: Jim's original method for plotting this was:
                # Width should really be plotted as a function of
                # distance along the skeleton
                cum_dist = h__getSkeletonDistance(skeleton_x, skeleton_y)

                ax2.plot(cum_dist./cum_dist[-1], h_widths[frame_index],
                         'r.-')
                hold on
                ax2.plot(np.linspace(0,1,49), nw_widths[:,iFrame], 'g.-')
                hold off
                """

                # Now let's plot each of the 200+ width values as the
                # y coordinate.
                ax3.set_title = "Worm width at each calculation point"
                ax3.set_xlabel("Calculation point")
                ax3.set_ylabel("Width")
                with plt.style.context('fivethirtyeight'):
                    ax3.plot(h_widths[frame_index], color='green',
                             linewidth=2)

                plt.show()

        # DEBUG
        # print(profile_times)
        return (h_widths, h_skeleton)

    #%%
    @staticmethod
    def h__getBounds(n1, n2, percent_left_search, percent_right_search):
        """
        Get contour "partner" search boundaries.

        Given two contour sides, having n1 and n2 points respectively, we
        want to pair off points on side n1 with points on side n2.  This
        method has a narrower goal: just to return the search boundaries
        for each point on n1 in its search for a match on the other side.

        For this, for each point i from 0 to n1-1 on the first side, we
        want start_indices[i] to be an integer giving the point index
        on the other side to start searching, and stop_indices[i], to be
        the final point searched on the other side.

        Parameters
        ---------------
        n1: int
            number of points along one side of the contour
        n2: int
            number of points along the other side of the contour
        percent_left_search: float
            percent to search backward
        percent_right_search: float
            percent to search forward

        Returns
        ---------------
        (start_indices, stop_indices): Two integer numpy arrays of shape (n1,)
            Giving the appropriate start and stop indices for a search
            through the other contour.  We only want to search for points
            along the opposing contour that are within
            [-percent_left_search, percent_right_search] of the point on
            the first contour.

        """
        # Create array of n1 evenly spaced numbers from 0 to 1 inclusive
        percentiles = np.linspace(0, 1, n1)
        start_percentiles = percentiles - percent_left_search
        stop_percentiles = percentiles + percent_right_search

        # An integer array giving the leftmost to navigate before stopping,
        # for each point along n2.
        start_indices = np.floor(start_percentiles * n2)
        stop_indices = np.ceil(stop_percentiles * n2)
        # Truncate any indices pointing outside the range between 0 and n2-1
        start_indices[start_indices < 0] = 0
        stop_indices[stop_indices >= n2] = n2 - 1

        return start_indices.astype(np.int), stop_indices.astype(np.int)

    #%%
    @staticmethod
    def h__getMatches(s1, s2,
                      norm_x, norm_y,
                      dx_across, dy_across, d_across,
                      left_I, right_I):
        """
        For a given frame,
        For each point on side 1, find which side 2 the point pairs with

        Parameters
        ---------------
        s1: list of numpy arrays, with the arrays having shape (2,ki)
            One side of the contour.  ki is the number of points in frame i
        s2: list of numpy arrays, with the arrays having shape (2,ji)
            The other side of the contour.  ji is the number of points in
            frame i
        norm_x:
        norm_y:
        dx_across:
        dy_across:
        d_across: 2d numpy array of shape (ki, ji)
            A lookup table giving the distance from a point on one
            of the contour to any point on the other side.
        left_I:
        right_I:

        Returns
        ---------------
        match_indices: numpy array of integers, of shape (ki) where
            ki is the number of contour points in s1

        """
        n_s1 = s1.shape[1]
        match_I = np.zeros(n_s1, dtype=int)
        match_I[0] = 0
        match_I[-1] = s2.shape[1]

        dp_values = np.zeros(n_s1)
        all_signs_used = np.zeros(n_s1)

        # There is no need to do the first and last point
        for I, (lb, rb) in enumerate(zip(left_I[1:-1], right_I[1:-1])):
            I = I + 1
            [abs_dp_value, dp_I, sign_used] = SkeletonCalculatorType1.\
                h__getProjectionIndex(norm_x[I], norm_y[I],
                                      dx_across[I, lb:rb], dy_across[I, lb:rb],
                                      lb,
                                      d_across[I, lb:rb], 0)
            all_signs_used[I] = sign_used
            dp_values[I] = abs_dp_value
            match_I[I] = dp_I

        if not np.all(all_signs_used[1:-1] == all_signs_used[1]):
            if np.sum(all_signs_used) > 0:
                sign_use = 1
                I_bad = utils.find(all_signs_used[1:-1] != 1) + 1
            else:
                I_bad = utils.find(all_signs_used[1:-1] != -1) + 1
                sign_use = -1

            for I in I_bad:
                lb = left_I[I]
                rb = right_I[I]
                [abs_dp_value, dp_I, sign_used] = SkeletonCalculatorType1.\
                    h__getProjectionIndex(norm_x[I], norm_y[I],
                                          dx_across[I, lb:rb], dy_across[I, lb:rb],
                                          lb,
                                          d_across[I, lb:rb], sign_use)
                all_signs_used[I] = sign_used
                dp_values[I] = abs_dp_value
                match_I[I] = dp_I

        return match_I

    #%%
    @staticmethod
    def h__getProjectionIndex(vc_dx_ortho, vc_dy_ortho,
                              dx_across_worm, dy_across_worm,
                              left_I, d_across, sign_use):
        """

        Matlab code:
        nvc_local = nvc(nvc_indices_use,:);

        dx_across_worm = cur_point(1) - nvc_local(:,1);
        dy_across_worm = cur_point(2) - nvc_local(:,2);

        d_magnitude = sqrt(dx_across_worm.^2+dy_across_worm.^2);

        dx_across_worm = dx_across_worm./d_magnitude;
        dy_across_worm = dy_across_worm./d_magnitude;

        """

        # SPEED: Compute normalized distances for all pairs ...
        # Might need to downsample

        dp = dx_across_worm * vc_dx_ortho + dy_across_worm * vc_dy_ortho

        # I'd like to not have to do this step, it has to do with
        # the relationship between the vulva and non-vulva side. This
        # should be consistent across the entire animal and could be
        # passed in, unless the worm rolls.
        sign_used = -1
        if sign_use == 0 and np.sum(dp) > 0:
            # Instead of multiplying by -1 we could hardcode the flip of
            # the logic below (e.g. max instead of min, > vs <)
            dp = -1 * dp
            sign_used = 1
        elif sign_use == 1:
            dp = -1 * dp
            sign_used = 1

        # This is slow, presumably due to the memory allocation ...
        #                   < right
        # possible = [dp(1:end-1) < dp(2:end) false] & \
        #                   < left
        #           [false dp(2:end) < dp(1:end-1)]

        # In Matlab:
        # possible = (dp(2:end-1) < dp(3:end)) & (dp(2:end-1) < dp(1:end-2));
        possible = (dp[1:-2] < dp[2:-1]) & (dp[1:-2] < dp[0:-3])

        Ip = utils.find(possible)
        if len(Ip) == 1:
            dp_I = Ip + 1
            dp_value = dp[dp_I]
        elif len(Ip) > 1:
            temp_I = np.argmin(d_across[Ip])
            dp_I = Ip[temp_I] + 1
            dp_value = dp[dp_I]
        else:
            dp_I = np.argmin(dp)
            dp_value = dp[dp_I]

        I = left_I + dp_I

        return (dp_value, I, sign_used)

    #%%
    @staticmethod
    def h__updateEndsByWalking(d_across, match_I1, s1, s2, END_S1_WALK_PCT):
        """
        Update ends by walking.

        Parameters
        ----------
        d_across: 2d numpy array of shape (ki, ji)
            A lookup table giving the distance from a point on one
            of the contour to any point on the other side.
        match_I1: numpy array of shape (ki,)
            current list of matches
        s1: list of numpy arrays, with the arrays having shape (2,ki)
            One side of the contour.  ki is the number of points in frame i
        s2: list of numpy arrays, with the arrays having shape (2,ji)
            The other side of the contour.  ji is the number of points in
            frame i
        END_S1_WALK_PCT: float
            At what % to end the walk along the first contour

        Returns
        -------
        (I_1, I_2): tuple of numpy arrays

        """
        n_s1 = s1.shape[1]
        n_s2 = s2.shape[1]

        end_s1_walk_I = np.ceil(n_s1 * END_S1_WALK_PCT)
        end_s1_walk_I = end_s1_walk_I.astype(np.int)
        end_s2_walk_I = 2 * end_s1_walk_I
        
        p1_I, p2_I = SkeletonCalculatorType1.h__getPartnersViaWalk(
            0, end_s1_walk_I,
            0, end_s2_walk_I,
            d_across,
            s1, s2)
        
        

        # Alter the matches somewhat
        match_I1[p1_I] = p2_I

        # Keep all our alterations
        keep_mask = np.zeros(len(match_I1), dtype=np.bool)
        keep_mask[p1_I] = True

        # Add
        end_s1_walk_backwards = n_s1 - end_s1_walk_I + 1
        end_s2_walk_backwards = n_s2 - end_s2_walk_I + 1

        p1_I, p2_I = SkeletonCalculatorType1.h__getPartnersViaWalk(
            n_s1 - 1, end_s1_walk_backwards,
            n_s2 - 1, end_s2_walk_backwards,
            d_across,
            s1, s2)

        match_I1[p1_I] = p2_I
        keep_mask[p1_I] = True

        # Anything in between we'll use the projection approach
        keep_mask[end_s1_walk_I + 1:end_s1_walk_backwards] = True

        # Always keep ends
        keep_mask[0] = True
        keep_mask[-1] = True

        match_I1[0] = 0
        match_I1[-1] = n_s2 - 1

        # This isn't perfect but it removes some back and forth behavior
        # of the matching. We'd rather drop points and smooth
        I_1 = np.flatnonzero(keep_mask)
        I_2 = match_I1[keep_mask]

        return (I_1, I_2)

    #%%
    @staticmethod
    def h__getPartnersViaWalk(s1, e1, s2, e2, d, xy1, xy2):
        """

        Intro
        -----
        This is is an implentation of:
        https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/
        %2Bseg_worm/%2Bcv/skeletonize.m

        In the SegWorm code this is the main algorithm for going from a
        contour to a skeleton (and widths). However in this implementation
        it only gets called for the head and tail. The original implementation
        calls this code many times after dividing the worm up into chunks
        based on finding large bend angles and the midpoint of the worm.

        Why used here
        -------------
        This is used at the ends rather than the projection as the projection
        is looking for a line that is orthogonal to the slope of the contour
        that makes a nice bisection of the worm. However at the ends this
        falls apart. Consider a worm that is slighly like a diamond and how
        the lines that are orthgonal to the end points form horrible lines
        for widths estimation.


        Example contour of the head or tail with a bit of the body:

              /\
             /\ \
            /  \ \
           |    \ |
           |     \| #Bad
           |------| #Good

        The hypen line  is orthogonal to the left contour and provides a
        good estimate of a line that is orthogonal to the skeleton.

        The inner diagonal line is orthogonal to the left contour but is
        a poor choice for a line that is orthogonal to the skeleton.

        Algorithm
        ---------
        More on this can be found in Ev Yemini's thesis. The basic idea is
        we start with pairs on both sides of the contour and ask whether or
        not each point on one side of the contour should partner with its
        current point or the next point on the other side, given the widths
        between them. Parterning

        Parameters
        ----------
        s1: start index for side 1
        e1: end index for side 1 (inclusive)
        s2: start index for side 2
        e2: end index for side 2 (inclusive)
        d: distance from I1 to I2 is d(I1,I2)

        Returns
        -------
        (p1_I, p2_I) tuple
            p1_I: numpy array
                Each element represents one of a pair of indices that
                belong together. In other words p1_I[i] goes with p2_I[i]

        """

        # TODO: remove hardcode, base on max of e1-s1+1
        p1_I = np.zeros(200, dtype=np.int)
        p2_I = np.zeros(200, dtype=np.int)

        c1 = s1  # Current 1 index
        c2 = s2  # Current 2 index
        cur_p_I = -1  # Current pair index

        while c1 != e1 and c2 != e2:
            cur_p_I += 1

            # We are either going up or down based on which end we are
            # starting from (beggining or end)
            if e1 < s1:
                next1 = c1 - 1
                next2 = c2 - 1
            else:
                next1 = c1 + 1
                next2 = c2 + 1

            # JAH: At this point
            # Need to handle indexing () vs [] and indexing spans (if any)
            # as well as 0 vs 1 based indexing (if any)
            try:
                v_n1c1 = xy1[:, next1] - xy1[:, c1]
            except:
                import pdb
                pdb.set_trace()

            v_n2c2 = xy2[:, next2] - xy2[:, c2]

            # 216,231
            d_n1n2 = d[next1, next2]
            d_n1c2 = d[next1, c2]
            d_n2c1 = d[c1, next2]

            if d_n1c2 == d_n2c1 or (d_n1n2 <= d_n1c2 and d_n1n2 <= d_n2c1):
                # Advance along both contours

                p1_I[cur_p_I] = next1
                p2_I[cur_p_I] = next2

                c1 = next1
                c2 = next2

            elif np.sum((v_n1c1 * v_n2c2) > 0):
                # Contours go similar directions
                # Follow smallest width
                if d_n1c2 < d_n2c1:
                    # Consume smaller distance, then move the base of the
                    # vector further forward
                    p1_I[cur_p_I] = next1
                    p2_I[cur_p_I] = c2

                    # This bit always confuses me
                    # c1  n1
                    #
                    #
                    # c2  x  x  x  n2
                    #
                    # Advance c1 so that d_n2_to_c1 is smaller next time
                    c1 = next1
                else:
                    p1_I[cur_p_I] = c1
                    p2_I[cur_p_I] = next2
                    c2 = next2
            else:

                if cur_p_I == 1:
                    prev_width = 0
                else:
                    prev_width = d[p1_I[cur_p_I - 1], p2_I[cur_p_I - 1]]

                if (d_n1c2 > prev_width and d_n2c1 > prev_width):
                    p1_I[cur_p_I] = next1
                    p2_I[cur_p_I] = next2

                    c1 = next1
                    c2 = next2
                elif d_n1c2 < d_n2c1:
                    p1_I[cur_p_I] = next1
                    p2_I[cur_p_I] = c2
                    c1 = next1
                else:
                    p1_I[cur_p_I] = c1
                    p2_I[cur_p_I] = next2
                    c2 = next2

        p1_I = p1_I[:cur_p_I]
        p2_I = p2_I[:cur_p_I]

        return (p1_I, p2_I)
        #p1_I[cur_p_I+1:] = []
        #p2_I[cur_p_I+1:] = []
