# -*- coding: utf-8 -*-
"""
This module handles the base (generic) Feature code that the actual
computed features inherit from.

Jim's notes on problematic features:
- missing dependency
- empty video (Let's not deal with this for now)
- no events
- missing_from_disk    #i.e. not present in loaded data
See:
    - locomotion_features.AverageBodyAngle
    - locomotion_features.MidbodyVelocityDistance
"""

from .. import utils

import numpy as np
import copy
import re

# Get everything until a period that is followed by no periods
#
# e.g. from:
#   temp.name.prop    TO
#   temp.name
_parent_feature_pattern = re.compile('(.*)\.([^\.]+)')


class Feature(object):

    """
    This is the parent class from which all features inherit.

    Unfortunately, with the current setup some of these features are
    populated in the Specs.compute_feature()

    Attributes
    ----------
    name :
    is_temporary :
    spec : worm_features.FeatureProcessingSpec
    value :
        This is the value of interest. This generally does not exist for
        features that are temporary
    dependencies : list
    missing_dependency :
        Unable to compute feature due to a missing dependency
    missing_from_disk :
        Unable to load the feature as it was not saved in the loaded file
    empty_video :
        Indicates that there was no data in the processed video. Note, I'm
        not sure that we'll use this one.
    no_events :
        Indicates that no events were observed in the video.



    Temporary features may have additional attributes that are essentially
    the values of the feature of interest. The child features grab these
    attributes and populate them in their 'value' attributes.
    TODO: provide example of what I mean by this

    See Also
    --------
    worm_features.FeatureProcessingSpec.compute_feature()

    """

    def __repr__(self):
        return utils.print_object(self)

    # This should be part of the features
    def __eq__(self, other):
        # TODO: Figure out how to pass parameters into this
        # We should probably just overload it ...
        return utils.correlation(self.value, other.value, self.name)

    def get_feature(self, wf, feature_name):
        """
        This was put in to do anything we need when getting a feature
        rather than calling the feature directly.

        For example, right now we'll log dependencies.

        Note, the dependencies are not recursive (i.e. we don't load in the
        dependencies of the featurew we are requesting)
        """

        # 1) Do logging - NYI
        # What is the easiest way to initialize without forcing a init super call?
        # NOTE: We could also support returning all depdencies, in which we get
        # the dependencies of the parent and add those as well
        if hasattr(self, 'dependencies'):
            self.dependencies.append(feature_name)
        else:
            self.dependencies = [feature_name]

        # 2) Make the call to WormFeatures
        # Note, we call wf.get_features rather than the spec to ensure that wf
        # is aware of the new features that have been computed
        return wf._get_and_log_feature(feature_name, internal_request=True)

    @property
    def is_valid(self):
        """
        Note, the properties here are currently assigned in the spec
        and dependent on the time of debugging may not exist.
        We might want to wrap with a try/catch
        """
        return not self.missing_from_disk and not self.missing_dependency

    @property
    def has_data(self):
        return self.is_valid and not self.no_events and not self.empty_video

    def copy(self):
        # TODO: We might want to do something special for value

        new_self = self.__new__(self.__class__)
        d = self.__dict__
        for key in d:
            temp = d[key]
            if key == 'spec':
                setattr(new_self, 'spec', temp.copy())
            else:
                setattr(new_self, key, copy.copy(temp))

        return new_self
        # return copy.copy(self)


def get_parent_feature_name(feature_name):
    """
    Go from something like:
    locomotion.crawling_bends.head.amplitude
    TO
    locomotion.crawling_bends.head
    """

    return get_feature_name_info(feature_name)[0]


def get_feature_name_info(feature_name):
    """
    Outputs
    -------
    (parent_name,specific_feature_name)
    """
    # We could make this more obvious by using split ...
    # I might want to also remove the parens and only get back the 1st string somehow
    # 0 - the entire match
    # 1 - the first parenthesized subgroup

    result = _parent_feature_pattern.match(feature_name)
    return result.group(1), result.group(2)

# How are we going to do from disk?
#
# 1) Need some reference to a saved file
# 2)

#    @classmethod
#    def from_disk(cls,width_ref):
#
#        self = cls.__new__(cls)
#
#        for partition in self.fields:
#            widths_in_partition = utils._extract_time_from_disk(width_ref,
#                                                                partition)
#            setattr(self, partition, widths_in_partition)

# event_durations
# distance_during_events
# time_between_events
# distance_between_events
# frequency
# time_ratio
# data_ratio


def get_event_attribute(event_object, attribute_name):

    # We might want to place some logic in here

    if event_object.is_null:
        return None
    else:
        return getattr(event_object, attribute_name)


class EventFeature(Feature):
    """
    This covers features that come from events. This is NOT the temporary
    event feature parent.

    TODO: Insert example

    temp main event list:
    locomotion_features.MotionEvent
    locomotion_turns.(upsilon and omega)
    posture_features.Coils

    Attributes
    ----------


    """

    def __init__(self, wf, feature_name):

        # This is a bit messy :/
        # We might want to eventually obtain this some other way
        cur_spec = wf.specs[feature_name]

        self.name = feature_name
        event_name, feature_type = get_feature_name_info(feature_name)
        event_name = get_parent_feature_name(feature_name)

        temp_parent_feature = self.get_feature(wf, event_name)

        if temp_parent_feature.no_events:
            self.no_events = True
            self.keep_mask = None
            self.value = None
            return

        # TODO: I'd like a better name for this
        #---------------------------
        # event_parent?
        # event_main?
        event_value = temp_parent_feature.value
        # event_value : EventListWithFeatures

        self.value = get_event_attribute(event_value, feature_type)
        start_frames = get_event_attribute(event_value, 'start_frames')
        end_frames = get_event_attribute(event_value, 'end_frames')

        try:
            self.num_video_frames = event_value.num_video_frames
        except:
            import pdb
            pdb.set_trace()

        # event_durations - filter on starts and stops
        #distance_during_events - "  "
        # time_between_events - filter on the breaks
        # distance_between_events - filter on the breaks
        #
        # Ideally

        # TODO: I think we should hide the 'keep_mask' => '_keep_mask'

        # TODO: The filtering should maybe be identified by type
        #-------------------------
        #  event-main - summary of an event itself
        #  event-inter - summary of something between events
        #  event-summary - summary statistic over all events
        # or something like this ...

        # TODO: This is different behavior than the original
        # In the original the between events were filtered the same as the 1st
        # event. In other words, if the 1st event was a partial, the time
        # between the 1st and 2nd event was considered a partial
        #
        # 1) Document difference
        # 2) Build in temporary support for the old behavior flag

        # print(cur_spec.name)

        # This will eventually be removed when we move to empty_features
        if self.value is None or isinstance(
                self.value, float) or self.value.size == 0:
            self.keep_mask = None
            self.signing_mask = None
        else:
            if cur_spec.is_signed:
                signing_field_name = cur_spec.signing_field
                signing_mask = get_event_attribute(
                    event_value, signing_field_name)
                if feature_type in [
                        'event_durations',
                        'distance_during_events']:
                    self.signing_mask = signing_mask
                else:
                    # TODO: We should check on scalar vs inter-event here
                    # but only inter-events are signed
                    #
                    # Signing them makes little sense and should be removed
                    #
                    # This is the old behavior
                    # Signing the inter-events doesn't make much sense
                    #
                    # See note below regarding inter-events and their relation to
                    # events
                    # This behavior signs the interevent based on the proceeding
                    # event since I'm 99% sure you only ever have:
                    #   - event, interevent, event AND NOT
                    #   - interevent, event, interevent OR
                    #   - event, interevent, etc.
                    #
                    #   i.e. only have interevents between events
                    self.signing_mask = signing_mask[0:-1]
            else:
                self.signing_mask = None

            self.keep_mask = np.ones(self.value.shape, dtype=bool)
            if feature_type in ['event_durations', 'distance_during_events']:
                self.keep_mask[0] = start_frames[0] != 0
                self.keep_mask[-1] = end_frames[-1] != (
                    self.num_video_frames - 1)
            elif feature_type in ['time_between_events', 'distance_between_events']:

                # TODO: Verify that inter-events can be partial ...
                # i.e. if an event starts at frame 20, verify that we have an
                # inter-event from frames 1 - 19
                #
                #   Document this result (either way)
                #
                # I think we only ever include inter-event values that are
                # actually between events ...

                # First is partial if the main event starts after the first
                # frame
                self.keep_mask[0] = start_frames[0] == 0
                # Similarly, if the last event ends before the end of the
                # video, then anything after that is partial
                self.keep_mask[-1] = end_frames[-1] == (
                    self.num_video_frames - 1)
            else:
                # Should be a scalar value
                # e.g. => frequency
                self.keep_mask = np.ones(1, dtype=bool)

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        return cls(wf, feature_name)

    # TODO: Figure out how to make sure that on copy
    # we adjust the mask if only full_values are set ...
    # def copy(self):

    def __eq__(self, other):
        # TODO: We need to implement this ...
        # scalars - see if they are close, otherwise call super?
        return True

    def get_value(self, partials=False, signed=True):
        # TODO: Document this function
        if not self.has_data:
            return None

        temp_values = self.value
        if signed and (self.signing_mask is not None):
            # TODO: Not sure if we multiply by -1 for True or False
            temp_values[self.signing_mask] = - \
                1 * temp_values[self.signing_mask]

        if partials:
            temp_values = temp_values[self.keep_mask]

        return temp_values

# We might want to make things specific again but for now we'll use
# a single class

# class EventDuration(Feature):
#
#    def __init__(self, wf, feature_name):
#        parent_name = get_parent_feature_name(feature_name)
#        temp = wf[event_name]
#        self.value = get_event_attribute(temp.value,'event_durations')
#        self.name = event_name + '.event_durations'
#
# class DistanceDuringEvents(Feature):
#
#    def __init__(self,wf,event_name):
#        temp = wf[event_name]
#        self.value = get_event_attribute(temp.value,'distance_during_events')
#        self.name = event_name + '.distance_during_events'
#
# class TimeBetweenEvents(Feature):
#
#    def __init__(self,wf,event_name):
#        temp = wf[event_name]
#        self.value = get_event_attribute(temp.value,'time_between_events')
#        self.name = event_name + '.time_between_events'
#
# class DistanceBetweenEvents(Feature):
#
#    def __init__(self,wf,event_name):
#        temp = wf[event_name]
#        self.value = get_event_attribute(temp.value,'distance_between_events')
#        self.name = event_name + '.distance_between_events'
#
# class Frequency(Feature):
#
#    def __init__(self,wf,event_name):
#        temp = wf[event_name]
#        self.value = get_event_attribute(temp.value,'frequency')
#        self.name = event_name + '.frequency'
#
# class EventTimeRatio(Feature):
#
#    def __init__(self,wf,event_name):
#        temp = wf[event_name]
#        self.value = get_event_attribute(temp.value,'time_ratio')
#        self.name = event_name + '.time_ratio'
#
# class EventDataRatio(Feature):
#
#    def __init__(self,wf,event_name):
#        temp = wf[event_name]
#        self.value = get_event_attribute(temp.value,'data_ratio')
#        self.name = event_name + '.data_ratio'
