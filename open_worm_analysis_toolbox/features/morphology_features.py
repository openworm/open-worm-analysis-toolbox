# -*- coding: utf-8 -*-
"""
morphology_features.py

"""
import numpy as np

from .generic_features import Feature
from .. import utils


class Widths(object):
    """
    This is no longer used in the new code and can be deleted when ready

    Attributes
    ----------
    head :
    midbody :
    tail :
    """

    fields = ('head', 'midbody', 'tail')

    def __init__(self, features_ref):
        """
        Parameters
        ----------
        features_ref : WormFeatures instance

        Note the current approach just computes the mean of the
        different body section widths. Eventually this should be
        computed in this class.

        """
        nw = features_ref.nw

        for partition in self.fields:
            widths_in_partition = nw.get_partition(partition, 'widths')
            setattr(self, partition, np.mean(widths_in_partition, 0))

    @classmethod
    def from_disk(cls, width_ref):

        self = cls.__new__(cls)

        for partition in self.fields:
            widths_in_partition = utils._extract_time_from_disk(width_ref,
                                                                partition)
            setattr(self, partition, widths_in_partition)

        return self

    def __eq__(self, other):
        return (
            utils.correlation(self.head, other.head,
                              'morph.width.head') and
            utils.correlation(self.midbody, other.midbody,
                              'morph.width.midbody') and
            utils.correlation(self.tail, other.tail,
                              'morph.width.tail'))

    def __repr__(self):
        return utils.print_object(self)

#======================================================================
#                       NEW CODE
#======================================================================
#
#
#   Still need to handle equality comparison and loading from disk


class Length(Feature):

    def __init__(self, wf, feature_name):
        self.name = feature_name
        self.value = wf.nw.length

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        self = cls.__new__(cls)
        self.name = feature_name
        self.value = utils.get_nested_h5_field(wf.h, ['morphology', 'length'])
        return self


class WidthSection(Feature):
    """
    This should only be called by the subclasses

    Attributes
    ----------
    partition_name
    """

    def __init__(self, wf, feature_name, partition_name):
        """
        Parameters
        ----------
        wf : WormFeatures instance
        """

        self.name = feature_name
        self.partition_name = partition_name
        # I'm not thrilled with the name of this method
        widths_in_partition = wf.nw.get_partition(partition_name, 'widths')
        self.value = np.mean(widths_in_partition, 0)

    @classmethod
    def from_schafer_file(cls, wf, feature_name, partition_name):
        self = cls.__new__(cls)
        self.name = feature_name
        self.value = utils.get_nested_h5_field(
            wf.h, ['morphology', 'width', partition_name])
        return self


class Area(Feature):
    """
    Feature: morphology.area
    """

    def __init__(self, wf, feature_name):
        self.name = feature_name
        self.value = wf.nw.area
        # We should instead be using this, NYI
        #self.value = self.get_feature(wf,'nw.area').value

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        self = cls.__new__(cls)
        self.name = feature_name
        self.value = utils.get_nested_h5_field(wf.h, ['morphology', 'area'])
        return self


class AreaPerLength(Feature):
    """
    Feature: morphology.area_per_length
    """

    def __init__(self, wf, feature_name):
        self.name = feature_name
        area = self.get_feature(wf, 'morphology.area').value
        length = self.get_feature(wf, 'morphology.length').value
        self.value = area / length

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        return cls(wf, feature_name)


class WidthPerLength(Feature):
    """
    Feature: morphology.width_per_length
    """

    def __init__(self, wf, feature_name):
        self.name = feature_name
        width = self.get_feature(wf, 'morphology.width.midbody').value
        length = self.get_feature(wf, 'morphology.length').value
        self.value = width / length

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        return cls(wf, feature_name)
