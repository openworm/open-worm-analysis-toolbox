import h5py
import numpy as np
import six  # For compatibility with Python 2.x
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .. import utils
from ..features.worm_features import WormFeatures

from .histogram import Histogram, MergedHistogram
from .specifications import SimpleSpecs, EventSpecs, MovementSpecs


class BasePlot(object):

    def __init__(self, worm):
        if isinstance(worm, str):
            self._get_worm_from_file(worm)
        else:
            self.worm = worm

    def _get_worm_from_file(self, worms):
        self.worm = WormFeatures.from_disk(worm)


class BasePointsPlot(BasePlot):

    def __init__(self, worm):
        super(BasePointsPlot, self).__init__(worm)
        self.points = {'x': list(), 'y': list()}
        self._get_points()

    def _set_location_from_frame(self, frame):
        self.points['x'].append(self.worm.path.coordinates.x[frame])
        self.points['y'].append(self.worm.path.coordinates.y[frame])

    def _get_points(self):
        pass

    def plot_points(self, ax, size, color, symbol):
        ax.scatter(
            self.points['x'],
            self.points['y'],
            s=size,
            c=color,
            marker=symbol)


class UpsilonPointsPlot(BasePointsPlot):

    def _get_points(self):
        for frame in self.worm.locomotion.turns.upsilons.start_frames:
            self._set_location_from_frame(frame)

    def plot_points(self, ax, size=120, color='brown', symbol='o'):
        ax.scatter(
            self.points['x'],
            self.points['y'],
            s=size,
            c=color,
            marker=symbol)


class OmegaPointsPlot(BasePointsPlot):

    def _get_points(self):
        for frame in self.worm.locomotion.turns.omegas.start_frames:
            self._set_location_from_frame(frame)

    def plot_points(self, ax, size=240, color='brown', symbol='x'):
        ax.scatter(
            self.points['x'],
            self.points['y'],
            lw=6,
            s=size,
            c=color,
            marker=symbol)


class CoilPointsPlot(BasePointsPlot):

    def _get_points(self):
        for frame in self.worm.posture.coils.start_frames:
            self._set_location_from_frame(frame - 5)

    def plot_points(self, ax, size=240, color='brown', symbol='+'):
        ax.scatter(
            self.points['x'],
            self.points['y'],
            lw=6,
            s=size,
            c=color,
            marker=symbol)


class StartPointsPlot(BasePointsPlot):

    def _get_points(self):
        self.points['x'] = self.worm.posture.skeleton.x[:, 0]
        self.points['y'] = self.worm.posture.skeleton.y[:, 0]

    def plot_points(self, ax, size=100, color='gray', symbol='o'):
        ax.scatter(
            self.points['x'],
            self.points['y'],
            s=size,
            c=color,
            marker=symbol)


class EndPointsPlot(BasePointsPlot):

    def _get_points(self):
        self.points['x'] = self.worm.posture.skeleton.x[:, -1]
        self.points['y'] = self.worm.posture.skeleton.y[:, -1]

    def plot_points(self, ax, size=100, color='black', symbol='o'):
        ax.scatter(
            self.points['x'],
            self.points['y'],
            s=size,
            c=color,
            marker=symbol)


class BasePathPlot(BasePlot):

    def __init__(self, worm):
        super(BasePathPlot, self).__init__(worm)
        self.path = {'x': list(), 'y': list()}
        self._get_path()

    def _get_path(self):
        self.path['x'] = self.worm.path.coordinates.x
        self.path['y'] = self.worm.path.coordinates.y

    def plot_path(self, ax):
        ax.plot(self.path['x'], self.path['y'])


class MidbodyPathPlot(BasePathPlot):

    def _get_path(self):
        self.path['x'] = np.nanmean(
            self.worm.posture.skeleton.x[
                16:33], axis=0)
        self.path['y'] = np.nanmean(
            self.worm.posture.skeleton.y[
                16:33], axis=0)

    def plot_path(self, ax, color='chartreuse'):
        ax.plot(self.path['x'], self.path['y'], c=color)


class HeadPathPlot(BasePathPlot):

    def _get_path(self):
        self.path['x'] = np.nanmean(self.worm.posture.skeleton.x[0:8], axis=0)
        self.path['y'] = np.nanmean(self.worm.posture.skeleton.y[0:8], axis=0)

    def plot_path(self, ax, color='fuchsia'):
        ax.plot(self.path['x'], self.path['y'], c=color)


class TailPathPlot(BasePathPlot):

    def _get_path(self):
        self.path['x'] = np.nanmean(
            self.worm.posture.skeleton.x[
                41:49], axis=0)
        self.path['y'] = np.nanmean(
            self.worm.posture.skeleton.y[
                41:49], axis=0)

    def plot_path(self, ax, color='royalblue'):
        ax.plot(self.path['x'], self.path['y'], c=color)


class VelocityPathPlot(BasePathPlot):

    def _get_path(self):
        self.path['x'] = self.worm.path.coordinates.x
        self.path['y'] = self.worm.path.coordinates.y
        raw_colors = np.nan_to_num(self.worm.locomotion.velocity.midbody.speed)
        max_color = max(raw_colors)
        self.colors = raw_colors / max_color

    def plot_path(self, ax):
        ax.scatter(
            self.path['x'],
            self.path['y'],
            c=self.colors,
            cmap='RdYlGn',
            lw=0,
            marker='.')


def test_run():
    worm_file_path = "/Users/chris/Google Drive/example_data/30m_wait/L/tracker_1/2012-03-08___15_42_48/483 AQ2947 on food R_2012_03_08__15_42_48___1___8_features.mat"

    worm = WormFeatures.from_disk(worm_file_path)

    f, ((path, velocity, na), (head, mid, tail)) = plt.subplots(2, 3)

    sp = StartPointsPlot(worm)
    ep = EndPointsPlot(worm)
    up = UpsilonPointsPlot(worm)
    op = OmegaPointsPlot(worm)
    cp = CoilPointsPlot(worm)
    pp = BasePathPlot(worm)
    hp = HeadPathPlot(worm)
    mp = MidbodyPathPlot(worm)
    tp = TailPathPlot(worm)
    vp = VelocityPathPlot(worm)

    path.set_title("Path Plot")
    velocity.set_title("Velocity Plot")
    head.set_title("Path Plot Head")
    mid.set_title("Path Plot Midbody")
    tail.set_title("Path Plot Tail")

    pp.plot_path(path)
    up.plot_points(path)
    op.plot_points(path)
    cp.plot_points(path)
    sp.plot_points(path)
    ep.plot_points(path)

    up.plot_points(velocity)
    op.plot_points(velocity)
    cp.plot_points(velocity)
    sp.plot_points(velocity)
    ep.plot_points(velocity)
    vp.plot_path(velocity)

    hp.plot_path(head)
    up.plot_points(head)
    op.plot_points(head)
    cp.plot_points(head)
    sp.plot_points(head)
    ep.plot_points(head)

    mp.plot_path(mid)
    up.plot_points(mid)
    op.plot_points(mid)
    cp.plot_points(mid)
    sp.plot_points(mid)
    ep.plot_points(mid)

    tp.plot_path(tail)
    up.plot_points(tail)
    op.plot_points(tail)
    cp.plot_points(tail)
    tp.plot_path(tail)
    sp.plot_points(tail)
    ep.plot_points(tail)

    f.text(0, 1, "Omega Turns X")
    f.text(0, 0.5, "Coils +")

    plt.show()


if __name__ == "__main__":
    test_run()
