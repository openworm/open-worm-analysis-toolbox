[![GitHub license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE.md)
[![Travis-CI](https://travis-ci.org/openworm/open-worm-analysis-toolbox.svg?branch=master)](https://travis-ci.org/openworm/open-worm-analysis-toolbox)
[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/openworm/open-worm-analysis-toolbox?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![BountySource](https://api.bountysource.com/badge/team?team_id=23852)](https://www.bountysource.com/teams/openworm)
[![Waffle](https://badge.waffle.io/openworm/open-worm-analysis-toolbox.png?label=ready&title=Ready)](https://waffle.io/openworm/open-worm-analysis-toolbox)


| <img src="OpenWorm%20Analysis%20Toolbox%20logo.png" width="125"> | Open Worm Analysis Toolbox |
====================

The **Open Worm Analysis Toolbox** is a Python port of the Schafer Lab's [Worm Analysis Toolbox 1.3.4](http://www.mrc-lmb.cam.ac.uk/wormtracker/index.php?action=analysis).

It can be used to process videos of *C. elegans* into statistics so the behaviour of individual worms can be compared.

It is also the package used by the OpenWorm project to determine how closely its simulated worm behaves like real worms. It was started as a sub-project of the [OpenWorm project](https://github.com/openworm).

[OWAT is on PyPI](https://pypi.python.org/pypi/open_worm_analysis_toolbox), so to install, simply type:

```
pip install open_worm_analysis_toolbox
```

Contributors please see:

-   [Installation
    Guide](INSTALL.md)
-   [Guide for
    Contributors](documentation/Guide%20for%20contributors.md)
-   [Kanban Board with current GitHub
    Issues](https://waffle.io/openworm/open-worm-analysis-toolbox)
-   [Movement Validation Cloud](https://github.com/openworm/movement_validation_cloud) - Code for running this package on the cloud via Amazon Web Services

Usage Example
-------------

```Python
import open_worm_analysis_toolbox as mv

# Load a "basic" worm from a file
bw = mv.BasicWorm.from_schafer_file_factory("example_contour_and_skeleton_info.mat")
# Normalize the basic worm
nw = mv.NormalizedWorm.from_BasicWorm_factory(bw)
# Plot this normalized worm    
wp = mv.NormalizedWormPlottable(nw, interactive=False)
wp.show()
# Obtain features
wf = mv.WormFeatures(nw)
```

Later, if we have control worms, we can run statistics on our worm:

```Python
# Compute histograms
experiment_histograms = mv.HistogramManager([wf, wf])
control_histograms = mv.HistogramManager(control_worms)

# Compute statistics
stat = mv.StatisticsManager(experiment_histograms, control_histograms)

# Plot statistics for the first extended feature
stat[0].plot(ax=None, use_alternate_plot=True)

# Give an overall assessment of the worm's similarity to the control set
print("Nonparametric p and q values are %.2f and %.2f, respectively." %
      (stat.min_p_wilcoxon, stat.min_q_wilcoxon))
```

------------------------

![](documentation/images/Test%20process.png?raw=true)

Images: *C. elegans* by Bob Goldstein, UNC Chapel Hill
<http://bio.unc.edu/people/faculty/goldstein/> Freely licensed. Contour
credit: MRC Schafer Lab. Simulated worm: OpenWorm.
