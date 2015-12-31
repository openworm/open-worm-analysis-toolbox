Sources of worm movement content
================================

(Currently our project uses just the Schafer Lab data)

[Schafer Lab](http://www2.mrc-lmb.cam.ac.uk/groups/wschafer/)

[Wikimedia
Commons](https://commons.wikimedia.org/wiki/Category:Videos_of_Caenorhabditis_elegans),
thanks to @Daniel-Mietchen, updated frequently via his [crawler
bot](https://commons.wikimedia.org/wiki/User:Open_Access_Media_Importer_Bot)

Work Tracking Software Packages
===============================

Here's a 2012 review of available Worm Trackers - [Keeping track of worm
trackers](http://www.wormbook.org/chapters/www_tracking/tracking.html)

1.  WT 2.0 from [Dr William Schafer's
    lab](http://www2.mrc-lmb.cam.ac.uk/groups/wschafer/) at Cambridge
    University's MRC Laboratory of Molecular Biology

    > -   NOTE: The OpenWorm team has chosen to adapt the codebase of
    >     this tracker for our Open Worm Analysis Toolbox.

2.  <https://github.com/samuellab/EarlyVersionWormTracker> (this repo
    provides links to other recommended trackers)
3.  [3-D Worm Tracker for Freely Moving C.
    elegans](http://www.pubmedcentral.nih.gov/articlerender.fcgi?artid=3578814&tool=pmcentrez&rendertype=abstract)
4.  [Track-A-Worm, An Open-Source System for Quantitative Assessment of
    C. elegans Locomotory and Bending
    Behavior](http://www.plosone.org/article/info:doi/10.1371/journal.pone.0069653),
    from Dr. Zhao-Wen Wang at the University of Connecticut

    > -   (We had [blogged about this
    >     one](http://blog.openworm.org/post/60312568840/ios-game-looks-to-kickstart-neuroscience-education)
    >     on 9 September 2013)
    > -   On 23 Feb 2014 Dr. Zhao-Wen Wang, the co-author of this
    >     system, sent us the [link to download the
    >     software](http://zwwang.uchc.edu/wormtrack/index.html).
    > -   Unlike the Schafer Lab's WT 2.0, the Track-A-Worm system can
    >     apparently handle most omega bends. (although not the system
    >     linked to here; only the latest version which is available
    >     only by emailing Dr. Wang directly:)
    > -   From Sijie Jason Wang, the son of Dr. Wang: "Here's a link to
    >     my current version, which is capable of resolving omega bends.
    >     I'm still tweaking the algorithm so things will continue to
    >     change. It's a pretty significant improvement over the version
    >     posted on the website. Please let me know if you need any help
    >     getting it to run, or if you have any suggestions at all. I'm
    >     working alone so a second set of eyes would be very much
    >     appreciated!"
    >     <https://www.dropbox.com/sh/33v65ofzs7m66n8/vVB7DiVp_v>
    > -   Note that this code is under a restrictive license that
    >     prevents it being used in our codebase, since it does not
    >     permit commercial reuse, which the MIT license requires. See
    >     [a June 2014
    >     discussion](https://groups.google.com/forum/#!topic/openworm-discuss/Ab0MrGRCwoY)
    >     on the boards about this.

5.  Nicolas Roussel, BioImage Analytics Laboratory, Department of
    Electrical and Computer Engineering, University of Houston,
    “Farsight Toolkit”
    <http://www.farsight-toolkit.org/wiki/The_Worm_Project>
    <https://github.com/hocheung20/farsight>

    > -   On 2 March 2015 I (@MichaelCurrie) tracked down [Badri
    >     Roysam](broysam@central.uh.edu), of the University of Houston,
    >     who had the following to say about this project: "The worm
    >     tracking project is currently in hiatus. The wiki page and the
    >     code are not being actively worked on. Roussel has moved to a
    >     company MBF Biosciences, and developed a commercial product
    >     there. Yu Wang has also moved to a career in California. If
    >     you see collaboration potential we are happy to engage."

6.  Roussel himself working for WormLab to implement his model-based
    algorithm.
7.  Brown Lab is working on a new tracker. Since André Brown worked on
    WT2.0, this new tracker is basically WT3.0 and will have multi-worm
    tracking capabilities. They are working closely with OpenWorm; viz.
    [this meeting 16 Feb
    2015](<https://docs.google.com/document/d/1VRMnNHoLQ2oHD3wgGOQrE8KzyCSrEoMwChWPOhVNTj0/edit?usp=sharing>)
8.  A commercial competitor has arisen in this space, by MBF
    Biosciences: <http://www.mbfbioscience.com/wormlab>
9.  An open-source static worm image analysis package, that analyses still images rather than videos, is described in [Wählby et al., 2012](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3433711/), called "WormToolbox".

