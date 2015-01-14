Sources of worm movement content
--------------------------------

(Currently our project uses just the Schafer Lab data)

`Schafer Lab <http://www2.mrc-lmb.cam.ac.uk/groups/wschafer/>`__

`Wikimedia
Commons <https://commons.wikimedia.org/wiki/Category:Videos_of_Caenorhabditis_elegans>`__,
thanks to @Daniel-Mietchen, updated frequently via his `crawler
bot <https://commons.wikimedia.org/wiki/User:Open_Access_Media_Importer_Bot>`__

Work Tracking Software Packages
-------------------------------

Here's a 2012 review of available Worm Trackers - `Keeping track of worm
trackers <http://www.wormbook.org/chapters/www_tracking/tracking.html>`__

1. WT 2.0 from `Dr William Schafer's lab <http://www2.mrc-lmb.cam.ac.uk/groups/wschafer/>`__ at Cambridge University's MRC Laboratory of Molecular Biology

	-  NOTE: The OpenWorm team has chosen to adapt the codebase of this tracker for our movement validation functionality.

2. `3-D Worm Tracker for Freely Moving C. elegans <http://www.pubmedcentral.nih.gov/articlerender.fcgi?artid=3578814&tool=pmcentrez&rendertype=abstract>`__

3. `Track-A-Worm, An Open-Source System for Quantitative Assessment of C. elegans Locomotory and Bending Behavior <http://www.plosone.org/article/info:doi/10.1371/journal.pone.0069653>`__, from Dr. Zhao-Wen Wang at the University of Connecticut

	-  (We had `blogged about this one <http://blog.openworm.org/post/60312568840/ios-game-looks-to-kickstart-neuroscience-education>`__ on 9 September 2013)
	-  On 23 Feb 2014 Dr. Zhao-Wen Wang, the co-author of this system, sent us the `link to download the software <http://zwwang.uchc.edu/wormtrack/index.html>`__.
	-  Unlike the Schafer Lab's WT 2.0, the Track-A-Worm system can apparently handle most omega bends. (although not the system linked to here; only the latest version which is available only by emailing Dr. Wang directly:)
	-  From Sijie Jason Wang, the son of Dr. Wang: "Here's a link to my current version, which is capable of resolving omega bends. I'm still tweaking the algorithm so things will continue to change. It's a pretty significant improvement over the version posted on the website. Please let me know if you need any help getting it to run, or if you have any suggestions at all. I'm working alone so a second set of eyes would be very much appreciated!" https://www.dropbox.com/sh/33v65ofzs7m66n8/vVB7DiVp_v
	-  Note that this code is under a restrictive license that prevents it being used in our codebase, since it does not permit commercial reuse, which the MIT license requires. See `a June 2014 discussion <https://groups.google.com/forum/#!topic/openworm-discuss/Ab0MrGRCwoY>`__ on the boards about this.
