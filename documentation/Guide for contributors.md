##Guide for Contributors to this Repository##

Thank you for your interest in contributing!  Whether you are a *C. elegans* researcher who wishes to add a feature set to the calculation engine, or an interested person looking to contribute to the overall OpenWorm effort, you are welcome to join in on the fun.

###Objectives:###

1. This repository should house a test [pipeline](https://github.com/OpenWorm/movement_validation/blob/master/documentation/Processing%20Pipeline.md) for the OpenWorm project to run a behavioural phenotyping of its virtual worm, using the same statistical tests the Schafer lab used on their real worm data.  
[**Overall OpenWorm Milestone**: *#19*](https://github.com/openworm/OpenWorm/issues?milestone=19&state=open)  

2. In achieving goal #1, this repository will also be an open source version of the Schafer Lab's Worm Tracker 2 (WT2) analysis pipeline that goes from raw real worm videos to worm measurements detected by machine vision, to a selection of calculated worm behavioural features like average speed.

3. Also in achieving goal #1, we hope to have a system for tracking the statistics of hundreds of real and virtual worm strains in a database.


### History of this Repository ###

The genesis of this repository was an [OpenWorm Journal Club event featuring Dr. Eviatar "Ev" Yemini](https://www.youtube.com/watch?v=YdBGbn_g_ls), on 16 August 2013.  He described how his software can process raw videos of worms into statistics describing their movement quantitatively.  It was realized that OpenWorm could adapt this software to validate whether its simulated worms were behaving correctly. 

Consequently, to start, in August 2013 [Jim Hokanson](https://github.com/JimHokanson) cloned the WT2 software, written in Matlab, into a repository he called ["SegWorm"](https://github.com/openworm/SegWorm).

He took this repository as the starting point for his code development work on behalf of OpenWorm.  He revised this code from September 2013 to January 2014, to speed up and clarify the code.  This revised Matlab version is available in the ["SegWormMatlabClasses"](https://github.com/JimHokanson/SegwormMatlabClasses/) repo.

From October 2013, [Michael Currie](https://github.com/MichaelCurrie) started to translate Jim's SegWormMatlabClasses repository into Python so it would be fully open source.  To this end he started the [movement_validation](https://github.com/openworm/movement_validation) repository.

Currently only the movement_validation repository is being actively worked on.

An overview of the current status of the work on the movement_validation repository is available at [Code Progress.pdf](https://github.com/openworm/movement_validation/blob/master/documentation/Code%20Progress.pdf).

### Best practices for contributing to this repo ###

Just find an issue in the "Ready" column of the [waffle board](https://waffle.io/openworm/movement_validation) and dig in.  Ask questions of others early and often; you can do so on our message board, [OpenWorm-discuss](https://groups.google.com/forum/#!forum/openworm-discuss).

Please commit your code often, even if you've made a very small change, but only if you've verified that your change has not broken any [examples](https://github.com/openworm/movement_validation/tree/master/examples).

For more information, see ["Commit Often, Perfect Later, Publish Once: Git best practices"](http://sethrobertson.github.io/GitBestPractices/).

### Further Information ###

[White Paper Describing Movement Validation at a high level](https://github.com/openworm/movement_validation/blob/master/documentation/Movement%20Validation%20White%20Paper.md) *(June 2014)*

[Archived Monthly Progress Reports](https://drive.google.com/folderview?id=0B9dU7zPD0s_LMm5RMGZGX2JEeGc&usp=sharing) *(Maintained from September 2013 to March 2014, after which they were discontinued)*

[Movement Validation](https://github.com/openworm/openworm_docs/blob/master/Projects/worm-movement.rst) documented at the OpenWorm/openworm_docs repository.  *(Has not been updated since 29 Dec 2013)*