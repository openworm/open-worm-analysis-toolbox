%{

Main Processing:

progress_verbose.m  - runs when you click start in main_GUI_automatic
---------------------------------------------------------------------------
startAnalysis
    - updateRandomExperimentListTable ???
    - createAnalysisDir
    - findStageMovementProcess - SegWorm/Pipeline/findStageMovementProcess
        - video2Diff
        - findStageMovement
    - segmentationMain      - SegWorm/Pipeline/segmentationMain

        Parsing of the worm occurs. Every 500 frames the results are saved
        to disk as a "block"
    
        - readPixels2Microns
        - video2Vignette
        - initializeChunkHTstats - SegWorm/Pipeline/initializeChunkHTstats
        - segWorm
        - orientWormPostCoil
        - worm2cell
    - getVulvaSide
    - correctVulvalSide
    - normWormProcess - SegWorm/Pipeline/normWormProcess.m
        - normWorms    
    - featureProcess  - SegWorm/Pipeline/featureProcess.m



%}