GarminPlotter
=============

A handy collection of python scripts to analyse and plot cycling data exported in .tcx format from the Garmin Connect website

Will only work as intended if you have speed, cadence, elevation, and heart rate data in your files. If elevation is missing, use the indoor version (since that is designed for trainer use).

More information can be found on my blog at:

http://papics.eu/?p=2142

http://papics.eu/?p=2221

http://papics.eu/?p=2247

Tested on full Anaconda (https://store.continuum.io/cshop/anaconda/) install:

             platform : osx-64
        conda version : 3.5.5
       python version : 2.7.8.final.0
       
Necessary extra package:

       conda install -c https://conda.binstar.org/piyanatk pyephem
