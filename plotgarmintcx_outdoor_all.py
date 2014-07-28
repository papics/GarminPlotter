import glob
import os
import sys

def listfiles(filenamepattern):
	"""
	Creates a sorted list from files with a filepattern in their name (e.g. '*.fits')
	
	@param filenamepattern: Common part of filenames of fiels which you want to put into the list.
	@type filenamepattern:	string
	@return:		List of files (filenames)
	@rtype:			list
	"""
	filelist = glob.glob(filenamepattern) #Path for the files.
	filelist.sort() #to make sure that the list is sorted
	return filelist
	
if __name__=="__main__":
	"""
	Runs the GARMIN plotter for all outdoor .tcx files (make sure you have only outdoor files in the directory where you run this)
	>python plotgarmintcx_outdoor_all.py
	"""
	
	filenamepattern = './*.tcx'
	filelist = listfiles(filenamepattern)
	print filelist
	for file in filelist:
		print file
		os.system('python plotgarmintcx_outdoor.py %s' %(file))
