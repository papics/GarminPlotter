import glob
import os
import sys
import datetime
import numpy as np
import pylab as py
import math as math
from plotgarmintcx_par import *

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

def get_tss(filename):
	date = filename.split('_')[3]
	date = datetime.datetime(int(date[0:4]),int(date[4:6]),int(date[6:]))
	f = open(filename)
	for line in f:
		if line[0:10] == 'TSS (est.)':
			tss = float(line[21:30])
	f.close()
	return date,tss
	
def pathcheck(path):
	"""
	Function checks if directory in a path (for a file) already exists, if not, it creates it.
	
	@param path:	The filename (containing the path relative to the library of the main script) of which path have to be checked or created.
	@type path:	string
	"""
	dir = os.path.dirname(path)
	if not os.path.exists(dir):
		os.makedirs(dir)
	return
	
if __name__=="__main__":
	"""
	Prepares Performance Management Chart for all the workouts in the directories (se you need all workouts processed with the plotter scripts beforehand)
	>python plotgarmintcx_makePMC.py
	By default it will plot all your workouts, but yoiu can also specify an interval for plotting, like:
	>python plotgarmintcx_makePMC.py 20110101 20110731
	"""
	
	filenamepattern = './*/statistics*'
	filelist = listfiles(filenamepattern)
	for file in filelist:
		date,tss = get_tss(file)
		try:
			date_all = np.append(date_all,date)
			tss_all = np.append(tss_all,tss)
		except:
			date_all = np.array(date)
			tss_all = np.array(tss)
	
	#here we calculate daily TSS values, so if there are more workouts from the same days, the values will be summed
	i = 1
	date_all_processed = np.array([date_all[0]])
	tss_all_processed = np.array([tss_all[0]])
	while i < date_all.shape[0]:
		if date_all[i] != date_all_processed[-1]:
			date_all_processed = np.append(date_all_processed,date_all[i])
			tss_all_processed = np.append(tss_all_processed,tss_all[i])
		else:
			tss_all_processed[-1] = tss_all_processed[-1] + tss_all[i]
		i = i+1
	date = date_all_processed
	tss = tss_all_processed
	
	date_start = date[0]-datetime.timedelta(days = 1)
	date_end = date[-1]
	
	try:
		date_start_forplotting = datetime.datetime(int(sys.argv[1][0:4]),int(sys.argv[1][4:6]),int(sys.argv[1][6:]))-datetime.timedelta(days = 1)
	except:
		date_start_forplotting = date_start-datetime.timedelta(days = 1)
	try:
		date_end_forplotting = datetime.datetime(int(sys.argv[2][0:4]),int(sys.argv[2][4:6]),int(sys.argv[2][6:]))+datetime.timedelta(days = 1)
	except:
		date_end_forplotting = date_end+datetime.timedelta(days = 1)
	
	#calculate values for the PMC
	date_current = date_start
	while date_current <= date_end:
		try:
			calendar_date = np.append(calendar_date,date_current)
			try:
				tss_current = tss[np.argwhere(date == date_current)[0]]
			except:
				tss_current = 0
			calendar_tss = np.append(calendar_tss,tss_current)
			atl_previous = calendar_atl[np.argwhere(calendar_date == (date_current - datetime.timedelta(days = 1)))[0]]
			atl_current = atl_previous + (tss_current - atl_previous)*(1-math.exp(-1./atl_timeconstant))
			calendar_atl = np.append(calendar_atl,atl_current)
			ctl_previous = calendar_ctl[np.argwhere(calendar_date == (date_current - datetime.timedelta(days = 1)))[0]]
			ctl_current = ctl_previous + (tss_current - ctl_previous)*(1-math.exp(-1./ctl_timeconstant))
			calendar_ctl = np.append(calendar_ctl,ctl_current)
			tsb_current = ctl_previous - atl_previous
			calendar_tsb = np.append(calendar_tsb,tsb_current)
		except:
			calendar_date = np.array(date_current)
			calendar_tss = np.array([0])
			calendar_atl = np.array([0])
			calendar_ctl = np.array([0])
			calendar_tsb = np.array([0])
		date_current = date_current+datetime.timedelta(days = 1)
	
	#do the plotting
	fig_size = [20,10]
	params = {'axes.labelsize': 12,'text.fontsize': 12,'legend.fontsize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12, 'figure.figsize': fig_size}
	py.rcParams.update(params)
	py.clf()
	fig = py.figure(1)
	host = fig.add_subplot(111)
	host.set_xlabel("Date")
	par1 = host.twinx()
	py.subplots_adjust(right=0.92,left=0.08)
	i = 0
	for entyr in calendar_date:
		p1, = host.plot([calendar_date[i],calendar_date[i]],[0,calendar_tss[i]],lw=1,color="0.5")
		i = i+1
	p2, = par1.plot(calendar_date,calendar_atl,lw=1,color="m")
	p2, = par1.plot(calendar_date,calendar_ctl,lw=1.5,color="b")
	p2, = par1.plot(calendar_date,calendar_tsb,lw=1,color="y")
	p2, = par1.plot([date_start_forplotting,date_end_forplotting],[0,0],"k--",lw=1)
	#host.set_xticks()
	host.set_xlim(date_start_forplotting,date_end_forplotting)
	#host.set_ylim(0,500)
	#par1.set_ylim(-100,150)
	host.set_ylabel("TSS")
	par1.set_ylabel("ATL, CTL, TSB (TSS/d)")
	plotfilename = './PMC/PMC_'+str(date_start_forplotting+datetime.timedelta(days = 1))[0:4]+str(date_start_forplotting+datetime.timedelta(days = 1))[5:7]+str(date_start_forplotting+datetime.timedelta(days = 1))[8:10]+'_'+str(date_end_forplotting-datetime.timedelta(days = 1))[0:4]+str(date_end_forplotting-datetime.timedelta(days = 1))[5:7]+str(date_end_forplotting-datetime.timedelta(days = 1))[8:10]+'.png'
	pathcheck(plotfilename)
	py.savefig(plotfilename,dpi=150)
	#py.show()
	
	#save the data as a file too...
	table = np.column_stack((calendar_date,calendar_tss,calendar_atl,calendar_ctl,calendar_tsb))
	tablefilename = './PMC/PMC.txt'
	np.savetxt(tablefilename,table,fmt='%20s %7.1f %6.2f %6.2f %6.2f')