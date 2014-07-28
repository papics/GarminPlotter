import numpy as np
import pprint
import xml.dom.minidom
import pylab as py
from xml.dom.minidom import Node
import dateutil
import ephem
import math
import sys
import os
import matplotlib.cm as cm
from matplotlib import mpl
from plotgarmintcx_par import *
from sigproc.base import sigfilt as filt

def openpathcheck(path):
	"""
	Function checks if directory in a path (for a file) already exists, if not, it creates it, then opens the file for writing. Use instead of simple C{open(path)} command.
	
	@param path:	The filename (containing the path relative to the library of the main script) of which path have to be checked or created.
	@type path:	string
	"""
	dir = os.path.dirname(path)
	if not os.path.exists(dir):
		os.makedirs(dir)
	return open(path,'w')

def calc_climb(elevation,distance,threshold):
	"""
	Calculates clever ascent and descent values with a given threshold, based on the description here: http://djconnel.blogspot.com/2010/11/total-climbing-and-descending-algorithm_18.html
	Distance is only given to enable us plotting the preserved points over the original elevation profile (if you want to check what the script is doing...)
	"""
	elev = np.array(elevation,dtype='float')
	dist = np.array(distance,dtype='float')
	i = 0
	e = elev.shape[0]
	while i < e-2:
		if (elev[i+1]-elev[i])/(elev[i+2]-elev[i+1]) > 0:
			elev = np.delete(elev,i+1)
			dist = np.delete(dist,i+1)
			e = elev.shape[0]
		elif (elev[i+1]==elev[i]):
			elev = np.delete(elev,i+1)
			dist = np.delete(dist,i+1)
			e = elev.shape[0]
		elif (elev[i+2]==elev[i+1]):
			elev = np.delete(elev,i+1)
			dist = np.delete(dist,i+1)
			e = elev.shape[0]
		else:
			i = i+1
	i = 0
	while i <= elev.shape[0]-4:
		if (abs(elev[i+1]-elev[i+2])<threshold) & (elev[i+1]>=elev[i]) & (elev[i+2]>=elev[i]) & (elev[i+1]<=elev[i+3]) & (elev[i+2]<=elev[i+3]):
			elev = np.delete(elev,[i+1,i+2])
			dist = np.delete(dist,[i+1,i+2])
			if i == 0:
				i = 0
			elif i == 1:
				i = 0
			else:
				i = i-2
		elif (abs(elev[i+1]-elev[i+2])<threshold) & (elev[i+1]<=elev[i]) & (elev[i+2]<=elev[i]) & (elev[i+1]>=elev[i+3]) & (elev[i+2]>=elev[i+3]):
			elev = np.delete(elev,[i+1,i+2])
			dist = np.delete(dist,[i+1,i+2])
			if i == 0:
				i = 0
			elif i == 1:
				i = 0
			else:
				i = i-2
		else:
			i = i+1
	elev_diff = np.diff(elev)
	ascent,descent = np.sum(elev_diff[elev_diff>0]),abs(np.sum(elev_diff[elev_diff<0]))
	return ascent,descent,dist,elev

def decdeg2dms(dd):
	mnt,sec = divmod(dd*3600,60)
	deg,mnt = divmod(mnt,60)
	return deg,mnt,sec

def make_patch_spines_invisible2(ax):
	par2.set_frame_on(True)
	par2.patch.set_visible(False)
	for sp in par2.spines.itervalues():
		sp.set_visible(False)

def make_patch_spines_invisible3(ax):
	par3.set_frame_on(True)
	par3.patch.set_visible(False)
	for sp in par3.spines.itervalues():
		sp.set_visible(False)

def make_patch_spines_invisible4(ax):
	par4.set_frame_on(True)
	par4.patch.set_visible(False)
	for sp in par4.spines.itervalues():
		sp.set_visible(False)

def make_spine_invisible(ax, direction):
	if direction in ["right", "left"]:
		ax.yaxis.set_ticks_position(direction)
		ax.yaxis.set_label_position(direction)
	elif direction in ["top", "bottom"]:
		ax.xaxis.set_ticks_position(direction)
		ax.xaxis.set_label_position(direction)
	else:
		raise ValueError("Unknown Direction : %s" % (direction,))
	ax.spines[direction].set_visible(True)

if __name__=="__main__":
	"""
	Indoor cycling workout data plotter
	use with GARMIN TCX files exported from Garmin Connect
	run like >python plotgarmintcx_indoor.py './activity_66337347.tcx'
	"""
	
	##################
	#read in xml file#
	##################
	if comparisonmode == 'yes':
		plottype = 'comparison'
	else:
		comparisonmode = 'no'
		plottype = 'single'
	try:
		doc = xml.dom.minidom.parse(sys.argv[1])
		garminconnectid = sys.argv[1]
		garminconnectid = garminconnectid.split('_')
		garminconnectid = garminconnectid[1]
		garminconnectid = garminconnectid.split('.')
		garminconnectid = garminconnectid[0] #and we have the Garmin Connect file Id too, we will use it to name the output folder
	except:
		print "Invalid file, or no file given"
		sys.exit()
	mapping = {}
	#first get an Id for the track
	for node in doc.getElementsByTagName('Activities'):
		ID = node.getElementsByTagName('Id')
		for node18 in ID:
			for node19 in node18.childNodes:
				id = str(node19.data)
	id = id.replace('-','')
	id = id.replace('T','_')
	id = id.replace(':','')
	id = id.split('.')
	id = id[0]
	print 'Garmin Connect ID:  %s' %(garminconnectid)
	print 'File ID:     %s' %(id)
	#then read in the data
	i = 0
	for node in doc.getElementsByTagName('Trackpoint'):
		try: #this is needed because the first point usually has only time, and no other data
			H = node.getElementsByTagName('AltitudeMeters')
			for node2 in H:
				for node3 in node2.childNodes:
					ele = float(node3.data)
			T = node.getElementsByTagName('Time')
			for node4 in T:
				for node5 in node4.childNodes:
					time = node5.data
					time = str(dateutil.parser.parse(time))
					time = time.replace('-','/')
					time = ephem.date(time) #ephem.date stores dates in days from last day of 1899 noon UT
			TA = node.getElementsByTagName('DistanceMeters')
			for node6 in TA:
				for node7 in node6.childNodes:
					distance = float(node7.data)/1000.
			HR = node.getElementsByTagName('Value')
			for node8 in HR:
				for node9 in node8.childNodes:
					heartrate = float(node9.data)
			CA = node.getElementsByTagName('Cadence')
			for node10 in CA:
				for node11 in node10.childNodes:
					cadence = float(node11.data)
			V = node.getElementsByTagName('Speed')
			for node12 in V:
				for node13 in node12.childNodes:
					speed = float(node13.data)*3.6
			if i > 0:
				time_all = np.column_stack((time_all,time))
				distance_all = np.column_stack((distance_all,distance))
				heartrate_all = np.column_stack((heartrate_all,heartrate))
				cadence_all = np.column_stack((cadence_all,cadence))
				speed_all = np.column_stack((speed_all,speed))
				i = i+1
			else:
				time_all = time
				distance_all = distance
				heartrate_all = heartrate
				cadence_all = cadence
				speed_all = speed
				i = i+1
		except:
			continue
	time = time_all[0]
	distance = distance_all[0]
	hr = heartrate_all[0]
	cad = cadence_all[0]
	speed = speed_all[0]
	print 'File consists of: %10d    data points' %(i)
	
	#cut section of data out if needed
	if plotonlysection == 'yes':
		time = time[(distance >= section[0]) & (distance <= section[1])]
		time = time - time[0] #so section starts at 00:00:00
		hr = hr[(distance >= section[0]) & (distance <= section[1])]
		cad = cad[(distance >= section[0]) & (distance <= section[1])]
		speed = speed[(distance >= section[0]) & (distance <= section[1])]
		distance = distance[(distance >= section[0]) & (distance <= section[1])]
		distance = distance - distance[0] #so section starts at 0 km
		print 'Section consists of: %7d    data points' %(distance.shape[0])
	
	###########################
	#create logfile and folder#
	###########################
	if testmode != 'yes':
		if plotonlysection == 'yes':
			workfoldername = str(id)+'_'+str(garminconnectid)+'/'+str(section[0])+'_'+str(section[1])
		else:
			workfoldername = str(id)+'_'+str(garminconnectid)
		logfilename = './'+workfoldername+'/statistics_'+str(id)+'_'+str(garminconnectid)+'.txt'
		logfile = openpathcheck(logfilename)
		#and start to write out some data
		logfile.write('Garmin Connect ID:  %s\n' %(garminconnectid))
		logfile.write('File ID:     %s\n' %(id))
		if plotonlysection == 'yes':
			logfile.write('Section consists of: %7d    data points\n' %(distance.shape[0]))
		else:
			logfile.write('File consists of: %10d    data points\n' %(i))
	
	#########################################################
	#calculate values which are not present in the data file#
	#########################################################
	i = 0
	distance_step = np.ones_like(time)
	time_step = np.ones_like(time)
	ele_change = np.ones_like(time)
	for element in time:
		if i == 0:
			distance_step[i] = 0
			time_step[i] = 0
			i = i+1
		else:
			distance_step[i] = distance[i]-distance[i-1]
			time_step[i] = (time[i]-time[i-1])*24
			i = i+1
	#check if the thresholds are fine
	if testmode == 'yes':
		py.plot(distance,time_step)
		py.xlabel("Distance (km)")
		py.ylabel("Stoppage time (h)")
		py.show()
		exit()
	
	############
	#make plots#
	############
	
	#plot with two axes v. distance in parts
	distance_slice = 10 # the length of one slice in kilometers
	fig_size = [20,10]
	params = {'axes.labelsize': 12,'text.fontsize': 12,'legend.fontsize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12, 'figure.figsize': fig_size}
	py.rcParams.update(params)
	py.clf()
	s = 0
	while s*distance_slice < distance[-1]:
		fig = py.figure(1)
		host = fig.add_subplot(111)
		host.set_xlabel("Distance")
		par1 = host.twinx()
		par2 = host.twinx()
		par2.spines["right"].set_position(("axes", 1.07))
		make_patch_spines_invisible2(par2)
		make_spine_invisible(par2, "right")
		py.subplots_adjust(right=0.8,left=0.05)
		p1, = host.plot(distance,speed,lw=0.5,color='b')
		p2, = par1.plot(distance,cad,lw=0.5,color="0.5")
		p3, = par2.plot(distance,hr,lw=0.5,color="r")
		host.set_xticks(np.linspace(0+s*distance_slice+1, 0+(s+1)*distance_slice-1,((distance_slice-2)/1+1)))
		host.set_xlim(0+s*distance_slice, 0+(s+1)*distance_slice)
		host.set_ylim(0, max(speed)+0.1*max(speed))
		par1.set_ylim(0, max(cad)+0.1*max(cad))
		par2.set_ylim(0, max(hr)+0.1*max(hr))
		host.set_xlabel("Distance (km)")
		host.set_ylabel("Speed (km/h)")
		par1.set_ylabel("Cadence (RPM)")
		par2.set_ylabel("Heart rate (BPM)")
		host.yaxis.label.set_color(p1.get_color())
		par1.yaxis.label.set_color(p2.get_color())
		par2.yaxis.label.set_color(p3.get_color())
		plotfilename = './'+workfoldername+'/plotvdistance_part'+str('%02.0d' %(s+1))+'_'+str(id)+'_'+str(garminconnectid)+'.png'
		py.savefig(plotfilename,dpi=150)
		if distance[-1]<distance_slice: #if it was a short ride, save alse one slice as the 'overall' plot
			plotfilename = './'+workfoldername+'/plotvdistance_all_'+str(id)+'_'+str(garminconnectid)+'.png'
			py.savefig(plotfilename,dpi=72)
		py.clf()
		py.close(1)
		s = s+1
		#py.show()
	
	#plot with two axes v. time in parts
	time_slice = 0.5 #length of slice in hours
	fig_size = [20,10]
	params = {'axes.labelsize': 12,'text.fontsize': 12,'legend.fontsize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12, 'figure.figsize': fig_size}
	py.rcParams.update(params)
	py.clf()
	s = 0
	while s*time_slice < ((time[-1]-time[0])*24):
		fig = py.figure(1)
		host = fig.add_subplot(111)
		host.set_xlabel("Distance")
		par1 = host.twinx()
		par2 = host.twinx()
		par2.spines["right"].set_position(("axes", 1.07))
		make_patch_spines_invisible2(par2)
		make_spine_invisible(par2, "right")
		py.subplots_adjust(right=0.8,left=0.05)
		p1, = host.plot((time-time[0])*24,speed,lw=0.5,color='b')
		p2, = par1.plot((time-time[0])*24,cad,lw=0.5,color="0.5")
		p3, = par2.plot((time-time[0])*24,hr,lw=0.5,color="r")
		host.set_xticks(np.round(np.linspace(0.1,24.0,(24.0-0.1)/0.1+1),decimals=1))
		host.set_xlim(0+s*time_slice, 0+(s+1)*time_slice)
		host.set_ylim(0, max(speed)+0.1*max(speed))
		par1.set_ylim(0, max(cad)+0.1*max(cad))
		par2.set_ylim(0, max(hr)+0.1*max(hr))
		host.set_xlabel("Time (h)")
		host.set_ylabel("Speed (km/h)")
		par1.set_ylabel("Cadence (RPM)")
		par2.set_ylabel("Heart rate (BPM)")
		host.yaxis.label.set_color(p1.get_color())
		par1.yaxis.label.set_color(p2.get_color())
		par2.yaxis.label.set_color(p3.get_color())
		plotfilename = './'+workfoldername+'/plotvtime_part'+str('%02.0d' %(s+1))+'_'+str(id)+'_'+str(garminconnectid)+'.png'
		py.savefig(plotfilename,dpi=150)
		if ((time[-1]-time[0])*24)<time_slice: #if it was a short ride, save alse one slice as the 'overall' plot
			plotfilename = './'+workfoldername+'/plotvtime_all_'+str(id)+'_'+str(garminconnectid)+'.png'
			py.savefig(plotfilename,dpi=72)
		py.clf()
		py.close(1)
		s = s+1
		#py.show()
		
	#plot with two axes v. movingtime in parts
	fig_size = [20,10]
	params = {'axes.labelsize': 12,'text.fontsize': 12,'legend.fontsize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12, 'figure.figsize': fig_size}
	py.rcParams.update(params)
	py.clf()
	s = 0
	while s*time_slice < np.cumsum(time_step[(speed>1) & (time_step<time_step_threshold)])[-1]:
		fig = py.figure(1)
		host = fig.add_subplot(111)
		host.set_xlabel("Distance")
		par1 = host.twinx()
		par2 = host.twinx()
		par2.spines["right"].set_position(("axes", 1.07))
		make_patch_spines_invisible2(par2)
		make_spine_invisible(par2, "right")
		py.subplots_adjust(right=0.8,left=0.05)
		p1, = host.plot(np.cumsum(time_step[(speed>1) & (time_step<time_step_threshold)]),speed[(speed>1) & (time_step<time_step_threshold)],lw=0.5,color='b')
		p2, = par1.plot(np.cumsum(time_step[(speed>1) & (time_step<time_step_threshold)]),cad[(speed>1) & (time_step<time_step_threshold)],lw=0.5,color="0.5")
		p3, = par2.plot(np.cumsum(time_step[(speed>1) & (time_step<time_step_threshold)]),hr[(speed>1) & (time_step<time_step_threshold)],lw=0.5,color="r")
		host.set_xticks(np.round(np.linspace(0.1,24.0,(24.0-0.1)/0.1+1),decimals=1))
		host.set_xlim(0+s*time_slice, 0+(s+1)*time_slice)
		host.set_ylim(0, max(speed)+0.1*max(speed))
		par1.set_ylim(0, max(cad)+0.1*max(cad))
		par2.set_ylim(0, max(hr)+0.1*max(hr))
		host.set_xlabel("Moving time (h)")
		host.set_ylabel("Speed (km/h)")
		par1.set_ylabel("Cadence (RPM)")
		par2.set_ylabel("Heart rate (BPM)")
		host.yaxis.label.set_color(p1.get_color())
		par1.yaxis.label.set_color(p2.get_color())
		par2.yaxis.label.set_color(p3.get_color())
		plotfilename = './'+workfoldername+'/plotvmovingtime_part'+str('%02.0d' %(s+1))+'_'+str(id)+'_'+str(garminconnectid)+'.png'
		py.savefig(plotfilename,dpi=150)
		if np.cumsum(time_step[(speed>1) & (time_step<time_step_threshold)])[-1]<time_slice: #if it was a short ride, save alse one slice as the 'overall' plot
			plotfilename = './'+workfoldername+'/plotvmovingtime_all_'+str(id)+'_'+str(garminconnectid)+'.png'
			py.savefig(plotfilename,dpi=72)
		py.clf()
		py.close(1)
		s = s+1
		#py.show()
	
	#overview plot  v. distance
	fig_size = [12.5,17.5]
	params = {'axes.labelsize': 12,'text.fontsize': 12,'legend.fontsize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12, 'figure.figsize': fig_size}
	py.rcParams.update(params)
	py.subplots_adjust(hspace=0.4,bottom=0.07,top=0.95)
	py.clf()
	py.subplot(511)
	py.plot(distance,speed,lw=0.5,color='b')
	py.xlim(min(distance), max(distance))
	py.ylim(0, max(speed)+0.1*max(speed))
	py.xlabel("Distance (km)")
	py.ylabel("Speed (km/h)")
	py.subplot(512)
	py.plot(distance,hr,lw=0.5,color="r")
	py.xlim(min(distance), max(distance))
	py.ylim(min(hr)-0.1*min(hr), max(hr)+0.1*max(hr))
	py.xlabel("Distance (km)")
	py.ylabel("Heart rate (BPM)")
	py.subplot(513)
	py.plot(distance,cad,lw=0.5,color="0.5")
	py.xlim(min(distance), max(distance))
	py.ylim(0, max(cad)+0.1*max(cad))
	py.xlabel("Distance (km)")
	py.ylabel("Cadence (RPM)")
	plotfilename = './'+workfoldername+'/overviewplotvdistance_'+str(id)+'_'+str(garminconnectid)+'.png'
	py.savefig(plotfilename,dpi=150)
	py.close()
	#py.show()
	
	#overview plot  v. time
	fig_size = [12.5,17.5]
	params = {'axes.labelsize': 12,'text.fontsize': 12,'legend.fontsize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12, 'figure.figsize': fig_size}
	py.rcParams.update(params)
	py.subplots_adjust(hspace=0.4,bottom=0.07,top=0.95)
	py.clf()
	py.subplot(511)
	py.plot((time-time[0])*24,speed,lw=0.5,color='b')
	py.xlim(min((time-time[0])*24), max((time-time[0])*24))
	py.ylim(0, max(speed)+0.1*max(speed))
	py.xlabel("Time (h)")
	py.ylabel("Speed (km/h)")
	py.subplot(512)
	py.plot((time-time[0])*24,hr,lw=0.5,color="r")
	py.xlim(min((time-time[0])*24), max((time-time[0])*24))
	py.ylim(min(hr)-0.1*min(hr), max(hr)+0.1*max(hr))
	py.xlabel("Time (h)")
	py.ylabel("Heart rate (BPM)")
	py.subplot(513)
	py.plot((time-time[0])*24,cad,lw=0.5,color="0.5")
	py.xlim(min((time-time[0])*24), max((time-time[0])*24))
	py.ylim(0, max(cad)+0.1*max(cad))
	py.xlabel("Time (h)")
	py.ylabel("Cadence (RPM)")
	plotfilename = './'+workfoldername+'/overviewplotvtime_'+str(id)+'_'+str(garminconnectid)+'.png'
	py.savefig(plotfilename,dpi=150)
	py.close()
	#py.show()
	
	#overview plot  v. movingtime
	fig_size = [12.5,17.5]
	params = {'axes.labelsize': 12,'text.fontsize': 12,'legend.fontsize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12, 'figure.figsize': fig_size}
	py.rcParams.update(params)
	py.subplots_adjust(hspace=0.4,bottom=0.07,top=0.95)
	py.clf()
	py.subplot(511)
	py.plot(np.cumsum(time_step[(speed>1) & (time_step<time_step_threshold)]),speed[(speed>1) & (time_step<time_step_threshold)],lw=0.5,color='b')
	py.xlim(min(np.cumsum(time_step[(speed>1) & (time_step<time_step_threshold)])), max(np.cumsum(time_step[(speed>1) & (time_step<time_step_threshold)])))
	py.ylim(0, max(speed)+0.1*max(speed))
	py.xlabel("Moving time (h)")
	py.ylabel("Speed (km/h)")
	py.subplot(512)
	py.plot(np.cumsum(time_step[(speed>1) & (time_step<time_step_threshold)]),hr[(speed>1) & (time_step<time_step_threshold)],lw=0.5,color="r")
	py.xlim(min(np.cumsum(time_step[(speed>1) & (time_step<time_step_threshold)])), max(np.cumsum(time_step[(speed>1) & (time_step<time_step_threshold)])))
	py.ylim(min(hr)-0.1*min(hr), max(hr)+0.1*max(hr))
	py.xlabel("Moving time (h)")
	py.ylabel("Heart rate (BPM)")
	py.subplot(513)
	py.plot(np.cumsum(time_step[(speed>1) & (time_step<time_step_threshold)]),cad[(speed>1) & (time_step<time_step_threshold)],lw=0.5,color="0.5")
	py.xlim(min(np.cumsum(time_step[(speed>1) & (time_step<time_step_threshold)])), max(np.cumsum(time_step[(speed>1) & (time_step<time_step_threshold)])))
	py.ylim(0, max(cad)+0.1*max(cad))
	py.xlabel("Moving time (h)")
	py.ylabel("Cadence (RPM)")
	plotfilename = './'+workfoldername+'/overviewplotvmovingtime_'+str(id)+'_'+str(garminconnectid)+'.png'
	py.savefig(plotfilename,dpi=150)
	py.close()
	#py.show()
	
	####################
	#calculate averages#
	####################
	distance_total = distance[-1]
	time_moving = np.cumsum(time_step[(speed>1) & (time_step<time_step_threshold)])[-1]
	time_total = (time[-1]-time[0])*24
	time_stopped = time_total-time_moving
	time_stopped_percent = time_stopped/(time_total/100)
	try: #because there migh be no section with 0 cadence
		time_moving_zerocadence = np.cumsum(time_step[(cad==0) & (speed>1) & (time_step<time_step_threshold)])[-1]
	except:
		time_moving_zerocadence = 0
	time_moving_zerocadence_percent = time_moving_zerocadence/(time_moving/100)
	speed_average_moving = np.average(speed[(speed>1) & (time_step<time_step_threshold)],weights = time_step[(speed>1) & (time_step<time_step_threshold)])
	speed_max = max(speed)
	hr_average_moving = np.average(hr[(speed>1) & (time_step<time_step_threshold)],weights = time_step[(speed>1) & (time_step<time_step_threshold)])
	hr_max = max(hr)
	cad_average_nonzero = np.average(cad[(cad>0) & (speed>1) & (time_step<time_step_threshold)],weights = time_step[(cad>0) & (speed>1) & (time_step<time_step_threshold)])
	cad_max = max(cad)
	
	############
	#histograms#
	############
	bins_speed = np.linspace(bins_speed_min,bins_speed_max,(bins_speed_max-bins_speed_min)/bins_speed_step+1) #number of bins in the histograms
	bins_hr = np.linspace(bins_hr_min,bins_hr_max,(bins_hr_max-bins_hr_min)/bins_hr_step+1) #number of bins in the histograms
	bins_cad = np.linspace(bins_cad_min,bins_cad_max,(bins_cad_max-bins_cad_min)/bins_cad_step+1) #number of bins in the histograms
	
	fig_size = [15,10]
	params = {'axes.labelsize': 12,'text.fontsize': 12,'legend.fontsize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12, 'figure.figsize': fig_size}
	py.rcParams.update(params)
	py.clf()
	py.hist(speed[(speed>1) & (time_step<time_step_threshold)],weights = time_step[(speed>1) & (time_step<time_step_threshold)], bins=bins_speed)
	py.xlim((bins_speed_min,bins_speed_max))
	py.xlabel('Speed (km/h)')
	py.ylabel("Time (h)")
	plotfilename = './'+workfoldername+'/histogram_s_'+str(id)+'_'+str(garminconnectid)+'.png'
	py.savefig(plotfilename,dpi=150)
	py.close()
	#py.show()
	
	fig_size = [15,10]
	params = {'axes.labelsize': 12,'text.fontsize': 12,'legend.fontsize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12, 'figure.figsize': fig_size}
	py.rcParams.update(params)
	py.clf()
	py.hist(hr[(speed>1) & (time_step<time_step_threshold)],weights = time_step[(speed>1) & (time_step<time_step_threshold)], bins=bins_hr)
	py.xlim((bins_hr_min,bins_hr_max))
	py.xlabel('Heart rate (BPM)')
	py.ylabel("Time (h)")
	plotfilename = './'+workfoldername+'/histogram_h_'+str(id)+'_'+str(garminconnectid)+'.png'
	py.savefig(plotfilename,dpi=150)
	py.close()
	#py.show()
	
	fig_size = [15,10]
	params = {'axes.labelsize': 12,'text.fontsize': 12,'legend.fontsize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12, 'figure.figsize': fig_size}
	py.rcParams.update(params)
	py.clf()
	py.hist(hr[(speed>1) & (time_step<time_step_threshold)],weights = time_step[(speed>1) & (time_step<time_step_threshold)], bins=[bins_hrzones[0],bins_hrzones[1],bins_hrzones[2],bins_hrzones[3],bins_hrzones[4],bins_hrzones[5],bins_hrzones[8],bins_hrzones[9]])
	py.hist(hr[(speed>1) & (time_step<time_step_threshold)],weights = time_step[(speed>1) & (time_step<time_step_threshold)], bins=[bins_hrzones[5],bins_hrzones[6],bins_hrzones[7],bins_hrzones[8]])
	py.xlim((bins_hrzones[0],bins_hrzones[-1]))
	py.xlabel('Heart rate (BPM)')
	py.ylabel("Time (h)")
	plotfilename = './'+workfoldername+'/histogram_hz_'+str(id)+'_'+str(garminconnectid)+'.png'
	py.savefig(plotfilename,dpi=150)
	py.close()
	#py.show()
	
	fig_size = [15,10]
	params = {'axes.labelsize': 12,'text.fontsize': 12,'legend.fontsize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12, 'figure.figsize': fig_size}
	py.rcParams.update(params)
	py.clf()
	py.hist(cad[(speed>1) & (time_step<time_step_threshold)],weights = time_step[(speed>1) & (time_step<time_step_threshold)], bins=bins_cad)
	py.xlim((bins_cad_min,bins_cad_max))
	py.xlabel('Cadence (RPM)')
	py.ylabel("Time (h)")
	plotfilename = './'+workfoldername+'/histogram_c_'+str(id)+'_'+str(garminconnectid)+'.png'
	py.savefig(plotfilename,dpi=150)
	py.close()
	#py.show()
	
	###############
	#2D histograms#
	###############
	
	fig_size = [12.5,10]
	params = {'axes.labelsize': 12,'text.fontsize': 12,'legend.fontsize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12, 'figure.figsize': fig_size}
	py.rcParams.update(params)
	py.clf()
	cmap=py.set_cmap('gist_earth')
	b = 0
	while bins_hr[b] < bins_hr[-1]:
		n_sh,bins_speed_edges = np.histogram(speed[(speed>1) & (time_step<time_step_threshold) & (hr>=bins_hr[b]) & (hr<bins_hr[b+1])],weights = time_step[(speed>1) & (time_step<time_step_threshold) & (hr>=bins_hr[b]) & (hr<bins_hr[b+1])], bins=bins_speed)
		try:
			n_sh_all = np.row_stack((n_sh_all,n_sh))
		except:
			n_sh_all = n_sh
		b = b+1
	py.imshow(n_sh_all[::-1],aspect='auto',extent=(bins_speed[0],bins_speed[-1],bins_hr[0],bins_hr[-1]),interpolation='nearest')
	py.subplots_adjust(left=0.1,right=0.9)
	cb = py.colorbar(fraction=0.05,pad=0.05)
	cb.set_label('Time (h)')
	py.xlabel('Speed (km/s)')
	py.ylabel('Heart rate (BPM)')
	plotfilename = './'+workfoldername+'/histogram2D_sh_'+str(id)+'_'+str(garminconnectid)+'.png'
	py.savefig(plotfilename,dpi=150)
	py.close()
	#py.show()
	
	fig_size = [12.5,10]
	params = {'axes.labelsize': 12,'text.fontsize': 12,'legend.fontsize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12, 'figure.figsize': fig_size}
	py.rcParams.update(params)
	py.clf()
	cmap=py.set_cmap('gist_earth')
	b = 0
	while bins_cad[b] < bins_cad[-1]:
		n_sc,bins_speed_edges = np.histogram(speed[(speed>1) & (time_step<time_step_threshold) & (cad>=bins_cad[b]) & (cad<bins_cad[b+1])],weights = time_step[(speed>1) & (time_step<time_step_threshold) & (cad>=bins_cad[b]) & (cad<bins_cad[b+1])], bins=bins_speed)
		try:
			n_sc_all = np.row_stack((n_sc_all,n_sc))
		except:
			n_sc_all = n_sc
		b = b+1
	py.imshow(n_sc_all[::-1],aspect='auto',extent=(bins_speed[0],bins_speed[-1],bins_cad[0],bins_cad[-1]),interpolation='nearest')
	py.subplots_adjust(left=0.1,right=0.9)
	cb = py.colorbar(fraction=0.05,pad=0.05)
	g = 0
	for frontgear in gears_front:
		for reargear in gears_rear:
			gearsetspeed_low = 45*(float(frontgear)/float(reargear))*(wheel_circ/1000000.)*60.
			gearsetspeed_high= 125*(float(frontgear)/float(reargear))*(wheel_circ/1000000.)*60.
			if np.mod(g,2) == 0:
				py.plot([gearsetspeed_low,gearsetspeed_high],[45,125],'--',lw=0.3,color='0.33')
				py.text(gearsetspeed_high,125+4,'%d:%d' %(frontgear,reargear),rotation=90,va='center',ha='center',size=7,color='0.33')
			else:
				py.plot([gearsetspeed_low,gearsetspeed_high],[45,125],'-',lw=0.3,color='0.33')
				py.text(gearsetspeed_low,45-4,'%d:%d' %(frontgear,reargear),rotation=90,va='center',ha='center',size=7,color='0.33')
		g = g+1
	cb.set_label('Time (h)')
	py.xlabel('Speed (km/s)')
	py.ylabel('Cadence (RPM)')
	py.xlim((bins_speed[0],bins_speed[-1]))
	py.ylim((bins_cad[0],bins_cad[-1]))
	plotfilename = './'+workfoldername+'/histogram2D_sc_'+str(id)+'_'+str(garminconnectid)+'.png'
	py.savefig(plotfilename,dpi=150)
	py.close()
	#py.show()
	
	fig_size = [12.5,10]
	params = {'axes.labelsize': 12,'text.fontsize': 12,'legend.fontsize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12, 'figure.figsize': fig_size}
	py.rcParams.update(params)
	py.clf()
	cmap=py.set_cmap('gist_earth')
	b = 0
	while bins_hr[b] < bins_hr[-1]:
		n_ch,bins_cad_edges = np.histogram(cad[(speed>1) & (time_step<time_step_threshold) & (hr>=bins_hr[b]) & (hr<bins_hr[b+1])],weights = time_step[(speed>1) & (time_step<time_step_threshold) & (hr>=bins_hr[b]) & (hr<bins_hr[b+1])], bins=bins_cad)
		try:
			n_ch_all = np.row_stack((n_ch_all,n_ch))
		except:
			n_ch_all = n_ch
		b = b+1
	py.imshow(n_ch_all[::-1],aspect='auto',extent=(bins_cad[0],bins_cad[-1],bins_hr[0],bins_hr[-1]),interpolation='nearest')
	py.subplots_adjust(left=0.1,right=0.9)
	cb = py.colorbar(fraction=0.05,pad=0.05)
	cb.set_label('Time (h)')
	py.xlabel('Cadence (RPM)')
	py.ylabel('Heart rate (BPM)')
	plotfilename = './'+workfoldername+'/histogram2D_ch_'+str(id)+'_'+str(garminconnectid)+'.png'
	py.savefig(plotfilename,dpi=150)
	py.close()
	#py.show()
	
	#####################
	#'Correlation' plots#
	#####################
	
	fig_size = [12.5,10]
	params = {'axes.labelsize': 12,'text.fontsize': 12,'legend.fontsize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12, 'figure.figsize': fig_size}
	py.rcParams.update(params)
	py.clf()
	cmap=py.set_cmap('gist_rainbow')
	sc = py.scatter(speed[(speed>1) & (time_step<time_step_threshold)],hr[(speed>1) & (time_step<time_step_threshold)],c=cad[(speed>1) & (time_step<time_step_threshold)],s=3,edgecolors='none',cmap=cmap)
	#sc.set_alpha(0.75)
	py.subplots_adjust(left=0.1,right=0.9)
	cb = py.colorbar(fraction=0.05,pad=0.05)
	cb.set_label('Cadence (RPM)')
	py.xlabel('Speed (km/h)')
	py.ylabel('Heart rate (BPM)')
	if comparisonmode == 'yes': # this makes the plots identical, so for comparing workouts, it is handy
		cb.set_clim((plots_cad_min,plots_cad_max))
		py.xlim((plots_speed_min,plots_speed_max))
		py.ylim((plots_hr_min,plots_hr_max))
	plotfilename = './'+workfoldername+'/3Dplots_shc_'+str(plottype)+'_'+str(id)+'_'+str(garminconnectid)+'.png'
	py.savefig(plotfilename,dpi=150)
	py.close()
	#py.show()
	
	fig_size = [12.5,10]
	params = {'axes.labelsize': 12,'text.fontsize': 12,'legend.fontsize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12, 'figure.figsize': fig_size}
	py.rcParams.update(params)
	py.clf()
	cmap=py.set_cmap('gist_rainbow')
	if comparisonmode == 'yes': # this makes the plots identical, so for comparing workouts, it is handy
		g = 0
		for frontgear in gears_front:
			for reargear in gears_rear:
				gearsetspeed_low = 45*(float(frontgear)/float(reargear))*(wheel_circ/1000000.)*60.
				gearsetspeed_high= 125*(float(frontgear)/float(reargear))*(wheel_circ/1000000.)*60.
				if np.mod(g,2) == 0:
					py.plot([gearsetspeed_low,gearsetspeed_high],[45,125],'--',lw=0.3,color='0.33')
					py.text(gearsetspeed_high,125+4,'%d:%d' %(frontgear,reargear),rotation=90,va='center',ha='center',size=7,color='0.33')
				else:
					py.plot([gearsetspeed_low,gearsetspeed_high],[45,125],'-',lw=0.3,color='0.33')
					py.text(gearsetspeed_low,45-4,'%d:%d' %(frontgear,reargear),rotation=90,va='center',ha='center',size=7,color='0.33')
			g = g+1
	sc = py.scatter(speed[(speed>1) & (time_step<time_step_threshold)],cad[(speed>1) & (time_step<time_step_threshold)],c=hr[(speed>1) & (time_step<time_step_threshold)],s=3,edgecolors='none',cmap=cmap)
	#sc.set_alpha(0.75)
	py.subplots_adjust(left=0.1,right=0.9)
	cb = py.colorbar(fraction=0.05,pad=0.05)
	cb.set_label('Heart rate (BPM)')
	py.xlabel('Speed (km/h)')
	py.ylabel('Cadence (RPM)')
	if comparisonmode == 'yes': # this makes the plots identical, so for comparing workouts, it is handy
		cb.set_clim((bins_hrzones[1],bins_hrzones[6]))
		py.xlim((plots_speed_min,plots_speed_max))
		py.ylim((plots_cad_min,plots_cad_max))
	plotfilename = './'+workfoldername+'/3Dplots_sch_'+str(plottype)+'_'+str(id)+'_'+str(garminconnectid)+'.png'
	py.savefig(plotfilename,dpi=150)
	py.close()
	#py.show()
	
	fig_size = [12.5,10]
	params = {'axes.labelsize': 12,'text.fontsize': 12,'legend.fontsize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12, 'figure.figsize': fig_size}
	py.rcParams.update(params)
	py.clf()
	cmap=py.set_cmap('Paired')
	sc = py.scatter(cad[(speed>1) & (time_step<time_step_threshold)],hr[(speed>1) & (time_step<time_step_threshold)],c=speed[(speed>1) & (time_step<time_step_threshold)],s=6,edgecolors='none',cmap=cmap)
	#sc.set_alpha(0.75)
	py.subplots_adjust(left=0.1,right=0.9)
	cb = py.colorbar(fraction=0.05,pad=0.05)
	cb.set_label('Speed (km/h)')
	py.xlabel('Cadence (RPM)')
	py.ylabel('Heart rate (BPM)')
	if comparisonmode == 'yes': # this makes the plots identical, so for comparing workouts, it is handy
		cb.set_clim((plots_speed_min,plots_speed_max))
		py.xlim((plots_cad_min,plots_cad_max))
		py.ylim((plots_hr_min,plots_hr_max))
	plotfilename = './'+workfoldername+'/3Dplots_chs_'+str(plottype)+'_'+str(id)+'_'+str(garminconnectid)+'.png'
	py.savefig(plotfilename,dpi=150)
	py.close()
	#py.show()
	
	#Times (and percentages) in the given zones
	(time_in_hrzones, hrzones) = np.histogram(hr[(speed>1) & (time_step<time_step_threshold)],weights = time_step[(speed>1) & (time_step<time_step_threshold)], bins=bins_hrzones)
	time_in_hrzones_percent = time_in_hrzones/(time_moving/100)
	
	####################
	#print workout data#
	####################
	print 'Total distance: %15.2f km' %(distance_total)
	logfile.write('Total distance: %15.2f km\n' %(distance_total))
	print 'Moving time:            %02d h %02.2d m %0.2d s ' %(decdeg2dms(time_moving)[0],decdeg2dms(time_moving)[1],decdeg2dms(time_moving)[2])
	logfile.write('Moving time:            %02d h %02.2d m %0.2d s\n' %(decdeg2dms(time_moving)[0],decdeg2dms(time_moving)[1],decdeg2dms(time_moving)[2]))
	print 'Total time:             %02d h %02.2d m %0.2d s ' %(decdeg2dms(time_total)[0],decdeg2dms(time_total)[1],decdeg2dms(time_total)[2])
	logfile.write('Total time:             %02d h %02.2d m %0.2d s\n' %(decdeg2dms(time_total)[0],decdeg2dms(time_total)[1],decdeg2dms(time_total)[2]))
	print 'Resting time:           %02d h %02.2d m %0.2d s (%4.1f percent of total time)' %(decdeg2dms(time_stopped)[0],decdeg2dms(time_stopped)[1],decdeg2dms(time_stopped)[2],time_stopped_percent)
	logfile.write('Resting time:           %02d h %02.2d m %0.2d s (%4.1f percent of total time)\n' %(decdeg2dms(time_stopped)[0],decdeg2dms(time_stopped)[1],decdeg2dms(time_stopped)[2],time_stopped_percent))
	print 'Average moving speed: %9.2f km/h' %(speed_average_moving)
	logfile.write('Average moving speed: %9.2f km/h\n' %(speed_average_moving))
	print 'Maximum speed: %15.1f  km/h' %(speed_max)
	logfile.write('Maximum speed: %15.1f  km/h\n' %(speed_max))
	print 'Average heart rate: %10.1f  BPM' %(hr_average_moving)
	logfile.write('Average heart rate: %10.1f  BPM\n' %(hr_average_moving))
	print 'Maximum heart rate: %8d    BPM' %(hr_max)
	logfile.write('Maximum heart rate: %8d    BPM\n' %(hr_max))
	print 'Time in HRZ0 [%3d:%3d]: %02d h %02.2d m %0.2d s (%4.1f percent of moving time)' %(hrzones[0],hrzones[1],decdeg2dms(time_in_hrzones[0])[0],decdeg2dms(time_in_hrzones[0])[1],decdeg2dms(time_in_hrzones[0])[2],time_in_hrzones_percent[0])
	logfile.write('Time in HRZ0 [%3d:%3d]: %02d h %02.2d m %0.2d s (%4.1f percent of moving time)\n' %(hrzones[0],hrzones[1],decdeg2dms(time_in_hrzones[0])[0],decdeg2dms(time_in_hrzones[0])[1],decdeg2dms(time_in_hrzones[0])[2],time_in_hrzones_percent[0]))
	print 'Time in HRZ1 [%3d:%3d]: %02d h %02.2d m %0.2d s (%4.1f percent of moving time)' %(hrzones[1],hrzones[2],decdeg2dms(time_in_hrzones[1])[0],decdeg2dms(time_in_hrzones[1])[1],decdeg2dms(time_in_hrzones[1])[2],time_in_hrzones_percent[1])
	logfile.write('Time in HRZ1 [%3d:%3d]: %02d h %02.2d m %0.2d s (%4.1f percent of moving time)\n' %(hrzones[1],hrzones[2],decdeg2dms(time_in_hrzones[1])[0],decdeg2dms(time_in_hrzones[1])[1],decdeg2dms(time_in_hrzones[1])[2],time_in_hrzones_percent[1]))
	print 'Time in HRZ2 [%3d:%3d]: %02d h %02.2d m %0.2d s (%4.1f percent of moving time)' %(hrzones[2],hrzones[3],decdeg2dms(time_in_hrzones[2])[0],decdeg2dms(time_in_hrzones[2])[1],decdeg2dms(time_in_hrzones[2])[2],time_in_hrzones_percent[2])
	logfile.write('Time in HRZ2 [%3d:%3d]: %02d h %02.2d m %0.2d s (%4.1f percent of moving time)\n' %(hrzones[2],hrzones[3],decdeg2dms(time_in_hrzones[2])[0],decdeg2dms(time_in_hrzones[2])[1],decdeg2dms(time_in_hrzones[2])[2],time_in_hrzones_percent[2]))
	print 'Time in HRZ3 [%3d:%3d]: %02d h %02.2d m %0.2d s (%4.1f percent of moving time)' %(hrzones[3],hrzones[4],decdeg2dms(time_in_hrzones[3])[0],decdeg2dms(time_in_hrzones[3])[1],decdeg2dms(time_in_hrzones[3])[2],time_in_hrzones_percent[3])
	logfile.write('Time in HRZ3 [%3d:%3d]: %02d h %02.2d m %0.2d s (%4.1f percent of moving time)\n' %(hrzones[3],hrzones[4],decdeg2dms(time_in_hrzones[3])[0],decdeg2dms(time_in_hrzones[3])[1],decdeg2dms(time_in_hrzones[3])[2],time_in_hrzones_percent[3]))
	print 'Time in HRZ4 [%3d:%3d]: %02d h %02.2d m %0.2d s (%4.1f percent of moving time)' %(hrzones[4],hrzones[5],decdeg2dms(time_in_hrzones[4])[0],decdeg2dms(time_in_hrzones[4])[1],decdeg2dms(time_in_hrzones[4])[2],time_in_hrzones_percent[4])
	logfile.write('Time in HRZ4 [%3d:%3d]: %02d h %02.2d m %0.2d s (%4.1f percent of moving time)\n' %(hrzones[4],hrzones[5],decdeg2dms(time_in_hrzones[4])[0],decdeg2dms(time_in_hrzones[4])[1],decdeg2dms(time_in_hrzones[4])[2],time_in_hrzones_percent[4]))
	print 'Time in HRZ5 [%3d:%3d]: %02d h %02.2d m %0.2d s (%4.1f percent of moving time)' %(hrzones[5],hrzones[8],decdeg2dms(time_in_hrzones[5]+time_in_hrzones[6]+time_in_hrzones[7])[0],decdeg2dms(time_in_hrzones[5]+time_in_hrzones[6]+time_in_hrzones[7])[1],decdeg2dms(time_in_hrzones[5]+time_in_hrzones[6]+time_in_hrzones[7])[2],time_in_hrzones_percent[5]+time_in_hrzones_percent[6]+time_in_hrzones_percent[7])
	logfile.write('Time in HRZ5 [%3d:%3d]: %02d h %02.2d m %0.2d s (%4.1f percent of moving time)\n' %(hrzones[5],hrzones[8],decdeg2dms(time_in_hrzones[5]+time_in_hrzones[6]+time_in_hrzones[7])[0],decdeg2dms(time_in_hrzones[5]+time_in_hrzones[6]+time_in_hrzones[7])[1],decdeg2dms(time_in_hrzones[5]+time_in_hrzones[6]+time_in_hrzones[7])[2],time_in_hrzones_percent[5]+time_in_hrzones_percent[6]+time_in_hrzones_percent[7]))
	print 'Time in HRZ5a[%3d:%3d]: %02d h %02.2d m %0.2d s (%4.1f percent of moving time)' %(hrzones[5],hrzones[6],decdeg2dms(time_in_hrzones[5])[0],decdeg2dms(time_in_hrzones[5])[1],decdeg2dms(time_in_hrzones[5])[2],time_in_hrzones_percent[5])
	logfile.write('Time in HRZ5a[%3d:%3d]: %02d h %02.2d m %0.2d s (%4.1f percent of moving time)\n' %(hrzones[5],hrzones[6],decdeg2dms(time_in_hrzones[5])[0],decdeg2dms(time_in_hrzones[5])[1],decdeg2dms(time_in_hrzones[5])[2],time_in_hrzones_percent[5]))
	print 'Time in HRZ5b[%3d:%3d]: %02d h %02.2d m %0.2d s (%4.1f percent of moving time)' %(hrzones[6],hrzones[7],decdeg2dms(time_in_hrzones[6])[0],decdeg2dms(time_in_hrzones[6])[1],decdeg2dms(time_in_hrzones[6])[2],time_in_hrzones_percent[6])
	logfile.write('Time in HRZ5b[%3d:%3d]: %02d h %02.2d m %0.2d s (%4.1f percent of moving time)\n' %(hrzones[6],hrzones[7],decdeg2dms(time_in_hrzones[6])[0],decdeg2dms(time_in_hrzones[6])[1],decdeg2dms(time_in_hrzones[6])[2],time_in_hrzones_percent[6]))
	print 'Time in HRZ5c[%3d:%3d]: %02d h %02.2d m %0.2d s (%4.1f percent of moving time)' %(hrzones[7],hrzones[8],decdeg2dms(time_in_hrzones[7])[0],decdeg2dms(time_in_hrzones[7])[1],decdeg2dms(time_in_hrzones[7])[2],time_in_hrzones_percent[7])
	logfile.write('Time in HRZ5c[%3d:%3d]: %02d h %02.2d m %0.2d s (%4.1f percent of moving time)\n' %(hrzones[7],hrzones[8],decdeg2dms(time_in_hrzones[7])[0],decdeg2dms(time_in_hrzones[7])[1],decdeg2dms(time_in_hrzones[7])[2],time_in_hrzones_percent[7]))
	print 'Time in HRZ6 [%3d:%3d]: %02d h %02.2d m %0.2d s (%4.1f percent of moving time)' %(hrzones[8],hrzones[9],decdeg2dms(time_in_hrzones[8])[0],decdeg2dms(time_in_hrzones[8])[1],decdeg2dms(time_in_hrzones[8])[2],time_in_hrzones_percent[8])
	logfile.write('Time in HRZ6 [%3d:%3d]: %02d h %02.2d m %0.2d s (%4.1f percent of moving time)\n' %(hrzones[8],hrzones[9],decdeg2dms(time_in_hrzones[8])[0],decdeg2dms(time_in_hrzones[8])[1],decdeg2dms(time_in_hrzones[8])[2],time_in_hrzones_percent[8]))
	print 'Average cadence: %13.1f  RPM' %(cad_average_nonzero)
	logfile.write('Average cadence: %13.1f  RPM\n' %(cad_average_nonzero))
	print 'Maximum cadence: %11d    RPM' %(cad_max)
	logfile.write('Maximum cadence: %11d    RPM\n' %(cad_max))
	print 'Rolling time:           %02d h %02.2d m %0.2d s (%4.1f percent of moving time)' %(decdeg2dms(time_moving_zerocadence)[0],decdeg2dms(time_moving_zerocadence)[1],decdeg2dms(time_moving_zerocadence)[2],time_moving_zerocadence_percent)
	logfile.write('Rolling time:           %02d h %02.2d m %0.2d s (%4.1f percent of moving time)\n' %(decdeg2dms(time_moving_zerocadence)[0],decdeg2dms(time_moving_zerocadence)[1],decdeg2dms(time_moving_zerocadence)[2],time_moving_zerocadence_percent))
	#calculate TSS (Training Stress Score, which is described here: http://home.trainingpeaks.com/articles/cycling/estimating-training-stress-score-(tss)-by-joe-friel.aspx, while the categories are taken from http://home.trainingpeaks.com/articles/cycling/normalized-power,-intensity-factor,-training-stress-score.aspx)
	bins_hrzones_tss = [bins_hrzones[0],bins_hrzones[1],bins_hrzones[1]+(1./3.)*(bins_hrzones[2]-bins_hrzones[1]),bins_hrzones[1]+(2./3.)*(bins_hrzones[2]-bins_hrzones[1]),bins_hrzones[2],bins_hrzones[2]+(1./2.)*(bins_hrzones[3]-bins_hrzones[2]),bins_hrzones[3],bins_hrzones[4],bins_hrzones[5],bins_hrzones[6],bins_hrzones[7],bins_hrzones[8],bins_hrzones[9]] #this is needed for the 3 and 2 subzones in HRZ1 and HRZ2
	(time_in_hrzones_tss, hrzones_tss) = np.histogram(hr[(speed>1) & (time_step<time_step_threshold)],weights = time_step[(speed>1) & (time_step<time_step_threshold)], bins=bins_hrzones_tss)
	tss = 10*time_in_hrzones_tss[0]+20*time_in_hrzones_tss[1]+30*time_in_hrzones_tss[2]+40*time_in_hrzones_tss[3]+50*time_in_hrzones_tss[4]+60*time_in_hrzones_tss[5]+70*time_in_hrzones_tss[6]+80*time_in_hrzones_tss[7]+100*time_in_hrzones_tss[8]+120*time_in_hrzones_tss[9]+140*time_in_hrzones_tss[10]
	tss_perhour = tss/time_moving #a TSS of 100 is a one hour ride at FTP, so TSS/h can not be higher than 100 for rides longer than 1 hour!
	tss_intensity = (tss_perhour/100.)**(0.5) #this basically gives the average power for the ride in units of the FTP power (ref.: http://www.twowheelblogs.com/what-does-100-tss-mean)
	tss_work = tss_intensity*time_moving #this is the work in units of FTP power Wh
	if tss >= 0:
		tss_category = 'low training load'
	if tss >= 150:
		tss_category = 'medium training load'
	if tss >= 300:
		tss_category = 'high training load'
	if tss >= 450:
		tss_category = 'very high training load'
	print 'TSS (est.): %18.1f  points (%s)' %(tss,tss_category)
	logfile.write('TSS (est.): %18.1f  points (%s)\n' %(tss,tss_category))
	print 'TSS/h (est.): %16.1f  points/h' %(tss_perhour)
	logfile.write('TSS/h (est.): %16.1f  points/h\n' %(tss_perhour))
	print 'Average power (est.): %9.2f*FTP' %(tss_intensity)
	logfile.write('Average power (est.): %9.2f*FTP\n' %(tss_intensity))
	print 'Work (est.): %18.2f*FTPh' %(tss_work)
	logfile.write('Work (est.): %18.2f*FTPh\n' %(tss_work))
	logfile.close()
