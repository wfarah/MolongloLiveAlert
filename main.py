import numpy as np
import threading,Queue
import logging
from time import sleep
from time import time as timeNow
from datetime import datetime
import Stage1Filter
import pylab as plb


#Globals:
pulsar_monitor_on = True			#If True, Pulsar monitor is switched on, False otherwise
pulsar_refresh_time = 1000		#Refresh time of pulsar monitor thread
utc_directory = "/data/mopsr/archives/"
plotting_thread_on = True			#If True, plotting thread will be activated


def observing():
	""""Function that returns a bool whether observation is running"""
	return True


def updatePulsarList(queue,pulsar_refresh_time,boresight_ra,boresight_dec,
			utc='Now'):
	""" Function to be executed by a thread, computes whether pulsar might be visible in a fanbeam.
	    Goes through an infinite loop, executes every 'pulsar_refresh_time', and puts output in queue,
	    (replaces in case queue is not empty)
		Args:
			queue (Queue.Queue instance): Queue that holds pulsar object (see Stage1Filter.py)
			pulsar_refresh_time (int): Wait time for 
			boresight_ra (float): RA of boresight
			boresight_dec (float): Dec of boresight
		Returns:
			(None)
			
	"""
	print "Pulsar List Thread spawned"
	while True:
		sleep(pulsar_refresh_time)
		print "Thread woke up, fetching data"
		if utc=='Now':
			utc = datetime.utcnow()
			utc = datetimeToStr(utc)
		if queue.empty():	#Ensures only 1 pulsar list in queue
			pulsar_list = Stage1Filter.getPotentialPulsars(utc,boresight_ra,boresight_dec)
			queue.put(pulsar_list)
			queue.task_done()
		else:
			pulsar_list = Stage1Filter.getPotentialPulsars(utc,boresight_ra,boresight_dec)
			_ = queue.get()
			queue.put(pulsar_list)
			queue.task_done()
		print "task done, data in queue"

def datetimeToStr(datetime_utc):
	"""Function that convert datetime object to str with specific format"""
	fmt = "%Y-%m-%d-%H:%M:%S"
	return datetime.strftime(datetime_utc,fmt)


def getBoresight(utc):
        """Function that returns the boresight of telescope at the particular observation.
	
	Args:
		utc (str): format yyyy-mm-dd-hh-mm-ss should be the start UTC of observation!
	
	Returns:
		ra (str): RA of boresight
		dec (str): Dec of boresight
	"""
        command="/home/wfarah/soft/print_ObsInfo "+utc
        output = Stage1Filter._cmdline(command)
        output = output.split("\n")
        ra,dec=None,None
        for line in output:
                line=line.split()
                if 'RA' in line:
                        ra=line[1]
                elif 'DEC' in line:
                        dec=line[1]
                        break
        if (ra,dec)==(None,None):
                raise KeyError("'"+utc+"' is not a valid start observation UTC")
        else:
                return ra,dec


def getObsInfo(utc):
	"""Funtion that reads obs.info file from obs directory

	Args:
		utc (str): format yyyy-mm-dd-hh-mm-ss should be the start UTC of observation!

	Returns:
		obs_info (dict): Dictionary with keys being the first column of 
						 obs.info file, and values the second column
	"""
	obs_info={}
	with open(utc_directory+utc+"/obs.info") as o:
		for line in o:
			if line[0] not in ["\n","#"]:
				i = line.split(" ")
				if i[-1].rstrip():
					obs_info[i[0]] = i[-1].rstrip()
				else:
					obs_info[i[0]] = " "
	return obs_info

def getCandidates(i):
	"""Function that returns HEIMDAL candidates to main"""
	mask=np.where((all_candidates[:,2]>i) & (all_candidates[:,2]<i+4))
	print "From time %s to time %s" %(all_candidates[:,2][mask].max(),all_candidates[:,2][mask].min())
	return all_candidates[mask]


def realTimePlot(queue):
	plot_array = []
	eps=10**(-2)
	plb.ion()
	while True:
		sleep(0.05)
		plot_array.append(queue.get()+eps)
		plb.clf()
		plb.bar(range(1,1+len(plot_array)),plot_array)
		plb.title("Candidate Rate")
		#plb.ylim(-1,np.max(PlotArray))
		plb.draw()


###############################################################################
#######################      	     MAIN               #######################
###############################################################################

if __name__=="__main__":
	all_candidates=np.loadtxt("/data/mopsr/results/2016-07-16-08:16:21/all_candidates.dat")
	start_utc = "2016-07-16-08:16:21"
	obs_info = getObsInfo(start_utc)
	boresight_ra , boresight_dec = obs_info['RA'] , obs_info['DEC'] 
	utc_now = datetimeToStr(datetime.utcnow())
	if pulsar_monitor_on:
		pulsar_list = Stage1Filter.getPotentialPulsars(utc_now,boresight_ra,boresight_dec)
		
		pulsar_list_queue = Queue.Queue()
		pulsar_monitor_thread = threading.Thread(name = 'PulsarMonitorThread', 
				   target=updatePulsarList,
				   args=(pulsar_list_queue,pulsar_refresh_time,boresight_ra,boresight_dec,))
		pulsar_monitor_thread.setDaemon(True)
	else:
		pulsar_list = []
	
	if plotting_thread_on:
		import pylab as plb
		plotting_queue = Queue.Queue()
		plotting_thread = threading.Thread(name = 'realTimePlot', target=realTimePlot,
				   args=(plotting_queue,))
		plotting_thread.setDaemon(True)
	
	#Listens for socket to commence observation
	#start_observing = socket_listen	
	if pulsar_monitor_on:
		pulsar_monitor_thread.start()
	if plotting_thread_on:
		plotting_thread.start()
	t=timeNow()		#Testing
	i=0			#Testing
	observing = True	#Testing
	lst=[]			#Testing
	cands=[]
	while observing:
		#print "iterating"
		sleep(0.001)
		if not pulsar_list_queue.empty() and pulsar_monitor_on:
			pulsar_list = pulsar_list_queue.get()
			print "Got data from Pulsar thread"
		else:
			pass
			#print "no data available yet"
		utc_now = datetime.utcnow()
		#if timeNow()-t > 0.5:
		heimdal_candidates = getCandidates(i) # 4 second Heimdal candidates
		i+=4
		print "Got Heimdal Candidates"
		#t=timeNow()
		#else:
		#heimdal_candidates = "empty"
		if heimdal_candidates is not "empty":
			filtered_candidates = Stage1Filter.candidateFilter(heimdal_candidates,pulsar_list)
			if filtered_candidates is not None:
				print filtered_candidates.shape
				lst.append(filtered_candidates.shape[0])
				cands.append(filtered_candidates)
				if plotting_thread_on:
					plotting_queue.put(filtered_candidates.shape[0])
			else:
				lst.append(0)
				#plotting_queue.put(0)
			print "Processing"
		"""Candidates after the first stage filter"""
		
	
		#for candidate in filtered_candidates:
		


#if __name__=="__main__":
#	main()
