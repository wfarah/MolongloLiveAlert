import numpy as np
import threading,Queue
from time import sleep
from time import time as TimeNow
from datetime import datetime
import Stage1Filter
import pylab as plb


#Globals:
PulsarRefreshTime = 10		#Refresh time of pulsar monitor thread


def Observing():
	""""Function that returns a bool whether observation is running"""
	return True


def UpdatePulsarList(queue,PulsarRefreshTime,boresight_ra,boresight_dec,
			utc='Now'):
	print "Pulsar List Thread spawned"
	while True:
		sleep(PulsarRefreshTime)
		print "Thread woke up, fetching data"
		if utc=='Now':
			utc = datetime.utcnow()
			utc = FormatUTC(utc)
		if queue.empty():	#Ensures only 1 pulsar list in queue
			pulsar_list = Stage1Filter.PotentialPulsars(utc,boresight_ra,boresight_dec)
			queue.put(pulsar_list)
			queue.task_done()
		else:
			pulsar_list = Stage1Filter.PotentialPulsars(utc,boresight_ra,boresight_dec)
			_ = queue.get()
			queue.put(pulsar_list)
			queue.task_done()
		print "task done, data in queue"

def FormatUTC(DatetimeUTC):
	"""Function that convert datetime object to str with specific format"""
	fmt = "%Y-%m-%d-%H:%M:%S"
	return datetime.strftime(DatetimeUTC,fmt)


def get_boresight(utc):
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


def get_candidates(i):
	"""Function that returns HEIMDAL candidates to main"""
	mask=np.where((all_candidates[:,2]>i) & (all_candidates[:,2]<i+4))
	print "From time %s to time %s" %(all_candidates[:,2][mask].max(),all_candidates[:,2][mask].min())
	return all_candidates[mask]


def RealTimePlot(queue):
	PlotArray = []
	plb.ion()
	while True:
		sleep(0.05)	
		PlotArray.append(queue.get())
		plb.clf()
		plb.bar(range(1,1+len(PlotArray)),PlotArray)
		plb.title("Candidate Rate")
		plb.ylim(-1,np.max(PlotArray))
		plb.draw()


###############################################################################
####################### 	     MAIN               #######################
###############################################################################

all_candidates=np.loadtxt("/data/mopsr/results/2016-07-16-08:16:21/all_candidates.dat")
if __name__=="__main__":
	start_utc = "2016-07-16-08:16:21"
	boresight_ra , boresight_dec = get_boresight(start_utc)
	utc_now = datetime.utcnow()
	utc_now = FormatUTC(utc_now)
	PulsarList = Stage1Filter.PotentialPulsars(utc_now,boresight_ra,boresight_dec)
	
	PulsarListQueue = Queue.Queue()
	PulsarMonitorThread = threading.Thread(name = 'PulsarMonitorThread', 
			   target=UpdatePulsarList,
			   args=(PulsarListQueue,PulsarRefreshTime,boresight_ra,boresight_dec,))
	PulsarMonitorThread.setDaemon(True)
	
	PlottingQueue = Queue.Queue()
	PlottingThread = threading.Thread(name = 'RealTimePlot', target=RealTimePlot,
			   args=(PlottingQueue,))
	PlottingThread.setDaemon(True)
	
	#Listens for socket to commence observation
	#start_observing = socket_listen	
	
	PulsarMonitorThread.start()
	PlottingThread.start()
	t=TimeNow()		#Testing
	i=0			#Testing
	observing = True	#Testing
	while Observing:
		#print "iterating"
		sleep(0.001)
		if not PulsarListQueue.empty():
			PulsarList = PulsarListQueue.get()
			print "Got data from Pulsar thread"
		else:
			pass
			#print "no data available yet"
		utc_now = datetime.utcnow()
		if TimeNow()-t > 4:
			HeimdalCandidates = get_candidates(i) # 4 second Heimdal candidates
			i+=4
			print "Got Heimdal Candidates"
			t=TimeNow()
		else:
			HeimdalCandidates = "empty"
		if HeimdalCandidates is not "empty":
			filtered_candidates = Stage1Filter.CandidateFilter(HeimdalCandidates,PulsarList)
			if filtered_candidates is not None:
				print filtered_candidates.shape
				PlottingQueue.put(filtered_candidates.shape[0])
			else:
				PlottingQueue.put(0)
			print "Processing"
		"""Candidates after the first stage filter"""
		
	
		#for candidate in filtered_candidates:
		


#if __name__=="__main__":
#	main()
