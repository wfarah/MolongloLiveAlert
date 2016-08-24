import sys
sys.path.append("/home/wfarah/MolongloLiveAlert/")

import numpy as np
from Stage1Filter import thresholdFilter
import time
from multiprocessing import Queue,Process
from Stage2Filter import process_candidate
import threading
#import fil_functions
#execfile("/home/wfarah/MolongloLiveAlert/Stage2Filter.py")
from Stage2Filter import process_candidate


def terminate_procs(proc_list):
	for proc in proc_list:
		proc.terminate()

def saving_thread(out_queue):
	print "Saving queue initiated"
	while True:
		k = out_queue.get()
		if k != "poison_pill":
			k = np.array(k).tolist()
			o.write(" ".join(k)+"\n")
		else:
			return

def candidate_counter(in_queue):
	while in_queue.qsize() != 0:
		print "%i Candidates still in queue" %in_queue.qsize()
		time.sleep(5)
	return

o = open("output.txt","a")
o.write("#FB samp H_DM H_W H_SNR DM W SNR SNR_0 F1 F2 F3 index utc MEANf1 MEANf2 MEANf3 MEANr STDf1 STDf2 STDf3 STDr SN_zap n_zap Mod_ind Mod_indT\n")

n_processes = 1

in_queue = Queue()
out_queue = Queue()

p_list = [Process(target = process_candidate, args = (in_queue,out_queue)) for i in range(n_processes)]

for i in p_list:
	i.start()


savingThread = threading.Thread(name = 'SavingThread',
		target=saving_thread,
		args=(out_queue,))
savingThread.setDaemon(True)
savingThread.start()

coutingThread = threading.Thread(name = 'CountingThread',
		target=candidate_counter,
		args=(in_queue,))


utc_list = np.loadtxt("utc_list",np.str)
print "started"
for utc in utc_list:
	print utc
	all_candidates = np.loadtxt("/data/mopsr/results/"+utc+"/all_candidates.dat")
	f_candidates = thresholdFilter(all_candidates)
	for cand in f_candidates:
		in_queue.put([cand,utc])


while in_queue.qsize() != 0:
	print "%i Candidates still in queue" %in_queue.qsize()
	time.sleep(5)

if in_queue.qsize() == 0:
	out_queue.put("poison_pill")
print "waiting for save queue"
savingThread.join()
assert out_queue.empty() and in_queue.empty()

print "Save complete"
o.close()

terminate_procs(p_list)

"""
loop = True
while loop:
	if in_queue.empty():
		time.sleep(1)
		print "Processing finished\nSaving..."
		o = open("output.txt","a")
		loop = False
		while not out_queue.empty():
			time.sleep(0.001)
			k = np.array(out_queue.get()).tolist()
			o.write(" ".join(k)+"\n")
	else:
		time.sleep(1)
"""
