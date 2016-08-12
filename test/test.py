import numpy as np
from Stage1Filter import ThresholdFilter
import time
from multiprocessing import Queue,Process
#import fil_functions

execfile("Stage2Filter.py")
utc = "2016-08-04-05:25:19"

all_candidates = np.loadtxt("/data/mopsr/archives/"+utc+"/all_candidates.dat")
f_candidates = ThresholdFilter(all_candidates)
"""
for i in range(f_candidates.shape[0]):
	process_candidate(q,f_candidates[i])
	if not q.empty():
		k=q.get()
		block=fil_functions.get_fil(utc=k[13],beam=k[0],sample=k[1]-200)
		plb.imshow(block,interpolation='nearest',aspect='auto')
		plb.show()

def run_1core():
	t=time.time()
	for candidate in f_candidates:
		process_candidate(candidate)
	print time.time() - t

def run_cores(n=6):
	t = time.time()
	pool = Pool(processes = n)
	k = pool.map(process_candidate,f_candidates)
	print time.time() - t
	return k	

"""

def terminate_procs(proc_list):
	for proc in proc_list:
		proc.terminate()

n_processes = 4

in_queue = Queue()
out_queue = Queue()

p_list = [Process(target = process_candidate, args = (in_queue,out_queue)) for i in range(n_processes)]

for i in p_list:
	i.start()


