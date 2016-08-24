import sys
sys.path.append("/home/wfarah/MolongloLiveAlert/")

import numpy as np
from Stage1Filter import thresholdFilter
import time
from multiprocessing import Queue,Process
from Stage2Filter import process_candidate
import threading
import fil_functions
import pylab as plb

def width_range(w):
	width_range=np.linspace(2**(w-1),2**(w+1),7)
	width_range=np.round(width_range)
	width_range=np.unique(width_range)
	width_range=np.array(width_range,np.int)
	print width_range


in_queue = Queue()
out_queue = Queue()

utc = "2016-06-08-12:15:12"
utc = "2016-08-16-20:07:21"
utc = "2016-08-05-07:48:39"
"""
#candidate = np.array([  1.75671000e+01,   3.07590000e+04,   2.01582000e+01,   8.00000000e+00,  4.40000000e+01,   8.84492000e+01,   4.00000000e+00,   3.07580000e+04, 3.07630000e+04,   2.00000000e+00,   1.00000000e+00,   2.00889000e+01,1.77000000e+02])
candidate = np.array([  2.74032000e+01,   3.07600000e+04,   2.01589000e+01,   2.00000000e+00,  4.40000000e+01,   8.84492000e+01,   3.50000000e+02,   3.07570000e+04,	     3.07640000e+04,   1.00000000e+00,   1.77000000e+02,   2.74032000e+01,    1.77000000e+02])
in_queue.put([candidate,utc])

process_candidate(in_queue,out_queue)
"""
p = Process(target = process_candidate, args = (in_queue,out_queue))
p.start()

all_candidates = np.loadtxt("/data/mopsr/results/"+utc+"/all_candidates.dat")
#f_candidates = thresholdFilter(all_candidates)

#k = f_candidates[3]
k = np.array([14.32,51072,51072*0.00065536,3,-1,58.1129,-1,-1,-1,-1,-1,-1,177])
#dm_list = np.loadtxt("/home/wfarah/dm_list.dat")

#in_queue.put([k,utc])
"""
for i in range(dm_list[37],len(dm_list)):
	k[5] = dm_list[i]
	in_queue.put([k,utc])
	time.sleep(0.01)
#p.terminate()
"""
