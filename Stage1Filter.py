import numpy as np
from subprocess import PIPE, Popen
import argparse
import datetime
import time



"""Globals:"""
directory="/home/wfarah/pulsar_fanbeam/"
ns_threshold = 5	#Max difference in degrees between pulsar and boresight NS
md_threshold = 2.5	#Max difference in degrees between pulsar and boresight MD
PulsarDatabase = np.loadtxt(directory+"psrcat.dat",np.str)   #Pulsars with S400>30 mJy or S1400>5 mJy, and Dec < 20 deg
sn_threshold=9
boxcar_threshold=6
dm_threshold=50


def _cmdline(command):	
    """Function that captures output from screen
    """
    process = Popen(
        args=command,
        stdout=PIPE,
        shell=True
    )
    return process.communicate()[0]



def _get_nsmd(utc,ra,dec):
	"""Function that returns ns and md coordinates of boresight
	"""
        dec="-- "+dec[:]       #This is done for the mopsr_getns command. 
	cmd_getns="mopsr_getns "+utc+" "+ra+" "+dec
	cmd_getmd="mopsr_getmd "+utc+" "+ra+" "+dec
	#print cmd_getns,cmd_getmd
	return float(_cmdline(cmd_getns)),float(_cmdline(cmd_getmd)) #captures mopsr_getns output from screen


class PulsarObject:
        def __init__(self,pulsar_info):
                self.name=pulsar_info[0]
                self.ra=pulsar_info[1]	
                self.dec=pulsar_info[2]			
                self.dm=float(pulsar_info[3])
                self.fb=int(np.round(float(pulsar_info[4])))



def PotentialPulsars(utc,boresight_ra,boresight_dec):
	"""Function that computes whether a potential pulsar might be in fanbeam
	
	Args:
		utc (str): Current UTC 
		boresight_ra (str): RA of boresight
		boresight_dec (str): Dec of boresight
	
	Returns:
		list: Potential PulsarObjest list
			(see PulsarObject class)
	"""
	telescope_ns,telescope_md = _get_nsmd(utc,boresight_ra,boresight_dec)
	t=time.time()
	IndexToFlag = []	#index of pulsars from pulsars_list to flag
	estimated_fb = []		#Fanbeam where a pulsar is estimated to be located
	i=0
	for PSRJ,RAJ,DECJ,DMJ in PulsarDatabase:
		#print PSRJ,RAJ,DECJ
		pulsar_ns,pulsar_md = _get_nsmd(utc,RAJ,DECJ)
		if np.abs(telescope_ns-pulsar_ns)<ns_threshold and np.abs(telescope_md-pulsar_md)<md_threshold:
			print pulsar_ns,pulsar_md
			IndexToFlag.append(i)
			estimated_fb.append(352.0*(((pulsar_md-telescope_md)/(2.0/np.cos(np.radians((pulsar_md+telescope_md)/2)))+1)/2.0) + 1)
		i+=1
	
	PulsarsToFlag=PulsarDatabase[IndexToFlag]
	t1=time.time()
	pulsar_list = np.column_stack((PulsarsToFlag,estimated_fb))
	PulsarList = [PulsarObject(i) for i in pulsar_list]
	return PulsarList


def CandidateFilter(candidates,PulsarList):
    """Function that performs stage 1 masking
    
    Returns:
            np.array: Array of the good candidates
    """
    GoodCandidates = ThresholdFilter(candidates)
    if len(PulsarList) == 0:
        """ No pulsars in FOV """
        return GoodCandidates
    else:
        mask = np.ones(len(GoodCandidates),np.bool)
	"""if pulsar is present, candidate index will be replaced by False"""
        i = 0
        for candidate in GoodCandidates:
            beam, H_dm = int(candidate[12]), candidate[5]
            for Pulsar in PulsarList:
                if CandidateIsPulsar(beam,H_dm,Pulsar):
                    mask[i] = False
                else:
                    pass
            i+=1
        return GoodCandidates[mask] #Great Candidates



def ThresholdFilter(candidates):
        """Function that returns candidates HEIMDAL's output above threshold
        
        Args:
                candidates (np.array): output line from HEIMDAL
			   (  None  ): If no candidates exist

        """
        mask = np.where((candidates[:,12]!=1) & (candidates[:,0] > sn_threshold) & 
	                (candidates[:,3] < boxcar_threshold) & (candidates[:,5] > dm_threshold))[0]
	if len(mask) != 0:
        	return candidates[mask]
	else:
		return None


def CandidateIsPulsar(beam,H_dm,Pulsar):
	"""Function that returns bool, whether pulsar in FB or not """
	if Pulsar.name is 'J0835-4510' and H_dm < 100:
		""" Vela alert! Discard all candidates with DM<100"""
		return True
	if (beam in range(Pulsar.fb-1,Pulsar.fb+2)) and (H_dm<1.2*Pulsar.dm and H_dm>0.8*Pulsar.dm):
		return True
	else:
		return False











if __name__=="__main__":
	utc = "2016-07-10-03:02:42"
	ra,dec = "11:57:15.240" , "-62:24:50.87"
	pulsar_list = PotentialPulsars(utc,ra,dec)
	print pulsar_list
	#np.savetxt(directory+utc+"_pulsar.sel",output,fmt="%s")
        candidates=np.loadtxt("/data/mopsr/results/2016-07-16-08:16:21/all_candidates.dat")
        print CandidateFilter(candidates,[])

