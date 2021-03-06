import numpy as np
from sigpyproc.Readers import FilReader
from multiprocessing import Pool
import datetime,os,time
from wrapper import dedisp
from wrapper import dedisp_norm_base_conv
from wrapper import norm_base_conv_sn0
from wrapper import load_block
from wrapper import delete_block




"""Globals"""
F1=range(7,13)
F2=range(14,19)
F3=range(20,25)
F_rst=[0,1,2,3,4,5,6,13,19,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
t_epoch=datetime.datetime(1970,1,1)

dm_list=np.loadtxt("/home/wfarah/dm_list.dat")
backstep=100
archiveDirectory="/data/mopsr/archives/"


def my_mean(dis,nstart,nend):
	m=np.ones(dis.shape[1],dtype=np.bool)
	for i in range(len(m)):
		if i in range(nstart,nend):
			m[i]=False
	return dis[:,m].mean(axis=1)

def my_std(dis,nstart,nend):
	m=np.ones(dis.shape[1],dtype=np.bool)
	for i in range(len(m)):
		if i in range(nstart,nend):
			m[i]=False
	return dis[:,m].std(axis=1)


def rms(x):
	"""Returns rms of the 1-D array x"""
	x=list(x)
	del x[(backstep-50):(backstep+100)]
	x=np.array(x)
	s=0
	for i in x:
		s+=i**2
	s/=x.size
	s=np.sqrt(s)
	return s

def MAD(filterbank_block,ax=1): #Frequency is axis 1 for filterbank files
	"""Takes in a 2D filterbank file, and computes the MAD of the frequency"""
	return np.median(abs(filterbank_block.T - np.median(filterbank_block,axis=ax)).T, axis=ax)

def normalize(time_series,how="mad"):
	"""Takes in time_series, and returns it normalized"""
	if how=="rms":
		return time_series/rms(time_series)
	if how=="mad":
		return time_series/(1.4826*MAD(time_series,ax=None))

def to_ms(nbin):
	"""Converts number of bins to ms"""
	return nbin*0.65536

def to_bin(t):
	"""Converts time in ms to number of bins"""
	return int(np.round(t/0.65536))

def remove_baseline(time_series,how="median",sig=1):
	"""Takes in time_series, and return it baseline subtracted"""
	if how=="sigma_clip":
		return time_series-np.mean(time_series[np.where(time_series<sig*np.std(time_series)+np.mean(time_series))[0]])
	elif how=="median":
		return time_series-np.median(time_series)

def convolve(time_series,width,start=backstep-50,bin_search=100):
	"""Convolves a time_series with a boxcar of a given width"""
	convolved_ts=np.zeros(time_series.size)
	for i in range(start,start+bin_search): #boxcar search from sample -50 to +50 (event centered at 0)
		ss=0
		for j in range(width):
			ss+=time_series[i+j]
		convolved_ts[i]=ss
	return convolved_ts

def fscrunch(x,start_ch,end_ch): #x should be dedispersed
	crunched=x[start_ch:end_ch+1]
	return crunched.sum(axis=1)

def get_power(event):
	ch1=fscrunch(event,F1[0],F1[len(F1)-1])
	ch2=fscrunch(event,F2[0],F2[len(F2)-1])
	ch3=fscrunch(event,F3[0],F3[len(F3)-1])
	ch4=fscrunch(event,31,32)
	ch1_sum=ch1.sum()
	ch2_sum=ch2.sum()
	ch3_sum=ch3.sum()
	ch4_sum=ch4.sum()
	all_ch=event.sum()
	return ch1_sum,ch2_sum,ch3_sum,ch4_sum,all_ch




def HUNTER(fil_block,dm,w,dispersion_delay):
	"""Hunts around dm and w (units power of 2, 2**w) for the best dm and width, and returns the refined dm and width, the SNR, SNR at 0 dm, start and end event gates"""
	dm_index=np.where(dm_list==dm)[0]
	if not dm_index:
		dm_range=np.linspace(dm*0.85,dm*1.15,15)
		print "DM is not on HEIMDAL DM list"
	elif dm_index==len(dm_list)-1:
		dm_range=np.linspace(dm_list[dm_index-1],dm_list[dm_index]*1.03,9)
	else:
		dm_range=np.linspace(dm_list[dm_index-1],dm_list[dm_index+1],9)
	width_range=np.linspace(2**(w-1),2**(w+1),7)
	width_range=np.round(width_range)
	width_range=np.unique(width_range)
	width_range=np.array(width_range,np.int)

	sn=np.zeros(len(width_range)*len(dm_range))
	nstart=np.zeros(len(width_range)*len(dm_range))
	nend=np.zeros(len(width_range)*len(dm_range))

	i=0
	for dm in dm_range:
		for width in width_range:
			#dis=fil_block.dedisperse(dm)
			#s=dis.sum(axis=0)
			#s=remove_baseline(s)
			#d=normalize(s,how="mad")
			#c=convolve(d,width)
            c = dedisp_norm_base_conv(width,dm)
			sn[i]=c.max()/np.sqrt(width)
			nstart[i]=c.argmax()
			nend[i]=c.argmax()+width
			i+=1
	sn=sn.reshape((len(dm_range),len(width_range)))
	nstart=nstart.reshape((len(dm_range),len(width_range)))
	nend=nend.reshape((len(dm_range),len(width_range)))
	try:
		nstart=int(nstart[np.where(sn==sn.max())])
		nend=int(nend[np.where(sn==sn.max())]) #bin excluded from event
		dm=dm_range[np.where(sn==sn.max())[0]][0]
		width=width_range[np.where(sn==sn.max())[1]][0]
		sn0=get_snr0(fil_block,width,dispersion_delay)
		return dm,width,sn.max(),sn0,nstart,nend
	except:
		return -1,-1,-1,-1,-1,-1

def get_snr0(fil_block,width,dispersion_delay):
	"""Returns the sn at 0 zero DM and specificied width, for a given Filterbank file block, and for a given dispersion delay in units of bins"""
	#dis=fil_block.dedisperse(0) #For SN at 0 dm
	#s=dis.sum(axis=0)
	#s=remove_baseline(s)
	#d=normalize(s,how="mad")
	#c=convolve(d,width,backstep,dispersion_delay)
	c = norm_base_conv_sn0(width,backstep,dispersion_delay)
    sn0=c.max()/np.sqrt(width)
	return sn0

def mod_index(event,t_crunch=False):    #event should be median subtracted, crunches in time when t_crunch is True
	"""Computes the modulation index of a 2D event window. Crunches time if t_crunch=Trues"""
	if t_crunch:
		event=event.sum(axis=1)
	return np.sqrt((event**2).mean()-(event.mean())**2)/event.mean()

def get_index(utc,sample):
	"""Returns a time stamp, given the utc and sample number, that is used roughly as a time index"""
	t_obs=datetime.datetime(int(utc[0:4]),int(utc[5:7]),int(utc[8:10]),int(utc[11:13]),int(utc[14:16]),int(utc[17:19]))
	t_event=t_obs+datetime.timedelta(seconds=sample*0.00065536)
	td=t_event-t_epoch
	total_seconds=(td.microseconds + (td.seconds + td.days * 24. * 3600.) * 10.**6.) / 10.**6.
	return np.log(total_seconds)*1000000-21100000

def get_snr(time_series,start,width): #x is a time series
	"""Computes the snr of a time series, given the start bin and the width (in bins)
	This is called snr in fil_functions
	"""
	a=remove_baseline(time_series)
	a=normalize(a)
	sn=np.sum(a[start:start+width])/np.sqrt(width)
	return sn



def process_candidate(in_queue,out_queue):
	print "%s Initiated" %os.getpid()
	while True:
		candidate = in_queue.get()
        if len(candidate) == 2:
            utc = candidate[1]
            candidate = candidate[0]
		H_sn , H_w, H_dm = candidate[0],candidate[3],candidate[5]
		sample_bin, H_t, beam = int(candidate[1]), float(candidate[2]), int(candidate[12])
		direc="/data/mopsr/archives/"+utc+"/BEAM_"+str(int(beam)).zfill(3)+"/"+utc+".fil"
		fil_file=FilReader(direc)
		dispersion_delay=(31.25*8.3*H_dm)/(0.840)**3
		dispersion_delay=np.ceil(dispersion_delay/655.36)
		w=2**H_w
		min_afterEvent=300
		time_extract=np.max([min_afterEvent,(dispersion_delay+w)*2])

		fil_block=fil_file.readBlock(sample_bin-backstep,int(backstep+time_extract))
        load_block(direc,sample_bin-backstep,int(backstep+time_extract))
		dis = dedisp(H_dm)
        dis=fil_block.dedisperse(H_dm)
		av_p=np.mean(dis)
		dis=dis-np.median(dis)
		event=dis[:,backstep-(2**H_w)/2:backstep+(2**H_w)/2]
		ch1,ch2,ch3,ch4,all_ch=get_power(event)
		if (ch1/all_ch>0.7 or ch2/all_ch>0.7 or ch3/all_ch>0.7 or ch1/all_ch+ch2/all_ch>0.9 or ch1/all_ch+ch3/all_ch>0.9 or ch2/all_ch+ch3/all_ch>0.9):
			delete_block()
            pass
		else:
			dm,width,sn,snr_0,nstart,nend=HUNTER(fil_block,H_dm,H_w,dispersion_delay)
			if sn == -1:
				print "Error at UTC: "+utc+", sample: "+str(sample_bin)
				print "Passing..."
				raise ValueError()
			dis=fil_block.dedisperse(dm)
			rms_freq=np.hstack((dis[:,:nstart],dis[:,nend:])).std(axis=1)   #RMS as function of frequency channels (without event)
			rms_freq -= np.median(rms_freq)
			rms_freq /= MAD(rms_freq,ax=None) #Normalized
			rms_mask=np.where(rms_freq<3)[0]
			n_mask=np.where(rms_freq>3)[0]
			sn_highrms=get_snr(dis[rms_mask].sum(axis=0),nstart,width)    #Calculate snr of non-high rms channels
			Mean=my_mean(dis,nstart,nend)
			mean_f1,mean_f2,mean_f3=Mean[F1].mean(),Mean[F2].mean(),Mean[F3].mean()
			Std=my_std(dis,nstart,nend)
			std_f1,std_f2,std_f3=Std[F1].mean(),Std[F2].mean(),Std[F3].mean()
			mean_rst=Mean[F_rst].mean()
			std_rst=Std[F_rst].mean()
			Mod_index=mod_index(event)
			Mod_tscrunch=mod_index(event,True)
			dis=dis-np.median(dis)
			event=dis[:,nstart:nend]
			ch1,ch2,ch3,ch4,all_ch=get_power(event)
			index=get_index(utc,sample_bin)
            delete_block()
			out_queue.put([beam,sample_bin,H_dm,H_w,H_sn,dm,width,sn,snr_0,(ch1/all_ch)*100,(ch2/all_ch)*100,(ch3/all_ch)*100,index,utc,mean_f1,mean_f2,mean_f3,mean_rst,std_f1,std_f2,std_f3,std_rst,sn_highrms,len(n_mask),Mod_index,Mod_tscrunch])
			#return beam,sample_bin,H_dm,H_w,H_sn,dm,width,sn,snr_0,(ch1/all_ch)*100,(ch2/all_ch)*100,(ch3/all_ch)*100,index,utc,mean_f1,mean_f2,mean_f3,mean_rst,std_f1,std_f2,std_f3,std_rst,sn_highrms,len(n_mask),Mod_index,Mod_tscrunch

def get_candidates():
	return 1,2,3

"""Globals"""
F1=range(7,13)
F2=range(14,19)
F3=range(20,25)
F_rst=[0,1,2,3,4,5,6,13,19,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]

dm_list=np.loadtxt("/home/wfarah/dm_list.dat")
backstep=100
archiveDirectory="/data/mopsr/archives/"

n_cores=2

"""
if __name__=="__main__":
	pool = Pool(processes = n_cores)
	while True:
		candidates = get_candidates() #Wait for candidates, socket?
		for candidate in candidates:
			processing

"""
