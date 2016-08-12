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
			dis=fil_block.dedisperse(dm)
			sn[i],nstart[i],nend[i]=7,100,100+width
			s=dis.sum(axis=0)
			s=remove_baseline(s)
			d=normalize(s,how="mad")
			c=convolve(d,width)
			#sn[i]=c.max()/np.sqrt(width)
			#nstart[i]=c.argmax()
			#nend[i]=c.argmax()+width
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

