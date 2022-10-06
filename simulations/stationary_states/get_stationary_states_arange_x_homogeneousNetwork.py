##################################
# written by: Justus Kromer
##################################
# written and tested for Python 2.7.13
##################################
#
#	content:
#
#		simulation script to gen initial networks and simulate until a stationary state is reaced, indices are aranged according to x direction
#
#	run: python gen_stationary_states_arange_x.py outputDirectory initialSeed

# imports
import numpy as np 
import numexpr as ne
import os
from scipy.sparse import csc_matrix, csr_matrix
import scipy.sparse
import scipy

import sys
sys.path.append( 'functions' )
import functions_genNetwork
import functions_sim
import functions_pars


#############################################
# load shell input
print('output directory')
outputDirectory = sys.argv[1]
print('initial seed')
initialSeed = int(sys.argv[2])

#############################################
## load parameter set
system_parameters = functions_pars.gen_Parameter_set_Chaos_Paper( initialSeed )

#############################################
## set seed of random number generator
np.random.seed( system_parameters['initialSeed'] )


#############################################
## initializes system
STDPon, N, N_STN, VR, tauVT, VTspike, VTRest, tau_spike, V_spike, Vexc, Vinh, tauNoise, noiseIntensities, synaptic_offset_after_spike, tauSyn, dt, Tmax, t, recordedSpikes, kSave, StepsTauSynDelaySTNSTN, StepsTauSynDelayGPeGPe, StepsTauSynDelayGPeSTN, StepsTauSynDelaySTNGPe, a_delaySteps, b_delaySteps, c_delaySteps, delayedSpikingNeurons, numberOfDelayedTimeSteps , est_Spike_Times, switchOffSpikeTimes, basicFilterSpikingNeurons, lastSpikeTimeStep, evenPriorLastSpikeTimeStep, kWeightOutput, NeuronIndices, start_time, borderForScaningForSpikeValues, borderForScaningNeuronsPassingSpikingThreshold, STDPAlreadyOn = functions_sim.initialize_system( system_parameters )
# maximum number of integration steps
nSteps= int( Tmax*1000./dt )	

################################################################
# apply initial conditions
Vrest , tau , v, s, Snoise, VT, Vnoise =  functions_sim.set_initial_conditions_for_nodes( system_parameters )

################################################################
# initialize Poisson input
spikeArrival , InputRates, numberOfPredrawnSpikeArrivals = functions_sim.initialize_Poisson_noise( system_parameters )

################################################################
# generate connectivity and weight matrix
rnd_state_for_network_generation = np.random.get_state()
synConnections , cMatrix , neuronLoc , sim_objects = functions_genNetwork.generate_connectivity_and_weight_matrix_homogeneous( system_parameters , rnd_state_for_network_generation )

################################################################
# initialize synaptic plasticity
weightUpDatesPostSynSpike , weightUpDatesPreSynSpike , STDPCutOffSteps = functions_sim.initialize_asymmetric_Hebbian_STDP( system_parameters )

################################################################
# initialize sim objects
maxNumberOfPreSynapticNeurons, maxNumberOfPostSynapticNeurons, numpyPreSynapticNeurons, numpyPostSynapticNeurons, transmissionDelaysPreSynNeurons, transmissionDelaysPostSynNeurons, csc_Zero, csc_Ones = functions_sim.initialize_sim_objects( sim_objects )

################################################################
# generate output folder
outputFolder = functions_sim.create_outputFolder( outputDirectory, STDPon,  initialSeed )
# save parameter set
np.save(outputFolder+'/parameterSet.npy', system_parameters) 




################################################################
# initialize output data
################################################################
#  spike trains, first column neuron index, second column spike time in ms
spikeTimes=np.zeros( (est_Spike_Times, 2) )     
# mean synaptic weights:    int , float, float, float
# entries are:     current time step        
meanWeightTimeSeries=np.zeros(  ( int( float( kSave )/float( kWeightOutput ) ) , 5 ) )
#  calculate initial mean weights
meanWeightTimeSeries[0,:]=np.array( [0, np.mean(cMatrix[:N_STN,:N_STN]), np.mean(cMatrix[N_STN:,N_STN:]) , np.mean(cMatrix[N_STN:,:N_STN]) , np.mean(cMatrix[:N_STN,N_STN:]) ] )


#recordTrajectory=np.zeros( (nSteps, 7) ) 














#############################################################################
########### start main loop
#############################################################################
print('starting main loop')
for kStep in range( nSteps ):


	##################################################
	# switch STDP on  
	# STDP is switched off in the very beginning (STDPAlreadyOn ==False)
	# switch it on after 20 sec
	if (STDPAlreadyOn == False):
		if ( kStep > int( 20000.0/dt) ) and (STDPon == True):
			STDPAlreadyOn=True
	STDPAlreadyOn=True

	# # save state
	# recordTrajectory[kStep,0]=t
	# recordTrajectory[kStep,1]=v[0]
	# recordTrajectory[kStep,2]=s[0,0]
	# recordTrajectory[kStep,3]=s[0,1]
	# recordTrajectory[kStep,4]=Snoise[0]
	# recordTrajectory[kStep,5]=VT[0]
	# recordTrajectory[kStep,6]=switchOffSpikeTimes[0]

	##################################################
	# generate output
	# save results every 'kSave' time steps
	if (kStep % kSave) == 0:

		# save spike train		
		if kStep != 0:
			# first column is index of spiking neurons, second column is spike time
			# spike times are saved as integers referring to the time step the neurons spiked
			# translation to integers
			backupEdSpikeTimes=spikeTimes[ spikeTimes[:,1] != 0 ]
			backupEdSpikeTimes[:,1]*=1./dt
			backupEdSpikeTimes=backupEdSpikeTimes.astype(int)
			# save 'spikeTimes' to 'spikeTimes_XXXX_sec.npy'
			np.save(outputFolder+'/spikeTimes_'+str(int((kStep+1)*0.001*dt))+'_sec.npy', backupEdSpikeTimes )
			# reset spikeTimes and recordedSpikes after backup. Spike times are only stored in memory until back up is done to save memoory
			spikeTimes=np.zeros( (est_Spike_Times, 2) )
			recordedSpikes=0
			# save mean synaptic weights
			# first column is time step, second column mean recurrent STN weight, third column mean recurrent GPe weigth, fourth column mean STN -> GPe weight, and fifth column mean GPe -> STN weight
			# save 'meanWeightTimeSeries' to 'meanWeightTimeSeries_XXXX_sec.npy'
			np.save(outputFolder+'/meanWeightTimeSeries_'+str(int((kStep+1)*0.001*dt))+'_sec.npy', meanWeightTimeSeries )
			meanWeightTimeSeries=np.zeros(  ( int( float( kSave )/float( kWeightOutput ) ) , 5 ) )

		# generate backup Folder
		functions_sim.genBackup_Files( outputFolder+'/'+str(int((kStep*0.001+1)*dt))+'_sec', kStep, v, s, Snoise, VT, switchOffSpikeTimes, cMatrix, synConnections, neuronLoc['STN_center_mm'], neuronLoc['GPe_center_mm'], lastSpikeTimeStep, evenPriorLastSpikeTimeStep , delayedSpikingNeurons)
		# finally, file 'listOfBackupTimeSteps.txt' contains ifnormation about times when backups where done
		# first column is time step and second column is corresponding backup folder
		with open( outputFolder+'/listOfBackupTimeSteps.txt', 'a') as myfile:
			myfile.write(str(kStep)+' '+str(int((kStep*0.001+1)*dt))+'_sec'+'\n')
			myfile.close()

    # mean synaptic weight is calculated every 'kWeightOutput' time steps
	if kStep % kWeightOutput == 0:	
		# first column is time step, second column mean recurrent STN weight, third column mean recurrent GPe weigth, fourth column mean STN -> GPe weight, and fifth column mean GPe -> STN weight
		meanWeightTimeSeries[int( float(kStep % kSave)/float(kWeightOutput)),:]=np.array( [kStep, np.mean(cMatrix[:N_STN,:N_STN]), np.mean(cMatrix[N_STN:,N_STN:]) , np.mean(cMatrix[N_STN:,:N_STN]) , np.mean(cMatrix[:N_STN,N_STN:]) ] )





	##################################################
	# evaluate noisy Poisson input
	# use preevaluated spike trains if not at end of it
	currentSlotInSpikeArrival=kStep % numberOfPredrawnSpikeArrivals
	# if done with precalculated Poisson input, calculate the new one
	if currentSlotInSpikeArrival == 0:
		spikeArrival=np.random.poisson(lam=(InputRates))
		spikeArrival=np.transpose( spikeArrival )

	# first column of s contains values for exc. conductances
	# second column values for inhibitory conducances
	Sexc=s[:,0]
	Sinh=s[:,1]

	##################################################
	# Euler step
	# 1)
	# calculate right-hand site of equation for membrane potential
	v_Offset=ne.evaluate('1/tau*((Vrest-v) +(Vnoise-v)*Snoise+(Vexc-v)*Sexc+(Vinh-v)*Sinh)')
	# synaptic conductances
	s_Offset=-np.multiply(s,tauSyn)
	# noise conductances
	Snoise_Offset=-Snoise/tauNoise
	# dynamic thresholds
	VT_Offset=-(VT-VTRest)/tauVT

	# check whether neuron is in refractory mode
	# dim N, returs True if neuron is not in refractory mode
	neuronsNotAtRefractory=switchOffSpikeTimes<=t   

	# 2)
	# reset neurons that just stopped spiking
	# those are not in refractory but their membrane potential is still high
	neuronsToBeSetToResetValue=NeuronIndices[np.logical_and( neuronsNotAtRefractory, v>borderForScaningForSpikeValues )]
	# set their membrane potential to VR
	v[neuronsToBeSetToResetValue]=VR*np.ones( len(neuronsToBeSetToResetValue) )

	# update membrane potentials of all neurons that are not in spike mode
	v[neuronsNotAtRefractory]=v[neuronsNotAtRefractory]+dt*v_Offset[neuronsNotAtRefractory]
	s=ne.evaluate('s+dt*s_Offset')
	Snoise=ne.evaluate('Snoise+dt*Snoise_Offset')
	VT=ne.evaluate('VT+dt*VT_Offset')

	# update noise conductances if Poisson spikes arrive
	# spikeArrival[ currentSlotInSpikeArrival ] cotains the number of arriving spikes, i.e. 0, 1, ...
	Snoise+=noiseIntensities/tauNoise*spikeArrival[ currentSlotInSpikeArrival ]

	
	##################################################
	# evaluate threshold crossing
	# get indices of neurons that chrossed the spiking threshold
	passedThreshold=NeuronIndices[np.logical_and( v > VT, v<borderForScaningNeuronsPassingSpikingThreshold)]
	# get number of such neurons
	numberOfSpikingNeurons=len(passedThreshold)


	##################################################
	# evaluate spiking neurons
	# 1) neurons that are just spiking
	if numberOfSpikingNeurons>0:

		# apply reset conditions
		v[passedThreshold]=V_spike*np.ones( numberOfSpikingNeurons )
		VT[passedThreshold]=VTspike*np.ones( numberOfSpikingNeurons )
		
		# add new spikes to spikeTimes in order to get complete spike train
		# spikeTimes is initialized as a block of 'est_Spike_Times' slots
		# if adding current spikes would exceed number of slots new 'est_Spike_Times 'slots are added
		# if est_Spike_Times
		if (recordedSpikes+numberOfSpikingNeurons)> len(spikeTimes):
			spikeTimes=np.concatenate( (spikeTimes,np.zeros( (est_Spike_Times, 2) )), axis=0 )
		# now add new indices of spiking neurons and spike times to 'spikeTimes'
		spikeTimes[recordedSpikes:(recordedSpikes+numberOfSpikingNeurons),0]=passedThreshold
		spikeTimes[recordedSpikes:(recordedSpikes+numberOfSpikingNeurons),1]=t*np.ones(numberOfSpikingNeurons)

		# keep track of last two spike times (we assume that transmission delay is much shorter than interspike intervals)
		evenPriorLastSpikeTimeStep[passedThreshold]=np.copy(lastSpikeTimeStep[passedThreshold]) # contains time steps spike times before last spike times
		lastSpikeTimeStep[passedThreshold]=kStep*np.ones(numberOfSpikingNeurons) # contains time steps last spike times
		
		# keep track of how many spikes were already recorded
		recordedSpikes+=numberOfSpikingNeurons

		# set spiking neurons to spiking mode
		# contains time when neuron will leave spiking mode
		switchOffSpikeTimes[passedThreshold]=(t+tau_spike)*np.ones(numberOfSpikingNeurons) # ms

		##################################################
		# activated synaptic connections neurons
		# 'activatedSynconnections' contains all necessary information about outgoing synapses that are active
		# it is a list of "outgoing synapses"
		#   first column:  index of presynaptic neuron
		#	second column: index of postsynaptic neuron
		#	thrid column:  synaptic transmission delay in time steps    
		activatedSynconnections=np.zeros((len(passedThreshold)*maxNumberOfPostSynapticNeurons,3))
		activatedSynconnections[:,0]=np.repeat(passedThreshold, maxNumberOfPostSynapticNeurons)
		activatedSynconnections[:,1]=numpyPostSynapticNeurons[passedThreshold].reshape(len(passedThreshold)*maxNumberOfPostSynapticNeurons)
		activatedSynconnections[:,2]=transmissionDelaysPostSynNeurons[passedThreshold].reshape(len(passedThreshold)*maxNumberOfPostSynapticNeurons)	
		# delete zero elements (theire delays are set to -1)
		activatedSynconnections=activatedSynconnections[activatedSynconnections[:,2]!=-1]



		# sort synapses accroding to their delays
		# FIXME: it is important that the three delays differ, i.e. StepsTauSynDelaySTNSTN !=  StepsTauSynDelayGPeGPe != StepsTauSynDelaySTNGPe
		a_delayInput=activatedSynconnections[ activatedSynconnections[:,2]==StepsTauSynDelaySTNSTN]
		b_delayInput=activatedSynconnections[ activatedSynconnections[:,2]==StepsTauSynDelayGPeGPe]
		c_delayInput=activatedSynconnections[ activatedSynconnections[:,2]==StepsTauSynDelaySTNGPe]
		
		# similar to 'activatedSynconnections' but for "incoming synapses" 
		# it is a list of "incoming synapses"
		#   first column:  index of postsynaptic neuron
		#	second column: index of presynaptic neuron
		#	thrid column:  synaptic transmission delay in time steps 
		presynapticNeuronsOfSpikingNeurons=np.zeros((len(passedThreshold)*maxNumberOfPreSynapticNeurons,3))
		presynapticNeuronsOfSpikingNeurons[:,0]=np.repeat(passedThreshold, maxNumberOfPreSynapticNeurons)
		presynapticNeuronsOfSpikingNeurons[:,1]=numpyPreSynapticNeurons[passedThreshold].reshape(len(passedThreshold)*maxNumberOfPreSynapticNeurons)
		presynapticNeuronsOfSpikingNeurons[:,2]=transmissionDelaysPreSynNeurons[passedThreshold].reshape(len(passedThreshold)*maxNumberOfPreSynapticNeurons)
		# delete zero elements(empty presynaptic neurons are initialized with N+1 )		
		presynapticNeuronsOfSpikingNeurons=presynapticNeuronsOfSpikingNeurons[presynapticNeuronsOfSpikingNeurons[:,1]!=N+1]
		
		#########################################################
		#
		# 	update weights according to stdp rule
		#	note: at spike times ingoing synapses are updated
		########################################################
		if STDPAlreadyOn==True:

			# evaluate ingoing synapses to 'passedThreshold' neurons
			# evaluate pre synaptic neurons (spiking neuron is postsynaptic neuron)
			indicesOfPreSynNeurons=presynapticNeuronsOfSpikingNeurons[:,1].astype(int)

			# delayed spike times of presynaptic neurons
			delayedSpikeTimesPresyanpticNeurons=np.copy(lastSpikeTimeStep[ indicesOfPreSynNeurons ]+presynapticNeuronsOfSpikingNeurons[:,2].astype(int))

			# filter spikes that are not arrived at synapse yet and take times from spikes before last spikes if necessary
			notArrivedSpikeIndices=delayedSpikeTimesPresyanpticNeurons > kStep
			delayedSpikeTimesPresyanpticNeurons[ notArrivedSpikeIndices ]=np.copy( evenPriorLastSpikeTimeStep[ indicesOfPreSynNeurons ]+presynapticNeuronsOfSpikingNeurons[:,2].astype(int) )[notArrivedSpikeIndices]

			# consider cutoff for STDP functions, weightupdates due to longer time lags are considered as if they would equal the cutoff time STDPCutOffSteps 
			lastSpikeStepsOfPreSynNeurons=np.clip((kStep-delayedSpikeTimesPresyanpticNeurons),0,STDPCutOffSteps-1)

			# calculate resulting weight updates using precalculated STDP functions 'weightUpDatesPostSynSpike'
			weigthUpdates=weightUpDatesPostSynSpike[lastSpikeStepsOfPreSynNeurons]

			# calculate indices in weight matrix corresponding to currently updated synapses
			# cMatrixIndizes   = list of indices of postsynaptic neurons, list of indices of presynaptic neurons
			cMatrixIndizes = presynapticNeuronsOfSpikingNeurons[:,0].astype(int), indicesOfPreSynNeurons

			# generate scipy.sparse matrix containing weight updates for respective synapses
			weigthUpdatesPre=scipy.sparse.csc_matrix((weigthUpdates, (cMatrixIndizes)), shape=(N, N))

			# update weight matrix
			cMatrix+= weigthUpdatesPre 


	##################################################
	# evaluate spikes that arrive at the synapse 
	# 2) neurons that are just spiking
	# process arriving spikes
	# read theses synapses out of 'delayedSpikingNeurons' 
	synapsesOfArrivingSpikes=delayedSpikingNeurons[ kStep % numberOfDelayedTimeSteps ]
	numOfArrivingSpikes = len( synapsesOfArrivingSpikes )-1	# -1 because of zeros at 0th entry

	# process arriving spikes if there are any
	if numOfArrivingSpikes>0:

		# spikes are arriving and cause upades of synaptic conductances
		activeConnectionWeights=scipy.sparse.csc_matrix( ( np.array(cMatrix[synapsesOfArrivingSpikes[1:,1],synapsesOfArrivingSpikes[1:,0]])[0], (synapsesOfArrivingSpikes[1:,1],synapsesOfArrivingSpikes[1:,0]) ), shape=(N,N) )
		
		# update synaptic weight
		# this matrix product effectively calculates the sum over weights of active synapses 
		# this sum is then added to 's' while rescaling with 'synaptic_offset_after_spike' = 1/N
		# first column of s are excitatory and second column are inhibitory synapses
		s+=synaptic_offset_after_spike*activeConnectionWeights.dot( basicFilterSpikingNeurons )

		#########################################################
		#
		# 	update weights according to STDP rule
		#   note: at spike arriving times outgoing synapses are updated
		########################################################
		if STDPAlreadyOn==True:

			########################################################
			# THIS HAS TO BE DONE AT DELAYED SPIKE TIME
			# evaluate post synaptic neurons (neuron that was spiking is presynaptic neuron)
			indicesOfPostSynNeurons=synapsesOfArrivingSpikes[1:,1].astype(int)

			# consider cutoff for STDP functions, weightupdates due to longer time lags are considered as if they would equal the cutoff time STDPCutOffSteps 
			lastSpikeStepsOfPostSynNeurons=np.clip((kStep-lastSpikeTimeStep[ indicesOfPostSynNeurons ]),0,STDPCutOffSteps-1)

			# calculate resulting weight updates using precalculated STDP functions 'weightUpDatesPostSynSpike'
			weigthUpdates=weightUpDatesPreSynSpike[lastSpikeStepsOfPostSynNeurons]

			# calculate indices in weight matrix corresponding to currently updated synapses
			# cMatrixIndizes   = list of indices of postsynaptic neurons, list of indices of presynaptic neurons
			cMatrixIndizes=indicesOfPostSynNeurons, synapsesOfArrivingSpikes[1:,0].astype(int)

			# generate scipy.sparse matrix containing weight updates for respective synapses
			weigthUpdatesPost=scipy.sparse.csc_matrix((weigthUpdates, (cMatrixIndizes)), shape=(N, N))  

			# update weight matrix			
			cMatrix+= weigthUpdatesPost 


	##################################################
	# apply the hard bounds and keep weights between 0 and 1		
	# apply hard bounds
	if STDPAlreadyOn==True:
		# check whether any weight update was done
		if (numOfArrivingSpikes>0) or (numberOfSpikingNeurons>0):
			# get indices of weights that violate bound
			indicesAbove=cMatrix>1.0          # the resulting array contains ones at indices that fulfill condition
			indicesBelowLowerThreshold=cMatrix<0.0 # the resulting array contains ones at indices that fulfill condition

			# if any weight violates lower bound
			if len(indicesBelowLowerThreshold.nonzero()[0])!=0:
				indicesAboveLowerThreshold=csc_Ones-indicesBelowLowerThreshold   # get array with ones at weights that are above lower threshold
				cMatrix=cMatrix.multiply( indicesAboveLowerThreshold ) # this leaves only the weights that are above lower threshold, other entries are zero

			# if any weight violates upper bound
			if len(indicesAbove.nonzero()[0])!=0:  
				cMatrix=cMatrix+(csc_Ones-cMatrix).multiply( indicesAbove ) # substract difference to upper bound and we are done


	##################################################
	# update delayedSpikingNeurons
	# first, we delete what was just processed 
	delayedSpikingNeurons[ kStep % numberOfDelayedTimeSteps ]=np.zeros((1,3))
	# next, consider delayed arrival of current spikes
	# add outgoing synapses of neurons that spiked in current time step to delayedSpikingNeurons
	if numberOfSpikingNeurons > 0:
		# we updated arrays 'a_delayInput', 'b_delayInput', and 'c_delayInput' earlier
		# now add them to the corresponding slot in 'delayedSpikingNeurons'
		if len(a_delayInput)>0:
			delayedSpikingNeurons[ (kStep+a_delaySteps) % numberOfDelayedTimeSteps ]=np.concatenate( (delayedSpikingNeurons[ (kStep+a_delaySteps) % numberOfDelayedTimeSteps ],a_delayInput), axis=0 )
		if len(b_delayInput)>0:
			delayedSpikingNeurons[ (kStep+b_delaySteps) % numberOfDelayedTimeSteps ]=np.concatenate( (delayedSpikingNeurons[ (kStep+b_delaySteps) % numberOfDelayedTimeSteps ],b_delayInput), axis=0 )
		if len(c_delayInput)>0:
			delayedSpikingNeurons[ (kStep+c_delaySteps) % numberOfDelayedTimeSteps ]=np.concatenate( (delayedSpikingNeurons[ (kStep+c_delaySteps) % numberOfDelayedTimeSteps ],c_delayInput), axis=0 )


	# update time
	t+=dt # ms

##################################################
# generate final backup
# first column is index of spiking neurons, second column is spike time
# spike times are saved as integers referring to the time step the neurons spiked
# translation to integers
backupEdSpikeTimes=spikeTimes[ spikeTimes[:,1] != 0 ]
backupEdSpikeTimes[:,1]*=1./dt
backupEdSpikeTimes=backupEdSpikeTimes.astype(int)
# save 'spikeTimes' to 'spikeTimes_'XXXX_sec.npy'
np.save(outputFolder+'/spikeTimes_FinalBackup.npy', backupEdSpikeTimes )
# save 'meanWeightTimeSeries' to 'meanWeightTimeSeries_XXXX_sec.npy'
np.save(outputFolder+'/meanWeightTimeSeries_FinalBackup.npy', meanWeightTimeSeries )
		
# generate backup Folder
functions_sim.genBackup_Files( outputFolder+'/FinalBackup', kStep, v, s, Snoise, VT, switchOffSpikeTimes, cMatrix, synConnections, neuronLoc['STN_center_mm'], neuronLoc['GPe_center_mm'], lastSpikeTimeStep, evenPriorLastSpikeTimeStep, delayedSpikingNeurons)
# finally, file 'listOfBackupTimeSteps.txt' contains ifnormation about times when backups where done
# first column is time step and second column is corresponding backup folder
with open( outputFolder+'/listOfBackupTimeSteps.txt', 'a') as myfile:
	myfile.write(str(kStep)+' '+'FinalBackup'+'\n')
	myfile.close()

# print (real) time needed in seconds create 
print( str(" comput. time (s) "+str(time.time() - start_time) ) )









