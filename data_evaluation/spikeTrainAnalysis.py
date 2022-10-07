import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
import scipy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# try:
# 	import elephant
# except:
# 	print 'WARNING: elephant package not found'
# import os
# from quantities import Hz, s, ms

dt = 0.1 # ms
stepsToSec = 0.001

from scipy.interpolate import interp1d

# returns a list of unique backup directory file names from 
#   Directory+'/listOfBackupTimeSteps.txt'
def loadListOfBackupTimesteps( Directory ):
	
	lines=[]
	backupSWithDuplicates=[]

	with open( Directory+'/listOfBackupTimeSteps.txt','r') as f:
		lines.append( f.read() )
		lines=(lines[0].split('\n'))
		for line in lines:
			backupSWithDuplicates.append(line.split(' '))

		backupS = []
		for i in backupSWithDuplicates:
			if i not in backupS:
				backupS.append(i)
	return backupS
	




# loads complete spike train and weight sequence for order of listed directories
def load_Complete_SpikeTrain_And_Weigth_Trajectories( sortedListOfDirectories ):
	
	import os

	# initialize spikeTrain and weightData
	spikeTrain=[]
	weightData=[]
				
	for Directory in sortedListOfDirectories:
		# load first Trajectory
		if os.path.isfile( Directory+'/listOfBackupTimeSteps.txt' ): 
			# load short data
			# load long data
			backupS =loadListOfBackupTimesteps( Directory )
	
			# load data
			for fileEndings in range(len(backupS)-1):
				
				# get actual file ending
				fileEndingString=backupS[ fileEndings ][1]
				if fileEndingString!='0_sec':
					
					#print fileEndingString
					spikeTimesFile=Directory+'/spikeTimes_'+str(fileEndingString)+'.npy'
						
					if os.path.isfile( spikeTimesFile ):
						if len(spikeTrain)==0:
							spikeTrain=np.load( spikeTimesFile )
						else:

							# ensure that spike trains dont overlap
							newSpikeTrain = np.load( spikeTimesFile )
							if len(newSpikeTrain) != 0:
								#print len(newSpikeTrain), len(spikeTrain), spikeTrain[-1,1], newSpikeTrain[0,1]
								oldSpikeTrain = spikeTrain[ spikeTrain[:,1]<newSpikeTrain[0,1] ]

								spikeTrain=np.concatenate( ( oldSpikeTrain, newSpikeTrain ), axis=0 )
					
					weightFile=Directory+'/meanWeightTimeSeries_'+str(fileEndingString)+'.npy'
					if os.path.isfile( weightFile ):
						if len(weightData)==0:
							weightData=np.load( weightFile )
						else:
							weightData=np.concatenate( ( weightData,np.load(weightFile ) ), axis=0 )
	
		else:
			print('Error: no data found in', Directory )
		

	# returns spike train and weight trajectory. 
	# times are in simulation time steps
	return spikeTrain, weightData        
		
# loads complete spike train and weight sequence for order of listed directories
def load_Complete_Weight_Trajectories( sortedListOfDirectories ):
    
    import os

    # initialize  aweightData
    weightData=[]
                
    for Directory in sortedListOfDirectories:
        # load first Trajectory
        if os.path.isfile( Directory+'/listOfBackupTimeSteps.txt' ): 
            # load short data
            # load long data
            backupS =loadListOfBackupTimesteps( Directory )
    
            # load data
            for fileEndings in range(len(backupS)-1):
                
                # get actual file ending
                fileEndingString=backupS[ fileEndings ][1]
                if fileEndingString!='0_sec':
                    
                    weightFile=Directory+'/meanWeightTimeSeries_'+str(fileEndingString)+'.npy'
                    if os.path.isfile( weightFile ):
                        if len(weightData)==0:
                            weightData=np.load( weightFile )
                        else:
                            weightData=np.concatenate( ( weightData,np.load(weightFile ) ), axis=0 )
    
        else:
            print('Error: no data found in', Directory)
        

    # returns spike train and weight trajectory. 
    # times are in simulation time steps
    return weightData 

# time interval for averaging
# KuramotoArray ... original data
# Delta ... length of averaging window in seconds
def calc_time_averaged_Kuramoto_order_parameter( KuramotoArray , Delta ):
    
    avSteps = int( Delta*100 )
    nav_Intervals = int( len(KuramotoArray)/float(avSteps) )
    process=KuramotoArray[  :(len(KuramotoArray) - (len(KuramotoArray)) % avSteps) ]
    t = process[:,0]
    KuramotoOrder =  process[:,1]
    t = t.reshape( (nav_Intervals, avSteps) )
    KuramotoOrder = KuramotoOrder.reshape( (nav_Intervals, avSteps) )
    time_av_Kuramoto_order = np.mean( KuramotoOrder , axis = 1 )
    t = t[:,0]

    return t, time_av_Kuramoto_order



# same as calcKuramotoOrderParameter( spikeTimes, tmin, tmax, NinterPolSteps, arrayOfNeuronIndixes, outputFilename )
# but saves memory
# resolution ... temporal distance between  (ms)
def piece_wise_calcKuramotoOrderParameter( spikeTimes, tmin, tmax, resolution, arrayOfNeuronIndixes, outputFilename ):

    # delete empty entries
    spikeTimes=spikeTimes[ spikeTimes[:,1]!= 0 ]

    ################################################################################
    ######## the following two lines were added later ##############################
    populationSize=len(arrayOfNeuronIndixes)
    
    # number of grid points for which Kuramoto order parameter is evaluated at once
    NinterPolSteps = int( 1000.0*( tmax-tmin )/float(resolution) )
    processAtOnce_NinterPolSteps = 100000

    # if too many gridpoints are given a piece-wise calculation is performed
    if NinterPolSteps > processAtOnce_NinterPolSteps:
        
        KuramotoOutArray = []
        
        currentInterPolSteps = 0
        lengthOfTimeIntervals = resolution * processAtOnce_NinterPolSteps # ms
        # in order to exclude boundary effects when combining arrays, we consider an overlap of 2000 ms
        TimeStepsOfOverlap = int( 2000.0/resolution ) # time steps

        if 2*TimeStepsOfOverlap > processAtOnce_NinterPolSteps:
            print('ERROR: overlap for piece-wise Kuramoto order parameter calculation too long compared to processAtOnce_NinterPolSteps!')
            return 0
        
        current_Tmin = 1000.0*tmin # ms
        current_Tmax = current_Tmin + lengthOfTimeIntervals # ms
        
        while current_Tmax < 1000*tmax:

            # initialize phases at fixed points
            phases=np.zeros( (populationSize, processAtOnce_NinterPolSteps) )
            arrayOfGridPoints=np.arange( current_Tmin, current_Tmax , resolution)

            # consider only spikes that are between tmin and tmax
            processedSpikeTimes=spikeTimes[ np.logical_and( spikeTimes[:,1]>=current_Tmin , spikeTimes[:,1]<= current_Tmax )  ]        

            # calculate phases in current interval
            krecPhases=0
            for kNeuron in arrayOfNeuronIndixes:

                # get spike train of corresponding neuron
                spikeTrainTemp = processedSpikeTimes[processedSpikeTimes[:,0].astype(int)== kNeuron ][:,1]

                # calc phase function
                if len(spikeTrainTemp) != 0:
                    phaseNPiCrossings=np.concatenate( ( np.full( 1, 1000.0*(2*tmin-tmax) ) , spikeTrainTemp, np.full( 1, 1000.0*(2*tmax-tmin) ) ), axis=0 )
                    PhaseValues=np.linspace(0,len(phaseNPiCrossings)-1, len(phaseNPiCrossings))
                else:
                    phaseNPiCrossings=np.array([-np.inf, np.inf])
                    PhaseValues=np.array([0, 1])

                # linear interpolate phaseNPiCrossings
                phaseFunctionKNeuron=interp1d(phaseNPiCrossings,2*np.pi*PhaseValues)

                phases[krecPhases,:]=phaseFunctionKNeuron(arrayOfGridPoints) 
                krecPhases+=1

            # calc Kuramoto order parameter
            TotalArrayOfKuramotoOrderParameterAtGridPoints=1/float(populationSize)*np.absolute(np.sum( np.exp( 1j*phases ), axis=0 ))
            current_KuramotoOutArray=np.array( [arrayOfGridPoints, TotalArrayOfKuramotoOrderParameterAtGridPoints] )

            current_KuramotoOutArray=np.transpose( current_KuramotoOutArray )

            if len(KuramotoOutArray) == 0:
                KuramotoOutArray = current_KuramotoOutArray[:-TimeStepsOfOverlap]
            else:
                # add to previous Kuramoto array
                KuramotoOutArray = np.concatenate( ( KuramotoOutArray, current_KuramotoOutArray[ TimeStepsOfOverlap:-TimeStepsOfOverlap ] ), axis = 0 )

            # prepare boundaries of next interval 
            current_Tmin = current_Tmax - resolution * 2*TimeStepsOfOverlap # ms
            current_Tmax = current_Tmin + lengthOfTimeIntervals # ms
     
            print('current interval (left boundary)' , current_Tmin * 0.001)
                
                
    # calculate in one single run    
    else:
        phases=np.zeros( (populationSize, NinterPolSteps) )
        arrayOfGridPoints=np.linspace(1000.0*tmin,1000.0*tmax,NinterPolSteps)

        processedSpikeTimes=spikeTimes[ np.logical_and( spikeTimes[:,1]>=1000.0*tmin , spikeTimes[:,1]<= 1000.0*tmax )  ]

            
        krecPhases=0
        for kNeuron in arrayOfNeuronIndixes:

            # get spike train of corresponding neuron
            spikeTrainTemp=processedSpikeTimes[processedSpikeTimes[:,0].astype(int)== kNeuron ][:,1]
            # calc phase function
            if len(spikeTrainTemp) != 0:
                phaseNPiCrossings=np.concatenate( ( np.full( 1, 1000.0*(2*tmin-tmax) ) , spikeTrainTemp, np.full( 1, 1000.0*(2*tmax-tmin) ) ), axis=0 )
                PhaseValues=np.linspace(0,len(phaseNPiCrossings)-1, len(phaseNPiCrossings))
            else:
                phaseNPiCrossings=np.array([-np.inf, np.inf])
                PhaseValues=np.array([0, 1])

            # linear interpolate phaseNPiCrossings
            phaseFunctionKNeuron=interp1d(phaseNPiCrossings,2*np.pi*PhaseValues)

            phases[krecPhases,:]=phaseFunctionKNeuron(arrayOfGridPoints) 
            krecPhases+=1

        # calc Kuramoto order parameter
        TotalArrayOfKuramotoOrderParameterAtGridPoints=1/float(populationSize)*np.absolute(np.sum( np.exp( 1j*phases ), axis=0 ))
        KuramotoOutArray=np.array( [arrayOfGridPoints, TotalArrayOfKuramotoOrderParameterAtGridPoints] )
        
        KuramotoOutArray=np.transpose( KuramotoOutArray )

        
    if outputFilename!='':
       np.save( outputFilename+'.npy' , KuramotoOutArray )
    
    return KuramotoOutArray

# calculates the sequence of mean population weights resulting from all backup directories in 'directory'
# listOfPresynapticPopulationIndices is a list of lists of indices of all subpopulations of presynaptic neurons to be considered
# listOfPostsynapticPopulationIndices is a list of lists of indices of all subpopulations of postsynaptic neurons to be considered
def calcPopulationWeightsFromCMatrix( directory, listOfPresynapticPopulationIndices, listOfPostsynapticPopulationIndices ):
	
	import os
	import scipy.sparse
	
	listOfBackupTimes = loadListOfBackupTimesteps( directory )
	
	meanPopulationWeights=np.zeros( (len( listOfBackupTimes ), len(listOfPresynapticPopulationIndices)+1) ) 
	
	for kBackupDirectories in range( len( listOfBackupTimes ) ):
	
		if len(listOfBackupTimes[kBackupDirectories])==2:
			# load respective cMatrix
			cMatrixFilename=directory+'/'+listOfBackupTimes[kBackupDirectories][1]+'/cMatrix.npz'
		
			if os.path.isfile( cMatrixFilename ):
				cMatrix = scipy.sparse.load_npz( cMatrixFilename )
			
				meanPopulationWeights[kBackupDirectories,0]=np.int(listOfBackupTimes[kBackupDirectories][0])
			
				for kSubpopulations in range(len(listOfPresynapticPopulationIndices)):
				
					meanPopulationWeights[kBackupDirectories,kSubpopulations+1]= np.mean( ( cMatrix[listOfPostsynapticPopulationIndices[kSubpopulations]][:, listOfPresynapticPopulationIndices[kSubpopulations] ]
 ) )
			
	return meanPopulationWeights


##################################################################################################
# 	returns array of time in ms (first) and population av Kuramoto order parmameter (second)
#	spikeTimes  ... spike train in ms
# 	tmin ... start time of kuramoto time trace in seconds
# 	tmax ... end time of kuramoto time trace in seconds
#  	NinterPolSteps .. number of equ spaced interpolation steps for time trace 
# 	arrayOfNeuronIndixes ... speciefies all neurons that are considered for calculation of Kuramoto order parameter
# 	outputFilename ... saves data in outputFilename+'.npy'
def calcKuramotoOrderParameter( spikeTimes, tmin, tmax, NinterPolSteps, arrayOfNeuronIndixes, outputFilename ):

	# delete empty entries
	spikeTimes=spikeTimes[ spikeTimes[:,1]!= 0 ]

	################################################################################
	######## the following two lines were added later ##############################
	# consider only spikes that are between tmin and tmax
	processedSpikeTimes=spikeTimes[ np.logical_and( spikeTimes[:,1]>=1000.0*tmin , spikeTimes[:,1]<= 1000.0*tmax )  ]
	populationSize=len(arrayOfNeuronIndixes)

	#print processedSpikeTimes
	#print NinterPolSteps
	phases=np.zeros( (populationSize, NinterPolSteps) )
	arrayOfGridPoints=np.linspace(1000.0*tmin,1000.0*tmax,NinterPolSteps)

	krecPhases=0
	for kNeuron in arrayOfNeuronIndixes:

		#print '#############################'
		#print kNeuron
		# get spike train of corresponding neuron
		spikeTrainTemp=processedSpikeTimes[processedSpikeTimes[:,0].astype(int)== kNeuron ][:,1]
		#print len(spikeTrainTemp)
		#print 'neuron', kNeuron
		#print spikeTrainTemp
		# calc phase function
		if len(spikeTrainTemp) != 0:
			phaseNPiCrossings=np.concatenate( ( np.full( 1, 1000.0*(2*tmin-tmax) ) , spikeTrainTemp, np.full( 1, 1000.0*(2*tmax-tmin) ) ), axis=0 )

			PhaseValues=np.linspace(0,len(phaseNPiCrossings)-1, len(phaseNPiCrossings))
		else:
			phaseNPiCrossings=np.array([-np.inf, np.inf])
			PhaseValues=np.array([0, 1])
		
		# linear interpolate phaseNPiCrossings
		phaseFunctionKNeuron=interp1d(phaseNPiCrossings,2*np.pi*PhaseValues)

		#print phases.shape, arrayOfGridPoints.shape
		phases[krecPhases,:]=phaseFunctionKNeuron(arrayOfGridPoints) 
		krecPhases+=1
		
		#print arrayOfGridPoints
		#print phaseNPiCrossings
		#print PhaseValues
		#print phases[krecPhases-1,:]

		#print '####', kNeuron
		#print phaseNPiCrossings.min(), phaseNPiCrossings.max()
		#print arrayOfGridPoints.min(), arrayOfGridPoints.max()
		#print phases.min(), phases.max()


	# calc Kuramoto order parameter
	TotalArrayOfKuramotoOrderParameterAtGridPoints=1/float(populationSize)*np.absolute(np.sum( np.exp( 1j*phases ), axis=0 ))
	#print 'Kuramoto order parameter'
	#print TotalArrayOfKuramotoOrderParameterAtGridPoints
	KuramotoOutArray=np.array( [arrayOfGridPoints, TotalArrayOfKuramotoOrderParameterAtGridPoints] )
	#print 'phases'
	#print arrayOfGridPoints
	#print TotalArrayOfKuramotoOrderParameterAtGridPoints
	
	KuramotoOutArray=np.transpose( KuramotoOutArray )
	
	if outputFilename!='':
	   np.save( outputFilename+'.npy' , KuramotoOutArray )
	
	return KuramotoOutArray



def MCluster_calcKuramotoOrderParameter( spikeTimes, tmin, tmax, NinterPolSteps, arrayOfNeuronIndixes, outputFilename, m ):

	# delete empty entries
	#print tmin, tmax
	spikeTimes=spikeTimes[ spikeTimes[:,1]!= 0 ]


	################################################################################
	######## the following two lines were added later ##############################
	# consider only spikes that are between tmin and tmax
	processedSpikeTimes=spikeTimes[ np.logical_and( spikeTimes[:,1]>=1000.0*tmin , spikeTimes[:,1]<= 1000.0*tmax )  ]
	populationSize=len(arrayOfNeuronIndixes)

	#print populationSize
	#print NinterPolSteps
	phases=np.zeros( (populationSize, NinterPolSteps) )
	arrayOfGridPoints=np.linspace(1000.0*tmin,1000.0*tmax,NinterPolSteps)

	krecPhases=0
	for kNeuron in arrayOfNeuronIndixes:

		#print '#############################'
		#print kNeuron
		# get spike train of corresponding neuron
		spikeTrainTemp=processedSpikeTimes[processedSpikeTimes[:,0].astype(int)== kNeuron ][:,1]
		#print len(spikeTrainTemp)
		#print 'neuron', kNeuron
		#print spikeTrainTemp
		# calc phase function
		if len(spikeTrainTemp) != 0:
			phaseNPiCrossings=np.concatenate( ( np.full( 1, 1000.0*(2*tmin-tmax) ) , spikeTrainTemp, np.full( 1, 1000.0*(2*tmax-tmin) ) ), axis=0 )

			PhaseValues=np.linspace(0,len(phaseNPiCrossings)-1, len(phaseNPiCrossings))
		else:
			phaseNPiCrossings=np.array([-np.inf, np.inf])
			PhaseValues=np.array([0, 1])
		
		# linear interpolate phaseNPiCrossings
		phaseFunctionKNeuron=interp1d(phaseNPiCrossings,2*np.pi*PhaseValues)

		#print phases.shape, arrayOfGridPoints.shape
		phases[krecPhases,:]=phaseFunctionKNeuron(arrayOfGridPoints) 
		krecPhases+=1
		
		#print arrayOfGridPoints
		#print phaseNPiCrossings
		#print PhaseValues
		#print phases[krecPhases-1,:]

		#print '####', kNeuron
		#print phaseNPiCrossings.min(), phaseNPiCrossings.max()
		#print arrayOfGridPoints.min(), arrayOfGridPoints.max()
		#print phases.min(), phases.max()


	# calc Kuramoto order parameter
	TotalArrayOfKuramotoOrderParameterAtGridPoints=1/float(populationSize)*np.absolute(np.sum( np.exp( m*1j*phases ), axis=0 ))
	#print 'Kuramoto order parameter'
	#print TotalArrayOfKuramotoOrderParameterAtGridPoints
	KuramotoOutArray=np.array( [arrayOfGridPoints, TotalArrayOfKuramotoOrderParameterAtGridPoints] )
	#print 'phases'
	#print arrayOfGridPoints
	#print TotalArrayOfKuramotoOrderParameterAtGridPoints
	
	KuramotoOutArray=np.transpose( KuramotoOutArray )
	
	if outputFilename!='':
	   np.save( outputFilename+'.npy' , KuramotoOutArray )
	
	return KuramotoOutArray



# tmin, tmax in sec
def plotTimeTraceOfKuramotoOrderParameter( spikeTrain, tmin, tmax, NinterPolSteps, arrayOfNeuronIndixes, axis, color, alpha ):

	# calculate time trace of Kuramoto order parameter
	print( NinterPolSteps )
	KuramotoOutArray = calcKuramotoOrderParameter( spikeTrain*np.array([1.0, 0.1]), tmin-1,tmax+1, NinterPolSteps, arrayOfNeuronIndixes, '' )

	# calc mean and standard deviation
	times=KuramotoOutArray[:,0]
	rho=KuramotoOutArray[:,1]

	reshaped_times=times.reshape( ( int( len(KuramotoOutArray)/float(2000) ), 2000 ) )
	reshaped_rho=rho.reshape( ( int( len(KuramotoOutArray)/float(2000) ), 2000 ) )

	print( reshaped_rho.shape )
	meantimes, stdtimes= np.mean(reshaped_times, axis=1), np.std(reshaped_times, axis=1)
	meanRho, stdRho= np.mean(reshaped_rho, axis=1), np.std(reshaped_rho, axis=1)
	print( len(KuramotoOutArray) )

	#plt.plot( KuramotoOutArray[:,0], KuramotoOutArray[:,1] )

	x=meantimes*0.001 # sec
	y=meanRho
	yerr=stdRho
	y1=meanRho-stdRho
	y2=meanRho+stdRho

	axis.plot( x, y, color=color )

	axis.fill_between(x, y1, y2, alpha=alpha, color=color)

	return x, y, yerr





# plots sequence of red lines indicated RVS stimulation pulses given in electrode data
#	electrodeData has shape :,Nele+1,   zeros column is time, other columns are currents to electrode
#    pulses are associated with current > 0.9				
def plotCRRVSsequence(electrodeData):
	kPulses=0
	Nelectrodes=4
	electrodedSize=1000/float(Nelectrodes)

	while kPulses < len(electrodeData):
		pulses=electrodeData[kPulses]
		for kelektrodes in range(Nelectrodes):
			if pulses[kelektrodes+1]>0.9:
				plt.plot( [pulses[0],pulses[0]],[(kelektrodes)*electrodedSize,(kelektrodes+1)*electrodedSize], color='red')
				kPulses+=3
		kPulses+=1
	return 0	

# calculates the mean weight of all synapses from listOfPresynapticPopulationIndices to listOfPostsynapticPopulationIndices
# for the connectivity matrix given by cMatrixFilename
def calcMeanWeights( cMatrixFilename, listOfPresynapticPopulationIndices, listOfPostsynapticPopulationIndices ):
	
	import os
	import scipy.sparse
	
	meanPopulationWeights=np.zeros( len(listOfPresynapticPopulationIndices) )
	
	if os.path.isfile( cMatrixFilename ):
		try:
			cMatrix = scipy.sparse.load_npz( cMatrixFilename )
		except:
			return meanPopulationWeights
			
		for kSubpopulations in range(len(listOfPresynapticPopulationIndices)):
				
			meanPopulationWeights[kSubpopulations]= np.mean( ( cMatrix[listOfPostsynapticPopulationIndices[kSubpopulations]][:, listOfPresynapticPopulationIndices[kSubpopulations] ]
 ) )
	#print meanPopulationWeights
	return meanPopulationWeights






# weight reduction as function of mDt
# TmaxStim .... maximum duration of stimulation in sec
# directoryEqu=''
# stimPattern='MixedRPEPoissonTstim'
# samp=2000.0
# mDtArray=[ 20.0, 40.0, 60.0, 80.0, 100.0 ]
# evaluationTimePoints=[ 2020.0, 2040.0 ]
def get_mean_Weights_and_Kuramoto_as_Function_of_time_and_mDT( directoryEqu, stimPattern, samp, mDt_array, evaluationTimePoints, TmaxStim ):
	
	# possible stimulation patterns
	# stimPattern='MixedRPEPoissonTstim'
	
	import numpy as np
	import os
	import sys
	
	# load evaluation scripts
	#import MixingNeurons.mixing_Indizes as mixing_Indizes
	import evaluationScripts.spikeTrainAnalysis as spikeTrainAnalysis

	# load parameter set
	parameterSetFile = directoryEqu+'/parameterSet.npy'
	parameterSet=np.load( parameterSetFile ).item()

	N_Slow, N_fast=parameterSet['N_STN']-parameterSet['NSTN_fast'] , parameterSet['NSTN_fast']

	PSlow=np.arange(N_Slow)
	PFast=np.arange(N_Slow, N_Slow+N_fast)

	postSynPopulations=[PSlow, PFast, PSlow, PFast]
	preSynPopulations=[PSlow, PFast,PFast , PSlow]

	TendOfEquilibration=parameterSet['Tend'] # seconds

	weightData=[]
	KuramototData=[]

	for mDt in mDt_array:
	
		mean_InterPulseInterval=mDt   # ms

		if stimPattern=='MixedRPEPoissonTstim':
			directoryStim=directoryEqu+'/FinalBackup_MixedRPEPoissonTstim/electrodeSize_500/PulsesPerBurst_1_mTimeBetBursts_'+str(mean_InterPulseInterval)+'/seedseq_100_seedShuffelIndizes_1_samp_'+str(samp)+'_Tsim_'+str(TmaxStim)
		elif stimPattern=='MixedRPEPoissonTstim_IstimRescaled':
			directoryStim=directoryEqu+'/FinalBackup_MixedRPEPoissonTstim/electrodeSize_500/PulsesPerBurst_1_mTimeBetBursts_'+str(mean_InterPulseInterval)+'_IstimRescaled/seedseq_100_seedShuffelIndizes_1_samp_'+str(samp)+'_Tsim_'+str(TmaxStim)
		else:
			print( 'Error: choose proper stimulation pattern' )
			exit()
			
		#print directoryStim

		for kData in range( len(evaluationTimePoints) ):
		
			timePoints=evaluationTimePoints[kData]
			
			# load connectivity matrix
			cMatrixFilename=directoryStim+'/'+str( int(timePoints) )+'_sec/cMatrix.npz'

			#print cMatrixFilename

			# calculate mean weights and generate output Array
			meanPopulationWeights=calcMeanWeights( cMatrixFilename, preSynPopulations, postSynPopulations )
			meanPopulationWeights=np.concatenate( (np.array([mDt]), np.array([timePoints]), meanPopulationWeights), axis=0 )
			#print directoryStim
			# calculate Kuramoto order parameters
			spikeTrainFilename=directoryStim+'/spikeTimes_'+str( int( timePoints ) )+'_sec.npy'
	   
			if os.path.isfile( spikeTrainFilename ):
				#print 'spikeTrain found'
				spikeTrain=np.load( spikeTrainFilename )
			
				# set time frame over which Kuramoto order parameter values are averaged
				Tav=10.0 # sec
				
				Resolution=int(Tav*100)
			
				#print spikeTrain.shape

				KuramotoSlowAcute=spikeTrainAnalysis.calcKuramotoOrderParameter( spikeTrain*np.array([1.0, 0.1]) , timePoints, timePoints+Tav, Resolution, PSlow , '' )
				KuramotoFastAcute=spikeTrainAnalysis.calcKuramotoOrderParameter( spikeTrain*np.array([1.0, 0.1]) , timePoints, timePoints+Tav, Resolution, PFast , '' )

				meanPopulationKuramoto=np.concatenate( (np.array([mDt]), np.array([timePoints]), np.array([ np.mean( KuramotoSlowAcute[:,1] ), np.std( KuramotoSlowAcute[:,1] ), np.mean( KuramotoFastAcute[:,1] ), np.std( KuramotoFastAcute[:,1] ) ] ) ), axis = 0 )
				KuramototData.append(meanPopulationKuramoto)

			#print meanPopulationWeights
			weightData.append(meanPopulationWeights)
		   
		
	# transform into numpy array
	weightData=np.array(weightData)
	KuramototData=np.array(KuramototData)
													  
	return KuramototData, weightData  	





# inputs are two list of spike times   in ms
# timeResolution                       in ms
# plotRange                            in ms
# returns correlation function in units of 1/Hz   
def calcSpikeTrainCorrelationFunction( spikeTrain1, spikeTrain2, timeResolution, plotRange ):
	
	# estimates of firing rate
	r1=float(len(spikeTrain1)*1000)/float( (spikeTrain1[-1]-spikeTrain1[0]) )   # Hz
	r2=float(len(spikeTrain2)*1000)/float( (spikeTrain2[-1]-spikeTrain2[0]) )   # Hz
	
	#print r1
	#print r2
	# calc corelation function
	x, y = np.meshgrid( spikeTrain1, spikeTrain2)
	
	tRange=plotRange  # ms
	dt=timeResolution

	c12_dt, binsValues=np.histogram((-x+y).flatten(), bins=np.arange(0,tRange,dt))
	#print len(c12_dt), len(binsValues)
	c12_dt=( r1*c12_dt/float( len(spikeTrain1) )*1000.0*(1/dt) )/(r1*r2)-1
	bins_dt1=0.5*(binsValues[1:]+binsValues[:-1])
	
	#print len(c12_dt), len(bins_dt1)
	
	return 0.001*timeResolution*c12_dt, bins_dt1


# inputs are two list of spike times   in ms
# timeResolution                       in ms
# plotRange                            in ms
# returns correlation function in units of 1/Hz   
# tested for x1(t) (Poisson) and x2(t)=x1(t+tau)
# works for long spike trains
# integrate over time in seconds to get correct probabilities
# output in 1/sec and ms
def check_Normalization_calcSpikeTrainCorrelationFunction( spikeTrain1, spikeTrain2, dt, tRange ):
	
	# estimates of firing rate
	r1=float(len(spikeTrain1))/float( 0.001*(spikeTrain1[-1]-spikeTrain1[0]) )   # Hz
	r2=float(len(spikeTrain2))/float( 0.001*(spikeTrain2[-1]-spikeTrain2[0]) )   # Hz
	
	
	# calc corelation function
	#print spikeTrain1, spikeTrain2
	x, y = np.meshgrid( spikeTrain1, spikeTrain2)     # x, y in ms
	#print x, y
	c12_dt, binsValues=np.histogram((-x+y).flatten(), bins=np.arange(-tRange,tRange,dt))
	#print sum(c12_dt), len(spikeTrain1), len(spikeTrain2), len(spikeTrain1)*len(spikeTrain2)
	#print len(c12_dt), len(binsValues)
	#print max( spikeTrain1[-1], spikeTrain2[-1] ), min( spikeTrain1[0], spikeTrain2[0] )
	averagingTime=float( max( spikeTrain1[-1], spikeTrain2[-1] ) - min( spikeTrain1[0], spikeTrain2[0] ) )/(1000. ) # sec
	#print averagingTime
	binsize=dt*0.001 # sec
	c12_dt= ( c12_dt/ ( averagingTime*binsize ) ) /( r1*r2 )-1
	bins_dt1=0.5*( binsValues[1:]+binsValues[:-1] )
	
	#print len(c12_dt), len(bins_dt1)
	
	return c12_dt, bins_dt1, r1, r2



# inputs are two list of spike times   in ms
# timeResolution                       in ms
# plotRange                            in ms
# returns correlation function in units of 1/Hz   
def calcSpikeTrainCorrelationFunctionFull( spikeTrain1, spikeTrain2, timeResolution, plotRange ):
	
	# estimates of firing rate
	r1=float(len(spikeTrain1)*1000)/float( (spikeTrain1[-1]-spikeTrain1[0]) )   # Hz
	r2=float(len(spikeTrain2)*1000)/float( (spikeTrain2[-1]-spikeTrain2[0]) )   # Hz
	
	#print r1
	#print r2
	# calc corelation function
	x, y = np.meshgrid( spikeTrain1, spikeTrain2)
	
	tRange=plotRange  # ms
	dt=timeResolution

	c12_dt, binsValues=np.histogram((-x+y).flatten(), bins=np.arange(-tRange,tRange+dt,dt))
	#print len(c12_dt), len(binsValues)
	c12_dt=( r1*c12_dt/float( len(spikeTrain1) )*1000.0*(1/dt) )/(r1*r2)-1
	bins_dt1=0.5*(binsValues[1:]+binsValues[:-1])
	
	#print len(c12_dt), len(bins_dt1)
	
	return 0.001*timeResolution*c12_dt, bins_dt1, r1, r2



# inputs are two list of spike times   in ms
# timeResolution                       in ms
# plotRange                            in ms
# returns correlation function (for negative t) in units of 1/Hz   
def calcSpikeTrainCorrelationFunction_NegT( spikeTrain1, spikeTrain2, timeResolution, plotRange ):
	
	# estimates of firing rate
	r1=float(len(spikeTrain1)*1000)/float( (spikeTrain1[-1]-spikeTrain1[0]) )   # Hz
	r2=float(len(spikeTrain2)*1000)/float( (spikeTrain2[-1]-spikeTrain2[0]) )   # Hz
	
	#print r1
	#print r2
	# calc corelation function
	x, y = np.meshgrid( spikeTrain1, spikeTrain2)
	
	tRange=plotRange  # ms
	dt=timeResolution

	c12_dt, binsValues=np.histogram((-x+y).flatten(), bins=np.arange(0,tRange,dt))
	#print len(c12_dt), len(binsValues)
	c12_dt=( r1*c12_dt/float( len(spikeTrain1) )*1000.0*(1/dt) )/(r1*r2)-1
	bins_dt1=0.5*(binsValues[1:]+binsValues[:-1])
	
	#print len(c12_dt), len(bins_dt1)
	
	return 0.001*timeResolution*c12_dt, bins_dt1




# calculates firing rates for a given type of stimulation
# protocol = CRRVS

def getFriringRates( protocol ):
	# get estimate of firing rates during CR as function of Delta for strong Samp
	sampArray=np.linspace(0.0, 5000.0, 11)
	DeltaArray=np.linspace(0.0, 500.0, 41) # ms

	seedArray=[10,12,14,16]
	seedSequenceArray=[100,110,120,130]

	timePointArray=np.arange(2020.0, 5600.0, 20).astype(int) # sec
	#sampArray=[2000.0]
	#DeltaArray=[50.0] # ms

	#timePointArray=[2020] # sec
	N=1002

	firingRates=np.zeros( ( len(sampArray),len(DeltaArray),len(seedArray), len(seedSequenceArray), len(timePointArray) ) ) # Hz

	dataDirectory='/Users/jkromer/Desktop/Projects/Stanford/scratch/largeNetworks/Hardbounds_only_STN_adjustParameters/STDP_Scan_Bimodel_scanParameters_only_STN_on_N_STN_1000_N_GPe_2_sigmaP_0.05/VRest_-38.0_tau_150_noiseIntSTN_1.3_STDPbeta_1.4_STDPtauR_4.0_STDPdelta_0.01_STDPtauPlus_10.0/cExcMax_400_cInhMax_0/excW_200.0_inhW_0.0/'

	counter=0
	maxCounter=len(sampArray)*len(DeltaArray)*len(seedArray)*len(seedSequenceArray)*len(timePointArray)


	for kSamp in range( len(sampArray) ):
		samp=sampArray[kSamp]
		for kDelta in range( len(DeltaArray) ):
			Delta=DeltaArray[kDelta]
			for kSeed in range( len(seedArray) ):
				seed = seedArray[kSeed]
				for kSeedSeq in range( len(seedSequenceArray) ):
					seedSeq=seedSequenceArray[kSeedSeq]
					
					timePoint=timePointArray[0]
					directory=dataDirectory+'seed_'+str(seed)

					if protocol=='CRRVS':
						
						directory+'/FinalBackup_RVSTstim/PulsesPerBurst_1_TimeBetBursts_'+str(Delta)+'/seedseq_'+str(seedSeq)+'_samp_'+str(samp)+'_Tstim_3700.0/'
						spikeTrainFile=directory+'spikeTimes_'+str(timePoint)+'_sec.npy'
						#print spikeTrainFile
						print(counter, '/',maxCounter )
						counter+=len(timePointArray)
						#print samp, Delta, seed, seedSeq, timePoint, r
						
						if os.path.isfile( spikeTrainFile ):
							print( 'data set found' )
							spikeTrain=np.load( spikeTrainFile )
							r=len(spikeTrain[:,0])/float(20.0*N)
							#print samp, Delta, seed, seedSeq, timePoint, r
							firingRates[kSamp, kDelta, kSeed, kSeedSeq, kTimePoint]=r
						

							for kTimePoint in range( 1,len(timePointArray) ):
								timePoint = timePointArray[ kTimePoint ]
								if protocol=='CRRVS':
									directory=dataDirectory+'seed_'+str(seed)+'/FinalBackup_RVSTstim/PulsesPerBurst_1_TimeBetBursts_'+str(Delta)+'/seedseq_'+str(seedSeq)+'_samp_'+str(samp)+'_Tstim_3700.0/'
									spikeTrainFile=directory+'spikeTimes_'+str(timePoint)+'_sec.npy'
									#print spikeTrainFile
									if os.path.isfile( spikeTrainFile ):
										spikeTrain=np.load( spikeTrainFile )
										r=len(spikeTrain[:,0])/float(20.0*N)
										
										firingRates[kSamp, kDelta, kSeed, kSeedSeq, kTimePoint]=r
								else:
									print( 'Error: unknown protocol')
									return 0
					else:
						print( 'Error: unknown protocol')
						return 0





	return firingRates




################# getSpikeTimeDifferences #################
def getSpikeTimeDifferences(spikesTimesOfPresynNeuron, spikesTimesOfPostsynNeuron, axonalDelay, dendriticDelay):
	
	# reverse order of spikes and spike times
	reverseSpikeTimesPre=np.flip( spikesTimesOfPresynNeuron, 0 )
	reverseSpikeTimesPost=np.flip( spikesTimesOfPostsynNeuron, 0 )
	
	DeltaT=[]
	
	while (len(reverseSpikeTimesPre)>1) and (len(reverseSpikeTimesPost)>1):
	
		# consider latest spike
		latestPreSynSpikeArrivalTime=reverseSpikeTimesPre[0]+axonalDelay
		latestPostSynSpikeArrivalTime=reverseSpikeTimesPost[0]+dendriticDelay
	
		# LTP
		if ( latestPostSynSpikeArrivalTime > latestPreSynSpikeArrivalTime ):
			DeltaT.append( latestPostSynSpikeArrivalTime-latestPreSynSpikeArrivalTime )
			reverseSpikeTimesPost=reverseSpikeTimesPost[1:]
		
		# LTD
		if ( latestPreSynSpikeArrivalTime > latestPostSynSpikeArrivalTime):
			DeltaT.append( -(latestPreSynSpikeArrivalTime-latestPostSynSpikeArrivalTime) )
			reverseSpikeTimesPre=reverseSpikeTimesPre[1:]
		
		# no influence
		if ( latestPreSynSpikeArrivalTime == latestPostSynSpikeArrivalTime ):
			DeltaT.append(0)
			DeltaT.append(0)
			reverseSpikeTimesPre=reverseSpikeTimesPre[1:]
			reverseSpikeTimesPost=reverseSpikeTimesPost[1:]
			
	return np.array(DeltaT)

################# getSpikeTimeDifferencesSequence #################
# generates one list of spike timing differences for the first 'NSynapses' synapses in the list of synapses 'synapses'
# a statistics of spike time differendces
def getSpikeTimeDifferencesSequence(spiketrain, synapses, NSynapses, NStart, NEnd): 

	# list of spike timing differences in ms
	DeltaTArray=[]
	combinedDeltaTArray=np.array([-1000])
	
	for kSynapse in range( NSynapses ): #len(synapses) ):
		#kSynapse=10
		print( kSynapse )
		kPreNeuron=synapses[kSynapse,1]
		kPostNeuron=synapses[kSynapse,0]

		#print kPreNeuron, kPostNeuron

		spikesTimesOfPresynNeuron=spiketrain[ spiketrain[:,0]==kPreNeuron ][:,1]*0.1   # ms
		spikesTimesOfPostsynNeuron=spiketrain[ spiketrain[:,0]==kPostNeuron ][:,1]*0.1   # ms
	
		# get time differences
		DeltaSpikeTimes=getSpikeTimeDifferences(spikesTimesOfPresynNeuron, spikesTimesOfPostsynNeuron, 3.0, 0.0)
		DeltaTArray.append( DeltaSpikeTimes )
	
		if len(combinedDeltaTArray)==0:
			spikeTimeDifferencesArray=np.flip( np.array(DeltaSpikeTimes), axis=0 )
			combinedDeltaTArray=np.concatenate(  (combinedDeltaTArray, spikeTimeDifferencesArray[NStart:NEnd] ), axis=1 )
		else:
			spikeTimeDifferencesArray=np.flip( np.array(DeltaSpikeTimes), axis=0 )
			combinedDeltaTArray=spikeTimeDifferencesArray[NStart:NEnd]
			
	return combinedDeltaTArray[1:], DeltaTArray



def plotSpikeTrain(directory, spikeTrainFileTime, Tend, TintervalLength, markersize):
	# check spikeTrain
	#Tend=2020
	spikeTrain=np.load(directory+'/spikeTimes_'+str(spikeTrainFileTime)+'.npy')

	PlotSpikeTrain=spikeTrain[ np.logical_and( 0.001*0.1*spikeTrain[:,1] > Tend-TintervalLength, 0.001*0.1*spikeTrain[:,1] < Tend)   ]
	x, y = 0.001*0.1*spikeTrain[:,1], spikeTrain[:,0]
	plt.scatter( x, y, s=markersize, color='black' , zorder=1)
	plt.xlim(Tend-TintervalLength,Tend)
	plt.xticks([Tend-TintervalLength,Tend], [Tend-TintervalLength,Tend])
	plt.xlabel('t in sec')

	return x, y



def getTrajectoryOfIntraAndInterPopWeights(  directory  ):
	
	weightTimeTrace=[]
	try: 
		parameters=np.load( directory+'/parameterSet.npy').item()
		synMatrix=scipy.sparse.load_npz( directory+'/2000_sec/synConnections.npz' )
	except: 
		print( "ERROR: no parameterSet.npy file found" )
		return 0
	N=parameters['N_GPe']+parameters['N_STN']
	
	filterIntraPopulationWeights=np.zeros( (N,N))
	Nelectordes=4
	electrodeSize=int( float(parameters['N_STN'])/float(Nelectordes) )
	for kElectro in range(4):
		filterIntraPopulationWeights[kElectro*electrodeSize:( kElectro+1 )*electrodeSize, kElectro*electrodeSize:( kElectro+1 )*electrodeSize ]=np.ones( (electrodeSize, electrodeSize) )

	filterInterPopulationWeights=np.ones( (N,N))-filterIntraPopulationWeights
	
	listOfBackupFiles=loadListOfBackupTimesteps( directory )
	for kBackup in range( len(listOfBackupFiles)-2 ):
		#print listOfBackupFiles[kBackup][1]
		cMatrixFilename=directory+'/'+listOfBackupFiles[kBackup][1]+'/cMatrix.npz'
		
		try: 
			cMatrix=scipy.sparse.load_npz(cMatrixFilename)
			
			
			# get ensemble averaged inter population weights
			#interpol
			# get ensemble averaged intra population weights
			
		except: break
		numberOfIntraPopulationWeights=np.sum( synMatrix.multiply(filterIntraPopulationWeights).A )
		intraopulationWeights=cMatrix.multiply( filterIntraPopulationWeights )
		intraopulationWeights=intraopulationWeights.A
	   

		meanIntraPopulationWeight=np.sum( intraopulationWeights)/numberOfIntraPopulationWeights
	   
		numberOfInterPopulationWeights=np.sum( synMatrix.multiply(filterInterPopulationWeights).A )
		interpopulationWeights=cMatrix.multiply( filterInterPopulationWeights ).multiply(synMatrix)
		meanInterPopulationWeight=interpopulationWeights.A
		meanInterPopulationWeight=np.sum( meanInterPopulationWeight )/numberOfInterPopulationWeights
		#print listOfBackupFiles[kBackup][1][:-4]
		weightTimeTrace.append([ float(listOfBackupFiles[kBackup][1][:-4]), meanIntraPopulationWeight, meanInterPopulationWeight ])
		
	return np.array( weightTimeTrace )



def getTrajectoryOfWeights_InterconnectingNeuronsWithDistance(  directory, distance  ):
	
	weightTimeTrace=[]
	print( directory+'/parameterSet.npy' )
	try: 
		parameters=np.load( directory+'/parameterSet.npy').item()
	except: 
		print( "ERROR: no parameterSet.npy file found" )
		return 0
	try: 
		synMatrix=scipy.sparse.load_npz( directory+'/2000_sec/synConnections.npz' )
	except: 
		print( "ERROR: no synConnections.npz file found" )
		return 0

	N=parameters['N_GPe']+parameters['N_STN']
	
	# matrix with ones at shifted diagonal
	filterDistancePopulationWeights=np.eye(N, k=distance)+np.eye(N, k=-distance)

	listOfBackupFiles=loadListOfBackupTimesteps( directory )
	for kBackup in range( len(listOfBackupFiles)-2 ):
		#print listOfBackupFiles[kBackup][1]
		cMatrixFilename=directory+'/'+listOfBackupFiles[kBackup][1]+'/cMatrix.npz'
		
		try: 
			cMatrix=scipy.sparse.load_npz(cMatrixFilename)
			
			# get ensemble averaged inter population weights
			#interpol
			# get ensemble averaged intra population weights
			
		except: break
		numberOfDistancePopulationWeights=np.sum( synMatrix.multiply(filterDistancePopulationWeights).A )
		distancePopulationWeights=cMatrix.multiply( filterDistancePopulationWeights )
		distancePopulationWeights=distancePopulationWeights.A
		meanDistancePopulationWeight=np.sum( distancePopulationWeights)/numberOfDistancePopulationWeights

		#print listOfBackupFiles[kBackup][1][:-4]
		weightTimeTrace.append([ float(listOfBackupFiles[kBackup][1][:-4]), meanDistancePopulationWeight, meanDistancePopulationWeight ])
		
	return np.array( weightTimeTrace )


def plotFiringRates( spiketrain, dt, neuronIndizes, limits ):

	import matplotlib.pyplot as plt

	# get limits
	kTimeStepMin=spiketrain[0,1]
	kTimeStepMax=spiketrain[-1,1]

	# T interval length
	msToSec=0.001
	T = msToSec*dt*( kTimeStepMax-kTimeStepMin )

	firingRates=[]

	for kNeuron in neuronIndizes:

		kNeuronSpikeTrain=spiketrain[ spiketrain[:,0] == kNeuron ]
		print( len(kNeuronSpikeTrain) )
		firingRates.append( [ kNeuron , float( len(kNeuronSpikeTrain) )/T ] )

	firingRates = np.array( firingRates )


	plt.hist(  firingRates[:,1] , bins=np.linspace( limits[0] , limits[1] ) )
	plt.show()

	return firingRates


def getPartOfSpikeTrain( spiketrain, tmin, tmax ):

    #stepsToSec = 0.0001
    return spiketrain[ np.logical_and(  stepsToSec*spiketrain[:,1]>tmin   ,  stepsToSec*spiketrain[:,1]<tmax  )]

def plotPartOfSpikeTrain( spiketrain, tmin, tmax ):

    import matplotlib.pyplot as plt

    #stepsToSec = 0.0001
    plotSpiketrain = getPartOfSpikeTrain( spiketrain, tmin, tmax )

    plt.scatter( stepsToSec*plotSpiketrain[:,1] , plotSpiketrain[:,0], color = 'black' )




