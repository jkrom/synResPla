##################################
# written by: Justus Kromer
##################################
# written and tested for Python 2.7.13
##################################
#
#	content:
#
#		simulation script to gen initial networks and simulate until a stationary state is reaced
#
#		genBackup_Files( str , int, float_1d, s, float_1d, VT, float_1d, scipy.sparse.csc_matrix, np.array float_2d, float_1d, float_1d, float_1d, float_1d , delayedSpikingNeurons ):
#
#       def initialize_asymmetric_Hebbian_STDP( system_parameters )
#
#       def initialize_Poisson_noise( system_parameters )
#
#       def initialize_system( system_parameters )
#       
#       def startFromBackup( BackupFolderName )
#
#	run: python gen_stationary_states.py 0.05 200.0 400 True -38.0 150 test 1.3 1.4 4.0 10.0 18
# calc backup

# imports 
import os
import scipy
import numpy as np
import pickle
import time
import scipy.sparse

##################################
# function: genBackup_Files( BackupFolderName, kStep, v, s, snoise, VT, switchOffSpikeTimes, cMatrix, synConnections, STNCenter, GPeCenter, lastSpikeTimeStep, evenPriorLastSpikeTimeStep , delayedSpikingNeurons)
def genBackup_Files( BackupFolderName, kStep, v, s, snoise, VT, switchOffSpikeTimes, cMatrix, synConnections, STNCenter, GPeCenter, lastSpikeTimeStep, evenPriorLastSpikeTimeStep , delayedSpikingNeurons):
#
#       Generates backup for restarting simulation later.
#       The backup is generated in folder BackupFolderName
#
#       input: NSTN, NGPe
#           BackupFolderName ... complete path to folder used for backup 
#           kStep ... current time step of simulation
#           v ... float_1d containing membrane potentials of all neurons in mv
# FIXME: add units for conductances
#           s ... current state of synaptic conductances  
#           snoise ... current state of noise conductances
#           VT ... current state of dynamic threshold
#           switchOffSpikeTimes ... contains time in ms when spike of neuron ends
#           cMatrix ... scipy.sparse.csc_matrix    current state of all synaptic weights    
#           synConnections ... number of GPe neurons that need to be placed
#           STNCenter ... np.array of float_3d     3d center coordinates of all neurons in STN
#           GPeCenter ... np.array of float_3d     3d center coordinates of all neurons in GPe
#           lastSpikeTimeStep ... time steps of last spikes
#           evenPriorLastSpikeTimeStep ... time steps of spikes before last spikes
#           delayedSpikingNeurons ... array realizing delayed input
#       return: non   

    # generate backup directory if it doesn't exist
    if os.path.exists( BackupFolderName ) == False:
        os.makedirs( BackupFolderName )

    # save state of synaptic weights
    scipy.sparse.save_npz(BackupFolderName+'/cMatrix.npz', cMatrix)

    # save connectivity matrix     entries:    1 excitatory connection,  -1 inhibitory connections,  0 no connections
    # synConnectionsBackup[i,j]  contains entries for connection from presynaptic neuron j to postsynaptic neuron i
    synConnectionsBackup=scipy.sparse.csc_matrix(synConnections.astype(int))
    scipy.sparse.save_npz(BackupFolderName+'/synConnections.npz', synConnectionsBackup)

    # save neuron positions
    # 3d postions in mm
    np.save(BackupFolderName+'/STNCenter.npy', STNCenter)
    np.save(BackupFolderName+'/GPeCenter.npy', GPeCenter)

    # time steps of last spikes
    np.save(BackupFolderName+'/lastSpikeTimeStep.npy', lastSpikeTimeStep.astype(int) )
    np.save(BackupFolderName+'/evenPriorLastSpikeTimeStep.npy', evenPriorLastSpikeTimeStep.astype(int) )

     # save delayed synaptic interactions
    np.save(BackupFolderName+'/delayedSpikingNeurons.npy', delayedSpikingNeurons)
     
   # save system state
   # generates binary file that contains 
   #    [kStep] ...  [time step]
   #       v    ...  current state of membrane voltages in mV
   #       s    ...  synpatic conductances
   #     snoise ...  noise conductances
   #      VT    ...  threshold potentials in mV
   #    switchOffSpikeTimes ...  end of the neurons' spikes in ms
   # threshold potentials
    systemState=(np.array([kStep]), v, s, snoise, VT, switchOffSpikeTimes)
    np.save(BackupFolderName+'/systemState.npy', systemState)

    # save state of random number generator
    stateOfRandomNumberGenerator= np.random.get_state()
    with open(BackupFolderName+'/npRandomState.pickle','wb') as f:
        pickle.dump( stateOfRandomNumberGenerator, f )


##################################
# function:startFromBackup( BackupFolderName )
def startFromBackup( BackupFolderName ):
#
#       Loads backup for restarting simulation later.
#       The backup is generated in folder BackupFolderName
#
#       input: NSTN, NGPe
#           BackupFolderName ... complete path to folder used for backup 
#       return:
#           kStep ... current time step of simulation
#           v ... float_1d containing membrane potentials of all neurons in mv
# FIXME: add units for conductances
#           s ... current state of synaptic conductances  
#           snoise ... current state of noise conductances
#           VT ... current state of dynamic threshold
#           switchOffSpikeTimes ... contains time in ms when spike of neuron ends
#           cMatrix ... scipy.sparse.csc_matrix    current state of all synaptic weights    
#           synConnections ... number of GPe neurons that need to be placed
#           STNCenter ... np.array of float_3d     3d center coordinates of all neurons in STN
#           GPeCenter ... np.array of float_3d     3d center coordinates of all neurons in GPe
#           lastSpikeTimeStep ... time steps of last spikes
#           evenPriorLastSpikeTimeStep ... time steps of spikes before last spikes
#           npRandomState ... state of random number generator
#           delayedSpikingNeurons ... array realizing delayed input
#           numberOfDelayedTimeSteps ... length of 'delayedSpikingNeurons'
   
    # load cMatrix
    cMatrixFile=BackupFolderName+'/cMatrix.npz'
    if os.path.isfile( cMatrixFile ):
        cMatrix = scipy.sparse.load_npz( cMatrixFile )
    else:
        print( cMatrixFile )
        print('Error: cMatrixFile not found.')

    # load connectivity matrix     entries:    1 excitatory connection,  -1 inhibitory connections,  0 no connections
    # synConnectionsBackup[i,j]  contains entries for connection from presynaptic neuron j to postsynaptic neuron i
    synConnectionsFile=BackupFolderName+'/synConnections.npz'
    if os.path.isfile( synConnectionsFile ):
        synConnectionsBackup = scipy.sparse.load_npz( synConnectionsFile )
        synConnections=synConnectionsBackup.A
    else:
        print( synConnectionsFile )
        print( 'Error: synConnectionsFile not found.')

    # load neuron positions
    # 3d postions in mm
    STNCenterFile=BackupFolderName+'/STNCenter.npy'
    if os.path.isfile( STNCenterFile ):
        STNCenter = np.load( STNCenterFile )
    else:
        print( STNCenterFile )
        print( 'Error: STNCenterFile not found.' )

    # load neuron positions
    # 3d postions in mm
    GPeCenterFile=BackupFolderName+'/GPeCenter.npy'
    if os.path.isfile( GPeCenterFile ):
        GPeCenter = np.load( GPeCenterFile )
    else:
        print( GPeCenterFile )
        print( 'Error: GPeCenterFile not found.' )

    # time steps of last spikes
    lastSpikeTimeStepFile=BackupFolderName+'/lastSpikeTimeStep.npy'
    if os.path.isfile( lastSpikeTimeStepFile ):
        lastSpikeTimeStep=np.load( lastSpikeTimeStepFile )
        evenPriorLastSpikeTimeStepFile=BackupFolderName+'/evenPriorLastSpikeTimeStep.npy'
        if os.path.isfile( evenPriorLastSpikeTimeStepFile ):
            evenPriorLastSpikeTimeStep=np.load( evenPriorLastSpikeTimeStepFile )
        else:
            print( 'Warning: no data for evenPriorLastSpikeTimeStep found. Using evenPriorLastSpikeTimeStep = lastSpikeTimeStep instead.' )
            evenPriorLastSpikeTimeStep = np.copy( lastSpikeTimeStep )

    else:
        print( lastSpikeTimeStepFile )
        print( 'Error: lastSpikeTimeStepFile not found.' )
        exit()



    # load delayed synaptic interactions
    delayedSpikingNeuronsFile=BackupFolderName+'/delayedSpikingNeurons.npy'
    if os.path.isfile( delayedSpikingNeuronsFile ):
        delayedSpikingNeurons=np.load( delayedSpikingNeuronsFile )
    else:
        print( delayedSpikingNeuronsFile)
        print( 'Error: delayedSpikingNeuronsFile not found.')
    numberOfDelayedTimeSteps=len(delayedSpikingNeurons)
    # transform to list so does each entry can contain different number of elements
    delayedSpikingNeurons = delayedSpikingNeurons.tolist()

    # load system state
    systemStateFile=BackupFolderName+'/systemState.npy'
    if os.path.isfile( systemStateFile ):
        systemState = np.load( systemStateFile )
        kStep, v, s, snoise, VT, switchOffSpikeTimes = systemState
        kStep=kStep[ 0 ]+1   # backup is taken at end of time step
    else:
        print( systemStateFile)
        print( 'Error: systemStateFile not found.')
     

    # save state of random number generator
    stateOfRandomNumberGeneratorFile=BackupFolderName+'/npRandomState.pickle'
    if os.path.isfile( stateOfRandomNumberGeneratorFile ):
        with open( stateOfRandomNumberGeneratorFile,'rb') as f:
            npRandomState=pickle.load( f ) 
    else:
        print( stateOfRandomNumberGeneratorFile)
        print( 'Error: stateOfRandomNumberGeneratorFile not found.')

    # return loaded stateVariables
    return kStep, v, s, snoise, VT, switchOffSpikeTimes, cMatrix, synConnections, STNCenter, GPeCenter, lastSpikeTimeStep, evenPriorLastSpikeTimeStep, npRandomState, delayedSpikingNeurons, numberOfDelayedTimeSteps



########################################################
#  function:  initialize_asymmetric_Hebbian_STDP
#       precalculates weight updates in order to speed up simulations
#
#       input:   system_parameters 
#           system_parameters  .. parameter set used for simulations
#
#       output: weightUpDatesPostSynSpike , weightUpDatesPreSynSpike , STDPCutOffSteps
#             weightUpDatesPostSynSpike ... weight update according to STDP functions for case tpost > tpre
#             weightUpDatesPreSynSpike ... weight update according to STDP functions for case tpost < tpre   
#             STDPCutOffSteps   ... maximal time lag considered in units of time steps           
def initialize_asymmetric_Hebbian_STDP( system_parameters ):

    # load needed parameters from system_parameters
    weightUpdateRateSTDP=system_parameters['STDP_weightUpdateRateSTDP']
    # ratio between overall depression and potentiation
    beta=system_parameters['STDP_beta']
    # STDP decay time for tpost > tpre in ms
    tauPlus=system_parameters['STDP_tauPlus']  # ms
    # ration between decay times 
    tauRatio=system_parameters['STDP_tauRatio']
    # cutoff time, weight updates for larger time lags are neglected
    tCutoff=system_parameters['STDP_tCutoff']  # ms
    # STDP decay time for tpost < tpre in ms
    tauMinus=tauRatio*tauPlus   # ms
    # integration time step in ms
    dt = system_parameters['dt']

    # asymmetric Hebbian STDP function  
    # dt in units of ms
    # 1) dt = |tpost - tpre| and tpost > tpre
    def RuleForPosDeltaT(dt ):
        return weightUpdateRateSTDP*np.exp( -(dt)/(tauPlus) )

    # 2) dt = |tpost - tpre| and tpost < tpre   
    def RuleForNegDeltaT(dt):
        return -weightUpdateRateSTDP*beta/tauRatio*np.exp( (-dt)/(tauRatio*tauPlus) )

    # preevaluate in order to speed up simulations
    times=np.arange( 0, tCutoff, dt )
    STDPCutOffSteps=len(times)

    weightUpDatesPostSynSpike=RuleForPosDeltaT( times )
    # dt = 0 does not lead to weight update
    weightUpDatesPostSynSpike[0]=0

    weightUpDatesPreSynSpike=RuleForNegDeltaT( times )
     # dt = 0 does not lead to weight update
    weightUpDatesPreSynSpike[0]=0

    # return results
    return weightUpDatesPostSynSpike , weightUpDatesPreSynSpike , STDPCutOffSteps




########################################################
#  function:  initialize_Poisson_noise
#       Calculates Poisson rates used throughout simulation. 
#       generates initial realization of Poisson input spike trains
#
#       input:   system_parameters 
#           system_parameters  .. parameter set used for simulations
#
#       output: spikeArrival , InputRates, numberOfPredrawnSpikeArrivals
#             spikeArrival ... contains current realization of number of arriving spikes per time bin
#             InputRates ... matrix of floats N x numberOfPredrawnSpikeArrivals with mean spike count per time bin
#             numberOfPredrawnSpikeArrivals   ... int number of time bins for which Poisson spike trains are calculated simultaneously  
def initialize_Poisson_noise( system_parameters ):

    # load needed parameters from system_parameters
    InputRateSTN = system_parameters['InputRateSTN']
    InputRateGPe = system_parameters['InputRateGPe']

    # network
    N_STN=system_parameters['N_STN']
    N_GPe=system_parameters['N_GPe']

    # total number of neurons
    N=N_STN+N_GPe

    dt = system_parameters['dt'] # ms
    kSave=int( system_parameters['Trec']*1000./dt)

    ################################################################
    # noise should be a ratio of kSave
    numberOfPredrawnSpikeArrivals=10000           # draw noise realization for one second of sim time
    # check whethet it is ratio of kSave
    if kSave % numberOfPredrawnSpikeArrivals != 0:
        print( 'Warning numberOfPredrawnSpikeArrivals is not a ratio of kSave')

    # mean spike count per time bin for GPe
    AvNumberOfSpikesPerTimeStepGPe=0.001*InputRateGPe*dt  # 
    # mean spike count per time bin for STN
    AvNumberOfSpikesPerTimeStepSTN=0.001*InputRateSTN*dt  # 
          
    #  mean spike count per time bin for each neuron
    InputRates=np.full( N , AvNumberOfSpikesPerTimeStepSTN)
    InputRates[N_STN:]=np.full( N_GPe ,AvNumberOfSpikesPerTimeStepGPe )

    # generate matrix of floats N x numberOfPredrawnSpikeArrivals with mean spike count per time bin
    xRates, yRates=np.meshgrid(InputRates, np.zeros(numberOfPredrawnSpikeArrivals))
    InputRates=np.transpose( xRates ) # average number of incoming spikes used for poisson random numbers

    # initial noise realization
    spikeArrival=np.random.poisson(lam=(InputRates))
    spikeArrival=np.transpose( spikeArrival )

    return spikeArrival , InputRates, numberOfPredrawnSpikeArrivals




########################################################
#  function:  create_outputFolder
#       Creates output folder 
#       outputDirectory + '/STDP_on/seed+'+str(initialSeed)
#
#       input:   outputDirectory , STDPon, initialSeed  
#           outputDirectory  .. path to parent directory
#           STDP_on  ...  bool True if STDP is considered
#           initialSeed ... inital seed for random number generator
#
#       output: outputFolder
#             outputFolder ...  outputDirectory + '/STDP_on/seed+'+str(initialSeed)
def create_outputFolder( outputDirectory , STDPon, initialSeed ):

    # set output folder file name
    if STDPon == True:
        outputFolder= outputDirectory + '/STDP_on/seed_'+str(initialSeed)
        #outputFolder= outputDirectory + 'STDP_Scan_Bimodel_scanParameters_only_STN_on_N_STN_'+str(N_STN)+'_N_GPe_'+str(N_GPe)+'_sigmaP_'+ str(system_parameters['sigmaP']) + '/VRest_'+sys.argv[5]+'_tau_'+sys.argv[6]+'_noiseIntSTN_'+str(system_parameters['noiseSTN'])+'_STDPbeta_'+sys.argv[9]+'_STDPtauR_'+sys.argv[10]+'_STDPdelta_'+str(system_parameters['STDP_weightUpdateRateSTDP'])+'_STDPtauPlus_'+str(system_parameters['STDP_tauPlus'])+'/cExcMax_' + sys.argv[3] + '_cInhMax_0/excW_'+ sys.argv[2] + '_inhW_0.0/seed_'+sys.argv[12]
    if STDPon == False:
        outputFolder= outputDirectory + '/STDP_off/seed_'+str(initialSeed)
        #outputFolder= outputDirectory + 'STDP_Scan_Bimodel_scanParameters_only_STN_off_N_STN_'+str(N_STN)+'_N_GPe_'+str(N_GPe)+'_sigmaP_'+ str(system_parameters['sigmaP']) + '/VRest_'+sys.argv[5]+'_tau_'+sys.argv[6]+'_noiseIntSTN_'+str(system_parameters['noiseSTN'])+'_STDPbeta_'+sys.argv[9]+'_STDPtauR_'+sys.argv[10]+'_STDPdelta_'+str(system_parameters['STDP_weightUpdateRateSTDP'])+'_STDPtauPlus_'+str(system_parameters['STDP_tauPlus'])+'/cExcMax_' + sys.argv[3] + '_cInhMax_0/excW_'+ sys.argv[2] + '_inhW_0.0/seed_'+sys.argv[12]

    # generate outputFolder if it doesn't alreade exist
    if os.path.exists( outputFolder ) == False:
        os.makedirs( outputFolder )

    return outputFolder



########################################################
#  function:  create_outputFolder
#       Creates output folder 
#       outputDirectory + '/STDP_on/seed+'+str(initialSeed)
#
#       input:   outputDirectory , STDPon, initialSeed  
#           outputDirectory  .. path to parent directory
#           STDP_on  ...  bool True if STDP is considered
#           initialSeed ... inital seed for random number generator
#
#       output: outputFolder
#             outputFolder ...  outputDirectory + '/STDP_on/seed+'+str(initialSeed)
def create_outputFolder_short( outputDirectory  ):

    # generate outputFolder if it doesn't alreade exist
    if os.path.exists( outputDirectory ) == False:
        os.makedirs( outputDirectory )

    return outputDirectory


########################################################
#  function:  set_initial_conditions_for_nodes
#       Sets initial conditions as used in manuscript
#
#       input:   system_parameters
#           system_parameters  .. parameter set used for simulations
#
#       output: Vrest , tau , v, s, Snoise, VT, Vnoise
#             Vrest ...  float_1d with resting potentiaons for STN an GPe neurons
#             tau   ...  float_1d membrane time constant in ms
#             v   ...  float_1d membrane potential in mV
#             s   ...  np.array Nx2   first column is conductance of exc. and secodn of inh. neurons
#             Snoise   ...  float_1d   conductance of noisy input synapses
#             VT ... float_1d    threshold potential in mV
#             Vnoise .. reversal potential for noise conductances
def set_initial_conditions_for_nodes( system_parameters ):

    # numbers of STN and GPe neurons
    N_STN=system_parameters['N_STN']
    N_GPe=system_parameters['N_GPe']    

    # total number of neurons
    N=N_STN+N_GPe

    Vrest=np.full( N, system_parameters['VRestSTN'] )
    Vrest[N_STN:]=np.full( N_GPe, system_parameters['VRestGPe'] )

    tau=np.random.normal( system_parameters['tauSTN'] , system_parameters['tauSTN']*system_parameters['sigmaP'], N )
    # and STN neurons
    tau[N_STN:]=np.random.normal( system_parameters['tauGPe'], system_parameters['tauGPe']*system_parameters['sigmaP'], N_GPe )

    v=np.random.uniform( system_parameters['Vreset'] , system_parameters['VTRest'], N )
    s=np.zeros( (N,2) )
    Snoise=np.zeros( N )
    # threshold
    VT=np.full( N, system_parameters['VTRest'] )
    # reversal potential for noisy input
    Vnoise=np.full( N, system_parameters['Vinh'] )
    Vnoise[:N_STN]=np.full( N_STN ,system_parameters['Vexc'] )

    ################################################################


    return Vrest , tau , v, s, Snoise, VT, Vnoise




########################################################
#  function:  initialize_system
#       Initializes system variables. mostly to speed up simulation
#
#       input:   system_parameters
#           system_parameters  .. parameter set used for simulations
#
#       output: STDPon, N, N_STN, VR, tauVT, VTspike, VTRest, tau_spike, V_spike, Vexc, Vinh, tauNoise, noiseIntensities, synaptic_offset_after_spike, tauSyn, dt, Tmax, t, recordedSpikes, kSave, StepsTauSynDelaySTNSTN, StepsTauSynDelayGPeGPe, StepsTauSynDelayGPeSTN, StepsTauSynDelaySTNGPe, a_delaySteps, b_delaySteps, c_delaySteps, delayedSpikingNeurons, numberOfDelayedTimeSteps , est_Spike_Times, switchOffSpikeTimes, basicFilterSpikingNeurons, lastSpikeTimeStep, evenPriorLastSpikeTimeStep, kWeightOutput, NeuronIndices, start_time, borderForScaningForSpikeValues, borderForScaningNeuronsPassingSpikingThreshold, STDPAlreadyOn
#            FIXME: add description
def initialize_system( system_parameters ):
    
    # bool that is set True
    STDPon=True
    STDPAlreadyOn=False


    # number of neurons in STN and GPe
    N_STN=system_parameters['N_STN']
    N_GPe=system_parameters['N_GPe']

    # total number of neurons
    N=N_STN+N_GPe

    # maximum coupling strengths
    # exc coupling
    cMaxExc=system_parameters['cMaxExc']    
    # inh coupling
    cMaxInh=system_parameters['cMaxInh']

    # reset potential for membrane
    VR=system_parameters['Vreset'] 

    # dynamic threshold
    tauVT=system_parameters['tauVT']
    VTspike=system_parameters['VTspike'] # mV
    VTRest=system_parameters['VTRest'] # mV

    # shape of spikes
    tau_spike=system_parameters['tau_spike']
    V_spike=system_parameters['V_spike']

    # resting potentials for synapses
    Vexc=system_parameters['Vexc']
    Vinh=system_parameters['Vinh']

    # synaptic time scales
    tauSynExc=system_parameters['tauSynExc'] # synaptic time constant [ms] according to Ebert et al. 2014
    tauSynInh=system_parameters['tauSynInh']
    tauNoise = system_parameters['tauNoise']

    # noise intensity
    noiseIntensities=np.full( N, system_parameters['noiseSTN'] )
    noiseIntensities[N_STN:]=np.full( N_GPe, system_parameters['noiseGPe'] )


    # preevaluate synaptic quantities to speed up simulation
    synaptic_offset_after_spike=1./(float(N)) # mV
    tauSyn=np.array([1/tauSynExc,1/tauSynInh])

    # integration
    dt=system_parameters['dt'] # ms
    # maximum integration time 
    Tmax=system_parameters['Tend']-system_parameters['Tinit']   # sec

    # current time
    t=system_parameters['Tinit']*1000   # ms

    # initialize number of recorded spikes
    recordedSpikes=0

    # integer specifying how often backups are done
    kSave=int( system_parameters['Trec']*1000./dt) # time steps


    # initialize synaptic transmission delay 
    tauSynDelaySTNSTN=system_parameters['tauSynDelaySTNSTN']
    StepsTauSynDelaySTNSTN=int(tauSynDelaySTNSTN/dt) # ms
    tauSynDelayGPeGPe=system_parameters['tauSynDelayGPeGPe']
    StepsTauSynDelayGPeGPe=int(tauSynDelayGPeGPe/dt) # ms
    tauSynDelayGPeSTN=system_parameters['tauSynDelayGPeSTN']
    StepsTauSynDelayGPeSTN=int(tauSynDelayGPeSTN/dt) # ms
    tauSynDelaySTNGPe=system_parameters['tauSynDelaySTNGPe']
    StepsTauSynDelaySTNGPe=int(tauSynDelaySTNGPe/dt) # ms

    # FIXME: currently only the case where delays differ is implemented 
    # this the the corresponding time delays in time steps
    a_delaySteps=StepsTauSynDelaySTNSTN
    b_delaySteps=StepsTauSynDelayGPeGPe
    c_delaySteps=StepsTauSynDelaySTNGPe

    if (a_delaySteps) == b_delaySteps or (a_delaySteps == c_delaySteps) or (b_delaySteps == c_delaySteps):

        print( 'Error: STN, GPE, and inter network delays have to differ! Or corresponding parts of the code have to be rewritten' )

    #################################
    # generate delayedSpikingNeurons
    # maximum delay in time steps
    tauSynDelay=max([tauSynDelaySTNSTN,tauSynDelayGPeGPe,tauSynDelayGPeSTN,tauSynDelaySTNGPe])
    # is a list of list of synapses containing information about pre and postsynaptic neurons and related delay times
    delayedSpikingNeurons=[np.zeros((1,3)) for x in np.arange(0,tauSynDelay, 0.1)]
    numberOfDelayedTimeSteps=len(delayedSpikingNeurons)

    # size at which 'spikeTimes' is generated
    est_Spike_Times=1000000
    
    # switchOffSpikeTimes contains the times at which neurons stop being in piking mode
    switchOffSpikeTimes=np.full( N, -tau_spike ) # ms

    # create filter for exc/inh neurons
    basicFilterSpikingNeurons=np.zeros( (N,2) )
    # filter already consideres coupling strengths and synaptic timescales according to differential equations
    # excitatory weights
    basicFilterSpikingNeurons[:N_STN,0]=cMaxExc/tauSynExc*np.ones(N_STN)
    # inhibitory weights
    basicFilterSpikingNeurons[N_STN:,1]=cMaxInh/tauSynInh*np.ones(N_GPe)

    # initialize array of last spike times
    lastSpikeTimeStep=np.zeros( N ).astype(int)
    # and spikes before last spike times
    evenPriorLastSpikeTimeStep=np.zeros( N ).astype(int)

    # every 'kWeightOutput' mean synaptic weights are calculated and stored
    kWeightOutput=system_parameters['mWeightOutputEverySteps']

    # array with indices of all neurons
    NeuronIndices=np.arange(N)

    # start time of the simulation (real time)
    start_time = time.time()

    # threshold to scan whether neurons are currently spiking
    borderForScaningForSpikeValues=0.5*(V_spike+VTspike) # mV
    # threshold to scan whether membrane potential has passed threshold
    borderForScaningNeuronsPassingSpikingThreshold=(VTspike-1.0) # mV

    # return objects
    return STDPon, N, N_STN, VR, tauVT, VTspike, VTRest, tau_spike, V_spike, Vexc, Vinh, tauNoise, noiseIntensities, synaptic_offset_after_spike, tauSyn, dt, Tmax, t, recordedSpikes, kSave, StepsTauSynDelaySTNSTN, StepsTauSynDelayGPeGPe, StepsTauSynDelayGPeSTN, StepsTauSynDelaySTNGPe, a_delaySteps, b_delaySteps, c_delaySteps, delayedSpikingNeurons, numberOfDelayedTimeSteps , est_Spike_Times, switchOffSpikeTimes, basicFilterSpikingNeurons, lastSpikeTimeStep, evenPriorLastSpikeTimeStep, kWeightOutput, NeuronIndices, start_time, borderForScaningForSpikeValues, borderForScaningNeuronsPassingSpikingThreshold, STDPAlreadyOn



########################################################
#  function:  initialize_sim_objects
#        basically unpacks sim_objects
#
#       Initializes system variables. mostly to speed up simulation
#
#       input:   sim_objects
#           sim_objects  .. struct containing objects related to the network structure 
#
#       output: maxNumberOfPreSynapticNeurons, maxNumberOfPostSynapticNeurons, numpyPreSynapticNeurons, numpyPostSynapticNeurons, transmissionDelaysPreSynNeurons, transmissionDelaysPostSynNeurons, csc_Zero, csc_Ones
#            FIXME: add description
def initialize_sim_objects( sim_objects ):

    # maximum number of presynatpic neurons
    maxNumberOfPreSynapticNeurons = sim_objects['max_N_pre']
    # maximum number of postsynaptic neurons
    maxNumberOfPostSynapticNeurons = sim_objects['max_N_post']

    # numpy array of dimension  N x maxNumberOfPreSynapticNeurons
    # entry i contains indices of all presynaptic neurons of neuron i
    numpyPreSynapticNeurons = sim_objects['numpyPreSynapticNeurons']
    # numpy array of dimension  N x maxNumberOfPostSynapticNeurons
    # entry i contains indices of all postsynaptic neurons of neuron i
    numpyPostSynapticNeurons  = sim_objects['numpyPostSynapticNeurons']
    # array of dimension  N x maxNumberOfPreSynapticNeurons
    # entry i j contains transmission delay in timesteps for synapse j - > i
    # empty elements are set to -1
    transmissionDelaysPreSynNeurons  = sim_objects['td_PreSynNeurons']
    # array of dimension  N x maxNumberOfPreSynapticNeurons
    # entry i j contains transmission delay in timesteps for synapse i - > j
    # empty elements are set to -1
    transmissionDelaysPostSynNeurons  = sim_objects['td_PostSynNeurons']
    # scipy sparse matrix containing zeros
    csc_Zero = sim_objects['csc_Zero'] 
    # scipy sparse matrix containing ones
    csc_Ones = sim_objects['csc_Ones']

    return maxNumberOfPreSynapticNeurons, maxNumberOfPostSynapticNeurons, numpyPreSynapticNeurons, numpyPostSynapticNeurons, transmissionDelaysPreSynNeurons, transmissionDelaysPostSynNeurons, csc_Zero, csc_Ones

########################################################
#  function:  reproduce_initial_conditions
#
#       When a backup is loaded, all neuron parameters are generated as in original simulation.
#       Basically repeats beginning of 'get_stationary_states.py' for initial seed so that 
#       after loading a backup file same realization of original system can be reproduced without saving every single 
#       parameter.
#
#       input:   sim_objects
#           sim_objects  .. struct containing objects related to the network structure 
#           synConnections ... adjacency matric, entries are +1 (excitatory con. ), -1 (inhibitory connection), 0 (no connection)
#
#       output: maxNumberOfPreSynapticNeurons, maxNumberOfPostSynapticNeurons, numpyPreSynapticNeurons, numpyPostSynapticNeurons, transmissionDelaysPreSynNeurons, transmissionDelaysPostSynNeurons, csc_Zero, csc_Ones
#            FIXME: add description
def reproduce_initial_conditions( system_parameters , synConnections, cMatrix ):


    # set to initial seed in order to get the same distribution of node parameters
    np.random.seed( system_parameters['initialSeed'] )

    # numbers of STN and GPe neurons
    N_STN=system_parameters['N_STN']
    N_GPe=system_parameters['N_GPe']
    # total number of neurons
    N=N_STN+N_GPe

    # renerate esting potentials
    VRestGPe=system_parameters['VRestGPe'] # mV
    VRestSTN=system_parameters['VRestSTN'] # mV
    Vrest=np.full( N, VRestSTN ) # mV
    Vrest[N_STN:]=np.full( N_GPe, VRestGPe ) # mV

    # parameter diversity
    #sigmaP=0.05 # 5% of mean
    sigmaP=system_parameters['sigmaP']

    # reset value for membrane potential
    VR=system_parameters['Vreset']  # mV
    # set timescale for STN neurons
    tau=np.random.normal( system_parameters['tauSTN'] , system_parameters['tauSTN']*sigmaP, N )  # ms
    # and STN neurons
    tau[N_STN:]=np.random.normal( system_parameters['tauGPe'], system_parameters['tauGPe']*sigmaP, N_GPe )  # ms

    # dynamic threshold
    tauVT=system_parameters['tauVT']  # ms
    VTspike=system_parameters['VTspike'] # mV
    VTRest=system_parameters['VTRest'] # mV


    # synapses
    # synaptic reversal potentials
    Vexc=system_parameters['Vexc']  # mV
    Vinh=system_parameters['Vinh']  # mV
    # synaptic time scales
    tauSynExc=system_parameters['tauSynExc']  # ms # synaptic time constant [ms] according to Ebert et al. 2014
    tauSynInh=system_parameters['tauSynInh']  # ms
    tauNoise = system_parameters['tauNoise']  # ms
    # preevaluate 1/tau... to speed up simulation
    synaptic_offset_after_spike=1./(float(N)) # mV
    tauSyn=np.array([1/tauSynExc,1/tauSynInh])  # ms

    # noise intensity
    noiseIntensities=np.full( N, system_parameters['noiseSTN'] )
    noiseIntensities[N_STN:]=np.full( N_GPe, system_parameters['noiseGPe'] )
    # Input rates according to Ebert Front Comp Neuro 2014
    InputRateGPe=system_parameters['InputRateGPe'] 
    InputRateSTN=system_parameters['InputRateSTN']
    # reversal potential for noise conductances
    Vnoise=np.full( N,Vinh )  # mV
    Vnoise[:N_STN]=np.full( N_STN ,Vexc )  # mV

    # spike generation
    # shape of spikes
    tau_spike=system_parameters['tau_spike']  # ms
    V_spike=system_parameters['V_spike']  # mV
    #  first column neuron index, second column spike time in ms
    switchOffSpikeTimes=np.full( N, -tau_spike )
    borderForScaningForSpikeValues=0.5*(V_spike+VTspike)
    borderForScaningNeuronsPassingSpikingThreshold=(VTspike-1.0)
    # initialize stimuluation current
    Istim=np.zeros( N )


    # if True, STDP is applied
    STDPAlreadyOn=True     
    # indices of all neurons
    NeuronIndices=np.arange(N)
   # integration time step
    dt=system_parameters['dt']  # ms

    # calc same tmax as for original simulation
    system_parameters['Tinit']= 1

    ###############################################
    #   parameters used to generate output 
    recordedSpikes=0
    kSave=int( system_parameters['Trec']*1000./dt)
    # other
    est_Spike_Times=1000000
    ################################################################
    kWeightOutput=system_parameters['mWeightOutputEverySteps']
    meanWeightTimeSeries=np.zeros(  ( int( float( kSave )/float( kWeightOutput ) ) , 5 ) )

    #########################################################################################
    #    in the following additional arrays are introduced to speed up simulations
    #     get indicec post and presynaptic neurons to speed up STDP
    #     arrays are filled based on 'synConnections' loaded from backup file
    PostSynNeurons = {}
    PreSynNeurons = {}
    # initialize synaptic transmission delay 
    tauSynDelaySTNSTN=system_parameters['tauSynDelaySTNSTN']
    StepsTauSynDelaySTNSTN=int(tauSynDelaySTNSTN/dt) # ms
    tauSynDelayGPeGPe=system_parameters['tauSynDelayGPeGPe']
    StepsTauSynDelayGPeGPe=int(tauSynDelayGPeGPe/dt) # ms
    tauSynDelayGPeSTN=system_parameters['tauSynDelayGPeSTN']
    StepsTauSynDelayGPeSTN=int(tauSynDelayGPeSTN/dt) # ms
    tauSynDelaySTNGPe=system_parameters['tauSynDelaySTNGPe']
    StepsTauSynDelaySTNGPe=int(tauSynDelaySTNGPe/dt) # ms

    # max numbers of corresponding synapses
    maxNumberOfPostSynapticNeurons=0
    maxNumberOfPreSynapticNeurons=0

    for kNeuron in range(N_STN):

        PostSynNeurons[kNeuron]=np.nonzero( ( synConnections[:,kNeuron].astype(int) ).tolist() )[0].tolist()
        PreSynNeurons[kNeuron]=np.nonzero( ( synConnections[kNeuron,:].astype(int) ).tolist() )[0].tolist()

        if maxNumberOfPostSynapticNeurons<len(PostSynNeurons[kNeuron]):
            maxNumberOfPostSynapticNeurons=len(PostSynNeurons[kNeuron])
        if maxNumberOfPreSynapticNeurons<len(PreSynNeurons[kNeuron]):
            maxNumberOfPreSynapticNeurons=len(PreSynNeurons[kNeuron])

    # generate numpy array with post synaptic neurons to speed up simulations ...
    numpyPostSynapticNeurons=np.full((N,maxNumberOfPostSynapticNeurons),N+1)
    numpyPreSynapticNeurons=np.full((N,maxNumberOfPreSynapticNeurons),N+1)

    # ... and corresponding matrix containing transimission delays in time steps
    transmissionDelays=np.full((N,maxNumberOfPostSynapticNeurons),-1.0)
    transmissionDelaysPreSynNeurons=np.full((N,maxNumberOfPreSynapticNeurons),-1.0)

    # gen numpy array with post synaptic neurons
    for kPreSyn in range(N_STN):
        postSynNeuronskPre=PostSynNeurons[kPreSyn]
        for kPostSyn in range(len(postSynNeuronskPre)):
            numpyPostSynapticNeurons[kPreSyn, kPostSyn]=postSynNeuronskPre[kPostSyn]

            kPostNeuron=postSynNeuronskPre[kPostSyn]
            if (kPreSyn < N_STN):
                if (kPostNeuron < N_STN):
                    transmissionDelays[kPreSyn, kPostSyn]=StepsTauSynDelaySTNSTN
                    #print kPreSyn, kPostSyn, tauSynDelaySTNSTN
                if (kPostNeuron >= N_STN) and (kPostNeuron < N):
                    transmissionDelays[kPreSyn, kPostSyn]=StepsTauSynDelaySTNGPe
                    #print kPreSyn, kPostNeuron, tauSynDelaySTNGPe

            if (kPreSyn >= N_STN) and (kPreSyn < N):
                if (kPostNeuron < N_STN):
                    transmissionDelays[kPreSyn, kPostSyn]=StepsTauSynDelayGPeSTN
                    #print kPreSyn, kPostSyn, tauSynDelayGPeSTN
                if (kPostNeuron >= N_STN) and (kPostNeuron < N):
                    transmissionDelays[kPreSyn, kPostSyn]=StepsTauSynDelayGPeGPe
                    #print kPreSyn, kPostNeuron, tauSynDelayGPeGPe

    # gen numpy array with post synaptic neurons
    for kPostSyn in range(N_STN):
        preSynNeuronskPre=PreSynNeurons[kPostSyn]
        for kPreSyn in range(len(preSynNeuronskPre)):
            numpyPreSynapticNeurons[kPostSyn, kPreSyn]=preSynNeuronskPre[kPreSyn]

            kPreNeuron=preSynNeuronskPre[kPreSyn]
            if (kPostSyn < N_STN):
                if (kPreNeuron < N_STN):
                    transmissionDelaysPreSynNeurons[kPostSyn, kPreSyn]=StepsTauSynDelaySTNSTN

                if (kPreNeuron >= N_STN) and (kPreNeuron < N):
                    transmissionDelaysPreSynNeurons[kPostSyn, kPreSyn]=StepsTauSynDelayGPeSTN

            if (kPostSyn >= N_STN) and (kPostSyn < N):
                if (kPreNeuron < N_STN):
                    transmissionDelaysPreSynNeurons[kPostSyn, kPreSyn]=StepsTauSynDelaySTNGPe

                if (kPreNeuron >= N_STN) and (kPreNeuron < N):
                    transmissionDelaysPreSynNeurons[kPostSyn, kPreSyn]=StepsTauSynDelayGPeGPe

    ###############################################
    # prepare precalculated matrices to speed up STDP 
    # transform weight matrix to sparse matrix to save memory and speed up matrix multiplication
    cMatrix=scipy.sparse.csc_matrix(cMatrix)
    csc_Zero=scipy.sparse.csc_matrix(np.zeros( ( N,N ) ))
    csc_Ones=scipy.sparse.csc_matrix(np.ones( ( N,N ) ))
    # create filter to read out exc/inh neurons
    basicFilterSpikingNeurons=np.zeros( (N,2) )
    # exc coupling
    cMaxExc=system_parameters['cMaxExc']    # maximum range for sync activity
    cExcInit=system_parameters['cExcInit']
    # inh coupling
    cMaxInh=system_parameters['cMaxInh']
    cInhInit=system_parameters['cInhInit']
    # filter already consideres coupling strengths and synaptic timescales according to differential equations
    # excitatory weights
    basicFilterSpikingNeurons[:N_STN,0]=cMaxExc/tauSynExc*np.ones(N_STN)
    # inhibitory weights
    basicFilterSpikingNeurons[N_STN:,1]=cMaxInh/tauSynInh*np.ones(N_GPe)


    # set start time to get computation time after simulation
    start_time = time.time()


    return transmissionDelays, STDPAlreadyOn, tauNoise, Istim, borderForScaningNeuronsPassingSpikingThreshold, borderForScaningForSpikeValues, start_time, csc_Ones, kWeightOutput, transmissionDelaysPreSynNeurons, numpyPostSynapticNeurons, numpyPreSynapticNeurons, maxNumberOfPostSynapticNeurons, maxNumberOfPreSynapticNeurons, basicFilterSpikingNeurons, switchOffSpikeTimes, est_Spike_Times, NeuronIndices, StepsTauSynDelaySTNSTN, StepsTauSynDelayGPeGPe, StepsTauSynDelaySTNGPe, recordedSpikes, kSave, system_parameters, N_STN, N_GPe, N, Vrest, VR, tau, tauVT, VTspike, VTRest, tau_spike, V_spike, Vexc, Vinh, tauSynExc, noiseIntensities, synaptic_offset_after_spike, tauSyn, Vnoise, dt















