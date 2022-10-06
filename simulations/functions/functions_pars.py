##################################
# written by: Justus Kromer
##################################
# written and tested for Python 2.7.13
##################################

##################################
# function: gen_Parameter_set
#
#   Parameter set used for RR simulations
#
#   input:
#       initialSeed ... integer determing initial seed of random number generator
def gen_Parameter_set_Chaos_Paper( initialSeed ):

    ############################################################################
    #   set parameters
    ############################################################################
    # dictionary containing all parameters
    system_parameters={}
    # network
    system_parameters.update({'N_STN': 1000})
    system_parameters.update({'N_GPe': 2})
    # resting potentials-
    system_parameters.update({'VRestSTN': -38.0}) # mV
    system_parameters.update({'VRestGPe': -41.0}) # mV
    # parameter diversity
    system_parameters.update({'sigmaP': 0.05 })
    # exc coupling
    system_parameters.update({'cMaxExc': 400})
    system_parameters.update({'cExcInit': 200.0})
    # inh coupling
    system_parameters.update({'cMaxInh': 0.0})
    system_parameters.update({'cInhInit': 0.0})
    # probs for syn connections
    system_parameters.update({'P_STN_STN': 0.07})
    system_parameters.update({'P_STN_GPe': 0.02})
    system_parameters.update({'P_GPe_GPe': 0.01})
    system_parameters.update({'P_GPe_STN': 0.02})
    # neurons
    system_parameters.update({'Vreset': -67.})  # mV
    # set timescale for STN neurons
    system_parameters.update({'tauSTN': 150})  # ms  
    # and STN neurons
    system_parameters.update({'tauGPe': 30.})  # ms 
    # dynamic threshold
    system_parameters.update({'tauVT': 5.})  # ms 
    system_parameters.update({'VTspike': 0.})  # mV 
    system_parameters.update({'VTRest': -40.})  # mV 
    # shape of spikes
    system_parameters.update({'tau_spike': 1.0 })  # ms
    system_parameters.update({'V_spike': 20. })  # mV 
    # resting potentials
    system_parameters.update({'Vexc': 0. })  # mV 
    system_parameters.update({'Vinh': -80. })  # mV 
    # synaptic time scales
    system_parameters.update({'tauSynExc': 1.0 })  # ms
    system_parameters.update({'tauSynInh': 3.3 })  # ms
    system_parameters.update({'tauNoise': 1.0 })  # ms
    # noise intensity
    system_parameters.update({'noiseSTN': 1.3})
    system_parameters.update({'noiseGPe': 2.0})

    # Input rates according to Ebert Front Comp Neuro 2014
    system_parameters.update({'InputRateGPe': 40.})   # Hz
    system_parameters.update({'InputRateSTN': 20.})   # Hz
    # initialize synaptic transmission delay 
    system_parameters.update({'tauSynDelaySTNSTN': 3.0})  # ms 
    system_parameters.update({'tauSynDelayGPeGPe': 4.0})  # ms 
    system_parameters.update({'tauSynDelaySTNGPe': 6.0})  # ms 
    system_parameters.update({'tauSynDelayGPeSTN': 6.0})  # ms 

    # precalculate weight updates for STDP
    system_parameters.update({'STDP_beta': 1.4})
    system_parameters.update({'STDP_tauPlus': 10.0}) # ms
    system_parameters.update({'STDP_tauRatio': 4.0})

    system_parameters.update({'STDP_weightUpdateRateSTDP': 0.02})
    system_parameters.update({'STDP_tCutoff': 500.0}) #ms

    # integration
    system_parameters.update({'dt': 0.1})  # ms   
    system_parameters.update({'Tinit': 0.0}) # [sec]        # set to zero, otherwise, use ...contRun_ scripts
    system_parameters.update({'Tend': 10000.0}) # [sec]
    system_parameters.update({'Trec': 20.0}) # [sec]        # generate backup every Trec
    system_parameters.update({'initialSeed': initialSeed}) 
    system_parameters.update({'mWeightOutputEverySteps': 500}) # time steps

    # dimensions
    system_parameters.update({'x_STN_min': -2.5 }) # mm
    system_parameters.update({'x_STN_max': 2.5 }) # mm
    
    system_parameters.update({'x_GPe_min': -2.5 }) # mm
    system_parameters.update({'x_GPe_max': 2.5 }) # mm



    return system_parameters


