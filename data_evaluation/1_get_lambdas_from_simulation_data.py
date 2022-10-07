import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.sparse
import scipy.interpolate
import os
import pickle
import matplotlib.colors as mcolors
import sys

import spikeTrainAnalysis as sta
import handle_dictionaries



# input:
#   spike_trains        ... list of spike trains of neuronal responses  
#                               kth entry is spike train for kth run
#                               each spike train is of the following format 
#                               (first column is index of neuron, second column is spike time in integration time steps)
#   stimulus_onsets     ... list of corresponding stimulus onset times   
#                               one-dimensional array of stimulus onset times in time steps
#   responding_neurons  ... list of lists of indices of neurons to be considered from each spike train
#   B_max               ... maximal number of spikes per burst to be considered
#   filename            ... filename of dictionary in which results are saved
def get_lambda_from_spike_trains(  spike_trains, stimulus_onsets, responding_neurons, B_max, filename ):

    # number of time steps that are considered as one bin
    binsize=2 # steps
    # print(stimulus_onsets)
    T_steps = stimulus_onsets[0][1] - stimulus_onsets[0][0] # time steps
    #bins = np.arange( (T_steps/float(binsize) )+1 ) # time steps
    bins = np.arange( 0, T_steps+binsize , binsize)

    # print(bins, T_steps)
    # exit()

    if len(spike_trains) == len(stimulus_onsets) and len(stimulus_onsets) == len(responding_neurons):

        #####################################
        # initialize estimated probability distributions
        #### calc conditional probabilities
        # initialize lambdas
        # conditional probability to get kth ISI of lengths eps_k given that kth spike occured at x_k
        # integral for fixed x_k is probability that there is a k+1 th if the kth spike occurs at x_k
        est_lambda = [ np.zeros( len(bins) )]
        # distribution of x_k (integral is est. probabilty to find at least k spikes in neuronal response to stimulus)
        est_Lambda = [ np.zeros( len(bins) )]
        # counts total number of events. This is used for the correct normalization later on.
        counts_given_x_k = [0]

        # initialize all distributions
        kSpikeInBurst = 1

        while kSpikeInBurst < B_max:
            # lambda(eps|x)
            # first entry is eps, second entry is x
            est_lambda.append( np.zeros( ( len(bins),len(bins) ) ) ) 
            # Lambda(x)
            est_Lambda.append( np.zeros( len(bins) ) )
            # counts(x)
            counts_given_x_k.append( np.zeros( len(bins) ) )
            kSpikeInBurst+=1

        # probability to have exactly K spikes in stimulus
        # k = 0 , 1 ,..., B_max
        PK = np.arange( B_max+1 )

        # distribution of last spike in stimulus
        est_Lambda_last = np.zeros( len(bins) )
        # counts how often neurons do not fire spikes in response to stimulus
        N_no_spikes = 0    

        # run through all spike trains considered for data evalation
        for k_spike_train in range(len(spike_trains)):

            print( 'analyzing spiketrain ', k_spike_train )

            #####################################
            # get input for current trial
            spike_train = np.copy( spike_trains[k_spike_train] )
            sequence = np.copy( stimulus_onsets[k_spike_train] )
            neuron_ind = responding_neurons[k_spike_train]

            ########################################################
            # run through stimulus onsets and get estimates
            for k_stimOnsetTime in range(len(sequence)-1):

                # get stimulus onset times
                stimOnsetTime = sequence[k_stimOnsetTime]       # time steps
                next_stimOnsetTime = sequence[k_stimOnsetTime+1] # time steps

                # get part of spike train that corresponds to current stimulus
                splitted_spikeTrain = spike_train[ np.logical_and( spike_train[:,1]>=stimOnsetTime , spike_train[:,1]<=next_stimOnsetTime )]

                # check whether there are any spikes
                if len(splitted_spikeTrain) != 0:

                    # leave only neurons that receive stimuli
                    splitted_spikeTrain_stim = splitted_spikeTrain[ np.isin(splitted_spikeTrain[:,0], neuron_ind, assume_unique=False ) ]


                    # run through all neurons for current stimulus
                    for kNeuron in neuron_ind:

                        # consider only spikes of neuron 'kNeuron'
                        Neurons_SpikeTrain = splitted_spikeTrain_stim[ splitted_spikeTrain_stim[:,0]== kNeuron ]
                        
                        # the neurons fires at least one spike in response to the stimulus
                        if len(Neurons_SpikeTrain) !=0:
                    
                            # get number of spikes for 'kNeuron' during current cycle 
                            #list_of_nspikes_per_stimulus[kNeuron] = len(Neurons_SpikeTrain)

                            if len(Neurons_SpikeTrain) < B_max:
                                # add to estimate of PK
                                PK[len(Neurons_SpikeTrain)]+=1 
                            else:
                                # choose larger B_max
                                print('ERROR: B_max is chosen to small. Some stimuli cause at least '+str(len(Neurons_SpikeTrain))+' spikes.')

                            # consider first spike
                            kSpike = 0

                            # get epsilon and x values and consider them in estimated distributions
                            epsilon_1 = Neurons_SpikeTrain[0,1] - stimOnsetTime # steps
                            x_1 = epsilon_1 # steps

                            # consider estimate of lambda_1
                            est_lambda[kSpike][ int( epsilon_1/float(binsize) ) ]+=1
                            # consider estimate in Lambda_1
                            est_Lambda[kSpike][ int( epsilon_1/float(binsize) ) ]+=1
                            # consider this spike for normalization
                            counts_given_x_k[kSpike] += 1

                            # prepare run over spikes
                            x_k=x_1

                            # run over all spikes in response of current neuron to current stimulus
                            # starts with second spike and first interspike interval
                            for kSpike in range( 1, len(Neurons_SpikeTrain) ):

                                # get spike time of current spike
                                x_kp1 = Neurons_SpikeTrain[ kSpike , 1 ] - stimOnsetTime # time steps
                                # calculate epsilon
                                epsilon_kp1 = x_kp1  - x_k # time steps

                                # consider in distribution of next epsilon
                                est_lambda[ kSpike ][ int( epsilon_kp1/float(binsize) ), int( x_k/float(binsize) ) ]+=1
                                # consider in distribution of k+1 th spike
                                est_Lambda[ kSpike ][ int( x_kp1/float(binsize) ) ]+=1     
                                # update counts 
                                # here the case that the kth spike occurs at x_k and the k+1 th spike occurs at x_k+1 is considered
                                counts_given_x_k[ kSpike ] [ int( x_k/float(binsize) ) ] += 1

                                # get next spike time relative to stimulus onset
                                x_k=x_kp1 # steps

                            # update counts
                            # here the case that the kth spike occurs at x_k and no k+1 th spike occurs is considered
                            counts_given_x_k[ kSpike+1 ] [ int( x_k/float(binsize) ) ] += 1

                            # get distribution of last spike
                            est_Lambda_last[int( x_k/float(binsize) )]+=1 
                            #print(Neurons_SpikeTrain[:,1]- stimOnsetTime, 'last spike ', x_k, kSpike+1 )
                        else:
                            # there was no spike of kNeuron 
                            N_no_spikes+=1


        # normalize results
        # total number of trials is number of neurons times number of stimuli
        total_number_of_trials  = N_no_spikes + counts_given_x_k[0]
        est_lambda[ 0 ] = est_lambda[ 0 ]/float(total_number_of_trials)
        est_Lambda[ 0 ] = est_Lambda[ 0 ]/float(total_number_of_trials)
        PK = PK / float( total_number_of_trials )

        # normalize distribution of last spike
        est_Lambda_last = est_Lambda_last/float(np.sum(est_Lambda_last))

        # initialize all distributions
        kSpikeInBurst = 1

        while kSpikeInBurst < B_max:

            # print('##########')
            # print( est_lambda[ kSpikeInBurst ] )

            # normalize lambda(eps|x)
            for k_x in range( len( bins ) ):
                if float( counts_given_x_k[ kSpikeInBurst ] [ k_x ] ) != 0:
                    est_lambda[ kSpikeInBurst ][:,k_x] = est_lambda[ kSpikeInBurst ][:,k_x]/float( counts_given_x_k[ kSpikeInBurst ] [ k_x ] )
                else:
                    est_lambda[ kSpikeInBurst ][:,k_x] = 0.0*est_lambda[ kSpikeInBurst ][:,k_x]

            # print( est_lambda[ kSpikeInBurst ] )

            # normalize Lamba(x)
            est_Lambda[ kSpikeInBurst] = est_Lambda[ kSpikeInBurst ]/float(total_number_of_trials)

            # .append( np.zeros( ( len(bins),len(bins) ) ) ) 

            # counts_given_x_k
            # # Lambda(x)
            # est_Lambda.append( np.zeros( len(bins) ) )
            # # counts(x)
            # counts_given_x_k.append( np.zeros( len(bins) ) )
            kSpikeInBurst+=1        

        # iteration over spike trains done
        dic_results = { 'PK':PK, 'est_Lambda_last prob':est_Lambda_last, 'N_no_spikes':N_no_spikes, 'bins':bins, 'bins ms': 0.1*bins, 'B_max':B_max }

        dic_results['est_lambda prob'] = {}
        dic_results['est_Lambda prob'] = {}
        dic_results['counts_given_x_k'] = {}

        for kSpike in range(B_max):
            print( np.array( est_lambda[ kSpike ] ).shape )
            dic_results['est_lambda prob'][kSpike]  = np.copy( np.array( est_lambda[ kSpike ] ) )
            dic_results['est_Lambda prob'][kSpike]  = np.copy( np.array( est_Lambda[ kSpike ] ) )
            dic_results['counts_given_x_k'][kSpike] = np.copy( np.array( counts_given_x_k[ kSpike ] ) )




        dic_results['filename'] = filename

        handle_dictionaries.dic_save( dic_results )      
     
        return  dic_results              

    else:
        print('ERROR: input lists must have same first dimensions.')
        return 0


# returns stimulus onset times for electrical stimuli in times steps
def get_stimulus_onset_electrical( maxNtimestpes , fCR_Hz , dt, M ):
    
    time_steps_per_cycle_period = int(1000./(fCR_Hz * dt))
    
    timeStepsToNextStimulusOnset = int( float( time_steps_per_cycle_period )/float(M) )
    stimOnsets = (time_steps_per_cycle_period*0)
    
    # initialize start of next cycle
    startNextCycle = 0
    stimulus_onset_times = [startNextCycle]
    
    for timeStep in range(maxNtimestpes):
        
        if timeStep == startNextCycle:
            
            startNextCycle += time_steps_per_cycle_period
            
            stimulus_onset_times.append( startNextCycle )
            
    return np.array( stimulus_onset_times ) # time steps





if __name__ == "__main__":

    output_directory = '/scratch/users/jkromer/Phase_shifted_periodic_multisite_stimulation/theory/PLoS/data_lambda_burst/'
    output_directory = ''


    ########################################################################
    ############ PLoS electrical stimulation get dependence on pulse width
    ########################################################################
    # used for Figure 1
    if sys.argv[1] == 'electrical PLoS pulse width': 
        # load spike train
        Astim = 0.4
        e_pulse_scale = float( sys.argv[2] )
        pulses_per_burst = 1 # pulses per burst

        seed_array = [10,12,14,16,18]

        spike_trains = []
        stimulus_onsets = []
        responding_neurons = []

        B_max = 6
        startTime_stimulation_steps  = 30000000

        # calculate stimulus onset times
        fCR = float( sys.argv[3] ) # Hz
        stimulus_onset_time_steps = get_stimulus_onset_electrical( 1200001 , fCR , 0.1, 3 )

        print("################################")
        print(e_pulse_scale, fCR)

        for seed in seed_array:

            print('loading data for seed', seed)

            #directory = '/Users/jkromer/Desktop/Projects/Stanford/scratch/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_'+str(seed)+'/electrical_stimulation/phase_shifted_periodic_multisite_stimulation/Dalpha1_0.0_Dalpha2_0.0/fCR_7.0_M_3_e_pulse_scale_'+str(e_pulse_scale)+'/Astim_'+str(Astim)+'_Tstim_100.0'
            #directory = '/Users/jkromer/Desktop/Projects/Stanford/scratch/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_'+str(seed)+'/electrical_stimulation/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_0.0_Dalpha2_0.0/pulses_per_burst_1/fCR_'+str(fCR)+'_M_3_e_pulse_scale_'+str(e_pulse_scale)+'/Astim_'+str(Astim)+'_Tstim_100.0'
            directory = '/Users/jkromer/Desktop/Projects/Stanford/scratch/Output/distribution_of_spike_response_times/initial_seed_'+str(seed)+'/electrical_stimulation/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_0.0_Dalpha2_0.0/pulses_per_burst_'+str(pulses_per_burst)+'/fCR_'+str(fCR)+'_M_3_e_pulse_scale_'+str(e_pulse_scale)+'/Astim_'+str(Astim)+'_Tstim_100.0'

            #print(directory+'/meanWeightTimeSeries_3020_sec.npy')

            if os.path.isfile( directory+'/meanWeightTimeSeries_3020_sec.npy' ):
                spike_train, weights = sta.load_Complete_SpikeTrain_And_Weigth_Trajectories([directory])
                sequence = stimulus_onset_time_steps + startTime_stimulation_steps  
                neuron_ind = np.arange(333)

                print( len(spike_train), ' spikes found')
                spike_trains.append( spike_train )
                stimulus_onsets.append( sequence ) # time steps
                responding_neurons.append( neuron_ind )

        # filename of dictionary in which data are saved
        # filename = output_directory+'dic_lambda_normalized_PLoS_electrical_burst_fCR_'+str(fCR)+'_Astim_'+str(Astim)+'_de_'+str(e_pulse_scale)+'_ppb_'+str(pulses_per_burst)
        filename = 'dic_lambda_normalized_PLoS_electrical_burst_fCR_'+str(fCR)+'_Astim_'+str(Astim)+'_de_'+str(e_pulse_scale)+'_ppb_'+str(pulses_per_burst)

        # do evaluation
        print('estimating distributions ...')

        dic = get_lambda_from_spike_trains(  spike_trains, stimulus_onsets, responding_neurons, B_max, filename )




    # used for Figure 4
    if sys.argv[1] == 'multiple pulses and intraburst frequency PLoS_5Hz': 

        parameter_combinations = {}
        parameter_combinations['1']={'Astim':0.8, 'de':1.0, 'PpB':3, 'fintra_Hz':120.0, 'fCR_Hz':5.0}
        parameter_combinations['2']={'Astim':0.8, 'de':1.0, 'PpB':5, 'fintra_Hz':120.0, 'fCR_Hz':5.0}
        parameter_combinations['3']={'Astim':0.8, 'de':1.0, 'PpB':8, 'fintra_Hz':120.0, 'fCR_Hz':5.0}
        parameter_combinations['4']={'Astim':0.8, 'de':1.0, 'PpB':3, 'fintra_Hz':60.0, 'fCR_Hz':5.0}
        parameter_combinations['5']={'Astim':0.8, 'de':1.0, 'PpB':5, 'fintra_Hz':60.0, 'fCR_Hz':5.0}
        parameter_combinations['6']={'Astim':0.8, 'de':1.0, 'PpB':8, 'fintra_Hz':60.0, 'fCR_Hz':5.0}

        seed_array = [10,12,14,16,18]

        spike_trains = []
        stimulus_onsets = []
        responding_neurons = []

        
        startTime_stimulation_steps  = 30000000

        # select which parameter set is analyzed
        key = sys.argv[2]

        parameterset = parameter_combinations[key]
        fCR = parameterset['fCR_Hz']
        fintra = parameterset['fintra_Hz']
        Astim = parameterset['Astim']
        e_pulse_scale = parameterset['de']
        pulses_per_burst = parameterset['PpB']

        # calculate stimulus onset times
        stimulus_onset_time_steps = get_stimulus_onset_electrical( 1200001 , fCR , 0.1, 3 )


        B_max = pulses_per_burst + 2
        print('considering data set',parameterset)

        for seed in seed_array:

            print('loading data for seed', seed)

            #directory = '/Users/jkromer/Desktop/Projects/Stanford/scratch/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_'+str(seed)+'/electrical_stimulation/phase_shifted_periodic_multisite_stimulation/Dalpha1_0.0_Dalpha2_0.0/fCR_7.0_M_3_e_pulse_scale_'+str(e_pulse_scale)+'/Astim_'+str(Astim)+'_Tstim_100.0'
            #directory = '/Users/jkromer/Desktop/Projects/Stanford/scratch/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_'+str(seed)+'/electrical_stimulation/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_0.0_Dalpha2_0.0/pulses_per_burst_'+str(pulses_per_burst)+'/fCR_'+str(fCR)+'_M_3_e_pulse_scale_'+str(e_pulse_scale)+'/Astim_'+str(Astim)+'_Tstim_100.0'
            directory = '/Users/jkromer/Desktop/Projects/Stanford/scratch/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_'+str(seed)+'/electrical_stimulation_intra/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_0.0_Dalpha2_0.0/pulses_per_burst_'+str(pulses_per_burst)+'/intraburst_frequency_'+str(fintra)+'_fCR_'+str(fCR)+'_M_3_e_pulse_scale_'+str(e_pulse_scale)+'/Astim_'+str(Astim)+'_Tstim_100.0'
            #print(directory+'/meanWeightTimeSeries_3020_sec.npy')

            if os.path.isfile( directory+'/meanWeightTimeSeries_3020_sec.npy' ):
                spike_train, weights = sta.load_Complete_SpikeTrain_And_Weigth_Trajectories([directory])
                sequence = stimulus_onset_time_steps + startTime_stimulation_steps  
                neuron_ind = np.arange(333)

                print( len(spike_train), ' spikes found')
                spike_trains.append( spike_train )
                stimulus_onsets.append( sequence ) # time steps
                responding_neurons.append( neuron_ind )

        # filename of dictionary in which data are saved
        filename = output_directory+'dic_lambda_normalized_PLoS_electrical_burst_fCR_'+str(fCR)+'_fintra_'+str(fintra)+'_Astim_'+str(Astim)+'_de_'+str(e_pulse_scale)+'_ppb_'+str(pulses_per_burst)

        # do evaluation
        print('estimating distributions ...')
        dic = get_lambda_from_spike_trains(  spike_trains, stimulus_onsets, responding_neurons, B_max, filename )


    # used for Figure 4
    if sys.argv[1] == 'multiple pulses and intraburst frequency PLoS_10Hz': 

        parameter_combinations = {}
        parameter_combinations['1']={'Astim':0.8, 'de':1.0, 'PpB':3, 'fintra_Hz':120.0, 'fCR_Hz':10.0}
        parameter_combinations['2']={'Astim':0.8, 'de':1.0, 'PpB':5, 'fintra_Hz':120.0, 'fCR_Hz':10.0}
        parameter_combinations['3']={'Astim':0.8, 'de':1.0, 'PpB':8, 'fintra_Hz':120.0, 'fCR_Hz':10.0}
        parameter_combinations['4']={'Astim':0.8, 'de':1.0, 'PpB':3, 'fintra_Hz':60.0, 'fCR_Hz':10.0}
        parameter_combinations['5']={'Astim':0.8, 'de':1.0, 'PpB':5, 'fintra_Hz':60.0, 'fCR_Hz':10.0}
        parameter_combinations['6']={'Astim':0.8, 'de':1.0, 'PpB':8, 'fintra_Hz':60.0, 'fCR_Hz':10.0}

        seed_array = [10,12,14,16,18]

        spike_trains = []
        stimulus_onsets = []
        responding_neurons = []

        
        startTime_stimulation_steps  = 30000000

        # select which parameter set is analyzed
        key = sys.argv[2]

        parameterset = parameter_combinations[key]
        fCR = parameterset['fCR_Hz']
        fintra = parameterset['fintra_Hz']
        Astim = parameterset['Astim']
        e_pulse_scale = parameterset['de']
        pulses_per_burst = parameterset['PpB']

        # calculate stimulus onset times
        stimulus_onset_time_steps = get_stimulus_onset_electrical( 1200001 , fCR , 0.1, 3 )


        B_max = pulses_per_burst + 2
        print('considering data set',parameterset)

        for seed in seed_array:

            print('loading data for seed', seed)

            #directory = '/Users/jkromer/Desktop/Projects/Stanford/scratch/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_'+str(seed)+'/electrical_stimulation/phase_shifted_periodic_multisite_stimulation/Dalpha1_0.0_Dalpha2_0.0/fCR_7.0_M_3_e_pulse_scale_'+str(e_pulse_scale)+'/Astim_'+str(Astim)+'_Tstim_100.0'
            #directory = '/Users/jkromer/Desktop/Projects/Stanford/scratch/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_'+str(seed)+'/electrical_stimulation/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_0.0_Dalpha2_0.0/pulses_per_burst_'+str(pulses_per_burst)+'/fCR_'+str(fCR)+'_M_3_e_pulse_scale_'+str(e_pulse_scale)+'/Astim_'+str(Astim)+'_Tstim_100.0'
            directory = '/Users/jkromer/Desktop/Projects/Stanford/scratch/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_'+str(seed)+'/electrical_stimulation_intra/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_0.0_Dalpha2_0.0/pulses_per_burst_'+str(pulses_per_burst)+'/intraburst_frequency_'+str(fintra)+'_fCR_'+str(fCR)+'_M_3_e_pulse_scale_'+str(e_pulse_scale)+'/Astim_'+str(Astim)+'_Tstim_100.0'
            #print(directory+'/meanWeightTimeSeries_3020_sec.npy')

            if os.path.isfile( directory+'/meanWeightTimeSeries_3020_sec.npy' ):
                spike_train, weights = sta.load_Complete_SpikeTrain_And_Weigth_Trajectories([directory])
                sequence = stimulus_onset_time_steps + startTime_stimulation_steps  
                neuron_ind = np.arange(333)

                print( len(spike_train), ' spikes found')
                spike_trains.append( spike_train )
                stimulus_onsets.append( sequence ) # time steps
                responding_neurons.append( neuron_ind )

        # filename of dictionary in which data are saved
        filename = output_directory+'dic_lambda_normalized_PLoS_electrical_burst_fCR_'+str(fCR)+'_fintra_'+str(fintra)+'_Astim_'+str(Astim)+'_de_'+str(e_pulse_scale)+'_ppb_'+str(pulses_per_burst)

        # do evaluation
        print('estimating distributions ...')
        dic = get_lambda_from_spike_trains(  spike_trains, stimulus_onsets, responding_neurons, B_max, filename )


    # used for Figure 4
    if sys.argv[1] == 'multiple pulses and intraburst frequency PLoS_2.5Hz': 

        parameter_combinations = {}
        parameter_combinations['1']={'Astim':0.8, 'de':1.0, 'PpB':3, 'fintra_Hz':120.0, 'fCR_Hz':2.5}
        parameter_combinations['2']={'Astim':0.8, 'de':1.0, 'PpB':5, 'fintra_Hz':120.0, 'fCR_Hz':2.5}
        parameter_combinations['3']={'Astim':0.8, 'de':1.0, 'PpB':8, 'fintra_Hz':120.0, 'fCR_Hz':2.5}
        parameter_combinations['4']={'Astim':0.8, 'de':1.0, 'PpB':3, 'fintra_Hz':60.0, 'fCR_Hz':2.5}
        parameter_combinations['5']={'Astim':0.8, 'de':1.0, 'PpB':5, 'fintra_Hz':60.0, 'fCR_Hz':2.5}
        parameter_combinations['6']={'Astim':0.8, 'de':1.0, 'PpB':8, 'fintra_Hz':60.0, 'fCR_Hz':2.5}

        seed_array = [10,12,14,16,18]

        spike_trains = []
        stimulus_onsets = []
        responding_neurons = []

        
        startTime_stimulation_steps  = 30000000

        # select which parameter set is analyzed
        key = sys.argv[2]

        parameterset = parameter_combinations[key]
        fCR = parameterset['fCR_Hz']
        fintra = parameterset['fintra_Hz']
        Astim = parameterset['Astim']
        e_pulse_scale = parameterset['de']
        pulses_per_burst = parameterset['PpB']

        # calculate stimulus onset times
        stimulus_onset_time_steps = get_stimulus_onset_electrical( 1200001 , fCR , 0.1, 3 )


        B_max = pulses_per_burst + 3
        print('considering data set',parameterset)

        for seed in seed_array:

            print('loading data for seed', seed)

            #directory = '/Users/jkromer/Desktop/Projects/Stanford/scratch/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_'+str(seed)+'/electrical_stimulation/phase_shifted_periodic_multisite_stimulation/Dalpha1_0.0_Dalpha2_0.0/fCR_7.0_M_3_e_pulse_scale_'+str(e_pulse_scale)+'/Astim_'+str(Astim)+'_Tstim_100.0'
            #directory = '/Users/jkromer/Desktop/Projects/Stanford/scratch/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_'+str(seed)+'/electrical_stimulation/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_0.0_Dalpha2_0.0/pulses_per_burst_'+str(pulses_per_burst)+'/fCR_'+str(fCR)+'_M_3_e_pulse_scale_'+str(e_pulse_scale)+'/Astim_'+str(Astim)+'_Tstim_100.0'
            directory = '/scratch/users/jkromer/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_'+str(seed)+'/electrical_stimulation_intra/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_0.0_Dalpha2_0.0/pulses_per_burst_'+str(pulses_per_burst)+'/intraburst_frequency_'+str(fintra)+'_fCR_'+str(fCR)+'_M_3_e_pulse_scale_'+str(e_pulse_scale)+'/Astim_'+str(Astim)+'_Tstim_100.0'
            #print(directory+'/meanWeightTimeSeries_3020_sec.npy')

            if os.path.isfile( directory+'/meanWeightTimeSeries_3020_sec.npy' ):
                spike_train, weights = sta.load_Complete_SpikeTrain_And_Weigth_Trajectories([directory])
                sequence = stimulus_onset_time_steps + startTime_stimulation_steps  
                neuron_ind = np.arange(333)

                print( len(spike_train), ' spikes found')
                spike_trains.append( spike_train )
                stimulus_onsets.append( sequence ) # time steps
                responding_neurons.append( neuron_ind )

        # filename of dictionary in which data are saved
        filename = output_directory+'dic_lambda_normalized_PLoS_electrical_burst_fCR_'+str(fCR)+'_fintra_'+str(fintra)+'_Astim_'+str(Astim)+'_de_'+str(e_pulse_scale)+'_ppb_'+str(pulses_per_burst)


        # do evaluation
        print('estimating distributions ...')
        dic = get_lambda_from_spike_trains(  spike_trains, stimulus_onsets, responding_neurons, B_max, filename )


    # used for Figure 8
    ########################################################################
    ############ single-pulse stimuli as function of stimulation frequency
    ########################################################################
    if sys.argv[1] == 'single-pulse as function of frequency': 

        seed_array = [10,12,14,16,18]

        parameterset = {}

        # frequency values
        f_array = np.round( np.arange( 2.0, 20.0, 0.25 ) , 2 )

        for f in f_array:

            parameterset={'Astim':0.4, 'de':1.0, 'PpB':1, 'fCR_Hz':f }

            fCR = parameterset['fCR_Hz']
            # fintra = parameterset['fintra_Hz']
            Astim = parameterset['Astim']
            e_pulse_scale = parameterset['de']
            pulses_per_burst = parameterset['PpB']


            spike_trains = []
            stimulus_onsets = []
            responding_neurons = []

        
            startTime_stimulation_steps  = 30000000

            # calculate stimulus onset times
            stimulus_onset_time_steps = get_stimulus_onset_electrical( 1200001 , fCR , 0.1, 3 )

            #print(stimulus_onset_time_steps)

            # exit()

            B_max = pulses_per_burst + 3
            print('considering data set',parameterset)

            for seed in seed_array:

                print('loading data for seed', seed)

                #directory = '/Users/jkromer/Desktop/Projects/Stanford/scratch/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_'+str(seed)+'/electrical_stimulation/phase_shifted_periodic_multisite_stimulation/Dalpha1_0.0_Dalpha2_0.0/fCR_7.0_M_3_e_pulse_scale_'+str(e_pulse_scale)+'/Astim_'+str(Astim)+'_Tstim_100.0'
                #directory = '/Users/jkromer/Desktop/Projects/Stanford/scratch/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_'+str(seed)+'/electrical_stimulation/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_0.0_Dalpha2_0.0/pulses_per_burst_'+str(pulses_per_burst)+'/fCR_'+str(fCR)+'_M_3_e_pulse_scale_'+str(e_pulse_scale)+'/Astim_'+str(Astim)+'_Tstim_100.0'
                #directory = '/scratch/users/jkromer/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_'+str(seed)+'/PLOS_electrical_stimulation_intra/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_0.0_Dalpha2_0.0/pulses_per_burst_'+str(pulses_per_burst)+'/intraburst_frequency_'+str(fintra)+'_fCR_'+str(fCR)+'_M_3_e_pulse_scale_'+str(e_pulse_scale)+'/Astim_'+str(Astim)+'_Tstim_100.0'
                directory = '/scratch/users/jkromer/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_'+str(seed)+'/electrical_stimulation/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_0.0_Dalpha2_0.0/pulses_per_burst_'+str(pulses_per_burst)+'/fCR_'+str(fCR)+'_M_3_e_pulse_scale_'+str(e_pulse_scale)+'/Astim_'+str(Astim)+'_Tstim_100.0'

                #print(directory+'/meanWeightTimeSeries_3020_sec.npy')

                # print(directory)
                # exit()

                if os.path.isfile( directory+'/meanWeightTimeSeries_3020_sec.npy' ):
                    spike_train, weights = sta.load_Complete_SpikeTrain_And_Weigth_Trajectories([directory])
                    sequence = stimulus_onset_time_steps + startTime_stimulation_steps  
                    neuron_ind = np.arange(333)

                    print( len(spike_train), ' spikes found')
                    spike_trains.append( spike_train )
                    stimulus_onsets.append( sequence ) # time steps
                    responding_neurons.append( neuron_ind )


            # filename of dictionary in which data are saved
            filename = output_directory+'dic_lambda_normalized_PLoS_electrical_burst_fCR_'+str(fCR)+'_Astim_'+str(Astim)+'_de_'+str(e_pulse_scale)+'_ppb_'+str(pulses_per_burst)

            # do evaluation
            print('estimating distributions ...')
            dic = get_lambda_from_spike_trains(  spike_trains, stimulus_onsets, responding_neurons, B_max, filename )




    ########################################################################
    ############ burst stimuli as function of stimulation frequency (fintra = 120 Hz)
    ########################################################################
    if sys.argv[1] == 'multiple pulses as function of frequency 120 Hz intraburst frequency': 

        seed_array = [10,12,14,16,18]

        parameterset = {}

        # frequency values
        f_array = np.round( np.arange( 2.0, 20.0, 0.25 ) , 2 )
        f_array = f_array[ f_array > float(sys.argv[2]) ]

        for f in f_array:
            for PpB in [2,3,4]:

                parameterset={'Astim':0.8, 'de':1.0, 'PpB':PpB, 'fintra_Hz':120.0, 'fCR_Hz':f }

                fCR = parameterset['fCR_Hz']
                fintra = parameterset['fintra_Hz']
                Astim = parameterset['Astim']
                e_pulse_scale = parameterset['de']
                pulses_per_burst = parameterset['PpB']


                spike_trains = []
                stimulus_onsets = []
                responding_neurons = []

            
                startTime_stimulation_steps  = 30000000

                # calculate stimulus onset times
                stimulus_onset_time_steps = get_stimulus_onset_electrical( 1200001 , fCR , 0.1, 3 )

                #print(stimulus_onset_time_steps)

                # exit()

                B_max = pulses_per_burst + 3
                print('considering data set',parameterset)

                for seed in seed_array:

                    print('loading data for seed', seed)

                    #directory = '/Users/jkromer/Desktop/Projects/Stanford/scratch/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_'+str(seed)+'/electrical_stimulation/phase_shifted_periodic_multisite_stimulation/Dalpha1_0.0_Dalpha2_0.0/fCR_7.0_M_3_e_pulse_scale_'+str(e_pulse_scale)+'/Astim_'+str(Astim)+'_Tstim_100.0'
                    #directory = '/Users/jkromer/Desktop/Projects/Stanford/scratch/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_'+str(seed)+'/electrical_stimulation/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_0.0_Dalpha2_0.0/pulses_per_burst_'+str(pulses_per_burst)+'/fCR_'+str(fCR)+'_M_3_e_pulse_scale_'+str(e_pulse_scale)+'/Astim_'+str(Astim)+'_Tstim_100.0'
                    directory = '/scratch/users/jkromer/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_'+str(seed)+'/PLOS_electrical_stimulation_intra/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_0.0_Dalpha2_0.0/pulses_per_burst_'+str(pulses_per_burst)+'/intraburst_frequency_'+str(fintra)+'_fCR_'+str(fCR)+'_M_3_e_pulse_scale_'+str(e_pulse_scale)+'/Astim_'+str(Astim)+'_Tstim_100.0'
                    #print(directory+'/meanWeightTimeSeries_3020_sec.npy')

                    # print(directory)
                    # exit()

                    if os.path.isfile( directory+'/meanWeightTimeSeries_3020_sec.npy' ):
                        spike_train, weights = sta.load_Complete_SpikeTrain_And_Weigth_Trajectories([directory])
                        sequence = stimulus_onset_time_steps + startTime_stimulation_steps  
                        neuron_ind = np.arange(333)

                        print( len(spike_train), ' spikes found')
                        spike_trains.append( spike_train )
                        stimulus_onsets.append( sequence ) # time steps
                        responding_neurons.append( neuron_ind )


                # filename of dictionary in which data are saved
                filename = output_directory+'dic_lambda_normalized_PLoS_electrical_burst_fCR_'+str(fCR)+'_fintra_'+str(fintra)+'_Astim_'+str(Astim)+'_de_'+str(e_pulse_scale)+'_ppb_'+str(pulses_per_burst)

                # do evaluation
                print('estimating distributions ...')
                dic = get_lambda_from_spike_trains(  spike_trains, stimulus_onsets, responding_neurons, B_max, filename )


    ########################################################################
    ############ burst stimuli as function of stimulation frequency (fintra = 60 Hz)
    ########################################################################
    if sys.argv[1] == 'multiple pulses as function of frequency 60 Hz intraburst frequency': 

        seed_array = [10,12,14,16,18]

        parameterset = {}

        # frequency values
        f_array = np.round( np.arange( 2.0, 20.0, 0.25 ) , 2 )

        for f in f_array:
            for PpB in [2,3,4]:

                parameterset={'Astim':0.8, 'de':1.0, 'PpB':PpB, 'fintra_Hz':60.0, 'fCR_Hz':f }

                fCR = parameterset['fCR_Hz']
                fintra = parameterset['fintra_Hz']
                Astim = parameterset['Astim']
                e_pulse_scale = parameterset['de']
                pulses_per_burst = parameterset['PpB']


                spike_trains = []
                stimulus_onsets = []
                responding_neurons = []

            
                startTime_stimulation_steps  = 30000000

                # calculate stimulus onset times
                stimulus_onset_time_steps = get_stimulus_onset_electrical( 1200001 , fCR , 0.1, 3 )

                #print(stimulus_onset_time_steps)

                # exit()

                B_max = pulses_per_burst + 3
                print('considering data set',parameterset)

                for seed in seed_array:

                    print('loading data for seed', seed)

                    #directory = '/Users/jkromer/Desktop/Projects/Stanford/scratch/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_'+str(seed)+'/electrical_stimulation/phase_shifted_periodic_multisite_stimulation/Dalpha1_0.0_Dalpha2_0.0/fCR_7.0_M_3_e_pulse_scale_'+str(e_pulse_scale)+'/Astim_'+str(Astim)+'_Tstim_100.0'
                    #directory = '/Users/jkromer/Desktop/Projects/Stanford/scratch/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_'+str(seed)+'/electrical_stimulation/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_0.0_Dalpha2_0.0/pulses_per_burst_'+str(pulses_per_burst)+'/fCR_'+str(fCR)+'_M_3_e_pulse_scale_'+str(e_pulse_scale)+'/Astim_'+str(Astim)+'_Tstim_100.0'
                    directory = '/scratch/users/jkromer/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_'+str(seed)+'/PLOS_electrical_stimulation_intra/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_0.0_Dalpha2_0.0/pulses_per_burst_'+str(pulses_per_burst)+'/intraburst_frequency_'+str(fintra)+'_fCR_'+str(fCR)+'_M_3_e_pulse_scale_'+str(e_pulse_scale)+'/Astim_'+str(Astim)+'_Tstim_100.0'
                    #print(directory+'/meanWeightTimeSeries_3020_sec.npy')

                    # print(directory)
                    # exit()

                    if os.path.isfile( directory+'/meanWeightTimeSeries_3020_sec.npy' ):
                        spike_train, weights = sta.load_Complete_SpikeTrain_And_Weigth_Trajectories([directory])
                        sequence = stimulus_onset_time_steps + startTime_stimulation_steps  
                        neuron_ind = np.arange(333)

                        print( len(spike_train), ' spikes found')
                        spike_trains.append( spike_train )
                        stimulus_onsets.append( sequence ) # time steps
                        responding_neurons.append( neuron_ind )

                # filename of dictionary in which data are saved
                filename = output_directory+'dic_lambda_normalized_PLoS_electrical_burst_fCR_'+str(fCR)+'_fintra_'+str(fintra)+'_Astim_'+str(Astim)+'_de_'+str(e_pulse_scale)+'_ppb_'+str(pulses_per_burst)

                # do evaluation
                print('estimating distributions ...')
                dic = get_lambda_from_spike_trains(  spike_trains, stimulus_onsets, responding_neurons, B_max, filename )








