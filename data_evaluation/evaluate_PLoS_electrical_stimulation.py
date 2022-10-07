import numpy as np
import matplotlib.pyplot as plt
import sys
# sys.path.append( "/Users/jkromer/Desktop/Projects/Stanford/pythonPackages/python_Scripts/evaluationScripts" )
import spikeTrainAnalysis as sta
from scipy.interpolate import griddata
import scipy.interpolate
import os
import matplotlib.gridspec as gridspec

# define function that calculates the stimulation sequence for electrical stimulation
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
            
    return stimulus_onset_times    

### do evaluation for stimulation with a single pulse and different pulse widths.
if sys.argv[1] == 'single_spike':
    # set stimulation frequency and number of pulses per bursts
    fCR = float(sys.argv[2]) # Hz
    pulses_perBurst = [1]

    # calculate stimulation sequence
    stimulus_onset_time_steps = get_stimulus_onset_electrical( 1200001 , fCR , 0.1, 3 )


    Astim_array = np.round( np.arange(0,1,0.05) , 2)
    e_pulse_scale_array = np.arange( 1.0, 20.5, 1.0 )

    # seeds for network generation
    seed_array = [10, 12, 14, 16, 18]

    #seed_array=[int(sys.argv[1])]
    #seed_array=[10]
    # AuC_array = [7500.0]
    # burst_duration_array = [10.0]
    Nstim = 333

    maxNparSets = len(Astim_array)*len(e_pulse_scale_array)*len(seed_array)
    counter = 0
    
    for seed in seed_array: 
        for PpB in pulses_perBurst:
            result_array = []
            for e_pulse_scale in e_pulse_scale_array:
                for Astim in Astim_array:
                    print(counter,maxNparSets)
                    counter+=1
                    # filenames
                    # directory = "/scratch/users/jkromer/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_"+str(seed)+"/electrical_stimulation/phase_shifted_periodic_multisite_stimulation/Dalpha1_0.0_Dalpha2_0.0/fCR_7.0_M_3_e_pulse_scale_"+str(e_pulse_scale)+"/Astim_"+str(Astim)+"_Tstim_100.0"
                    # directory = "/Users/jkromer/Desktop/Projects/Stanford/scratch/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_"+str(seed)+"/electrical_stimulation/phase_shifted_periodic_multisite_stimulation/Dalpha1_0.0_Dalpha2_0.0/fCR_7.0_M_3_e_pulse_scale_"+str(e_pulse_scale)+"/Astim_"+str(Astim)+"_Tstim_100.0"
                    # directory = "/scratch/users/jkromer/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_"+str(seed)+"/electrical_stimulation/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_0.0_Dalpha2_0.0/pulses_per_burst_"+str(PpB)+"/fCR_"+str(fCR)+"_M_3_e_pulse_scale_"+str(e_pulse_scale)+"/Astim_"+str(Astim)+"_Tstim_100.0"
                    directory =             "/Users/jkromer/Desktop/Projects/Stanford/scratch/Output/distribution_of_spike_response_times/initial_seed_"+str(seed)+"/electrical_stimulation/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_0.0_Dalpha2_0.0/pulses_per_burst_"+str(PpB)+"/fCR_"+str(fCR)+"_M_3_e_pulse_scale_"+str(e_pulse_scale)+"/Astim_"+str(Astim)+"_Tstim_100.0/"



                    if os.path.isfile( directory+"/spikeTimes_3100_sec.npy" ):
                        print('load data')
                        # load data
                        spike_train, weights = sta.load_Complete_SpikeTrain_And_Weigth_Trajectories([directory])
                        sequence = stimulus_onset_time_steps # time steps

                        # create stimulus logged spike histograms
                        stimulus_logged_spiketrains = {}

                        startTime_stimulation_steps  = 30000000
                        binsize=2 # steps

                        for k_stimOnsetTime in range(len(sequence)-1):


                            stimOnsetTime = sequence[k_stimOnsetTime]+startTime_stimulation_steps
                            next_stimOnsetTime = sequence[k_stimOnsetTime+1]+startTime_stimulation_steps

                            if spike_train[-1,1] > next_stimOnsetTime:

                                #print next_stimOnsetTime-stimOnsetTime

                                # get corresponding part of spike train
                                splitted_spikeTrain = spike_train[ np.logical_and( spike_train[:,1]>=stimOnsetTime , spike_train[:,1]<=next_stimOnsetTime )]
                                # get corresponding part of mean weights
                                splitted_weights = weights[ np.logical_and( weights[:,0]>=stimOnsetTime , weights[:,0]<=next_stimOnsetTime )]

                                # calc stimulus-triggered histogram of spiking responses
                                # 1) stimulated subpopulations (neurons 0-299)
                                neuron_ind = np.arange(Nstim)
                                # leave only neurons that receive stimuli
                                splitted_spikeTrain_stim = splitted_spikeTrain[ np.isin(splitted_spikeTrain[:,0], neuron_ind, assume_unique=False ) ]


                                bins = np.arange( stimOnsetTime , next_stimOnsetTime, binsize )
                                counts, bins = np.histogram( splitted_spikeTrain_stim , bins , density = False )

                                #print k_stimOnsetTime

                                # get shifted bin centers
                                bin_centers_stimulus_triggered = 0.5*( bins[1:] + bins[:-1] ) - stimOnsetTime


                                stimulus_logged_spiketrains[k_stimOnsetTime] = {'onset_steps':stimOnsetTime, 'end_steps':next_stimOnsetTime, 'spikeTrain_steps':splitted_spikeTrain, 'spikeTrain_stim_steps':splitted_spikeTrain_stim, 'mw_steps':splitted_weights , 'hist':{} }
                                stimulus_logged_spiketrains[k_stimOnsetTime]['hist']['counts'] = counts
                                stimulus_logged_spiketrains[k_stimOnsetTime]['hist']['bins_steps'] = bins
                                stimulus_logged_spiketrains[k_stimOnsetTime]['hist']['bincenter_shifted_steps'] = bin_centers_stimulus_triggered
                                stimulus_logged_spiketrains[k_stimOnsetTime]['hist']['bincenter_shifted_ms'] = bin_centers_stimulus_triggered*sta.dt



                        # get averaged histogram
                        counts = []
                        nbins = 0

                        for k in stimulus_logged_spiketrains:

                            if k == 0:
                                x_ms = stimulus_logged_spiketrains[k]['hist']['bincenter_shifted_ms']
                                nbins = len(x_ms)
                            else:
                                if len(stimulus_logged_spiketrains[k]['hist']['counts']/float(Nstim)) != nbins:

                                    nbins = min( nbins , len(stimulus_logged_spiketrains[k]['hist']['counts']/float(Nstim)) )
                                    x_ms = x_ms[:nbins]

                        counts_array = np.zeros( (len(stimulus_logged_spiketrains),nbins) )
                        #print stimulus_logged_spiketrains
                        for k in stimulus_logged_spiketrains:
                            counts_array[k] = (stimulus_logged_spiketrains[k]['hist']['counts'][:nbins]/float(Nstim))

                        # calculate cumulative distribution of spike times
                        mean_counts = np.mean( counts_array , axis = 0 )

                        # cumulative
                        cumultative = np.zeros( len(mean_counts) )
                        cumultative[0]=mean_counts[0]

                        for k in range(1,len(cumultative)):
                            cumultative[k]= cumultative[k-1]+mean_counts[k]

                        # number of spikes per stimulus
                        pulseDuration_steps = 2+4*e_pulse_scale+2+8*e_pulse_scale
                        nSpikes_per_stimulus = np.sum( mean_counts[ x_ms<= pulseDuration_steps ] )
                        nSpikes_per_period   = np.sum( cumultative[-1] )

                        print([  Astim , e_pulse_scale , PpB, nSpikes_per_stimulus, nSpikes_per_period ])
                        result_array.append( [  Astim , e_pulse_scale , PpB , nSpikes_per_stimulus, nSpikes_per_period ] )

            result_array = np.array( result_array )

            np.save( 'fCR_'+str(fCR)+'_PpB_'+str(PpB)+'_backup_electrical_Stimulation_result_array_seed_'+str(seed)+'.npy', result_array )




### do evaluation for multiple electrical pulses per bursts
if sys.argv[1] == 'multiple_spikes 120 Hz':

    # set stimulation frequency and number of pulses per bursts
    fCR = float(sys.argv[2]) # Hz
    pulses_perBurst = np.arange(1,23)

    # calculate stimulation sequence
    stimulus_onset_time_steps = get_stimulus_onset_electrical( 1200001 , fCR , 0.1, 3 )


    Astim_array = np.round( np.arange(0,1,0.05) , 2)
    e_pulse_scale_array = [1.0]

    # seeds for network generation
    seed_array = [10, 12, 14, 16, 18]

    #seed_array=[int(sys.argv[1])]
    #seed_array=[10]
    # AuC_array = [7500.0]
    # burst_duration_array = [10.0]
    Nstim = 333
    for seed in seed_array:
        for e_pulse_scale in e_pulse_scale_array:
            result_array = []
            for PpB in pulses_perBurst:
                for Astim in Astim_array:

                    # filenames
                    # directory = "/scratch/users/jkromer/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_"+str(seed)+"/electrical_stimulation/phase_shifted_periodic_multisite_stimulation/Dalpha1_0.0_Dalpha2_0.0/fCR_7.0_M_3_e_pulse_scale_"+str(e_pulse_scale)+"/Astim_"+str(Astim)+"_Tstim_100.0"
                    # directory = "/Users/jkromer/Desktop/Projects/Stanford/scratch/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_"+str(seed)+"/electrical_stimulation/phase_shifted_periodic_multisite_stimulation/Dalpha1_0.0_Dalpha2_0.0/fCR_7.0_M_3_e_pulse_scale_"+str(e_pulse_scale)+"/Astim_"+str(Astim)+"_Tstim_100.0"
                    #directory = "/scratch/users/jkromer/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_"+str(seed)+"/electrical_stimulation/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_0.0_Dalpha2_0.0/pulses_per_burst_"+str(PpB)+"/fCR_"+str(fCR)+"_M_3_e_pulse_scale_"+str(e_pulse_scale)+"/Astim_"+str(Astim)+"_Tstim_100.0"
                    directory = "/scratch/users/jkromer/Phase_shifted_periodic_multisite_stimulation/distribution of spike response times/initial_seed_"+str(seed)+"/PLOS_electrical_stimulation_intra/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_0.0_Dalpha2_0.0/pulses_per_burst_"+str(PpB)+"/intraburst_frequency_120.0_fCR_"+str(fCR)+"_M_3_e_pulse_scale_"+str(e_pulse_scale)+"/Astim_"+str(Astim)+"_Tstim_100.0"

                    if os.path.isfile( directory+"/spikeTimes_3100_sec.npy" ):
                        print('load data')
                        # load data
                        spike_train, weights = sta.load_Complete_SpikeTrain_And_Weigth_Trajectories([directory])
                        sequence = stimulus_onset_time_steps # time steps

                        # create stimulus logged spike histograms
                        stimulus_logged_spiketrains = {}

                        startTime_stimulation_steps  = 30000000
                        binsize=2 # steps

                        for k_stimOnsetTime in range(len(sequence)-1):


                            stimOnsetTime = sequence[k_stimOnsetTime]+startTime_stimulation_steps
                            next_stimOnsetTime = sequence[k_stimOnsetTime+1]+startTime_stimulation_steps

                            if spike_train[-1,1] > next_stimOnsetTime:

                                #print next_stimOnsetTime-stimOnsetTime

                                # get corresponding part of spike train
                                splitted_spikeTrain = spike_train[ np.logical_and( spike_train[:,1]>=stimOnsetTime , spike_train[:,1]<=next_stimOnsetTime )]
                                # get corresponding part of mean weights
                                splitted_weights = weights[ np.logical_and( weights[:,0]>=stimOnsetTime , weights[:,0]<=next_stimOnsetTime )]

                                # calc stimulus-triggered histogram of spiking responses
                                # 1) stimulated subpopulations (neurons 0-299)
                                neuron_ind = np.arange(Nstim)
                                # leave only neurons that receive stimuli
                                splitted_spikeTrain_stim = splitted_spikeTrain[ np.isin(splitted_spikeTrain[:,0], neuron_ind, assume_unique=False ) ]


                                bins = np.arange( stimOnsetTime , next_stimOnsetTime, binsize )
                                counts, bins = np.histogram( splitted_spikeTrain_stim , bins , density = False )

                                #print k_stimOnsetTime

                                # get shifted bin centers
                                bin_centers_stimulus_triggered = 0.5*( bins[1:] + bins[:-1] ) - stimOnsetTime


                                stimulus_logged_spiketrains[k_stimOnsetTime] = {'onset_steps':stimOnsetTime, 'end_steps':next_stimOnsetTime, 'spikeTrain_steps':splitted_spikeTrain, 'spikeTrain_stim_steps':splitted_spikeTrain_stim, 'mw_steps':splitted_weights , 'hist':{} }
                                stimulus_logged_spiketrains[k_stimOnsetTime]['hist']['counts'] = counts
                                stimulus_logged_spiketrains[k_stimOnsetTime]['hist']['bins_steps'] = bins
                                stimulus_logged_spiketrains[k_stimOnsetTime]['hist']['bincenter_shifted_steps'] = bin_centers_stimulus_triggered
                                stimulus_logged_spiketrains[k_stimOnsetTime]['hist']['bincenter_shifted_ms'] = bin_centers_stimulus_triggered*sta.dt



                        # get averaged histogram
                        counts = []
                        nbins = 0

                        for k in stimulus_logged_spiketrains:

                            if k == 0:
                                x_ms = stimulus_logged_spiketrains[k]['hist']['bincenter_shifted_ms']
                                nbins = len(x_ms)
                            else:
                                if len(stimulus_logged_spiketrains[k]['hist']['counts']/float(Nstim)) != nbins:

                                    nbins = min( nbins , len(stimulus_logged_spiketrains[k]['hist']['counts']/float(Nstim)) )
                                    x_ms = x_ms[:nbins]

                        counts_array = np.zeros( (len(stimulus_logged_spiketrains),nbins) )
                        #print stimulus_logged_spiketrains
                        for k in stimulus_logged_spiketrains:
                            counts_array[k] = (stimulus_logged_spiketrains[k]['hist']['counts'][:nbins]/float(Nstim))

                        # calculate cumulative distribution of spike times
                        mean_counts = np.mean( counts_array , axis = 0 )

                        # cumulative
                        cumultative = np.zeros( len(mean_counts) )
                        cumultative[0]=mean_counts[0]

                        for k in range(1,len(cumultative)):
                            cumultative[k]= cumultative[k-1]+mean_counts[k]

                        # number of spikes per stimulus
                        pulseDuration_steps = 2+4*e_pulse_scale+2+30*e_pulse_scale
                        nSpikes_per_stimulus = np.sum( mean_counts[ x_ms<= pulseDuration_steps ] )
                        nSpikes_per_period   = np.sum( cumultative[-1] )

                        print([  Astim , e_pulse_scale , PpB, nSpikes_per_stimulus, nSpikes_per_period ])
                        result_array.append( [  Astim , e_pulse_scale, PpB , nSpikes_per_stimulus, nSpikes_per_period ] )

            result_array = np.array( result_array )

            np.save( '120_Hz_fCR_'+str(fCR)+'_e_pulse_scale_'+str(e_pulse_scale)+'_backup_electrical_Stimulation_result_array_seed_'+str(seed)+'.npy', result_array )



