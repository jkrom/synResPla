import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.sparse
import scipy.interpolate
import os
import pickle
#import pickle5
import matplotlib.colors as mcolors
import sys
#sys.path.append( "/Users/jkromer/Desktop/Projects/Stanford/pythonPackages/python_Scripts/")
import handle_dictionaries
#sys.path.append("/Users/jkromer/Desktop/Projects/Stanford/pythonPackages/python_Scripts/evaluationScripts/")
import spikeTrainAnalysis as sta


def calc_mean_weights( parameters  ):


    # calculate mean weight as function of firing rate
    f_array = np.arange(1.0,20.0,1.0) # Hz
    #f_array = [2.0]
    seed_array = [10,12,14,16,18]  

    # directory in which simulation data are stored
    # directory = '/Users/jkromer/Desktop/Projects/Stanford/scratch/Phase_shifted_periodic_multisite_stimulation/EL_phase_lags_TASS/'
    directory = '/scratch/users/jkromer/Phase_shifted_periodic_multisite_stimulation/EL_phase_lags_TASS/'
    outputdirectory = '/scratch/users/jkromer/Phase_shifted_periodic_multisite_stimulation/mean_weight_as_function_of_frequency'

    results_dic = {}

    # run through all stimulation frequencies
    for fCR in f_array:

        # add corresponding data sets to dictionary
        results_dic[ fCR ] = { 'seeds' : [] , 'mw' : [] }

        # run through different seeds
        for seed in seed_array:
        
            print( 'calc', fCR, seed )

            # calculate mean synaptic weight
            parameters[ 'seed' ] = seed
            parameters[ 'fCR' ] = fCR # Hz

            # evaluation time
            teval = 4000 # sec

            ### load original data and generate backup data
            # filenames for spike train and weight data
            if parameters[ 'ppb' ] == 1:
                filename_spk = directory + 'initial_seed_'+str(parameters[ 'seed' ])+'/electrical_stimulation/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_'+str(parameters[ 'Dalpha1' ])+'_Dalpha2_'+str(parameters[ 'Dalpha2' ])+'/pulses_per_burst_'+str(parameters[ 'ppb' ])+'/fCR_'+str(parameters[ 'fCR' ])+'_M_'+str(parameters[ 'M' ])+'_e_pulse_scale_'+str(parameters[ 'de' ])+'/Astim_'+str(parameters[ 'Astim' ])+'_Tstim_1020.0/spikeTimes_'+str(teval)+'_sec.npy'
                filename_weightMatrix = directory + 'initial_seed_'+str(parameters[ 'seed' ])+'/electrical_stimulation/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_'+str(parameters[ 'Dalpha1' ])+'_Dalpha2_'+str(parameters[ 'Dalpha2' ])+'/pulses_per_burst_'+str(parameters[ 'ppb' ])+'/fCR_'+str(parameters[ 'fCR' ])+'_M_'+str(parameters[ 'M' ])+'_e_pulse_scale_'+str(parameters[ 'de' ])+'/Astim_'+str(parameters[ 'Astim' ])+'_Tstim_1020.0/'+str(teval)+'_sec/cMatrix.npz'
            else:
                filename_spk =          directory + 'initial_seed_'+str(parameters[ 'seed' ])+'/electrical_stimulation/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_'+str(parameters[ 'Dalpha1' ])+'_Dalpha2_'+str(parameters[ 'Dalpha2' ])+'/pulses_per_burst_'+str(parameters[ 'ppb' ])+'/intraburst_frequency_'+str(parameters[ 'f_intra' ])+'_fCR_'+str(parameters[ 'fCR' ])+'_M_'+str(parameters[ 'M' ])+'_e_pulse_scale_'+str(parameters[ 'de' ])+'/Astim_'+str(parameters[ 'Astim' ])+'_Tstim_1020.0/spikeTimes_'+str(teval)+'_sec.npy'
                filename_weightMatrix = directory + 'initial_seed_'+str(parameters[ 'seed' ])+'/electrical_stimulation/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_'+str(parameters[ 'Dalpha1' ])+'_Dalpha2_'+str(parameters[ 'Dalpha2' ])+'/pulses_per_burst_'+str(parameters[ 'ppb' ])+'/intraburst_frequency_'+str(parameters[ 'f_intra' ])+'_fCR_'+str(parameters[ 'fCR' ])+'_M_'+str(parameters[ 'M' ])+'_e_pulse_scale_'+str(parameters[ 'de' ])+'/Astim_'+str(parameters[ 'Astim' ])+'_Tstim_1020.0/'+str(teval)+'_sec/cMatrix.npz'

            # load simulation data
            print(filename_spk)
            if os.path.isfile( filename_spk ):

                # load spike train
                spk_data = np.load( filename_spk )
                print('spike train loaded')

                if os.path.isfile( filename_weightMatrix ):

                    # load synaptic weight matrix
                    wMatrix = scipy.sparse.load_npz( filename_weightMatrix )

                    print('weight matrix loaded')

                    # generate backup filename   
                    # np.savez( filename_dataSet , spk_train = spk_data , wMatrix = wMatrix.A )

                    print('backup file generated')
                    
                    results_dic[ fCR ]['seeds'].append( seed ) 
                    results_dic[ fCR ]['mw'].append( np.mean(wMatrix)/0.07 )

                    print( fCR, seed, np.mean(wMatrix)/0.07  )

            else:
                print('WARNING: file not found')

        results_dic[ fCR ]['mw_mean'] = np.mean(results_dic[ fCR ]['mw'])
        results_dic[ fCR ]['mw_std'] = np.std(results_dic[ fCR ]['mw'])

    if parameters[ 'ppb' ] == 1:
        results_dic['filename'] = 'results_dic_M_'+str(parameters[ 'M' ])+'_de_'+str(parameters[ 'de' ])+'_ppb_'+str(parameters[ 'ppb' ])+'_Astim_'+str(parameters[ 'Astim' ])+'_Dalpha1_'+str(parameters[ 'Dalpha1' ])+'_Dalpha2_'+str(parameters[ 'Dalpha2' ])

    else:
        results_dic['filename'] = 'results_dic_M_'+str(parameters[ 'M' ])+'_de_'+str(parameters[ 'de' ])+'_ppb_'+str(parameters[ 'ppb' ])+'_Astim_'+str(parameters[ 'Astim' ])+'_fintra_'+str(parameters['f_intra'])+'_Dalpha1_'+str(parameters[ 'Dalpha1' ])+'_Dalpha2_'+str(parameters[ 'Dalpha2' ])

    handle_dictionaries.dic_save( results_dic )
    return results_dic



def calc_mean_weights_teval( parameters, teval  ):


    # calculate mean weight as function of firing rate
    f_array = np.arange(1.0,20.0,1.0) # Hz
    #f_array = [2.0]
    seed_array = [10,12,14,16,18]  

    # directory in which simulation data are stored
    # directory = '/Users/jkromer/Desktop/Projects/Stanford/scratch/Phase_shifted_periodic_multisite_stimulation/EL_phase_lags_TASS/'
    directory = '/scratch/users/jkromer/Phase_shifted_periodic_multisite_stimulation/EL_phase_lags_TASS/'
    outputdirectory = '/scratch/users/jkromer/Phase_shifted_periodic_multisite_stimulation/mean_weight_as_function_of_frequency/'

    results_dic = {}

    # run through all stimulation frequencies
    for fCR in f_array:

        # add corresponding data sets to dictionary
        results_dic[ fCR ] = { 'seeds' : [] , 'mw' : [] }

        # run through different seeds
        for seed in seed_array:
        
            print( 'calc', fCR, seed )

            # calculate mean synaptic weight
            parameters[ 'seed' ] = seed
            parameters[ 'fCR' ] = fCR # Hz

            # evaluation time
            # teval = 4000 # sec

            ### load original data and generate backup data
            # filenames for spike train and weight data
            if parameters[ 'ppb' ] == 1:
                filename_spk = directory + 'initial_seed_'+str(parameters[ 'seed' ])+'/electrical_stimulation/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_'+str(parameters[ 'Dalpha1' ])+'_Dalpha2_'+str(parameters[ 'Dalpha2' ])+'/pulses_per_burst_'+str(parameters[ 'ppb' ])+'/fCR_'+str(parameters[ 'fCR' ])+'_M_'+str(parameters[ 'M' ])+'_e_pulse_scale_'+str(parameters[ 'de' ])+'/Astim_'+str(parameters[ 'Astim' ])+'_Tstim_1020.0/spikeTimes_'+str(teval)+'_sec.npy'
                filename_weightMatrix = directory + 'initial_seed_'+str(parameters[ 'seed' ])+'/electrical_stimulation/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_'+str(parameters[ 'Dalpha1' ])+'_Dalpha2_'+str(parameters[ 'Dalpha2' ])+'/pulses_per_burst_'+str(parameters[ 'ppb' ])+'/fCR_'+str(parameters[ 'fCR' ])+'_M_'+str(parameters[ 'M' ])+'_e_pulse_scale_'+str(parameters[ 'de' ])+'/Astim_'+str(parameters[ 'Astim' ])+'_Tstim_1020.0/'+str(teval)+'_sec/cMatrix.npz'
            else:
                filename_spk =          directory + 'initial_seed_'+str(parameters[ 'seed' ])+'/electrical_stimulation/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_'+str(parameters[ 'Dalpha1' ])+'_Dalpha2_'+str(parameters[ 'Dalpha2' ])+'/pulses_per_burst_'+str(parameters[ 'ppb' ])+'/intraburst_frequency_'+str(parameters[ 'f_intra' ])+'_fCR_'+str(parameters[ 'fCR' ])+'_M_'+str(parameters[ 'M' ])+'_e_pulse_scale_'+str(parameters[ 'de' ])+'/Astim_'+str(parameters[ 'Astim' ])+'_Tstim_1020.0/spikeTimes_'+str(teval)+'_sec.npy'
                filename_weightMatrix = directory + 'initial_seed_'+str(parameters[ 'seed' ])+'/electrical_stimulation/multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/Dalpha1_'+str(parameters[ 'Dalpha1' ])+'_Dalpha2_'+str(parameters[ 'Dalpha2' ])+'/pulses_per_burst_'+str(parameters[ 'ppb' ])+'/intraburst_frequency_'+str(parameters[ 'f_intra' ])+'_fCR_'+str(parameters[ 'fCR' ])+'_M_'+str(parameters[ 'M' ])+'_e_pulse_scale_'+str(parameters[ 'de' ])+'/Astim_'+str(parameters[ 'Astim' ])+'_Tstim_1020.0/'+str(teval)+'_sec/cMatrix.npz'

            # load simulation data
            print(filename_spk)
            if os.path.isfile( filename_spk ):

                # load spike train
                spk_data = np.load( filename_spk )
                print('spike train loaded')

                if os.path.isfile( filename_weightMatrix ):

                    # load synaptic weight matrix
                    wMatrix = scipy.sparse.load_npz( filename_weightMatrix )

                    print('weight matrix loaded')

                    # generate backup filename   
                    # np.savez( filename_dataSet , spk_train = spk_data , wMatrix = wMatrix.A )

                    print('backup file generated')
                    
                    results_dic[ fCR ]['seeds'].append( seed ) 
                    results_dic[ fCR ]['mw'].append( np.mean(wMatrix)/0.07 )

                    print( fCR, seed, np.mean(wMatrix)/0.07  )

            else:
                print('WARNING: file not found')

        results_dic[ fCR ]['mw_mean'] = np.mean(results_dic[ fCR ]['mw'])
        results_dic[ fCR ]['mw_std'] = np.std(results_dic[ fCR ]['mw'])

    if parameters[ 'ppb' ] == 1:
        results_dic['filename'] = outputdirectory+'results_teval_'+str(teval)+'_dic_M_'+str(parameters[ 'M' ])+'_de_'+str(parameters[ 'de' ])+'_ppb_'+str(parameters[ 'ppb' ])+'_Astim_'+str(parameters[ 'Astim' ])+'_Dalpha1_'+str(parameters[ 'Dalpha1' ])+'_Dalpha2_'+str(parameters[ 'Dalpha2' ])

    else:
        results_dic['filename'] = outputdirectory+'results_'+str(teval)+'_dic_M_'+str(parameters[ 'M' ])+'_de_'+str(parameters[ 'de' ])+'_ppb_'+str(parameters[ 'ppb' ])+'_Astim_'+str(parameters[ 'Astim' ])+'_fintra_'+str(parameters['f_intra'])+'_Dalpha1_'+str(parameters[ 'Dalpha1' ])+'_Dalpha2_'+str(parameters[ 'Dalpha2' ])

    print('WRITE:',results_dic['filename'] )
    handle_dictionaries.dic_save( results_dic )
    #exit()
    return results_dic



def calc_mean_weights_CRRVS_teval( parameters, teval  ):

    # calculate mean weight as function of firing rate
    f_array = np.arange(1.0,20.0,1.0) # Hz
    #f_array = [2.0]
    seed_array = [10,12,14,16,18]  

    # directory in which simulation data are stored
    # directory = '/Users/jkromer/Desktop/Projects/Stanford/scratch/Phase_shifted_periodic_multisite_stimulation/EL_phase_lags_TASS/'
    directory = '/scratch/users/jkromer/Phase_shifted_periodic_multisite_stimulation/EL_phase_lags_TASS/'
    outputdirectory = '/scratch/users/jkromer/Phase_shifted_periodic_multisite_stimulation/mean_weight_as_function_of_frequency/'

    results_dic = {}

    # run through all stimulation frequencies
    for fCR in f_array:

        # add corresponding data sets to dictionary
        results_dic[ fCR ] = { 'seeds' : [] , 'mw' : [] }

        # run through different seeds
        for seed in seed_array:
        
            print( 'calc', fCR, seed )

            # calculate mean synaptic weight
            parameters[ 'seed' ] = seed
            parameters[ 'fCR' ] = fCR # Hz

            # evaluation time
            # teval = 4000 # sec

            ### load original data and generate backup data
            # filenames for spike train and weight data
            filename_spk =          directory + 'initial_seed_'+str(parameters[ 'seed' ])+'/electrical_stimulation/multiple_spikes_RVS_CR_TASS2012/pulses_per_burst_'+str(parameters[ 'ppb' ])+'/intraburst_frequency_'+str(parameters[ 'f_intra' ])+'_fCR_'+str(parameters[ 'fCR' ])+'_M_'+str(parameters[ 'M' ])+'_e_pulse_scale_'+str(parameters[ 'de' ])+'/Astim_'+str(parameters[ 'Astim' ])+'_Tstim_1020.0/spikeTimes_'+str(teval)+'_sec.npy'
            filename_weightMatrix = directory + 'initial_seed_'+str(parameters[ 'seed' ])+'/electrical_stimulation/multiple_spikes_RVS_CR_TASS2012/pulses_per_burst_'+str(parameters[ 'ppb' ])+'/intraburst_frequency_'+str(parameters[ 'f_intra' ])+'_fCR_'+str(parameters[ 'fCR' ])+'_M_'+str(parameters[ 'M' ])+'_e_pulse_scale_'+str(parameters[ 'de' ])+'/Astim_'+str(parameters[ 'Astim' ])+'_Tstim_1020.0/'+str(teval)+'_sec/cMatrix.npz'

            # load simulation data
            print(filename_spk)
            if os.path.isfile( filename_spk ):

                # load spike train
                spk_data = np.load( filename_spk )
                print('spike train loaded')

                if os.path.isfile( filename_weightMatrix ):

                    # load synaptic weight matrix
                    wMatrix = scipy.sparse.load_npz( filename_weightMatrix )

                    print('weight matrix loaded')

                    # generate backup filename   
                    # np.savez( filename_dataSet , spk_train = spk_data , wMatrix = wMatrix.A )

                    print('backup file generated')
                    
                    results_dic[ fCR ]['seeds'].append( seed ) 
                    results_dic[ fCR ]['mw'].append( np.mean(wMatrix)/0.07 )

                    print( fCR, seed, np.mean(wMatrix)/0.07  )

            else:
                print('WARNING: file not found')

        results_dic[ fCR ]['mw_mean'] = np.mean(results_dic[ fCR ]['mw'])
        results_dic[ fCR ]['mw_std'] = np.std(results_dic[ fCR ]['mw'])

    # filename
    results_dic['filename'] = outputdirectory+'results_CRRVS_teval_'+str(teval)+'_dic_M_'+str(parameters[ 'M' ])+'_de_'+str(parameters[ 'de' ])+'_ppb_'+str(parameters[ 'ppb' ])+'_Astim_'+str(parameters[ 'Astim' ])+'_fintra_'+str(parameters[ 'f_intra' ])

    print('WRITE:',results_dic['filename'] )
    handle_dictionaries.dic_save( results_dic )
    #exit()
    return results_dic



if __name__ == "__main__":

    if sys.argv[1] == 'PMCS':
        teval = int( sys.argv[2] )    

        parameters = {}

        # fixed parameters
        parameters[ 'M' ] = 3
        parameters[ 'de' ] = 1.0
        

        pattern_array = ['CR', 'anti_phase', 'periodic']
        fintra_array = [60.0, 120.0]
        ppb_array = [1,2,3,4,5]

        for pattern in pattern_array:
            for ppb in ppb_array:

                if pattern == 'CR':
                    parameters[ 'Dalpha1' ] = 0.33
                    parameters[ 'Dalpha2' ] = 0.33
                elif pattern =='anti_phase':
                    parameters[ 'Dalpha1' ] = 0.5
                    parameters[ 'Dalpha2' ] = 0.5
                elif pattern =='periodic':
                    parameters[ 'Dalpha1' ] = 0.0
                    parameters[ 'Dalpha2' ] = 0.0

                # fix parameters
                parameters[ 'ppb' ] = ppb
                if ppb > 1:
                    # burst stimulation
                    for fintra in fintra_array:
                        parameters[ 'Astim' ] = 0.8
                        parameters['f_intra'] = fintra # Hz
                        calc_mean_weights_teval( parameters, teval  )
                else:
                    # single-pulse stimulation
                    parameters[ 'Astim' ] = 0.4
                    calc_mean_weights_teval( parameters, teval  )
                
    if sys.argv[1] == 'CRRVS':  

        teval = int( sys.argv[2] ) # sec

        parameters = {}

        # fixed parameters
        parameters[ 'M' ] = 3
        parameters[ 'de' ] = 1.0
        parameters[ 'Astim' ] = 0.8
        
        fintra_array = [60.0, 120.0] # Hz
        ppb_array = [1,2,3,4]

        for ppb in ppb_array:

            # burst stimulation
            for fintra in fintra_array:
                parameters[ 'ppb' ] = ppb
                parameters['f_intra'] = fintra # Hz
                calc_mean_weights_CRRVS_teval( parameters, teval  )




