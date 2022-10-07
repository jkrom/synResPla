import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import scipy.sparse
import scipy.interpolate
import os
import pickle
import pickle5

# import matplotlib.colors as mcolors
import sys

def dic_save( dic ):
	with open(dic['filename'] + '.pickle', 'wb') as f:
		pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)

def dic_load(filename):
	with open(filename + '.pickle', 'rb') as f:
		return pickle.load(f)

def dic_load_pkl(filename):
	with open(filename + '.pkl', 'rb') as f:
		return pickle.load(f)


def dic_load_new(filename):
    with open(filename + '.pickle', 'rb') as f:
        return pickle5.load(f , encoding='bytes' )

# probability densities below 'lambda_threshold' will be considered as zero
lambda_threshold = 0
# maximum array size to be considered during calculation. This sets the upper bound
# for memory consumption
MaxBlockSize = 20000000

###########################
# STDP functions
###########################
beta = 1.4 # ratio between overall long-term depression and overall long-term potentiation
tauPlus = 10.0 # ms   time scale of synaptic potentiation
tauR = 4.0 # ration of time scales for synaptic depression and potentiation
tau1 = tauPlus # time scale for symmetric plasticty

# asymmetric Hebbian
def STDP_asym_Heb(t, dt):
   
	weightChange=np.zeros( len(t) )
   
	weightChange[ t>dt ] = np.exp(-t[ t>dt ]/tauPlus)
	weightChange[ t<-dt ] = -beta/(tauR)*np.exp(t[ t<-dt ]/(tauR*tauPlus) )
   
	return weightChange


# asymmetric anti Hebbian
def STDP_asym_anti_Heb(t, dt):
   
	weightChange=np.zeros( len(t) )
   
	weightChange[ t>dt ] = -beta*np.exp(-t[ t>dt ]/tauPlus)
	weightChange[ t<-dt ] = 1.0/(tauR)*np.exp(t[ t<-dt ]/(tauR*tauPlus) )
   
	return weightChange

# symmetric Hebbian
def STDP_sym_Heb(t, dt):
   
	weightChange=np.zeros( len(t) )
   
	weightChange = ( np.exp( -np.abs(t)/float(tau1) ) - 0.5*np.exp( -np.abs(t)/float( (1+beta)*tau1) )  )
	#weightChange = ( 2*np.exp( -np.abs(t)/float(tau1) ) - np.exp( -np.abs(t)/float( (2.-10*(1-beta)/float(tau1))*tau1) )  )

	return weightChange

# symmetric anti Hebbian
def STDP_sym_anti_Heb(t, dt):
   
	weightChange=np.zeros( len(t) )
   
	weightChange = -( np.exp( -np.abs(t)/float(tau1) ) - 0.5*np.exp( -np.abs(t)/float( (3-beta)*tau1) )  )
	#weightChange = ( -2*np.exp( -np.abs(t)/float(tau1) ) + np.exp( -np.abs(t)/float( (2.-10*(-1+beta)/float(tau1))*tau1) )  )

	return weightChange


## calculates weight updates per cycle for given phi value
# parameters:
# bins_ms 		... bin boundaries in milliseconds used for time descretization for all distributions
# phi 			... 0<=phi<1 current phase lag between stimuli delivered to postsynaptic and presynaptic neuron
# fHz 			... stimulation frequency in Hz
#
# all destributions need to be normalized to give probabilties for bins
# LAMBDA_array 	... precalculated distributions of spike times of the 1st, 2nd, .... , Bmaxth spike
# LambdaLast 	... distribution of the last spike 
# lambda_array 	... distribution of intraburst intervals (epsilon) given left spike time (x) (first index for intraburst interval bin, second index for spike time bin)
# Delta 		... difference between axonal delays and dendritic delays in ms
# B_max 		... maximal number of spikes that can occur during a neuronal response to a single simulus
# Pk_cum 		... cumulative probabilites that a neuronal response can have k bursts (Pk_cum[0] .. prob that only one spike occurs)
def normalized_fast_W_(bins_ms, phi,fHz, LAMBDA_array, LambdaLast, lambda_array, Delta, B_max, Pk_cum):
	
	# maximal distance of cycles between pre and postsynaptic bursts for which spikes 
	# are considered for weight updates
	kmax = 5

	# max number of bins for each x and eps
	ktMax = len(bins_ms)

	# bin size in ms
	bin_size_ms = bins_ms[1]-bins_ms[0] # ms

	# get frequency in 1/ms
	fHz_per_ms = 0.001*fHz
	
	# total number of spikes per burst
	nSpikesPerBurst = B_max
	
	# initialize W_plus and W_minus
	# asymmetric anti Heb
	W_plus_aaH = 0 	
	W_minus_aaH = 0

	# asymmetric Heb
	W_plus_aH = 0
	W_minus_aH = 0

	# symmetric Heb
	W_plus_sH = 0
	W_minus_sH = 0

	# symmetric anti Heb
	W_plus_saH = 0
	W_minus_saH = 0
		
	# get all possible combinations
	x1_values =  np.arange(ktMax)
	x2_values =  np.arange(ktMax)
	xlast_values = np.arange(ktMax)
	eps_values =  np.arange(ktMax)
 
	# consider only last spike time that actually occur
	all_xLast_values_nonzero = xlast_values[ LambdaLast > lambda_threshold ]

	# consider positive updates triggered by mth postsynaptic spike and 
	# negative updates triggered by the mth presynaptic spike
	for m in np.arange( nSpikesPerBurst ):

		Prob_for_mp1_spikes = Pk_cum[m]

		# restrict to timings x2 of mth postsynaptic spike that are actually possible
		x2_values_nonzero = x2_values[ LAMBDA_array[m] > lambda_threshold ]

		# restrict to timings x1 of presynaptic spikes that are possible
		ind_x1_nonzeroPro = LAMBDA_array[-1] > lambda_threshold  	
		for kBurst in np.arange(nSpikesPerBurst-1):
			ind_x1_nonzeroPro += LAMBDA_array[kBurst] > lambda_threshold		
		ind_x1_nonzeroPro = np.arange(ktMax)[ind_x1_nonzeroPro] 
		all_x1_values_nonzero = x1_values[ ind_x1_nonzeroPro ]

		# get maximum value for epsilon
		maxLength_eps_Val = max( np.arange(ktMax)[lambda_array[0] > lambda_threshold] )
		#print(maxLength_eps_Val)
		for kBurst in np.arange(1,nSpikesPerBurst):
			# epsilon_k                                             eps                    x1
			if len( np.nonzero( lambda_array[kBurst] > lambda_threshold  )[0] ) >0:
				new_max = max(np.nonzero( lambda_array[kBurst] > lambda_threshold  )[0])
			else:
				new_max = 0
			maxLength_eps_Val = max([maxLength_eps_Val, new_max])

		#print(maxLength_eps_Val)
		eps_values = eps_values[:maxLength_eps_Val+1]

		# total number of array elements to be evaluated
		TotalNumberOfElements_during = len(all_x1_values_nonzero)*len(x2_values_nonzero)*len(eps_values)
		TotalNumberOfElements_last = len(all_xLast_values_nonzero)*len(x2_values_nonzero)*len(eps_values)


		# print(all_x1_values_nonzero,x2_values_nonzero,eps_values)

		# print("TotalNumberOfElements_during", TotalNumberOfElements_during)
		# print("TotalNumberOfElements_last", TotalNumberOfElements_last)

		##############################################
		### start evaluation for 'during'  (first integral in Eqs. for Wplus and Wminus )
		# split up in blocks of length MaxBlockSize and evaluate integrals for individual blocks
		if (TotalNumberOfElements_during > MaxBlockSize):
			# calculate block size (size of blocks all_x1_values_nonzero is divided into)
			blockSize = int( MaxBlockSize/float(len(x2_values_nonzero)*len(eps_values)) )

			if blockSize < 2:
				print("WARNING: too high resolution, watch memory!")
				blockSize = 2
			#print('split up in blocks of size', blockSize)
		else:
			blockSize = len( all_x1_values_nonzero ) 

		# evaluate for indiviudal blocks
		currentEndOfEvaluation = 0

		while currentEndOfEvaluation < len( all_x1_values_nonzero ):

			#print('during',currentEndOfEvaluation,len( all_x1_values_nonzero ))

			# do everything for current block
			if (currentEndOfEvaluation+blockSize)< len(all_x1_values_nonzero):
				x1_values_nonzero = all_x1_values_nonzero[currentEndOfEvaluation:currentEndOfEvaluation+blockSize]
			else:
				x1_values_nonzero = all_x1_values_nonzero[currentEndOfEvaluation:]

			currentEndOfEvaluation = currentEndOfEvaluation+blockSize

			# construct on long array to speed up later summation
			x1_long = np.repeat(x1_values_nonzero,len(x2_values_nonzero)*len(eps_values))

			x2_long = np.tile(   np.repeat(x2_values_nonzero,len(eps_values) ) , len(x1_values_nonzero))
			eps_long = np.tile(eps_values, len(x2_values_nonzero)*len(x1_values_nonzero))


			# first index is x1, second is x2, third is eps
			x1_x2_eps_long = np.concatenate( ( np.transpose([x1_long]), np.transpose([x2_long]), np.transpose([eps_long]) ) , axis = 1     )

			for k in np.arange(-kmax+1,kmax+1,1):

				# get borders for Wplus
				#		x2min_Wplus = (k-phi)/fHz + Delta + x1
				x2min_Wplus = int( ( (k-phi)/fHz_per_ms + Delta)/bin_size_ms ) + x1_x2_eps_long[:,0]  # bins 
				#		x2max_Wplus = x2min_Wplus + eps
				x2max_Wplus = x2min_Wplus + x1_x2_eps_long[:,2] # bins 
																					# x2 >= x2min_Wplus 		   			x2 < x2max_Wplus																	x2 < x2max
				case2_x1_x2_eps_long_Wplus = x1_x2_eps_long[  np.logical_and( x1_x2_eps_long[:,1]  >= x2min_Wplus ,  x1_x2_eps_long[:,1] <  x2max_Wplus ) ]


				# get borders for Wminus
				#		x2min_Wminus = (k+phi)/fHz - Delta + x1
				x2min_Wminus = int( ( (k+phi)/fHz_per_ms - Delta)/bin_size_ms ) + x1_x2_eps_long[:,0]  # bins 
				#		x2max_Wminus = x2min_Wminus + eps
				x2max_Wminus = x2min_Wminus + x1_x2_eps_long[:,2] # bins 
																					# x2 >= x2min_Wminus 		   			x2 < x2max_Wminus																	x2 < x2max
				case2_x1_x2_eps_long_Wminus = x1_x2_eps_long[  np.logical_and( x1_x2_eps_long[:,1]  >= x2min_Wminus ,  x1_x2_eps_long[:,1] <  x2max_Wminus ) ]


				# run through all possible spikes of pre (Wplus) and postsynaptic (Wminus) bursts
				for n in np.arange(0,nSpikesPerBurst-1):

					x1_plus  = case2_x1_x2_eps_long_Wplus[:,0] # bins
					x2_plus  = case2_x1_x2_eps_long_Wplus[:,1] # bins
					eps_plus = case2_x1_x2_eps_long_Wplus[:,2] # bins

					Dt_plus = bin_size_ms*( x2_plus - x1_plus )- (k-phi)/fHz_per_ms - Delta # ms

					x1_minus  = case2_x1_x2_eps_long_Wminus[:,0] # bins
					x2_minus  = case2_x1_x2_eps_long_Wminus[:,1] # bins
					eps_minus = case2_x1_x2_eps_long_Wminus[:,2] # bins

					Dt_minus = bin_size_ms*( x1_minus - x2_minus )+ (k+phi)/fHz_per_ms - Delta # ms

					# calculate positive updates
					if ( len( case2_x1_x2_eps_long_Wplus ) != 0 ):
						#				  		LAMBDA_n(x1)			lambda_n+1( eps | x1 )			LAMBDA_m( x2 )		    W(  Dt )
						# STDP_asym_anti_Heb(t, dt)
						W_plus_aaH+=Prob_for_mp1_spikes*np.sum( LAMBDA_array[n][  x1_plus   ]*lambda_array[n+1][  eps_plus , x1_plus   ]*LAMBDA_array[m][  x2_plus ] * STDP_asym_anti_Heb( Dt_plus , bin_size_ms ) )
						# STDP_asym_Heb(t, dt)
						W_plus_aH+=Prob_for_mp1_spikes*np.sum( LAMBDA_array[n][  x1_plus   ]*lambda_array[n+1][  eps_plus , x1_plus   ]*LAMBDA_array[m][  x2_plus ] * STDP_asym_Heb( Dt_plus , bin_size_ms ) )
						# STDP_sym_anti_Heb(t, dt)
						W_plus_saH+=Prob_for_mp1_spikes*np.sum( LAMBDA_array[n][  x1_plus   ]*lambda_array[n+1][  eps_plus , x1_plus   ]*LAMBDA_array[m][  x2_plus ] * STDP_sym_anti_Heb( Dt_plus , bin_size_ms ) )
						# STDP_sym_Heb(t, dt)
						W_plus_sH+=Prob_for_mp1_spikes*np.sum( LAMBDA_array[n][  x1_plus   ]*lambda_array[n+1][  eps_plus , x1_plus   ]*LAMBDA_array[m][  x2_plus ] * STDP_sym_Heb( Dt_plus , bin_size_ms ) )

					# calculate negative updates
					if  ( len(case2_x1_x2_eps_long_Wminus) != 0):
						#				  		LAMBDA_n(x1)			lambda_n+1( eps | x1 )			LAMBDA_m( x2 )		    W(  Dt )
						# STDP_asym_anti_Heb(t, dt)
						W_minus_aaH+=Prob_for_mp1_spikes*np.sum( LAMBDA_array[n][  x1_minus  ]*lambda_array[n+1][  eps_minus , x1_minus   ]*LAMBDA_array[m][  x2_minus ] * STDP_asym_anti_Heb( Dt_minus  , bin_size_ms ) )
						# STDP_asym_Heb(t, dt)
						W_minus_aH+=Prob_for_mp1_spikes*np.sum( LAMBDA_array[n][  x1_minus  ]*lambda_array[n+1][  eps_minus , x1_minus   ]*LAMBDA_array[m][  x2_minus ] * STDP_asym_Heb( Dt_minus  , bin_size_ms ) )
						# STDP_sym_anti_Heb(t, dt)
						W_minus_saH+=Prob_for_mp1_spikes*np.sum( LAMBDA_array[n][  x1_minus  ]*lambda_array[n+1][  eps_minus , x1_minus   ]*LAMBDA_array[m][  x2_minus ] * STDP_sym_anti_Heb( Dt_minus  , bin_size_ms ) )
						# STDP_sym_Heb(t, dt)
						W_minus_sH+=Prob_for_mp1_spikes*np.sum( LAMBDA_array[n][  x1_minus  ]*lambda_array[n+1][  eps_minus , x1_minus   ]*LAMBDA_array[m][  x2_minus ] * STDP_sym_Heb( Dt_minus  , bin_size_ms ) )


	
		#print('#### during',W_plus,W_minus)

		##############################################
		### start evaluation for 'last' (second integral in Eqs. for Wplus and Wminus )
		# split up in blocks of length MaxBlockSize and evaluate integrals for individual blocks
		if (TotalNumberOfElements_last > MaxBlockSize):
			# calculate block size (size of blocks all_x1_values_nonzero is divided into)
			blockSize = int( MaxBlockSize/float(len(x2_values_nonzero)*len(eps_values)) )

			if blockSize < 2:
				print("WARNING: too high resolution, watch memory!")
				blockSize = 2
			#print('split up in blocks of size', blockSize)
		else:
			blockSize = len( all_xLast_values_nonzero )

		# evaluate for indiviudal blocks
		currentEndOfEvaluation = 0

		while currentEndOfEvaluation < len( all_xLast_values_nonzero ):

			#print('last',currentEndOfEvaluation,len( all_xLast_values_nonzero ))

			# do everything for current block
			if (currentEndOfEvaluation+blockSize)< len(all_xLast_values_nonzero):
				xLast_values_nonzero = all_xLast_values_nonzero[currentEndOfEvaluation:currentEndOfEvaluation+blockSize]
			else:
				xLast_values_nonzero = all_xLast_values_nonzero[currentEndOfEvaluation:]

			currentEndOfEvaluation = currentEndOfEvaluation+blockSize

			# construct on long array to speed up later summation
			xlast_long = np.repeat(xLast_values_nonzero,len(x2_values_nonzero)*len(eps_values))

			x2_long = np.tile(   np.repeat(x2_values_nonzero,len(eps_values) ) , len(xLast_values_nonzero))
			eps_long = np.tile(eps_values, len(x2_values_nonzero)*len(xLast_values_nonzero))

			# first index is xlast, second is x2, third is eps
			xlast_x2_eps_long = np.concatenate( ( np.transpose([xlast_long]), np.transpose([x2_long]), np.transpose([eps_long]) ) , axis = 1     )

			for k in np.arange(-kmax+1,kmax+1,1):

				# get borders for Wplus
				#		x2min_Wplus = (k-phi)/fHz + Delta + xlast
				x2min_Wplus = int( ( (k-phi)/fHz_per_ms + Delta)/bin_size_ms ) + xlast_x2_eps_long[:,0]  # bins 
				#		x2max_Wplus = (k+1-phi)/fHz + eps + Delta
				x2max_Wplus = int( ( (k+1-phi)/fHz_per_ms + Delta)/bin_size_ms ) + xlast_x2_eps_long[:,2] # bins 
																					# x2 >= x2min_Wplus 		   			x2 < x2max_Wplus																	x2 < x2max
				case2_xlast_x2_eps_long_Wplus = xlast_x2_eps_long[  np.logical_and( xlast_x2_eps_long[:,1]  >= x2min_Wplus ,  xlast_x2_eps_long[:,1] <  x2max_Wplus ) ]


				# get borders for Wminus
				#		x2min_Wminus = (k+phi)/fHz - Delta + x1
				x2min_Wminus = int( ( (k+phi)/fHz_per_ms - Delta)/bin_size_ms ) + xlast_x2_eps_long[:,0]  # bins 
				#		x2max_Wminus = x2min_Wminus + eps
				x2max_Wminus = int( ( (k+1+phi)/fHz_per_ms - Delta)/bin_size_ms ) + xlast_x2_eps_long[:,2] # bins 
																					# x2 >= x2min_Wminus 		   			x2 < x2max_Wminus																	x2 < x2max
				case2_xlast_x2_eps_long_Wminus = xlast_x2_eps_long[  np.logical_and( xlast_x2_eps_long[:,1]  >= x2min_Wminus ,  xlast_x2_eps_long[:,1] <  x2max_Wminus ) ]


				xlast_plus  = case2_xlast_x2_eps_long_Wplus[:,0] # bins
				x2_plus  = case2_xlast_x2_eps_long_Wplus[:,1] # bins
				eps_plus = case2_xlast_x2_eps_long_Wplus[:,2] # bins

				Dt_plus = bin_size_ms*( x2_plus - xlast_plus )- (k-phi)/fHz_per_ms - Delta # ms

				xlast_minus  = case2_xlast_x2_eps_long_Wminus[:,0] # bins
				x2_minus  = case2_xlast_x2_eps_long_Wminus[:,1] # bins
				eps_minus = case2_xlast_x2_eps_long_Wminus[:,2] # bins

				Dt_minus = bin_size_ms*( xlast_minus - x2_minus )+ (k+phi)/fHz_per_ms - Delta # ms

				# calculate positive updates
				if ( len( case2_xlast_x2_eps_long_Wplus ) != 0 ):
					#				  		LAMBDA_n(xlast)			lambda_n+1( eps | xlast )			LAMBDA_m( x2 )		    W(  Dt )
					# STDP_asym_anti_Heb(t, dt)
					W_plus_aaH+=Prob_for_mp1_spikes*np.sum( LambdaLast[  xlast_plus   ]*lambda_array[0][  eps_plus   ]*LAMBDA_array[m][  x2_plus ] * STDP_asym_anti_Heb( Dt_plus , bin_size_ms ) )
					# STDP_asym_Heb(t, dt)
					W_plus_aH+=Prob_for_mp1_spikes*np.sum( LambdaLast[  xlast_plus   ]*lambda_array[0][  eps_plus   ]*LAMBDA_array[m][  x2_plus ] * STDP_asym_Heb( Dt_plus , bin_size_ms ) )
					# STDP_sym_anti_Heb(t, dt)
					W_plus_saH+=Prob_for_mp1_spikes*np.sum( LambdaLast[  xlast_plus   ]*lambda_array[0][  eps_plus   ]*LAMBDA_array[m][  x2_plus ] * STDP_sym_anti_Heb( Dt_plus , bin_size_ms ) )
					# STDP_sym_Heb(t, dt)
					W_plus_sH+=Prob_for_mp1_spikes*np.sum( LambdaLast[  xlast_plus   ]*lambda_array[0][  eps_plus   ]*LAMBDA_array[m][  x2_plus ] * STDP_sym_Heb( Dt_plus , bin_size_ms ) )
		
				# calculate negative updates		
				if  ( len(case2_xlast_x2_eps_long_Wminus) != 0):
					# STDP_asym_anti_Heb(t, dt)
					W_minus_aaH+=Prob_for_mp1_spikes*np.sum( LambdaLast[  xlast_minus  ]*lambda_array[0][  eps_minus   ]*LAMBDA_array[m][  x2_minus ] * STDP_asym_anti_Heb( Dt_minus  , bin_size_ms ) )
					# STDP_asym_Heb(t, dt)
					W_minus_aH+=Prob_for_mp1_spikes*np.sum( LambdaLast[  xlast_minus  ]*lambda_array[0][  eps_minus   ]*LAMBDA_array[m][  x2_minus ] * STDP_asym_Heb( Dt_minus  , bin_size_ms ) )
					# STDP_sym_anti_Heb(t, dt)
					W_minus_saH+=Prob_for_mp1_spikes*np.sum( LambdaLast[  xlast_minus  ]*lambda_array[0][  eps_minus   ]*LAMBDA_array[m][  x2_minus ] * STDP_sym_anti_Heb( Dt_minus  , bin_size_ms ) )
					# STDP_sym_Heb(t, dt)
					W_minus_sH+=Prob_for_mp1_spikes*np.sum( LambdaLast[  xlast_minus  ]*lambda_array[0][  eps_minus   ]*LAMBDA_array[m][  x2_minus ] * STDP_sym_Heb( Dt_minus  , bin_size_ms ) )



	dic_W = {}
	dic_W['aaH'] = {'Wplus':W_plus_aaH, 'Wminus':W_minus_aaH, }
	dic_W['aH'] = {'Wplus':W_plus_aH, 'Wminus':W_minus_aH, }
	dic_W['saH'] = {'Wplus':W_plus_saH, 'Wminus':W_minus_saH, }
	dic_W['sH'] = {'Wplus':W_plus_sH, 'Wminus':W_minus_sH, }

	return dic_W


def combine_kphi_results_J( parameters ):

    directory = '/scratch/users/jkromer/Phase_shifted_periodic_multisite_stimulation/theory/PLoS/results_burst'
    # outputdirectory = '/scratch/users/jkromer/Phase_shifted_periodic_multisite_stimulation/theory/PLoS/results_burst'
    outputdirectory = '/scratch/users/jkromer/Phase_shifted_periodic_multisite_stimulation/theory/PLoS/results_burst_full'
    
    # array of phi values
    phiValues = np.arange(0,1,0.01)


    # initialize result array for different STDP functions
    phi_Wplus_Wminus_aaH = np.zeros( (len(phiValues),3) )
    phi_Wplus_Wminus_aH = np.zeros( (len(phiValues),3) )
    phi_Wplus_Wminus_saH = np.zeros( (len(phiValues),3) )
    phi_Wplus_Wminus_sH = np.zeros( (len(phiValues),3) )


    # start calculation of JofPhi 
    for kphi in range(len(phiValues)):

        if parameters['ppb'] == 1:
            filename = '/normalized_phi_PLoS_single_f_stim_fCR_'+str(parameters['f'])+'_Astim_'+str(parameters['Astim'])+'_de_'+str(parameters['de'])+'_ppb_'+str(parameters['ppb'])+'_kphi_'+str(kphi)+'.npz'
            outputfilename = '/normalized_phi_PLoS_single_fCR_'+str(parameters['f'])+'_Astim_'+str(parameters['Astim'])+'_de_'+str(parameters['de'])+'_ppb_'+str(parameters['ppb'])+'.npz'
        else:
            filename = '/normalized_phi_PLoS_burst_f_stim_fCR_'+str(parameters['f'])+'_fintra_'+str(parameters['fintra'])+'_Astim_'+str(parameters['Astim'])+'_de_'+str(parameters['de'])+'_ppb_'+str(parameters['ppb'])+'_kphi_'+str(kphi)+'.npz'
            outputfilename = '/normalized_phi_PLoS_intra_fCR_'+str(parameters['f'])+'_fintra_'+str(parameters['fintra'])+'_Astim_'+str(parameters['Astim'])+'_de_'+str(parameters['de'])+'_ppb_'+str(parameters['ppb'])+'.npz'

        # load filename
        if os.path.isfile( directory + filename ):
            data_set_J_kphi = np.load( directory + filename )

            phi_Wplus_Wminus_aaH[kphi,0] = data_set_J_kphi['aaH'][kphi,0]
            phi_Wplus_Wminus_aaH[kphi,1] = data_set_J_kphi['aaH'][kphi,1]
            phi_Wplus_Wminus_aaH[kphi,2] = data_set_J_kphi['aaH'][kphi,2]

            phi_Wplus_Wminus_aH[kphi,0] = data_set_J_kphi['aH'][kphi,0]
            phi_Wplus_Wminus_aH[kphi,1] = data_set_J_kphi['aH'][kphi,1]
            phi_Wplus_Wminus_aH[kphi,2] = data_set_J_kphi['aH'][kphi,2]

            phi_Wplus_Wminus_saH[kphi,0] = data_set_J_kphi['saH'][kphi,0]
            phi_Wplus_Wminus_saH[kphi,1] = data_set_J_kphi['saH'][kphi,1]
            phi_Wplus_Wminus_saH[kphi,2] = data_set_J_kphi['saH'][kphi,2]

            phi_Wplus_Wminus_sH[kphi,0] = data_set_J_kphi['sH'][kphi,0]
            phi_Wplus_Wminus_sH[kphi,1] = data_set_J_kphi['sH'][kphi,1]
            phi_Wplus_Wminus_sH[kphi,2] = data_set_J_kphi['sH'][kphi,2]
        else:
            print('ERROR: file not found')
            return 0


    # save results
    np.savez( outputdirectory + outputfilename , phi=phiValues, aaH=phi_Wplus_Wminus_aaH, aH=phi_Wplus_Wminus_aH, saH=phi_Wplus_Wminus_saH, sH=phi_Wplus_Wminus_sH )


# input arguments
# 1 ... specifies stimulus type to be analyzed. This only changes the precalulated lambdas used for the calculation
#		of JofPhi
# 2 ... Delta, difference between axonal and dendritic delays in ms
if __name__ == "__main__":

	mode = 'officeComputer'
	mode = 'sherlock'
	# mode = 'local'

	if mode == 'sherlock':
		# use this directory when running script on sherlock
		# outputdirectory = '/scratch/users/jkromer/Phase_shifted_periodic_multisite_stimulation/theory/PLoS/results_burst/'
		outputdirectory = '/scratch/users/jkromer/Output/data/WofPhi/'
		# directory from which precalculated lambdas are loaded
		#directory_lambda='data/'
		#directory_lambda = '/scratch/users/jkromer/Phase_shifted_periodic_multisite_stimulation/theory/PLoS/data_lambda_burst/'
		directory_lambda = '/scratch/users/jkromer/Output/data/WofPhi/'

	if mode == 'officeComputer':
		# use this directory when running on office computer
		outputdirectory = 'data_WofPhi/'
		# directory from which precalculated lambdas are loaded
		directory_lambda='data_lambda/'

	if mode == 'local':
		# use this directory when running on office computer
		outputdirectory = ''
		# directory from which precalculated lambdas are loaded
		directory_lambda='../figures/Fig6/data/'		

	if outputdirectory != '':
		if not os.path.exists(outputdirectory):
		  
		  # Create a new directory because it does not exist 
		  os.makedirs(outputdirectory)


	# calculate expected mean rate of weight change for estimated distributions 
	# load estimated distributions of spike response times from simulation data
	if sys.argv[1] == 'PLoS_single': 

		parameter_combinations = {}
		parameter_combinations['1']={'Astim':0.4, 'de':1.0, 'fCR_Hz':5.0, 'PpB':1, 'stimulus_type':'single', 'delay': 3.0}
		parameter_combinations['2']={'Astim':0.4, 'de':20.0, 'fCR_Hz':5.0, 'PpB':1, 'stimulus_type':'single', 'delay': 3.0}
		parameter_combinations['3']={'Astim':0.4, 'de':1.0, 'fCR_Hz':2.5, 'PpB':1, 'stimulus_type':'single', 'delay': 3.0}
		parameter_combinations['4']={'Astim':0.4, 'de':20.0, 'fCR_Hz':2.5, 'PpB':1, 'stimulus_type':'single', 'delay': 3.0}
		parameter_combinations['5']={'Astim':0.4, 'de':1.0, 'fCR_Hz':10.0, 'PpB':1, 'stimulus_type':'single', 'delay': 3.0}
		parameter_combinations['6']={'Astim':0.4, 'de':20.0, 'fCR_Hz':10.0, 'PpB':1, 'stimulus_type':'single', 'delay': 3.0}

		parameter_combinations['7']={'Astim':0.4, 'de':40.0, 'fCR_Hz':5.0, 'PpB':1, 'stimulus_type':'single', 'delay': 3.0}
		parameter_combinations['8']={'Astim':0.4, 'de':40.0, 'fCR_Hz':10.0, 'PpB':1, 'stimulus_type':'single', 'delay': 3.0}
		parameter_combinations['9']={'Astim':0.4, 'de':40.0, 'fCR_Hz':2.5, 'PpB':1, 'stimulus_type':'single', 'delay': 3.0}

		key = sys.argv[2]

		parameterset = parameter_combinations[key]

		#filename_est_lambda = directory_lambda+'dic_lambda_normalized_PLoS_electrical_burst_fCR_'+str(parameterset['fCR_Hz'])+'_fintra_'+str(parameterset['fintra_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])
		#filename_est_lambda = directory_lambda+'dic_lambda_normalized_PLoS_electrical_burst_fCR_'+str(parameterset['fCR_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])
		filename_est_lambda  = directory_lambda + 'dic_lambda_normalized_PLoS_electrical_burst_fCR_'+str(parameterset['fCR_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])
		print('####',directory_lambda)
		# stimulation parameters
		fHz = parameterset['fCR_Hz'] # Hz


		dic_lambda = {}

		print(filename_est_lambda)
		# load corresponding distributions from simulation data
		print("loading estimated lambdas from file ...")
		try:
			dic_lambda = dic_load( filename_est_lambda )
		except:
			print('using pickle5')
			dic_lambda = dic_load_new( filename_est_lambda )

		# get time resolutions
		bins_ms = dic_lambda['bins ms'] # ms
		dt_theory = bins_ms[1] - bins_ms[0] # ms

		# maximal number of spikes considered
		B_max = dic_lambda['B_max']

		# load lambda arrays from dictionary
		lambda_array = []
		Lambda_array = []
		for kB in range(B_max):
			lambda_array.append( dic_lambda['est_lambda prob'][kB] )
			Lambda_array.append( dic_lambda['est_Lambda prob'][kB] )
		lambdaLastSpike = dic_lambda['est_Lambda_last prob']

		# get cumulative probability
		PK = dic_lambda['PK']
		Pk_cum = np.zeros( PK.shape )
		Pk_cum[0] = 1.0
		for k in range( 1,len(Pk_cum) ):
			Pk_cum[k] = Pk_cum[k-1]-PK[k]

		### uncommend to loaded plot distributions
		# import matplotlib.pyplot as plt 
		# plt.plot( bins_ms ,Lambda_array[0], label = "Lambda_1" )
		# # plt.plot( bins_ms ,Lambda_array[1], label = "Lambda_2" )
		# # plt.plot( bins_ms ,Lambda_array[2], label = "Lambda_3" )
		# plt.plot( bins_ms ,lambdaLastSpike, label = 'last' )
		# plt.legend()
		# plt.show()
		# exit()

		# axonal minus dendritic delay in ms
		Delta = parameterset['delay']

		# phi values for which results are calculated
		phiValues = np.arange(0,1,0.01)

		# initialize result array for different STDP functions
		phi_Wplus_Wminus_aaH = np.zeros( (len(phiValues),3) )
		phi_Wplus_Wminus_aH = np.zeros( (len(phiValues),3) )
		phi_Wplus_Wminus_saH = np.zeros( (len(phiValues),3) )
		phi_Wplus_Wminus_sH = np.zeros( (len(phiValues),3) )


		# start calculation of JofPhi 
		#for kphi in range(len(phiValues)):
		for kphi in [int(sys.argv[3])]:
		#for kphi in np.arange(100):

			print( kphi , '/' , len(phiValues) )

			phi = phiValues[kphi]
			dic_W = normalized_fast_W_(bins_ms, phi,fHz, Lambda_array, lambdaLastSpike, lambda_array, Delta, B_max, Pk_cum)

			phi_Wplus_Wminus_aaH[kphi,0] = phi
			phi_Wplus_Wminus_aaH[kphi,1] = dic_W['aaH']['Wplus']
			phi_Wplus_Wminus_aaH[kphi,2] = dic_W['aaH']['Wminus']

			phi_Wplus_Wminus_aH[kphi,0] = phi
			phi_Wplus_Wminus_aH[kphi,1] = dic_W['aH']['Wplus']
			phi_Wplus_Wminus_aH[kphi,2] = dic_W['aH']['Wminus']

			phi_Wplus_Wminus_saH[kphi,0] = phi
			phi_Wplus_Wminus_saH[kphi,1] = dic_W['saH']['Wplus']
			phi_Wplus_Wminus_saH[kphi,2] = dic_W['saH']['Wminus']

			phi_Wplus_Wminus_sH[kphi,0] = phi
			phi_Wplus_Wminus_sH[kphi,1] = dic_W['sH']['Wplus']
			phi_Wplus_Wminus_sH[kphi,2] = dic_W['sH']['Wminus']

		# save results
		# np.savez( outputdirectory+'PLoS_normalized_phi_'+str(sys.argv[1])+'_fCR_'+str(parameterset['fCR_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])+'.npz' , phi=phiValues, aaH=phi_Wplus_Wminus_aaH, aH=phi_Wplus_Wminus_aH, saH=phi_Wplus_Wminus_saH, sH=phi_Wplus_Wminus_sH )
		#np.savez( outputdirectory+'normalized_phi_PLoS_'+str(sys.argv[1])+'_fCR_'+str(parameterset['fCR_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])+'.npz' )
		np.savez( outputdirectory+'normalized_phi_'+str(sys.argv[1])+'_fCR_'+str(parameterset['fCR_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])+'_kphi_'+str(sys.argv[3])+'.npz' , phi=phiValues, aaH=phi_Wplus_Wminus_aaH, aH=phi_Wplus_Wminus_aH, saH=phi_Wplus_Wminus_saH, sH=phi_Wplus_Wminus_sH )
		#np.savez( outputdirectory+'normalized_phi_'+str(sys.argv[1])+'_fCR_'+str(parameterset['fCR_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])+'.npz' , phi=phiValues, aaH=phi_Wplus_Wminus_aaH, aH=phi_Wplus_Wminus_aH, saH=phi_Wplus_Wminus_saH, sH=phi_Wplus_Wminus_sH )



	# calculate expected mean rate of weight change for estimated distributions 
	# load estimated distributions of spike response times from simulation data
	if sys.argv[1] == 'PLoS_single_all': 

		parameter_combinations = {}
		parameter_combinations['1']={'Astim':0.4, 'de':1.0, 'fCR_Hz':5.0, 'PpB':1, 'stimulus_type':'single', 'delay': 3.0}
		parameter_combinations['2']={'Astim':0.4, 'de':20.0, 'fCR_Hz':5.0, 'PpB':1, 'stimulus_type':'single', 'delay': 3.0}
		parameter_combinations['3']={'Astim':0.4, 'de':1.0, 'fCR_Hz':2.5, 'PpB':1, 'stimulus_type':'single', 'delay': 3.0}
		parameter_combinations['4']={'Astim':0.4, 'de':20.0, 'fCR_Hz':2.5, 'PpB':1, 'stimulus_type':'single', 'delay': 3.0}
		parameter_combinations['5']={'Astim':0.4, 'de':1.0, 'fCR_Hz':10.0, 'PpB':1, 'stimulus_type':'single', 'delay': 3.0}
		parameter_combinations['6']={'Astim':0.4, 'de':20.0, 'fCR_Hz':10.0, 'PpB':1, 'stimulus_type':'single', 'delay': 3.0}

		parameter_combinations['7']={'Astim':0.4, 'de':40.0, 'fCR_Hz':5.0, 'PpB':1, 'stimulus_type':'single', 'delay': 3.0}
		parameter_combinations['8']={'Astim':0.4, 'de':40.0, 'fCR_Hz':10.0, 'PpB':1, 'stimulus_type':'single', 'delay': 3.0}
		parameter_combinations['9']={'Astim':0.4, 'de':40.0, 'fCR_Hz':2.5, 'PpB':1, 'stimulus_type':'single', 'delay': 3.0}

		key = sys.argv[2]

		parameterset = parameter_combinations[key]

		#filename_est_lambda = directory_lambda+'dic_lambda_normalized_PLoS_electrical_burst_fCR_'+str(parameterset['fCR_Hz'])+'_fintra_'+str(parameterset['fintra_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])
		#filename_est_lambda = directory_lambda+'dic_lambda_normalized_PLoS_electrical_burst_fCR_'+str(parameterset['fCR_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])
		filename_est_lambda  = directory_lambda + 'dic_lambda_normalized_PLoS_electrical_burst_fCR_'+str(parameterset['fCR_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])
		print('####',directory_lambda)
		# stimulation parameters
		fHz = parameterset['fCR_Hz'] # Hz


		dic_lambda = {}

		print(filename_est_lambda)
		# load corresponding distributions from simulation data
		print("loading estimated lambdas from file ...")
		dic_lambda = dic_load( filename_est_lambda )

		# get time resolutions
		bins_ms = dic_lambda['bins ms'] # ms
		dt_theory = bins_ms[1] - bins_ms[0] # ms

		# maximal number of spikes considered
		B_max = dic_lambda['B_max']

		# load lambda arrays from dictionary
		lambda_array = []
		Lambda_array = []
		for kB in range(B_max):
			lambda_array.append( dic_lambda['est_lambda prob'][kB] )
			Lambda_array.append( dic_lambda['est_Lambda prob'][kB] )
		lambdaLastSpike = dic_lambda['est_Lambda_last prob']

		# get cumulative probability
		PK = dic_lambda['PK']
		Pk_cum = np.zeros( PK.shape )
		Pk_cum[0] = 1.0
		for k in range( 1,len(Pk_cum) ):
			Pk_cum[k] = Pk_cum[k-1]-PK[k]

		### uncommend to loaded plot distributions
		# import matplotlib.pyplot as plt 
		# plt.plot( bins_ms ,Lambda_array[0], label = "Lambda_1" )
		# # plt.plot( bins_ms ,Lambda_array[1], label = "Lambda_2" )
		# # plt.plot( bins_ms ,Lambda_array[2], label = "Lambda_3" )
		# plt.plot( bins_ms ,lambdaLastSpike, label = 'last' )
		# plt.legend()
		# plt.show()
		# exit()

		# axonal minus dendritic delay in ms
		Delta = parameterset['delay']

		# phi values for which results are calculated
		phiValues = np.arange(0,1,0.01)

		# initialize result array for different STDP functions
		phi_Wplus_Wminus_aaH = np.zeros( (len(phiValues),3) )
		phi_Wplus_Wminus_aH = np.zeros( (len(phiValues),3) )
		phi_Wplus_Wminus_saH = np.zeros( (len(phiValues),3) )
		phi_Wplus_Wminus_sH = np.zeros( (len(phiValues),3) )


		# start calculation of JofPhi 
		#for kphi in range(len(phiValues)):
		#for kphi in [int(sys.argv[3])]:
		for kphi in np.arange(100):

			print( kphi , '/' , len(phiValues) )

			phi = phiValues[kphi]
			dic_W = normalized_fast_W_(bins_ms, phi,fHz, Lambda_array, lambdaLastSpike, lambda_array, Delta, B_max, Pk_cum)

			phi_Wplus_Wminus_aaH[kphi,0] = phi
			phi_Wplus_Wminus_aaH[kphi,1] = dic_W['aaH']['Wplus']
			phi_Wplus_Wminus_aaH[kphi,2] = dic_W['aaH']['Wminus']

			phi_Wplus_Wminus_aH[kphi,0] = phi
			phi_Wplus_Wminus_aH[kphi,1] = dic_W['aH']['Wplus']
			phi_Wplus_Wminus_aH[kphi,2] = dic_W['aH']['Wminus']

			phi_Wplus_Wminus_saH[kphi,0] = phi
			phi_Wplus_Wminus_saH[kphi,1] = dic_W['saH']['Wplus']
			phi_Wplus_Wminus_saH[kphi,2] = dic_W['saH']['Wminus']

			phi_Wplus_Wminus_sH[kphi,0] = phi
			phi_Wplus_Wminus_sH[kphi,1] = dic_W['sH']['Wplus']
			phi_Wplus_Wminus_sH[kphi,2] = dic_W['sH']['Wminus']

		# save results
		# np.savez( outputdirectory+'PLoS_normalized_phi_'+str(sys.argv[1])+'_fCR_'+str(parameterset['fCR_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])+'.npz' , phi=phiValues, aaH=phi_Wplus_Wminus_aaH, aH=phi_Wplus_Wminus_aH, saH=phi_Wplus_Wminus_saH, sH=phi_Wplus_Wminus_sH )
		#np.savez( outputdirectory+'normalized_phi_PLoS_'+str(sys.argv[1])+'_fCR_'+str(parameterset['fCR_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])+'.npz' )
		#np.savez( outputdirectory+'normalized_phi_'+str(sys.argv[1])+'_fCR_'+str(parameterset['fCR_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])+'_kphi_'+str(sys.argv[3])+'.npz' , phi=phiValues, aaH=phi_Wplus_Wminus_aaH, aH=phi_Wplus_Wminus_aH, saH=phi_Wplus_Wminus_saH, sH=phi_Wplus_Wminus_sH )
		np.savez( outputdirectory+'normalized_phi_PLoS_single_fCR_'+str(parameterset['fCR_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])+'.npz' , phi=phiValues, aaH=phi_Wplus_Wminus_aaH, aH=phi_Wplus_Wminus_aH, saH=phi_Wplus_Wminus_saH, sH=phi_Wplus_Wminus_sH )




	if sys.argv[1] == 'PLoS_intra_2.5Hz': 

		parameter_combinations = {}
		parameter_combinations['1']={'Astim':0.8, 'de':1.0, 'PpB':3, 'fintra_Hz':120.0, 'fCR_Hz':2.5, 'delay': 3.0}
		parameter_combinations['2']={'Astim':0.8, 'de':1.0, 'PpB':5, 'fintra_Hz':120.0, 'fCR_Hz':2.5, 'delay': 3.0}
		parameter_combinations['3']={'Astim':0.8, 'de':1.0, 'PpB':8, 'fintra_Hz':120.0, 'fCR_Hz':2.5, 'delay': 3.0}
		parameter_combinations['4']={'Astim':0.8, 'de':1.0, 'PpB':3, 'fintra_Hz':60.0, 'fCR_Hz':2.5, 'delay': 3.0}
		parameter_combinations['5']={'Astim':0.8, 'de':1.0, 'PpB':5, 'fintra_Hz':60.0, 'fCR_Hz':2.5, 'delay': 3.0}
		parameter_combinations['6']={'Astim':0.8, 'de':1.0, 'PpB':8, 'fintra_Hz':60.0, 'fCR_Hz':2.5, 'delay': 3.0}

		key = sys.argv[2]

		parameterset = parameter_combinations[key]

		filename_est_lambda = directory_lambda+'dic_lambda_normalized_PLoS_electrical_burst_fCR_'+str(parameterset['fCR_Hz'])+'_fintra_'+str(parameterset['fintra_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])

		# stimulation parameters
		fHz = parameterset['fCR_Hz'] # Hz

		print(filename_est_lambda)
		# load corresponding distributions from simulation data
		print("loading estimated lambdas from file ...")
		dic_lambda = dic_load( filename_est_lambda )

		# get time resolutions
		bins_ms = dic_lambda['bins ms'] # ms
		dt_theory = bins_ms[1] - bins_ms[0] # ms

		# maximal number of spikes considered
		B_max = dic_lambda['B_max']

		# load lambda arrays from dictionary
		lambda_array = []
		Lambda_array = []
		for kB in range(B_max):
			lambda_array.append( dic_lambda['est_lambda prob'][kB] )
			Lambda_array.append( dic_lambda['est_Lambda prob'][kB] )
		lambdaLastSpike = dic_lambda['est_Lambda_last prob']

		# get cumulative probability
		PK = dic_lambda['PK']
		Pk_cum = np.zeros( PK.shape )
		Pk_cum[0] = 1.0
		for k in range( 1,len(Pk_cum) ):
			Pk_cum[k] = Pk_cum[k-1]-PK[k]

		### uncommend to loaded plot distributions
		# import matplotlib.pyplot as plt 
		# plt.plot( bins_ms ,Lambda_array[0], label = "Lambda_1" )
		# # plt.plot( bins_ms ,Lambda_array[1], label = "Lambda_2" )
		# # plt.plot( bins_ms ,Lambda_array[2], label = "Lambda_3" )
		# plt.plot( bins_ms ,lambdaLastSpike, label = 'last' )
		# plt.legend()
		# plt.show()
		# exit()

		# axonal minus dendritic delay in ms
		Delta = parameterset['delay']

		# phi values for which results are calculated
		phiValues = np.arange(0,1,0.01)

		# initialize result array for different STDP functions
		phi_Wplus_Wminus_aaH = np.zeros( (len(phiValues),3) )
		phi_Wplus_Wminus_aH = np.zeros( (len(phiValues),3) )
		phi_Wplus_Wminus_saH = np.zeros( (len(phiValues),3) )
		phi_Wplus_Wminus_sH = np.zeros( (len(phiValues),3) )

		# start calculation of JofPhi 
		for kphi in [int(sys.argv[3])]:
			#print(key, kphi, len(phiValues))
			phi = phiValues[kphi]
			dic_W = normalized_fast_W_(bins_ms, phi,fHz, Lambda_array, lambdaLastSpike, lambda_array, Delta, B_max, Pk_cum)

			phi_Wplus_Wminus_aaH[kphi,0] = phi
			phi_Wplus_Wminus_aaH[kphi,1] = dic_W['aaH']['Wplus']
			phi_Wplus_Wminus_aaH[kphi,2] = dic_W['aaH']['Wminus']

			phi_Wplus_Wminus_aH[kphi,0] = phi
			phi_Wplus_Wminus_aH[kphi,1] = dic_W['aH']['Wplus']
			phi_Wplus_Wminus_aH[kphi,2] = dic_W['aH']['Wminus']

			phi_Wplus_Wminus_saH[kphi,0] = phi
			phi_Wplus_Wminus_saH[kphi,1] = dic_W['saH']['Wplus']
			phi_Wplus_Wminus_saH[kphi,2] = dic_W['saH']['Wminus']

			phi_Wplus_Wminus_sH[kphi,0] = phi
			phi_Wplus_Wminus_sH[kphi,1] = dic_W['sH']['Wplus']
			phi_Wplus_Wminus_sH[kphi,2] = dic_W['sH']['Wminus']

		# save results
		np.savez( outputdirectory+'normalized_phi_'+str(sys.argv[1])+'_fCR_'+str(parameterset['fCR_Hz'])+'_fintra_'+str(parameterset['fintra_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])+'_kphi_'+str(sys.argv[3])+'.npz' , phi=phiValues, aaH=phi_Wplus_Wminus_aaH, aH=phi_Wplus_Wminus_aH, saH=phi_Wplus_Wminus_saH, sH=phi_Wplus_Wminus_sH )


	# calculate expected mean rate of weight change for estimated distributions 
	# load estimated distributions of spike response times from simulation data
	if sys.argv[1] == 'PLoS_intra_5.0Hz': 

		parameter_combinations = {}
		parameter_combinations['1']={'Astim':0.8, 'de':1.0, 'PpB':3, 'fintra_Hz':120.0, 'fCR_Hz':5.0, 'delay': 3.0 }
		parameter_combinations['2']={'Astim':0.8, 'de':1.0, 'PpB':5, 'fintra_Hz':120.0, 'fCR_Hz':5.0, 'delay': 3.0 }
		parameter_combinations['3']={'Astim':0.8, 'de':1.0, 'PpB':8, 'fintra_Hz':120.0, 'fCR_Hz':5.0, 'delay': 3.0 }
		parameter_combinations['4']={'Astim':0.8, 'de':1.0, 'PpB':3, 'fintra_Hz':60.0, 'fCR_Hz':5.0, 'delay': 3.0 }
		parameter_combinations['5']={'Astim':0.8, 'de':1.0, 'PpB':5, 'fintra_Hz':60.0, 'fCR_Hz':5.0, 'delay': 3.0 }
		parameter_combinations['6']={'Astim':0.8, 'de':1.0, 'PpB':8, 'fintra_Hz':60.0, 'fCR_Hz':5.0, 'delay': 3.0 }

		for key in [sys.argv[2]]:

			directory_lambda = '/scratch/users/jkromer/Phase_shifted_periodic_multisite_stimulation/theory/PLoS/data_lambda/'


			parameterset = parameter_combinations[key]

			filename_est_lambda = directory_lambda+'dic_lambda_normalized_PLoS_electrical_burst_fCR_'+str(parameterset['fCR_Hz'])+'_fintra_'+str(parameterset['fintra_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])

			# stimulation parameters
			fHz = parameterset['fCR_Hz'] # Hz

			print(filename_est_lambda)
			# load corresponding distributions from simulation data
			print("loading estimated lambdas from file ...")
			dic_lambda = dic_load( filename_est_lambda )

			# get time resolutions
			bins_ms = dic_lambda['bins ms'] # ms
			dt_theory = bins_ms[1] - bins_ms[0] # ms

			# maximal number of spikes considered
			B_max = dic_lambda['B_max']

			# load lambda arrays from dictionary
			lambda_array = []
			Lambda_array = []
			for kB in range(B_max):
				lambda_array.append( dic_lambda['est_lambda prob'][kB] )
				Lambda_array.append( dic_lambda['est_Lambda prob'][kB] )
			lambdaLastSpike = dic_lambda['est_Lambda_last prob']

			# get cumulative probability
			PK = dic_lambda['PK']
			Pk_cum = np.zeros( PK.shape )
			Pk_cum[0] = 1.0
			for k in range( 1,len(Pk_cum) ):
				Pk_cum[k] = Pk_cum[k-1]-PK[k]

			### uncommend to loaded plot distributions
			# import matplotlib.pyplot as plt 
			# plt.plot( bins_ms ,Lambda_array[0], label = "Lambda_1" )
			# # plt.plot( bins_ms ,Lambda_array[1], label = "Lambda_2" )
			# # plt.plot( bins_ms ,Lambda_array[2], label = "Lambda_3" )
			# plt.plot( bins_ms ,lambdaLastSpike, label = 'last' )
			# plt.legend()
			# plt.show()
			# exit()

			# axonal minus dendritic delay in ms
			Delta = parameterset['delay']

			# phi values for which results are calculated
			phiValues = np.arange(0,1,0.01)

			# initialize result array for different STDP functions
			phi_Wplus_Wminus_aaH = np.zeros( (len(phiValues),3) )
			phi_Wplus_Wminus_aH = np.zeros( (len(phiValues),3) )
			phi_Wplus_Wminus_saH = np.zeros( (len(phiValues),3) )
			phi_Wplus_Wminus_sH = np.zeros( (len(phiValues),3) )

			# start calculation of JofPhi 
			for kphi in range(len(phiValues)):

				phi = phiValues[kphi]
				dic_W = normalized_fast_W_(bins_ms, phi,fHz, Lambda_array, lambdaLastSpike, lambda_array, Delta, B_max, Pk_cum)

				phi_Wplus_Wminus_aaH[kphi,0] = phi
				phi_Wplus_Wminus_aaH[kphi,1] = dic_W['aaH']['Wplus']
				phi_Wplus_Wminus_aaH[kphi,2] = dic_W['aaH']['Wminus']

				phi_Wplus_Wminus_aH[kphi,0] = phi
				phi_Wplus_Wminus_aH[kphi,1] = dic_W['aH']['Wplus']
				phi_Wplus_Wminus_aH[kphi,2] = dic_W['aH']['Wminus']

				phi_Wplus_Wminus_saH[kphi,0] = phi
				phi_Wplus_Wminus_saH[kphi,1] = dic_W['saH']['Wplus']
				phi_Wplus_Wminus_saH[kphi,2] = dic_W['saH']['Wminus']

				phi_Wplus_Wminus_sH[kphi,0] = phi
				phi_Wplus_Wminus_sH[kphi,1] = dic_W['sH']['Wplus']
				phi_Wplus_Wminus_sH[kphi,2] = dic_W['sH']['Wminus']

			# save results
			np.savez( outputdirectory+'normalized_phi_'+str(sys.argv[1])+'_fCR_'+str(parameterset['fCR_Hz'])+'_fintra_'+str(parameterset['fintra_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])+'.npz' , phi=phiValues, aaH=phi_Wplus_Wminus_aaH, aH=phi_Wplus_Wminus_aH, saH=phi_Wplus_Wminus_saH, sH=phi_Wplus_Wminus_sH )


	if sys.argv[1] == 'PLoS_intra_10.0Hz': 

		parameter_combinations = {}
		parameter_combinations['1']={'Astim':0.8, 'de':1.0, 'PpB':3, 'fintra_Hz':120.0, 'fCR_Hz':10.0, 'delay': 3.0 }
		parameter_combinations['2']={'Astim':0.8, 'de':1.0, 'PpB':5, 'fintra_Hz':120.0, 'fCR_Hz':10.0, 'delay': 3.0 }
		parameter_combinations['3']={'Astim':0.8, 'de':1.0, 'PpB':8, 'fintra_Hz':120.0, 'fCR_Hz':10.0, 'delay': 3.0 }
		parameter_combinations['4']={'Astim':0.8, 'de':1.0, 'PpB':3, 'fintra_Hz':60.0, 'fCR_Hz':10.0, 'delay': 3.0 }
		parameter_combinations['5']={'Astim':0.8, 'de':1.0, 'PpB':5, 'fintra_Hz':60.0, 'fCR_Hz':10.0, 'delay': 3.0 }
		parameter_combinations['6']={'Astim':0.8, 'de':1.0, 'PpB':8, 'fintra_Hz':60.0, 'fCR_Hz':10.0, 'delay': 3.0 }

		for key in [sys.argv[2]]:

			directory_lambda = '/scratch/users/jkromer/Phase_shifted_periodic_multisite_stimulation/theory/PLoS/data_lambda/'


			parameterset = parameter_combinations[key]

			filename_est_lambda = directory_lambda+'dic_lambda_normalized_PLoS_electrical_burst_fCR_'+str(parameterset['fCR_Hz'])+'_fintra_'+str(parameterset['fintra_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])

			# stimulation parameters
			fHz = parameterset['fCR_Hz'] # Hz

			print(filename_est_lambda)
			# load corresponding distributions from simulation data
			print("loading estimated lambdas from file ...")
			dic_lambda = dic_load( filename_est_lambda )

			# get time resolutions
			bins_ms = dic_lambda['bins ms'] # ms
			dt_theory = bins_ms[1] - bins_ms[0] # ms

			# maximal number of spikes considered
			B_max = dic_lambda['B_max']

			# load lambda arrays from dictionary
			lambda_array = []
			Lambda_array = []
			for kB in range(B_max):
				lambda_array.append( dic_lambda['est_lambda prob'][kB] )
				Lambda_array.append( dic_lambda['est_Lambda prob'][kB] )
			lambdaLastSpike = dic_lambda['est_Lambda_last prob']

			# get cumulative probability
			PK = dic_lambda['PK']
			Pk_cum = np.zeros( PK.shape )
			Pk_cum[0] = 1.0
			for k in range( 1,len(Pk_cum) ):
				Pk_cum[k] = Pk_cum[k-1]-PK[k]

			### uncommend to loaded plot distributions
			# import matplotlib.pyplot as plt 
			# plt.plot( bins_ms ,Lambda_array[0], label = "Lambda_1" )
			# # plt.plot( bins_ms ,Lambda_array[1], label = "Lambda_2" )
			# # plt.plot( bins_ms ,Lambda_array[2], label = "Lambda_3" )
			# plt.plot( bins_ms ,lambdaLastSpike, label = 'last' )
			# plt.legend()
			# plt.show()
			# exit()

			# axonal minus dendritic delay in ms
			Delta = parameterset['delay']

			# phi values for which results are calculated
			phiValues = np.arange(0,1,0.01)

			# initialize result array for different STDP functions
			phi_Wplus_Wminus_aaH = np.zeros( (len(phiValues),3) )
			phi_Wplus_Wminus_aH = np.zeros( (len(phiValues),3) )
			phi_Wplus_Wminus_saH = np.zeros( (len(phiValues),3) )
			phi_Wplus_Wminus_sH = np.zeros( (len(phiValues),3) )

			# start calculation of JofPhi 
			for kphi in range(len(phiValues)):

				phi = phiValues[kphi]
				dic_W = normalized_fast_W_(bins_ms, phi,fHz, Lambda_array, lambdaLastSpike, lambda_array, Delta, B_max, Pk_cum)

				phi_Wplus_Wminus_aaH[kphi,0] = phi
				phi_Wplus_Wminus_aaH[kphi,1] = dic_W['aaH']['Wplus']
				phi_Wplus_Wminus_aaH[kphi,2] = dic_W['aaH']['Wminus']

				phi_Wplus_Wminus_aH[kphi,0] = phi
				phi_Wplus_Wminus_aH[kphi,1] = dic_W['aH']['Wplus']
				phi_Wplus_Wminus_aH[kphi,2] = dic_W['aH']['Wminus']

				phi_Wplus_Wminus_saH[kphi,0] = phi
				phi_Wplus_Wminus_saH[kphi,1] = dic_W['saH']['Wplus']
				phi_Wplus_Wminus_saH[kphi,2] = dic_W['saH']['Wminus']

				phi_Wplus_Wminus_sH[kphi,0] = phi
				phi_Wplus_Wminus_sH[kphi,1] = dic_W['sH']['Wplus']
				phi_Wplus_Wminus_sH[kphi,2] = dic_W['sH']['Wminus']

			# save results
			np.savez( outputdirectory+'normalized_phi_'+str(sys.argv[1])+'_fCR_'+str(parameterset['fCR_Hz'])+'_fintra_'+str(parameterset['fintra_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])+'.npz' , phi=phiValues, aaH=phi_Wplus_Wminus_aaH, aH=phi_Wplus_Wminus_aH, saH=phi_Wplus_Wminus_saH, sH=phi_Wplus_Wminus_sH )


















	# calculate expected mean rate of weight change for estimated distributions 
	# load estimated distributions of spike response times from simulation data
	if sys.argv[1] == 'PLoS_single_f_stim': 

		# array of all considered stimulation frequencies
		f_array = np.round( np.arange( 2.0, 20.0, 0.25 ) , 2 ) 

		# parameter combinations
		parameter_combinations = {}
		for kf_array in range( len(f_array) ):
			f = f_array[ kf_array ]
			parameter_combinations[ str(kf_array) ]={'Astim':0.4, 'de':1.0, 'fCR_Hz':f, 'PpB':1, 'stimulus_type':'single', 'delay': 3.0}

		key = sys.argv[2]
		print( 'calculating', key , len(f_array) )

		parameterset = parameter_combinations[key]

		#filename_est_lambda = directory_lambda+'dic_lambda_normalized_PLoS_electrical_burst_fCR_'+str(parameterset['fCR_Hz'])+'_fintra_'+str(parameterset['fintra_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])
		#filename_est_lambda = directory_lambda+'dic_lambda_normalized_PLoS_electrical_burst_fCR_'+str(parameterset['fCR_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])
		filename_est_lambda  = directory_lambda + 'dic_lambda_normalized_PLoS_electrical_burst_fCR_'+str(parameterset['fCR_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])
		print('####',directory_lambda)
		# stimulation parameters
		fHz = parameterset['fCR_Hz'] # Hz


		dic_lambda = {}

		print(filename_est_lambda)
		# load corresponding distributions from simulation data
		print("loading estimated lambdas from file ...")
		dic_lambda = dic_load_pkl( filename_est_lambda )

		# get time resolutions
		bins_ms = dic_lambda['bins ms'] # ms
		dt_theory = bins_ms[1] - bins_ms[0] # ms

		# maximal number of spikes considered
		B_max = dic_lambda['B_max']

		# load lambda arrays from dictionary
		lambda_array = []
		Lambda_array = []
		for kB in range(B_max):
			lambda_array.append( dic_lambda['est_lambda prob'][kB] )
			Lambda_array.append( dic_lambda['est_Lambda prob'][kB] )
		lambdaLastSpike = dic_lambda['est_Lambda_last prob']

		# get cumulative probability
		PK = dic_lambda['PK']
		Pk_cum = np.zeros( PK.shape )
		Pk_cum[0] = 1.0
		for k in range( 1,len(Pk_cum) ):
			Pk_cum[k] = Pk_cum[k-1]-PK[k]

		### uncommend to loaded plot distributions
		# import matplotlib.pyplot as plt 
		# plt.plot( bins_ms ,Lambda_array[0], label = "Lambda_1" )
		# # plt.plot( bins_ms ,Lambda_array[1], label = "Lambda_2" )
		# # plt.plot( bins_ms ,Lambda_array[2], label = "Lambda_3" )
		# plt.plot( bins_ms ,lambdaLastSpike, label = 'last' )
		# plt.legend()
		# plt.show()
		# exit()

		# axonal minus dendritic delay in ms
		Delta = parameterset['delay']

		# phi values for which results are calculated
		phiValues = np.arange(0,1,0.01)

		# initialize result array for different STDP functions
		phi_Wplus_Wminus_aaH = np.zeros( (len(phiValues),3) )
		phi_Wplus_Wminus_aH = np.zeros( (len(phiValues),3) )
		phi_Wplus_Wminus_saH = np.zeros( (len(phiValues),3) )
		phi_Wplus_Wminus_sH = np.zeros( (len(phiValues),3) )


		# start calculation of JofPhi 
		#for kphi in range(len(phiValues)):
		for kphi in [int(sys.argv[3])]:

			print( kphi , '/' , len(phiValues) )

			phi = phiValues[kphi]
			dic_W = normalized_fast_W_(bins_ms, phi,fHz, Lambda_array, lambdaLastSpike, lambda_array, Delta, B_max, Pk_cum)

			phi_Wplus_Wminus_aaH[kphi,0] = phi
			phi_Wplus_Wminus_aaH[kphi,1] = dic_W['aaH']['Wplus']
			phi_Wplus_Wminus_aaH[kphi,2] = dic_W['aaH']['Wminus']

			phi_Wplus_Wminus_aH[kphi,0] = phi
			phi_Wplus_Wminus_aH[kphi,1] = dic_W['aH']['Wplus']
			phi_Wplus_Wminus_aH[kphi,2] = dic_W['aH']['Wminus']

			phi_Wplus_Wminus_saH[kphi,0] = phi
			phi_Wplus_Wminus_saH[kphi,1] = dic_W['saH']['Wplus']
			phi_Wplus_Wminus_saH[kphi,2] = dic_W['saH']['Wminus']

			phi_Wplus_Wminus_sH[kphi,0] = phi
			phi_Wplus_Wminus_sH[kphi,1] = dic_W['sH']['Wplus']
			phi_Wplus_Wminus_sH[kphi,2] = dic_W['sH']['Wminus']

		# save results
		# np.savez( outputdirectory+'PLoS_normalized_phi_'+str(sys.argv[1])+'_fCR_'+str(parameterset['fCR_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])+'.npz' , phi=phiValues, aaH=phi_Wplus_Wminus_aaH, aH=phi_Wplus_Wminus_aH, saH=phi_Wplus_Wminus_saH, sH=phi_Wplus_Wminus_sH )
		#np.savez( outputdirectory+'normalized_phi_PLoS_'+str(sys.argv[1])+'_fCR_'+str(parameterset['fCR_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])+'.npz' )
		np.savez( outputdirectory+'normalized_phi_'+str(sys.argv[1])+'_fCR_'+str(parameterset['fCR_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])+'_kphi_'+str(sys.argv[3])+'.npz' , phi=phiValues, aaH=phi_Wplus_Wminus_aaH, aH=phi_Wplus_Wminus_aH, saH=phi_Wplus_Wminus_saH, sH=phi_Wplus_Wminus_sH )




	# calculate expected mean rate of weight change for estimated distributions 
	# load estimated distributions of spike response times from simulation data
	if sys.argv[1] == 'PLoS_burst_f_stim': 

		# array of all considered stimulation frequencies
		f_array = np.round( np.arange( 2.0, 20.0, 0.25 ) , 2 ) 
		fintra_array = [60.0, 120.0] # Hz
		ppb_array = [2,3,4]

		# parameter combinations
		parameter_combinations = {}

		kParSet = 0

		for f in f_array:
			for fintra in fintra_array:
				for ppb in ppb_array:
			
					parameter_combinations[ str(kParSet) ]={'Astim':0.8, 'de':1.0, 'fCR_Hz':f, 'PpB':ppb, 'fintra_Hz':fintra, 'stimulus_type':'burst', 'delay': 3.0}
					kParSet+=1


		key = sys.argv[2]
		# print( 'calculating', key , len(f_array) )
		# print( list( parameter_combinations.keys() ) )

		parameterset = parameter_combinations[key]

		# filename_est_lambda = directory_lambda+'dic_lambda_normalized_PLoS_electrical_burst_fCR_'+str(parameterset['fCR_Hz'])+'_fintra_'+str(parameterset['fintra_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])
		# filename_est_lambda = directory_lambda+'dic_lambda_normalized_PLoS_electrical_burst_fCR_'+str(parameterset['fCR_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])
		# filename_est_lambda  = directory_lambda + 'dic_lambda_normalized_PLoS_electrical_burst_fCR_'+str(parameterset['fCR_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])
		filename_est_lambda  = directory_lambda + 'dic_lambda_normalized_PLoS_electrical_burst_fCR_'+str(parameterset['fCR_Hz'])+'_fintra_'+str(parameterset['fintra_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])		

		print('####',directory_lambda)
		# stimulation parameters
		fHz = parameterset['fCR_Hz'] # Hz


		dic_lambda = {}

		# print(filename_est_lambda)
		# load corresponding distributions from simulation data
		print("loading estimated lambdas from file ...")
		dic_lambda = dic_load_pkl( filename_est_lambda )

		# get time resolutions
		bins_ms = dic_lambda['bins ms'] # ms
		dt_theory = bins_ms[1] - bins_ms[0] # ms

		# maximal number of spikes considered
		B_max = dic_lambda['B_max']

		# load lambda arrays from dictionary
		lambda_array = []
		Lambda_array = []
		for kB in range(B_max):
			lambda_array.append( dic_lambda['est_lambda prob'][kB] )
			Lambda_array.append( dic_lambda['est_Lambda prob'][kB] )
		lambdaLastSpike = dic_lambda['est_Lambda_last prob']

		# get cumulative probability
		PK = dic_lambda['PK']
		Pk_cum = np.zeros( PK.shape )
		Pk_cum[0] = 1.0
		for k in range( 1,len(Pk_cum) ):
			Pk_cum[k] = Pk_cum[k-1]-PK[k]

		### uncommend to loaded plot distributions
		# import matplotlib.pyplot as plt 
		# plt.plot( bins_ms ,Lambda_array[0], label = "Lambda_1" )
		# # plt.plot( bins_ms ,Lambda_array[1], label = "Lambda_2" )
		# # plt.plot( bins_ms ,Lambda_array[2], label = "Lambda_3" )
		# plt.plot( bins_ms ,lambdaLastSpike, label = 'last' )
		# plt.legend()
		# plt.show()
		# exit()

		# axonal minus dendritic delay in ms
		Delta = parameterset['delay']

		# phi values for which results are calculated
		phiValues = np.arange(0,1,0.01)

		# initialize result array for different STDP functions
		phi_Wplus_Wminus_aaH = np.zeros( (len(phiValues),3) )
		phi_Wplus_Wminus_aH = np.zeros( (len(phiValues),3) )
		phi_Wplus_Wminus_saH = np.zeros( (len(phiValues),3) )
		phi_Wplus_Wminus_sH = np.zeros( (len(phiValues),3) )


		# start calculation of JofPhi 
		#for kphi in range(len(phiValues)):
		for kphi in [int(sys.argv[3])]:

			print( kphi , '/' , len(phiValues) )

			phi = phiValues[kphi]
			dic_W = normalized_fast_W_(bins_ms, phi,fHz, Lambda_array, lambdaLastSpike, lambda_array, Delta, B_max, Pk_cum)

			phi_Wplus_Wminus_aaH[kphi,0] = phi
			phi_Wplus_Wminus_aaH[kphi,1] = dic_W['aaH']['Wplus']
			phi_Wplus_Wminus_aaH[kphi,2] = dic_W['aaH']['Wminus']

			phi_Wplus_Wminus_aH[kphi,0] = phi
			phi_Wplus_Wminus_aH[kphi,1] = dic_W['aH']['Wplus']
			phi_Wplus_Wminus_aH[kphi,2] = dic_W['aH']['Wminus']

			phi_Wplus_Wminus_saH[kphi,0] = phi
			phi_Wplus_Wminus_saH[kphi,1] = dic_W['saH']['Wplus']
			phi_Wplus_Wminus_saH[kphi,2] = dic_W['saH']['Wminus']

			phi_Wplus_Wminus_sH[kphi,0] = phi
			phi_Wplus_Wminus_sH[kphi,1] = dic_W['sH']['Wplus']
			phi_Wplus_Wminus_sH[kphi,2] = dic_W['sH']['Wminus']

		# save results
		# np.savez( outputdirectory+'PLoS_normalized_phi_'+str(sys.argv[1])+'_fCR_'+str(parameterset['fCR_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])+'.npz' , phi=phiValues, aaH=phi_Wplus_Wminus_aaH, aH=phi_Wplus_Wminus_aH, saH=phi_Wplus_Wminus_saH, sH=phi_Wplus_Wminus_sH )
		# np.savez( outputdirectory+'normalized_phi_PLoS_'+str(sys.argv[1])+'_fCR_'+str(parameterset['fCR_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])+'.npz' )
		# np.savez( outputdirectory+'normalized_phi_'+str(sys.argv[1])+'_fCR_'+str(parameterset['fCR_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])+'_kphi_'+str(sys.argv[3])+'.npz' , phi=phiValues, aaH=phi_Wplus_Wminus_aaH, aH=phi_Wplus_Wminus_aH, saH=phi_Wplus_Wminus_saH, sH=phi_Wplus_Wminus_sH )
		np.savez( outputdirectory+'normalized_phi_'+str(sys.argv[1])+'_fCR_'+str(parameterset['fCR_Hz'])+'_fintra_'+str(parameterset['fintra_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])+'_kphi_'+str(sys.argv[3])+'.npz' , phi=phiValues, aaH=phi_Wplus_Wminus_aaH, aH=phi_Wplus_Wminus_aH, saH=phi_Wplus_Wminus_saH, sH=phi_Wplus_Wminus_sH )



	if sys.argv[1] == 'combine_PLoS_burst_f_stim': 

		parameters = {}

		fCR_array = np.arange( 2.0, 20.0, 0.25)
		kcounter = 0

		parameters['Astim'] = 0.4
		parameters['Astim'] = float(sys.argv[2])

		parameters['de'] = 1.0
		parameters['ppb'] = 1
		parameters['ppb'] = int(sys.argv[3])
		parameters['fintra'] = 60.0 # Hz
		parameters['fintra'] = float(sys.argv[4]) # Hz

		for f in fCR_array:
		    print( kcounter , f, parameters['Astim'], parameters['ppb'], parameters['fintra']  )
		    parameters['f'] = f
		    
		    combine_kphi_results_J( parameters )
		    kcounter+=1

