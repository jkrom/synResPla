import os
import pickle
import numpy as np
import sys

def dic_save( dic ):
	with open(dic['filename'] + '.pickle', 'wb') as f:
		pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)

def dic_load(filename):
	with open(filename + '.pickle', 'rb') as f:
		return pickle.load(f)

# rate of weight change, loads precalculated weight updates from files
def JofPhi( stimulus_type, STDP_type, phi, fCR_Hz, eta, td):

	# directory from which data are loaded
	directoryWofPhi = 'data_WofPhi/'

	if stimulus_type == 'el_burst':
		# load data
		J_phi_data = np.load( directoryWofPhi+'phi_el_burst.npz' )
		phi_data = J_phi_data['phi']
		# get mean rate of weight change
		J_data = eta*fCR_Hz*(J_phi_data['Wminus'] + J_phi_data['Wplus'])

	
	elif stimulus_type == 'el_IBI':
		# load data     
		J_phi_data = np.load( directoryWofPhi+'normalized_phi_el_IBI_Delta_ms_'+str(td)+'.npz' )  
		phi_data = J_phi_data[STDP_type][:,0]
		# get mean rate of weight change
		J_data = eta*fCR_Hz*(J_phi_data[STDP_type][:,1] + J_phi_data[STDP_type][:,2])

	elif stimulus_type == 'el1':
		# load data      
		J_phi_data = np.load( directoryWofPhi+'normalized_phi_el1_Delta_ms_'+str(td)+'.npz' ) 
		phi_data = J_phi_data[STDP_type][:,0]
		# get mean rate of weight change
		J_data = eta*fCR_Hz*(J_phi_data[STDP_type][:,1] + J_phi_data[STDP_type][:,2])

	elif stimulus_type == 'el2':
		# load data      
		J_phi_data = np.load( directoryWofPhi+'normalized_phi_el2_Delta_ms_'+str(td)+'.npz' )         
		phi_data = J_phi_data[STDP_type][:,0]
		# get mean rate of weight change
		J_data = eta*fCR_Hz*(J_phi_data[STDP_type][:,1] + J_phi_data[STDP_type][:,2])

	elif stimulus_type == 'vib1':
		# load data      
		J_phi_data = np.load( directoryWofPhi+'normalized_phi_vib1_Delta_ms_'+str(td)+'.npz' ) 
		phi_data = J_phi_data[STDP_type][:,0]
		# get mean rate of weight change
		J_data = eta*fCR_Hz*(J_phi_data[STDP_type][:,1] + J_phi_data[STDP_type][:,2])
   
	elif stimulus_type == 'vib2':
		# load data      
		J_phi_data = np.load( directoryWofPhi+'normalized_phi_vib2_Delta_ms_'+str(td)+'.npz' ) 
		phi_data = J_phi_data[STDP_type][:,0]
		# get mean rate of weight change
		J_data = eta*fCR_Hz*(J_phi_data[STDP_type][:,1] + J_phi_data[STDP_type][:,2])
 
	elif stimulus_type == 'vib_burst':
		# load data     
		J_phi_data = np.load( directoryWofPhi+'normalized_phi_vib_burst_Delta_ms_'+str(td)+'.npz' ) 
		phi_data = J_phi_data[STDP_type][:,0]
		# get mean rate of weight change
		J_data = eta*fCR_Hz*(J_phi_data[STDP_type][:,1] + J_phi_data[STDP_type][:,2])
	else:
		print('ERROR: unknown stimulus model')

	# get J for arbitrary phi using linear interpolation
	J = np.interp(phi, phi_data , J_data)

	return J



# rate of weight change, loads precalculated weight updates from files
def JofPhi_par( phi, eta, parameterset):

	# directory from which data are loaded
	directoryWofPhi = '../figures/Fig6/data/'

	#print(parameterset['stimulus_type'])
	if parameterset['stimulus_type'] in ['3p120','5p120','8p120','3p60','5p60','8p60']:
		# load data     
		if parameterset['fCR_Hz'] == 2.5: # Hz
			filename = directoryWofPhi+'normalized_phi_PLoS_intra_fCR_'+str(parameterset['fCR_Hz'])+'_fintra_'+str(parameterset['fintra_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])+'_full.npz'
		else:
			filename = directoryWofPhi+'normalized_phi_PLoS_intra_fCR_'+str(parameterset['fCR_Hz'])+'_fintra_'+str(parameterset['fintra_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])+'.npz'


		J_phi_data = np.load( filename )

		phi_data = J_phi_data[parameterset['STDP_type']][:,0]
		# get mean rate of weight change
		J_data = eta*parameterset['fCR_Hz']*( J_phi_data[parameterset['STDP_type']][:,1] + J_phi_data[parameterset['STDP_type']][:,2] )
 
	elif parameterset['stimulus_type'] in ['single']:
		# load data     
		J_phi_data = np.load( directoryWofPhi+'normalized_phi_PLoS_single_fCR_'+str(parameterset['fCR_Hz'])+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_ppb_'+str(parameterset['PpB'])+'.npz')

		phi_data = J_phi_data[parameterset['STDP_type']][:,0]
		# get mean rate of weight change
		J_data = eta*parameterset['fCR_Hz']*( J_phi_data[parameterset['STDP_type']][:,1] + J_phi_data[parameterset['STDP_type']][:,2] )


	else:
		print('ERROR: unknown stimulus model')

	# get J for arbitrary phi using linear interpolation
	J = np.interp(phi, phi_data , J_data)

	return J





# calculate matrix of phase differences
def phiMat( Da_vec ):
	
	# get total number of subpopulations
	M = len(Da_vec)+1

	# initialize phi matrix
	phiMat = np.zeros( (M,M) )
	
	# run through all combinations of postsynaptic and presynaptic subpopulations
	for kPost in range(1,M):
		for kPre in range(0,kPost):
			phiMat[ kPost, kPre ] = np.mod( np.sum( Da_vec[kPre:kPost] ) , 1 )
			phiMat[ kPre, kPost ] = np.mod( -np.sum( Da_vec[kPre:kPost] ), 1 ) 

	return phiMat

# calculate matrix of estimated mean rate of weight change
def getJmat( phiMat , stimulus_type, fCR_Hz, STDP_type, xi):

	# initialize J matrix    
	J_mat = np.zeros( phiMat.shape )

	# STDP update per spike
	eta = 0.02 # STDP prefactor

	# run through all combinations of post and presynaptic subpopulations
	for x in range( len(J_mat) ):
		for y in range( len(J_mat) ):
			J_mat[x,y] = JofPhi(stimulus_type, STDP_type, phiMat[y,x], fCR_Hz, eta, xi)

	return J_mat


# calculate matrix of estimated mean rate of weight change
def getJmat_par( phiMat , parameterset):

	# initialize J matrix    
	J_mat = np.zeros( phiMat.shape )

	# STDP update per spike
	eta = 0.02 # STDP prefactor

	# run through all combinations of post and presynaptic subpopulations
	for x in range( len(J_mat) ):
		for y in range( len(J_mat) ):
			J_mat[x,y] = JofPhi_par( phiMat[y,x], eta, parameterset)

	return J_mat

# mean synaptic weight at time t for scalar J
def subPopMeanWeight( t, w0, J ):
	if J>0:
		return np.clip( w0 + (1.0-w0)*J*t , a_min = 0 , a_max=1)
	else:
		return np.clip( w0 + w0*J*t , a_min = 0 , a_max=1)

# overall mean synaptic weight
def meanWeight(t, M, w0_mat, J_mat):
	
	mw = 0

	for x in range(M):
		for y in range(M):
			wsub_t = subPopMeanWeight( t, w0_mat[x,y], J_mat[x,y] )
			mw+=wsub_t
	
	return mw/float(M*M) 

# overall mean synaptic weight
def meanSubWeights(t, M, w0_mat, J_mat):
	
	mw = 0
	#print  w0_mat, J_mat
	wt_mat = np.zeros( (M,M) )
	for x in range(M):
		for y in range(M):
			wsub_t = subPopMeanWeight( t, w0_mat[x,y], J_mat[x,y] )
			# weights for individual subpopulations
			wt_mat[x,y] = wsub_t
	
	return wt_mat 

if __name__ == "__main__":

	if sys.argv[1] == 'PLoS_single': 

		# output directory
		outputdirectory = '../figures/Fig6/data/'
			
		### initial mean weight
		w0 = 0.38
		### total number of subpopulations
		M = 3

		parameter_combinations = {}
		parameter_combinations['1']={'Astim':0.4, 'de':1.0, 'PpB':1, 'fCR_Hz':5.0, 'xi_ms':3.0, 'STDP_type':'aH', 'stimulus_type':'single'}
		parameter_combinations['2']={'Astim':0.4, 'de':20.0, 'PpB':1, 'fCR_Hz':5.0, 'xi_ms':3.0, 'STDP_type':'aH', 'stimulus_type':'single'}
		parameter_combinations['3']={'Astim':0.4, 'de':1.0, 'PpB':1, 'fCR_Hz':2.5, 'xi_ms':3.0, 'STDP_type':'aH', 'stimulus_type':'single'}
		parameter_combinations['4']={'Astim':0.4, 'de':20.0, 'PpB':1, 'fCR_Hz':2.5, 'xi_ms':3.0, 'STDP_type':'aH', 'stimulus_type':'single'}
		parameter_combinations['5']={'Astim':0.4, 'de':1.0, 'PpB':1, 'fCR_Hz':10.0, 'xi_ms':3.0, 'STDP_type':'aH', 'stimulus_type':'single'}
		parameter_combinations['6']={'Astim':0.4, 'de':20.0, 'PpB':1, 'fCR_Hz':10.0, 'xi_ms':3.0, 'STDP_type':'aH', 'stimulus_type':'single'}

		parameter_combinations['7']={'Astim':0.4, 'de':40.0, 'PpB':1, 'fCR_Hz':2.5, 'xi_ms':3.0, 'STDP_type':'aH', 'stimulus_type':'single'}
		parameter_combinations['8']={'Astim':0.4, 'de':40.0, 'PpB':1, 'fCR_Hz':5.0, 'xi_ms':3.0, 'STDP_type':'aH', 'stimulus_type':'single'}
		parameter_combinations['9']={'Astim':0.4, 'de':40.0, 'PpB':1, 'fCR_Hz':10.0, 'xi_ms':3.0, 'STDP_type':'aH', 'stimulus_type':'single'}

		key = sys.argv[2]

		parameterset = parameter_combinations[key]

		# stimulation frequency
		fCR_Hz = parameterset['fCR_Hz'] # Hz

		print(parameterset)

		# check whether results were already calculated
		outputFilename = 'PLoS_normalized_theory_phaselags_fCR_Hz_'+str(parameterset['fCR_Hz'])+'_single_'+parameterset['STDP_type']+'_xi_'+str(parameterset['xi_ms'])+'_M_'+str(M)+'_w0_'+str(w0)+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_PpB_'+str(parameterset['PpB'])
		if os.path.isfile( outputdirectory + outputFilename +'_single.pickle' ):
			print('results were already calculated')
		else:
			print('do calculations')
			print('single',parameterset['STDP_type'],parameterset['xi_ms'])
			
			# evaluation times at which minimal overall mean weight is desired
			Teval = np.array([20.0, 100.0, 1000.0]) # sec

			# initial block matrix of mean synaptic weights
			w0_mat = w0*np.ones( (M,M) )

			# bin size for phase differences
			dDa = 0.01
			#dDa = 0.0025
			Da_grid_values = np.arange(0,1,dDa)
   
			# initialize output dictionary
			dic_theory = {}																	
			dic_theory['filename']=outputdirectory + outputFilename
			dic_theory['pars'] = {'fCR_Hz' : str(parameterset['fCR_Hz']), 'stimulus_type': 'single', 'STDP_type':parameterset['STDP_type'], 'xi_ms':parameterset['xi_ms'],                         'M':M       ,'w0':w0 ,     'Astim' : parameterset['Astim'],        'de':parameterset['de']    , 'PpB': parameterset['PpB']  }

			# initialize grid for data results
			Da1_grid = np.zeros( ( len(Da_grid_values), len(Da_grid_values)) ) 
			Da2_grid = np.zeros( ( len(Da_grid_values), len(Da_grid_values)) ) 
			mwGrid = np.zeros( ( len(Da_grid_values), len(Da_grid_values) , len(Teval)) ) 

			# so far only M=3 is implemented
			for kDa1 in range(len(Da_grid_values)):

				Da1 = Da_grid_values[kDa1]   
				dic_theory[Da1] = {}
				print(Da1)

				for kDa2 in range(len(Da_grid_values)):
					Da2 = Da_grid_values[kDa2]
					dic_theory[Da1][Da2] = {}

					# define phase vector
					Da_vec = [Da1,Da2]

					# calculate matrix of phase differences between subpopulations
					phiMatrix = phiMat( Da_vec )

					# calculate corresponding matrix for estimated rates of weight changes
					JMat = getJmat_par( phiMat( Da_vec ) , parameterset )

					### calculate overall mean synaptic weight
					mw = meanWeight(Teval, len(Da_vec)+1, w0_mat, JMat)
					# add mean weight to dictionary
					dic_theory[Da1][Da2]['mw'] = mw

					# ##calculate mean synaptic weight for individual subpopulations
					wSub = meanSubWeights(Teval[-1], M, w0_mat, JMat)
					# add it to dictionary
					dic_theory[Da1][Da2]['wSub'] = wSub

					Da1_grid[kDa1,kDa2] = Da1
					Da2_grid[kDa1,kDa2] = Da2
					mwGrid[kDa1,kDa2] = mw

			# save grid data
			dic_theory['Da1_grid'] = Da1_grid
			dic_theory['Da2_grid'] = Da2_grid
			dic_theory['mwGrid'] = mwGrid
			dic_theory['Da_grid_values'] = Da_grid_values

			dic_save( dic_theory )

	
	if sys.argv[1] == 'PLoS_intra_5.0Hz': 

		# output directory
		outputdirectory = 'data_mwalpha1alpha2/'
			
		### initial mean weight
		w0 = 0.38
		### total number of subpopulations
		M = 3

		parameter_combinations = {}
		parameter_combinations['1']={'Astim':0.8, 'de':1.0, 'PpB':3, 'fintra_Hz':120.0, 'fCR_Hz':5.0, 'stimulus_type':'3p120', 'STDP_type':'aH', 'xi_ms':3.0}
		parameter_combinations['2']={'Astim':0.8, 'de':1.0, 'PpB':5, 'fintra_Hz':120.0, 'fCR_Hz':5.0, 'stimulus_type':'5p120', 'STDP_type':'aH', 'xi_ms':3.0}
		# parameter_combinations['3']={'Astim':0.8, 'de':1.0, 'PpB':8, 'fintra_Hz':120.0, 'fCR_Hz':5.0, 'stimulus_type':'8p120', 'STDP_type':'aH', 'xi_ms':3.0}
		parameter_combinations['4']={'Astim':0.8, 'de':1.0, 'PpB':3, 'fintra_Hz':60.0, 'fCR_Hz':5.0, 'stimulus_type':'3p60', 'STDP_type':'aH', 'xi_ms':3.0}
		parameter_combinations['5']={'Astim':0.8, 'de':1.0, 'PpB':5, 'fintra_Hz':60.0, 'fCR_Hz':5.0, 'stimulus_type':'5p60', 'STDP_type':'aH', 'xi_ms':3.0}
		# parameter_combinations['6']={'Astim':0.8, 'de':1.0, 'PpB':8, 'fintra_Hz':60.0, 'fCR_Hz':5.0, 'stimulus_type':'8p60', 'STDP_type':'aH', 'xi_ms':3.0}

		key = sys.argv[2]

		parameterset = parameter_combinations[key]

		# stimulation frequency
		fCR_Hz = parameterset['fCR_Hz'] # Hz

		print(parameterset)

		# check whether results were already calculated
		if os.path.isfile(outputdirectory+'NEW2_PLoS_normalized_theory_phaselags_fCR_Hz_'+str(parameterset['fCR_Hz'])+'_'+parameterset['stimulus_type']+'_'+parameterset['STDP_type']+'_xi_'+str(parameterset['xi_ms'])+'_M_'+str(M)+'_w0_'+str(w0)+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_PpB_'+str(parameterset['PpB'])+'_fintra_Hz_'+str(parameterset['fintra_Hz'])+'.pickle'):
			print('results were already calculated')
		else:
			print('do calculations')
			print(parameterset['stimulus_type'],parameterset['STDP_type'],parameterset['xi_ms'])
			
			# evaluation times at which minimal overall mean weight is desired
			Teval = np.array([20.0, 100.0, 1000.0]) # sec

			# initial block matrix of mean synaptic weights
			w0_mat = w0*np.ones( (M,M) )

			# bin size for phase differences
			dDa = 0.01
			Da_grid_values = np.arange(0,1,dDa)
   
			# initialize output dictionary
			dic_theory = {}																	
			dic_theory['filename']=outputdirectory+'NEW2_PLoS_normalized_theory_phaselags_fCR_Hz_'+str(parameterset['fCR_Hz'])+'_'+parameterset['stimulus_type']+'_'+parameterset['STDP_type']+'_xi_'+str(parameterset['xi_ms'])+'_M_'+str(M)+'_w0_'+str(w0)+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_PpB_'+str(parameterset['PpB'])+'_fintra_Hz_'+str(parameterset['fintra_Hz'])
			dic_theory['pars'] = {'fCR_Hz' : str(parameterset['fCR_Hz']), 'stimulus_type': parameterset['stimulus_type'], 'STDP_type':parameterset['STDP_type'], 'xi_ms':parameterset['xi_ms'],                         'M':M       ,'w0':w0 ,     'Astim' : parameterset['Astim'],        'de':parameterset['de']    , 'PpB': parameterset['PpB']    , 'fintra_Hz': parameterset['fintra_Hz'] }

			# initialize grid for data results
			Da1_grid = np.zeros( ( len(Da_grid_values), len(Da_grid_values)) ) 
			Da2_grid = np.zeros( ( len(Da_grid_values), len(Da_grid_values)) ) 
			mwGrid = np.zeros( ( len(Da_grid_values), len(Da_grid_values) , len(Teval)) ) 

			# so far only M=3 is implemented
			for kDa1 in range(len(Da_grid_values)):

				Da1 = Da_grid_values[kDa1]   
				dic_theory[Da1] = {}
				print(Da1)

				for kDa2 in range(len(Da_grid_values)):
					Da2 = Da_grid_values[kDa2]
					dic_theory[Da1][Da2] = {}

					# define phase vector
					Da_vec = [Da1,Da2]

					# calculate matrix of phase differences between subpopulations
					phiMatrix = phiMat( Da_vec )

					# calculate corresponding matrix for estimated rates of weight changes
					JMat = getJmat_par( phiMat( Da_vec ) , parameterset )

					### calculate overall mean synaptic weight
					mw = meanWeight(Teval, len(Da_vec)+1, w0_mat, JMat)
					# add mean weight to dictionary
					dic_theory[Da1][Da2]['mw'] = mw

					# ##calculate mean synaptic weight for individual subpopulations
					wSub = meanSubWeights(Teval[-1], M, w0_mat, JMat)
					# add it to dictionary
					dic_theory[Da1][Da2]['wSub'] = wSub

					Da1_grid[kDa1,kDa2] = Da1
					Da2_grid[kDa1,kDa2] = Da2
					mwGrid[kDa1,kDa2] = mw

			# save grid data
			dic_theory['Da1_grid'] = Da1_grid
			dic_theory['Da2_grid'] = Da2_grid
			dic_theory['mwGrid'] = mwGrid
			dic_theory['Da_grid_values'] = Da_grid_values

			dic_save( dic_theory )

	if sys.argv[1] == 'PLoS_intra_2.5Hz': 

		# output directory
		outputdirectory = 'data_mwalpha1alpha2/'
			
		### initial mean weight
		w0 = 0.38
		### total number of subpopulations
		M = 3

		parameter_combinations = {}
		parameter_combinations['1']={'Astim':0.8, 'de':1.0, 'PpB':3, 'fintra_Hz':120.0, 'fCR_Hz':2.5, 'stimulus_type':'3p120', 'STDP_type':'aH', 'xi_ms':3.0}
		parameter_combinations['2']={'Astim':0.8, 'de':1.0, 'PpB':5, 'fintra_Hz':120.0, 'fCR_Hz':2.5, 'stimulus_type':'5p120', 'STDP_type':'aH', 'xi_ms':3.0}
		parameter_combinations['3']={'Astim':0.8, 'de':1.0, 'PpB':8, 'fintra_Hz':120.0, 'fCR_Hz':2.5, 'stimulus_type':'8p120', 'STDP_type':'aH', 'xi_ms':3.0}
		parameter_combinations['4']={'Astim':0.8, 'de':1.0, 'PpB':3, 'fintra_Hz':60.0, 'fCR_Hz':2.5, 'stimulus_type':'3p60', 'STDP_type':'aH', 'xi_ms':3.0}
		parameter_combinations['5']={'Astim':0.8, 'de':1.0, 'PpB':5, 'fintra_Hz':60.0, 'fCR_Hz':2.5, 'stimulus_type':'5p60', 'STDP_type':'aH', 'xi_ms':3.0}
		parameter_combinations['6']={'Astim':0.8, 'de':1.0, 'PpB':8, 'fintra_Hz':60.0, 'fCR_Hz':2.5, 'stimulus_type':'8p60', 'STDP_type':'aH', 'xi_ms':3.0}

		key = sys.argv[2]

		parameterset = parameter_combinations[key]

		# stimulation frequency
		fCR_Hz = parameterset['fCR_Hz'] # Hz

		print(parameterset)

		# check whether results were already calculated
		if os.path.isfile(outputdirectory+'NEW2_PLoS_normalized_theory_phaselags_fCR_Hz_'+str(parameterset['fCR_Hz'])+'_'+parameterset['stimulus_type']+'_'+parameterset['STDP_type']+'_xi_'+str(parameterset['xi_ms'])+'_M_'+str(M)+'_w0_'+str(w0)+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_PpB_'+str(parameterset['PpB'])+'_fintra_Hz_'+str(parameterset['fintra_Hz'])+'.pickle'):
			print('results were already calculated')
		else:
			print('do calculations')
			print(parameterset['stimulus_type'],parameterset['STDP_type'],parameterset['xi_ms'])
			
			# evaluation times at which minimal overall mean weight is desired
			Teval = np.array([20.0, 100.0, 1000.0]) # sec

			# initial block matrix of mean synaptic weights
			w0_mat = w0*np.ones( (M,M) )

			# bin size for phase differences
			dDa = 0.01
			Da_grid_values = np.arange(0,1,dDa)
   
			# initialize output dictionary
			dic_theory = {}																	
			dic_theory['filename']=outputdirectory+'NEW2_PLoS_normalized_theory_phaselags_fCR_Hz_'+str(parameterset['fCR_Hz'])+'_'+parameterset['stimulus_type']+'_'+parameterset['STDP_type']+'_xi_'+str(parameterset['xi_ms'])+'_M_'+str(M)+'_w0_'+str(w0)+'_Astim_'+str(parameterset['Astim'])+'_de_'+str(parameterset['de'])+'_PpB_'+str(parameterset['PpB'])+'_fintra_Hz_'+str(parameterset['fintra_Hz'])
			dic_theory['pars'] = {'fCR_Hz' : str(parameterset['fCR_Hz']), 'stimulus_type': parameterset['stimulus_type'], 'STDP_type':parameterset['STDP_type'], 'xi_ms':parameterset['xi_ms'],                         'M':M       ,'w0':w0 ,     'Astim' : parameterset['Astim'],        'de':parameterset['de']    , 'PpB': parameterset['PpB']    , 'fintra_Hz': parameterset['fintra_Hz'] }

			# initialize grid for data results
			Da1_grid = np.zeros( ( len(Da_grid_values), len(Da_grid_values)) ) 
			Da2_grid = np.zeros( ( len(Da_grid_values), len(Da_grid_values)) ) 
			mwGrid = np.zeros( ( len(Da_grid_values), len(Da_grid_values) , len(Teval)) ) 

			# so far only M=3 is implemented
			for kDa1 in range(len(Da_grid_values)):

				Da1 = Da_grid_values[kDa1]   
				dic_theory[Da1] = {}
				print(Da1)

				for kDa2 in range(len(Da_grid_values)):
					Da2 = Da_grid_values[kDa2]
					dic_theory[Da1][Da2] = {}

					# define phase vector
					Da_vec = [Da1,Da2]

					# calculate matrix of phase differences between subpopulations
					phiMatrix = phiMat( Da_vec )

					# calculate corresponding matrix for estimated rates of weight changes
					JMat = getJmat_par( phiMat( Da_vec ) , parameterset )
					#exit()

					### calculate overall mean synaptic weight
					mw = meanWeight(Teval, len(Da_vec)+1, w0_mat, JMat)
					# add mean weight to dictionary
					dic_theory[Da1][Da2]['mw'] = mw

					# ##calculate mean synaptic weight for individual subpopulations
					wSub = meanSubWeights(Teval[-1], M, w0_mat, JMat)
					# add it to dictionary
					dic_theory[Da1][Da2]['wSub'] = wSub

					Da1_grid[kDa1,kDa2] = Da1
					Da2_grid[kDa1,kDa2] = Da2
					mwGrid[kDa1,kDa2] = mw

			# save grid data
			dic_theory['Da1_grid'] = Da1_grid
			dic_theory['Da2_grid'] = Da2_grid
			dic_theory['mwGrid'] = mwGrid
			dic_theory['Da_grid_values'] = Da_grid_values

			dic_save( dic_theory )
