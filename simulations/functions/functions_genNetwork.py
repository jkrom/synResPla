##################################
# written by: Justus Kromer
##################################
# written and tested for Python 2.7.13
##################################
#
#	content:
#
#		contains all scripts for generating connectivity matrices
#
#		class ellipsoid() 	l 34
#							def __init__(self, float , float, float )
#							def isInVolumen(self, Points)
#
#		def placeNeurons( int_1d , int_1d ) 	
#
# 		def conProbability( float_1d, float )    	
#
#		def cartesianProduct( element_1d , element_1d )	 
#
#		def synReversalsCmatrix( float_3d, float_3d, int_1d, int_1d, float, float, float, float)	
#
#		def generate_connectivity_and_weight_matrix_Ebert( system_parameters , rnd_state_for_network_generation )
#
#		def placeNeurons_3D_cuboid( NSTN , NGPe, rx_STN, ry_STN, rz_STN, rx_GPe, ry_GPe, rz_GPe )
#
# 		def synReversalsCmatrix_3D_cuboid(positionsSTNNeurons, positionsGPeNeurons, NSTN, NGPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN, r_STN, r_GPe)
#
#		def generate_connectivity_and_weight_matrix_3D_cuboid( system_parameters , rnd_state_for_network_generation )
#
#	if implemented run 'python functions_genNetwork.py CLASS_OR_FUNCTIONNAME' to test function or class with 'CLASS_OR_FUNCTIONNAME'

# imports
from scipy.interpolate import interp1d
import numpy as np 
import itertools
import scipy


##################################
# class: ellipsoid
##################################
class ellipsoid():
# Description:
#   Ellipsoidal volume representing brain regions in three dimensions.
#
	def __init__(self, xSP=6.0, ySP=3.0, zSP=2.5 ): 
	# default input are dimensions for STN according to Ebert et al. 2014. p4 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe
	# order changed, to make x axes longest axes

		#   ellipsoid.XSPAxes ... length of semi-principle axes in x direction (mm)
		self.XSPAxes=xSP	# mm
		#   ellipsoid.YSPAxes ... length of semi-principle axes in y direction (mm)
		self.YSPAxes=ySP	# mm
		#   ellipsoid.ZSPAxes ... length of semi-principle axes in z direction (mm)
		self.ZSPAxes=zSP	# mm


	##################################
	#   function: isInVolumen(self, Point)
	def isInVolumen(self, Points):
	#
	#       Checks whether elements of 'Point' are in volume
	#
	#       input: Points
	#           Points ... numpy array of  x,y,z values representing positions in three dimension ( same units as principle axes lengths)
	#       return: bools
	#			bools ... numpy array of bools ('True' if corresponding element in Points is in volume)
	#   
		# rescale to position in sphere
		x=Points[:,0]/self.XSPAxes	# mm
		y=Points[:,1]/self.YSPAxes	# mm
		z=Points[:,2]/self.ZSPAxes	# mm

		# return true if element in sphere
		return np.power(z,2) < 1-np.power(x,2)-np.power(y,2)





##################################
# function: placeNeurons
def placeNeurons( NSTN , NGPe ):
#
#       Places NSTN neurons in ellipsoid associated with subthalamic nucleus (STN) and 
#		NGPe in ellipsoid associated with globus pallidus externus (GPe).
#	    Dimensions of ellipsoids as in Ebert et al. 2014. p4 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe, 
#		however, they are centered at origin and semi-principle axes are aligned with coordinate system.	
#
#       input: NSTN, NGPe
#           NSTN ... number of STN neurons that need to be placed
#			NGPe ... number of GPe neurons that need to be placed
#       return: STNNeuronPositons, GPeNeuronPositons	
# 			STNNeuronPositons ... numpy array of STN neuron centers in 3d (mm) 
# 			GPeNeuronPositons ... numpy array of GPe neuron centers in 3d (mm) 

# 1) place STN neurons
	# generate ellipsoidal volume
	STNArea=ellipsoid()

	# (i) start with uniformly distributed positions in 3d .. 
	NeuronToPlace=np.random.uniform([-STNArea.XSPAxes, -STNArea.YSPAxes, -STNArea.ZSPAxes], [STNArea.XSPAxes, STNArea.YSPAxes, STNArea.ZSPAxes], [NSTN, 3])
	
	# .. and (ii) keep those that are inside STN Area 
	# STNNeuronPositons ... contains neuron positions in 3d of neurons that are already placed inside STN volume
	STNNeuronPositons=np.copy(NeuronToPlace[ STNArea.isInVolumen(NeuronToPlace) ])

	# kSTNNeuronsPlaced ... number of Neurons that are already placed inside STN volume
	kSTNNeuronsPlaced=len(STNNeuronPositons)

	# repeat steps (i,ii) for additional neurons until all NSTN neurons are placed inside STN volume
	while kSTNNeuronsPlaced<NSTN:

		# (i) for 'next neuron'
		NeuronToPlace=np.random.uniform([-STNArea.XSPAxes, -STNArea.YSPAxes, -STNArea.ZSPAxes], [STNArea.XSPAxes, STNArea.YSPAxes, STNArea.ZSPAxes], [1, 3])
		
		# (ii)
		# check whether 'next neuron' is placed inside STN volume
		if STNArea.isInVolumen(NeuronToPlace)[0]==True:

			# if placed correctly, add to list of already-placed neurons
			# consider the case that no neuron has been placed yet
			if len(STNNeuronPositons) > 0:
				STNNeuronPositons=np.concatenate(  (STNNeuronPositons, NeuronToPlace)  , axis=0 )
			else:
				STNNeuronPositons = np.array([NeuronToPlace])
			# keep track of the number of already-placed neurons
			kSTNNeuronsPlaced+=1

### repeat the same steps for GPe neurons
# 2) place GPe neurons
	# generate ellipsoidal volume
	# generate ellipsoidal volume
	GPeArea=ellipsoid(xSP=4.6, ySP=12.3, zSP=3.2) # input in mm according to Ebert et al. 2014. p4 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe

	# (i) start with uniformly distributed positions in 3d .. 
	NeuronToPlace=np.random.uniform([-GPeArea.XSPAxes, -GPeArea.YSPAxes, -GPeArea.ZSPAxes], [GPeArea.XSPAxes, GPeArea.YSPAxes, GPeArea.ZSPAxes], [NGPe, 3])
	
	# .. and (ii) keep those that are inside GPe Area 
	# GPeNeuronPositons ... contains neuron positions in 3d of neurons that are already placed inside GPe volume
	GPeNeuronPositons=np.copy(NeuronToPlace[ GPeArea.isInVolumen(NeuronToPlace) ])

	# kGPeNeuronsPlaced ... number of Neurons that are already placed inside GPe volume
	kGPeNeuronsPlaced=len(GPeNeuronPositons)

	# repeat steps (i,ii) for additional neurons until all NGPe neurons are placed inside GPe volume
	while kGPeNeuronsPlaced<NGPe:

		# (i) for 'next neuron'
		NeuronToPlace=np.random.uniform([-GPeArea.XSPAxes, -GPeArea.YSPAxes, -GPeArea.ZSPAxes], [GPeArea.XSPAxes, GPeArea.YSPAxes, GPeArea.ZSPAxes], [1, 3])
		
		# (ii)
		# check whether 'next neuron' is placed inside GPe volume
		if GPeArea.isInVolumen(NeuronToPlace)[0]==True:

			# if placed correctly, add to list of already-placed neurons
			# consider the case that no neuron has been placed yet
			if len(GPeNeuronPositons) > 0:
				GPeNeuronPositons=np.concatenate(  (GPeNeuronPositons, NeuronToPlace)  , axis=0 )
			else:
				GPeNeuronPositons = np.array([NeuronToPlace])
			
			# keep track of the number of already-placed neurons
			kGPeNeuronsPlaced+=1

	# return lists of neuron positions in 3d that were placed in STN and GPe volume, respectively 
	return STNNeuronPositons, GPeNeuronPositons



##################################
# function: conProbability
def conProbability(d, Cd):
#
#       Returns a the probability for neurons to connect at a certain distance d.
#		Considers exponential decay as in 
#		Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe	
#
#       input: d, Cd
#           d ... 1d array of distances (same units as Cd)
#			Cd ... scale for exponential decay (same units as d)
#       return: STNNeuronPositons, GPeNeuronPositons	
# 			p ... 1d array of connection probability for distances d
	# connection probability decays exponentially
	p=1/Cd*np.exp(-d/Cd )
	
	# return probabilities
	return p



##################################
# function: conProbability
#
#	Returns the cartesian product of a and b (list of all possible combinations of elements of 'a' an 'b')
#
#       input: a, b
#			a ... list of elements
#			b ... list of elements
#		return:  
#			list of all possible combinations of elements of a and b
def cartesianProduct(a,b):
	return np.array([x for x in itertools.product(a, b)])



##################################
# function: synReversalsCmatrix
# 	Returns a 2d matrix of integers indicating which neurons are connected by an excitatory synapse (entry 1),
#   which neurons are connected by inhibitory synapses (entry -1) or which neurons are not connected (entry = 0).
#   
#	The shape of that matrix is a block matrix with dimension (NSTN+NGPe, NSTN+NGPe)
#
#	Recurrent STN and GPe connections are implemented according to a 
#	distance-dependent connection probability with exponential shape taken from Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe	
#
#	Probability for internetwork connections does not depend on the distance, as in Ebert et al. 2014
#	
#	Distance-dependent connection are randomly implemented by calculating the probability for each possible connections and then
#	drawing the desired number of connections without replacement from the pool of all possible connections.
#
#		input:   positionsSTNNeurons, positionsGPeNeurons, NSTN, NGPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN
#			positionsSTNNeurons ... 1d np.array of STN neuron positions in 3d in units of mm
#			positionsGPeNeurons ... 1d np.array of GPe neuron positions in 3d in units of mm
#			NSTN ... total number of STN neurons
#			NGPe ... total number of GPe neurons
#			P_STN_STN ... Probability for STN -> STN connections (total number of connections is P_STN_STN * ( NSTN * NSTN ) )
#			P_STN_GPe ... Probability for STN -> GPe connections (total number of connections is P_STN_GPe * ( NSTN * NGPe ) )
#			P_GPe_GPe ... Probability for GPe -> GPe connections (total number of connections is P_GPe_GPe * ( NGPe * NGPe ) )
#			P_GPe_STN ... Probability for GPe -> STN connections (total number of connections is P_GPe_STN * ( NGPe * BSTN ) )
#
#		return:
#			synReversals ... block matrix of integers and dimension (NSTN+NGPe, NSTN+NGPe). synReversals[i,j] contains information about the 
#							 connection between presynatpic neuron j and postsynatpic neuron i
#							 synReversals[i,j] = 1 -> exc. connections from j to i
#							 synReversals[i,j] = -1 -> inh. connections from j to i
#							 synReversals[i,j] = 0 -> no connections from j to i
def synReversalsCmatrix(positionsSTNNeurons, positionsGPeNeurons, NSTN, NGPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN):

	# initiallize return matrix containing synReversals[i,j]
	# 1  ... presynaptic neuron j is connected to postsynapti neuron i by excitatory synapse
	# -1 ... presynaptic neuron j is connected to postsynapti neuron i  by inhibitory synapse
	# 0 ... no connections from j to i
	synReversals=np.zeros( (NSTN+NGPe, NSTN+NGPe) ) 

	##################################
	# CONNECTIONS FOR STN -> STN
	##################################
	# distance-dependent connection probability according to Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe	

	# total number of STN -> STN connections (round is actually not necessary, just to avoid non integer input )
	totNumberOfConnection=int( np.round( P_STN_STN * NSTN * NSTN ) )

	# implement array of all possible STN -> STN connections
	# first index for pre, second for post synaptic neuron
	allPossibleSTNtoSTNconnections=cartesianProduct( np.arange(NSTN),np.arange(NSTN) )

	# calculate distances related to these connection  in units of mm
	distances=np.linalg.norm( positionsSTNNeurons[allPossibleSTNtoSTNconnections[:,1]]-positionsSTNNeurons[allPossibleSTNtoSTNconnections[:,0]] , axis=1 )

	# probability densities to implement a connection with those lengths according to 
	# Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe	
	Cd_STN = 0.5 # characteristic scale in mm
	probs=conProbability( distances, Cd_STN ) # connection probability density in 1/mm

	# exclude self connections
	indizesOfNonSelfconnections=allPossibleSTNtoSTNconnections[:,0]!=allPossibleSTNtoSTNconnections[:,1]
	probs=probs[indizesOfNonSelfconnections]	# 1/mm
	allPossibleSTNtoSTNconnections=allPossibleSTNtoSTNconnections[indizesOfNonSelfconnections]

	# normalize probabiltiy to one
	probs=1/np.sum(probs)*probs   # probs contains the probability for each connection to be selected if only a single connection was implemented
	
	# implement synaptic connections according to probabilities
	STNSTNconnectionsIndizes=np.random.choice( len(allPossibleSTNtoSTNconnections) , totNumberOfConnection, p=probs, replace=False)
	
	# STNSTNconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
	STNSTNconnections=allPossibleSTNtoSTNconnections[ STNSTNconnectionsIndizes ]

	# add these connections to the return matrix 'synReversals'
	for connection in STNSTNconnections:
		# add excitatory connection
		# note that synReversals[i,j] is refers to connections from presynaptic neuron j to presynaptic neuron i
		synReversals[ connection[1],  connection[0] ]=1


	##################################	
	# CONNECTIONS FOR STN-> GPe
	##################################
	#  the connections probability for STN -> GPe connections does not depend on the distance

	# total number of STN-> GPe connections (round is actually not necessary, just to avoid non integer input )
	totNumberOfConnection=int(np.round(P_STN_GPe*NSTN*NGPe))

	# implement array of all possible STN -> GPe connections
	# first index for pre, second for post synaptic neuron      neuron indices for STN neurons are 0 - NSTN-1 and for GPe neurons NSTN - NSTN+NGPe-1
	allPossibleSTNtoGPeconnections=cartesianProduct( np.arange(NSTN),np.arange(NSTN, NSTN+NGPe ) )

	# calculate distances related to these connection  in units of mm
	distances=np.linalg.norm( positionsGPeNeurons[allPossibleSTNtoGPeconnections[:,1]-NSTN]-positionsSTNNeurons[allPossibleSTNtoGPeconnections[:,0]] , axis=1 )

	# all STN-> GPe are implemented with the same probability
	probs=np.full( len(allPossibleSTNtoGPeconnections), 1/float(totNumberOfConnection) )

	# normalize to probs one
	probs=1/np.sum(probs)*probs # probs contains the probability for each connection to be selected if only a single connection was implemented
	
	# implement synaptic connections according to probabilities
	STNGPeConnectionsIndizes=np.random.choice(len(allPossibleSTNtoGPeconnections), totNumberOfConnection, p=probs, replace=False)
	
	# STNGPeconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
	STNGPeconnections=allPossibleSTNtoGPeconnections[ STNGPeConnectionsIndizes ]

	# add these connections to the return matrix 'synReversals'
	for connection in STNGPeconnections:
		# add excitatory connection
		# note that synReversals[i,j] is refers to connections from presynaptic neuron j to presynaptic neuron i
		synReversals[ connection[1],  connection[0] ]=1




	##################################	
	# CONNECTIONS FOR GPe-> GPe
	##################################
	# distance-dependent connection probability according to Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe	

	# total number of GPe-> GPe connections (round is actually not necessary, just to avoid non integer input )
	totNumberOfConnection=int(np.round(P_GPe_GPe*NGPe*NGPe))

	# implement array of all possible GPe -> GPe connections
	# first index for pre, second for post synaptic neuron      neuron indices for STN neurons are 0 - NSTN-1 and for GPe neurons NSTN - NSTN+NGPe-1
	allPossibleGPetoGPeconnections=cartesianProduct( np.arange( NSTN, NSTN+NGPe ) ,np. arange( NSTN, NSTN+NGPe )  )

	# calculate distances related to these connection  in units of mm
	distances=np.linalg.norm( positionsGPeNeurons[allPossibleGPetoGPeconnections[:,1]-NSTN]-positionsGPeNeurons[allPossibleGPetoGPeconnections[:,0]-NSTN] , axis=1 )

	# probability densities to implement a connection with those lengths according to 
	# Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe	
	Cd_GPe = 0.63 # characteristic scale in mm
	probs=conProbability(distances, Cd_GPe ) # connection probability density in 1/mm

	# exclude self connections
	indizesOfNonSelfconnections=allPossibleGPetoGPeconnections[:,0]!=allPossibleGPetoGPeconnections[:,1]
	probs=probs[indizesOfNonSelfconnections]
	allPossibleGPetoGPeconnections=allPossibleGPetoGPeconnections[indizesOfNonSelfconnections]

	# normalize to probs one
	probs=1/np.sum(probs)*probs # probs contains the probability for each connection to be selected if only a single connection was implemented
	
	# implement synaptic connections according to probabilities
	GPeGPeconnectionsIndizes=np.random.choice(len(allPossibleGPetoGPeconnections), totNumberOfConnection, p=probs, replace=False)
	
	# GPeGPeconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
	GPeGPeconnections=allPossibleGPetoGPeconnections[ GPeGPeconnectionsIndizes ]

	# add these connections to the return matrix 'synReversals'
	for connection in GPeGPeconnections:
		# add inhibitory connection
		# note that synReversals[i,j] is refers to connections from presynaptic neuron j to presynaptic neuron i
		synReversals[ connection[1],  connection[0] ]=-1


	##################################	
	# CONNECTIONS FOR GPe-> STN
	##################################
	#  the connections probability for STN -> GPe connections does not depend on the distance

	# total number of GPe-> STN connections (round is actually not necessary, just to avoid non integer input )
	totNumberOfConnection=int(np.round(P_GPe_STN*NGPe*NSTN))

	# implement array of all possible GPe -> STN connections
	# first index for pre, second for post synaptic neuron      neuron indices for STN neurons are 0 - NSTN-1 and for GPe neurons NSTN - NSTN+NGPe-1
	allPossibleGPetoSTNconnections=cartesianProduct( np.arange(NSTN, NSTN+NGPe ) ,np.arange(NSTN )  )

	# calculate distances related to these connection  in units of mm
	distances=np.linalg.norm( positionsSTNNeurons[allPossibleGPetoSTNconnections[:,1]]-positionsGPeNeurons[allPossibleGPetoSTNconnections[:,0]-NSTN] , axis=1 )

	# all GPe -> STN are implemented with the same probability	
	probs=np.full( len(allPossibleGPetoSTNconnections), 1/float(totNumberOfConnection) )

	# normalize to probs one
	probs=1/np.sum(probs)*probs # probs contains the probability for each connection to be selected if only a single connection was implemented

	# GPeSTNconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
	GPeSTNconnectionsIndizes=np.random.choice(len(allPossibleGPetoSTNconnections), totNumberOfConnection, p=probs, replace=False)
	
	# GPeSTNconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
	GPeSTNconnections=allPossibleGPetoSTNconnections[ GPeSTNconnectionsIndizes ]

	# add these connections to the return matrix 'synReversals'
	for connection in GPeSTNconnections:
		# add inhibitory connection
		# note that synReversals[i,j] is refers to connections from presynaptic neuron j to presynaptic neuron i
		synReversals[ connection[1],  connection[0] ]=-1
	
	# return result
	return synReversals






################################################################
#  function: generate_connectivity_and_weight_matrix_Ebert
# 		places neurons in elipsoidal volumns and generates the connectivity matrix 
#		according to   Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe	  
#		Also generates synaptic weight matrix with initial weights set to either 0 or one such that a given 
#		mean weight is realized
#
#		input:   system_parameters , rnd_state_for_network_generation 
#			system_parameters  .. parameter set used for simulations
#			rnd_state_for_network_generation .. state of random number generator to be used for network generation
#
#		output: synConnections , cMatrix , neuronLoc , sim_objects
#   		synConnections  ... connectivity matrix    entries 1, -1 , 0 for exc., inh. and no connections, respectively
#			cMatrix         ... scipy.sparse matrix synaptic weight matrix  entries [0,1]
#			neuronLoc       ... struct containing  locations of STN and GPe neurons
#			sim_objects		... struct containing objects related to network structure that are generated to 
#								speed up simulations
def generate_connectivity_and_weight_matrix_Ebert( system_parameters , rnd_state_for_network_generation ):

	# load needed parameters from system_parameters
	# number of STN neurons
	N_STN = system_parameters['N_STN']
	# number of GPe neurons
	N_GPe = system_parameters['N_GPe']
	# total number of neurons
	N = N_STN+ N_GPe
	# probabilityh for STN -> STN connection
	P_STN_STN = system_parameters['P_STN_STN']
	# probabilityh for STN -> GPe connection
	P_STN_GPe = system_parameters['P_STN_GPe']
	# probabilityh for GPe -> GPe connection
	P_GPe_GPe = system_parameters['P_GPe_GPe']
	# probabilityh for GPe -> STN connection
	P_GPe_STN = system_parameters['P_GPe_STN']

	# synaptic transmission delay in time steps
	StepsTauSynDelaySTNSTN=int(system_parameters['tauSynDelaySTNSTN']/system_parameters['dt']) # time steps
	StepsTauSynDelayGPeGPe=int(system_parameters['tauSynDelayGPeGPe']/system_parameters['dt']) # time steps
	StepsTauSynDelayGPeSTN=int(system_parameters['tauSynDelayGPeSTN']/system_parameters['dt']) # time steps
	StepsTauSynDelaySTNGPe=int(system_parameters['tauSynDelaySTNGPe']/system_parameters['dt']) # time steps

	# max strengths exc coupling
	cMaxExc=system_parameters['cMaxExc']	
	# mean initial strengths exc coupling
	cExcInit=system_parameters['cExcInit']

	# max inh coupling
	cMaxInh=system_parameters['cMaxInh']
	# mean initial strengths inh coupling
	cInhInit=system_parameters['cInhInit']

	# set state of random number generator
	np.random.set_state( rnd_state_for_network_generation  )

	# get connectivity matrix
	STNCenter, GPeCenter= placeNeurons( N_STN, N_GPe )


	synConnections= synReversalsCmatrix(STNCenter, GPeCenter, N_STN, N_GPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN)

	# set diagonal to zero 
	diaZero=np.ones( (N,N) )-np.diag( np.ones( N ) )
	synConnections=synConnections*diaZero

	# decouple GPe  ( this is done since we only used STN neurons in our simulations, uncomment if not needed)
	for kNeuron in range(N_STN,N):
		synConnections[:,kNeuron]=np.zeros(N)
		synConnections[kNeuron,:]=np.zeros(N)

	#########################################################################################
	#    in the following additional arrays are introduced to speed up simulations
	# get indicec post and presynaptic neurons to speed up STDP
	PostSynNeurons = {}
	PreSynNeurons = {}

	# max numbers of corresponding synapses
	maxNumberOfPostSynapticNeurons=0
	maxNumberOfPreSynapticNeurons=0

	for kNeuron in range(N_STN):

		PostSynNeurons[kNeuron]=np.nonzero( ( synConnections[:,kNeuron].astype(int) ).tolist() )[0].tolist()
		PreSynNeurons[kNeuron]=np.nonzero( ( synConnections[kNeuron,:].astype(int) ).tolist() )[0].tolist()

		# add random intra-network connections in case of no post/pre neurons
		# this is to help getting fully connected networks
		if len(PostSynNeurons[kNeuron])==0:

			# add random connection
			if kNeuron < N_STN:
				kPost=np.random.choice( range(kNeuron)+range(kNeuron+1,N_STN) )
				synConnections[kPost,kNeuron]=1


		if len(PreSynNeurons[kNeuron])==0:
			# add random connection
			# this is to help getting fully connected networks
			if kNeuron < N_STN:
				kPre=np.random.choice( range(kNeuron)+range(kNeuron+1,N_STN) )
				synConnections[kNeuron,kPre]=1

		# update max numbers of connections
		if maxNumberOfPostSynapticNeurons<len(PostSynNeurons[kNeuron]):
			maxNumberOfPostSynapticNeurons=len(PostSynNeurons[kNeuron])
		if maxNumberOfPreSynapticNeurons<len(PreSynNeurons[kNeuron]):
			maxNumberOfPreSynapticNeurons=len(PreSynNeurons[kNeuron])

	# generate numpy array with post synaptic neurons to speed up simulations ...
	numpyPostSynapticNeurons=np.full((N,maxNumberOfPostSynapticNeurons),N+1)
	numpyPreSynapticNeurons=np.full((N,maxNumberOfPreSynapticNeurons),N+1)

	# ... and corresponding matrix containing transimission delays in time steps
	transmissionDelaysPostSynNeurons=np.full((N,maxNumberOfPostSynapticNeurons),-1.0)
	transmissionDelaysPreSynNeurons=np.full((N,maxNumberOfPreSynapticNeurons),-1.0)

	# gen numpy array with post synaptic neurons
	for kPreSyn in range(N_STN):
		postSynNeuronskPre=PostSynNeurons[kPreSyn]
		for kPostSyn in range(len(postSynNeuronskPre)):
			numpyPostSynapticNeurons[kPreSyn, kPostSyn]=postSynNeuronskPre[kPostSyn]

			kPostNeuron=postSynNeuronskPre[kPostSyn]
			if (kPreSyn < N_STN):
				if (kPostNeuron < N_STN):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelaySTNSTN

				if (kPostNeuron >= N_STN) and (kPostNeuron < N):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelaySTNGPe


			if (kPreSyn >= N_STN) and (kPreSyn < N):
				if (kPostNeuron < N_STN):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelayGPeSTN

				if (kPostNeuron >= N_STN) and (kPostNeuron < N):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelayGPeGPe

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

	# synaptic weight matrix
	cMatrix=np.zeros( (N , N) )

	# initialize synaptic weights by setting random weights to zero so that the mean initial
	# synaptic weights are cExcInit/cMaxExc and cInhInit/cMaxInh for excitatory and inhibitory connections, 
	# respectively
	# mean inital weights
	if cMaxExc != 0:
		meanInitalExcWeight=cExcInit/cMaxExc
	else:
		meanInitalExcWeight=0

	if cMaxInh != 0:
		meanInitalInhWeight=cInhInit/cMaxInh
	else:
		meanInitalInhWeight=0

	# initialize excitatory connections
	P1e=meanInitalExcWeight
	P0e=1-P1e
	cMatrix[:,:N_STN]=np.random.choice([0.0,1.0],(N,N_STN),p=[P0e,P1e])

	# initialize inhibitory connections
	P1i=meanInitalInhWeight
	P0i=1-P1i
	cMatrix[:,N_STN:]=np.random.choice([0.0,1.0],(N,N_GPe),p=[P0i,P1i])

	# filter weits with actual connections according to connectivity matrix
	cMatrix=cMatrix*synConnections
	cMatrix=scipy.sparse.csc_matrix(cMatrix)
	csc_Zero=scipy.sparse.csc_matrix(np.zeros( ( N,N ) ))
	csc_Ones=scipy.sparse.csc_matrix(np.ones( ( N,N ) ))

	# output struct containing neuron positions in mm
	neuronLoc = { 'STN_center_mm' : STNCenter , 'GPe_center_mm' : GPeCenter }

	# output struct containing objects that are related to network structure but only needed during simulation
	sim_objects = { 'max_N_pre' : maxNumberOfPreSynapticNeurons ,'max_N_post' : maxNumberOfPostSynapticNeurons , 'numpyPostSynapticNeurons' : numpyPostSynapticNeurons , 'numpyPreSynapticNeurons' : numpyPreSynapticNeurons , 'td_PostSynNeurons' : transmissionDelaysPostSynNeurons , 'td_PreSynNeurons' : transmissionDelaysPreSynNeurons , 'csc_Zero' : csc_Zero , 'csc_Ones' : csc_Ones	}
	

	# return output
	return synConnections , cMatrix , neuronLoc , sim_objects



################################################################
#  function: generate_connectivity_and_weight_matrix_Ebert
# 		same as 'generate_connectivity_and_weight_matrix_Ebert' but neurons are sorted according to x coordinate
#
#		input:   system_parameters , rnd_state_for_network_generation 
#			system_parameters  .. parameter set used for simulations
#			rnd_state_for_network_generation .. state of random number generator to be used for network generation
#
#		output: synConnections , cMatrix , neuronLoc , sim_objects
#   		synConnections  ... connectivity matrix    entries 1, -1 , 0 for exc., inh. and no connections, respectively
#			cMatrix         ... scipy.sparse matrix synaptic weight matrix  entries [0,1]
#			neuronLoc       ... struct containing  locations of STN and GPe neurons
#			sim_objects		... struct containing objects related to network structure that are generated to 
#								speed up simulations
def generate_connectivity_and_weight_matrix_Ebert_sort_x( system_parameters , rnd_state_for_network_generation ):

	# load needed parameters from system_parameters
	# number of STN neurons
	N_STN = system_parameters['N_STN']
	# number of GPe neurons
	N_GPe = system_parameters['N_GPe']
	# total number of neurons
	N = N_STN+ N_GPe
	# probabilityh for STN -> STN connection
	P_STN_STN = system_parameters['P_STN_STN']
	# probabilityh for STN -> GPe connection
	P_STN_GPe = system_parameters['P_STN_GPe']
	# probabilityh for GPe -> GPe connection
	P_GPe_GPe = system_parameters['P_GPe_GPe']
	# probabilityh for GPe -> STN connection
	P_GPe_STN = system_parameters['P_GPe_STN']

	# synaptic transmission delay in time steps
	StepsTauSynDelaySTNSTN=int(system_parameters['tauSynDelaySTNSTN']/system_parameters['dt']) # time steps
	StepsTauSynDelayGPeGPe=int(system_parameters['tauSynDelayGPeGPe']/system_parameters['dt']) # time steps
	StepsTauSynDelayGPeSTN=int(system_parameters['tauSynDelayGPeSTN']/system_parameters['dt']) # time steps
	StepsTauSynDelaySTNGPe=int(system_parameters['tauSynDelaySTNGPe']/system_parameters['dt']) # time steps

	# max strengths exc coupling
	cMaxExc=system_parameters['cMaxExc']	
	# mean initial strengths exc coupling
	cExcInit=system_parameters['cExcInit']

	# max inh coupling
	cMaxInh=system_parameters['cMaxInh']
	# mean initial strengths inh coupling
	cInhInit=system_parameters['cInhInit']

	# set state of random number generator
	np.random.set_state( rnd_state_for_network_generation  )

	# get connectivity matrix
	STNCenter, GPeCenter= placeNeurons( N_STN, N_GPe )

	# sort neurons according to x-coordinate
	STNCenter = STNCenter[STNCenter[:,0].argsort()]
	GPeCenter = GPeCenter[GPeCenter[:,0].argsort()]

	synConnections= synReversalsCmatrix(STNCenter, GPeCenter, N_STN, N_GPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN)

	# set diagonal to zero 
	diaZero=np.ones( (N,N) )-np.diag( np.ones( N ) )
	synConnections=synConnections*diaZero

	# decouple GPe  ( this is done since we only used STN neurons in our simulations, uncomment if not needed)
	for kNeuron in range(N_STN,N):
		synConnections[:,kNeuron]=np.zeros(N)
		synConnections[kNeuron,:]=np.zeros(N)

	#########################################################################################
	#    in the following additional arrays are introduced to speed up simulations
	# get indicec post and presynaptic neurons to speed up STDP
	PostSynNeurons = {}
	PreSynNeurons = {}

	# max numbers of corresponding synapses
	maxNumberOfPostSynapticNeurons=0
	maxNumberOfPreSynapticNeurons=0

	for kNeuron in range(N_STN):

		PostSynNeurons[kNeuron]=np.nonzero( ( synConnections[:,kNeuron].astype(int) ).tolist() )[0].tolist()
		PreSynNeurons[kNeuron]=np.nonzero( ( synConnections[kNeuron,:].astype(int) ).tolist() )[0].tolist()

		# add random intra-network connections in case of no post/pre neurons
		# this is to help getting fully connected networks
		if len(PostSynNeurons[kNeuron])==0:

			# add random connection
			if kNeuron < N_STN:
				kPost=np.random.choice( range(kNeuron)+range(kNeuron+1,N_STN) )
				synConnections[kPost,kNeuron]=1


		if len(PreSynNeurons[kNeuron])==0:
			# add random connection
			# this is to help getting fully connected networks
			if kNeuron < N_STN:
				kPre=np.random.choice( range(kNeuron)+range(kNeuron+1,N_STN) )
				synConnections[kNeuron,kPre]=1

		# update max numbers of connections
		if maxNumberOfPostSynapticNeurons<len(PostSynNeurons[kNeuron]):
			maxNumberOfPostSynapticNeurons=len(PostSynNeurons[kNeuron])
		if maxNumberOfPreSynapticNeurons<len(PreSynNeurons[kNeuron]):
			maxNumberOfPreSynapticNeurons=len(PreSynNeurons[kNeuron])

	# generate numpy array with post synaptic neurons to speed up simulations ...
	numpyPostSynapticNeurons=np.full((N,maxNumberOfPostSynapticNeurons),N+1)
	numpyPreSynapticNeurons=np.full((N,maxNumberOfPreSynapticNeurons),N+1)

	# ... and corresponding matrix containing transimission delays in time steps
	transmissionDelaysPostSynNeurons=np.full((N,maxNumberOfPostSynapticNeurons),-1.0)
	transmissionDelaysPreSynNeurons=np.full((N,maxNumberOfPreSynapticNeurons),-1.0)

	# gen numpy array with post synaptic neurons
	for kPreSyn in range(N_STN):
		postSynNeuronskPre=PostSynNeurons[kPreSyn]
		for kPostSyn in range(len(postSynNeuronskPre)):
			numpyPostSynapticNeurons[kPreSyn, kPostSyn]=postSynNeuronskPre[kPostSyn]

			kPostNeuron=postSynNeuronskPre[kPostSyn]
			if (kPreSyn < N_STN):
				if (kPostNeuron < N_STN):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelaySTNSTN

				if (kPostNeuron >= N_STN) and (kPostNeuron < N):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelaySTNGPe


			if (kPreSyn >= N_STN) and (kPreSyn < N):
				if (kPostNeuron < N_STN):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelayGPeSTN

				if (kPostNeuron >= N_STN) and (kPostNeuron < N):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelayGPeGPe

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

	# synaptic weight matrix
	cMatrix=np.zeros( (N , N) )

	# initialize synaptic weights by setting random weights to zero so that the mean initial
	# synaptic weights are cExcInit/cMaxExc and cInhInit/cMaxInh for excitatory and inhibitory connections, 
	# respectively
	# mean inital weights
	if cMaxExc != 0:
		meanInitalExcWeight=cExcInit/cMaxExc
	else:
		meanInitalExcWeight=0

	if cMaxInh != 0:
		meanInitalInhWeight=cInhInit/cMaxInh
	else:
		meanInitalInhWeight=0

	# initialize excitatory connections
	P1e=meanInitalExcWeight
	P0e=1-P1e
	cMatrix[:,:N_STN]=np.random.choice([0.0,1.0],(N,N_STN),p=[P0e,P1e])

	# initialize inhibitory connections
	P1i=meanInitalInhWeight
	P0i=1-P1i
	cMatrix[:,N_STN:]=np.random.choice([0.0,1.0],(N,N_GPe),p=[P0i,P1i])

	# filter weits with actual connections according to connectivity matrix
	cMatrix=cMatrix*synConnections
	cMatrix=scipy.sparse.csc_matrix(cMatrix)
	csc_Zero=scipy.sparse.csc_matrix(np.zeros( ( N,N ) ))
	csc_Ones=scipy.sparse.csc_matrix(np.ones( ( N,N ) ))

	# output struct containing neuron positions in mm
	neuronLoc = { 'STN_center_mm' : STNCenter , 'GPe_center_mm' : GPeCenter }

	# output struct containing objects that are related to network structure but only needed during simulation
	sim_objects = { 'max_N_pre' : maxNumberOfPreSynapticNeurons ,'max_N_post' : maxNumberOfPostSynapticNeurons , 'numpyPostSynapticNeurons' : numpyPostSynapticNeurons , 'numpyPreSynapticNeurons' : numpyPreSynapticNeurons , 'td_PostSynNeurons' : transmissionDelaysPostSynNeurons , 'td_PreSynNeurons' : transmissionDelaysPreSynNeurons , 'csc_Zero' : csc_Zero , 'csc_Ones' : csc_Ones	}
	

	# return output
	return synConnections , cMatrix , neuronLoc , sim_objects



##################################
# function: placeNeurons_3D_cuboid
def placeNeurons_3D_cuboid( NSTN , NGPe, rx_STN, ry_STN, rz_STN, rx_GPe, ry_GPe, rz_GPe ):
#
#       Places NSTN neurons in cuboid associated with subthalamic nucleus (STN) and 
#		NGPe in cuboid associated with globus pallidus externus (GPe).
#	    axes are aligned with coordinate system.	
#
#       input: NSTN, NGPe
#           NSTN ... number of STN neurons that need to be placed
#			NGPe ... number of GPe neurons that need to be placed
#			rx ... max distance from center in x-direction
#			ry ... max distance from center in y-direction
#			rz ... max distance from center in z-direction
#       return: STNNeuronPositons, GPeNeuronPositons	
# 			STNNeuronPositons ... numpy array of STN neuron centers in 3d (mm) 
# 			GPeNeuronPositons ... numpy array of GPe neuron centers in 3d (mm) 

# 1) place STN neurons
	rx, ry, rz = rx_STN, ry_STN, rz_STN
	# (i) start with uniformly distributed positions in 3d .. 
	STNNeuronPositons=np.random.uniform([-rx, -ry, -rz], [rx, ry, rz], [NSTN, 3])
	
# 2) place GPe neurons
	rx, ry, rz = rx_GPe, ry_GPe, rz_GPe
	# (i) start with uniformly distributed positions in 3d .. 
	GPeNeuronPositons=np.random.uniform([-rx, -ry, -rz], [rx, ry, rz], [NGPe, 3])

	# return lists of neuron positions in 3d that were placed in STN and GPe volume, respectively 
	return STNNeuronPositons, GPeNeuronPositons


##################################
# function: synReversalsCmatrix
# 	Returns a 2d matrix of integers indicating which neurons are connected by an excitatory synapse (entry 1),
#   which neurons are connected by inhibitory synapses (entry -1) or which neurons are not connected (entry = 0).
#   
#	The shape of that matrix is a block matrix with dimension (NSTN+NGPe, NSTN+NGPe)
#
#	Recurrent STN and GPe connections are implemented according to a 
#	distance-dependent connection probability with exponential shape taken from Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe	
#
#	Probability for internetwork connections does not depend on the distance, as in Ebert et al. 2014
#	
#	Distance-dependent connection are randomly implemented by calculating the probability for each possible connections and then
#	drawing the desired number of connections without replacement from the pool of all possible connections.
#
#	periodic boundary conditions are applied
#
#		input:   positionsSTNNeurons, positionsGPeNeurons, NSTN, NGPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN
#			positionsSTNNeurons ... 1d np.array of STN neuron positions in 3d in units of mm
#			positionsGPeNeurons ... 1d np.array of GPe neuron positions in 3d in units of mm
#			NSTN ... total number of STN neurons
#			NGPe ... total number of GPe neurons
#			P_STN_STN ... Probability for STN -> STN connections (total number of connections is P_STN_STN * ( NSTN * NSTN ) )
#			P_STN_GPe ... Probability for STN -> GPe connections (total number of connections is P_STN_GPe * ( NSTN * NGPe ) )
#			P_GPe_GPe ... Probability for GPe -> GPe connections (total number of connections is P_GPe_GPe * ( NGPe * NGPe ) )
#			P_GPe_STN ... Probability for GPe -> STN connections (total number of connections is P_GPe_STN * ( NGPe * BSTN ) )
#			r_STN ... list containing max x,y,z distances from center for STN volume
#			r_GPe ... list containing max x,y,z distances from center for GPe volume
#
#		return:
#			synReversals ... block matrix of integers and dimension (NSTN+NGPe, NSTN+NGPe). synReversals[i,j] contains information about the 
#							 connection between presynatpic neuron j and postsynatpic neuron i
#							 synReversals[i,j] = 1 -> exc. connections from j to i
#							 synReversals[i,j] = -1 -> inh. connections from j to i
#							 synReversals[i,j] = 0 -> no connections from j to i
def synReversalsCmatrix_3D_cuboid(positionsSTNNeurons, positionsGPeNeurons, NSTN, NGPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN, r_STN, r_GPe):

	# initiallize return matrix containing synReversals[i,j]
	# 1  ... presynaptic neuron j is connected to postsynapti neuron i by excitatory synapse
	# -1 ... presynaptic neuron j is connected to postsynapti neuron i  by inhibitory synapse
	# 0 ... no connections from j to i
	synReversals=np.zeros( (NSTN+NGPe, NSTN+NGPe) ) 

	# sort neurons according to x-coordinate
	positionsSTNNeurons = positionsSTNNeurons[positionsSTNNeurons[:,0].argsort()]
	positionsGPeNeurons = positionsGPeNeurons[positionsGPeNeurons[:,0].argsort()]


	##################################
	# CONNECTIONS FOR STN -> STN
	##################################
	# distance-dependent connection probability according to Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe	

	# total number of STN -> STN connections (round is actually not necessary, just to avoid non integer input )
	totNumberOfConnection=int( np.round( P_STN_STN * NSTN * NSTN ) )

	# implement array of all possible STN -> STN connections
	# first index for pre, second for post synaptic neuron
	allPossibleSTNtoSTNconnections=cartesianProduct( np.arange(NSTN),np.arange(NSTN) )

	# calculate distances related to these connection  in units of mm
	# apply periodic boundary conditions
	vectors_connecting_neurons = positionsSTNNeurons[allPossibleSTNtoSTNconnections[:,1]]-positionsSTNNeurons[allPossibleSTNtoSTNconnections[:,0]]
	# find vectors that are too long for all directions
	# these are replaced by shortest connections under consideration of periodic boundary conditions
	for kdirection in range(3):

		neg_too_log_vectors = vectors_connecting_neurons[:,kdirection]<-r_STN[kdirection]
		vectors_connecting_neurons[ neg_too_log_vectors,kdirection ] = vectors_connecting_neurons[ neg_too_log_vectors,kdirection ] + 2.0*r_STN[kdirection]

		pos_too_log_vectors = vectors_connecting_neurons[:,kdirection]>=r_STN[kdirection]
		vectors_connecting_neurons[ pos_too_log_vectors,kdirection ] = vectors_connecting_neurons[ pos_too_log_vectors,kdirection ] - 2.0*r_STN[kdirection]

	# calculate distances related to these connection  in units of mm
	distances=np.linalg.norm( vectors_connecting_neurons , axis=1 )

	# probability densities to implement a connection with those lengths according to 
	# Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe
	# we consider the same value here. It is important that Cd_STN is significantly shorter than the dimension of the volume though
	Cd_STN = 0.5 # characteristic scale in mm
	probs=conProbability( distances, Cd_STN ) # connection probability density in 1/mm

	# exclude self connections
	indizesOfNonSelfconnections=allPossibleSTNtoSTNconnections[:,0]!=allPossibleSTNtoSTNconnections[:,1]
	probs=probs[indizesOfNonSelfconnections]	# 1/mm
	allPossibleSTNtoSTNconnections=allPossibleSTNtoSTNconnections[indizesOfNonSelfconnections]

	# normalize probabiltiy to one
	probs=1/np.sum(probs)*probs   # probs contains the probability for each connection to be selected if only a single connection was implemented
	
	# implement synaptic connections according to probabilities
	STNSTNconnectionsIndizes=np.random.choice( len(allPossibleSTNtoSTNconnections) , totNumberOfConnection, p=probs, replace=False)
	
	# STNSTNconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
	STNSTNconnections=allPossibleSTNtoSTNconnections[ STNSTNconnectionsIndizes ]

	# add these connections to the return matrix 'synReversals'
	for connection in STNSTNconnections:
		# add excitatory connection
		# note that synReversals[i,j] is refers to connections from presynaptic neuron j to presynaptic neuron i
		synReversals[ connection[1],  connection[0] ]=1


	##################################	
	# CONNECTIONS FOR STN-> GPe
	##################################
	#  the connections probability for STN -> GPe connections does not depend on the distance

	# total number of STN-> GPe connections (round is actually not necessary, just to avoid non integer input )
	totNumberOfConnection=int(np.round(P_STN_GPe*NSTN*NGPe))

	# implement array of all possible STN -> GPe connections
	# first index for pre, second for post synaptic neuron      neuron indices for STN neurons are 0 - NSTN-1 and for GPe neurons NSTN - NSTN+NGPe-1
	allPossibleSTNtoGPeconnections=cartesianProduct( np.arange(NSTN),np.arange(NSTN, NSTN+NGPe ) )


	# all STN-> GPe are implemented with the same probability
	probs=np.full( len(allPossibleSTNtoGPeconnections), 1/float(totNumberOfConnection) )

	# normalize to probs one
	probs=1/np.sum(probs)*probs # probs contains the probability for each connection to be selected if only a single connection was implemented
	
	# implement synaptic connections according to probabilities
	STNGPeConnectionsIndizes=np.random.choice(len(allPossibleSTNtoGPeconnections), totNumberOfConnection, p=probs, replace=False)
	
	# STNGPeconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
	STNGPeconnections=allPossibleSTNtoGPeconnections[ STNGPeConnectionsIndizes ]

	# add these connections to the return matrix 'synReversals'
	for connection in STNGPeconnections:
		# add excitatory connection
		# note that synReversals[i,j] is refers to connections from presynaptic neuron j to presynaptic neuron i
		synReversals[ connection[1],  connection[0] ]=1




	##################################	
	# CONNECTIONS FOR GPe-> GPe
	##################################
	# distance-dependent connection probability according to Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe	

	# total number of GPe-> GPe connections (round is actually not necessary, just to avoid non integer input )
	totNumberOfConnection=int(np.round(P_GPe_GPe*NGPe*NGPe))

	# implement array of all possible GPe -> GPe connections
	# first index for pre, second for post synaptic neuron      neuron indices for STN neurons are 0 - NSTN-1 and for GPe neurons NSTN - NSTN+NGPe-1
	allPossibleGPetoGPeconnections=cartesianProduct( np.arange( NSTN, NSTN+NGPe ) ,np. arange( NSTN, NSTN+NGPe )  )


	# calculate distances related to these connection  in units of mm
	# apply periodic boundary conditions
	vectors_connecting_neurons = positionsGPeNeurons[allPossibleGPetoGPeconnections[:,1]-NSTN]-positionsGPeNeurons[allPossibleGPetoGPeconnections[:,0]-NSTN]
	# find vectors that are too long for all directions
	# these are replaced by shortest connections under consideration of periodic boundary conditions
	for kdirection in range(3):

		neg_too_log_vectors = vectors_connecting_neurons[:,kdirection]<-r_GPe[kdirection]
		vectors_connecting_neurons[ neg_too_log_vectors,kdirection ] = vectors_connecting_neurons[ neg_too_log_vectors,kdirection ] + 2.0*r_GPe[kdirection]

		pos_too_log_vectors = vectors_connecting_neurons[:,kdirection]>=r_GPe[kdirection]
		vectors_connecting_neurons[ pos_too_log_vectors,kdirection ] = vectors_connecting_neurons[ pos_too_log_vectors,kdirection ] - 2.0*r_GPe[kdirection]


	# calculate distances related to these connection  in units of mm
	distances=np.linalg.norm( vectors_connecting_neurons , axis=1 )

	# probability densities to implement a connection with those lengths according to 
	# Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe
	# we consider the same value here. It is important that Cd_GPe is significantly shorter than the dimension of the volume though	
	Cd_GPe = 0.63 # characteristic scale in mm
	probs=conProbability(distances, Cd_GPe ) # connection probability density in 1/mm

	# exclude self connections
	indizesOfNonSelfconnections=allPossibleGPetoGPeconnections[:,0]!=allPossibleGPetoGPeconnections[:,1]
	probs=probs[indizesOfNonSelfconnections]
	allPossibleGPetoGPeconnections=allPossibleGPetoGPeconnections[indizesOfNonSelfconnections]

	# normalize to probs one
	probs=1/np.sum(probs)*probs # probs contains the probability for each connection to be selected if only a single connection was implemented
	
	# implement synaptic connections according to probabilities
	GPeGPeconnectionsIndizes=np.random.choice(len(allPossibleGPetoGPeconnections), totNumberOfConnection, p=probs, replace=False)
	
	# GPeGPeconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
	GPeGPeconnections=allPossibleGPetoGPeconnections[ GPeGPeconnectionsIndizes ]

	# add these connections to the return matrix 'synReversals'
	for connection in GPeGPeconnections:
		# add inhibitory connection
		# note that synReversals[i,j] is refers to connections from presynaptic neuron j to presynaptic neuron i
		synReversals[ connection[1],  connection[0] ]=-1


	##################################	
	# CONNECTIONS FOR GPe-> STN
	##################################
	#  the connections probability for STN -> GPe connections does not depend on the distance

	# total number of GPe-> STN connections (round is actually not necessary, just to avoid non integer input )
	totNumberOfConnection=int(np.round(P_GPe_STN*NGPe*NSTN))

	# implement array of all possible GPe -> STN connections
	# first index for pre, second for post synaptic neuron      neuron indices for STN neurons are 0 - NSTN-1 and for GPe neurons NSTN - NSTN+NGPe-1
	allPossibleGPetoSTNconnections=cartesianProduct( np.arange(NSTN, NSTN+NGPe ) ,np.arange(NSTN )  )

	# all GPe -> STN are implemented with the same probability	
	probs=np.full( len(allPossibleGPetoSTNconnections), 1/float(totNumberOfConnection) )

	# normalize to probs one
	probs=1/np.sum(probs)*probs # probs contains the probability for each connection to be selected if only a single connection was implemented

	# GPeSTNconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
	GPeSTNconnectionsIndizes=np.random.choice(len(allPossibleGPetoSTNconnections), totNumberOfConnection, p=probs, replace=False)
	
	# GPeSTNconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
	GPeSTNconnections=allPossibleGPetoSTNconnections[ GPeSTNconnectionsIndizes ]

	# add these connections to the return matrix 'synReversals'
	for connection in GPeSTNconnections:
		# add inhibitory connection
		# note that synReversals[i,j] is refers to connections from presynaptic neuron j to presynaptic neuron i
		synReversals[ connection[1],  connection[0] ]=-1
	
	# return result
	return synReversals





################################################################
#  function: generate_connectivity_and_weight_matrix for 3D cuboid volume with periodic boundary conditions
# 		places neurons in 3D cuboid volume and generates the connectivity matrix 
#		distance dependent connection probabiltiy is chosen according to   Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe	  
#		Also generates synaptic weight matrix with initial weights set to either 0 or one such that a given 
#		mean weight is realized
#
#		input:   system_parameters , rnd_state_for_network_generation 
#			system_parameters  .. parameter set used for simulations
#			rnd_state_for_network_generation .. state of random number generator to be used for network generation
#
#		output: synConnections , cMatrix , neuronLoc , sim_objects
#   		synConnections  ... connectivity matrix    entries 1, -1 , 0 for exc., inh. and no connections, respectively
#			cMatrix         ... scipy.sparse matrix synaptic weight matrix  entries [0,1]
#			neuronLoc       ... struct containing  locations of STN and GPe neurons
#			sim_objects		... struct containing objects related to network structure that are generated to 
#								speed up simulations
def generate_connectivity_and_weight_matrix_3D_cuboid( system_parameters , rnd_state_for_network_generation ):

	# load needed parameters from system_parameters
	# number of STN neurons
	N_STN = system_parameters['N_STN']
	# number of GPe neurons
	N_GPe = system_parameters['N_GPe']
	# total number of neurons
	N = N_STN+ N_GPe
	# probabilityh for STN -> STN connection
	P_STN_STN = system_parameters['P_STN_STN']
	# probabilityh for STN -> GPe connection
	P_STN_GPe = system_parameters['P_STN_GPe']
	# probabilityh for GPe -> GPe connection
	P_GPe_GPe = system_parameters['P_GPe_GPe']
	# probabilityh for GPe -> STN connection
	P_GPe_STN = system_parameters['P_GPe_STN']

	# synaptic transmission delay in time steps
	StepsTauSynDelaySTNSTN=int(system_parameters['tauSynDelaySTNSTN']/system_parameters['dt']) # time steps
	StepsTauSynDelayGPeGPe=int(system_parameters['tauSynDelayGPeGPe']/system_parameters['dt']) # time steps
	StepsTauSynDelayGPeSTN=int(system_parameters['tauSynDelayGPeSTN']/system_parameters['dt']) # time steps
	StepsTauSynDelaySTNGPe=int(system_parameters['tauSynDelaySTNGPe']/system_parameters['dt']) # time steps

	# max strengths exc coupling
	cMaxExc=system_parameters['cMaxExc']	
	# mean initial strengths exc coupling
	cExcInit=system_parameters['cExcInit']

	# max inh coupling
	cMaxInh=system_parameters['cMaxInh']
	# mean initial strengths inh coupling
	cInhInit=system_parameters['cInhInit']

	# set state of random number generator
	np.random.set_state( rnd_state_for_network_generation  )


	rx_STN =system_parameters['rx_STN']
	ry_STN =system_parameters['ry_STN']
	rz_STN =system_parameters['rz_STN']
	
	rx_GPe =system_parameters['rx_GPe']
	ry_GPe =system_parameters['ry_GPe']
	rz_GPe =system_parameters['rz_GPe']

	# get connectivity matrix
	STNCenter, GPeCenter= placeNeurons_3D_cuboid( N_STN , N_GPe , rx_STN, ry_STN, rz_STN, rx_GPe, ry_GPe, rz_GPe )

	# sort neurons according to x-coordinate
	STNCenter = STNCenter[STNCenter[:,0].argsort()]
	GPeCenter = GPeCenter[GPeCenter[:,0].argsort()]


	synConnections= synReversalsCmatrix_3D_cuboid(STNCenter, GPeCenter, N_STN, N_GPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN, [rx_STN, ry_STN, rz_STN], [rx_GPe, ry_GPe, rz_GPe])


	# set diagonal to zero 
	diaZero=np.ones( (N,N) )-np.diag( np.ones( N ) )
	synConnections=synConnections*diaZero

	# decouple GPe  ( this is done since we only used STN neurons in our simulations, uncomment if not needed)
	for kNeuron in range(N_STN,N):
		synConnections[:,kNeuron]=np.zeros(N)
		synConnections[kNeuron,:]=np.zeros(N)

	#########################################################################################
	#    in the following additional arrays are introduced to speed up simulations
	# get indicec post and presynaptic neurons to speed up STDP
	PostSynNeurons = {}
	PreSynNeurons = {}

	# max numbers of corresponding synapses
	maxNumberOfPostSynapticNeurons=0
	maxNumberOfPreSynapticNeurons=0

	for kNeuron in range(N_STN):

		PostSynNeurons[kNeuron]=np.nonzero( ( synConnections[:,kNeuron].astype(int) ).tolist() )[0].tolist()
		PreSynNeurons[kNeuron]=np.nonzero( ( synConnections[kNeuron,:].astype(int) ).tolist() )[0].tolist()

		# add random intra-network connections in case of no post/pre neurons
		# this is to help getting fully connected networks
		if len(PostSynNeurons[kNeuron])==0:

			# add random connection
			if kNeuron < N_STN:
				kPost=np.random.choice( range(kNeuron)+range(kNeuron+1,N_STN) )
				synConnections[kPost,kNeuron]=1


		if len(PreSynNeurons[kNeuron])==0:
			# add random connection
			# this is to help getting fully connected networks
			if kNeuron < N_STN:
				kPre=np.random.choice( range(kNeuron)+range(kNeuron+1,N_STN) )
				synConnections[kNeuron,kPre]=1

		# update max numbers of connections
		if maxNumberOfPostSynapticNeurons<len(PostSynNeurons[kNeuron]):
			maxNumberOfPostSynapticNeurons=len(PostSynNeurons[kNeuron])
		if maxNumberOfPreSynapticNeurons<len(PreSynNeurons[kNeuron]):
			maxNumberOfPreSynapticNeurons=len(PreSynNeurons[kNeuron])

	# generate numpy array with post synaptic neurons to speed up simulations ...
	numpyPostSynapticNeurons=np.full((N,maxNumberOfPostSynapticNeurons),N+1)
	numpyPreSynapticNeurons=np.full((N,maxNumberOfPreSynapticNeurons),N+1)

	# ... and corresponding matrix containing transimission delays in time steps
	transmissionDelaysPostSynNeurons=np.full((N,maxNumberOfPostSynapticNeurons),-1.0)
	transmissionDelaysPreSynNeurons=np.full((N,maxNumberOfPreSynapticNeurons),-1.0)

	# gen numpy array with post synaptic neurons
	for kPreSyn in range(N_STN):
		postSynNeuronskPre=PostSynNeurons[kPreSyn]
		for kPostSyn in range(len(postSynNeuronskPre)):
			numpyPostSynapticNeurons[kPreSyn, kPostSyn]=postSynNeuronskPre[kPostSyn]

			kPostNeuron=postSynNeuronskPre[kPostSyn]
			if (kPreSyn < N_STN):
				if (kPostNeuron < N_STN):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelaySTNSTN

				if (kPostNeuron >= N_STN) and (kPostNeuron < N):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelaySTNGPe


			if (kPreSyn >= N_STN) and (kPreSyn < N):
				if (kPostNeuron < N_STN):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelayGPeSTN

				if (kPostNeuron >= N_STN) and (kPostNeuron < N):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelayGPeGPe

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

	# synaptic weight matrix
	cMatrix=np.zeros( (N , N) )

	# initialize synaptic weights by setting random weights to zero so that the mean initial
	# synaptic weights are cExcInit/cMaxExc and cInhInit/cMaxInh for excitatory and inhibitory connections, 
	# respectively
	# mean inital weights
	if cMaxExc != 0:
		meanInitalExcWeight=cExcInit/cMaxExc
	else:
		meanInitalExcWeight=0

	if cMaxInh != 0:
		meanInitalInhWeight=cInhInit/cMaxInh
	else:
		meanInitalInhWeight=0

	# initialize excitatory connections
	P1e=meanInitalExcWeight
	P0e=1-P1e
	cMatrix[:,:N_STN]=np.random.choice([0.0,1.0],(N,N_STN),p=[P0e,P1e])

	# initialize inhibitory connections
	P1i=meanInitalInhWeight
	P0i=1-P1i
	cMatrix[:,N_STN:]=np.random.choice([0.0,1.0],(N,N_GPe),p=[P0i,P1i])

	# filter weits with actual connections according to connectivity matrix
	cMatrix=cMatrix*synConnections
	cMatrix=scipy.sparse.csc_matrix(cMatrix)
	csc_Zero=scipy.sparse.csc_matrix(np.zeros( ( N,N ) ))
	csc_Ones=scipy.sparse.csc_matrix(np.ones( ( N,N ) ))

	# output struct containing neuron positions in mm
	neuronLoc = { 'STN_center_mm' : STNCenter , 'GPe_center_mm' : GPeCenter }

	# output struct containing objects that are related to network structure but only needed during simulation
	sim_objects = { 'max_N_pre' : maxNumberOfPreSynapticNeurons ,'max_N_post' : maxNumberOfPostSynapticNeurons , 'numpyPostSynapticNeurons' : numpyPostSynapticNeurons , 'numpyPreSynapticNeurons' : numpyPreSynapticNeurons , 'td_PostSynNeurons' : transmissionDelaysPostSynNeurons , 'td_PreSynNeurons' : transmissionDelaysPreSynNeurons , 'csc_Zero' : csc_Zero , 'csc_Ones' : csc_Ones	}
	

	# return output
	return synConnections , cMatrix , neuronLoc , sim_objects




























#################################
# function: placeNeurons_3D_cuboid
def placeNeurons_1D( NSTN , NGPe, x_STN_min, x_STN_max, x_GPe_min, x_GPe_max ):
#
#       Places NSTN neurons in cuboid associated with subthalamic nucleus (STN) and 
#		NGPe in cuboid associated with globus pallidus externus (GPe).
#	    axes are aligned with coordinate system.	
#
#       input: NSTN, NGPe
#           NSTN ... number of STN neurons that need to be placed
#			NGPe ... number of GPe neurons that need to be placed
#			rx ... max distance from center in x-direction
#			ry ... max distance from center in y-direction
#			rz ... max distance from center in z-direction
#       return: STNNeuronPositons, GPeNeuronPositons	
# 			STNNeuronPositons ... numpy array of STN neuron centers in 3d (mm) 
# 			GPeNeuronPositons ... numpy array of GPe neuron centers in 3d (mm) 

# 1) place STN neurons
	# (i) start with uniformly distributed positions in 1d .. 
	STNNeuronPositons=np.random.uniform( x_STN_min, x_STN_max, NSTN )
	
# 2) place GPe neurons
	# (i) start with uniformly distributed positions in 1d .. 
	GPeNeuronPositons=np.random.uniform( x_GPe_min, x_GPe_max, NGPe )

	# return lists of neuron positions in 3d that were placed in STN and GPe volume, respectively 
	return STNNeuronPositons, GPeNeuronPositons


##################################
# function: synReversalsCmatrix
# 	Returns a 2d matrix of integers indicating which neurons are connected by an excitatory synapse (entry 1),
#   which neurons are connected by inhibitory synapses (entry -1) or which neurons are not connected (entry = 0).
#   
#	The shape of that matrix is a block matrix with dimension (NSTN+NGPe, NSTN+NGPe)
#
#	Recurrent STN and GPe connections are implemented according to a 
#	distance-dependent connection probability with exponential shape taken from Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe	
#
#	Probability for internetwork connections does not depend on the distance, as in Ebert et al. 2014
#	
#	Distance-dependent connection are randomly implemented by calculating the probability for each possible connections and then
#	drawing the desired number of connections without replacement from the pool of all possible connections.
#
#	periodic boundary conditions are applied
#
#		input:   positionsSTNNeurons, positionsGPeNeurons, NSTN, NGPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN
#			positionsSTNNeurons ... 1d np.array of STN neuron positions in 3d in units of mm
#			positionsGPeNeurons ... 1d np.array of GPe neuron positions in 3d in units of mm
#			NSTN ... total number of STN neurons
#			NGPe ... total number of GPe neurons
#			P_STN_STN ... Probability for STN -> STN connections (total number of connections is P_STN_STN * ( NSTN * NSTN ) )
#			P_STN_GPe ... Probability for STN -> GPe connections (total number of connections is P_STN_GPe * ( NSTN * NGPe ) )
#			P_GPe_GPe ... Probability for GPe -> GPe connections (total number of connections is P_GPe_GPe * ( NGPe * NGPe ) )
#			P_GPe_STN ... Probability for GPe -> STN connections (total number of connections is P_GPe_STN * ( NGPe * BSTN ) )
#			r_STN ... list containing max x,y,z distances from center for STN volume
#			r_GPe ... list containing max x,y,z distances from center for GPe volume
#
#		return:
#			synReversals ... block matrix of integers and dimension (NSTN+NGPe, NSTN+NGPe). synReversals[i,j] contains information about the 
#							 connection between presynatpic neuron j and postsynatpic neuron i
#							 synReversals[i,j] = 1 -> exc. connections from j to i
#							 synReversals[i,j] = -1 -> inh. connections from j to i
#							 synReversals[i,j] = 0 -> no connections from j to i
def synReversalsCmatrix_1D(positionsSTNNeurons, positionsGPeNeurons, NSTN, NGPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN ):

	# initiallize return matrix containing synReversals[i,j]
	# 1  ... presynaptic neuron j is connected to postsynapti neuron i by excitatory synapse
	# -1 ... presynaptic neuron j is connected to postsynapti neuron i  by inhibitory synapse
	# 0 ... no connections from j to i
	synReversals=np.zeros( (NSTN+NGPe, NSTN+NGPe) ) 

	# sort neurons according to x-coordinate
	positionsSTNNeurons = np.sort( positionsSTNNeurons )
	positionsGPeNeurons = np.sort( positionsGPeNeurons )

	##################################
	# CONNECTIONS FOR STN -> STN
	##################################
	# distance-dependent connection probability according to Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe	

	# total number of STN -> STN connections (round is actually not necessary, just to avoid non integer input )
	totNumberOfConnection=int( np.round( P_STN_STN * NSTN * NSTN ) )

	# implement array of all possible STN -> STN connections
	# first index for pre, second for post synaptic neuron
	allPossibleSTNtoSTNconnections=cartesianProduct( np.arange(NSTN),np.arange(NSTN) )

	# calculate distances related to these connections  in units of mm
	# apply periodic boundary conditions
	vectors_connecting_neurons = positionsSTNNeurons[allPossibleSTNtoSTNconnections[:,1]]-positionsSTNNeurons[allPossibleSTNtoSTNconnections[:,0]]

	# calculate distances related to these connections  in units of mm
	distances=np.abs( vectors_connecting_neurons )

	# probability densities to implement a connection with those lengths according to 
	# Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe
	# we consider the same value here. It is important that Cd_STN is significantly shorter than the dimension of the volume though
	Cd_STN = 0.5 # characteristic scale in mm
	probs=conProbability( distances, Cd_STN ) # connection probability density in 1/mm

	# exclude self connections
	indizesOfNonSelfconnections=allPossibleSTNtoSTNconnections[:,0]!=allPossibleSTNtoSTNconnections[:,1]
	probs=probs[indizesOfNonSelfconnections]	# 1/mm
	allPossibleSTNtoSTNconnections=allPossibleSTNtoSTNconnections[indizesOfNonSelfconnections]

	# normalize probabiltiy to one
	probs=1/np.sum(probs)*probs   # probs contains the probability for each connection to be selected if only a single connection was implemented
	
	# implement synaptic connections according to probabilities
	STNSTNconnectionsIndizes=np.random.choice( len(allPossibleSTNtoSTNconnections) , totNumberOfConnection, p=probs, replace=False)
	
	# STNSTNconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
	STNSTNconnections=allPossibleSTNtoSTNconnections[ STNSTNconnectionsIndizes ]

	# add these connections to the return matrix 'synReversals'
	for connection in STNSTNconnections:
		# add excitatory connection
		# note that synReversals[i,j] is refers to connections from presynaptic neuron j to presynaptic neuron i
		synReversals[ connection[1],  connection[0] ]=1

	##################################	
	# CONNECTIONS FOR STN-> GPe
	##################################
	#  the connections probability for STN -> GPe connections does not depend on the distance

	# total number of STN-> GPe connections (round is actually not necessary, just to avoid non integer input )
	totNumberOfConnection=int(np.round(P_STN_GPe*NSTN*NGPe))

	# implement array of all possible STN -> GPe connections
	# first index for pre, second for post synaptic neuron      neuron indices for STN neurons are 0 - NSTN-1 and for GPe neurons NSTN - NSTN+NGPe-1
	allPossibleSTNtoGPeconnections=cartesianProduct( np.arange(NSTN),np.arange(NSTN, NSTN+NGPe ) )


	# all STN-> GPe are implemented with the same probability
	probs=np.full( len(allPossibleSTNtoGPeconnections), 1/float(totNumberOfConnection) )

	# normalize to probs one
	probs=1/np.sum(probs)*probs # probs contains the probability for each connection to be selected if only a single connection was implemented
	
	# implement synaptic connections according to probabilities
	STNGPeConnectionsIndizes=np.random.choice(len(allPossibleSTNtoGPeconnections), totNumberOfConnection, p=probs, replace=False)
	
	# STNGPeconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
	STNGPeconnections=allPossibleSTNtoGPeconnections[ STNGPeConnectionsIndizes ]

	# add these connections to the return matrix 'synReversals'
	for connection in STNGPeconnections:
		# add excitatory connection
		# note that synReversals[i,j] is refers to connections from presynaptic neuron j to presynaptic neuron i
		synReversals[ connection[1],  connection[0] ]=1



	##################################	
	# CONNECTIONS FOR GPe-> GPe
	##################################
	# distance-dependent connection probability according to Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe	

	# total number of GPe-> GPe connections (round is actually not necessary, just to avoid non integer input )
	totNumberOfConnection=int(np.round(P_GPe_GPe*NGPe*NGPe))

	# implement array of all possible GPe -> GPe connections
	# first index for pre, second for post synaptic neuron      neuron indices for STN neurons are 0 - NSTN-1 and for GPe neurons NSTN - NSTN+NGPe-1
	allPossibleGPetoGPeconnections=cartesianProduct( np.arange( NSTN, NSTN+NGPe ) ,np. arange( NSTN, NSTN+NGPe )  )


	# calculate distances related to these connection  in units of mm
	# apply periodic boundary conditions
	vectors_connecting_neurons = positionsGPeNeurons[allPossibleGPetoGPeconnections[:,1]-NSTN]-positionsGPeNeurons[allPossibleGPetoGPeconnections[:,0]-NSTN]

	# calculate distances related to these connection  in units of mm
	distances=np.abs( vectors_connecting_neurons )

	# probability densities to implement a connection with those lengths according to 
	# Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe
	# we consider the same value here. It is important that Cd_GPe is significantly shorter than the dimension of the volume though	
	Cd_GPe = 0.63 # characteristic scale in mm
	probs=conProbability(distances, Cd_GPe ) # connection probability density in 1/mm

	# exclude self connections
	indizesOfNonSelfconnections=allPossibleGPetoGPeconnections[:,0]!=allPossibleGPetoGPeconnections[:,1]
	probs=probs[indizesOfNonSelfconnections]
	allPossibleGPetoGPeconnections=allPossibleGPetoGPeconnections[indizesOfNonSelfconnections]

	# normalize to probs one
	probs=1/np.sum(probs)*probs # probs contains the probability for each connection to be selected if only a single connection was implemented
	
	# implement synaptic connections according to probabilities
	GPeGPeconnectionsIndizes=np.random.choice(len(allPossibleGPetoGPeconnections), totNumberOfConnection, p=probs, replace=False)
	
	# GPeGPeconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
	GPeGPeconnections=allPossibleGPetoGPeconnections[ GPeGPeconnectionsIndizes ]

	# add these connections to the return matrix 'synReversals'
	for connection in GPeGPeconnections:
		# add inhibitory connection
		# note that synReversals[i,j] is refers to connections from presynaptic neuron j to presynaptic neuron i
		synReversals[ connection[1],  connection[0] ]=-1


	##################################	
	# CONNECTIONS FOR GPe-> STN
	##################################
	#  the connections probability for STN -> GPe connections does not depend on the distance

	# total number of GPe-> STN connections (round is actually not necessary, just to avoid non integer input )
	totNumberOfConnection=int(np.round(P_GPe_STN*NGPe*NSTN))

	# implement array of all possible GPe -> STN connections
	# first index for pre, second for post synaptic neuron      neuron indices for STN neurons are 0 - NSTN-1 and for GPe neurons NSTN - NSTN+NGPe-1
	allPossibleGPetoSTNconnections=cartesianProduct( np.arange(NSTN, NSTN+NGPe ) ,np.arange(NSTN )  )

	# all GPe -> STN are implemented with the same probability	
	probs=np.full( len(allPossibleGPetoSTNconnections), 1/float(totNumberOfConnection) )

	# normalize to probs one
	probs=1/np.sum(probs)*probs # probs contains the probability for each connection to be selected if only a single connection was implemented

	# GPeSTNconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
	GPeSTNconnectionsIndizes=np.random.choice(len(allPossibleGPetoSTNconnections), totNumberOfConnection, p=probs, replace=False)
	
	# GPeSTNconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
	GPeSTNconnections=allPossibleGPetoSTNconnections[ GPeSTNconnectionsIndizes ]

	# add these connections to the return matrix 'synReversals'
	for connection in GPeSTNconnections:
		# add inhibitory connection
		# note that synReversals[i,j] is refers to connections from presynaptic neuron j to presynaptic neuron i
		synReversals[ connection[1],  connection[0] ]=-1
	
	# return result
	return synReversals



##################################
# function: synReversalsCmatrix
#   Returns a 2d matrix of integers indicating which neurons are connected by an excitatory synapse (entry 1),
#   which neurons are connected by inhibitory synapses (entry -1) or which neurons are not connected (entry = 0).
#   
#   The shape of that matrix is a block matrix with dimension (NSTN+NGPe, NSTN+NGPe)
#
#   Recurrent STN and GPe connections are implemented according to a 
#   distance-dependent connection probability with exponential shape taken from Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe  
#
#   Probability for internetwork connections does not depend on the distance, as in Ebert et al. 2014
#   
#   Distance-dependent connection are randomly implemented by calculating the probability for each possible connections and then
#   drawing the desired number of connections without replacement from the pool of all possible connections.
#
#   periodic boundary conditions are not applied
#
#       input:   positionsSTNNeurons, positionsGPeNeurons, NSTN, NGPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN
#           positionsSTNNeurons ... 1d np.array of STN neuron positions in 3d in units of mm
#           positionsGPeNeurons ... 1d np.array of GPe neuron positions in 3d in units of mm
#           NSTN ... total number of STN neurons
#           NGPe ... total number of GPe neurons
#           P_STN_STN ... Probability for STN -> STN connections (total number of connections is P_STN_STN * ( NSTN * NSTN ) )
#           P_STN_GPe ... Probability for STN -> GPe connections (total number of connections is P_STN_GPe * ( NSTN * NGPe ) )
#           P_GPe_GPe ... Probability for GPe -> GPe connections (total number of connections is P_GPe_GPe * ( NGPe * NGPe ) )
#           P_GPe_STN ... Probability for GPe -> STN connections (total number of connections is P_GPe_STN * ( NGPe * BSTN ) )
#           r_STN ... list containing max x,y,z distances from center for STN volume
#           r_GPe ... list containing max x,y,z distances from center for GPe volume
#           Cd_STN... characteristic distance for synaptic connections
#
#       return:
#           synReversals ... block matrix of integers and dimension (NSTN+NGPe, NSTN+NGPe). synReversals[i,j] contains information about the 
#                            connection between presynatpic neuron j and postsynatpic neuron i
#                            synReversals[i,j] = 1 -> exc. connections from j to i
#                            synReversals[i,j] = -1 -> inh. connections from j to i
#                            synReversals[i,j] = 0 -> no connections from j to i
def variable_distance_synReversalsCmatrix_1D(positionsSTNNeurons, positionsGPeNeurons, NSTN, NGPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN, Cd_STN ):

	# initiallize return matrix containing synReversals[i,j]
	# 1  ... presynaptic neuron j is connected to postsynapti neuron i by excitatory synapse
	# -1 ... presynaptic neuron j is connected to postsynapti neuron i  by inhibitory synapse
	# 0 ... no connections from j to i
	synReversals=np.zeros( (NSTN+NGPe, NSTN+NGPe) ) 

	# sort neurons according to x-coordinate
	positionsSTNNeurons = np.sort( positionsSTNNeurons )
	positionsGPeNeurons = np.sort( positionsGPeNeurons )

	##################################
	# CONNECTIONS FOR STN -> STN
	##################################
	# distance-dependent connection probability according to Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe 

	# total number of STN -> STN connections (round is actually not necessary, just to avoid non integer input )
	totNumberOfConnection=int( np.round( P_STN_STN * NSTN * NSTN ) )

	if totNumberOfConnection != 0:
		# implement array of all possible STN -> STN connections
		# first index for pre, second for post synaptic neuron
		allPossibleSTNtoSTNconnections=cartesianProduct( np.arange(NSTN),np.arange(NSTN) )

		# calculate distances related to these connections  in units of mm
		# apply periodic boundary conditions
		vectors_connecting_neurons = positionsSTNNeurons[allPossibleSTNtoSTNconnections[:,1]]-positionsSTNNeurons[allPossibleSTNtoSTNconnections[:,0]]

		# calculate distances related to these connections  in units of mm
		distances=np.abs( vectors_connecting_neurons )

		# probability densities to implement a connection with those lengths according to 
		# Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe
		# we consider the same value here. It is important that Cd_STN is significantly shorter than the dimension of the volume though
		#Cd_STN = 0.5 # characteristic scale in mm
		probs=conProbability( distances, Cd_STN ) # connection probability density in 1/mm

		# exclude self connections
		indizesOfNonSelfconnections=allPossibleSTNtoSTNconnections[:,0]!=allPossibleSTNtoSTNconnections[:,1]
		probs=probs[indizesOfNonSelfconnections]    # 1/mm
		allPossibleSTNtoSTNconnections=allPossibleSTNtoSTNconnections[indizesOfNonSelfconnections]

		# normalize probabiltiy to one
		probs=1/np.sum(probs)*probs   # probs contains the probability for each connection to be selected if only a single connection was implemented
		
		# implement synaptic connections according to probabilities
		STNSTNconnectionsIndizes=np.random.choice( len(allPossibleSTNtoSTNconnections) , totNumberOfConnection, p=probs, replace=False)
		
		# STNSTNconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
		STNSTNconnections=allPossibleSTNtoSTNconnections[ STNSTNconnectionsIndizes ]

		# add these connections to the return matrix 'synReversals'
		for connection in STNSTNconnections:
			# add excitatory connection
			# note that synReversals[i,j] is refers to connections from presynaptic neuron j to presynaptic neuron i
			synReversals[ connection[1],  connection[0] ]=1

	##################################  
	# CONNECTIONS FOR STN-> GPe
	##################################
	#  the connections probability for STN -> GPe connections does not depend on the distance

	# total number of STN-> GPe connections (round is actually not necessary, just to avoid non integer input )
	totNumberOfConnection=int(np.round(P_STN_GPe*NSTN*NGPe))

	if totNumberOfConnection != 0:
		# implement array of all possible STN -> GPe connections
		# first index for pre, second for post synaptic neuron      neuron indices for STN neurons are 0 - NSTN-1 and for GPe neurons NSTN - NSTN+NGPe-1
		allPossibleSTNtoGPeconnections=cartesianProduct( np.arange(NSTN),np.arange(NSTN, NSTN+NGPe ) )


		# all STN-> GPe are implemented with the same probability
		probs=np.full( len(allPossibleSTNtoGPeconnections), 1/float(totNumberOfConnection) )

		# normalize to probs one
		probs=1/np.sum(probs)*probs # probs contains the probability for each connection to be selected if only a single connection was implemented
		
		# implement synaptic connections according to probabilities
		STNGPeConnectionsIndizes=np.random.choice(len(allPossibleSTNtoGPeconnections), totNumberOfConnection, p=probs, replace=False)
		
		# STNGPeconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
		STNGPeconnections=allPossibleSTNtoGPeconnections[ STNGPeConnectionsIndizes ]

		# add these connections to the return matrix 'synReversals'
		for connection in STNGPeconnections:
			# add excitatory connection
			# note that synReversals[i,j] is refers to connections from presynaptic neuron j to presynaptic neuron i
			synReversals[ connection[1],  connection[0] ]=1



	##################################  
	# CONNECTIONS FOR GPe-> GPe
	##################################
	# distance-dependent connection probability according to Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe 

	# total number of GPe-> GPe connections (round is actually not necessary, just to avoid non integer input )
	totNumberOfConnection=int(np.round(P_GPe_GPe*NGPe*NGPe))

	if totNumberOfConnection != 0:
		# implement array of all possible GPe -> GPe connections
		# first index for pre, second for post synaptic neuron      neuron indices for STN neurons are 0 - NSTN-1 and for GPe neurons NSTN - NSTN+NGPe-1
		allPossibleGPetoGPeconnections=cartesianProduct( np.arange( NSTN, NSTN+NGPe ) ,np. arange( NSTN, NSTN+NGPe )  )


		# calculate distances related to these connection  in units of mm
		# apply periodic boundary conditions
		vectors_connecting_neurons = positionsGPeNeurons[allPossibleGPetoGPeconnections[:,1]-NSTN]-positionsGPeNeurons[allPossibleGPetoGPeconnections[:,0]-NSTN]

		# calculate distances related to these connection  in units of mm
		distances=np.abs( vectors_connecting_neurons )

		# probability densities to implement a connection with those lengths according to 
		# Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe
		# we consider the same value here. It is important that Cd_GPe is significantly shorter than the dimension of the volume though 
		Cd_GPe = 0.63 # characteristic scale in mm
		probs=conProbability(distances, Cd_GPe ) # connection probability density in 1/mm

		# exclude self connections
		indizesOfNonSelfconnections=allPossibleGPetoGPeconnections[:,0]!=allPossibleGPetoGPeconnections[:,1]
		probs=probs[indizesOfNonSelfconnections]
		allPossibleGPetoGPeconnections=allPossibleGPetoGPeconnections[indizesOfNonSelfconnections]

		# normalize to probs one
		probs=1/np.sum(probs)*probs # probs contains the probability for each connection to be selected if only a single connection was implemented
		
		# implement synaptic connections according to probabilities
		GPeGPeconnectionsIndizes=np.random.choice(len(allPossibleGPetoGPeconnections), totNumberOfConnection, p=probs, replace=False)
		
		# GPeGPeconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
		GPeGPeconnections=allPossibleGPetoGPeconnections[ GPeGPeconnectionsIndizes ]

		# add these connections to the return matrix 'synReversals'
		for connection in GPeGPeconnections:
			# add inhibitory connection
			# note that synReversals[i,j] is refers to connections from presynaptic neuron j to presynaptic neuron i
			synReversals[ connection[1],  connection[0] ]=-1


	##################################  
	# CONNECTIONS FOR GPe-> STN
	##################################
	#  the connections probability for STN -> GPe connections does not depend on the distance

	# total number of GPe-> STN connections (round is actually not necessary, just to avoid non integer input )
	totNumberOfConnection=int(np.round(P_GPe_STN*NGPe*NSTN))

	if totNumberOfConnection != 0:
		# implement array of all possible GPe -> STN connections
		# first index for pre, second for post synaptic neuron      neuron indices for STN neurons are 0 - NSTN-1 and for GPe neurons NSTN - NSTN+NGPe-1
		allPossibleGPetoSTNconnections=cartesianProduct( np.arange(NSTN, NSTN+NGPe ) ,np.arange(NSTN )  )

		# all GPe -> STN are implemented with the same probability  
		probs=np.full( len(allPossibleGPetoSTNconnections), 1/float(totNumberOfConnection) )

		# normalize to probs one
		probs=1/np.sum(probs)*probs # probs contains the probability for each connection to be selected if only a single connection was implemented

		# GPeSTNconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
		GPeSTNconnectionsIndizes=np.random.choice(len(allPossibleGPetoSTNconnections), totNumberOfConnection, p=probs, replace=False)
		
		# GPeSTNconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
		GPeSTNconnections=allPossibleGPetoSTNconnections[ GPeSTNconnectionsIndizes ]

		# add these connections to the return matrix 'synReversals'
		for connection in GPeSTNconnections:
			# add inhibitory connection
			# note that synReversals[i,j] is refers to connections from presynaptic neuron j to presynaptic neuron i
			synReversals[ connection[1],  connection[0] ]=-1
		
	# return result
	return synReversals


def circular_network_synReversalsCmatrix_1D(positionsSTNNeurons, positionsGPeNeurons, NSTN, NGPe, Pcon, M ):

	# initiallize return matrix containing synReversals[i,j]
	# 1  ... presynaptic neuron j is connected to postsynapti neuron i by excitatory synapse
	# -1 ... presynaptic neuron j is connected to postsynapti neuron i  by inhibitory synapse
	# 0 ... no connections from j to i
	synReversals=np.zeros( (NSTN+NGPe, NSTN+NGPe) ) 

	# sort neurons according to x-coordinate
	positionsSTNNeurons = np.sort( positionsSTNNeurons )
	positionsGPeNeurons = np.sort( positionsGPeNeurons )

	# randomly fill matrix
	PopulationSize = int( float(NSTN)/float(M) )
	synReversals=np.zeros( (NSTN+NGPe, NSTN+NGPe) ) 
	synReversals[:NSTN,:NSTN] = np.random.choice( [0,1], (NSTN,NSTN), p=[1-Pcon,Pcon] )
	synReversals[PopulationSize:3*PopulationSize,:PopulationSize] = 0
	synReversals[2*PopulationSize:4*PopulationSize,PopulationSize:2*PopulationSize] = 0
	synReversals[3*PopulationSize:4*PopulationSize,2*PopulationSize:3*PopulationSize] = 0
	synReversals[:PopulationSize,3*PopulationSize:4*PopulationSize] = 0
	# # the total number of connections
	# totNumberOfConnection=int( np.round( Pcon * NSTN * NSTN ) )

	# set diagonal to zero
	np.fill_diagonal(synReversals, 0)

	# return result
	return synReversals


def circular_network_2_synReversalsCmatrix_1D(positionsSTNNeurons, positionsGPeNeurons, NSTN, NGPe, Pcon, M ):

	# initiallize return matrix containing synReversals[i,j]
	# 1  ... presynaptic neuron j is connected to postsynapti neuron i by excitatory synapse
	# -1 ... presynaptic neuron j is connected to postsynapti neuron i  by inhibitory synapse
	# 0 ... no connections from j to i
	synReversals=np.zeros( (NSTN+NGPe, NSTN+NGPe) ) 

	# sort neurons according to x-coordinate
	positionsSTNNeurons = np.sort( positionsSTNNeurons )
	positionsGPeNeurons = np.sort( positionsGPeNeurons )

	# randomly fill matrix
	PopulationSize = int( float(NSTN)/float(M) )
	synReversals=np.zeros( (NSTN+NGPe, NSTN+NGPe) ) 
	synReversals[:NSTN,:NSTN] = np.random.choice( [0,1], (NSTN,NSTN), p=[1-Pcon,Pcon] )
	synReversals[PopulationSize:3*PopulationSize,:PopulationSize] = 0
	synReversals[2*PopulationSize:4*PopulationSize,PopulationSize:2*PopulationSize] = 0
	synReversals[3*PopulationSize:4*PopulationSize,2*PopulationSize:3*PopulationSize] = 0
	synReversals[:PopulationSize,3*PopulationSize:4*PopulationSize] = 0
	synReversals[:PopulationSize,2*PopulationSize:3*PopulationSize] = 0
	synReversals[PopulationSize:2*PopulationSize,3*PopulationSize:4*PopulationSize] = 0
	# # the total number of connections
	# totNumberOfConnection=int( np.round( Pcon * NSTN * NSTN ) )

	# set diagonal to zero
	np.fill_diagonal(synReversals, 0)

	# return result
	return synReversals


##################################
# function: synReversalsCmatrix
# 	Returns a 2d matrix of integers indicating which neurons are connected by an excitatory synapse (entry 1),
#   which neurons are connected by inhibitory synapses (entry -1) or which neurons are not connected (entry = 0).
#   
#	The shape of that matrix is a block matrix with dimension (NSTN+NGPe, NSTN+NGPe)
#
#	Recurrent STN and GPe connections are implemented according to a 
#	distance-dependent connection probability with exponential shape taken from Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe	
#
#	Probability for internetwork connections does not depend on the distance, as in Ebert et al. 2014
#	
#	Distance-dependent connection are randomly implemented by calculating the probability for each possible connections and then
#	drawing the desired number of connections without replacement from the pool of all possible connections.
#
#	periodic boundary conditions are applied
#
#		input:   positionsSTNNeurons, positionsGPeNeurons, NSTN, NGPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN
#			positionsSTNNeurons ... 1d np.array of STN neuron positions in 3d in units of mm
#			positionsGPeNeurons ... 1d np.array of GPe neuron positions in 3d in units of mm
#			NSTN ... total number of STN neurons
#			NGPe ... total number of GPe neurons
#			P_STN_STN ... Probability for STN -> STN connections (total number of connections is P_STN_STN * ( NSTN * NSTN ) )
#			P_STN_GPe ... Probability for STN -> GPe connections (total number of connections is P_STN_GPe * ( NSTN * NGPe ) )
#			P_GPe_GPe ... Probability for GPe -> GPe connections (total number of connections is P_GPe_GPe * ( NGPe * NGPe ) )
#			P_GPe_STN ... Probability for GPe -> STN connections (total number of connections is P_GPe_STN * ( NGPe * BSTN ) )
#			r_STN ... list containing max x,y,z distances from center for STN volume
#			r_GPe ... list containing max x,y,z distances from center for GPe volume
#
#		return:
#			synReversals ... block matrix of integers and dimension (NSTN+NGPe, NSTN+NGPe). synReversals[i,j] contains information about the 
#							 connection between presynatpic neuron j and postsynatpic neuron i
#							 synReversals[i,j] = 1 -> exc. connections from j to i
#							 synReversals[i,j] = -1 -> inh. connections from j to i
#							 synReversals[i,j] = 0 -> no connections from j to i
def synReversalsCmatrix_homogeneous(NSTN, NGPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN ):

	# initiallize return matrix containing synReversals[i,j]
	# 1  ... presynaptic neuron j is connected to postsynapti neuron i by excitatory synapse
	# -1 ... presynaptic neuron j is connected to postsynapti neuron i  by inhibitory synapse
	# 0 ... no connections from j to i
	synReversals=np.zeros( (NSTN+NGPe, NSTN+NGPe) ) 

	##################################
	# CONNECTIONS FOR STN -> STN
	##################################

	# total number of STN -> STN connections (round is actually not necessary, just to avoid non integer input )
	totNumberOfConnection=int( np.round( P_STN_STN * NSTN * NSTN ) )

	# implement array of all possible STN -> STN connections
	# first index for pre, second for post synaptic neuron
	allPossibleSTNtoSTNconnections=cartesianProduct( np.arange(NSTN),np.arange(NSTN) )

	# same probabilty for all connections
	probs = np.ones( len(allPossibleSTNtoSTNconnections) )

	# exclude self connections
	indizesOfNonSelfconnections=allPossibleSTNtoSTNconnections[:,0]!=allPossibleSTNtoSTNconnections[:,1]
	probs=probs[indizesOfNonSelfconnections]	# 1/mm
	allPossibleSTNtoSTNconnections=allPossibleSTNtoSTNconnections[indizesOfNonSelfconnections]

	# normalize probabiltiy to one
	probs=1/np.sum(probs)*probs   # probs contains the probability for each connection to be selected if only a single connection was implemented
	
	# implement synaptic connections according to probabilities
	STNSTNconnectionsIndizes=np.random.choice( len(allPossibleSTNtoSTNconnections) , totNumberOfConnection, p=probs, replace=False)
	
	# STNSTNconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
	STNSTNconnections=allPossibleSTNtoSTNconnections[ STNSTNconnectionsIndizes ]

	# add these connections to the return matrix 'synReversals'
	for connection in STNSTNconnections:
		# add excitatory connection
		# note that synReversals[i,j] is refers to connections from presynaptic neuron j to presynaptic neuron i
		synReversals[ connection[1],  connection[0] ]=1

	##################################	
	# CONNECTIONS FOR STN-> GPe
	##################################
	#  the connections probability for STN -> GPe connections does not depend on the distance

	# total number of STN-> GPe connections (round is actually not necessary, just to avoid non integer input )
	totNumberOfConnection=int(np.round(P_STN_GPe*NSTN*NGPe))

	# implement array of all possible STN -> GPe connections
	# first index for pre, second for post synaptic neuron      neuron indices for STN neurons are 0 - NSTN-1 and for GPe neurons NSTN - NSTN+NGPe-1
	allPossibleSTNtoGPeconnections=cartesianProduct( np.arange(NSTN),np.arange(NSTN, NSTN+NGPe ) )

	# same probabilty for all connections
	probs = np.ones( len(allPossibleSTNtoGPeconnections) )
	# normalize probabiltiy to one
	probs=1/np.sum(probs)*probs   # probs contains the probability for each connection to be selected if only a single connection was implemented
	
	# implement synaptic connections according to probabilities
	STNGPeConnectionsIndizes=np.random.choice(len(allPossibleSTNtoGPeconnections), totNumberOfConnection, p=probs, replace=False)
	
	# STNGPeconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
	STNGPeconnections=allPossibleSTNtoGPeconnections[ STNGPeConnectionsIndizes ]

	# add these connections to the return matrix 'synReversals'
	for connection in STNGPeconnections:
		# add excitatory connection
		# note that synReversals[i,j] is refers to connections from presynaptic neuron j to presynaptic neuron i
		synReversals[ connection[1],  connection[0] ]=1



	##################################	
	# CONNECTIONS FOR GPe-> GPe
	##################################

	# total number of GPe-> GPe connections (round is actually not necessary, just to avoid non integer input )
	totNumberOfConnection=int(np.round(P_GPe_GPe*NGPe*NGPe))

	# implement array of all possible GPe -> GPe connections
	# first index for pre, second for post synaptic neuron      neuron indices for STN neurons are 0 - NSTN-1 and for GPe neurons NSTN - NSTN+NGPe-1
	allPossibleGPetoGPeconnections=cartesianProduct( np.arange( NSTN, NSTN+NGPe ) ,np. arange( NSTN, NSTN+NGPe )  )

	# same probabilty for all connections
	probs = np.ones( len(allPossibleGPetoGPeconnections) )

	# exclude self connections
	indizesOfNonSelfconnections=allPossibleGPetoGPeconnections[:,0]!=allPossibleGPetoGPeconnections[:,1]
	probs=probs[indizesOfNonSelfconnections]
	allPossibleGPetoGPeconnections=allPossibleGPetoGPeconnections[indizesOfNonSelfconnections]

	# normalize to probs one
	probs=1/np.sum(probs)*probs # probs contains the probability for each connection to be selected if only a single connection was implemented
	
	# implement synaptic connections according to probabilities
	GPeGPeconnectionsIndizes=np.random.choice(len(allPossibleGPetoGPeconnections), totNumberOfConnection, p=probs, replace=False)
	
	# GPeGPeconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
	GPeGPeconnections=allPossibleGPetoGPeconnections[ GPeGPeconnectionsIndizes ]

	# add these connections to the return matrix 'synReversals'
	for connection in GPeGPeconnections:
		# add inhibitory connection
		# note that synReversals[i,j] is refers to connections from presynaptic neuron j to presynaptic neuron i
		synReversals[ connection[1],  connection[0] ]=-1


	##################################	
	# CONNECTIONS FOR GPe-> STN
	##################################
	#  the connections probability for STN -> GPe connections does not depend on the distance

	# total number of GPe-> STN connections (round is actually not necessary, just to avoid non integer input )
	totNumberOfConnection=int(np.round(P_GPe_STN*NGPe*NSTN))

	# implement array of all possible GPe -> STN connections
	# first index for pre, second for post synaptic neuron      neuron indices for STN neurons are 0 - NSTN-1 and for GPe neurons NSTN - NSTN+NGPe-1
	allPossibleGPetoSTNconnections=cartesianProduct( np.arange(NSTN, NSTN+NGPe ) ,np.arange(NSTN )  )

	# same probabilty for all connections
	probs = np.ones( len(allPossibleGPetoSTNconnections) )

	# normalize to probs one
	probs=1/np.sum(probs)*probs # probs contains the probability for each connection to be selected if only a single connection was implemented

	# GPeSTNconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
	GPeSTNconnectionsIndizes=np.random.choice(len(allPossibleGPetoSTNconnections), totNumberOfConnection, p=probs, replace=False)
	
	# GPeSTNconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
	GPeSTNconnections=allPossibleGPetoSTNconnections[ GPeSTNconnectionsIndizes ]

	# add these connections to the return matrix 'synReversals'
	for connection in GPeSTNconnections:
		# add inhibitory connection
		# note that synReversals[i,j] is refers to connections from presynaptic neuron j to presynaptic neuron i
		synReversals[ connection[1],  connection[0] ]=-1
	
	# return result
	return synReversals


################################################################
#  function: generate_connectivity_and_weight_matrix_homogeneous for 3D cuboid volume with periodic boundary conditions
# 		places neurons in 3D cuboid volume and generates the connectivity matrix 
#		distance dependent connection probabiltiy is chosen according to   Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe	  
#		Also generates synaptic weight matrix with initial weights set to either 0 or one such that a given 
#		mean weight is realized
#
#		input:   system_parameters , rnd_state_for_network_generation 
#			system_parameters  .. parameter set used for simulations
#			rnd_state_for_network_generation .. state of random number generator to be used for network generation
#
#		output: synConnections , cMatrix , neuronLoc , sim_objects
#   		synConnections  ... connectivity matrix    entries 1, -1 , 0 for exc., inh. and no connections, respectively
#			cMatrix         ... scipy.sparse matrix synaptic weight matrix  entries [0,1]
#			neuronLoc       ... struct containing  locations of STN and GPe neurons
#			sim_objects		... struct containing objects related to network structure that are generated to 
#								speed up simulations
def generate_connectivity_and_weight_matrix_homogeneous( system_parameters , rnd_state_for_network_generation ):

	# load needed parameters from system_parameters
	# number of STN neurons
	N_STN = system_parameters['N_STN']
	# number of GPe neurons
	N_GPe = system_parameters['N_GPe']
	# total number of neurons
	N = N_STN+ N_GPe
	# probabilityh for STN -> STN connection
	P_STN_STN = system_parameters['P_STN_STN']
	# probabilityh for STN -> GPe connection
	P_STN_GPe = system_parameters['P_STN_GPe']
	# probabilityh for GPe -> GPe connection
	P_GPe_GPe = system_parameters['P_GPe_GPe']
	# probabilityh for GPe -> STN connection
	P_GPe_STN = system_parameters['P_GPe_STN']


	# synaptic transmission delay in time steps
	StepsTauSynDelaySTNSTN=int(system_parameters['tauSynDelaySTNSTN']/system_parameters['dt']) # time steps
	StepsTauSynDelayGPeGPe=int(system_parameters['tauSynDelayGPeGPe']/system_parameters['dt']) # time steps
	StepsTauSynDelayGPeSTN=int(system_parameters['tauSynDelayGPeSTN']/system_parameters['dt']) # time steps
	StepsTauSynDelaySTNGPe=int(system_parameters['tauSynDelaySTNGPe']/system_parameters['dt']) # time steps

	# max strengths exc coupling
	cMaxExc=system_parameters['cMaxExc']	
	# mean initial strengths exc coupling
	cExcInit=system_parameters['cExcInit']

	# max inh coupling
	cMaxInh=system_parameters['cMaxInh']
	# mean initial strengths inh coupling
	cInhInit=system_parameters['cInhInit']

	# set state of random number generator
	np.random.set_state( rnd_state_for_network_generation  )

	# sort neurons according to x-coordinate
	STNCenter = np.zeros( N_STN )
	GPeCenter = np.zeros( N_GPe )

	synConnections= synReversalsCmatrix_homogeneous(N_STN, N_GPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN )


	# set diagonal to zero 
	diaZero=np.ones( (N,N) )-np.diag( np.ones( N ) )
	synConnections=synConnections*diaZero

	# decouple GPe  ( this is done since we only used STN neurons in our simulations, uncomment if not needed)
	for kNeuron in range(N_STN,N):
		synConnections[:,kNeuron]=np.zeros(N)
		synConnections[kNeuron,:]=np.zeros(N)

	#########################################################################################
	#    in the following additional arrays are introduced to speed up simulations
	# get indicec post and presynaptic neurons to speed up STDP
	PostSynNeurons = {}
	PreSynNeurons = {}

	# max numbers of corresponding synapses
	maxNumberOfPostSynapticNeurons=0
	maxNumberOfPreSynapticNeurons=0

	for kNeuron in range(N_STN):

		PostSynNeurons[kNeuron]=np.nonzero( ( synConnections[:,kNeuron].astype(int) ).tolist() )[0].tolist()
		PreSynNeurons[kNeuron]=np.nonzero( ( synConnections[kNeuron,:].astype(int) ).tolist() )[0].tolist()

		# add random intra-network connections in case of no post/pre neurons
		# this is to help getting fully connected networks
		if len(PostSynNeurons[kNeuron])==0:

			# add random connection
			if kNeuron < N_STN:
				kPost=np.random.choice( range(kNeuron)+range(kNeuron+1,N_STN) )
				synConnections[kPost,kNeuron]=1


		if len(PreSynNeurons[kNeuron])==0:
			# add random connection
			# this is to help getting fully connected networks
			if kNeuron < N_STN:
				kPre=np.random.choice( range(kNeuron)+range(kNeuron+1,N_STN) )
				synConnections[kNeuron,kPre]=1

		# update max numbers of connections
		if maxNumberOfPostSynapticNeurons<len(PostSynNeurons[kNeuron]):
			maxNumberOfPostSynapticNeurons=len(PostSynNeurons[kNeuron])
		if maxNumberOfPreSynapticNeurons<len(PreSynNeurons[kNeuron]):
			maxNumberOfPreSynapticNeurons=len(PreSynNeurons[kNeuron])

	# generate numpy array with post synaptic neurons to speed up simulations ...
	numpyPostSynapticNeurons=np.full((N,maxNumberOfPostSynapticNeurons),N+1)
	numpyPreSynapticNeurons=np.full((N,maxNumberOfPreSynapticNeurons),N+1)

	# ... and corresponding matrix containing transimission delays in time steps
	transmissionDelaysPostSynNeurons=np.full((N,maxNumberOfPostSynapticNeurons),-1.0)
	transmissionDelaysPreSynNeurons=np.full((N,maxNumberOfPreSynapticNeurons),-1.0)

	# gen numpy array with post synaptic neurons
	for kPreSyn in range(N_STN):
		postSynNeuronskPre=PostSynNeurons[kPreSyn]
		for kPostSyn in range(len(postSynNeuronskPre)):
			numpyPostSynapticNeurons[kPreSyn, kPostSyn]=postSynNeuronskPre[kPostSyn]

			kPostNeuron=postSynNeuronskPre[kPostSyn]
			if (kPreSyn < N_STN):
				if (kPostNeuron < N_STN):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelaySTNSTN

				if (kPostNeuron >= N_STN) and (kPostNeuron < N):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelaySTNGPe


			if (kPreSyn >= N_STN) and (kPreSyn < N):
				if (kPostNeuron < N_STN):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelayGPeSTN

				if (kPostNeuron >= N_STN) and (kPostNeuron < N):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelayGPeGPe

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

	# synaptic weight matrix
	cMatrix=np.zeros( (N , N) )

	# initialize synaptic weights by setting random weights to zero so that the mean initial
	# synaptic weights are cExcInit/cMaxExc and cInhInit/cMaxInh for excitatory and inhibitory connections, 
	# respectively
	# mean inital weights
	if cMaxExc != 0:
		meanInitalExcWeight=cExcInit/cMaxExc
	else:
		meanInitalExcWeight=0

	if cMaxInh != 0:
		meanInitalInhWeight=cInhInit/cMaxInh
	else:
		meanInitalInhWeight=0

	# initialize excitatory connections
	P1e=meanInitalExcWeight
	P0e=1-P1e
	cMatrix[:,:N_STN]=np.random.choice([0.0,1.0],(N,N_STN),p=[P0e,P1e])

	# initialize inhibitory connections
	P1i=meanInitalInhWeight
	P0i=1-P1i
	cMatrix[:,N_STN:]=np.random.choice([0.0,1.0],(N,N_GPe),p=[P0i,P1i])

	# filter weits with actual connections according to connectivity matrix
	cMatrix=cMatrix*synConnections
	cMatrix=scipy.sparse.csc_matrix(cMatrix)
	csc_Zero=scipy.sparse.csc_matrix(np.zeros( ( N,N ) ))
	csc_Ones=scipy.sparse.csc_matrix(np.ones( ( N,N ) ))

	# output struct containing neuron positions in mm
	neuronLoc = { 'STN_center_mm' : STNCenter , 'GPe_center_mm' : GPeCenter }

	# output struct containing objects that are related to network structure but only needed during simulation
	sim_objects = { 'max_N_pre' : maxNumberOfPreSynapticNeurons ,'max_N_post' : maxNumberOfPostSynapticNeurons , 'numpyPostSynapticNeurons' : numpyPostSynapticNeurons , 'numpyPreSynapticNeurons' : numpyPreSynapticNeurons , 'td_PostSynNeurons' : transmissionDelaysPostSynNeurons , 'td_PreSynNeurons' : transmissionDelaysPreSynNeurons , 'csc_Zero' : csc_Zero , 'csc_Ones' : csc_Ones	}
	

	# return output
	return synConnections , cMatrix , neuronLoc , sim_objects



################################################################
#  function: generate_connectivity_and_weight_matrix for 3D cuboid volume with periodic boundary conditions
# 		places neurons in 3D cuboid volume and generates the connectivity matrix 
#		distance dependent connection probabiltiy is chosen according to   Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe	  
#		Also generates synaptic weight matrix with initial weights set to either 0 or one such that a given 
#		mean weight is realized
#
#		input:   system_parameters , rnd_state_for_network_generation 
#			system_parameters  .. parameter set used for simulations
#			rnd_state_for_network_generation .. state of random number generator to be used for network generation
#
#		output: synConnections , cMatrix , neuronLoc , sim_objects
#   		synConnections  ... connectivity matrix    entries 1, -1 , 0 for exc., inh. and no connections, respectively
#			cMatrix         ... scipy.sparse matrix synaptic weight matrix  entries [0,1]
#			neuronLoc       ... struct containing  locations of STN and GPe neurons
#			sim_objects		... struct containing objects related to network structure that are generated to 
#								speed up simulations
def generate_connectivity_and_weight_matrix_1D( system_parameters , rnd_state_for_network_generation ):

	# load needed parameters from system_parameters
	# number of STN neurons
	N_STN = system_parameters['N_STN']
	# number of GPe neurons
	N_GPe = system_parameters['N_GPe']
	# total number of neurons
	N = N_STN+ N_GPe
	# probabilityh for STN -> STN connection
	P_STN_STN = system_parameters['P_STN_STN']
	# probabilityh for STN -> GPe connection
	P_STN_GPe = system_parameters['P_STN_GPe']
	# probabilityh for GPe -> GPe connection
	P_GPe_GPe = system_parameters['P_GPe_GPe']
	# probabilityh for GPe -> STN connection
	P_GPe_STN = system_parameters['P_GPe_STN']

	# synaptic transmission delay in time steps
	StepsTauSynDelaySTNSTN=int(system_parameters['tauSynDelaySTNSTN']/system_parameters['dt']) # time steps
	StepsTauSynDelayGPeGPe=int(system_parameters['tauSynDelayGPeGPe']/system_parameters['dt']) # time steps
	StepsTauSynDelayGPeSTN=int(system_parameters['tauSynDelayGPeSTN']/system_parameters['dt']) # time steps
	StepsTauSynDelaySTNGPe=int(system_parameters['tauSynDelaySTNGPe']/system_parameters['dt']) # time steps

	# max strengths exc coupling
	cMaxExc=system_parameters['cMaxExc']	
	# mean initial strengths exc coupling
	cExcInit=system_parameters['cExcInit']

	# max inh coupling
	cMaxInh=system_parameters['cMaxInh']
	# mean initial strengths inh coupling
	cInhInit=system_parameters['cInhInit']

	# set state of random number generator
	np.random.set_state( rnd_state_for_network_generation  )


	x_STN_min =system_parameters['x_STN_min']
	x_STN_max =system_parameters['x_STN_max']
	
	x_GPe_min =system_parameters['x_GPe_min']
	x_GPe_max =system_parameters['x_GPe_max']

	# get connectivity matrix
	STNCenter, GPeCenter= placeNeurons_1D( N_STN , N_GPe, x_STN_min, x_STN_max, x_GPe_min, x_GPe_max )

	# sort neurons according to x-coordinate
	STNCenter = np.sort( STNCenter )
	GPeCenter = np.sort( GPeCenter )

	synConnections= synReversalsCmatrix_1D(STNCenter, GPeCenter, N_STN, N_GPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN )


	# set diagonal to zero 
	diaZero=np.ones( (N,N) )-np.diag( np.ones( N ) )
	synConnections=synConnections*diaZero

	# decouple GPe  ( this is done since we only used STN neurons in our simulations, uncomment if not needed)
	for kNeuron in range(N_STN,N):
		synConnections[:,kNeuron]=np.zeros(N)
		synConnections[kNeuron,:]=np.zeros(N)

	#########################################################################################
	#    in the following additional arrays are introduced to speed up simulations
	# get indicec post and presynaptic neurons to speed up STDP
	PostSynNeurons = {}
	PreSynNeurons = {}

	# max numbers of corresponding synapses
	maxNumberOfPostSynapticNeurons=0
	maxNumberOfPreSynapticNeurons=0

	for kNeuron in range(N_STN):

		PostSynNeurons[kNeuron]=np.nonzero( ( synConnections[:,kNeuron].astype(int) ).tolist() )[0].tolist()
		PreSynNeurons[kNeuron]=np.nonzero( ( synConnections[kNeuron,:].astype(int) ).tolist() )[0].tolist()

		# add random intra-network connections in case of no post/pre neurons
		# this is to help getting fully connected networks
		if len(PostSynNeurons[kNeuron])==0:

			# add random connection
			if kNeuron < N_STN:
				kPost=np.random.choice( range(kNeuron)+range(kNeuron+1,N_STN) )
				synConnections[kPost,kNeuron]=1


		if len(PreSynNeurons[kNeuron])==0:
			# add random connection
			# this is to help getting fully connected networks
			if kNeuron < N_STN:
				kPre=np.random.choice( range(kNeuron)+range(kNeuron+1,N_STN) )
				synConnections[kNeuron,kPre]=1

		# update max numbers of connections
		if maxNumberOfPostSynapticNeurons<len(PostSynNeurons[kNeuron]):
			maxNumberOfPostSynapticNeurons=len(PostSynNeurons[kNeuron])
		if maxNumberOfPreSynapticNeurons<len(PreSynNeurons[kNeuron]):
			maxNumberOfPreSynapticNeurons=len(PreSynNeurons[kNeuron])

	# generate numpy array with post synaptic neurons to speed up simulations ...
	numpyPostSynapticNeurons=np.full((N,maxNumberOfPostSynapticNeurons),N+1)
	numpyPreSynapticNeurons=np.full((N,maxNumberOfPreSynapticNeurons),N+1)

	# ... and corresponding matrix containing transimission delays in time steps
	transmissionDelaysPostSynNeurons=np.full((N,maxNumberOfPostSynapticNeurons),-1.0)
	transmissionDelaysPreSynNeurons=np.full((N,maxNumberOfPreSynapticNeurons),-1.0)

	# gen numpy array with post synaptic neurons
	for kPreSyn in range(N_STN):
		postSynNeuronskPre=PostSynNeurons[kPreSyn]
		for kPostSyn in range(len(postSynNeuronskPre)):
			numpyPostSynapticNeurons[kPreSyn, kPostSyn]=postSynNeuronskPre[kPostSyn]

			kPostNeuron=postSynNeuronskPre[kPostSyn]
			if (kPreSyn < N_STN):
				if (kPostNeuron < N_STN):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelaySTNSTN

				if (kPostNeuron >= N_STN) and (kPostNeuron < N):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelaySTNGPe


			if (kPreSyn >= N_STN) and (kPreSyn < N):
				if (kPostNeuron < N_STN):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelayGPeSTN

				if (kPostNeuron >= N_STN) and (kPostNeuron < N):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelayGPeGPe

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

	# synaptic weight matrix
	cMatrix=np.zeros( (N , N) )

	# initialize synaptic weights by setting random weights to zero so that the mean initial
	# synaptic weights are cExcInit/cMaxExc and cInhInit/cMaxInh for excitatory and inhibitory connections, 
	# respectively
	# mean inital weights
	if cMaxExc != 0:
		meanInitalExcWeight=cExcInit/cMaxExc
	else:
		meanInitalExcWeight=0

	if cMaxInh != 0:
		meanInitalInhWeight=cInhInit/cMaxInh
	else:
		meanInitalInhWeight=0

	# initialize excitatory connections
	P1e=meanInitalExcWeight
	P0e=1-P1e
	cMatrix[:,:N_STN]=np.random.choice([0.0,1.0],(N,N_STN),p=[P0e,P1e])

	# initialize inhibitory connections
	P1i=meanInitalInhWeight
	P0i=1-P1i
	cMatrix[:,N_STN:]=np.random.choice([0.0,1.0],(N,N_GPe),p=[P0i,P1i])

	# filter weits with actual connections according to connectivity matrix
	cMatrix=cMatrix*synConnections
	cMatrix=scipy.sparse.csc_matrix(cMatrix)
	csc_Zero=scipy.sparse.csc_matrix(np.zeros( ( N,N ) ))
	csc_Ones=scipy.sparse.csc_matrix(np.ones( ( N,N ) ))

	# output struct containing neuron positions in mm
	neuronLoc = { 'STN_center_mm' : STNCenter , 'GPe_center_mm' : GPeCenter }

	# output struct containing objects that are related to network structure but only needed during simulation
	sim_objects = { 'max_N_pre' : maxNumberOfPreSynapticNeurons ,'max_N_post' : maxNumberOfPostSynapticNeurons , 'numpyPostSynapticNeurons' : numpyPostSynapticNeurons , 'numpyPreSynapticNeurons' : numpyPreSynapticNeurons , 'td_PostSynNeurons' : transmissionDelaysPostSynNeurons , 'td_PreSynNeurons' : transmissionDelaysPreSynNeurons , 'csc_Zero' : csc_Zero , 'csc_Ones' : csc_Ones	}
	

	# return output
	return synConnections , cMatrix , neuronLoc , sim_objects




def sequence_paper_generate_connectivity_and_weight_matrix_1D( system_parameters , rnd_state_for_network_generation, d_synaptic_length_scale ):

	# load needed parameters from system_parameters
	# number of STN neurons
	N_STN = system_parameters['N_STN']
	# number of GPe neurons
	N_GPe = system_parameters['N_GPe']
	# total number of neurons
	N = N_STN+ N_GPe
	# probabilityh for STN -> STN connection
	P_STN_STN = system_parameters['P_STN_STN']
	# probabilityh for STN -> GPe connection
	P_STN_GPe = system_parameters['P_STN_GPe']
	# probabilityh for GPe -> GPe connection
	P_GPe_GPe = system_parameters['P_GPe_GPe']
	# probabilityh for GPe -> STN connection
	P_GPe_STN = system_parameters['P_GPe_STN']

	# synaptic transmission delay in time steps
	StepsTauSynDelaySTNSTN=int(system_parameters['tauSynDelaySTNSTN']/system_parameters['dt']) # time steps
	StepsTauSynDelayGPeGPe=int(system_parameters['tauSynDelayGPeGPe']/system_parameters['dt']) # time steps
	StepsTauSynDelayGPeSTN=int(system_parameters['tauSynDelayGPeSTN']/system_parameters['dt']) # time steps
	StepsTauSynDelaySTNGPe=int(system_parameters['tauSynDelaySTNGPe']/system_parameters['dt']) # time steps

	# max strengths exc coupling
	cMaxExc=system_parameters['cMaxExc']	
	# mean initial strengths exc coupling
	cExcInit=system_parameters['cExcInit']

	# max inh coupling
	cMaxInh=system_parameters['cMaxInh']
	# mean initial strengths inh coupling
	cInhInit=system_parameters['cInhInit']

	# set state of random number generator
	np.random.set_state( rnd_state_for_network_generation  )


	x_STN_min =system_parameters['x_STN_min']
	x_STN_max =system_parameters['x_STN_max']
	
	x_GPe_min =system_parameters['x_GPe_min']
	x_GPe_max =system_parameters['x_GPe_max']

	# get connectivity matrix
	STNCenter, GPeCenter= placeNeurons_1D( N_STN , N_GPe, x_STN_min, x_STN_max, x_GPe_min, x_GPe_max )

	# sort neurons according to x-coordinate
	STNCenter = np.sort( STNCenter )
	GPeCenter = np.sort( GPeCenter )

	synConnections= variable_distance_synReversalsCmatrix_1D(STNCenter, GPeCenter, N_STN, N_GPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN, d_synaptic_length_scale )


	# set diagonal to zero 
	diaZero=np.ones( (N,N) )-np.diag( np.ones( N ) )
	synConnections=synConnections*diaZero

	# decouple GPe  ( this is done since we only used STN neurons in our simulations, uncomment if not needed)
	for kNeuron in range(N_STN,N):
		synConnections[:,kNeuron]=np.zeros(N)
		synConnections[kNeuron,:]=np.zeros(N)

	#########################################################################################
	#    in the following additional arrays are introduced to speed up simulations
	# get indicec post and presynaptic neurons to speed up STDP
	PostSynNeurons = {}
	PreSynNeurons = {}

	# max numbers of corresponding synapses
	maxNumberOfPostSynapticNeurons=0
	maxNumberOfPreSynapticNeurons=0

	for kNeuron in range(N_STN):

		PostSynNeurons[kNeuron]=np.nonzero( ( synConnections[:,kNeuron].astype(int) ).tolist() )[0].tolist()
		PreSynNeurons[kNeuron]=np.nonzero( ( synConnections[kNeuron,:].astype(int) ).tolist() )[0].tolist()

		# add random intra-network connections in case of no post/pre neurons
		# this is to help getting fully connected networks
		if len(PostSynNeurons[kNeuron])==0:

			# add random connection
			if kNeuron < N_STN:
				kPost=np.random.choice( range(kNeuron)+range(kNeuron+1,N_STN) )
				synConnections[kPost,kNeuron]=1


		if len(PreSynNeurons[kNeuron])==0:
			# add random connection
			# this is to help getting fully connected networks
			if kNeuron < N_STN:
				kPre=np.random.choice( range(kNeuron)+range(kNeuron+1,N_STN) )
				synConnections[kNeuron,kPre]=1

		# update max numbers of connections
		if maxNumberOfPostSynapticNeurons<len(PostSynNeurons[kNeuron]):
			maxNumberOfPostSynapticNeurons=len(PostSynNeurons[kNeuron])
		if maxNumberOfPreSynapticNeurons<len(PreSynNeurons[kNeuron]):
			maxNumberOfPreSynapticNeurons=len(PreSynNeurons[kNeuron])

	# generate numpy array with post synaptic neurons to speed up simulations ...
	numpyPostSynapticNeurons=np.full((N,maxNumberOfPostSynapticNeurons),N+1)
	numpyPreSynapticNeurons=np.full((N,maxNumberOfPreSynapticNeurons),N+1)

	# ... and corresponding matrix containing transimission delays in time steps
	transmissionDelaysPostSynNeurons=np.full((N,maxNumberOfPostSynapticNeurons),-1.0)
	transmissionDelaysPreSynNeurons=np.full((N,maxNumberOfPreSynapticNeurons),-1.0)

	# gen numpy array with post synaptic neurons
	for kPreSyn in range(N_STN):
		postSynNeuronskPre=PostSynNeurons[kPreSyn]
		for kPostSyn in range(len(postSynNeuronskPre)):
			numpyPostSynapticNeurons[kPreSyn, kPostSyn]=postSynNeuronskPre[kPostSyn]

			kPostNeuron=postSynNeuronskPre[kPostSyn]
			if (kPreSyn < N_STN):
				if (kPostNeuron < N_STN):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelaySTNSTN

				if (kPostNeuron >= N_STN) and (kPostNeuron < N):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelaySTNGPe


			if (kPreSyn >= N_STN) and (kPreSyn < N):
				if (kPostNeuron < N_STN):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelayGPeSTN

				if (kPostNeuron >= N_STN) and (kPostNeuron < N):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelayGPeGPe

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

	# synaptic weight matrix
	cMatrix=np.zeros( (N , N) )

	# initialize synaptic weights by setting random weights to zero so that the mean initial
	# synaptic weights are cExcInit/cMaxExc and cInhInit/cMaxInh for excitatory and inhibitory connections, 
	# respectively
	# mean inital weights
	if cMaxExc != 0:
		meanInitalExcWeight=cExcInit/cMaxExc
	else:
		meanInitalExcWeight=0

	if cMaxInh != 0:
		meanInitalInhWeight=cInhInit/cMaxInh
	else:
		meanInitalInhWeight=0

	# initialize excitatory connections
	P1e=meanInitalExcWeight
	P0e=1-P1e
	cMatrix[:,:N_STN]=np.random.choice([0.0,1.0],(N,N_STN),p=[P0e,P1e])

	# initialize inhibitory connections
	P1i=meanInitalInhWeight
	P0i=1-P1i
	cMatrix[:,N_STN:]=np.random.choice([0.0,1.0],(N,N_GPe),p=[P0i,P1i])

	# filter weits with actual connections according to connectivity matrix
	cMatrix=cMatrix*synConnections
	cMatrix=scipy.sparse.csc_matrix(cMatrix)
	csc_Zero=scipy.sparse.csc_matrix(np.zeros( ( N,N ) ))
	csc_Ones=scipy.sparse.csc_matrix(np.ones( ( N,N ) ))

	# output struct containing neuron positions in mm
	neuronLoc = { 'STN_center_mm' : STNCenter , 'GPe_center_mm' : GPeCenter }

	# output struct containing objects that are related to network structure but only needed during simulation
	sim_objects = { 'max_N_pre' : maxNumberOfPreSynapticNeurons ,'max_N_post' : maxNumberOfPostSynapticNeurons , 'numpyPostSynapticNeurons' : numpyPostSynapticNeurons , 'numpyPreSynapticNeurons' : numpyPreSynapticNeurons , 'td_PostSynNeurons' : transmissionDelaysPostSynNeurons , 'td_PreSynNeurons' : transmissionDelaysPreSynNeurons , 'csc_Zero' : csc_Zero , 'csc_Ones' : csc_Ones	}
	

	# return output
	return synConnections , cMatrix , neuronLoc , sim_objects







def sequence_paper_generate_circular_network( system_parameters , rnd_state_for_network_generation, Pcon ):

	# load needed parameters from system_parameters
	# number of STN neurons
	N_STN = system_parameters['N_STN']
	# number of GPe neurons
	N_GPe = system_parameters['N_GPe']
	# total number of neurons
	N = N_STN+ N_GPe
	# probabilityh for STN -> STN connection
	P_STN_STN = system_parameters['P_STN_STN']
	# probabilityh for STN -> GPe connection
	P_STN_GPe = system_parameters['P_STN_GPe']
	# probabilityh for GPe -> GPe connection
	P_GPe_GPe = system_parameters['P_GPe_GPe']
	# probabilityh for GPe -> STN connection
	P_GPe_STN = system_parameters['P_GPe_STN']

	# synaptic transmission delay in time steps
	StepsTauSynDelaySTNSTN=int(system_parameters['tauSynDelaySTNSTN']/system_parameters['dt']) # time steps
	StepsTauSynDelayGPeGPe=int(system_parameters['tauSynDelayGPeGPe']/system_parameters['dt']) # time steps
	StepsTauSynDelayGPeSTN=int(system_parameters['tauSynDelayGPeSTN']/system_parameters['dt']) # time steps
	StepsTauSynDelaySTNGPe=int(system_parameters['tauSynDelaySTNGPe']/system_parameters['dt']) # time steps

	# max strengths exc coupling
	cMaxExc=system_parameters['cMaxExc']	
	# mean initial strengths exc coupling
	cExcInit=system_parameters['cExcInit']

	# max inh coupling
	cMaxInh=system_parameters['cMaxInh']
	# mean initial strengths inh coupling
	cInhInit=system_parameters['cInhInit']

	# set state of random number generator
	np.random.set_state( rnd_state_for_network_generation  )


	x_STN_min =system_parameters['x_STN_min']
	x_STN_max =system_parameters['x_STN_max']
	
	x_GPe_min =system_parameters['x_GPe_min']
	x_GPe_max =system_parameters['x_GPe_max']

	# get connectivity matrix
	STNCenter, GPeCenter= placeNeurons_1D( N_STN , N_GPe, x_STN_min, x_STN_max, x_GPe_min, x_GPe_max )

	# sort neurons according to x-coordinate
	STNCenter = np.sort( STNCenter )
	GPeCenter = np.sort( GPeCenter )

	# synConnections= variable_distance_synReversalsCmatrix_1D(STNCenter, GPeCenter, N_STN, N_GPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN, d_synaptic_length_scale )
	synConnections= circular_network_synReversalsCmatrix_1D(STNCenter, GPeCenter, N_STN, N_GPe, 2*P_STN_STN, 4 )


	# set diagonal to zero 
	diaZero=np.ones( (N,N) )-np.diag( np.ones( N ) )
	synConnections=synConnections*diaZero

	# decouple GPe  ( this is done since we only used STN neurons in our simulations, uncomment if not needed)
	for kNeuron in range(N_STN,N):
		synConnections[:,kNeuron]=np.zeros(N)
		synConnections[kNeuron,:]=np.zeros(N)

	#########################################################################################
	#    in the following additional arrays are introduced to speed up simulations
	# get indicec post and presynaptic neurons to speed up STDP
	PostSynNeurons = {}
	PreSynNeurons = {}

	# max numbers of corresponding synapses
	maxNumberOfPostSynapticNeurons=0
	maxNumberOfPreSynapticNeurons=0

	for kNeuron in range(N_STN):

		PostSynNeurons[kNeuron]=np.nonzero( ( synConnections[:,kNeuron].astype(int) ).tolist() )[0].tolist()
		PreSynNeurons[kNeuron]=np.nonzero( ( synConnections[kNeuron,:].astype(int) ).tolist() )[0].tolist()

		# add random intra-network connections in case of no post/pre neurons
		# this is to help getting fully connected networks
		if len(PostSynNeurons[kNeuron])==0:

			# add random connection
			if kNeuron < N_STN:
				kPost=np.random.choice( range(kNeuron)+range(kNeuron+1,N_STN) )
				synConnections[kPost,kNeuron]=1


		if len(PreSynNeurons[kNeuron])==0:
			# add random connection
			# this is to help getting fully connected networks
			if kNeuron < N_STN:
				kPre=np.random.choice( range(kNeuron)+range(kNeuron+1,N_STN) )
				synConnections[kNeuron,kPre]=1

		# update max numbers of connections
		if maxNumberOfPostSynapticNeurons<len(PostSynNeurons[kNeuron]):
			maxNumberOfPostSynapticNeurons=len(PostSynNeurons[kNeuron])
		if maxNumberOfPreSynapticNeurons<len(PreSynNeurons[kNeuron]):
			maxNumberOfPreSynapticNeurons=len(PreSynNeurons[kNeuron])

	# generate numpy array with post synaptic neurons to speed up simulations ...
	numpyPostSynapticNeurons=np.full((N,maxNumberOfPostSynapticNeurons),N+1)
	numpyPreSynapticNeurons=np.full((N,maxNumberOfPreSynapticNeurons),N+1)

	# ... and corresponding matrix containing transimission delays in time steps
	transmissionDelaysPostSynNeurons=np.full((N,maxNumberOfPostSynapticNeurons),-1.0)
	transmissionDelaysPreSynNeurons=np.full((N,maxNumberOfPreSynapticNeurons),-1.0)

	# gen numpy array with post synaptic neurons
	for kPreSyn in range(N_STN):
		postSynNeuronskPre=PostSynNeurons[kPreSyn]
		for kPostSyn in range(len(postSynNeuronskPre)):
			numpyPostSynapticNeurons[kPreSyn, kPostSyn]=postSynNeuronskPre[kPostSyn]

			kPostNeuron=postSynNeuronskPre[kPostSyn]
			if (kPreSyn < N_STN):
				if (kPostNeuron < N_STN):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelaySTNSTN

				if (kPostNeuron >= N_STN) and (kPostNeuron < N):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelaySTNGPe


			if (kPreSyn >= N_STN) and (kPreSyn < N):
				if (kPostNeuron < N_STN):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelayGPeSTN

				if (kPostNeuron >= N_STN) and (kPostNeuron < N):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelayGPeGPe

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

	# synaptic weight matrix
	cMatrix=np.zeros( (N , N) )

	# initialize synaptic weights by setting random weights to zero so that the mean initial
	# synaptic weights are cExcInit/cMaxExc and cInhInit/cMaxInh for excitatory and inhibitory connections, 
	# respectively
	# mean inital weights
	if cMaxExc != 0:
		meanInitalExcWeight=cExcInit/cMaxExc
	else:
		meanInitalExcWeight=0

	if cMaxInh != 0:
		meanInitalInhWeight=cInhInit/cMaxInh
	else:
		meanInitalInhWeight=0

	# initialize excitatory connections
	P1e=meanInitalExcWeight
	P0e=1-P1e
	cMatrix[:,:N_STN]=np.random.choice([0.0,1.0],(N,N_STN),p=[P0e,P1e])

	# initialize inhibitory connections
	P1i=meanInitalInhWeight
	P0i=1-P1i
	cMatrix[:,N_STN:]=np.random.choice([0.0,1.0],(N,N_GPe),p=[P0i,P1i])

	# filter weits with actual connections according to connectivity matrix
	cMatrix=cMatrix*synConnections
	cMatrix=scipy.sparse.csc_matrix(cMatrix)
	csc_Zero=scipy.sparse.csc_matrix(np.zeros( ( N,N ) ))
	csc_Ones=scipy.sparse.csc_matrix(np.ones( ( N,N ) ))

	# output struct containing neuron positions in mm
	neuronLoc = { 'STN_center_mm' : STNCenter , 'GPe_center_mm' : GPeCenter }

	# output struct containing objects that are related to network structure but only needed during simulation
	sim_objects = { 'max_N_pre' : maxNumberOfPreSynapticNeurons ,'max_N_post' : maxNumberOfPostSynapticNeurons , 'numpyPostSynapticNeurons' : numpyPostSynapticNeurons , 'numpyPreSynapticNeurons' : numpyPreSynapticNeurons , 'td_PostSynNeurons' : transmissionDelaysPostSynNeurons , 'td_PreSynNeurons' : transmissionDelaysPreSynNeurons , 'csc_Zero' : csc_Zero , 'csc_Ones' : csc_Ones	}
	

	# return output
	return synConnections , cMatrix , neuronLoc , sim_objects


def sequence_paper_generate_circular_network_2( system_parameters , rnd_state_for_network_generation, Pcon ):

	# load needed parameters from system_parameters
	# number of STN neurons
	N_STN = system_parameters['N_STN']
	# number of GPe neurons
	N_GPe = system_parameters['N_GPe']
	# total number of neurons
	N = N_STN+ N_GPe
	# probabilityh for STN -> STN connection
	P_STN_STN = system_parameters['P_STN_STN']
	# probabilityh for STN -> GPe connection
	P_STN_GPe = system_parameters['P_STN_GPe']
	# probabilityh for GPe -> GPe connection
	P_GPe_GPe = system_parameters['P_GPe_GPe']
	# probabilityh for GPe -> STN connection
	P_GPe_STN = system_parameters['P_GPe_STN']

	# synaptic transmission delay in time steps
	StepsTauSynDelaySTNSTN=int(system_parameters['tauSynDelaySTNSTN']/system_parameters['dt']) # time steps
	StepsTauSynDelayGPeGPe=int(system_parameters['tauSynDelayGPeGPe']/system_parameters['dt']) # time steps
	StepsTauSynDelayGPeSTN=int(system_parameters['tauSynDelayGPeSTN']/system_parameters['dt']) # time steps
	StepsTauSynDelaySTNGPe=int(system_parameters['tauSynDelaySTNGPe']/system_parameters['dt']) # time steps

	# max strengths exc coupling
	cMaxExc=system_parameters['cMaxExc']	
	# mean initial strengths exc coupling
	cExcInit=system_parameters['cExcInit']

	# max inh coupling
	cMaxInh=system_parameters['cMaxInh']
	# mean initial strengths inh coupling
	cInhInit=system_parameters['cInhInit']

	# set state of random number generator
	np.random.set_state( rnd_state_for_network_generation  )


	x_STN_min =system_parameters['x_STN_min']
	x_STN_max =system_parameters['x_STN_max']
	
	x_GPe_min =system_parameters['x_GPe_min']
	x_GPe_max =system_parameters['x_GPe_max']

	# get connectivity matrix
	STNCenter, GPeCenter= placeNeurons_1D( N_STN , N_GPe, x_STN_min, x_STN_max, x_GPe_min, x_GPe_max )

	# sort neurons according to x-coordinate
	STNCenter = np.sort( STNCenter )
	GPeCenter = np.sort( GPeCenter )

	# synConnections= variable_distance_synReversalsCmatrix_1D(STNCenter, GPeCenter, N_STN, N_GPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN, d_synaptic_length_scale )
	synConnections= circular_network_2_synReversalsCmatrix_1D(STNCenter, GPeCenter, N_STN, N_GPe, 2*P_STN_STN, 4 )


	# set diagonal to zero 
	diaZero=np.ones( (N,N) )-np.diag( np.ones( N ) )
	synConnections=synConnections*diaZero

	# decouple GPe  ( this is done since we only used STN neurons in our simulations, uncomment if not needed)
	for kNeuron in range(N_STN,N):
		synConnections[:,kNeuron]=np.zeros(N)
		synConnections[kNeuron,:]=np.zeros(N)

	#########################################################################################
	#    in the following additional arrays are introduced to speed up simulations
	# get indicec post and presynaptic neurons to speed up STDP
	PostSynNeurons = {}
	PreSynNeurons = {}

	# max numbers of corresponding synapses
	maxNumberOfPostSynapticNeurons=0
	maxNumberOfPreSynapticNeurons=0

	for kNeuron in range(N_STN):

		PostSynNeurons[kNeuron]=np.nonzero( ( synConnections[:,kNeuron].astype(int) ).tolist() )[0].tolist()
		PreSynNeurons[kNeuron]=np.nonzero( ( synConnections[kNeuron,:].astype(int) ).tolist() )[0].tolist()

		# add random intra-network connections in case of no post/pre neurons
		# this is to help getting fully connected networks
		if len(PostSynNeurons[kNeuron])==0:

			# add random connection
			if kNeuron < N_STN:
				kPost=np.random.choice( range(kNeuron)+range(kNeuron+1,N_STN) )
				synConnections[kPost,kNeuron]=1


		if len(PreSynNeurons[kNeuron])==0:
			# add random connection
			# this is to help getting fully connected networks
			if kNeuron < N_STN:
				kPre=np.random.choice( range(kNeuron)+range(kNeuron+1,N_STN) )
				synConnections[kNeuron,kPre]=1

		# update max numbers of connections
		if maxNumberOfPostSynapticNeurons<len(PostSynNeurons[kNeuron]):
			maxNumberOfPostSynapticNeurons=len(PostSynNeurons[kNeuron])
		if maxNumberOfPreSynapticNeurons<len(PreSynNeurons[kNeuron]):
			maxNumberOfPreSynapticNeurons=len(PreSynNeurons[kNeuron])

	# generate numpy array with post synaptic neurons to speed up simulations ...
	numpyPostSynapticNeurons=np.full((N,maxNumberOfPostSynapticNeurons),N+1)
	numpyPreSynapticNeurons=np.full((N,maxNumberOfPreSynapticNeurons),N+1)

	# ... and corresponding matrix containing transimission delays in time steps
	transmissionDelaysPostSynNeurons=np.full((N,maxNumberOfPostSynapticNeurons),-1.0)
	transmissionDelaysPreSynNeurons=np.full((N,maxNumberOfPreSynapticNeurons),-1.0)

	# gen numpy array with post synaptic neurons
	for kPreSyn in range(N_STN):
		postSynNeuronskPre=PostSynNeurons[kPreSyn]
		for kPostSyn in range(len(postSynNeuronskPre)):
			numpyPostSynapticNeurons[kPreSyn, kPostSyn]=postSynNeuronskPre[kPostSyn]

			kPostNeuron=postSynNeuronskPre[kPostSyn]
			if (kPreSyn < N_STN):
				if (kPostNeuron < N_STN):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelaySTNSTN

				if (kPostNeuron >= N_STN) and (kPostNeuron < N):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelaySTNGPe


			if (kPreSyn >= N_STN) and (kPreSyn < N):
				if (kPostNeuron < N_STN):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelayGPeSTN

				if (kPostNeuron >= N_STN) and (kPostNeuron < N):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelayGPeGPe

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

	# synaptic weight matrix
	cMatrix=np.zeros( (N , N) )

	# initialize synaptic weights by setting random weights to zero so that the mean initial
	# synaptic weights are cExcInit/cMaxExc and cInhInit/cMaxInh for excitatory and inhibitory connections, 
	# respectively
	# mean inital weights
	if cMaxExc != 0:
		meanInitalExcWeight=cExcInit/cMaxExc
	else:
		meanInitalExcWeight=0

	if cMaxInh != 0:
		meanInitalInhWeight=cInhInit/cMaxInh
	else:
		meanInitalInhWeight=0

	# initialize excitatory connections
	P1e=meanInitalExcWeight
	P0e=1-P1e
	cMatrix[:,:N_STN]=np.random.choice([0.0,1.0],(N,N_STN),p=[P0e,P1e])

	# initialize inhibitory connections
	P1i=meanInitalInhWeight
	P0i=1-P1i
	cMatrix[:,N_STN:]=np.random.choice([0.0,1.0],(N,N_GPe),p=[P0i,P1i])

	# filter weits with actual connections according to connectivity matrix
	cMatrix=cMatrix*synConnections
	cMatrix=scipy.sparse.csc_matrix(cMatrix)
	csc_Zero=scipy.sparse.csc_matrix(np.zeros( ( N,N ) ))
	csc_Ones=scipy.sparse.csc_matrix(np.ones( ( N,N ) ))

	# output struct containing neuron positions in mm
	neuronLoc = { 'STN_center_mm' : STNCenter , 'GPe_center_mm' : GPeCenter }

	# output struct containing objects that are related to network structure but only needed during simulation
	sim_objects = { 'max_N_pre' : maxNumberOfPreSynapticNeurons ,'max_N_post' : maxNumberOfPostSynapticNeurons , 'numpyPostSynapticNeurons' : numpyPostSynapticNeurons , 'numpyPreSynapticNeurons' : numpyPreSynapticNeurons , 'td_PostSynNeurons' : transmissionDelaysPostSynNeurons , 'td_PreSynNeurons' : transmissionDelaysPreSynNeurons , 'csc_Zero' : csc_Zero , 'csc_Ones' : csc_Ones	}
	

	# return output
	return synConnections , cMatrix , neuronLoc , sim_objects



##################################
# run script to test functions
if __name__ == "__main__":

	import sys

	if len(sys.argv[1]) == 0:
		print( 'ERROR: CLASS_OR_FUNCTIONNAME needed' )
		print( 'run python functions_genNetwork.py CLASS_OR_FUNCTIONNAME to test function or class with CLASS_OR_FUNCTIONNAME' )
		print( 'possible values for CLASS_OR_FUNCTIONNAME:' )
		print( '			placeNeurons     ' )
		print( '			conProbability   ' )
		print( '         cartesianProduct ' )
		print( '         synReversalsCmatrix' )
		print( ' 		placeNeurons_3D_cuboid' )
		print( ' 		synReversalsCmatrix_3D_cuboid' )
		exit()

	else:

		if sys.argv[1] == 'circular_network_2_synReversalsCmatrix_1D':

			matrix = circular_network_2_synReversalsCmatrix_1D( np.random.uniform(0,1,1000) , np.random.uniform(0,1,2), 1000, 2, 0.07, 4 )
			
			# import matplotlib.pyplot as plt
			# plt.imshow( matrix.A )
			# plt.show()

		if sys.argv[1] == 'synReversalsCmatrix_1D':

			# gen positions
			NSTN = 1000
			NGPe = 10

			x_min = -2.5 # mm
			x_max = 2.5 # mm

			xSTN, xGPe = placeNeurons_1D( NSTN , NGPe, x_min, x_max, x_min, x_max )

			P_STN_STN = 0.07
			P_STN_GPe = 0.02
			P_GPe_GPe = 0.02
			P_GPe_STN = 0.01

			synReversals = synReversalsCmatrix_1D(xSTN, xGPe, NSTN, NGPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN )


			print( np.sum( synReversals[synReversals>0] ) )
			# check probabilities
			print( ' ' )
			print( '############################' )
			print( ' ' )
			print( 'prob for connection:   output / input' )
			print( np.abs( np.mean( synReversals[:NSTN,:NSTN] ) ) , '/' , P_STN_STN )
			print( np.abs( np.mean( synReversals[NSTN:(NSTN+NGPe),:NSTN] ) ) , '/' , P_STN_GPe )
			print( np.abs( np.mean( synReversals[NSTN:(NSTN+NGPe),NSTN:(NSTN+NGPe)] ) ) , '/' , P_GPe_GPe )
			print( np.abs( np.mean( synReversals[:NSTN,NSTN:(NSTN+NGPe)] ) ) , '/' , P_GPe_STN )
			print( ' ' )
			print( 'number of self connections' )
			print( np.sum( np.diag( synReversals ) ) )

			# plot synReversals
			import functions_plot
			functions_plot.plot_mat_2d( synReversals )




		if sys.argv[1] == 'placeNeurons1D':
			# gen positions
			NSTN = 1000
			NGPe = 1000

			x_min = -2.5 # mm
			x_max = 2.5 # mm

			xSTN, xGPe = placeNeurons_1D( NSTN , NGPe, x_min, x_max, x_min, x_max )
			import matplotlib.pyplot as plt 

			plt.hist(xSTN, 100)
			plt.show()

			plt.hist(xGPe, 100)
			plt.show()

		##################################
		# test placeNeurons by plotting 3d positions
		if sys.argv[1] == 'placeNeurons':

			# gen positions
			NSTN = 1000
			NGPe = 1000

			xyzSTN , xyzGPe = placeNeurons( NSTN , NGPe )

			import functions_plot
			functions_plot.scatter_3d( xyzSTN )
			
			exit()


		##################################
		# test placeNeurons by plotting 3d positions
		if sys.argv[1] == 'placeNeurons':

			# gen positions
			NSTN = 1000
			NGPe = 1000

			xyzSTN , xyzGPe = placeNeurons( NSTN , NGPe )

			import functions_plot
			functions_plot.scatter_3d( xyzSTN )
			
			exit()

		##################################
		# test conProbability by plotting shape of probability for connection as function of distance
		if sys.argv[1] == 'conProbability':

			import functions_plot
			# gen distances
			# spatial resolution
			dd = 0.01 # mm
			d = np.arange( 0, 10, dd ) # mm
			Cd = 0.5  # mm
			p = conProbability(d, Cd)

			# check normalization
			print( 'p is normalized to' )
			print( np.sum(p)*dd )

			# plot distance dependence
			functions_plot.plot_xy( d, p )

			exit()

		##################################
		# test cartesianProduct by an example
		if sys.argv[1] == 'cartesianProduct':

			a=['a','b','c']
			b=['1','2']

			print( cartesianProduct(a,b) )

			exit()

		##################################
		# test synReversalsCmatrix by plotting connectivity matrix and check connection probabilties
		if sys.argv[1] == 'synReversalsCmatrix':

			import functions_plot

			NSTN = 10
			NGPe = 25

			P_STN_STN = 0.2
			P_STN_GPe = 0.7
			P_GPe_GPe = 0.3
			P_GPe_STN = 0.5

			positionsSTNNeurons = np.random.uniform( -1, 1, (NSTN,3) )
			positionsGPeNeurons = np.random.uniform( -1, 1, (NGPe,3) )

			synReversals = synReversalsCmatrix(positionsSTNNeurons, positionsGPeNeurons, NSTN, NGPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN)

			# check probabilities
			print( ' ' )
			print( '############################' )
			print( ' ' )
			print( 'prob for connection:   output / input' )
			print( np.abs( np.mean( synReversals[:NSTN,:NSTN] ) ) , '/' , P_STN_STN )
			print( np.abs( np.mean( synReversals[NSTN:(NSTN+NGPe),:NSTN] ) ) , '/' , P_STN_GPe )
			print( np.abs( np.mean( synReversals[NSTN:(NSTN+NGPe),NSTN:(NSTN+NGPe)] ) ) , '/' , P_GPe_GPe )
			print( np.abs( np.mean( synReversals[:NSTN,NSTN:(NSTN+NGPe)] ) ) , '/' , P_GPe_STN )
			print( ' ' )
			print( 'number of self connections' )
			print( np.sum( np.diag( synReversals ) ) )

			# plot synReversals
			functions_plot.plot_mat_2d( synReversals )

			exit()

	

		##################################
		# test placeNeurons_3D_cuboid by plotting 3d positions
		if sys.argv[1] == 'placeNeurons_3D_cuboid':

			# gen positions
			NSTN = 1000
			NGPe = 2

			rx_STN, ry_STN, rz_STN = 4.0, 2.0, 2.0 # mm
			rx_GPe, ry_GPe, rz_GPe = 4.0, 2.0, 2.0 # mm

			xyzSTN , xyzGPe = placeNeurons_3D_cuboid( NSTN , NGPe , rx_STN, ry_STN, rz_STN, rx_GPe, ry_GPe, rz_GPe )

			import functions_plot
			functions_plot.scatter_3d( xyzSTN )
			
			exit()


		##################################
		# test synReversalsCmatrix_3D_cuboid
		if sys.argv[1] == 'synReversalsCmatrix_3D_cuboid':

			# gen positions
			import functions_plot

			NSTN = 1000
			NGPe = 2

			P_STN_STN = 0.07
			P_STN_GPe = 0.001
			P_GPe_GPe = 0.01
			P_GPe_STN = 0.001

			rx_STN, ry_STN, rz_STN = 2.0, 1.0, 1.0 # mm
			rx_GPe, ry_GPe, rz_GPe = 2.0, 1.0, 1.0 # mm

			xyzSTN , xyzGPe = placeNeurons_3D_cuboid( NSTN , NGPe , rx_STN, ry_STN, rz_STN, rx_GPe, ry_GPe, rz_GPe )

			synReversals = 	synReversalsCmatrix_3D_cuboid(xyzSTN, xyzGPe, NSTN, NGPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN, [rx_STN, ry_STN, rz_STN], [rx_GPe, ry_GPe, rz_GPe])

			# check probabilities
			print( ' ' )
			print( '############################' )
			print( ' ' )
			print( 'prob for connection:   output / input' )
			print( np.abs( np.mean( synReversals[:NSTN,:NSTN] ) ) , '/' , P_STN_STN )
			print( np.abs( np.mean( synReversals[NSTN:(NSTN+NGPe),:NSTN] ) ) , '/' , P_STN_GPe )
			print( np.abs( np.mean( synReversals[NSTN:(NSTN+NGPe),NSTN:(NSTN+NGPe)] ) ) , '/' , P_GPe_GPe )
			print( np.abs( np.mean( synReversals[:NSTN,NSTN:(NSTN+NGPe)] ) ) , '/' , P_GPe_STN )
			print( ' ' )
			print( 'number of self connections' )
			print( np.sum( np.diag( synReversals ) ) )

			# plot synReversals
			functions_plot.plot_mat_2d( synReversals )

			exit()
			exit()



		##################################
		# test synReversalsCmatrix_3D_cuboid
		if sys.argv[1] == 'homogeneous':

			# gen positions
			import functions_plot

			NSTN = 1000
			NGPe = 2

			P_STN_STN = 0.07
			P_STN_GPe = 0.001
			P_GPe_GPe = 0.01
			P_GPe_STN = 0.001


			synReversals = 	synReversalsCmatrix_homogeneous(NSTN, NGPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN)

			# check probabilities
			print( ' ' )
			print( '############################' )
			print( ' ' )
			print( 'prob for connection:   output / input' )
			print( np.abs( np.mean( synReversals[:NSTN,:NSTN] ) ) , '/' , P_STN_STN )
			print( np.abs( np.mean( synReversals[NSTN:(NSTN+NGPe),:NSTN] ) ) , '/' , P_STN_GPe )
			print( np.abs( np.mean( synReversals[NSTN:(NSTN+NGPe),NSTN:(NSTN+NGPe)] ) ) , '/' , P_GPe_GPe )
			print( np.abs( np.mean( synReversals[:NSTN,NSTN:(NSTN+NGPe)] ) ) , '/' , P_GPe_STN )
			print( ' ' )
			print( 'number of self connections' )
			print( np.sum( np.diag( synReversals ) ) )

			# plot synReversals
			functions_plot.plot_mat_2d( synReversals )

			exit()
			exit()


		##################################
		# test 1D check distance
		if sys.argv[1] == '1D distance':
					
			# gen positions
			NSTN = 2000
			NGPe = 2

			# array of indices
			ind = np.arange(NSTN)

			# number of subpopulations
			M=4

			# x range
			x_min = -2.5 # mm
			x_max = 2.5 # mm

			# connection probabilities
			P_STN_STN = 0.07
			P_STN_GPe = 0.0
			P_GPe_GPe = 0.0 
			P_GPe_STN = 0.0

			# length scale of synaptic connections
			d = float(sys.argv[2])# mm


			print('d=',d,'mm') 
			ds = (x_max-x_min)/4.0
			print('ds=',ds) 

			print('d/ds=',d/ds)

			Estimate_Bxy=np.zeros( (M,M) )
			Nruns = 100

			for kRun in range(Nruns):
				print(kRun)

				xSTN, xGPe = placeNeurons_1D( NSTN , NGPe, x_min, x_max, x_min, x_max )
				
				# sort neurons according to x coordinates 
				xSTN, xGPe = np.sort(xSTN), np.sort(xGPe)
				
				# sort neurons according to four subpopulations
				subpop_STN = ( M*(xSTN-x_min)/(x_max-x_min) ).astype( int )
				subpop_GPe = ( M*(xGPe-x_min)/(x_max-x_min) ).astype( int )

				# calculate connectivity matrix
				# synReversals[ kPost, kPre ]
				synReversals = variable_distance_synReversalsCmatrix_1D(xSTN, xGPe, NSTN, NGPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN, d )

				# #import matplotlib.pyplot as plt 
				# calculate Bxy
				# list of arrays of indices of neurons belonging to the same subpopulations
				ind_pop_k = []
				for k in range(M):
					ind_pop_k.append( ind[ subpop_STN == k ] )

				# total number of synapses
				tot_n_syn_STN_STN = len( np.nonzero(synReversals[:NSTN,:NSTN])[0] )

				# Bxy[ kPost, kPre ]
				Bxy = np.zeros( (M,M) )

				for kPre in range(M):

					indMinPre = ind_pop_k[kPre][0]
					indMaxPre = ind_pop_k[kPre][-1]+1
					#print(indMinPre, indMaxPre)
					for kPost in range(M):
						#print(ind_pop_k[kPre])
						indMinPost = ind_pop_k[kPost][0]	
						indMaxPost = ind_pop_k[kPost][-1]+1

						#print(synReversals[ indMinPost:indMaxPost+1 , indMinPre:indMaxPre+1 ].shape)
						Bxy[ kPost, kPre ] = len( np.nonzero( synReversals[ indMinPost:indMaxPost , indMinPre:indMaxPre ] )[0] )

				#print(np.round( Bxy/float(tot_n_syn_STN_STN) ,3)  )
				#print(np.sum(Bxy/float(tot_n_syn_STN_STN)))
				
				Estimate_Bxy+=Bxy/float(tot_n_syn_STN_STN)

			print( 'result' )
			print( np.round( Estimate_Bxy/float(Nruns), 3) )

				#print(np.sum(Bxy))
  	# 		print tot_n_syn
  			# import matplotlib.pyplot as plt 
  			# plt.imshow(synReversals)
  				
  			# plt.show()
