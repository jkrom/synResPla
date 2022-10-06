##################################
# written by: Justus Kromer
##################################
# written and tested for Python 2.7.13
##################################
#
#	content:
#
#		classes and functions for stimulation protocols
#
#		class RVS_CR       ( nonoverlapping stimuli )
#
#		gen_wSignal_for_CR_contacts_along_x_axes( Nelectrodes, system_parameters, pos_neurons )
#
#	run: python gen_stationary_states.py outputDirectory initialSeed

# imports
import sys
import numpy as np


##################################
# class: RVS_CR_overlapping   ( same as RVS_CR but stimuli delivered to different subpopulations can overlap )
##################################
# Description:
#   Rapidly varying sequence CR protocol
#
class RVS_CR_Burst_overlapping_PLOS_paper:

	## input:           
	#	timeBetweenBursts      ... time between end of burst and beginning of next burst
	#   totalStimulationTime   ... total stimulation time in sec
	#	Nelectrodes            ... number of stimulation sites  (4 for standard CR stimultion)
	# 	NpB 				   ... number of pulses per burst
	def __init__(self, fCR, totalStimulationTime, M, NpB, dt, e_pulse_scale, intraburst_frequency ):

		# time interval between end of burst and beginning of next burst in ms
		self.fCR=fCR    # Hz

		# total stimulation time in sec
		self.Tstim=totalStimulationTime*1000.0 # ms

		# number of stimulation sites
		self.M=M

		# integration time step used in simulations
		self.dt=dt # ms

		# number of pulses per burst
		self.NpB = NpB

		# pulse scaling ratio
		self.e_pulse_scale = e_pulse_scale

		# inverse time between first spikes of subsequent bursts
		self.burstFrequency = 0        # Hz

		# array that contains pre-calculated stimulus
		# kth element is value of stimulation current delivered at k*dt after stimulus onset
		self.signalOnTrain=np.zeros(1)

		# number of time steps in 'signalOnTrain'
		self.lengthSignalOneElectrode=0

		# intraburst_frequency
		self.intraburst_frequency = intraburst_frequency

		# integration time step at which current stimulus is delivered (beginning of next stimulus)
		# is used to get end of current stimulus and beginning of next one
		self.CurrentStimulusOnset=np.arange(self.M)	# time steps

		# directory name for output from simulations using this stimulation protocol
		self.signalName = ''

		# start of next cycle
		self.startNextCycle = 0

	##################################
	#   function: initialize_RVS_CR_protocol(self )
	def initialize_RVS_CR_Burst_overlapping_PLOS_paper( self ):
	#
	# 		initializes signal calculated the full signal of one electrode for one/Nelectrodes signal periods

		########## generate pulse shape ##########
		### single pulse characteristics
		# pos rectangular pulses of unit amplitude, duration 0.2 ms followed by a 
		# negative counterpart of length 3 ms and amplitude 1/15, interpulse interval 1/130 s
		# positive rectangular pulse
		tStartPosPuls=0.2 # ms
		tStepStartPosPuls= int(tStartPosPuls/self.dt)
		lengthsPosRect=0.4*self.e_pulse_scale # ms
		tSteplengthsPosRect= int(lengthsPosRect/self.dt)

		# normalized such that integral over time in ms yields one
		AmpPosPuls=1.0/(0.4*self.e_pulse_scale)

		# negative pulse
		tStartNegPuls=lengthsPosRect+0.2 # ms
		tStepStartNegPuls= int(tStartNegPuls/self.dt)
		lengthsNegRect=0.8*self.e_pulse_scale # ms         # motivated by Tass et al. 2012 (monkey study)
		tSteplengthsNegRect= int(lengthsNegRect/self.dt)
		AmpNegPuls= -(AmpPosPuls*lengthsPosRect)/lengthsNegRect # ensures charge balance by scaling the amplitude

		### minimal interval between subsequent pulses is adjusted to 130 Hz DBS pulsFrequency
		pulsFrequency=self.intraburst_frequency #  Hz
		pulsPeriod=1.0/float(pulsFrequency*0.001) # ms  // approx 7.69 ms

		self.pulsLength = int( (pulsPeriod)/self.dt ) # (number of timesteps until next pulse starts 

		#### in case of burst stimuli one stimulus consistes of 
		# several bursts
		# create signal for one electrode
		self.lengthSignalOneElectrode = self.pulsLength * self.NpB
		self.signalOnTrain = np.zeros(self.lengthSignalOneElectrode)


		# directory in which output for this stimulation protocol is saved
		# self.signalName='multisite_RVS_CR_overlap_Tstim/fCR_'+str(self.fCR)+'_M_'+str(self.M)
		self.signalName='multiple_spikes_RVS_CR_TASS2012/pulses_per_burst_'+str(self.NpB)+'/intraburst_frequency_'+str(self.intraburst_frequency)+'_fCR_'+str(self.fCR)+'_M_'+str(self.M)+'_e_pulse_scale_'+str(self.e_pulse_scale)

		### construct a single stimulus
		# this incorporates a burst stimulus or a single-pulse stimulus
		for ktimeSteps in range( self.pulsLength ):

			kStep=ktimeSteps

			for kPulse in range( self.NpB ):

				offSetPulse = kPulse * self.pulsLength

				# add pos pulse
				if ((ktimeSteps) < tSteplengthsPosRect+tStepStartPosPuls) and (tStepStartPosPuls) <= (ktimeSteps):

					self.signalOnTrain[ kStep + offSetPulse ]=AmpPosPuls

				# add neg pulse
				if ((ktimeSteps)<tSteplengthsNegRect+tStepStartNegPuls) and (tStepStartNegPuls) <= (ktimeSteps):

					self.signalOnTrain[kStep+offSetPulse]=AmpNegPuls


		self.rec_stim_Sequence = []

		### initialize array that contains the stimulation currents per integration time step
		# number of integration time steps in a CR cycle
		self.time_steps_per_cycle_period = int( 1000./(self.fCR * self.dt) ) # time steps 
		self.lengthOfCurrentArray = max( int(1000./(self.fCR * self.dt)) , self.M * self.lengthSignalOneElectrode ) # time steps

		# time steps of stimulus onset
		timeStepsToNextStimulusOnset = int( float( self.time_steps_per_cycle_period )/float(self.M) )
		self.stimOnsets = timeStepsToNextStimulusOnset*np.arange( self.M ) # time steps

		###  CR sequence
		self.CR_Sequence = np.arange(self.M)

		### original currents to contact
		# rows contain currents for all time steps during one CR cycle 
		# m th row contains the currents for the m th stimulus activation in the CR cycle
		self.originalCurrentsToContacts = np.zeros( ( self.M , self.lengthOfCurrentArray ) )

		# for m in range(self.M):

		# 	for kSignal in range( len(self.signalOnTrain) ):

		# 		if kSignal+self.stimOnsets[m] < len( self.originalCurrentsToContacts[ m ] ):
		# 			self.originalCurrentsToContacts[ m, kSignal+self.stimOnsets[m] ] = self.signalOnTrain[kSignal]
		# 		else:
		# 			print( 'Warning: '+str(m)+'th stimulus would reaches into next cycle and is cut off' )

		# import matplotlib.pyplot as plt 
		# fig = plt.figure()
		# ax = fig.add_subplot(111)
		# ax.imshow (self.originalCurrentsToContacts )

		# ax.set_aspect( 100 )
		# plt.show()

		return 0

	##################################
	#   function: enterNewCycle(self)
	def enterNewCycle(self, timeStep):

		### create new realization of CR sequence
		np.random.shuffle( self.CR_Sequence )
		# print(self.CR_Sequence)

		### generate stimulus pattern for next cycle
		# run through M slots during CR cycle
		for kpop in np.arange( self.M ):

			# get onset time for stimulus delivered to that slot
			tOnsetStimToPopulation = timeStep + self.stimOnsets[  kpop ]
			# get stimulation contact that is activated at that slot
			kStimPop = self.CR_Sequence[ kpop ]

			# record stimulation sequence
			self.rec_stim_Sequence.append( [ tOnsetStimToPopulation , kStimPop ] )

			# run through future time steps
			for kStep in range( self.lengthSignalOneElectrode ):
				self.originalCurrentsToContacts[ kStimPop ,  (tOnsetStimToPopulation + kStep) % self.lengthOfCurrentArray   ] = self.signalOnTrain[kStep]

		# 	print( kStimPop , tOnsetStimToPopulation )

		# exit()



		# currents to contacts
		# self.current_Currents = np.copy( self.originalCurrentsToContacts )
		# np.random.shuffle( self.current_Currents )

		#print self.current_Currents
		self.startNextCycle += self.time_steps_per_cycle_period

	##################################
	#   function: getCurrent( self, timeStep )
	def getCurrent( self, timeStep ):

		#print timeStep, self.startNextCycle
		if timeStep == self.startNextCycle:
			currentOutputTimeStep = 0
			self.enterNewCycle( timeStep )

		# stimulation is turned on
		if timeStep > 0:
			currentOutputTimeStep = timeStep % self.lengthOfCurrentArray
			lastOutputTimeStep = (currentOutputTimeStep-1) % self.lengthOfCurrentArray 
			self.originalCurrentsToContacts[ :, lastOutputTimeStep  ] = 0
		else:
			currentOutputTimeStep = 0

		# stimulation is turned off 
		if timeStep < 0:
			return 0*self.originalCurrentsToContacts[:,0]

		# set stimulus currents that where already delivered to zero
		
		return self.originalCurrentsToContacts[ :,currentOutputTimeStep ]


class multiple_spikes_phase_shifted_periodic_multisite_stimulation:

	## input:           
	#   timeBetweenBursts      ... time between end of burst and beginning of next burst
	#   totalStimulationTime   ... total stimulation time in sec
	#   Nelectrodes            ... number of stimulation sites  (4 for standard CR stimultion)
	#   phase_shifts           ... array containing phases shiftes Delta alpha1-Delta till alphaM-1
	#							   values are between zero and one
	def __init__(self, fCR, totalStimulationTime, M, dt, phase_shifts, e_pulse_scale, pulses_per_burst ):

		# time interval between end of burst and beginning of next burst in ms
		self.fCR=fCR    # Hz

		# total stimulation time in sec
		self.Tstim=totalStimulationTime*1000.0 # ms

		# number of stimulation sites
		self.M=M

		# integration time step used in simulations
		self.dt=dt # ms

		# array that contains pre-calculated stimulus
		# kth element is value of stimulation current delivered at k*dt after stimulus onset
		self.signalOnTrain=np.zeros(1)

		# number of time steps in 'signalOnTrain'
		self.lengthSignalOneElectrode=0

		# integration time step at which current stimulus is delivered (beginning of next stimulus)
		# is used to get end of current stimulus and beginning of next one
		self.CurrentStimulusOnset=np.arange(self.M) # time steps

		# directory name for output from simulations using this stimulation protocol
		self.signalName=''

		# start of next cycle
		self.startNextCycle = 0
 
		self.phase_shifts = phase_shifts

		self.e_pulse_scale = e_pulse_scale

		self.pulses_per_burst = pulses_per_burst
		# print 'sequence'
		# print self.Sequence

	##################################
	#   function: initialize_Fixed_Sequence_CR_overlapping_Chaos_Paper(self )
	def initialize_multiple_spikes_phase_shifted_periodic_multisite_stimulation(self ):
	#
	#       initializes signal calculated the full signal of one electrode for one/Nelectrodes signal periods

		# number of integration time steps in a cycle
		self.time_steps_per_cycle_period = int(1000./(self.fCR * self.dt)) # steps

		self.Sequence = [0]
		# time steps of stimulus onsets
		phases_of_stimulus_onsets_in_first_cycle = np.zeros( self.M )
		# obda, first stimulus is delivered at t=0
		# now calculated the others
		directory_phaseShifts=''
		if self.M > 1:
			for k_Dalpha in range( len( self.phase_shifts ) ):
				Dalpha = self.phase_shifts[k_Dalpha]

				phases_of_stimulus_onsets_in_first_cycle[ k_Dalpha+1 ] = np.mod( phases_of_stimulus_onsets_in_first_cycle[ k_Dalpha ] + Dalpha , 1.0 )
			
				directory_phaseShifts+='Dalpha'+str(k_Dalpha+1)+'_'+str(Dalpha)+'_'
				self.Sequence.append(k_Dalpha+1)
			directory_phaseShifts=directory_phaseShifts[:-1]
		else:
			directory_phaseShifts = 'single'

		timeStepsToNextStimulusOnset = int( float( self.time_steps_per_cycle_period )/float(self.M) )
		self.stimOnsets = (self.time_steps_per_cycle_period*phases_of_stimulus_onsets_in_first_cycle).astype(int)

		########## generate pulse shape ##########
		### single pulse characteristics
		# pos rectangular pulses of unit amplitude, duration 0.2 ms followed by a 
		# negative counterpart of length 3 ms and amplitude 1/15, interpulse interval 1/130 s
		# positive rectangular pulse
		tStartPosPuls=0.2 # ms
		tStepStartPosPuls= int(tStartPosPuls/self.dt)
		lengthsPosRect=0.4*self.e_pulse_scale # ms
		tSteplengthsPosRect= int(lengthsPosRect/self.dt)

		# normalized such that integral over time in ms yields one
		AmpPosPuls=1.0/(0.4*self.e_pulse_scale)

		# negative pulse
		tStartNegPuls=lengthsPosRect+0.2 # ms
		tStepStartNegPuls= int(tStartNegPuls/self.dt)
		lengthsNegRect=0.8*self.e_pulse_scale # ms         # motivated by Tass et al. 2012 (monkey study)
		tSteplengthsNegRect= int(lengthsNegRect/self.dt)
		AmpNegPuls= -(AmpPosPuls*lengthsPosRect)/lengthsNegRect # ensures charge balance by scaling the amplitude

		### minimal interval between subsequent pulses is adjusted to 130 Hz DBS pulsFrequency
		pulsFrequency = 150.0 # cut off frequency accounting for finite pulse durations #  Hz
		# set lower than typical DBS frequencies to display very long pulses
		pulsPeriod=1.0/float(pulsFrequency*0.001) # ms  // approx 7.69 ms

		self.pulsLength = tStepStartNegPuls + tSteplengthsNegRect + 10 #  int( (pulsPeriod)/self.dt ) # (number of timesteps until next pulse starts 
		#self.pulsLength = tStepStartNegPuls + tSteplengthsNegRect + 39
		# print self.pulsLength

		#### in case of burst stimuli one stimulus consistes of 
		# create signal for one electrode
		self.lengthSignalOneElectrode=self.pulsLength*self.pulses_per_burst+1
		self.signalOnTrain=np.zeros(self.lengthSignalOneElectrode)


		# directory in which output for this stimulation protocol is saved
		self.signalName='multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/'+directory_phaseShifts+'/pulses_per_burst_'+str(self.pulses_per_burst)+'/fCR_'+str(self.fCR)+'_M_'+str(self.M)+'_e_pulse_scale_'+str(self.e_pulse_scale)


		# construct a single stimulus
		for ktimeSteps in range( self.pulsLength ):

			for kPulse in range( self.pulses_per_burst ):

				kStep = ktimeSteps + kPulse*self.pulsLength

				# add pos pulse
				if ( ktimeSteps < tSteplengthsPosRect+tStepStartPosPuls) and ( tStepStartPosPuls <= ktimeSteps ):

					self.signalOnTrain[kStep]=AmpPosPuls

				# add neg pulse
				if ( ktimeSteps<tSteplengthsNegRect+tStepStartNegPuls ) and ( tStepStartNegPuls <= ktimeSteps ):

					self.signalOnTrain[kStep]=AmpNegPuls


		# import matplotlib.pyplot as plt 
		# plt.scatter( self.dt*np.arange( len(self.signalOnTrain) ), self.signalOnTrain )
		# plt.show()
		# exit()


		# original currents to contact
		# rows contain currents for all time steps during one CR cycle 
		# m th row contains the currents for the m th stimulus activation in the CR cycle
		self.originalCurrentsToContacts = np.zeros( ( self.M , self.time_steps_per_cycle_period ) )
		#print self.signalOnTrain
		for k in range( self.M ):

			m = self.Sequence[k]

			for kSignal in range( len(self.signalOnTrain) ):

				if kSignal+self.stimOnsets[m] < len( self.originalCurrentsToContacts[ m ] ):
					self.originalCurrentsToContacts[ k, kSignal+self.stimOnsets[m] ] = self.signalOnTrain[kSignal]
				else:
					kSignalshifted = np.mod( kSignal+self.stimOnsets[m] , len( self.originalCurrentsToContacts[ m ] ) )
					self.originalCurrentsToContacts[ k, kSignalshifted ] = self.signalOnTrain[kSignal]

		# import matplotlib.pyplot as plt 
		# fig = plt.figure()
		# ax = fig.add_subplot(111)
		# ax.imshow (self.originalCurrentsToContacts )

		# ax.set_aspect( 10*len(self.originalCurrentsToContacts)/float(M) )
		# plt.show()
		# exit()

		return 0

	##################################
	#   function: enterNewCycle(self)
	def enterNewCycle(self, timeStep):

		# currents to contacts
		self.current_Currents = np.copy( self.originalCurrentsToContacts )

		#print self.current_Currents
		self.startNextCycle += self.time_steps_per_cycle_period

	##################################
	#   function: getCurrent( self, timeStep )
	def getCurrent( self, timeStep ):
		#print timeStep, self.startNextCycle
		if timeStep == self.startNextCycle:
			currentOutputTimeStep = 0
			self.enterNewCycle( timeStep )
		if timeStep > 0:
			currentOutputTimeStep = timeStep % self.time_steps_per_cycle_period
			#print currentOutputTimeStep
		else:
			currentOutputTimeStep = 0
		if timeStep < 0:
			#print 'exit 1'
			return 0*self.originalCurrentsToContacts[:,0]
		# return current
		#print 'exit 2'
		#print timeStep
		return self.current_Currents[ :,currentOutputTimeStep ]


class intra_burst_frequency_multiple_spikes_phase_shifted_periodic_multisite_stimulation:

	## input:           
	#   timeBetweenBursts      ... time between end of burst and beginning of next burst
	#   totalStimulationTime   ... total stimulation time in sec
	#   Nelectrodes            ... number of stimulation sites  (4 for standard CR stimultion)
	#   phase_shifts           ... array containing phases shiftes Delta alpha1-Delta till alphaM-1
	#							   values are between zero and one
	def __init__(self, fCR, totalStimulationTime, M, dt, phase_shifts, e_pulse_scale, pulses_per_burst, intraburst_frequency ):

		# time interval between end of burst and beginning of next burst in ms
		self.fCR=fCR    # Hz

		# total stimulation time in sec
		self.Tstim=totalStimulationTime*1000.0 # ms

		# number of stimulation sites
		self.M=M

		# integration time step used in simulations
		self.dt=dt # ms

		# array that contains pre-calculated stimulus
		# kth element is value of stimulation current delivered at k*dt after stimulus onset
		self.signalOnTrain=np.zeros(1)

		# number of time steps in 'signalOnTrain'
		self.lengthSignalOneElectrode=0

		# integration time step at which current stimulus is delivered (beginning of next stimulus)
		# is used to get end of current stimulus and beginning of next one
		self.CurrentStimulusOnset=np.arange(self.M) # time steps

		# directory name for output from simulations using this stimulation protocol
		self.signalName=''

		# start of next cycle
		self.startNextCycle = 0
 
		self.phase_shifts = phase_shifts

		self.e_pulse_scale = e_pulse_scale

		self.pulses_per_burst = pulses_per_burst

		self.intraburst_frequency = intraburst_frequency
		# print 'sequence'
		# print self.Sequence

	##################################
	#   function: initialize_Fixed_Sequence_CR_overlapping_Chaos_Paper(self )
	def initialize_intra_burst_frequency_multiple_spikes_phase_shifted_periodic_multisite_stimulation(self ):
	#
	#       initializes signal calculated the full signal of one electrode for one/Nelectrodes signal periods

		# number of integration time steps in a cycle
		self.time_steps_per_cycle_period = int(1000./(self.fCR * self.dt)) # steps

		self.Sequence = [0]
		# time steps of stimulus onsets
		phases_of_stimulus_onsets_in_first_cycle = np.zeros( self.M )
		# obda, first stimulus is delivered at t=0
		# now calculated the others
		directory_phaseShifts=''
		if self.M > 1:
			for k_Dalpha in range( len( self.phase_shifts ) ):
				Dalpha = self.phase_shifts[k_Dalpha]

				phases_of_stimulus_onsets_in_first_cycle[ k_Dalpha+1 ] = np.mod( phases_of_stimulus_onsets_in_first_cycle[ k_Dalpha ] + Dalpha , 1.0 )
			
				directory_phaseShifts+='Dalpha'+str(k_Dalpha+1)+'_'+str(Dalpha)+'_'
				self.Sequence.append(k_Dalpha+1)
			directory_phaseShifts=directory_phaseShifts[:-1]
		else:
			directory_phaseShifts = 'single'

		timeStepsToNextStimulusOnset = int( float( self.time_steps_per_cycle_period )/float(self.M) )
		self.stimOnsets = (self.time_steps_per_cycle_period*phases_of_stimulus_onsets_in_first_cycle).astype(int)

		########## generate pulse shape ##########
		### single pulse characteristics
		# pos rectangular pulses of unit amplitude, duration 0.2 ms followed by a 
		# negative counterpart of length 3 ms and amplitude 1/15, interpulse interval 1/130 s
		# positive rectangular pulse
		tStartPosPuls=0.2 # ms
		tStepStartPosPuls= int(tStartPosPuls/self.dt)
		lengthsPosRect=0.4*self.e_pulse_scale # ms
		tSteplengthsPosRect= int(lengthsPosRect/self.dt)

		# normalized such that integral over time in ms yields one
		AmpPosPuls=1.0/(0.4*self.e_pulse_scale)

		# negative pulse
		tStartNegPuls=lengthsPosRect+0.2 # ms
		tStepStartNegPuls= int(tStartNegPuls/self.dt)
		lengthsNegRect=0.8*self.e_pulse_scale # ms         # motivated by Tass et al. 2012 (monkey study)
		tSteplengthsNegRect= int(lengthsNegRect/self.dt)
		AmpNegPuls= -(AmpPosPuls*lengthsPosRect)/lengthsNegRect # ensures charge balance by scaling the amplitude

		### minimal interval between subsequent pulses is adjusted to 130 Hz DBS pulsFrequency
		pulsFrequency=self.intraburst_frequency #  Hz
		pulsPeriod=1.0/float(pulsFrequency*0.001) # ms  // approx 7.69 ms

		self.pulsLength = int( (pulsPeriod)/self.dt ) # (number of timesteps until next pulse starts 
		#self.pulsLength = tStepStartNegPuls + tSteplengthsNegRect + 39
		# print self.pulsLength

		#### in case of burst stimuli one stimulus consistes of 
		# create signal for one electrode
		self.lengthSignalOneElectrode=self.pulsLength*self.pulses_per_burst+1
		self.signalOnTrain=np.zeros(self.lengthSignalOneElectrode)


		# directory in which output for this stimulation protocol is saved
		self.signalName='multiple_spikes_phase_shifted_periodic_multisite_stimulation_TASS2012/'+directory_phaseShifts+'/pulses_per_burst_'+str(self.pulses_per_burst)+'/intraburst_frequency_'+str(self.intraburst_frequency)+'_fCR_'+str(self.fCR)+'_M_'+str(self.M)+'_e_pulse_scale_'+str(self.e_pulse_scale)


		# construct a single stimulus
		for ktimeSteps in range( self.pulsLength ):

			for kPulse in range( self.pulses_per_burst ):

				kStep = ktimeSteps + kPulse*self.pulsLength

				# add pos pulse
				if ( ktimeSteps < tSteplengthsPosRect+tStepStartPosPuls) and ( tStepStartPosPuls <= ktimeSteps ):

					self.signalOnTrain[kStep]=AmpPosPuls

				# add neg pulse
				if ( ktimeSteps<tSteplengthsNegRect+tStepStartNegPuls ) and ( tStepStartNegPuls <= ktimeSteps ):

					self.signalOnTrain[kStep]=AmpNegPuls


		# import matplotlib.pyplot as plt 
		# plt.scatter( self.dt*np.arange( len(self.signalOnTrain) ), self.signalOnTrain )
		# plt.show()
		# exit()


		# original currents to contact
		# rows contain currents for all time steps during one CR cycle 
		# m th row contains the currents for the m th stimulus activation in the CR cycle
		self.originalCurrentsToContacts = np.zeros( ( self.M , self.time_steps_per_cycle_period ) )
		#print self.signalOnTrain
		for k in range( self.M ):

			m = self.Sequence[k]

			for kSignal in range( len(self.signalOnTrain) ):

				if kSignal+self.stimOnsets[m] < len( self.originalCurrentsToContacts[ m ] ):
					self.originalCurrentsToContacts[ k, kSignal+self.stimOnsets[m] ] = self.signalOnTrain[kSignal]
				else:
					kSignalshifted = np.mod( kSignal+self.stimOnsets[m] , len( self.originalCurrentsToContacts[ m ] ) )
					self.originalCurrentsToContacts[ k, kSignalshifted ] = self.signalOnTrain[kSignal]

		# import matplotlib.pyplot as plt 
		# fig = plt.figure()
		# ax = fig.add_subplot(111)
		# ax.imshow (self.originalCurrentsToContacts )

		# ax.set_aspect( 10*len(self.originalCurrentsToContacts)/float(M) )
		# plt.show()
		# exit()

		return 0

	##################################
	#   function: enterNewCycle(self)
	def enterNewCycle(self, timeStep):

		# currents to contacts
		self.current_Currents = np.copy( self.originalCurrentsToContacts )

		#print self.current_Currents
		self.startNextCycle += self.time_steps_per_cycle_period

	##################################
	#   function: getCurrent( self, timeStep )
	def getCurrent( self, timeStep ):
		#print timeStep, self.startNextCycle
		if timeStep == self.startNextCycle:
			currentOutputTimeStep = 0
			self.enterNewCycle( timeStep )
		if timeStep > 0:
			currentOutputTimeStep = timeStep % self.time_steps_per_cycle_period
			#print currentOutputTimeStep
		else:
			currentOutputTimeStep = 0
		if timeStep < 0:
			#print 'exit 1'
			return 0*self.originalCurrentsToContacts[:,0]
		# return current
		#print 'exit 2'
		#print timeStep
		return self.current_Currents[ :,currentOutputTimeStep ]





################################################################
#  function: gen_wSignal_for_CR_contacts_along_x_axes for 3D cuboid volume with periodic boundary conditions
# 		sorts neurons to separately stimulated subpopulations according to their x coordinate
#
#		input:   Nelectrodes , system_parameters, pos_neurons 
#			Nelectrodes  .. number of CR electrodes
#			system_parameters ... parameter set 
#			pos_neurons	... 3d neuron positions
#
#		output: wSignalArray
#   		wSignalArray  ... matrix with 'Nelectrodes' columns and 'total number of neurons' rows containing relative stimulation amplitudes [0,1]
#							  the first len(pos_neurons) rows are filed with 1 at kColumn if kColumn*(width of electrode) <= pos_neurons[:,0] < (kColumn+1)*(width of electrode) 
#	
def gen_wSignal_for_CR_contacts_along_x_axes( Nelectrodes, system_parameters, pos_neurons ):

	# calculate total number of neurons
	N = system_parameters['N_STN'] + system_parameters['N_GPe']

	# initialize 'wSignalArray'
	# multiplying wSignalArray with the vector of currents delivered to the electrods
	# yields stimulation currents for each neuron
	wSignalArray=np.zeros( (N, Nelectrodes) )
	# width of an stimulation contact in x-direction
	widthOfElectrode = 2*system_parameters['rx_STN']/float(Nelectrodes)

	# sort neurons to electrodes
	# indices need to match with those in 'pos_neurons;
	for kNeuron in range( system_parameters['N_STN'] ):

		# sort according to electrodes
		correspondingElectrodeIndex = int( (pos_neurons[kNeuron,0]+system_parameters['rx_STN'])/widthOfElectrode )
		#print pos_neurons[kNeuron,0], pos_neurons[kNeuron,0]+system_parameters['rx_STN']
		# set relative stimulation amplitude
		wSignalArray[ kNeuron , correspondingElectrodeIndex  ]=1

	return wSignalArray



# for test runs
if __name__ == "__main__":


	# test RVS_CR_Burst_overlapping_PLOS_paper
	if sys.argv[1]=='RVS_CR_Burst_overlapping_PLOS_paper':

		# test sequence
		print( 'test RVS_CR_Burst_overlapping_PLOS_paper')
		print('parameters:')

		timeBetweenBursts=20.0 # ms
		print( 'timeBetweenBursts', timeBetweenBursts)
		
		NumberOfPulsesPerBurst=4 
		print( 'NumberOfPulsesPerBurst', NumberOfPulsesPerBurst)

		Nelectrodes=3
		print( 'Nelectrodes', Nelectrodes)

		# integration time step ( currently things are adjusted for dt=0.1 ms )
		dt=0.1 # ms

		totalStimulationTime=330.0  # ms
		fCR = 5.0 # Hz
		e_pulse_scale = 1.0
		intraburst_frequency = 60.0 # Hz
	
		print( 'totalStimulationTime' , totalStimulationTime)
									                		
		sequence=RVS_CR_Burst_overlapping_PLOS_paper( fCR, totalStimulationTime, Nelectrodes, NumberOfPulsesPerBurst, dt, e_pulse_scale, intraburst_frequency )
		sequence.initialize_RVS_CR_Burst_overlapping_PLOS_paper( )

		# prepare plot of CR sequence
		# time
		t0 = -100. # ms

		times = np.arange(t0, 1.5*totalStimulationTime, dt)
		currents=np.zeros( (len(times), Nelectrodes) )

		for kstep in range( len( times ) ):
			kstep = int(times[kstep]/dt)
			# print(sequence.getCurrent( kstep + int(t0/dt) ))
			currents[ kstep ] = sequence.getCurrent( kstep + int(t0/dt) )

		import matplotlib.pyplot as plt

		figPulseShape=plt.figure()

		# spatio temporal sequence
		ax1=figPulseShape.add_subplot(311)
		ax1.imshow( np.transpose( currents ) )
		ax1.set_aspect(  float(len( times ))/float(Nelectrodes)  ) 
		ax1.set_xlabel('t in time steps')
		ax1.set_aspect(  float(0.1*len( times ))/float(Nelectrodes)  ) 
		ax1.set_yticks(  np.arange(Nelectrodes)   ) 
		ax1.set_ylabel( 'stim site' )

		# applied current to individual neurons
		ax2=figPulseShape.add_subplot(313)

		for kNeuron in range(Nelectrodes):
			ax2.plot( times, currents[:,kNeuron] )
		
		ax2.set_xlabel('t in ms')
		plt.show()

	if sys.argv[1]=='gen_wSignal_for_CR_contacts_along_x_axes':

		nElectrodes = 5
		NSTN = 100
		system_parameters = {}

		system_parameters['N_STN'] = NSTN
		system_parameters['N_GPe'] = 10
		system_parameters['rx_STN'] = 1
		

		STNCenter = positionsSTNNeurons = np.random.uniform( -system_parameters['rx_STN'], system_parameters['rx_STN'], (NSTN,3) )
		STNCenter = STNCenter[STNCenter[:,0].argsort()]

		wSignalArray = gen_wSignal_for_CR_contacts_along_x_axes( nElectrodes, system_parameters, STNCenter )

		import matplotlib.pyplot as plt 
		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax1.imshow(wSignalArray)
		ax1.set_aspect( 1./float(NSTN) )


		print( STNCenter)

		plt.show()

		exit()


	# test phase_shifted_periodic_multisite_stimulation
	if sys.argv[1]=='multiple_spikes_phase_shifted_periodic_multisite_stimulation':

		fCR =10.0 # Hz

		totalStimulationTime = 0.5 # sec

		M = 3
		dt = 0.1 # ms
		NumberOfPulsesPerBurst =1

		phaseShifts = [0.8,0.1]
		e_pulse_scale = 20.0  # scales width of electrical pulse. for e_pulse_scale=1 we have de=0.4 ms and di = 0.8 ms
		#phaseShifts = []		

		sequence=multiple_spikes_phase_shifted_periodic_multisite_stimulation(fCR, totalStimulationTime, M, dt, phaseShifts , e_pulse_scale , NumberOfPulsesPerBurst )
		sequence.initialize_multiple_spikes_phase_shifted_periodic_multisite_stimulation( )


		totNumberOfNeurons = 22

		# prepare plot of CR sequence
		# time
		t0 = -100. # ms
		times = np.arange(t0, 1.5*totalStimulationTime*1000, dt)
		currents=np.zeros( (len(times), totNumberOfNeurons) )

		wmatrix = np.zeros( (totNumberOfNeurons,M) )
		wmatrix[1:4,0] = 1 
		wmatrix[6:18,1] = 1 
		wmatrix[18:20,2] = 1 

		for kstep in range( len( times ) ):
			kstep = int(times[kstep]/dt)
			#print kstep + int(t0/dt)
			currents[ kstep ] = wmatrix.dot( sequence.getCurrent( kstep + int(t0/dt) ) )
			#print currents[ kstep ]

		#print currents
		import matplotlib.pyplot as plt

		figPulseShape=plt.figure()

		# spatio temporal sequence
		ax1=figPulseShape.add_subplot(311)
		ax1.imshow( np.transpose( currents ), origin='lower' )
		ax1.set_aspect(  float(len( times ))/float(totNumberOfNeurons)  ) 
		ax1.set_xlabel('t in time steps')
		ax1.set_aspect(  float(0.1*len( times ))/float(totNumberOfNeurons)  ) 
		ax1.set_yticks(  np.arange(totNumberOfNeurons)   ) 
		ax1.set_ylabel( 'stim site' )

		# applied current to individual neurons
		ax2=figPulseShape.add_subplot(313)

		for kNeuron in range(totNumberOfNeurons):
			ax2.plot( times, currents[:,kNeuron] )
		
		ax2.set_xlabel('t in ms')
		plt.show()

	# test phase_shifted_periodic_multisite_stimulation
	if sys.argv[1]=='intra_burst_frequency_multiple_spikes_phase_shifted_periodic_multisite_stimulation':

		fCR =5.0 # Hz
		intraburst_frequency = 120.0 # Hz

		totalStimulationTime = 0.5 # sec

		M = 3
		dt = 0.1 # ms
		NumberOfPulsesPerBurst =3

		phaseShifts = [0.3,0.1]
		e_pulse_scale = 1.0  # scales width of electrical pulse. for e_pulse_scale=1 we have de=0.4 ms and di = 3.0 ms
		#phaseShifts = []		

		sequence=intra_burst_frequency_multiple_spikes_phase_shifted_periodic_multisite_stimulation(fCR, totalStimulationTime, M, dt, phaseShifts , e_pulse_scale , NumberOfPulsesPerBurst, intraburst_frequency )
		sequence.initialize_intra_burst_frequency_multiple_spikes_phase_shifted_periodic_multisite_stimulation( )


		totNumberOfNeurons = 20

		# prepare plot of CR sequence
		# time
		t0 = -100. # ms
		times = np.arange(t0, 1.5*totalStimulationTime*1000, dt)
		currents=np.zeros( (len(times), totNumberOfNeurons) )

		wmatrix = np.zeros( (totNumberOfNeurons,M) )
		wmatrix[1:4,0] = 1 
		wmatrix[6:18,1] = 1 
		wmatrix[18:20,2] = 1 

		for kstep in range( len( times ) ):
			kstep = int(times[kstep]/dt)
			#print kstep + int(t0/dt)
			currents[ kstep ] = wmatrix.dot( sequence.getCurrent( kstep + int(t0/dt) ) )
			#print currents[ kstep ]

		#print currents
		import matplotlib.pyplot as plt

		figPulseShape=plt.figure()

		# spatio temporal sequence
		ax1=figPulseShape.add_subplot(311)
		ax1.imshow( np.transpose( currents ), origin='lower' )
		ax1.set_aspect(  float(len( times ))/float(totNumberOfNeurons)  ) 
		ax1.set_xlabel('t in time steps')
		ax1.set_aspect(  float(0.1*len( times ))/float(totNumberOfNeurons)  ) 
		ax1.set_yticks(  np.arange(totNumberOfNeurons)   ) 
		ax1.set_ylabel( 'stim site' )

		# applied current to individual neurons
		ax2=figPulseShape.add_subplot(313)

		for kNeuron in range(totNumberOfNeurons):
			ax2.plot( times, currents[:,kNeuron] )
		
		ax2.set_xlabel('t in ms')
		plt.show()
