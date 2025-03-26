import enum
import logging

class ExcitatoryType(enum.Enum):
    AMPA = 1
    NMDA = 2
    NONE = 3
    def __str__(self):
        return self.name

class InhibitoryType(enum.Enum):
    GABA_A  = 1
    GABA_B  = 2
    GABA_AB = 3
    NONE    = 4
    def __str__(self):
        return self.name

class SimulationParameters:
    """General simulation parameters
    Properties are used to calculate values from adapted params
    """

    @property
    def sim_duration_sec(self):
        return self.totalSimDurationInSec

    @property
    def sim_duration_ms(self):
        return 1000 * self.sim_duration_sec

    @property
    def totalSimDurationInMS(self):
        return 1000 * self.totalSimDurationInSec

    def __init__(self, numSimulations=1, totalSimDurationInSec=1, AMPA_gMax=0.0004, NMDA_gMax=0.0004, GABA_gMax=0.001,
                 initial_voltage=-76):
        self.numSimulations = numSimulations  # 128
        self.totalSimDurationInSec = totalSimDurationInSec  # 6
        self.AMPA_gMax = AMPA_gMax
        self.NMDA_gMax = NMDA_gMax
        self.GABA_gMax = GABA_gMax
        self.initial_voltage = initial_voltage

    def to_dict(self):
        experimentParams = {}
        experimentParams['numSimulations'] = self.numSimulations
        experimentParams['AMPA_gMax'] = self.AMPA_gMax
        experimentParams['NMDA_gMax'] = self.NMDA_gMax
        experimentParams['GABA_gMax'] = self.GABA_gMax
        experimentParams['initial_voltage'] = self.initial_voltage
        experimentParams['totalSimDurationInSec'] = self.totalSimDurationInSec
        experimentParams['collectAndSaveDVTs'] = self.collectAndSaveDVTs
        experimentParams['numSamplesPerMS_HighRes'] = self.numSamplesPerMS_HighRes
        experimentParams['excitatorySynapseType'] = self.excitatorySynapseType
        experimentParams['inhibitorySynapseType'] = self.inhibitorySynapseType
        experimentParams['useActiveDendrites'] = self.useActiveDendrites
        experimentParams['Ih_vshift'] = self.Ih_vshift
        experimentParams['instRateSamplingTmIntrvsMs'] = \
            self.inst_rate_sampling_time_interval_options_ms
        experimentParams['num_bas_ex_spikes_100ms'] = self.num_bas_ex_spikes_per_100ms_range
        experimentParams['num_bas_e_i_spike_diff_100ms'] = self.num_bas_ex_inh_spike_diff_per_100ms_range
        experimentParams['num_apic_ex_spikes_100ms'] = self.num_apic_ex_spikes_per_100ms_range
        experimentParams['num_apic_e_i_spike_diff_100ms'] = self.num_apic_ex_inh_spike_diff_per_100ms_range
        return experimentParams

    collectAndSaveDVTs = True  # store dendritic voltage traces (DVTs), which take up a lot of storage
    #showPlots = False
    showPlots = True
    useCvode = True

    numSamplesPerMS_HighRes = 8  # high resolution sampling of the voltage and nexus voltages

    # synapse type
    # excitatorySynapseType = 'NMDA'  # supported options: {'AMPA','NMDA'}
    excitatorySynapseType = 'AMPA'    # supported options: {'AMPA','NMDA'}
    inhibitorySynapseType = None#'GABA_A'

    useActiveDendrites = True  # use active dendritic conductance

    # attenuation factor for the conductance of the SK channel
    SKE2_mult_factor = 1.0
    # SKE2_mult_factor = 0.1

    # determine the voltage activation curve of the Ih current (HCN channel)
    Ih_vshift = 0

    # "regularization" param for the segment lengths (mainly used to not divide by very small numbers)
    min_seg_length_um = 10.0

    # define inst rate between change interval and smoothing sigma options
    inst_rate_sampling_time_interval_options_ms = [25, 30, 35, 40, 45, 55, 60, 65, 70, 75, 80, 85, 90, 100, 150, 200,
                                                   300, 450]
    temporal_inst_rate_smoothing_sigma_options_ms = [25, 30, 35, 40, 50, 60, 80, 100, 150, 200, 300, 400, 500, 600]

    inst_rate_sampling_time_interval_jitter_range = 20
    temporal_inst_rate_smoothing_sigma_jitter_range = 20

    # todo refactor
    # number of spike ranges for the simulation

    # AMPA with attenuated SK_E2 conductance
    # num_bas_ex_spikes_per_100ms_range = [0,1900]
    # num_bas_ex_inh_spike_diff_per_100ms_range = [-1500,300]
    # num_apic_ex_spikes_per_100ms_range = [0,2000]
    # num_apic_ex_inh_spike_diff_per_100ms_range = [-1500,300]

    # AMPA
    # num_bas_ex_spikes_per_100ms_range = [0,1900]
    # num_bas_ex_inh_spike_diff_per_100ms_range = [-1650,150]
    # num_apic_ex_spikes_per_100ms_range = [0,2000]
    # num_apic_ex_inh_spike_diff_per_100ms_range = [-1650,150]

    # NMDA
    num_bas_ex_spikes_per_100ms_range = [0, 800]
    num_bas_ex_inh_spike_diff_per_100ms_range = [-600, 200]
    num_apic_ex_spikes_per_100ms_range = [0, 800]
    num_apic_ex_inh_spike_diff_per_100ms_range = [-600, 200]
