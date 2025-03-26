import glob
import json
import logging
import argparse
import os
import sys

import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from tqdm import tqdm
import numpy as np
import tifffile
from scipy import signal
import pickle
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

import neuron
from neuron import h
from neuron.units import mV, ms

from utils.dendogram import compute_distances, morph_distance, electrical_distance, plot
from utils.simulation_parameters import SimulationParameters, ExcitatoryType, InhibitoryType
from neuron_model import NeuronCell
from utils.morpholog_visualization import get_morphology, plot_morphology, plot_morphology_from_cell


def build_excitatory_inhibitory_synapses(allSegments, params, exSpikeTimesMap, inhSpikeTimesMap):
    def ConnectEmptyEventGenerator(synapse):
        netConnection = h.NetCon(None, synapse)
        netConnection.delay = 0
        netConnection.weight[0] = 1  # todo this is strength
        return netConnection

    def excitatory(seg):
        if params.excitatorySynapseType is None:
            return None
        if params.excitatorySynapseType == 'AMPA':
            return NeuronCell.DefineSynapse_AMPA(seg, gMax=params.AMPA_gMax)
        elif params.excitatorySynapseType == 'NMDA':
            return NeuronCell.DefineSynapse_NMDA(seg, gMax=params.NMDA_gMax)
        else:
            assert False, 'Not supported Excitatory Synapse Type'

    def inhibitory(seg):
        if params.inhibitorySynapseType is None:
            return None
        if params.inhibitorySynapseType == 'GABA_A':
            return NeuronCell.DefineSynapse_GABA_A(seg, gMax=params.GABA_gMax)
        elif params.inhibitorySynapseType == 'GABA_B':
            return NeuronCell.DefineSynapse_GABA_B(seg, gMax=params.GABA_gMax)
        elif params.inhibitorySynapseType == 'GABA_AB':
            return NeuronCell.DefineSynapse_GABA_AB(seg, gMax=params.GABA_gMax)
        else:
            assert False, 'Not supported Inhibitory Synapse Type'

    allSomaNetCons, allSomaNetConEventLists = [], []
    allExNetCons, allExNetConEventLists = [], []
    allInhNetCons, allInhNetConEventLists = [], []
    allSomaSynapses, allExSynapses, allInhSynapses = [], [], []

    for segInd, segment in enumerate(allSegments):
        # define synapse and connect it to a segment
        exSynapse = excitatory(segment)
        if exSynapse is not None:
            allExSynapses.append(exSynapse)
            netConnection = ConnectEmptyEventGenerator(exSynapse)  # connect synapse

            # update lists
            allExNetCons.append(netConnection)
            if segInd in exSpikeTimesMap.keys():
                allExNetConEventLists.append(exSpikeTimesMap[segInd])
            else:
                allExNetConEventLists.append([])

        # define synapse and connect it to a segment
        inhSynapse = inhibitory(segment)
        if inhSynapse is not None:
            allInhSynapses.append(inhSynapse)
            netConnection = ConnectEmptyEventGenerator(inhSynapse)  # connect synapse

            # update lists
            allInhNetCons.append(netConnection)
            if segInd in inhSpikeTimesMap.keys():
                allInhNetConEventLists.append(inhSpikeTimesMap[segInd])
            else:
                allInhNetConEventLists.append([])  # insert empty list if no event

    return allExNetCons, allExNetConEventLists, allInhNetCons, allInhNetConEventLists, allExSynapses, allInhSynapses, \
      allSomaNetCons, allSomaNetConEventLists, allSomaSynapses


def CreateCombinedColorImage(dendriticVoltageTraces, excitatoryInputSpikes, inhibitoryInputSpikes, minV=-85, maxV=35):
    excitatoryInputSpikes = signal.fftconvolve(excitatoryInputSpikes, np.ones((3, 3)), mode='same')
    inhibitoryInputSpikes = signal.fftconvolve(inhibitoryInputSpikes, np.ones((3, 3)), mode='same')

    stimulationImage = np.zeros((np.shape(excitatoryInputSpikes)[0], np.shape(excitatoryInputSpikes)[1], 3))
    stimulationImage[:, :, 0] = 0.98 * (dendriticVoltageTraces - minV) / (maxV - minV) + inhibitoryInputSpikes
    stimulationImage[:, :, 1] = 0.98 * (dendriticVoltageTraces - minV) / (maxV - minV) + excitatoryInputSpikes
    stimulationImage[:, :, 2] = 0.98 * (dendriticVoltageTraces - minV) / (maxV - minV)
    stimulationImage[stimulationImage > 1] = 1

    return stimulationImage


def PlotSimulation(selected_segment_index=635, experimentParams={}, save_dir=None, add_name=""):
    plt.figure(figsize=(30, 15))
    plt.subplot(2, 1, 1);
    plt.title('input spike trains')
    plt.imshow(CreateCombinedColorImage(dendriticVoltages, inputSpikeTrains_ex, inputSpikeTrains_inh))
    plt.subplot(2, 1, 2);
    plt.title('interpulated time - high res')
    plt.plot(recordingTimeHighRes, somaVoltageHighRes)
    plt.plot(recordingTimeHighRes, nexusVoltageHighRes)
    plt.xlim(0, params.totalSimDurationInMS)
    plt.ylabel('Voltage [mV]');
    plt.legend(['soma', 'nexus'])
    add_name += "seg{0}_".format(selected_segment_index)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, add_name + "input_spike_train_high_res.jpg"))
        plt.close()
    else:
        plt.show()

    plt.figure(figsize=(30, 15))
    plt.subplot(3, 1, 1);
    plt.title('dendritic voltage traces - low res')
    for segInd in range(len(recVoltage_allSegments)):
        plt.plot(recordingTimeLowRes, dendriticVoltages[segInd, :])
    plt.ylabel('Voltage [mV]')
    plt.subplot(3, 1, 2);
    plt.title('interpulated time - low res')
    plt.plot(recordingTimeLowRes, somaVoltageLowRes)
    plt.plot(recordingTimeLowRes, nexusVoltageLowRes)
    plt.xlabel('time [msec]');
    plt.ylabel('Voltage [mV]');
    plt.legend(['soma', 'nexus', 'soma LowRes', 'nexus LowRes'])
    plt.subplot(3, 1, 3);
    plt.title('voltage histogram')
    #plt.hist(somaVoltageHighRes.ravel(), normed=True, bins=200, color='b', alpha=0.7)
    #plt.hist(nexusVoltageHighRes.ravel(), normed=True, bins=200, color='r', alpha=0.7)
    plt.xlabel('Voltage [mV]');
    plt.legend(['soma', 'nexus'])
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, add_name + "input_spike_train_low_res.jpg"))
        plt.close()
    else:
        plt.show()

    X_exc_segment = inputSpikeTrains_ex[selected_segment_index, :]
    X_inh_segment = inputSpikeTrains_inh[selected_segment_index, :]
    y_DVT_segment = dendriticVoltages[selected_segment_index, :]
    min_voltage = y_DVT_segment.min()

    plt.figure(figsize=(20, 12))
    plt.subplot(2, 1, 1)
    plt.title('segment %d' % (selected_segment_index), fontsize=24)
    plt.plot(5 * X_exc_segment + min_voltage - 2, color='r')
    plt.plot(5 * X_inh_segment + min_voltage - 2, color='b')
    plt.plot(y_DVT_segment, color='k')
    plt.xlim(0, X_exc_segment.shape[0])
    plt.xlabel('time [ms]', fontsize=20)
    plt.ylabel('voltage [mV]', fontsize=20)
    plt.subplot(2, 1, 2)
    plt.plot(5 * X_exc_segment + min_voltage - 2, color='r')
    plt.plot(5 * X_inh_segment + min_voltage - 2, color='b')
    plt.plot(y_DVT_segment, color='k')
    plt.xlim(250, 1750)
    plt.xlabel('time [ms]', fontsize=20)
    plt.ylabel('voltage [mV]', fontsize=20)
    plt.legend(['exc input directly onto segment', 'inh input directly onto segment', 'segment voltage'], fontsize=20)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, add_name + "segment{0}.jpg".format(selected_segment_index)))
        plt.close()
    else:
        plt.show()

    seg_ind_to_xyz_coords_map, seg_ind_to_sec_ind_map, section_index, distance_from_soma, is_basal = \
        get_morphology(experiment_dict={"Params": experimentParams})
    num_segments = len(experimentParams['allSegmentsType'])

    num_nearby_segments = 30
    y_DVTs = np.zeros((num_segments, params.sim_duration_ms, 1), dtype=np.float16)
    y_DVTs[:, :, 0] = currSimulationResultsDict['dendriticVoltagesLowRes']
    corr_matrix_DVTs = np.corrcoef(y_DVTs.reshape([y_DVTs.shape[0], -1]))
    segment_distance_matrix = squareform(pdist(corr_matrix_DVTs, 'correlation'))

    sorted_segments_by_distance = np.argsort(segment_distance_matrix[selected_segment_index, :])
    selected_nearby_segment_inds = sorted_segments_by_distance[:num_nearby_segments]

    # all segments in sequential order colors and widths
    segment_colors_in_order = np.arange(num_segments)

    # nearby (similar) segments colors and widths
    segment_colors_nearby = np.zeros(segment_colors_in_order.shape)
    segment_colors_nearby[selected_nearby_segment_inds] = 1

    segment_widths_nearby = 1.2 * np.ones(segment_colors_nearby.shape)
    segment_widths_nearby[selected_nearby_segment_inds] = 3.0

    # selected index segment colors and widths
    segment_colors_selected = np.zeros(segment_colors_in_order.shape)
    segment_colors_selected[selected_segment_index] = 1.0

    segment_widths_selected = 1.2 * np.ones(segment_colors_nearby.shape)
    segment_widths_selected[selected_segment_index] = 5.0

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 12))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01, hspace=0.01, wspace=0.2)

    names = [str(i) for i in np.arange(num_segments)]
    plot_morphology(ax[0], segment_colors_selected, width_mult_factors=segment_widths_selected,
                    seg_ind_to_xyz_coords_map=seg_ind_to_xyz_coords_map, names=names)
    plot_morphology(ax[1], segment_colors_nearby, width_mult_factors=segment_widths_nearby,
                    seg_ind_to_xyz_coords_map=seg_ind_to_xyz_coords_map)
    plot_morphology(ax[2], segment_colors_in_order, seg_ind_to_xyz_coords_map=seg_ind_to_xyz_coords_map)

    ax[0].set_title('selected segment', fontsize=24)
    ax[1].set_title('"nearby" segments', fontsize=24)
    ax[2].set_title('all segments (by seq order)', fontsize=24)

    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, add_name + "neuron_morph.jpg"))
        plt.close()
    else:
        plt.show()


def dict2bin(row_inds_spike_times_map, num_segments, sim_duration_ms):
    bin_spikes_matrix = np.zeros((num_segments, sim_duration_ms), dtype='bool')
    for row_ind in row_inds_spike_times_map.keys():
        for spike_time in row_inds_spike_times_map[row_ind]:
            bin_spikes_matrix[row_ind, spike_time] = 1.0
    return bin_spikes_matrix


def bin2dict(inputSpikeTrains):
    inhSpikeSegInds, inhSpikeTimes = np.nonzero(inputSpikeTrains)
    inhSpikeTimesMap = {}
    for segInd, synTime in zip(inhSpikeSegInds, inhSpikeTimes):
        if segInd in inhSpikeTimesMap.keys():
            inhSpikeTimesMap[segInd].append(synTime)
        else:
            inhSpikeTimesMap[segInd] = [synTime]
    return inhSpikeTimesMap


def plot_morph_tiff_stack(save_dir, file_name, num_segments, seg_ind_to_xyz_coords_map, num_nearby_segments=30):
    segment_colors_in_order = np.arange(num_segments)

    # nearby (similar) segments colors and widths
    for curr_selected_segment_index in tqdm(range(len(listOfSingleSimulationDicts))):
        logging.info("{0}/{1}".format(curr_selected_segment_index, len(listOfSingleSimulationDicts)))
        currSimulationResultsDict = listOfSingleSimulationDicts[curr_selected_segment_index]

        inputSpikeTrains_ex_c = dict2bin(currSimulationResultsDict['exInputSpikeTimes'], num_segments, params.sim_duration_ms)
        inputSpikeTrains_inh_c = dict2bin(currSimulationResultsDict['inhInputSpikeTimes'], num_segments, params.sim_duration_ms)

        dendriticVoltages_c = currSimulationResultsDict['dendriticVoltagesLowRes']
        # outputSpikeTimes = currSimulationResultsDict['outputSpikeTimes']

        y_DVTs = np.zeros((num_segments, params.sim_duration_ms, 1), dtype=np.float16)
        y_DVTs[:, :, 0] = currSimulationResultsDict['dendriticVoltagesLowRes']
        corr_matrix_DVTs = np.corrcoef(y_DVTs.reshape([y_DVTs.shape[0], -1]))
        segment_distance_matrix = squareform(pdist(corr_matrix_DVTs, 'correlation'))

        sorted_segments_by_distance = np.argsort(segment_distance_matrix[curr_selected_segment_index, :])
        selected_nearby_segment_inds = sorted_segments_by_distance[:num_nearby_segments]
        segment_colors_nearby = np.zeros(segment_colors_in_order.shape)
        segment_colors_nearby[selected_nearby_segment_inds] = 1

        segment_widths_nearby = 1.2 * np.ones(segment_colors_nearby.shape)
        segment_widths_nearby[selected_nearby_segment_inds] = 3.0

        # all segments in sequential order colors and widths
        segment_colors_in_order = np.arange(num_segments)
        segment_colors_selected = np.zeros(segment_colors_in_order.shape)
        segment_colors_selected[curr_selected_segment_index] = 1.0

        segment_widths_selected = 1.2 * np.ones(segment_colors_nearby.shape)
        segment_widths_selected[curr_selected_segment_index] = 5.0

        # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 12))
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 2)
        ax = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]
        fig.subplots_adjust(left=0.01, right=0.98, top=0.96, hspace=0.01, wspace=0.2)
        names = []# [str(i) for i in np.arange(num_segments)]
        plot_morphology(ax[0], segment_colors_selected, width_mult_factors=segment_widths_selected,
                        seg_ind_to_xyz_coords_map=seg_ind_to_xyz_coords_map, names=names)

        X_exc_segment = inputSpikeTrains_ex_c[curr_selected_segment_index, :]
        X_inh_segment = inputSpikeTrains_inh_c[curr_selected_segment_index, :]
        y_DVT_segment = dendriticVoltages_c[curr_selected_segment_index, :]
        min_voltage = y_DVT_segment.min()
        ax[1].plot(5 * X_exc_segment + min_voltage - 2, color='r')
        ax[1].plot(5 * X_inh_segment + min_voltage - 2, color='b')
        ax[1].plot(y_DVT_segment, color='k')
        ax[1].set_xlim(0, X_exc_segment.shape[0])
        ax[1].set_xlabel('time [ms]', fontsize=20)
        ax[1].set_ylabel('voltage [mV]', fontsize=20)
        ax[1].set_title("Segment {0}".format(curr_selected_segment_index))

        ax[2].set_title('Interpulated time - high res')
        ax[2].plot(currSimulationResultsDict['recordingTimeHighRes'], currSimulationResultsDict['somaVoltageHighRes'])
        ax[2].plot(currSimulationResultsDict['recordingTimeHighRes'], currSimulationResultsDict['nexusVoltageHighRes'])
        ax[2].set_xlabel('Time [ms]')
        ax[2].set_ylabel('Voltage [mV]')
        ax[2].legend(['soma', 'nexus'])

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
        if curr_selected_segment_index == 0:
            tifffile.imwrite(os.path.join(save_dir, file_name + "_morph_stack.tiff"), img, append=False, bigtiff=True)
        else:
            tifffile.imwrite(os.path.join(save_dir, file_name + "_stack.tiff"), img, append=True, bigtiff=True)
        plt.close('all')
    logging.info(os.path.join(save_dir, file_name + "_morph_stack.tiff"))


def plot_simulation_plots(save_dir, results_dict, cell, species_prefix="", diam_factor=1,
                          colors_dict={"soma": "k", "apical": "blue", "oblique": "cyan", "trunk": "purple",
                                       "basal": "r", "axon": "green", "else": "gold", "synapse": "grey"}):
    import sys
    import traceback
    xlim=[390, 440] if not is_long_pulse else [190, 600]
    try:
      fig, ax = plt.subplots(3, len(results_dict.keys()), sharex=True, figsize=(30, 10))
      if len(results_dict.keys()) == 1:
        ax = np.array([ax]).T
      for i, name in enumerate(results_dict.keys()):
        ax[0, i].plot(results_dict[name]["time_soma_stim"], results_dict[name]["soma_stim_voltage_traces"].T)
        l = ax[0, i].plot(results_dict[name]["recordingTimeHighRes"][-1, :], results_dict[name]["somaVoltageHighRes"][-1, :], '--k', linewidth=2)
        ax[0, i].legend(l, ["Somatic stim"])
        ax[1, i].plot(results_dict[name]["recordingTimeHighRes"].T, results_dict[name]["somaVoltageHighRes"].T)
        ax[2, i].plot(results_dict[name]["time_soma_stim"].T, results_dict[name]["dendriticVoltagesHighRes"].T)
        ax[0, i].set_title(name)
        ax[0, i].set_ylabel("Sim 1: Dendritic traces (mV)")
        ax[1, i].set_ylabel("Sim 2: Somatic response (mV)")
        ax[2, i].set_ylabel("Sim 2: Dendritic stimulus (mV)")
        ax[2, i].set_xlabel("time (ms)")
      #ax[0, -1].set_xlim(xlim)
      ax[0, 0].get_shared_y_axes().join(ax[0, 0], *ax[0, :])
      ax[0, 0].get_shared_y_axes().join(ax[0, 0], *ax[1, :])
      ax[2, 0].get_shared_y_axes().join(ax[2, 0], *ax[2, :])
      plt.savefig(os.path.join(save_dir, species_prefix + "_all_cells_" + "visusalize_simulation.jpg"), dpi=600)
      print(os.path.join(save_dir, species_prefix + "_all_cells_" + "visusalize_simulation.jpg"))
    except Exception as e:
      pass
      #traceback.print_exc(file=sys.stdout)
      #exc_type, exc_value, exc_traceback = sys.exc_info()
      #traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
    
    try:
      for i, name in enumerate(results_dict.keys()):
        print(name, results_dict[name].keys(), len(ax), ax)
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
        ax[0].plot(results_dict[name]["time_soma_stim"], results_dict[name]["soma_stim_voltage_traces"].T)
        l = ax[0].plot(results_dict[name]["recordingTimeHighRes"][-1, :], results_dict[name]["somaVoltageHighRes"][-1, :], '--k', linewidth=2)
        ax[0].legend(l, ["Somatic stim"])
        ax[1].plot(results_dict[name]["recordingTimeHighRes"].T, results_dict[name]["somaVoltageHighRes"].T)
        ax[2].plot(results_dict[name]["time_soma_stim"].T, results_dict[name]["dendriticVoltagesHighRes"].T)
        ax[0].get_shared_y_axes().join(ax[0], ax[1])
        if is_short_pulse:
            for i in range(3):
                ax[i].plot(402, ax[i].get_ylim()[1], '*k')
        ax[0].set_title(name)
        ax[0].set_ylabel("Sim 1: Dendritic response (mV)")
        ax[1].set_ylabel("Sim 2: Somatic response (mV)")
        ax[2].set_ylabel("Sim 2: Dendritic stimulus (mV)")
        ax[2].set_xlabel("time (ms)")
        ax[0].set_xlim(xlim)
        plt.savefig(os.path.join(save_dir, species_prefix + "_" + name + "_visusalize_simulation.jpg"), dpi=600)
        plt.close('all')
        print(os.path.join(save_dir, species_prefix + "_" + name + "_visusalize_simulation.jpg"))
    except Exception as e:
      traceback.print_exc(file=sys.stdout)
      exc_type, exc_value, exc_traceback = sys.exc_info()
      traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)

    #
    try:
      for name in results_dict.keys():
        fig, ax = plt.subplots(1, 3, figsize=(10 * 3, 10))
        tree_dendogram_morph_dist = dict()
        tree_dendogram_morph_dist[(cell.soma.name(), 0.5)] = 0
        compute_distances(cell.soma, distance_func=morph_distance,
                          tree_dendogram_dist=tree_dendogram_morph_dist, cell=cell)
        tree_dendogram_electrical_dist = dict()
        tree_dendogram_electrical_dist[(cell.soma.name(), 0.5)] = 0
        compute_distances(cell.soma, distance_func=electrical_distance,
                          tree_dendogram_dist=tree_dendogram_electrical_dist, cell=cell)

        legend_elements = [Line2D([0], [0], color=colors_dict["apical"], lw=2, label="apical"),
                           Line2D([0], [0], color=colors_dict["basal"], lw=2, label="basal"),]
        ax[0].legend(handles=legend_elements, loc="best")
        plot('', tree_dendogram_morph_dist, add="morph", diam_factor=diam_factor,
             ax=ax[1], ylbl="Distance (um)", cell=cell, cell_name=name, colors_dict=colors_dict)
        plot('', tree_dendogram_electrical_dist, add="electrical", diam_factor=diam_factor,
             ax=ax[2], cell=cell, cell_name=name, colors_dict=colors_dict)
        plot_morphology_from_cell(ax[0], cell, fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, species_prefix + "_" + name +"_morph_and_distances.jpg"), dpi=600)
        plt.close('all')
    except Exception as e:
      traceback.print_exc(file=sys.stdout)
      exc_type, exc_value, exc_traceback = sys.exc_info()
      traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)

    try:
      for name in list(results_dict.keys()):
        tree_dendogram_morph_dist = dict()
        tree_dendogram_morph_dist[(cell.soma.name(), 0.5)] = 0
        compute_distances(cell.soma, distance_func=morph_distance,
                          tree_dendogram_dist=tree_dendogram_morph_dist, cell=cell)
        tree_dendogram_electrical_dist = dict()
        tree_dendogram_electrical_dist[(cell.soma.name(), 0.5)] = 0
        compute_distances(cell.soma, distance_func=electrical_distance,
                          tree_dendogram_dist=tree_dendogram_electrical_dist, cell=cell)

        # quickfix patch template number to match
        result_template_num = results_dict[name]["segment_names"][0][0].split("[")[1].split("]")[0]
        tree_dend_template_num = list(tree_dendogram_morph_dist.keys())[0][0].split("[")[1].split("]")[0]
        results_dict[name]["segment_names"] = [(k[0].replace("Celltemplate[{0}]".format(result_template_num),
                                                             "Celltemplate[{0}]".format(tree_dend_template_num)), round(k[1], 7))
                                               for k in results_dict[name]["segment_names"]]

        results_dict[name]["tree_dendogram_morph_dist"] = [np.nan] * len(results_dict[name]["segment_names"])
        results_dict[name]["tree_dendogram_elect_dist"] = [np.nan] * len(results_dict[name]["segment_names"])
        for i, (tree_dend_name, x) in enumerate(results_dict[name]["segment_names"]):
            ks = [(n, xx) for (n, xx) in tree_dendogram_morph_dist.keys() if n == tree_dend_name]
            if (tree_dend_name, x) in [(n, round(xx, 7)) for (n, xx) in ks]:
                xx = [xx for (n, xx) in ks if x == round(xx, 7)][0]
                results_dict[name]["tree_dendogram_morph_dist"][i] = tree_dendogram_morph_dist[(tree_dend_name, xx)]
                results_dict[name]["tree_dendogram_elect_dist"][i] = tree_dendogram_electrical_dist[(tree_dend_name, xx)]
            else:
                print("Missing ", (tree_dend_name, x), " in ",
                      [(n, xx) for (n, xx) in tree_dendogram_morph_dist.keys() if n == tree_dend_name])

        plt.close('all')
    except Exception as e:
      traceback.print_exc(file=sys.stdout)
      exc_type, exc_value, exc_traceback = sys.exc_info()
      traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)

    try:
      for f, f_n in zip([lambda trace, t: trace[t > 200].argmax(axis=0), 
                         lambda trace, t: trace[t > 200].argmax(axis=0),
                         lambda trace, t: abs(trace - trace.mean(axis=0))[t > 200].argmin(axis=0)], # "argmean"
                      ["peak", "peak-I", "com"]):
        for name in list(results_dict.keys()):
            # pair 1: Somatic stim:
            time_ms_response_somatic_stim = results_dict[name]["time_soma_stim"][
                f(results_dict[name]["soma_stim_voltage_traces"].T, results_dict[name]["time_soma_stim"])]
            time_ms_somatic_stim = results_dict[name]["recordingTimeHighRes"][-1, :][
                f(results_dict[name]["somaVoltageHighRes"][-1, :], results_dict[name]["recordingTimeHighRes"][-1, :])]
            if f_n != "peak-I":
                time_ms_somatic_stim_diff = time_ms_response_somatic_stim - time_ms_somatic_stim  # vector shape n segments
            else:
                time_ms_somatic_stim_diff = time_ms_response_somatic_stim - 402  # todo

            # pair 2: Dendritic stim:
            time_ms_dend_stim_diff = np.zeros((len(results_dict[name]["recordingTimeHighRes"]), 1))
            for i in range(len(results_dict[name]["recordingTimeHighRes"])):
                time_ms_response_dend_stim = results_dict[name]["recordingTimeHighRes"][i, :][
                    f(results_dict[name]["somaVoltageHighRes"][i, :], results_dict[name]["recordingTimeHighRes"][i, :])]
                time_ms_dend_stim = results_dict[name]["time_soma_stim"][
                    f(results_dict[name]["dendriticVoltagesHighRes"][i, :], results_dict[name]["time_soma_stim"])]
                if f_n != "peak-I":
                    time_ms_dend_stim_diff[i] = time_ms_response_dend_stim - time_ms_dend_stim
                else:
                    time_ms_dend_stim_diff[i] = time_ms_response_dend_stim - 402  # todo
            # print(time_ms_dend_stim, time_ms_response_dend_stim, time_ms_dend_stim_diff.T)  # todo soma stim res is not good
            inds = {}
            for (i, (n, x)) in enumerate(results_dict[name]["segment_names"]):
                inds[n] = i

            # Plot
            sec_to_vis_name = lambda n: "{1}".format(n.split("[")[0].replace("dend", "basal")[0], n.split("[")[1].replace("]", ""))
            to_name = lambda v: np.array(["{0}".format(sec_to_vis_name(n.split(".")[1]), x) for (n, x) in v])
            fontsize=3
            y_lbl = "Time diff - {0} (ms)".format(f_n)
            postfix = "_" + f_n + ".jpg"
            for inds_only in [True, False]:
                indices = list(inds.values()) if inds_only else np.arange(0, len(time_ms_dend_stim_diff))
                add = "_small" if inds_only else ""
                plt.figure()
                plt.plot(time_ms_dend_stim_diff[indices], 'o', label="Dendritic stim")
                plt.plot(time_ms_somatic_stim_diff[indices], 'o', label="Somatic stim")
                plt.xlabel("# Section"); plt.ylabel(y_lbl)
                plt.title(name)
                plt.legend()
                plt.savefig(os.path.join(save_dir, species_prefix + "_" + name + add + "_time_diff_scatter_f_number" + postfix), dpi=600)
                for field, x_lbl, filename in zip(["tree_dendogram_morph_dist", "tree_dendogram_elect_dist"],
                                                  ["Dendogram morph distance (um)", "Dendogram electrical distance ($\lambda$)"],
                                                  ["_time_diff_scatter_f_morph_dist", "_time_diff_scatter_f_elect_dist"]):
                    plt.figure()
                    x_inds = np.array(results_dict[name][field])[indices]
                    ap_inds = [ii for ii, n in enumerate(np.array(results_dict[name]["segment_names"])[indices]) if "apic" in n[0]]
                    ba_inds = [ii for ii, n in enumerate(np.array(results_dict[name]["segment_names"])[indices]) if "dend" in n[0]]
                    plt.plot(x_inds[ap_inds], time_ms_dend_stim_diff[indices][ap_inds], '.', color="blue", label="Dendritic stim (apical)")
                    plt.plot(x_inds[ba_inds], time_ms_dend_stim_diff[indices][ba_inds], '.', color="red", label="Dendritic stim (basal)")
                    plt.plot(x_inds[ap_inds], time_ms_somatic_stim_diff[indices][ap_inds], 's', mfc='none', color="blue", label="Somatic stim (apical)")
                    plt.plot(x_inds[ba_inds], time_ms_somatic_stim_diff[indices][ba_inds], 's', mfc='none', color="red", label="Somatic stim (basal)")
                    for i in range(len(indices)):
                        plt.text(np.array(results_dict[name][field])[indices][i],
                                 time_ms_dend_stim_diff[indices][i],
                                 to_name(np.array(results_dict[name]["segment_names"])[indices])[i], fontsize=fontsize)
                        plt.text(np.array(results_dict[name][field])[indices][i],
                                 time_ms_somatic_stim_diff[indices][i],
                                 to_name(np.array(results_dict[name]["segment_names"])[indices])[i], fontsize=fontsize)
                    plt.xlabel(x_lbl); plt.ylabel(y_lbl)
                    plt.title(name)
                    plt.legend(fontsize=8)
                    plt.savefig(os.path.join(save_dir, species_prefix + "_" + name + add + filename + postfix), dpi=600)

            sec_to_vis_full_name = lambda n: "{0}".format(n.split("[")[0].replace("dend", "basal").replace("apic", "apical"), n.split("[")[1].replace("]", ""))
            time_diff_df = pd.DataFrame.from_records(
                [{"name": "Dendritic stim",
                  y_lbl: time_ms_dend_stim_diff[i][0],
                  "seg_name": results_dict[name]["segment_names"][i],
                  "seg_type": sec_to_vis_full_name(results_dict[name]["segment_names"][i][0].split(".")[1]),
                  "seg_type_name": "Dendritic stim: " + sec_to_vis_full_name(results_dict[name]["segment_names"][i][0].split(".")[1]),
                  "Dendogram morph distance (um)": results_dict[name]["tree_dendogram_morph_dist"][i],
                  "Dendogram electrical distance (lambda)": results_dict[name]["tree_dendogram_elect_dist"][i]} for i in range(len(results_dict[name]["recordingTimeHighRes"]))] + \
                [{"name": "Somatic stim",
                  y_lbl: time_ms_somatic_stim_diff[i],
                  "seg_name": results_dict[name]["segment_names"][i],
                  "seg_type": sec_to_vis_full_name(results_dict[name]["segment_names"][i][0].split(".")[1]),
                  "seg_type_name": "Somatic stim: " + sec_to_vis_full_name(results_dict[name]["segment_names"][i][0].split(".")[1]),
                  "Dendogram morph distance (um)": results_dict[name]["tree_dendogram_morph_dist"][i],
                  "Dendogram electrical distance (lambda)": results_dict[name]["tree_dendogram_elect_dist"][i]} for i in range(len(results_dict[name]["recordingTimeHighRes"]))])

            plt.figure()
            sns.displot(time_diff_df, hue="name", x="Dendogram morph distance (um)", y=y_lbl)
            plt.title(name)
            plt.savefig(os.path.join(save_dir, species_prefix + "_" + name + "_time_diff_displot_f_morph_dist" + postfix), dpi=600)
            plt.figure()
            sns.displot(time_diff_df, hue="name", x="Dendogram electrical distance (lambda)", y=y_lbl)
            plt.title(name)
            plt.savefig(os.path.join(save_dir, species_prefix + "_" + name + "_time_diff_displot_f_elect_dist" + postfix), dpi=600)

            plt.figure()
            sns.displot(time_diff_df, hue="seg_type_name", x="Dendogram morph distance (um)", y=y_lbl,
                        palette="coolwarm")
            plt.title(name)
            plt.savefig(os.path.join(save_dir, species_prefix + "_" + name + "_time_diff_displot_f_morph_dist_sep" + postfix), dpi=600)
            plt.figure()
            sns.displot(time_diff_df, hue="seg_type_name", x="Dendogram electrical distance (lambda)", y=y_lbl,
                        palette="coolwarm")
            plt.title(name)
            plt.savefig(os.path.join(save_dir, species_prefix + "_" + name + "_time_diff_displot_f_elect_dist_sep" + postfix), dpi=600)
            plt.close('all')
            print("Done ", species_prefix, " ", name)
    except Exception as e:
      traceback.print_exc(file=sys.stdout)
      exc_type, exc_value, exc_traceback = sys.exc_info()
      traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)

    try:
      for i, name in enumerate(list(results_dict.keys())):
        # pair 1: Somatic stim:
        time_ms_response_somatic_stim = results_dict[name]["time_soma_stim"][results_dict[name]["soma_stim_voltage_traces"].T.argmax(axis=0)]
        time_ms_somatic_stim = results_dict[name]["recordingTimeHighRes"][-1, :][results_dict[name]["somaVoltageHighRes"][-1, :].argmax(axis=0)]
        time_ms_somatic_stim_diff = time_ms_response_somatic_stim - time_ms_somatic_stim  # vector shape n segments
        # pair 2: Dendritic stim:
        time_ms_dend_stim_diff = np.zeros((len(results_dict[name]["recordingTimeHighRes"]), 1))
        for i in range(len(results_dict[name]["recordingTimeHighRes"])):
            time_ms_response_dend_stim = results_dict[name]["recordingTimeHighRes"][i, :][results_dict[name]["somaVoltageHighRes"][i, :].argmax(axis=0)]
            time_ms_dend_stim = results_dict[name]["time_soma_stim"][results_dict[name]["dendriticVoltagesHighRes"][i, :].argmax(axis=0)]
            time_ms_dend_stim_diff[i] = time_ms_response_dend_stim - time_ms_dend_stim
        inds = {}
        for (i, (n, x)) in enumerate(results_dict[name]["segment_names"]):
            inds[n] = i

        fontsize=3
        for inds_only in [True]:
            indices = list(inds.values()) if inds_only else np.arange(0, len(time_ms_dend_stim_diff))
            add = "_small" if inds_only else ""
            for field, x_lbl, filename in zip(["tree_dendogram_morph_dist", "tree_dendogram_elect_dist"],
                                              ["Dendogram morph distance (um)", "Dendogram electrical distance ($\lambda$)"],
                                              ["_time_diff_scatter_f_morph_dist.jpg", "_time_diff_scatter_f_elect_dist.jpg"]):
                # todo these are sub traces
                min_dist = 0.05
                max_dist = 0.15
                #                 min_dist = 0.25
                #                 max_dist = 0.35
                indices_within_lambda = np.where((min_dist <= np.array(results_dict[name][field])[indices]) &
                                                 (np.array(results_dict[name][field])[indices] <= max_dist))[0]

                if len(indices_within_lambda) == 0:
                    continue

                ap_inds = [ii for ii, n in enumerate(np.array(results_dict[name]["segment_names"])[indices][indices_within_lambda]) if "apic" in n[0]]
                ba_inds = [ii for ii, n in enumerate(np.array(results_dict[name]["segment_names"])[indices][indices_within_lambda]) if "dend" in n[0]]

                to_type = lambda n: "apical" if "apic" in n[0] else ("basal" if "dend" in n[0] else "default")
                seg_types = [to_type(n) for ii, n in enumerate(np.array(results_dict[name]["segment_names"])[indices][indices_within_lambda])]
                c_to_t = {"axon": "gray", "apical": plt.cm.Blues, "basal": plt.cm.Reds, "default": plt.cm.jet}  # default jet

                segment_colors = np.arange(len(seg_types))
                segment_colors = segment_colors / segment_colors.max()
                colors = plt.cm.jet(segment_colors)
                for curr in np.unique(seg_types):
                    cmap = c_to_t[curr] if curr in c_to_t.keys() else plt.cm.jet
                    locs = np.where(curr == np.array(seg_types))[0]
                    from_ind = 1 if len(locs) == 1 else len(locs) // 3
                    segment_colors = np.arange(from_ind, from_ind + len(locs))
                    colors[locs] = cmap(segment_colors / segment_colors.max()) if not isinstance(cmap, str) else plt.cm.gray(0.5)

                from cycler import cycler
                fig = plt.figure(figsize=(5 + 10 + 10, 10))
                gs = fig.add_gridspec(3, 5)
                axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[2, 0])]
                ax_dendogram = fig.add_subplot(gs[:, 1:3])
                ax_morph = fig.add_subplot(gs[:, 3:5], adjustable='box', aspect=1)
                lines = axes[0].plot(results_dict[name]["time_soma_stim"], results_dict[name]["soma_stim_voltage_traces"][indices][indices_within_lambda].T)
                for i in range(len(lines)):
                    lines[i].set_color(colors[i])
                l = axes[0].plot(results_dict[name]["recordingTimeHighRes"][-1, :], results_dict[name]["somaVoltageHighRes"][-1, :], '--k', linewidth=2)
                axes[0].legend(l, ["Somatic stim"])
                lines = axes[1].plot(results_dict[name]["recordingTimeHighRes"][indices][indices_within_lambda, :].T,
                                     results_dict[name]["somaVoltageHighRes"][indices][indices_within_lambda, :].T)
                for i in range(len(lines)):
                    lines[i].set_color(colors[i])
                lines = axes[2].plot(results_dict[name]["time_soma_stim"].T, results_dict[name]["dendriticVoltagesHighRes"][indices][indices_within_lambda, :].T)
                for i in range(len(lines)):
                    lines[i].set_color(colors[i])
                axes[0].set_title(name)
                axes[0].set_ylabel("Sim 1: Dendritic response (mV)")
                axes[1].set_ylabel("Sim 2: Somatic response (mV)")
                axes[2].set_ylabel("Sim 2: Dendritic stimulus (mV)")
                axes[2].set_xlabel("time (ms)")
                for ax in axes:
                    ax.set_xlim(xlim)
                axes[0].get_shared_y_axes().join(axes[0], axes[1])

                plot_morphology_from_cell(ax_morph, cell, fontsize=12)

                add = "electrical" if "elect_dist" in field else "morphological"
                tree_dendogram_electrical_dist = dict()
                tree_dendogram_electrical_dist[(cell.soma.name(), 0.5)] = 0
                compute_distances(cell.soma, distance_func=electrical_distance,
                                  tree_dendogram_dist=tree_dendogram_electrical_dist, cell=cell)
                plot('', tree_dendogram_electrical_dist, add=add, diam_factor=diam_factor, ax=ax_dendogram,
                     cell=cell, cell_name=name, colors_dict=colors_dict)
                ax_dendogram.axhline(min_dist, color="k", linestyle="--")
                ax_dendogram.axhline(max_dist, color="k", linestyle="--")

                plt.suptitle(str(min_dist)+ "$\leq\lambda\leq$" + str(max_dist))
                plt.tight_layout()

                plt.savefig(os.path.join(save_dir, species_prefix + "_" + name +
                                         "_visusalize_lambda_{0}_{1}_traces.jpg".format(min_dist, max_dist)), dpi=600)
        plt.close('all')
    except Exception as e:
      traceback.print_exc(file=sys.stdout)
      exc_type, exc_value, exc_traceback = sys.exc_info()
      traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)


def GetDirNameAndFileName(numOutputSpikes, randomSeed, resultsSavedIn_rootFolder, cellType='L5PC'):  # names cant be too long in windows, it will fail
    def get_n(w):
        return w if w is not None else ""

    # string to describe model name based on params
    synapseTypes = get_n(params.excitatorySynapseType) + "_" + get_n(params.inhibitorySynapseType) + '_A_gM_' + str(params.AMPA_gMax) + '_N_gM_' + str(
        params.NMDA_gMax) + '_' + get_n(params.inhibitorySynapseType) + '_G_gM_' + str(params.GABA_gMax) + \
                   '_Iv_m' + str(abs(params.initial_voltage))
    dendritesKind = 'activeDend'
    if not params.useActiveDendrites:
        dendritesKind = 'passiveDend'
    else:
        dendritesKind += '_Ih_vshift_%d_SKE2_mult_%d' % (params.Ih_vshift, 100 * params.SKE2_mult_factor)

    modelString = '__' + dendritesKind + '__' + synapseTypes
    dirToSaveIn = resultsSavedIn_rootFolder + "full_data_dt" + str(h.dt) #modelString

    # string to describe input
    # string1 = 'exBas_%d_%d_inhBasDiff_%d_%d' %(params.num_bas_ex_spikes_per_100ms_range[0], params.num_bas_ex_spikes_per_100ms_range[1],
    #                                            params.num_bas_ex_inh_spike_diff_per_100ms_range[0], params.num_bas_ex_inh_spike_diff_per_100ms_range[1])
    # string2 = 'exApic_%d_%d_inhApicDiff_%d_%d' %(params.num_apic_ex_spikes_per_100ms_range[0], params.num_apic_ex_spikes_per_100ms_range[1],
    #                                              params.num_apic_ex_inh_spike_diff_per_100ms_range[0], params.num_apic_ex_inh_spike_diff_per_100ms_range[1])
    inputString = cellType + '__'  # string1 + '__' + string2

    # string to describe simulation
    savedDVTs = 'DVTs' if params.collectAndSaveDVTs else ''
    simulationString = 'InpSpikes_%s_%d_outSpikes_%d_simuRuns_%d_secDur_randSeed_%d' % (
    savedDVTs, numOutputSpikes, params.numSimulations, params.totalSimDurationInSec, randomSeed)

    filenameToSave = inputString + '' + simulationString + '.p'

    return dirToSaveIn, filenameToSave


def parse_input_from_command():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--seed', default=None, help='Use random seed if number not given')
    parser.add_argument('--ampa_gmax', default=AMPA_gMax, help='Default {0}'.format(AMPA_gMax))
    parser.add_argument('--nmda_gmax', default=NMDA_gMax, help='Default {0}'.format(NMDA_gMax))
    parser.add_argument('--gaba_gmax', default=GABA_gMax, help='Default {0}'.format(GABA_gMax))
    parser.add_argument('--path', default=os.path.join(".", "Human_Gabor2021", "morphologies_yoni", "human_morph"))
    parser.add_argument('--name', default="*", help='Default {0}'.format("170830HuSHS2C1IN0toIN3_postsynaptic_reconstruction_with_putative_synapses"))
    parser.add_argument('--init_v', default=initial_voltage, help='Default {0}'.format(initial_voltage))
    parser.add_argument('--scale_input_pulse', default=scale_input_pulse, help='Default {0}'.format(scale_input_pulse))
    parser.add_argument('--save_trunk', default=True, action='store_true', help='Default {0}'.format(False))
    parser.add_argument('--save_oblique', default=False, action='store_true', help='Default {0}'.format(False))
    parser.add_argument('--passive', default=False, action='store_true', help='Default {0}'.format(False))
    parser.add_argument('--flip_params', default=False, action='store_true', help='Default {0}. Flip mouse/human params'.format(False))
    parser.add_argument('--static_params', default=False, action='store_true', help='Default {0}. Static mouse/human params'.format(False))
    parser.add_argument('--long_pulse', default=False, action='store_true', help='Default {0}. Activate long pulse, not synaptic input'.format(False))
    parser.add_argument('--short_pulse', default=False, action='store_true', help='Default {0}. Activate short pulse, not synaptic input'.format(False))
    parser.add_argument('--alpha_pulse', default=False, action='store_true', help='Default {0}. Activate short alpha pulse, not synaptic input'.format(False))
    parser.add_argument('--alpha_voltage', default=False, action='store_true', help='Default {0}. Activate short alpha voltage clamp'.format(False))
    parser.add_argument('--fake_soma', default=False, action='store_true', help='Default {0}. Activate short alpha pulse, not synaptic input'.format(False))
    parser.add_argument('--steal_human_soma', default=False, action='store_true', help='Default {0}. Activate short alpha pulse, not synaptic input'.format(False))
    parser.add_argument('--steal_rat_soma', default=False, action='store_true', help='Default {0}. Activate short alpha pulse, not synaptic input'.format(False))
    parser.add_argument('--steal_name', default=None, help='Default {0}'.format("None"))
    parser.add_argument('--human', default=False, action='store_true', help='Default {0}'.format(False))
    parser.add_argument('--on_spine', default=False, action='store_true', help='Default {0}'.format(False))
    parser.add_argument('--derivative', default=False, action='store_true', help='Default {0}'.format(False))
    parser.add_argument('--spine_ra_factor', type=float, default=1, help='Default {0}'.format(1))
    parser.add_argument('--fake_name', default=None, help='Default {0}'.format("None"))
    parser.add_argument('--ex_type', default=ExcitatoryType.AMPA,
                        type=lambda v: ExcitatoryType[v], choices=list(ExcitatoryType))
    parser.add_argument('--inh_type', default=InhibitoryType.NONE,
                        type=lambda v: InhibitoryType[v], choices=list(InhibitoryType))
    return parser.parse_args(sys.argv[1:])


def get_simulation_recordings(cell, allSegments, fake_name, nexusSectionInd=0):  # 50
    # record time
    recTime = h.Vector()
    recTime.record(h._ref_t)
    # record soma voltage
    recVoltageSoma = h.Vector()
    recVoltageSoma.record(cell.soma(0.5)._ref_v)
    # record nexus voltage
    recVoltageNexus = h.Vector()
    if nexusSectionInd is not None:
        recVoltageNexus.record(cell.allSegments[nexusSectionInd]._ref_v)
    # record all segments voltage
    recVoltage_allSegments = []
    if True:#params.collectAndSaveDVTs:
        for segInd, segment in enumerate(allSegments):
            voltageRecSegment = h.Vector()
            voltageRecSegment.record(segment._ref_v)
            recVoltage_allSegments.append(voltageRecSegment)
    return recTime, recVoltageSoma, recVoltageNexus, recVoltage_allSegments, allSegments


def run_neuron(fake_name, name, args_path, skip=False, with_shrinkage_fix=True, fix_diam=True, is_on_spine=False,
               SPINE_RA_factor=1, is_fast=True):  # todo not enough space
    model_folder = os.path.dirname(args_path)
    name_full = os.path.join(args_path, name)
    name = os.path.basename(name_full).strip()
    model_part_path = os.path.dirname(os.path.dirname(args_path)) + os.sep
    morph_relative_path = model_path.replace(model_part_path, "")

    steal_human_soma, steal_rat_soma = None, None
    morph_names=dict(rat=[], human=[])
    for curr in glob.glob(os.path.join(model_folder, "*_morph")):
        if "human" in os.path.basename(curr):
            morph_names["human"] = [a for a in os.listdir(curr) if a.lower().endswith(".asc")]
        elif "rat" in os.path.basename(curr):
            morph_names["rat"] = [a for a in os.listdir(curr) if a.lower().endswith(".asc")]
        else:
            print("Error unknown folder", curr)

    assert len(morph_names["human"]) > 0 and len(morph_names["rat"]) > 0
    if is_steal_human_soma:
        steal_human_soma = [a for a in morph_names["human"] if st_human_name in a][0]
    if is_steal_rat_soma:
        steal_rat_soma = [a for a in morph_names["rat"] if st_rat_name in a][0]
    logging.info(f"Run neuron fake name: {fake_name}, name: {name}, skip? {skip}, shrinkage? {with_shrinkage_fix}"
                 f", diam? {fix_diam}, on spine? {is_on_spine} with its factor {SPINE_RA_factor}")

    morph_name = name
    if os.path.isfile(os.path.join(model_part_path, "model_morph_map.json")):  # if there is mapping - take it
        with open(os.path.join(model_part_path, "model_morph_map.json")) as f:
            map_names = json.loads(f.read())
            name_2 = name[:name.lower().find(".asc")] + ".asc"
            if name in list(map_names.keys()):
                morph_name = map_names.get(name[:name.lower().find(".asc")], name)
            elif name[:name.lower().find(".asc")] in list(map_names.keys()):
                morph_name = map_names.get(name[:name.lower().find(".asc")], name)
            elif name_2 in list(map_names.values()):
                ks = [k for (k, v) in map_names.items() if name_2 == v]
                if len(ks) == 1:
                    name = ks[0] + ".ASC"
                else:
                    raise Exception("Cant patch name ks={0} name={1}".format(ks, name))
            else:
                raise Exception("Cant patch {0} for {1} or {2}".format(name_2, list(map_names.keys()), list(map_names.values())))
    else:
        print("Doesnt have ", os.path.join(model_part_path, "model_morph_map.json"))

    biophysicalModelFilename = None
    if os.path.isfile(os.path.join(model_part_path, "generic_template.hoc")):
        main_template, template_name = "generic_template.hoc", "Celltemplate"
    elif os.path.isfile(os.path.join(model_part_path, name[:name.lower().find(".asc")] + ".hoc")):
        main_template, template_name = name[:name.lower().find(".asc")] + ".hoc", name[:name.lower().find(".asc")]
    elif "L5PC_NEURON_simulation" in model_path and os.path.isfile(os.path.join(model_part_path, "L5PCtemplate_2.hoc")):
        main_template, template_name = "L5PCtemplate_2.hoc", "L5PCtemplate"
        biophysicalModelFilename = "L5PCbiophys5b.hoc"
    else:
        raise Exception("Cant find hoc file path for " + name + " in part file " + model_part_path)

    print("Mech: ", os.path.join(model_part_path.replace("\\", "/"), "x86_64", "libnrnmech.so"))

    diam_diff_threshold_um = 0.2
    stop_trunk_at=None
    # if "L5PC_NEURON_simulation" in model_path:
    #     diam_diff_threshold_um = 0.5

    logging.info(dict(is_init_passive=is_init_passive, is_delete_axon=is_delete_axon, ih_type=ih_type,
                      is_init_active=is_init_active, is_init_hcn=is_init_hcn, is_init_sk=is_init_sk,
                      is_init_k=is_init_k, use_cvode=use_cvode, sim_with_soma=True, sim_with_axon=True,
                      with_shrinkage_fix=with_shrinkage_fix, fix_diam=fix_diam, 
                      is_active_na_kv_only=is_active_na_kv_only,
                      is_init_trunk_oblique=True, diam_diff_threshold_um=diam_diff_threshold_um, stop_trunk_at=stop_trunk_at,
    ))
    cell = NeuronCell(is_init_passive=is_init_passive, is_delete_axon=is_delete_axon, ih_type=ih_type,
                      is_init_active=is_init_active, is_init_hcn=is_init_hcn, is_init_sk=is_init_sk,
                      is_init_k=is_init_k, use_cvode=use_cvode, sim_with_soma=True, sim_with_axon=True,
                      with_shrinkage_fix=with_shrinkage_fix, fix_diam=fix_diam, 
                      is_active_na_kv_only=is_active_na_kv_only,
                      replace_soma_dendrites_with_fake=is_fake_soma,
                      steal_rat_soma=steal_rat_soma, steal_human_soma=steal_human_soma,
                      is_init_trunk_oblique=True, diam_diff_threshold_um=diam_diff_threshold_um, stop_trunk_at=stop_trunk_at,
                      model_path=model_part_path.replace("\\", "/"),
                      morphologyFilename=os.path.join(morph_relative_path, morph_name).replace("\\", "/"),
                      biophysicalModelFilename=biophysicalModelFilename,  # inside template already
                      name=model_folder,  # means we need to adapt params
                      templateName=template_name,  # from hoc file
                      biophysicalModelTemplateFilename=main_template)

    name = name.replace(".ASC", "").replace(".asc", "")
    print("Cell loaded: ", name, cell)
    cell.SPINE_START = 60  # um

    all_spines, all_spine_psds = [], []
    if is_on_spine:
        for dend in cell.apic + cell.basal:
            if cell._get_distance_between_segments(origin_segment=cell.soma(0.5), to_segment=dend(0.5)) > cell.SPINE_START:
                spines, spine_psds = cell.add_spines_to_sec(dend, SPINE_RA_factor=SPINE_RA_factor)
                all_spines.extend(spines)
                all_spine_psds.extend(spine_psds)

    # patch specific fixes per hoc (depends on passive vs active)
    scale_pulse = 1
    if "generic_template" in main_template:
        logging.info("Fixes for generic template: delete axon and init passive params with defaults")
        # cell.L5PC.delete_axon()
        # cell.init_passive_params()
        if is_active_na_kv_only:
            na_gbar_list=[na, na, 200, 150, na]
            k_gbar_list=[kv, kv, kv, kv, kv]
            cell.update_na_kv(na_gbar_list=na_gbar_list, k_gbar_list=k_gbar_list)
            logging.info(f"Update cell generic template: Na {na_gbar_list} Kv {k_gbar_list}")
    elif "L5PC_NEURON_simulation" in model_path:
        if is_passive:
            logging.info("Fixes for Etay-Hay Rat L5: passive")
            cell.L5PC.biophys(0)
        else:
            logging.info("Fixes for Etay-Hay Rat L5: active set Ih and SK to 0 and 1")
            cell.L5PC.biophys(1)
            cell.ih_type = "Ih"  # match biophysics file
            cell.set_sk_multiplicative_factor(params.SKE2_mult_factor)
            cell.set_voltage_activation_curve_of_Ih_current(params.Ih_vshift)
            cell.active_mechanisms = {'gbar': cell.get_mechanism_data(startswith_='g', endswith_='bar')}  # keep previous
            cell.set_active()

    # scale_pulse = 10
    #elif "Human_L5PC_model" not in model_path:
    #    logging.info("Fixes for Guy-Eyal human: set Ih and SK")
    #    scale_pulse=10
    elif "Human_L5PC_model" in model_path:
        logging.info("Fixes for Guy-Eyal human: none")
        # scale_pulse = 10
    if scale_input_pulse != scale_pulse:
        logging.info("Scale input pulse: {0}".format(scale_input_pulse))
        scale_pulse = scale_input_pulse

    full_passive = os.path.join(model_folder, "passive_params.json")
    if os.path.exists(full_passive):
        with open(full_passive) as f:
            data = json.load(f)
            print("Passive params ", data)
            if not is_flip_params and not is_static_params and fake_name is None:
                ks = [c for c in list(data.keys()) if c in name]
                if len(ks) == 1:
                    name=ks[0]
                    curr = data.get(name, {})
                    logging.info("Normal params. I'm {0} ({1}): {2}.".format(name, "human" if is_human else "mouse/rat", curr))
                    cell.change_passive_params(CM=curr["CM"], RM=curr["RM"], RA=curr["RA"],
                                               F_factor=curr["F_factor"], SPINE_START=cell.SPINE_START, E_PAS=-70)
                else:
                    raise Exception("Missing name {0} in passive keys.".format(name))
            else:
                if fake_name is None:
                    if is_flip_params:
                        fake_name = "180305MoSHS1C1IN2toIN3" if is_human else "171101HuSHS2C1IN0toIN1"
                    else: # static
                        fake_name = "180305MoSHS1C1IN2toIN3" if not is_human else "171101HuSHS2C1IN0toIN1"
                if fake_name in data.keys():
                    curr = data.get(fake_name, {})
                    logging.info("Flip params. I'm {0} ({1}). Flip to {2}: {3}".format(
                        name, "human" if is_human else "mouse/rat", fake_name, curr))
                    cell.change_passive_params(CM=curr["CM"], RM=curr["RM"], RA=curr["RA"],
                                               F_factor=curr["F_factor"], SPINE_START=cell.SPINE_START, E_PAS=-70)
                else:
                    raise Exception("Missing fake name {0} in passive keys.".format(fake_name))

    name = name.replace("_reconstruction", "")
    if fix_diam:
        name += "_fixed_diam"
        logging.info(f"Fixed diam. name: {name}")
    if cell.fixed_shrinkage:
        name += "_fixed_d_L"
        logging.info(f"Fixed shrinkage. name: {name}")
    if not is_human:
        #cell.set_voltage_activation_curve_of_Ih_current(params.Ih_vshift)
        #cell.set_sk_multiplicative_factor(params.SKE2_mult_factor)
        logging.info("Cell not human. Changed  Ih to {0} and SK to {1}".format(params.Ih_vshift, params.SKE2_mult_factor))

    if is_passive:
        cell.set_passive()
        logging.info("Set cell to passive")

    if initial_voltage != cell.default_v:  #
        cell.add_current_clamp(initial_voltage, params.totalSimDurationInMS)
        logging.info("Setting cell's voltage with current in soma")

    colors_dict = {"soma": "k", "apical": "darkblue", "oblique": "cyan", "trunk": "gold", "tuft": "blue",
                   "basal": "r", "axon": "grey", "else": "gold", "synapse": "green"}
    # fig, ax = plt.subplots(1, 2, figsize=(10 * 2, 10))
    # plot_morphology_from_cell(ax[0], cell, fontsize=10, color_by_type=False, with_legend=True, is_scalebar=True)
    # plot_morphology_from_cell(ax[1], cell, fontsize=10, color_by_type=True, colors_dict=colors_dict, with_legend=True, is_scalebar=True, with_text=True)
    # sapir patch
    if fake_name is not None:
        dirToSaveIn= fake_data_dir# ".\\Research\\Gabor_sim_Pas_Hum_short_p\\fake_cells_comparison\\"
        if not os.path.isdir(dirToSaveIn):
            os.makedirs(dirToSaveIn)
        add_name = f"{name.replace('_marker', '')}_{fake_name}" if name not in fake_name else fake_name
        filenameToSave = f"{add_name}_passive.jpg"
        plt.savefig(os.path.join(dirToSaveIn, filenameToSave), dpi=300)

    if skip:
        return

    if is_on_spine:
        allSegments = cell.allSegments + [sp_psd_sec(0.5) for sp_psd_sec in all_spine_psds]  # simulate both on and off spine with spine extension
        if is_fast:
            spine_segs = [sp_psd_sec(0.5) for sp_psd_sec in all_spine_psds]
            spine_segs = spine_segs[-5:] + spine_segs[:5]
            allSegments = cell.allSegments[-5:] + [a(0.5) for a in cell.trunk] + \
                          spine_segs + [sp_psd_sec.sec.parentseg().sec.parentseg().sec.parentseg() for sp_psd_sec in spine_segs]
    elif is_derivative:
        is_terminal = lambda seg: len(seg.sec.children()) == 0 and seg.x == max([a.x for a in seg.sec])
        allSegments = [a for a in cell.allSegments if (is_terminal(a) and "dend" not in str(a)) or "soma" in str(a)]
        # allSegments = allSegments[0:3] + [a for a in allSegments if "soma" in str(a)] # todo remove me
        logging.info("Derivative on teminals only")
    else:
        allSegments = cell.allSegments

    num_segments = len(allSegments)
    logging.info("Total segments: {0}. Basal {1}, Apical {2}, Somatic {3}, Axonal {4}".format(
        num_segments, len(cell.BasalSectionsList), len(cell.ApicalSectionsList),
        len(cell.SomaSectionsList), len(cell.AxonalSectionsList)))

    inputSpikeTrains_ex = np.zeros([num_segments, params.totalSimDurationInMS], dtype=bool)
    inputSpikeTrains_inh = np.zeros([num_segments, params.totalSimDurationInMS], dtype=bool)

    params.numSimulations = inputSpikeTrains_ex.shape[0]  # one sim per segment

    # Simulation result folders
    f_name = 'Gabor_sim'
    f_name += '_Pas' if is_passive else ''
    f_name += "_deriv" if is_derivative else ''
    f_name += f'_NaKv_{na}_{kv}' if is_active_na_kv_only else ''
    f_name += '_Hum' if is_human else ''
    f_name += "_flip_params" if is_flip_params else ''
    f_name += "_static_params" if is_static_params else ''
    f_name += "_long_pulse" if is_long_pulse else ''
    f_name += "_short_p" if is_short_pulse else ''
    f_name += "_alpha_p" if is_alpha_pulse else ''
    f_name += "_alpha_v" if is_alpha_voltage else ''
    f_name += "_scale_I_{0}".format(scale_pulse) if scale_pulse != 1 else ''
    f_name += "_spine" + f"_F{args.spine_ra_factor}" if args.on_spine else ''
    f_name += "_fast" if is_fast else ''
    f_name += "_fake_soma" if is_fake_soma else ''
    if not is_fake_soma:  # steal fake not constant fake
        f_name += ("_fake_soma_hum_st_" + st_human_name) if is_steal_human_soma and not is_steal_rat_soma else ''
        f_name += ("_fake_soma_rat_st_" + st_rat_name) if not is_steal_human_soma and is_steal_rat_soma else ''
        logging.info(f"Fake soma stealing: human? {is_steal_human_soma} rat? {is_steal_rat_soma} => name {f_name}")

    if "morphologies_fig1" in args.path:
        print("Using morphologies only from figure 1. path:", args.path)
        f_name += "_fig1"
        if "human_morph" in args.path:
            f_name += "_hum"
        else:
            f_name += "_rat"

    logging.info(f"F_name {f_name}")
    dd = "{0}_a_{1}_n_{2}".format(ex_type, AMPA_gMax, NMDA_gMax) if ex_type != ExcitatoryType.NONE else "{0}_g_{1}".format(in_type, GABA_gMax) if in_type != InhibitoryType.NONE else ''
    resultsSavedIn_rootFolder = ""
    save_dir = ""
    
    if sub_dir is None:  # normal outputs
        resultsSavedIn_rootFolder = os.path.join('Research', f_name, '')
        save_dir = os.path.join('Research', f_name, 'Plots', dd + 'm{0}v'.format(abs(initial_voltage)))
    else:
        resultsSavedIn_rootFolder = os.path.join('Research', f_name, sub_dir, fake_name, '')
        save_dir = os.path.join('Research', f_name, sub_dir, fake_name, 'Plots', dd + 'm{0}v'.format(abs(initial_voltage)))

    inner_plots_save_dir = os.path.join(save_dir, "per_simulation")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir(inner_plots_save_dir):
        os.makedirs(inner_plots_save_dir)

    # collect results
    totalNumOutputSpikes = 0
    numOutputSpikesPerSim = []
    listOfISIs = []
    listOfSingleSimulationDicts = []

    ## run all simulations
    experimentStartTime = time.time()
    print('-------------------------------------\\')
    print('temperature is %.2f degrees celsius' % (h.celsius))
    print('dt is %.4f ms' % (h.dt))
    print('-------------------------------------/')
    search_result_soma = [(curr[0], allSegments[curr[0]]) for curr in [np.where([b == a for b in allSegments])[0]
                                                                       for a in cell.soma]][0]
    try:
        if len(cell.trunk) > 0:
            search_result_trunk = [(curr[0], allSegments[curr[0]]) for curr in [np.where([b == a for b in allSegments])[0]
                                                                                for a in cell.trunk[-1]]]
            search_result_trunk = max(search_result_trunk, key=lambda segment: segment[1].x)  # tuple (index, segment)
            nexus_sec_ind = search_result_trunk[0]
        else:
            search_result_trunk = [(curr[0], allSegments[curr[0]]) for curr in [np.where([b == a for b in allSegments])[0]
                                                                                for a in cell.apic[-1]]]
            search_result_trunk = max(search_result_trunk, key=lambda segment: segment[1].x)  # tuple (index, segment)
            nexus_sec_ind = search_result_trunk[0]
    except:
        search_result_trunk = [0, allSegments[0]]
        nexus_sec_ind = search_result_trunk[0]
    # search_apical = [(curr, allSegments[curr]) for curr in np.where(["apic[10]" in b.sec.name() for b in allSegments])[0]]
    # search_apical = max(search_apical, key=lambda segment: segment[1].x)  # tuple (index, segment)
    is_minimal = False
    is_super_minimal = False #True
    for simInd in tqdm(range(params.numSimulations)):
        if is_super_minimal and simInd != search_result_soma[0] and simInd != search_result_trunk[0]: # \
            #and simInd != search_apical[0]:
            listOfSingleSimulationDicts.append({})
            logging.debug(f"Skip index {simInd} which is not soma/trunk/apical (super minimal run) {allSegments[simInd]}")
            continue
        logging.info(f"Sim index {simInd} which {allSegments[simInd]}")
        currSimulationResultsDict = {}
        preparationStartTime = time.time()
        print('...')
        print('------------------------------------------------------------------------------\\')
        # redefine input spikes - 639 x sec*1000 (segments x time) - only has 1 spike per simulation
        inputSpikeTrains_ex[:] = 0
        inputSpikeTrains_inh[:] = 0
        if not is_long_pulse and not is_short_pulse and not is_alpha_pulse and not is_alpha_voltage:
            logging.info("Index {0} delay {1} of synaptic input".format(simInd, delay_ms))
            if ex_type != ExcitatoryType.NONE:
                inputSpikeTrains_ex[simInd, delay_ms] = True
            if in_type != InhibitoryType.NONE:
                inputSpikeTrains_inh[simInd, delay_ms] = True
        else:
            seg = allSegments[simInd]
            delay_ms_ = 200 if is_long_pulse else delay_ms
            dur = 900-delay_ms_ if is_long_pulse else 2  # 2 todo sapir 10 doesnt help
            if is_alpha_voltage:
                cell.add_alpha_voltage_stim(seg, delay_ms=delay_ms_, dur_from_delay_ms=dur, amp_ns=25 * scale_pulse,
                                            base_voltage=initial_voltage)  # voltage height
                logging.info("Index {0} delay {1} of segment {2} input {3}ms {4} voltage init {5}".format(
                    simInd, delay_ms, seg, dur, 25 * scale_pulse, initial_voltage))
            elif not is_alpha_pulse:
                cell.add_current_stim(seg, delay_ms=delay_ms_, dur_from_delay_ms=dur, amp_ns=0.5 * scale_pulse)  # 0.03
                logging.info("Index {0} delay {1} of segment {2} input {3}ms {4}nS".format(simInd, delay_ms, seg, dur, 0.5 * scale_pulse))
            else:
                cell.add_alpha_current_stim(seg, delay_ms=delay_ms_, dur_from_delay_ms=dur, amp_ns=1.5 * scale_pulse,
                                            tau0=5, tau1=8)
                logging.info("Index alpha {0} delay {1} of segment {2} input {3}ms {4}nS".format(simInd, delay_ms, seg, dur, 0.5 * scale_pulse))

        ## convert binary vectors to dict of spike times for each seg ind
        exSpikeTimesMap = bin2dict(inputSpikeTrains_ex)
        inhSpikeTimesMap = bin2dict(inputSpikeTrains_inh)

        allExNetCons, allExNetConEventLists, allInhNetCons, allInhNetConEventLists, allExSynapses, allInhSynapses, \
            allSomaNetCons, allSomaNetConEventLists, allSomaSynapses = \
            build_excitatory_inhibitory_synapses(allSegments, params, exSpikeTimesMap, inhSpikeTimesMap)

        # define function to be run at the begining of the simulation to add synaptic events
        def AddAllSynapticEvents():  # todo doesnt work
            for exNetCon, eventsList in zip(allExNetCons, allExNetConEventLists):
                for eventTime in eventsList:
                    exNetCon.event(eventTime)
            for inhNetCon, eventsList in zip(allInhNetCons, allInhNetConEventLists):
                for eventTime in eventsList:
                    inhNetCon.event(eventTime)

        # add voltage and time recordings
        recTime, recVoltageSoma, recVoltageNexus, recVoltage_allSegments, rec_segments = \
            get_simulation_recordings(cell, allSegments=allSegments if not is_derivative else cell.allSegments,
                                      fake_name=fake_name, nexusSectionInd=nexus_sec_ind)
        print("preparing for single simulation took %.4f seconds" % (time.time() - preparationStartTime))

        if fake_name is None:
            add_name = name.split("_fixed")[0]
        else:
            add_name = f"{name.replace('_marker', '')}_{fake_name}" if name not in fake_name else fake_name

        ## simulate the cell
        simulationStartTime = time.time()
        # make sure the following line will be run after h.finitialize()
        if not is_long_pulse and not is_short_pulse and not is_alpha_pulse and not is_alpha_voltage:  # todo doesnt
            fih = h.FInitializeHandler('nrnpython("AddAllSynapticEvents()")')
        h.finitialize(params.initial_voltage * mV)
        print("Init neuron with {0}V and run {1}ms".format(params.initial_voltage, params.totalSimDurationInMS))
        h.continuerun(params.totalSimDurationInMS * ms)  # todo before interp this is 70 points for small dt
        print("single simulation took %.2f minutes" % ((time.time() - simulationStartTime) / 60))

        # collect all relevent recoding vectors (input spike times, dendritic voltage traces, soma voltage trace)
        collectionStartTime = time.time()

        origRecordingTime = np.array(recTime.to_python())
        origSomaVoltage = np.array(recVoltageSoma.to_python())
        origNexusVoltage = np.array(recVoltageNexus.to_python())
        logging.info("Orig shapes: Time {0}, SomaV {1}, NexusV {2}".format(origRecordingTime.shape, origSomaVoltage.shape, origNexusVoltage.shape))

        # high res - origNumSamplesPerMS per ms
        recordingTimeHighRes = np.arange(0, params.totalSimDurationInMS, 1.0 / params.numSamplesPerMS_HighRes)
        somaVoltageHighRes = np.interp(recordingTimeHighRes, origRecordingTime, origSomaVoltage)
        nexusVoltageHighRes = np.interp(recordingTimeHighRes, origRecordingTime, origNexusVoltage)

        # low res - 1 sample per ms
        recordingTimeLowRes = np.arange(0, params.totalSimDurationInMS)
        # somaVoltageLowRes = np.interp(recordingTimeLowRes, origRecordingTime, origSomaVoltage)
        # nexusVoltageLowRes = np.interp(recordingTimeLowRes, origRecordingTime, origNexusVoltage)

        if params.collectAndSaveDVTs:
            dendriticVoltages = np.zeros((len(recVoltage_allSegments), recordingTimeLowRes.shape[0]))
            dendriticVoltagesHighRes = np.zeros((len(recVoltage_allSegments), recordingTimeHighRes.shape[0]))
            for segInd, recVoltageSeg in enumerate(recVoltage_allSegments):
                dendriticVoltages[segInd, :] = np.interp(recordingTimeLowRes, origRecordingTime,
                                                         np.array(recVoltageSeg.to_python()))
                dendriticVoltagesHighRes[segInd, :] = np.interp(recordingTimeHighRes, origRecordingTime,
                                                         np.array(recVoltageSeg.to_python()))
            logging.info("Max dendritic V {0}, max somatic V {1}".format(dendriticVoltagesHighRes.max(), somaVoltageHighRes.max()))

        if is_fast:  # debug fast spine check
            curr_n = ".".join(str(allSegments[simInd]).split(".")[1:]) if "Celltemplate" in str(allSegments[simInd]) else str(allSegments[simInd])
            f, axes = plt.subplots(1, 2, sharex=True, sharey=True)
            fr, to = np.where(recordingTimeHighRes >= (delay_ms - 20))[0][0], np.where(recordingTimeHighRes >= 450)[0][0]
            axes[0].plot(recordingTimeHighRes[fr:to], somaVoltageHighRes[fr:to], label="soma")
            axes[1].plot(recordingTimeHighRes[fr:to], dendriticVoltagesHighRes[:, fr:to].T)
            if "spine" in curr_n:
                spine_parent = allSegments[simInd].sec.parentseg().sec.parentseg().sec.parentseg()
                simInd2 = np.where([str(b) == str(spine_parent) for b in allSegments])[0][0]
                axes[0].plot(recordingTimeHighRes[fr:to],
                             dendriticVoltagesHighRes[simInd, fr:to], "--", label=curr_n)
                curr_n2 = ".".join(str(allSegments[simInd2]).split(".")[1:]) if "Celltemplate" in str(
                    allSegments[simInd2]) else str(allSegments[simInd2])
                axes[0].plot(recordingTimeHighRes[fr:to], dendriticVoltagesHighRes[simInd2, fr:to], "--", label=curr_n2)
            else:
                axes[0].plot(recordingTimeHighRes[fr:to], dendriticVoltagesHighRes[simInd, fr:to], "--", label=curr_n)
            plt.title(f"{simInd}: {curr_n}")
            axes[0].legend()
            plt.savefig(f"for_sapir/{add_name}_{simInd}_{curr_n.replace('[','_').replace(']','_').replace('(','_').replace(')','')}.jpg", dpi=300)
        # detect soma spike times
        risingBefore = np.hstack((0, somaVoltageHighRes[1:] - somaVoltageHighRes[:-1])) > 0
        fallingAfter = np.hstack((somaVoltageHighRes[1:] - somaVoltageHighRes[:-1], 0)) < 0
        localMaximum = np.logical_and(fallingAfter, risingBefore)
        largerThanThresh = somaVoltageHighRes > -25

        binarySpikeVector = np.logical_and(localMaximum, largerThanThresh)
        outputSpikeTimes = recordingTimeHighRes[np.nonzero(binarySpikeVector)]

        currSimulationResultsDict['recordingTimeHighRes'] = recordingTimeHighRes.astype(np.float64)
        currSimulationResultsDict['somaVoltageHighRes'] = somaVoltageHighRes.astype(np.float64)
        currSimulationResultsDict['nexusVoltageHighRes'] = nexusVoltageHighRes.astype(np.float64)

        #currSimulationResultsDict['recordingTimeLowRes'] = recordingTimeLowRes.astype(np.float32)
        #currSimulationResultsDict['somaVoltageLowRes'] = somaVoltageLowRes.astype(np.float16)
        #currSimulationResultsDict['nexusVoltageLowRes'] = nexusVoltageLowRes.astype(np.float16)

        #currSimulationResultsDict['exInputSpikeTimes'] = exSpikeTimesMap
        #currSimulationResultsDict['inhInputSpikeTimes'] = inhSpikeTimesMap
        #currSimulationResultsDict['outputSpikeTimes'] = outputSpikeTimes.astype(np.float16)

        if params.collectAndSaveDVTs:
            # currSimulationResultsDict['dendriticVoltagesLowRes'] = dendriticVoltages.astype(np.float32)
            currSimulationResultsDict['dendriticVoltagesHighRes'] = dendriticVoltagesHighRes.astype(np.float64)
            if is_minimal and not is_fast and not is_derivative and simInd != search_result_soma[0] and simInd != search_result_trunk[0]: # and simInd != search_apical[0]:
                # currSimulationResultsDict['dendriticVoltagesLowRes'] = dendriticVoltages.astype(np.float32)[]
                currSimulationResultsDict['dendriticVoltagesHighRes'] = dendriticVoltagesHighRes.astype(np.float64)[simInd, :]

        if (is_short_pulse or is_alpha_pulse or is_alpha_voltage) and is_cut_by_time:
            minu, pls = 10, 100
            if not is_alpha_voltage:
                delay_in_ms, dur = cell.icl.delay, cell.icl.dur
            else:
                delay_in_ms, dur = 0, cell.icl.dur1
            if is_alpha_pulse or is_alpha_voltage:
                delay_in_ms, dur = cell.alpha_params["delay_ms"], cell.alpha_params["dur_from_delay_ms"]
            t_vec = currSimulationResultsDict['recordingTimeHighRes'].copy()
            for field in ["recordingTimeHighRes", "dendriticVoltagesHighRes", "somaVoltageHighRes", "nexusVoltageHighRes"]:
                if len(currSimulationResultsDict[field].shape) == 2:
                    currSimulationResultsDict[field] = \
                        currSimulationResultsDict[field][:, (t_vec >= delay_in_ms - minu) &
                                                            (t_vec <= delay_in_ms + dur + pls)]
                else:
                    currSimulationResultsDict[field] = \
                        currSimulationResultsDict[field][(t_vec >= delay_in_ms - minu) &
                                                         (t_vec <= delay_in_ms + dur + pls)]

        numOutputSpikes = len(outputSpikeTimes)
        numOutputSpikesPerSim.append(numOutputSpikes)
        listOfISIs += list(np.diff(outputSpikeTimes))

        listOfSingleSimulationDicts.append(currSimulationResultsDict)

        print("data collection per single simulation took %.4f seconds" % (time.time() - collectionStartTime))
        print('-----------------------------------------------------------')
        print('finished simulation %d: num output spikes = %d' % (simInd + 1, numOutputSpikes))
        print("entire simulation took %.2f minutes" % ((time.time() - preparationStartTime) / 60))
        print('------------------------------------------------------------------------------/')
        print("dendriticVoltagesHighRes shape ", currSimulationResultsDict['dendriticVoltagesHighRes'].shape)

        # show the results
        if params.collectAndSaveDVTs and params.showPlots:
            # plt.close('all')
            experimentParams = params.to_dict()
            experimentParams.update(cell.to_dict())
            PlotSimulation(experimentParams=experimentParams, save_dir=inner_plots_save_dir,
                           selected_segment_index=simInd)

    # %% all simulations have ended, pring some statistics
    totalNumOutputSpikes = sum(numOutputSpikesPerSim)
    totalNumSimulationSeconds = params.totalSimDurationInSec * params.numSimulations
    averageOutputFrequency = totalNumOutputSpikes / float(totalNumSimulationSeconds)
    ISICV = np.std(listOfISIs) / np.mean(listOfISIs)
    entireExperimentDurationInMinutes = (time.time() - experimentStartTime) / 60

    # calculate some collective meassures of the experiment
    print('-------------------------------------------------\\')
    print("entire experiment took %.2f minutes" % (entireExperimentDurationInMinutes))
    print('-----------------------------------------------')
    print('total number of collected spikes is ' + str(totalNumOutputSpikes))
    print('average output frequency is %.2f [Hz]' % (averageOutputFrequency))
    print('number of spikes per simulation minute is %.2f' % (totalNumOutputSpikes / entireExperimentDurationInMinutes))
    print('ISI-CV is ' + str(ISICV))
    print('-------------------------------------------------/')
    sys.stdout.flush()

    # create a simulation parameters dict
    experimentParams = params.to_dict()
    experimentParams['randomSeed'] = randomSeed
    experimentParams.update(cell.to_dict())
    experimentParams['ISICV'] = ISICV
    experimentParams['listOfISIs'] = listOfISIs
    experimentParams['numOutputSpikesPerSim'] = numOutputSpikesPerSim
    experimentParams['totalNumOutputSpikes'] = totalNumOutputSpikes
    experimentParams['totalNumSimulationSeconds'] = totalNumSimulationSeconds
    experimentParams['averageOutputFrequency'] = averageOutputFrequency
    experimentParams['entireExprDurationInMin'] = entireExperimentDurationInMinutes

    # the important things to store
    experimentResults = {}
    #experimentResults['listOfSingleSimulationDicts'] = listOfSingleSimulationDicts

    num_segments = len(experimentParams['allSegmentsType'])

    dirToSaveIn, filenameToSave = GetDirNameAndFileName(totalNumOutputSpikes, randomSeed, cellType=name,
                                                        resultsSavedIn_rootFolder=resultsSavedIn_rootFolder)
    if not os.path.exists(dirToSaveIn):
        os.makedirs(dirToSaveIn)
    logging.info("Done expr. Save in folder {0}".format(dirToSaveIn))

    # pickle everything
    experimentDict = {'Params': experimentParams, 'Results': experimentResults}
    # this is too big

    # sapir patch
    if fake_name is not None:
        dirToSaveIn= fake_data_dir  #".\\Research\\Gabor_sim_Pas_Hum_short_p\\fake_cells_comparison\\"
        if not os.path.isdir(dirToSaveIn):
            os.makedirs(dirToSaveIn)
        add_name = f"{name.replace('_marker', '')}_{fake_name}" if name not in fake_name else fake_name
        filenameToSave = f"{add_name}_passive.p"

    #pickle.dump(experimentDict, open(os.path.join(dirToSaveIn, filenameToSave), "wb"), protocol=2)
    pickle.dump(experimentParams, open(os.path.join(dirToSaveIn, filenameToSave.replace(".p", "_params.p")), "wb"),
                protocol=2)
    logging.info("Done saving. Check {0} {1}".format(os.path.join(dirToSaveIn, filenameToSave), os.path.join(dirToSaveIn, filenameToSave.replace(".p", "_params.p"))))

    # save results subset (combine multiple simulations as one file)
    soma_stim_voltage_traces, time_soma_stim = None, None
    trunk_stim_voltage_traces, time_trunk_stim = None, None
    far_apic_stim_voltage_traces, time_far_apic_stim = None, None
    #voltage_traces = np.zeros([len(listOfSingleSimulationDicts), params.totalSimDurationInMS])
    voltage_traces = np.zeros([len(listOfSingleSimulationDicts), len(listOfSingleSimulationDicts[search_result_soma[0]]['dendriticVoltagesHighRes'][0, :])])
    if is_on_spine:
        voltage_traces = np.zeros([len(listOfSingleSimulationDicts), len(
            listOfSingleSimulationDicts[search_result_soma[0]]['dendriticVoltagesHighRes'][0, :]), 2])
    elif is_derivative:
        data_shape = listOfSingleSimulationDicts[search_result_soma[0]]['dendriticVoltagesHighRes'].shape
        voltage_traces = np.zeros([len(listOfSingleSimulationDicts), data_shape[0], data_shape[1]])
    somaVhighRes = np.zeros([len(listOfSingleSimulationDicts), len(listOfSingleSimulationDicts[search_result_soma[0]]['somaVoltageHighRes'])])
    nexusVhighRes = np.zeros([len(listOfSingleSimulationDicts), len(listOfSingleSimulationDicts[search_result_soma[0]]['nexusVoltageHighRes'])])
    timeHighRes = np.zeros([len(listOfSingleSimulationDicts), len(listOfSingleSimulationDicts[search_result_soma[0]]['recordingTimeHighRes'])])
    voltage_traces[:] = np.nan
    somaVhighRes[:] = np.nan
    nexusVhighRes[:] = np.nan
    timeHighRes[:] = np.nan
    segment_names = [(seg.sec.name(), seg.x) for seg in (allSegments if not is_derivative else cell.allSegments)]
    print(segment_names)
    if np.array([a in allSegments for a in cell.soma]).all():  # unique case we save soma-stim response in all tree
        search_result = [np.where([b == a for b in allSegments])[0] for a in cell.soma][0]
        if len(search_result) == 1:  # found soma
            sim_ind = search_result[0]
            soma_stim_voltage_traces = listOfSingleSimulationDicts[sim_ind]['dendriticVoltagesHighRes']
            time_soma_stim = listOfSingleSimulationDicts[sim_ind]['recordingTimeHighRes']
            logging.info("Found somatic stim simulation for ind {0}. Result shape {1} time {2}".format(
                sim_ind, soma_stim_voltage_traces.shape, time_soma_stim.shape))
    if is_save_trunk:
        try:
            search_result_t = [(curr[0], allSegments[curr[0]]) for curr in [np.where([b == a for b in allSegments])[0]
                                                                            for a in cell.trunk[-1]]]
            search_result_t = max(search_result_t, key=lambda segment: segment[1].x)  # tuple (index, segment)
            sim_ind = search_result_t[0]
            trunk_stim_voltage_traces = listOfSingleSimulationDicts[sim_ind]['dendriticVoltagesHighRes']
            time_trunk_stim = listOfSingleSimulationDicts[sim_ind]['recordingTimeHighRes']
            logging.info("Found trunk stim simulation for ind {0} ({1}). Result shape {2} time {3}".format(
                sim_ind, search_result_t[1], trunk_stim_voltage_traces.shape, time_trunk_stim.shape))
        except Exception as e:
            print(e)
    # try:
    #     sim_ind = search_apical[0]
    #     far_apic_stim_voltage_traces = listOfSingleSimulationDicts[sim_ind]['dendriticVoltagesHighRes']
    #     time_far_apic_stim = listOfSingleSimulationDicts[sim_ind]['recordingTimeHighRes']
    #     logging.info("Found far apic stim simulation for ind {0} ({1}). Result shape {2} time {3}".format(
    #         sim_ind, search_apical[1], far_apic_stim_voltage_traces.shape, time_far_apic_stim.shape))
    # except Exception as e:
    #     print(e)

    if not is_super_minimal:  # skip simulation for most values (reduce mem)
        for i in range(len(listOfSingleSimulationDicts)):
            if params.collectAndSaveDVTs:
                if listOfSingleSimulationDicts[i]['dendriticVoltagesHighRes'] is None:
                    logging.error("None neuron {0}".format(i))
                else:
                    if is_minimal and not is_fast:  # todo!
                        voltage_traces[i, :] = listOfSingleSimulationDicts[i]['dendriticVoltagesHighRes']
                    elif is_derivative:
                        voltage_traces[i, :, :] = listOfSingleSimulationDicts[i]['dendriticVoltagesHighRes']
                    elif is_on_spine:
                        curr_inds = [np.where([str(b) == str(a) for b in allSegments])[0] for a in allSegments][i]
                        if len(curr_inds) != 1:
                            print(f"Error on spine indices {curr_inds}")
                            voltage_traces[i, :, 0] = listOfSingleSimulationDicts[i]['dendriticVoltagesHighRes'][i, :]
                            voltage_traces[i, :, 1] = np.nan
                        elif len(curr_inds) == 1 and "spine" in str(allSegments[curr_inds[0]]):
                            spine_parent = allSegments[curr_inds[0]].sec.parentseg().sec.parentseg().sec.parentseg()
                            spine_parent_ind = np.where([str(b) == str(spine_parent) for b in allSegments])[0]
                            voltage_traces[i, :, 0] = listOfSingleSimulationDicts[i]['dendriticVoltagesHighRes'][i, :]
                            if len(spine_parent_ind) != 1:
                                print(f"Error on spine indices {curr_inds} parent {spine_parent_ind}")
                                voltage_traces[i, :, 1] = np.nan
                            else:
                                voltage_traces[i, :, 1] = \
                                    listOfSingleSimulationDicts[i]['dendriticVoltagesHighRes'][spine_parent_ind[0], :]
                                print(f"{allSegments[curr_inds[0]]} parent {allSegments[spine_parent_ind[0]]}")
                        else:
                            voltage_traces[i, :, 0] = listOfSingleSimulationDicts[i]['dendriticVoltagesHighRes'][i, :]
                            voltage_traces[i, :, 1] = np.nan
                        print(voltage_traces[i, :, 0])
                        print(voltage_traces[i, :, 1])
                    else:
                        voltage_traces[i, :] = listOfSingleSimulationDicts[i]['dendriticVoltagesHighRes'][i, :]
            somaVhighRes[i, :] = listOfSingleSimulationDicts[i]['somaVoltageHighRes']
            nexusVhighRes[i, :] = listOfSingleSimulationDicts[i]['nexusVoltageHighRes']
            timeHighRes[i, :] = listOfSingleSimulationDicts[i]['recordingTimeHighRes']
    results_dict = {'dendriticVoltagesHighRes': voltage_traces, 'recordingTimeHighRes': timeHighRes,
                    'somaVoltageHighRes': somaVhighRes, 'nexusVoltageHighRes': nexusVhighRes,
                    'soma_stim_voltage_traces': soma_stim_voltage_traces, 'time_soma_stim': time_soma_stim,
                    'trunk_stim_voltage_traces': trunk_stim_voltage_traces, 'time_trunk_stim': time_trunk_stim,
                    'far_apic_stim_voltage_traces': far_apic_stim_voltage_traces, 'time_far_apic_stim': time_far_apic_stim,
                    'segment_names': segment_names}
    if is_alpha_pulse or is_alpha_voltage:
        results_dict["alpha_time"] = np.array(cell.alpha_time_vec.to_python())
        results_dict["alpha_current"] = np.array(cell.alpha_current_vec.to_python())
    with open(os.path.join(dirToSaveIn, filenameToSave.replace(".p", "_results.p")), "wb") as f:
        pickle.dump(results_dict, f, protocol=2)
        print("Saved ", os.path.join(dirToSaveIn, filenameToSave.replace(".p", "_results.p")))
        #save_mat_dict(os.path.join(dirToSaveIn, filenameToSave.replace(".p", "_results.mat")),
        #              {'dendriticVoltagesLowRes': voltage_traces, 'recordingTimeHighRes': timeHighRes,
        #               'somaVoltageHighRes': somaVhighRes, 'nexusVoltageHighRes': nexusVhighRes})

    return
    try:
        if sub_dir is None:  # normal outputs
            save_dir = os.path.join(dirToSaveIn, "vis")
        else:
            save_dir = os.path.join('Research', f_name, sub_dir, fake_name + "_vis", '')  # currently can ignore the other params - help download faster
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print("Plots dir: ", save_dir)
        plot_simulation_plots(save_dir, {"": results_dict}, cell, species_prefix=("human" if is_human else "mouse") + "_" + name)
    except Exception as e:
        print(e)

    seg_ind_to_xyz_coords_map, seg_ind_to_sec_ind_map, section_index, distance_from_soma, is_basal = \
        get_morphology(experiment_dict={"Params": experimentParams}, cell=cell)
    print("Printing:")
    #plot_morph_tiff_stack(save_dir, filenameToSave.replace(".p", ""), num_segments, seg_ind_to_xyz_coords_map)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    os.chdir(os.path.abspath(os.path.dirname(sys.argv[0])))

    is_fast = True

    # Defaults
    AMPA_gMax = 0.0004
    NMDA_gMax = 0.0004
    GABA_gMax = 0.001
    initial_voltage = -70  # cell's normal voltage
    ex_type = ExcitatoryType.AMPA
    in_type = InhibitoryType.NONE
    is_passive = False
    is_save_oblique, is_save_trunk = False, False
    is_shrinkage_fix=True
    is_human = False
    is_flip_params = False
    is_static_params = False
    is_long_pulse = False
    is_short_pulse = False
    is_alpha_pulse = False
    is_alpha_voltage = False
    is_fake_soma = False
    is_steal_human_soma = False
    is_steal_rat_soma = False
    is_derivative = False # few simulations but full traces for velocity derivative
    fake_name = None
    sub_dir = None  # for fake cells
    scale_input_pulse = 1
    st_human_name="2006252lr_4"
    st_rat_name="2207271lr_1"

    if len(sys.argv) > 1:
        args = parse_input_from_command()
        print(args)
        if args.seed is None:
            randomSeed = np.random.randint(100000)
            print('randomly choose seed - %d' % (randomSeed))
        else:
            randomSeed = args.seed
        AMPA_gMax = float(args.ampa_gmax)
        NMDA_gMax = float(args.nmda_gmax)
        GABA_gMax = float(args.gaba_gmax)
        initial_voltage = float(args.init_v)
        ex_type = args.ex_type
        in_type = args.inh_type
        is_passive = args.passive
        is_save_oblique = args.save_oblique
        is_save_trunk = args.save_trunk
        is_flip_params = args.flip_params
        is_static_params = args.static_params
        is_long_pulse = args.long_pulse
        is_short_pulse = args.short_pulse
        is_alpha_pulse = args.alpha_pulse
        is_alpha_voltage = args.alpha_voltage
        is_fake_soma = args.fake_soma
        is_steal_human_soma = args.steal_human_soma
        is_steal_rat_soma = args.steal_rat_soma
        is_derivative = args.derivative
        if args.steal_name is not None:
          if is_steal_human_soma:
            st_human_name=args.steal_name
          elif is_steal_rat_soma:
            st_rat_name=args.steal_name
          print("Stealing name ", args.steal_name, " Human? ", is_steal_human_soma, " , Rat? ", is_steal_rat_soma)
        is_human = args.human
        is_fast = is_fast and args.on_spine   # not good for normal beause I caused it to collect all data
        fake_name = args.fake_name
        scale_input_pulse = float(args.scale_input_pulse) if (0.1 <= float(args.scale_input_pulse) and float(args.scale_input_pulse) < 20) else 1
        sub_dir = None if fake_name is None else "fake_cells"
        print('random seed selected by user - %d' % (randomSeed))
    else:
        randomSeed = np.random.randint(100000)
        print('randomly choose seed - %d' % (randomSeed))

    is_init_passive = True  # is_passive - we init this for active as well
    is_delete_axon = False  # todo: delete cause error in nearest search - why?
    is_init_k = False
    is_init_hcn = False
    is_init_sk = False
    is_init_active = not is_passive
    is_active_na_kv_only = is_init_active and True
    na, kv = 8, 10
    ih_type = "Ih"
    use_cvode = False  #True
    h.dt = 1/ 100  #100

    randomSeed = 79598  # int(sys.argv[1])
    np.random.seed(randomSeed)

    if args.path.endswith(os.path.sep) and os.sep != "/":
        args.path = args.path[:-1]
    model_path = os.path.join(args.path, "")

    if "L5PC_NEURON_simulation" in model_path:
        diam_diff_threshold_um=0.5
        if not is_passive:
            is_init_active = False

    initial_voltage = -70  # cell's normal voltage
    if "L5PC_NEURON_simulation" in model_path:
        initial_voltage = -90  # cell's normal voltage
    elif "Human_L5PC_model" in model_path:
        initial_voltage = -70  # cell's normal voltage

    params: SimulationParameters = SimulationParameters(numSimulations=1, totalSimDurationInSec=1, AMPA_gMax=AMPA_gMax,
                                                        NMDA_gMax=NMDA_gMax, GABA_gMax=GABA_gMax,
                                                        initial_voltage=initial_voltage)
    is_cut_by_time = True
    delay_ms = 400  # spike time (this simulaiton has only 1 spike and this timepoint)
    params.numSamplesPerMS_HighRes = 1 / h.dt  # the interpolation should not change this if we set dt
    print("params.numSamplesPerMS_HighRes: ", params.numSamplesPerMS_HighRes)
    params.showPlots = False
    # params.useActiveDendrites = False  # for name

    params.inhibitorySynapseType = str(in_type) if in_type != InhibitoryType.NONE else None
    params.excitatorySynapseType = str(ex_type) if ex_type != ExcitatoryType.NONE else None
    logging.info(params.to_dict())

    inner_f_name = "Gabor_sim_Pas_Hum_short_p"
    if "morphologies_fig1" in args.path:
        inner_f_name = "Gabor_Pas_fig1_short"
        is_delete_axon = True
        if "human_morph" in args.path:
            inner_f_name += "_hum"
        else:
            inner_f_name += "_rat"
    if is_alpha_pulse or is_alpha_voltage:
        inner_f_name = "Gabor_Pas_Hum_alpha_" + ("p" if is_alpha_pulse else "v")
        if "morphologies_fig1" in args.path:
            is_delete_axon = True
            inner_f_name = "Gabor_Pas_fig1_alpha" + ("p" if is_alpha_pulse else "v")
            if "human_morph" in args.path:
                inner_f_name += "_hum"
            else:
                inner_f_name += "_rat"
    if args.on_spine:
        inner_f_name += ("_spine" + f"_F{args.spine_ra_factor}" )
    if args.derivative:
        inner_f_name += "_deriv"
    if args.fake_soma:
        inner_f_name += "_fake_soma"
    elif args.steal_human_soma:
        inner_f_name += "_fake_soma_hum_st_" + st_human_name
    elif args.steal_rat_soma:
        inner_f_name += "_fake_soma_rat_st_" + st_rat_name
    inner_f_name += "_fast" if is_fast else ''

    fake_data_dir = os.path.join(".", "Research", inner_f_name, "fake_cells_comparison", f"dt_{h.dt}", "")
    if is_shrinkage_fix:
        fake_data_dir = os.path.join(fake_data_dir, "shrinkage", "")
    if fake_name is not None:
        print(f"Fake data dir {fake_data_dir}")
    if args.name == "*":
        for name in os.listdir(args.path):
            skip = False
            if name.lower().endswith(".asc"):
                # sapir patch
                if fake_name is not None:
                    dirToSaveIn=fake_data_dir
                    if not os.path.isdir(dirToSaveIn):
                        os.makedirs(dirToSaveIn)
                    add_name = f"{name.replace('_reconstruction_marker.ASC', '')}_{fake_name}" if name not in fake_name else fake_name
                    add_name2 = f"{name.replace('_reconstruction_marker.ASC', '')}_fixed_diam_{fake_name}" if name not in fake_name else fake_name
                    if os.path.exists(os.path.join(dirToSaveIn, add_name + "_passive_results.p")) or \
                        os.path.exists(os.path.join(dirToSaveIn, add_name2 + "_passive_results.p")):
                        print("Skip fake ", add_name)
                        # skip=True
                run_neuron(fake_name=fake_name, name=name, args_path=args.path, skip=skip, SPINE_RA_factor=float(args.spine_ra_factor),
                           with_shrinkage_fix=is_shrinkage_fix, is_on_spine=args.on_spine, is_fast=is_fast)
    else:
        run_neuron(fake_name=fake_name, name=args.name, args_path=args.path, is_on_spine=args.on_spine, is_fast=is_fast,
                   SPINE_RA_factor=float(args.spine_ra_factor))
