# adjusted from yoni leibner code
import os
import pickle
import sys
import argparse

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar
from neuron import h

from neuron_model import NeuronCell
from utils.morpholog_visualization import get_morphology, plot_morphology


def get_segment_length_lamda(seg):  # passive only
    # g_total = seg.g_pas + self.g_Ih_record_dict[sec][seg]
    R_total = 1.0 / seg.g_pas  # 1 / g_total
    lamda = np.sqrt((R_total / seg.sec.Ra) * (seg.diam / 10000.0) / 4.0)
    return (float(seg.sec.L / seg.sec.nseg) / 10000.0) / lamda


def electrical_distance(to_segment, origin_segment, tree_dendogram_dist):
    sec = to_segment.sec
    x = sec.parentseg().x
    if (sec.parentseg().sec.name(), x) not in tree_dendogram_dist.keys():
        ks = [x for (n, x) in tree_dendogram_dist.keys() if n == sec.parentseg().sec.name()]
        if len(ks) == 0:
            print("Error. Cant find nearby name for ", sec.parentseg().sec.name())
            return np.nan
        x = ks[np.argmin(abs(np.array(ks) - x))]  # patch. x return 1 when doesnt exist
    return tree_dendogram_dist[(sec.parentseg().sec.name(), x)] + \
           sum([get_segment_length_lamda(seg) for seg in sec if seg.x <= to_segment.x])  # sec length as sum of seg length


def morph_distance(to_segment, origin_segment, tree_dendogram_dist):
    h.distance(0, origin_segment.x, sec=origin_segment.sec)
    return h.distance(to_segment.x, sec=to_segment.sec)


def compute_distances(base_sec, distance_func, tree_dendogram_dist, cell):
    origin_segment = cell.soma(0.5)
    for sec in h.SectionRef(sec=base_sec).child:
        for seg in reversed([seg for seg in sec]):
            tree_dendogram_dist[(seg.sec.name(), seg.x)] = \
                distance_func(seg, origin_segment=origin_segment, tree_dendogram_dist=tree_dendogram_dist)
        compute_distances(sec, distance_func=distance_func,
                          tree_dendogram_dist=tree_dendogram_dist, cell=cell)


def get_color(sec, cell, colors_dict):
    if sec in cell.apic:
        if sec in cell.trunk:
            return colors_dict.get("trunk", colors_dict["apical"])
        elif sec in cell.oblique:
            return colors_dict.get("oblique", colors_dict["apical"])
        return colors_dict["apical"]
    elif sec in cell.basal:
        return colors_dict["basal"]
    elif sec in cell.axon:
        return colors_dict["axon"]
    elif sec in cell.soma:
        return colors_dict["soma"]
    else:
        try:
            clean_name = sec.name().split(".")[1]
            clean_name = clean_name[:clean_name.find("[")]
            for k in colors_dict.keys():
                if k == clean_name:
                    return colors_dict[k]
        except Exception as e:
            print(e)
            pass
        return colors_dict["else"]


def plot_synapse(sec_start, sec_end, pos, x_axis, ax, cell, colors_dict):
    ax.scatter(x_axis, sec_start + abs(sec_end - sec_start) * float(pos), color=colors_dict["synapse"])


def plot_vertical(x_pos, sec, parent, tree_dendogram_dist, ax, cell, colors_dict, diam_factor=None):
    curr_name = sec.name().split(".")[1].split("[")[1].replace("]", "")
    x = max([x for (n, x) in tree_dendogram_dist.keys() if n == sec.name()])
    x2 = max([x for (n, x) in tree_dendogram_dist.keys() if n == parent.name()])
    ax.text(x_pos, tree_dendogram_dist[(sec.name(), x)], curr_name)
    ax.plot([x_pos, x_pos], [tree_dendogram_dist[(parent.name(), x2)], tree_dendogram_dist[(sec.name(), x)]],
                color=get_color(sec, cell=cell, colors_dict=colors_dict),  # plot vertical
                linewidth=1 if diam_factor is None else sec.diam * diam_factor)


def plot_func(sec, x_pos, color, done_section, tree_dendogram_dist, ax, cell, colors_dict, diam_factor=None, dots_loc=[]):
    parent = h.SectionRef(sec=sec).parent

    if sec in done_section:
        raise BaseException(f"problem with morph {done_section}, {sec}")
    else:
        done_section.add(sec)
    sec_name = sec.name()
    sec_name = sec_name[sec_name.find(".") + 1:]
    if h.SectionRef(sec=sec).nchild() == 0:
        plot_vertical(x_pos, sec, parent, tree_dendogram_dist=tree_dendogram_dist, diam_factor=diam_factor, ax=ax,
                      cell=cell, colors_dict=colors_dict)
        for sec_n, loc in dots_loc:
            if sec_name == sec_n:
                x = max([x for (n, x) in tree_dendogram_dist.keys() if n == sec.name()])
                x2 = max([x for (n, x) in tree_dendogram_dist.keys() if n == parent.name()])
                plot_synapse(tree_dendogram_dist[(parent.name(), x2)], tree_dendogram_dist[(sec.name(), x)], loc, x_pos,
                             ax=ax, cell=cell, colors_dict=colors_dict)

        return x_pos + 1.0, x_pos
    elif h.SectionRef(sec=sec).nchild() == 1:
        for sec_n, loc in dots_loc:
            if sec_name == sec_n:
                x = max([x for (n, x) in tree_dendogram_dist.keys() if n == sec.name()])
                x2 = max([x for (n, x) in tree_dendogram_dist.keys() if n == parent.name()])
                plot_synapse(tree_dendogram_dist[(parent.name(), x2)], tree_dendogram_dist[(sec.name(), x)], loc, x_pos, ax=ax,
                             cell=cell, colors_dict=colors_dict)
        x_pos, start_pos = plot_func(h.SectionRef(sec=sec).child[0], x_pos, color,
                                     done_section=done_section, ax=ax, cell=cell, colors_dict=colors_dict,
                                     tree_dendogram_dist=tree_dendogram_dist, diam_factor=diam_factor)
        plot_vertical(start_pos, sec, parent, tree_dendogram_dist=tree_dendogram_dist, diam_factor=diam_factor, ax=ax,
                      cell=cell, colors_dict=colors_dict)
        return x_pos, start_pos

    x_pos, start_pos = plot_func(h.SectionRef(sec=sec).child[0], x_pos, color,
                                 done_section=done_section, ax=ax, cell=cell, colors_dict=colors_dict,
                                 tree_dendogram_dist=tree_dendogram_dist, diam_factor=diam_factor)
    for i in range(1, int(h.SectionRef(sec=sec).nchild()) - 1, 1):
        x_pos, end_pos = plot_func(h.SectionRef(sec=sec).child[i], x_pos, color,
                                   done_section=done_section, ax=ax, cell=cell, colors_dict=colors_dict,
                                   tree_dendogram_dist=tree_dendogram_dist, diam_factor=diam_factor)

    x_pos, end_pos = plot_func(h.SectionRef(sec=sec).child[int(h.SectionRef(sec=sec).nchild()) - 1], x_pos,
                               color, done_section=done_section, ax=ax, cell=cell, colors_dict=colors_dict,
                               tree_dendogram_dist=tree_dendogram_dist, diam_factor=diam_factor)
    mid_x = start_pos + abs(end_pos - start_pos) / 2.0

    plot_vertical(mid_x, sec, parent, tree_dendogram_dist=tree_dendogram_dist, diam_factor=diam_factor, ax=ax,
                  cell=cell, colors_dict=colors_dict)
    x = max([x for (n, x) in tree_dendogram_dist.keys() if n == sec.name()])
    ax.plot([start_pos, end_pos], [tree_dendogram_dist[(sec.name(), x)]] * 2, color=get_color(sec, cell, colors_dict),
            linewidth=1 if diam_factor is None else sec.diam * diam_factor)  # plot horizontal

    for sec_n, loc in dots_loc:
        if sec_name == sec_n:
            x = max([x for (n, x) in tree_dendogram_dist.keys() if n == sec.name()])
            x2 = max([x for (n, x) in tree_dendogram_dist.keys() if n == parent.name()])
            plot_synapse(tree_dendogram_dist[(parent.name(), x2)], tree_dendogram_dist[(sec.name(), x)], loc, mid_x, ax=ax,
                         cell=cell, colors_dict=colors_dict)

    return x_pos, mid_x


def plot(save_folder, tree_dendogram_dist, add, cell, colors_dict, cell_name, max_y=None, diam_factor=None, ax=None,
         ylbl=r"Distance ($\lambda$)", with_title=True, figsize=(20, 20), is_scalebar=False):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    x_pos = 0.0
    done_section = set()
    sec = h.SectionRef(sec=cell.soma).child[0]
    x_pos, start_pos = plot_func(sec, x_pos, color=get_color(sec, cell, colors_dict), done_section=done_section, ax=ax,
                                 cell=cell, tree_dendogram_dist=tree_dendogram_dist, diam_factor=diam_factor,
                                 colors_dict=colors_dict)
    if int(h.SectionRef(sec=cell.soma).nchild()) == 1:
        end_pos = start_pos + .1
    for i in range(1, int(h.SectionRef(sec=cell.soma).nchild()), 1):
        sec = h.SectionRef(sec=cell.soma).child[i]
        x_pos, end_pos = plot_func(sec, x_pos, color=get_color(sec, cell, colors_dict), done_section=done_section, ax=ax,
                                   tree_dendogram_dist=tree_dendogram_dist, diam_factor=diam_factor,
                                   cell=cell, colors_dict=colors_dict)

    ax.plot([start_pos, end_pos], [0] * 2, color=colors_dict["soma"],
            linewidth=1 if diam_factor is None else cell.soma.diam * diam_factor)
    mid_x = start_pos + abs(end_pos - start_pos) / 2.0
    ax.plot([mid_x, mid_x], [-0.01, 0], color=colors_dict["soma"],
            linewidth=1 if diam_factor is None else cell.soma.diam * diam_factor)
    ax.set_xticks([])
    ax.set_ylabel(ylbl)
    if with_title:
        ax.set_title(add + " dendogram " + ' ' + cell_name)
    existing_colors = list(set([a.get_color() for a in ax.lines]))
    legend_elements = []
    for curr_color in existing_colors:
        if curr_color in colors_dict.values():
            key = [k for k, v in colors_dict.items() if v == curr_color][0]
            legend_elements.append(Line2D([0], [0], color=colors_dict[key], lw=2, label=key))
    ax.legend(handles=legend_elements, loc="best")
    if max_y is None:
        max_y = ax.set_ylim()[1]
    min_value = -0.001 if max_y < 10 else -0.5
    # ax.set_ylim([min_value * (1 if diam_factor is None else cell.soma.diam * diam_factor), max_y])  # todo min of soma line?

    if not is_scalebar:
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    else:
        ax.axis('off')
        fixed_value = 0.1 if max_y < 10 else 100
        scale_formatter = lambda value, unit: f"{value} $\lambda$" if max_y < 10 else f"{value} {unit}"
        ax.add_artist(ScaleBar(1, "um", fixed_value=fixed_value, scale_formatter=scale_formatter,
                               location="upper left", rotation="vertical"))

    if fig is not None:  # not subplots
        for ending in [".jpg", ".pdf"]:
            plt.savefig(os.path.join(save_folder, add + "_" + cell_name + ending), dpi=300)
        plt.close()
    done_section = set()
    return max_y


def parse_input_from_command():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--path', default=default_path, help='Path to data (and output)')
    parser.add_argument('--name', default="2005142lr_1_electrophisiology_for_modelling",
                        help='Name of data file (no ending)')
    return parser.parse_args(sys.argv[1:])


def plot_g_Ih_bar(g_Ih_record_dict):
    plt.figure()
    for sec in g_Ih_record_dict.keys():
        for seg in g_Ih_record_dict[sec].keys():
            dist=h.distance(sec(seg.x))
            plt.scatter(dist, g_Ih_record_dict[sec][seg])
    plt.xlabel('distance from soma um')
    plt.ylabel('g_Ih')
    plt.show()


if __name__ == '__main__':  # todo change thickness based on diam?
    default_path = os.path.join("..", "Research", "human_rat_data", "human_recordings")
    colors_dict = {"soma": "k",
                   "apical": "blue",
                   "oblique": "cyan",
                   "trunk": "purple",
                   "basal": "r",
                   "axon": "green",
                   "else": "gold",
                   "synapse": "grey"}
    diam_factor = 1 # None
    args = parse_input_from_command()
    p = args.path
    name = args.name
    t = [c for c in p.split("/") if "_morph" in c][0].replace("_morph", "")
    pref = str(t) + "_" + name
    save_folder = os.path.join(p, 'dendogram/')

    # cell_name = "L5PC_mouse"
    # dict_pickle = './morphology_dict.pickle'
    # cell = NeuronCell(use_cvode=True, model_path=os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])),
    #                                                           "../L5PC_NEURON_simulation/"))

    model_folder = os.path.basename(p[:p.find("morphologies") - 1]).replace("/", "").replace("\\", "")
    cell_name = name.replace(".asc", "")  #"L5PC_human_cell0603_11_model_937"
    dict_pickle = './morphology_human_' + name.replace(".asc", "") + '.pickle'
    cell = NeuronCell(use_cvode=True, model_path=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0]))),
                                                                  model_folder + "/"),
                          morphologyFilename=os.path.join(p[p.find("morphologies"):], name),
                          biophysicalModelFilename=None,  # inside template already
                          name=model_folder,  # means we need to adapt params
                          templateName="Celltemplate",  # from hoc file
                          biophysicalModelTemplateFilename="generic_template.hoc")

    #cell.L5PC.delete_axon()
    cell.init_passive_params()
    cell.SPINE_START = 60  # um
    cell.change_passive_params(CM=0.5)
    #cell.L5PC.biophys()

    #cell = NeuronCell(use_cvode=True,
    #                  model_path=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0]))),
    #                                          "Human_L5PC_model/"),
    #                 morphologyFilename="morphologies/2013_03_06_cell11_1125_H41_06.asc",
    #                 biophysicalModelFilename=None,  # inside template already
    #                 name="Human_Lp5", # means we need to adapt params
    #                 templateName="cell0603_11_model_937",  # from hoc file
    #                 biophysicalModelTemplateFilename="cell0603_11_model_937.hoc")

    tree_dendogram_morph_dist = dict()
    tree_dendogram_morph_dist[(cell.soma.name(), 0.5)] = 0
    compute_distances(cell.soma, distance_func=morph_distance,
                      tree_dendogram_dist=tree_dendogram_morph_dist)
    print(tree_dendogram_morph_dist)
    tree_dendogram_electrical_dist = dict()
    tree_dendogram_electrical_dist[(cell.soma.name(), 0.5)] = 0
    compute_distances(cell.soma, distance_func=electrical_distance,
                      tree_dendogram_dist=tree_dendogram_electrical_dist)
    print(tree_dendogram_electrical_dist)

    try:
        os.mkdir(save_folder)
    except:
        pass
    # plot('dendogram/', tree_dendogram_morph_dist, add="morph", ylbl="Distance (um)",
    #      cell=cell, cell_name=cell_name, colors_dict=colors_dict, diam_factor=diam_factor)
    # plot('dendogram/', tree_dendogram_electrical_dist, add="electrical",
    #      cell=cell, cell_name=cell_name, colors_dict=colors_dict, diam_factor=diam_factor)

    with open(os.path.join(save_folder, pref + "_dend_morph_dist" + ".pickle"), "wb") as f:
        pickle.dump(tree_dendogram_morph_dist, f, protocol=2)
    with open(os.path.join(save_folder, pref + "_dend_morph_elec_dist" + ".pickle"), "wb") as f:
        pickle.dump(tree_dendogram_electrical_dist, f, protocol=2)

    import pprint
    with open(os.path.join(save_folder, pref + "_dend_morph_dist" + ".json"), "w") as f:
        pprint.pprint(tree_dendogram_morph_dist, f, indent=4)
    with open(os.path.join(save_folder, pref + "_dend_morph_elec_dist" + ".json"), "w") as f:
        pprint.pprint(tree_dendogram_electrical_dist, f, indent=4)

    #sys.exit()
    cell_data = cell.to_dict()

    seg_ind_to_xyz_coords_map, seg_ind_to_sec_ind_map, section_index, distance_from_soma, is_basal = \
        get_morphology(experiment_dict={"Params": cell_data}, experiment_table=None,
                       morphology_filename=dict_pickle)

    fig, ax = plt.subplots(1, 3, figsize=(20 * 3, 20))
    # segment_colors_in_order = np.arange(len(cell_data['allSegmentsType']))
    segment_colors_in_order = np.zeros(len(cell_data['allSegmentsType']))
    segment_colors_in_order[is_basal] = 2

    plot('dendogram/', tree_dendogram_morph_dist, add="morph", diam_factor=diam_factor,
         ax=ax[1], ylbl="Distance (um)", cell=cell, cell_name=cell_name, colors_dict=colors_dict)
    plot('dendogram/', tree_dendogram_electrical_dist, add="electrical", diam_factor=diam_factor,
         ax=ax[2], cell=cell, cell_name=cell_name, colors_dict=colors_dict)

    plot_morphology(ax[0], segment_colors_in_order, fontsize=6,
                    seg_ind_to_xyz_coords_map=seg_ind_to_xyz_coords_map)
    legend_elements = [
        Line2D([0], [0], color=colors_dict["apical"], lw=2, label="apical"),
        Line2D([0], [0], color=colors_dict["basal"], lw=2, label="basal"),
    ]
    ax[0].legend(handles=legend_elements, loc="best")
    for ending in [".jpg", ".pdf"]:
        plt.savefig(os.path.join(save_folder, pref +"_all_figs" + ending), dpi=300)
    plt.close()
