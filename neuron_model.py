import json
import logging
import traceback

import numpy as np
from neuron import h
import os


def get_class_members(class_instance, with_values=True):
    if with_values:
        return [(k, getattr(class_instance, k)) for (k, v) in vars(class_instance).items()]
    return [k for (k, v) in vars(class_instance).items()]


def patch_sections_connected_to_soma(morphologyFilename, all_sections, is_logger=True):
    fix = []
    if '2005201lr_6' in morphologyFilename:
        fix = [(("apic[57]", 0), ("apic[40]", 0.2)),
               (("apic[56]", 0), ("apic[38]", 0.33)),
               (("apic[0]", 0), ("apic[33]", 1))]
        # todo apic[59] apic[58] are soma children and shouldn't be.apic[52] apic[51] are off?
    elif "2007091lr_2" in morphologyFilename:
        fix = []

    for (son, parent) in fix:
        son_sec = [a for a in all_sections if son[0] in a.name()]
        parent_sec = [a for a in all_sections if parent[0] in a.name()]
        if not (len(son_sec) == 1 and len(parent_sec) == 1):
            print(f"Error. Not found {son} or {parent} in cell to fix it")
            continue
        h.disconnect(sec=son_sec[0])
        son_sec[0].connect(parent_sec[0], parent[1], son[1])
    if len(fix) > 0:
        print("Done patch ", morphologyFilename)


def fix_diameter_manually(morphologyFilename, all_sections, is_logger=True):
    fix = []
    if '2006252lr_4' in morphologyFilename:
        fix = [("dend[9]", 0.03, None), ("dend[19]", 0.03, None), ("dend[21]", 0.045, None), ("dend[22]", 0.045, None),
               ("dend[79]", 0.038, None), ("dend[76]", 0.045, None), ("dend[26]", 0.033, None)]
    elif "2006253lr_3" in morphologyFilename:
        fix = [("dend[42]", 0.27, None), ("dend[43]", 0.033, None), ("dend[40]", 0.5, None),
               ("dend[52]", 0.23, None), ("dend[52]", 0.37, None), ("dend[52]", 0.5, None), ("dend[52]", 0.63, None)]
    elif "2012111lr_2" in morphologyFilename:
        fix2 = [("apic[50]", 0.1, 1.1), ("apic[50]", 0.3, 1.1), ("apic[50]", 0.5, 1.1),
               ("apic[50]", 0.7, 1.1), ("apic[50]", 0.9, 1.1),
               ("apic[53]", 0.1, 1.2), ("apic[53]", 0.3, 1.2), ("apic[53]", 0.5, 1.2),
               ("apic[53]", 0.7, 1.2), ("apic[53]", 0.9, 1.2), ("apic[57]", 0.1, 1.1),
               ("apic[36]", 0.17, 1.35), ("apic[36]", 0.5, 1.33), ("apic[36]", 0.83, 1.3), ("apic[37]", 0.5, 1.2),
               ("apic[49]", 0.17, 1.35), ("apic[49]", 0.5, 1.33), ("apic[49]", 0.83, 1.3),
               ("apic[47]", 0.1, 1), ("apic[46]", 0.07, 1),
                # ("dend[38]", 0.5, None), ("dend[39]", 0.1, 0.85), ("dend[36]", 0.5, 1.05), ("dend[30]", 0.5, 1.1),
               # ("dend[37]", 0.17, None), ("dend[37]", 0.5, .8), ("dend[37]", 0.7, None),
               # ("apic[36]", 0.17, 1.45), ("apic[49]", 0.17, 1.45), ("apic[36]", 0.83, 1.41), ("apic[49]", 0.83, 1.41),
               # ("apic[36]", 0.5, 1.43), ("apic[49]", 0.5, 1.43), ("apic[35]", 0.5, 1.55),
               ("apic[59]", 0.17, 1.75), ("apic[59]", 0.5, 1.5)]
    for sec in all_sections:
        if sec.name().split(".")[1] in [a for a, x, d in fix]:
            wanted_as = [x for a, x, d in fix if sec.name().split(".")[1] in a]
            segs_to_fix = [seg for w in wanted_as for seg in sec if np.isclose(seg.x, w, atol=1e-2)]
            # print(sec, wanted_as, segs_to_fix, [seg for seg in sec])
            for seg in segs_to_fix:
                fix_to = [d for a, x, d in fix if sec.name().split(".")[1] in a and np.isclose(seg.x, x, atol=1e-2)]
                next_seg = [s for s in sec if s.x > seg.x]
                prev_diam = seg.diam
                if len(fix_to) == 1 and fix_to[0] is not None:
                    seg.diam = fix_to[0]
                elif len(next_seg) > 0:
                    seg.diam = next_seg[0].diam
                if is_logger:
                    logging.info(f"Fixed diam {seg} {prev_diam} to {seg.diam}")
                else:
                    print(f"Fixed diam {seg} {prev_diam} to {seg.diam}")  # for ipparallel we need print


def fix_shrinkage(all_sections, axon_sections, morphologyFilename="", length_scale=None, diameter_scale=None,
                  chunkSize=40, is_logger=True):  # here for reuse
    for sec in [s for s in all_sections if s not in axon_sections]:  # axon is deleted usually so wont touch it
        if length_scale is not None:
            sec.L *= length_scale
            sec.nseg = int(sec.L / chunkSize) * 2 + 1  # same as in generic_template.hoc with checkSize=40
        if diameter_scale is not None:
            for seg in sec:
                seg.diam *= diameter_scale
    if length_scale is not None:
        if is_logger:
            logging.info(f"Scaled L by {length_scale} for {morphologyFilename}")
        else:
            print(f"Scaled L by {length_scale} for {morphologyFilename}")
    if diameter_scale is not None:
        if is_logger:
            logging.info(f"Scaled d by {diameter_scale} for {morphologyFilename}")
        else:
            print(f"Scaled d by {diameter_scale} for {morphologyFilename}")


def fix_morph(sim, icell):
    if not hasattr(sim, "data_for_morph_fix") or not isinstance(sim.data_for_morph_fix, dict) or \
      "cell_name" not in sim.data_for_morph_fix.keys() or \
      "cell_fixed_shrinkage_params" not in sim.data_for_morph_fix.keys():  # passed data
        logging.error("Cant fix morph for cell")
        return

    morphologyFilename = sim.data_for_morph_fix["cell_name"]
    length_scale = sim.data_for_morph_fix["cell_fixed_shrinkage_params"]["L"]
    diameter_scale = sim.data_for_morph_fix["cell_fixed_shrinkage_params"]["d"]

    soma_sec_list = [icell.soma[x] for x in range(len(icell.soma))]
    axonal_sec_list = [icell.axon[x] for x in range(len(icell.axon))]
    basal_sec_list = [icell.dend[x] for x in range(len(icell.dend))]
    apical_sec_list = [icell.apic[x] for x in range(len(icell.apic))]

    all_sections = basal_sec_list + apical_sec_list + soma_sec_list + axonal_sec_list
    fix_shrinkage(all_sections, axonal_sec_list, morphologyFilename=morphologyFilename,
                  length_scale=length_scale, diameter_scale=diameter_scale, chunkSize=40, is_logger=False)
    fix_diameter_manually(morphologyFilename, all_sections, is_logger=False)


class CellMorph:
    @property
    def soma(self):
        assert(len(self.SomaSectionsList) == 1)
        return self.SomaSectionsList[0]

    @property
    def all(self):
        return self.BasalSectionsList + self.ApicalSectionsList + self.SomaSectionsList + self.AxonalSectionsList

    @property
    def basal(self):
        return self.BasalSectionsList

    @property
    def apic(self):
        return self.ApicalSectionsList

    @property
    def axon(self):
        return self.AxonalSectionsList

    @property
    def oblique(self):
        if hasattr(self, "oblique_list"):
            return self.oblique_list
        return []

    @property
    def trunk(self):
        if hasattr(self, "trunk_list"):
            return self.trunk_list
        return []

    @property
    def tuft(self):
        if hasattr(self, "tuft_list"):
            return self.tuft_list
        return []

    @property
    def SomaSectionsList(self):
        return [self.L5PC.soma[x] for x in range(len(self.L5PC.soma))]

    @property
    def BasalSectionsList(self):
        return [self.L5PC.dend[x] for x in range(len(self.L5PC.dend)) if self.L5PC.dend[x].name() not in self.deleted_secs]

    @property
    def ApicalSectionsList(self):
        return [self.L5PC.apic[x] for x in range(len(self.L5PC.apic)) if self.L5PC.apic[x].name() not in self.deleted_secs]

    @property
    def AxonalSectionsList(self):
        return [self.L5PC.axon[x] for x in range(len(self.L5PC.axon))]

    def __init__(self, hoc_morph_instance):
        self.L5PC = hoc_morph_instance
        self.markers = []
        self.deleted_secs = []
        self.stolen_secs = []


class NeuronCell:  # todo refactor to extract common/generic things?
    """Wraps the basic neuron loading and model creation
    """
    @property
    def SomaSectionsList(self):
        return [curr for curr in self.L5PC.somatic]
        # return [self.L5PC.soma[x] for x in range(len(self.L5PC.soma))]

    @property
    def soma(self):
        assert(len(self.SomaSectionsList) == 1)
        return self.SomaSectionsList[0]

    @property
    def all(self):
        return self.BasalSectionsList + self.ApicalSectionsList + self.SomaSectionsList + self.AxonalSectionsList

    @property
    def basal(self):
        return self.BasalSectionsList

    @property
    def apic(self):
        return self.ApicalSectionsList

    @property
    def axon(self):
        return self.AxonalSectionsList

    @property
    def oblique(self):
        if hasattr(self, "oblique_list"):
            return self.oblique_list
        return []

    @property
    def trunk(self):
        if hasattr(self, "trunk_list"):
            return self.trunk_list
        return []

    @property
    def tuft(self):
        if hasattr(self, "tuft_list"):
            return self.tuft_list
        return []

    @property
    def BasalSectionsList(self):
        return [curr for curr in self.L5PC.basal if curr.name() not in self.deleted_secs] + [a for a in self.stolen_secs if "dend" in a.name()]

    @property
    def ApicalSectionsList(self):  # todo
        return [curr for curr in self.L5PC.apical if curr.name() not in self.deleted_secs] + [a for a in self.stolen_secs if "apic" in a.name()]

    @property
    def AxonalSectionsList(self):
        return [curr for curr in self.L5PC.axonal if curr.name() not in self.deleted_secs]

    # @property
    # def BasalSectionsList(self):
    #     return [self.L5PC.dend[x] for x in range(len(self.L5PC.dend))
    #             if self.L5PC.dend[x].name() not in self.deleted_secs] + [a for a in self.stolen_secs if "dend" in a.name()]
    #
    # @property
    # def ApicalSectionsList(self):
    #     return [self.L5PC.apic[x] for x in range(len(self.L5PC.apic))
    #             if self.L5PC.apic[x].name() not in self.deleted_secs] + [a for a in self.stolen_secs if "apic" in a.name()]
    #
    # @property
    # def AxonalSectionsList(self):
    #     return [self.L5PC.axon[x] for x in range(len(self.L5PC.axon))]

    @property
    def numBasalSegments(self):
        return len(self.basal_seg_length_um)

    @property
    def numApicalSegments(self):
        return len(self.apical_seg_length_um)

    @property
    def totalNumSegments(self):
        """Just wraps basal + apical sum"""
        return self.numBasalSegments + self.numApicalSegments

    @property
    def totalBasalDendriticLength(self):
        return np.sum(self.basal_seg_length_um)

    @property
    def totalApicalDendriticLength(self):
        return np.sum(self.apical_seg_length_um)

    @property
    def totalDendriticLength(self):
        """Just wraps basal + apical sum"""
        return self.totalBasalDendriticLength + self.totalApicalDendriticLength

    def __init__(self, sim_with_soma=False, sim_with_axon=False, ih_type="Ih", with_shrinkage_fix=True, fix_diam=True,
                 is_init_trunk_oblique=False, diam_diff_threshold_um=0.2, stop_trunk_at=None, is_init_k=False,  # diam_diff_threshold_um=0.5 for etay
                 is_delete_axon=False, is_init_passive=False, is_init_active=False, is_init_hcn=False, is_init_sk=False,
                 is_active_na_kv_only=False, replace_soma_dendrites_with_fake=False,
                 steal_rat_soma=None, steal_human_soma=None,
                 na_gbar_list=[6, 6, 10, 200, 6], k_gbar_list=[5, 5, 5, 5, 5], # basal, apical, soma, axon, tuft 
                 use_cvode=False, model_path="./L5PC_NEURON_simulation",
                 templateName='L5PCtemplate',
                 name = "L5PC_NEURON_simulation",
                 morphologyFilename="morphologies/cell1.ASC",
                 biophysicalModelFilename="L5PCbiophys5b.hoc",
                 biophysicalModelTemplateFilename="L5PCtemplate_2.hoc"):
        h.load_file('nrngui.hoc')
        h.load_file("import3d.hoc")
        h.load_file("stdrun.hoc")

        self.deleted_secs = []
        self.stolen_secs = []
        # Import2 = h.Import3d_SWC_read()
        # Import2.input("C:\\Users\\sapir\\Downloads\\" + "2006253lr_3_fitted-000.swc")
        # Out[9]: 0.0
        # imprt2 = h.Import3d_GUI(Import2, 0)
        # imprt2.instantiate(None)

        if os.path.isfile(os.path.join(model_path, "x86_64", "libnrnmech.so")):
            self.mods_dll = os.path.join(model_path, "x86_64", "libnrnmech.so")
        elif os.path.isfile(os.path.join(model_path, "nrnmech.dll")):
            self.mods_dll = os.path.join(model_path, "nrnmech.dll")
        if not "CaDynamics_E2" in dir(h):
            if os.path.isfile(os.path.join(model_path, "x86_64", "libnrnmech.so")):
                h.nrn_load_dll(os.path.join(model_path, "x86_64", "libnrnmech.so"))
                self.mods_dll = os.path.join(model_path, "x86_64", "libnrnmech.so")
                logging.info(f"Loaded {os.path.join(model_path, 'x86_64', 'libnrnmech.so')}")
            elif os.path.isfile(os.path.join(model_path, "nrnmech.dll")):
                h.nrn_load_dll(os.path.join(model_path, "nrnmech.dll"))
                self.mods_dll = os.path.join(model_path, "nrnmech.dll")
                logging.info(f"Loaded {os.path.join(model_path, 'nrnmech.dll')}")
            else:
                logging.error("Missing dll/so mechanisms")
        if biophysicalModelFilename is not None:
            if os.path.isfile(os.path.join(model_path, biophysicalModelFilename)):
                h.load_file(os.path.join(model_path, biophysicalModelFilename))
            else:
                logging.error("Missing biophysics file {0}".format(os.path.join(model_path, biophysicalModelFilename)))
        if not os.path.isfile(os.path.join(model_path, biophysicalModelTemplateFilename)):
            raise Exception("Missing model template {0}".format(os.path.join(model_path, biophysicalModelTemplateFilename)))
        if hasattr(h, templateName):
            pass
            # delattr(h, templateName)
        if morphologyFilename.startswith("/"):
            morphologyFilename = morphologyFilename[1:]
        h.load_file(os.path.join(model_path, biophysicalModelTemplateFilename))
        logging.info("Loading {0}".format(os.path.join(model_path, morphologyFilename)))
        cell_temp_func = getattr(h, templateName)  # todo can be automatic in the template as cell?
        # self.L5PC = h.L5PCtemplate(os.path.join(model_path, morphologyFilename))
        self.L5PC = cell_temp_func(os.path.join(model_path, morphologyFilename))
        logging.info("Loaded {0}".format(morphologyFilename))
        self.morphologyFilename = morphologyFilename
        self.fullMorphologyFilename = os.path.join(model_path, morphologyFilename)
        self.model_path = os.path.join(model_path, "mods")

        self.model_to_steal = None

        if is_delete_axon:
            self.L5PC.delete_axon()
            logging.info("Done delete axon")
            # logging.info(self.all)
            # logging.info("Done delete axon 2")

        #logging.info(self.all)

        # fix diam/L if needed
        logging.info("Fix shrinkage diam: {0}".format(morphologyFilename))
        self.fixed_shrinkage = False  # will be set to True if changed
        self.fixed_params = {}
        search_in = os.path.join(model_path, os.path.dirname(morphologyFilename))
        search_in = [search_in, os.path.dirname(search_in)]
        fix_json = [os.path.join(dirname, n) for dirname in search_in for n in os.listdir(dirname) if n == "morph_fix.json"]
        if len(fix_json) == 1 and with_shrinkage_fix:
            try:
                with open(fix_json[0]) as f:
                    data = json.load(f)
                    relevant_key = [k for k in data.keys() if k in self.morphologyFilename]
                    if len(relevant_key) == 1:
                        self.fixed_params = data[relevant_key[0]]
                        self.fix_shrinkage(length_scale=self.fixed_params["L"], diameter_scale=self.fixed_params["d"])
                        print("Fixed shrinkage for ", relevant_key[0] ," with ", self.fixed_params)
            except Exception as e: pass
        else:
            logging.info("Lack morph_fix.json")

        if fix_diam:
            self.fix_diam()
            logging.info(("Fixed diam for {0}", morphologyFilename))

        logging.info("Done fix diam")

        logging.info("Get_markers")
        if "Gabor" in self.fullMorphologyFilename:
            if False:  # todo add before resimulate!
                patch_sections_connected_to_soma(morphologyFilename=self.morphologyFilename, all_sections=self.all)

        self.markers = self.__get_markers(os.path.join(model_path, morphologyFilename))
        logging.info("Done get_markers")
        if is_init_trunk_oblique:
            if "L5PC_NEURON_simulation" in model_path:
                diam_diff_threshold_um = 0.5
            if "2005201lr_6" in morphologyFilename:
                stop_trunk_at = "apic[37]"  # patch- diams doesnt change - weird
            if "2006253lr_3" in morphologyFilename:
                diam_diff_threshold_um = 0.4
                # stop_trunk_at = "apic[8]"  # patch- diams doesnt change - weird
            logging.info("Set trunk oblique")
            self.__set_trunk_oblique_from_morph(diam_diff_threshold_um=diam_diff_threshold_um, stop_trunk_at=stop_trunk_at)

        if replace_soma_dendrites_with_fake:
            self.__replace_soma_dendrites_with_fake()
            print(self.BasalSectionsList)
        elif steal_rat_soma is not None or steal_human_soma is not None:
            self.__prepare_stolen_cell(cell_temp_func, model_path, self.morphologyFilename, is_fix_diam=fix_diam,
                                       is_delete_axon=is_delete_axon, with_shrinkage_fix=with_shrinkage_fix,
                                       steal_human_soma=steal_human_soma, steal_rat_soma=steal_rat_soma)
            prev_children = self.__replace_soma_dendrites_with_fake(
                fake_soma_diam=self.stolen_morph_cell.soma.diam, fake_soma_L=self.stolen_morph_cell.soma.L) # remove basal etc
            for sec in prev_children:
                # h.delete_section(sec=sec)
                self.deleted_secs.append(sec.name())  # for plot
                self.deleted_secs.extend([s.name() for s in sec.subtree()])
            # self.stolen_morph_cell.deleted_secs = self.deleted_secs
            basal_children_of_stolen_soma = [c for c in self.stolen_morph_cell.soma.children() if "dend" in c.name()]
            for ch in basal_children_of_stolen_soma:
                ch.connect(self.soma, .5, 0)
                self.stolen_secs.append(ch) # todo define if we ran on these as well
                self.stolen_secs.extend(ch.subtree())
            self.stolen_secs = list(set(self.stolen_secs))
            logging.info(f"Replaced children: {prev_children} => {[c for c in self.soma.children() if 'dend' in c.name()]}")

        cvode = h.CVode()
        if use_cvode:
            cvode.active(1)
        self.name = name
        self.default_v = -81.0

        # for loops, better to calc once and save
        self.basal_seg_length_um = np.array(self.__getSegmentsLength_um(self.BasalSectionsList))
        self.apical_seg_length_um = np.array(self.__getSegmentsLength_um(self.ApicalSectionsList))
        self.allSections = self.BasalSectionsList + self.ApicalSectionsList
        self.allSections += self.SomaSectionsList if sim_with_soma else []
        # self.allSections += self.AxonalSectionsList if sim_with_axon else []  # todo crash
        self.allSegments = self.__getAllSegments(self.allSections)
        self.icl = None
        logging.info("Done calc segments")

        # Common init
        self.ih_type = ih_type
        self.channel_param = {} # update names within init_channels
        if is_init_passive:  # todo won't work when stealing?
            self.init_passive_params()
            print("Done init passive")
            apical_channels=["Im", "SK_E2",  "SKv3_1", "Ca_LVAst", "Ca", "CaDynamics_E2"],
            somatic_channels=["Im", "SK_E2",  "SKv3_1", "Ca_LVAst", "Ca", "K_Pst", "K_Tst",  "Nap_Et2", "NaTg", "CaDynamics_E2"],
            h.distance(0, 0.5, sec=self.soma)
            if is_init_hcn:  # todo human ih?
                self.init_channels_params(basal_channels=[ih_type], apical_channels=[ih_type], axonal_channels=[ih_type],
                                          somatic_channels=[ih_type])
                self.ih_type = ih_type
                field_name = f"gIhbar_{ih_type}"
                print(f"Before: {field_name} soma=", [getattr(a, field_name) for a in self.L5PC.somatic], " apical ",
                      [getattr(a, field_name) for a in self.L5PC.apical])
                self.L5PC.distribute(self.L5PC.apical, field_name, "( 0 * %.6g  + 1 ) * 2e-5", 1)
                self.L5PC.distribute(self.L5PC.somatic, field_name, "( 0 * %.6g  + 1 ) * 1e-6", 1)
                print(f"After: {field_name} soma=", [getattr(a, field_name) for a in self.L5PC.somatic], " apical ",
                      [getattr(a, field_name) for a in self.L5PC.apical])
            if is_init_sk:
                ch = ["SK_E2"]  # SK_E2 = SK-type calcium-activated potassium current
                self.init_channels_params(basal_channels=ch, apical_channels=ch, axonal_channels=ch, somatic_channels=ch)
                print("Before: gSK_E2bar_SK_E2 soma=", self.soma.gSK_E2bar_SK_E2, " axon ", [a.gSK_E2bar_SK_E2 for a in self.axon])
                self.L5PC.distribute(self.L5PC.somatic, "gSK_E2bar_SK_E2","( 0 * %.6g  + 1 ) * 0.031068", 1)
                self.L5PC.distribute(self.L5PC.axonal, "gSK_E2bar_SK_E2","( 0 * %.6g  + 1 ) * 0.00640298", 1)
                print("After: gSK_E2bar_SK_E2 soma=", self.soma.gSK_E2bar_SK_E2, " axon ", [a.gSK_E2bar_SK_E2 for a in self.axon])
            if is_init_k:
                # SKv3_1 = Shaw-related potassium channel family in rat brain
                # Only soma/axon: K_Tst = transient component of the K current, K_Pst = persistent component of the K current
                ch = ["SKv3_1"]
                self.init_channels_params(basal_channels=ch, apical_channels=ch, axonal_channels=ch, somatic_channels=ch)
                print("Before: gSKv3_1bar_SKv3_1 soma=", self.soma.gSKv3_1bar_SKv3_1, " axon ", [a.gSKv3_1bar_SKv3_1 for a in self.axon])
                self.L5PC.distribute(self.L5PC.somatic, "gSKv3_1bar_SKv3_1", "( 0 * %.6g  + 1 ) * 0.132429", 1)
                self.L5PC.distribute(self.L5PC.axonal, "gSKv3_1bar_SKv3_1", "( 0 * %.6g  + 1 ) * 1.97907", 1)
                print("Before: gSKv3_1bar_SKv3_1 soma=", self.soma.gSKv3_1bar_SKv3_1, " axon ", [a.gSKv3_1bar_SKv3_1 for a in self.axon])
            if is_init_active:  # default list
                if is_active_na_kv_only:  # H&H channels, with parameters from spike
                    somatic_channels = ["kv", "na"]
                    apical_channels = ["kv", "na"]
                    # todo - parameters should not be equivalent! fixme
                    self.init_channels_params(somatic_channels=somatic_channels, apical_channels=apical_channels,
                                              axonal_channels=somatic_channels, basal_channels=apical_channels)
                    # gbar in na is 1000 which is 100 ms/cm^2
                    for what, curr_list in zip(["na", "kv"], [na_gbar_list, k_gbar_list]):
                        for mech_name, gbar_value, sections_list in zip([what]*5, curr_list, # [8, 4, 8, 140]
                                                                        [self.basal, self.apic, [self.soma], self.axon, self.tuft]):
                            for sec in sections_list:
                                for seg in sec:
                                    for mech in seg:
                                        if mech_name == mech.name():
                                            mech.gbar = gbar_value * 10  # scale to pS to um^2
                else: # todo basal?
                    somatic_channels=["Im", "SK_E2",  "SKv3_1", "Ca_LVAst", "Ca_HVA", "K_Pst", "K_Tst",
                                      "Nap_Et2", "NaTg", "CaDynamics_E2"]
                    apical_channels=["Im", "SK_E2",  "SKv3_1", "Ca_LVAst", "Ca_HVA", "K_Pst", "K_Tst",
                                      "NaTs2_t", "CaDynamics_E2"]
                    self.init_channels_params(somatic_channels=somatic_channels, apical_channels=apical_channels,
                                              axonal_channels=somatic_channels)
                    self.L5PC.biophys_active()
            # logging.info({'all': self.get_mechanism_data(startswith_='', endswith_='')})

        self.active_mechanisms = {'gbar': self.get_mechanism_data(startswith_='g', endswith_='bar')}  # keep previous
        # {'all': self.get_mechanism_data(startswith_='', endswith_='')}
        # distance between path points for xyz coord interpolation:
        # np.concatenate([[0], np.sqrt(np.diff(x_path) **2 + np.diff(y_path) **2 + np.diff(z_path) **2)])
        print("Done __init__")

    @staticmethod
    def __add_few_spines(sref_list, x_vec, neck_diam, neck_len, spine_head_area, ra, cm, rm, e_pas,
                         head_psd_percentage=0.1):
        def create(num, is_head=True, is_psd=False):
            if is_head and not is_psd:
                sp = h.Section(name=f"spine_head{num}")
                sp.L = (1 - head_psd_percentage) * L_head
                sp.diam = diam_head
            elif is_head and is_psd:
                sp = h.Section(name=f"spine_head_psd{num}")
                sp.L = head_psd_percentage * L_head
                sp.diam = diam_head
            else:
                sp = h.Section(name=f"spine_neck{num}")
                sp.L = neck_len
                sp.diam = neck_diam
            sp.insert('pas')
            sp.g_pas = 1 / rm  # 1/Rm - Rm ohm*cm^2
            sp.e_pas = e_pas
            sp.cm = cm
            sp.Ra = ra  # ohm*cm
            return sp

        L_head = 2 * np.sqrt(spine_head_area / 4 / np.pi)  # sphere has the same surface area as cylinder with L=diam
        diam_head = L_head
        spines = []
        spine_psds = []

        for i, sec in enumerate(sref_list):
            for j, shaft_x in enumerate(x_vec[i]):
                sp_head = create(j, is_head=True, is_psd=False)
                sp_head_psd = create(j, is_head=True, is_psd=True)
                sp_neck = create(j, is_head=False)
                spine_psds.append(sp_head_psd)
                spines.append(sp_neck)  # 2j
                spines.append(sp_head)  # 2j + 1
                sp_head_psd.connect(sp_head(1), 0)  # todo direction ok?
                sp_head.connect(sp_neck(1), 0)  # todo direction ok?
                sp_neck.connect(sec(shaft_x), 0)
                print(sp_head(1), " connect to begin of ", sp_head_psd, " with diam ", sp_head_psd.diam, " length ",
                      sp_head_psd.L)
                print(sp_neck(1), " connect to begin of ", sp_head, " with diam ", sp_head.diam, " length ",
                      sp_head.L)
                print(sec(shaft_x), " connect to begin of ", sp_neck, " with diam ", sp_neck.diam, " length ",
                      sp_neck.L)
        return spines, spine_psds

    def add_spines_to_sec(self, dend, SPINE_NECK_DIAM=0.25, SPINE_NECK_L=1.35, SPINE_HEAD_AREA=2.8,
                          SPINE_RA_factor=1, SPINE_RA=None, SPINE_RM=None, SPINE_CM=None, SPINE_EPAS=None,  # copy dend values if None
                          syn_segs=list(np.arange(start=0, stop=1.1, step=0.1))):
        print("Sapir ", type(SPINE_RA_factor), type(dend.Ra) if SPINE_RA is None else SPINE_RA)
        print(dend.Ra * SPINE_RA_factor if SPINE_RA is None else SPINE_RA)
        return self.__add_few_spines([dend], [syn_segs], SPINE_NECK_DIAM, SPINE_NECK_L, SPINE_HEAD_AREA,
                                     ra=dend.Ra * SPINE_RA_factor if SPINE_RA is None else SPINE_RA,
                                     rm=1 / dend.g_pas if SPINE_RM is None else SPINE_RM,
                                     cm=dend.cm if SPINE_CM is None else SPINE_CM,
                                     e_pas=dend.e_pas if SPINE_EPAS is None else SPINE_EPAS)

    def update_na_kv(self, na_gbar_list=[6, 6, 10, 200, 6], k_gbar_list=[5, 5, 5, 5, 5]): # basal, apical, soma, axon, tuft
        if not "na" in self.channel_param["apical_channels"]:
            print("Error. cant change non init param")
            return
        # gbar in na is 1000 which is 100 ms/cm^2
        for what, curr_list in zip(["na", "kv"], [na_gbar_list, k_gbar_list]):
            for mech_name, gbar_value, sections_list in zip([what]*5, curr_list, # [8, 4, 8, 140]
                                                                        [self.basal, self.apic, [self.soma], self.axon, self.tuft]):
                for sec in sections_list:
                    for seg in sec:
                        for mech in seg:
                            if mech_name == mech.name():
                                mech.gbar = gbar_value * 10  # scale to pS to um^2


    def __prepare_stolen_cell(self, cell_temp_func, model_path, morphologyFilename, is_delete_axon, with_shrinkage_fix,
                              is_fix_diam, steal_human_soma=None, steal_rat_soma=None):
        model_to_steal = steal_human_soma if steal_human_soma is not None else steal_rat_soma
        self.model_to_steal = morphologyFilename.replace(os.path.basename(morphologyFilename), model_to_steal)
        if os.path.isfile(os.path.join(model_path, self.model_to_steal)):
            self.stolen_morph_cell = CellMorph(cell_temp_func(os.path.join(model_path, self.model_to_steal)))
        elif os.path.isfile(os.path.join(model_path, self.model_to_steal.replace("human_morph", "rat_morph"))):
            self.model_to_steal = self.model_to_steal.replace("human_morph", "rat_morph")
            self.stolen_morph_cell = CellMorph(cell_temp_func(os.path.join(model_path, self.model_to_steal)))
        elif os.path.isfile(os.path.join(model_path, self.model_to_steal.replace("rat_morph", "human_morph"))):
            self.model_to_steal = self.model_to_steal.replace("rat_morph", "human_morph")
            self.stolen_morph_cell = CellMorph(cell_temp_func(os.path.join(model_path, self.model_to_steal)))
        else:
            print("Error ",  self.model_to_steal, os.listdir(model_path))
        logging.info(f"Stole soma from {self.model_to_steal}: {self.stolen_morph_cell}")
        if is_delete_axon:
            self.stolen_morph_cell.L5PC.delete_axon()
        if with_shrinkage_fix:
            try:
                chunkSize = 40
                fixed_params = None
                search_in_2 = os.path.join(model_path, os.path.dirname(self.model_to_steal))
                search_in_2 = [search_in_2, os.path.dirname(search_in_2)]
                fix_json_2 = [os.path.join(dirname, n) for dirname in search_in_2
                              for n in os.listdir(dirname) if n == "morph_fix.json"]
                if len(fix_json_2) == 1:
                    with open(fix_json_2[0]) as f:
                        data = json.load(f)
                        relevant_key = [k for k in data.keys() if k in self.model_to_steal]
                        fixed_params = data[relevant_key[0]] if len(relevant_key) == 1 else None
                if fixed_params is not None:
                    fix_shrinkage(all_sections=self.stolen_morph_cell.all, axon_sections=self.stolen_morph_cell.axon,
                                  morphologyFilename=self.model_to_steal, chunkSize=chunkSize,
                                  length_scale=fixed_params["L"], diameter_scale=fixed_params["d"])
            except Exception as e: print(e)
        if is_fix_diam:
            fix_diameter_manually(morphologyFilename=self.model_to_steal, all_sections=self.stolen_morph_cell.all)

    def __replace_soma_dendrites_with_fake(self, fake_soma_diam=10, fake_soma_L=10):
        basal_children_of_soma = [c for c in self.soma.children() if "dend" in c.name()]
        logging.info(f"Disconnect {basal_children_of_soma} and set soma diam to {fake_soma_diam} and L {fake_soma_L}")
        for son in basal_children_of_soma:
            h.disconnect(sec=son)
        assert len([c for c in self.soma.children() if "dend" in c.name()]) == 0
        self.soma.diam = fake_soma_diam
        self.soma.L = fake_soma_L
        self.soma.nseg = 1
        return basal_children_of_soma

    def fix_shrinkage(self, length_scale=None, diameter_scale=None, chunkSize=40):
        fix_shrinkage(all_sections=self.all, axon_sections=self.axon, morphologyFilename=self.morphologyFilename,
                      length_scale=length_scale, diameter_scale=diameter_scale, chunkSize=chunkSize)
        self.fixed_shrinkage = length_scale is not None and diameter_scale is not None

    def fix_diam(self):
        fix_diameter_manually(morphologyFilename=self.morphologyFilename, all_sections=self.all)

    def __set_trunk_oblique_from_morph(self, diam_diff_threshold_um=0.2, stop_trunk_at=None):
        """For clear tree like shape we can use morph to split apical
        :return:
        """
        apical_children_of_soma = [c for c in self.soma.children() if "apic" in c.name()]
        apical_children_of_soma = [s for s in apical_children_of_soma if len(s.children()) > 0]
        if len(apical_children_of_soma) != 1:
            logging.error("Can't find single apical trunk in soma children: {0}".format(self.soma.children()))
            return
        child = apical_children_of_soma[0]
        trunk = [child]
        while max(abs(np.diff([c.diam for c in child.children()]))) > diam_diff_threshold_um:
            child = max(child.children(), key=lambda c: c.diam)
            trunk.append(child)
            if len(child.children()) <= 1 or (stop_trunk_at is not None and stop_trunk_at in child.name()):
                break
        obliques = []  # choose those who are on the path but not on trunk - todo prone to errors in trunk detection
        for sec in trunk[:-1]:
            added_obliques = [c for c in sec.children() if c not in trunk]
            obliques.extend(added_obliques)
            for o in added_obliques:
                obliques.extend(o.subtree())
        self.trunk_list = trunk
        self.oblique_list = list(set(obliques))
        self.tuft_list = [tuft for tuft in trunk[-1].subtree() if tuft not in trunk]  # remove trunk[-1]

    @staticmethod
    def __record_Ih(cell):
        record_dict = dict()
        for sec in cell.all:
            record_dict[sec] = dict()
            for i, seg in enumerate(sec):
                try:
                    record_dict[sec][seg] = h.Vector()
                    record_dict[sec][seg].record(sec(seg.x)._ref_gIh_Ih_human_shifts_mul_add)  # todo unique?
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    record_dict[sec][seg] = 0  # no Ih here - todo nan?
        return record_dict

    def simulate_Ih_traces(self, run_time_ms=750):
        g_Ih_record_dict = self.__record_Ih(cell=self)
        h.tstop = run_time_ms
        h.run()
        g_Ih_record_dict_steady_state = dict()
        for sec in g_Ih_record_dict.keys():
            g_Ih_record_dict_steady_state[sec] = dict()
            for seg in g_Ih_record_dict[sec].keys():
                if g_Ih_record_dict[sec][seg] == 0: continue
                g_Ih_record_dict_steady_state[sec][seg] = np.array(g_Ih_record_dict[sec][seg])[-1]  # stady state opening
        return g_Ih_record_dict, g_Ih_record_dict_steady_state

    @staticmethod
    def __get_markers(morph_full_path):
        """Use blueBrain morphio package to parse asc (tbd not sure how to do the same in neuron loading)

        :param morph_full_path:
        :return: dict with markers data
        """
        def to_dict(cls):
            return dict([(k, getattr(cls, k)) for k in dir(cls) if not k.startswith("__")])
        from morphio import Morphology
        try:
            curr_cell_data = Morphology(morph_full_path)
            return [to_dict(marker) for marker in curr_cell_data.markers]
        except Exception as e:
            print(morph_full_path, e)
        return []

    def nearest_segment_to_marker_coords(self, marker_dict):
        closest_sec, sec_x, dist = None, None, np.inf
        x, y, z = marker_dict["points"][0]
        for sec_ind, sec in enumerate(self.all):
            x_path = [h.x3d(i, sec=sec) for i in range(int(h.n3d(sec=sec)))]
            y_path = [h.y3d(i, sec=sec) for i in range(int(h.n3d(sec=sec)))]
            z_path = [h.z3d(i, sec=sec) for i in range(int(h.n3d(sec=sec)))]
            distances = np.sqrt((np.array(x_path) - x) ** 2 + (np.array(y_path) - y) ** 2 + (np.array(z_path) - z) ** 2)
            if distances.min() < dist:
                dist = distances.min()
                closest_sec = sec
                sec_x = distances.argmin() / len(x_path)
        return closest_sec(sec_x), dist

    @staticmethod
    def get_impedance(x_sec, input_electrode_sec, freq_hz=0):
        imp = h.Impedance()
        imp.loc(input_electrode_sec.x, sec=input_electrode_sec.sec)  # location of input electrode (middle of section).
        imp.compute(freq_hz)
        return {"Rin": imp.input(x_sec.x, sec=x_sec.sec), "Rtr_M_ohm": imp.transfer(x_sec.x, sec=x_sec.sec)}

    def get_mechanism_data(self, startswith_='g', endswith_='bar'):
        result = {}
        for sec in self.BasalSectionsList + self.ApicalSectionsList:  # todo add soma axon?
            for seg in sec:
                for mech in seg:
                    if mech.name() not in result:
                        result[mech.name()] = {}
                    values = dict(get_class_members(mech))
                    data_to_save = dict([(k, values[k]) for k in values.keys()
                                         if k.startswith(startswith_) and k.endswith(endswith_)])
                    result[mech.name()][str(mech.segment())] = data_to_save

        return result

    def init_passive_params(self):
        for sec in self.all:
            sec.insert('pas')
            print(sec)
            for seg in sec:
                seg.pas.e = 0

    def init_channels_params(self,
                             basal_channels=["Im", "SK_E2",  "SKv3_1", "Ca_LVAst", "Ca_HVA",
                                              # todo why na k missing? "K_Pst", "K_Tst", "Nap_Et2", "NaTg",
                                              "CaDynamics_E2"],
                             apical_channels=["Im", "SK_E2",  "SKv3_1", "Ca_LVAst", "Ca_HVA",
                                              # todo why na k missing? "K_Pst", "K_Tst", "Nap_Et2", "NaTg",
                                              "CaDynamics_E2"],
                             somatic_channels=["Im", "SK_E2",  "SKv3_1", "Ca_LVAst", "Ca_HVA", "K_Pst", "K_Tst",
                                               "Nap_Et2", "NaTg", "CaDynamics_E2"],
                             axonal_channels=["Im", "SK_E2", "SKv3_1", "Ca_LVAst", "Ca_HVA", "K_Pst", "K_Tst",
                                              "Nap_Et2", "NaTg", "CaDynamics_E2"]):
        """MOO's result are the exact params of this fit.

        :param basal_channels:
        :param apical_channels:
        :param somatic_channels:
        :param axonal_channels:
        :return:
        """
        # todo NaTa_t or NaTg or NaTs2_t? Ca or Ca_HVA?
        loop_dict = {"soma": ([self.soma], somatic_channels), "apic": (self.apic, apical_channels),
                     "bas": (self.basal, basal_channels), "axon": (self.axon, axonal_channels)}
        for name, (where, channels_list) in loop_dict.items():
            for ch in channels_list:
                if ch == "Ih":
                    ch = self.ih_type
                try:
                    logging.info("Insert {0} channel {1}".format(name, ch))
                    for sec in where:
                        sec.insert(ch)
                except Exception as e:
                    logging.error(e)
                    traceback.print_exc()
        self.channel_param=dict(basal_channels=basal_channels, apical_channels=apical_channels, somatic_channels=somatic_channels, axonal_channels=axonal_channels)

    def change_passive_params(self, CM=1, RA=250, RM=20000.0, E_PAS=-70, SPINE_START=60, F_factor=1.9):
        h.distance(0, 0.5, sec=self.soma)

        for sec in self.all:
            print(sec, RA, RM)
            sec.Ra = RA  # Ohm-cm
            sec.cm = CM  # uF/cm^2
            sec.g_pas = 1.0 / RM  # RM: Ohm-cm^2
            sec.e_pas = E_PAS

        for sec in self.apic + self.basal:
            for seg in sec:
                if self._get_distance_between_segments(origin_segment=self.soma(0.5), to_segment=seg) > SPINE_START:
                    seg.cm *= F_factor
                    seg.g_pas *= F_factor
        # print(f"RA={RA:.2f} RM={RM:.2f} CM={CM:.4f} F_factor={F_factor}, SPINE_START={SPINE_START} E_PAS={E_PAS}")

    def __set_mechanism_to_value(self, mech_name, new_value=None, is_changing_mech=lambda mech: True):
        """Adapt a specific mechanism in cell. Change from given value to original value
        :param mech_name: as appears in self.active_mechanisms keys
        :param new_value: None for changing to saved value (from self.active_mechanisms)
        """
        def _log(mech_):
            print(mech_.name(), mech_.segment(), get_class_members(mech_))

        if mech_name not in self.active_mechanisms.keys():
            logging.error("Trying to set wrong mechanism name ({0} not in {1})".format(mech_name,
                                                                                       self.active_mechanisms.keys()))
            return

        for sec in self.BasalSectionsList + self.ApicalSectionsList:  # todo as before
            for seg in sec:
                for mech in seg:
                    if is_changing_mech(mech):
                        curr_data = self.active_mechanisms[mech_name][mech.name()][str(mech.segment())]
                        for key in curr_data.keys():  # only adapt saved mechs
                            if new_value is None:
                                setattr(mech, key, curr_data[key])
                            else:
                                setattr(mech, key, new_value)
                            # _log(mech)

    def set_passive(self):
        """Change all gbar values to 0"""
        mech_name = 'gbar'  # save and adapt conductance only
        self.__set_mechanism_to_value(mech_name='gbar', new_value=0.0,
                                      is_changing_mech=lambda mech: mech.name() != 'pas')

    def set_active(self):
        """Change all gbar values to original value"""
        self.__set_mechanism_to_value(mech_name='gbar', new_value=None,
                                      is_changing_mech=lambda mech: mech.name() != 'pas')

    def mechanisms_info(self):
        result = {}
        for sec in self.BasalSectionsList + self.ApicalSectionsList:  # todo as before
            result[sec.name()] = sec.psection().get('density_mechs', {})
        return result

    def add_current_clamp(self, initial_voltage, totalSimDurationInMS):
        voltage_to_somatic_current_map = {
            -81.0: 0.003,
            -76.0: 0.11,
            -70.0: 0.23,
            -62.0: 0.394
        }
        if self.name != "L5PC_NEURON_simulation":
            logging.error("Lack conversion for model {0} (dont know how to add current clamp)".format(self.name))
            return

        self.icl = h.IClamp(0.5, sec=self.soma)
        self.icl.delay = 200.0  # ms - let model stabilize before changing
        self.icl.dur = totalSimDurationInMS - self.icl.delay  # ms
        self.icl.amp = voltage_to_somatic_current_map[initial_voltage]  # nS

    @staticmethod
    def alpha_stim(delay_ms=200, dur_from_delay_ms=400, amp_ns=0.1, tau0=10, tau1=15, dt=0.1):
        """todo bug too small

        :param delay_ms:
        :param dur_from_delay_ms:
        :param amp_ns:
        :param tau0:
        :param tau1:
        :param dt:
        :return:
        """
        time = np.arange(0, delay_ms + dur_from_delay_ms, dt)
        time_for_exp = np.arange(0, dur_from_delay_ms, dt)
        current_vec = np.zeros(time.shape)
        from_t = int(np.round(delay_ms/dt))
        current_vec[from_t:] = amp_ns * ((1-np.exp(-(time_for_exp)/tau0))-(1-np.exp(-time_for_exp/tau1)))
        return time, current_vec

    def add_current_stim(self, seg, delay_ms=200, dur_from_delay_ms=400, amp_ns=0.1, tau0=10, tau1=15):
        if self.icl is not None:
            del self.icl
        self.icl = h.IClamp(seg.x, sec=seg.sec)
        self.icl.delay = delay_ms  # ms - let model stabilize before changing
        self.icl.dur = dur_from_delay_ms  # ms
        self.icl.amp = amp_ns  # nS

    def add_alpha_current_stim(self, seg, delay_ms=200, dur_from_delay_ms=400, amp_ns=0.1, tau0=5, tau1=8):
        if self.icl is not None:
            del self.icl

        if dur_from_delay_ms == 2:
            dur_from_delay_ms=5  # allow taus to "work"
            tau0 = 0.25  # short enough but still alpha
            tau1 = 1
            amp_ns *= 2  # reach same height
        time, current = self.alpha_stim(delay_ms=delay_ms, dur_from_delay_ms=dur_from_delay_ms, amp_ns=amp_ns,
                                        tau0=tau0, tau1=tau1, dt=h.dt)
        self.icl = h.IClamp(seg.x, sec=seg.sec)
        self.icl.dur = 1e4  # ms
        self.alpha_current_vec = h.Vector(current)
        self.alpha_time_vec = h.Vector(time)
        self.alpha_current_vec.play(self.icl._ref_amp, self.alpha_time_vec)
        self.alpha_params = dict(delay_ms=delay_ms, dur_from_delay_ms=dur_from_delay_ms, amp_ns=amp_ns)

    def add_alpha_voltage_stim(self, seg, delay_ms=200, dur_from_delay_ms=400, amp_ns=25, tau0=5, tau1=8,
                               base_voltage=-70):
        if self.icl is not None:
            del self.icl

        if dur_from_delay_ms == 2:
            dur_from_delay_ms = 5  # allow taus to "work"
            tau0 = 0.25  # short enough but still alpha - 0.25
            tau1 = 1
            amp_ns *= 2  # reach same height 2
        time, current = self.alpha_stim(delay_ms=delay_ms, dur_from_delay_ms=dur_from_delay_ms, amp_ns=amp_ns,
                                        tau0=tau0, tau1=tau1, dt=h.dt)
        current += base_voltage  # min is zero in this function

        self.icl = h.SEClamp(seg.x, sec=seg.sec)
        self.icl.dur1 = 1e9  # ms
        self.icl.rs = 1e-3  # series resistance should be much smaller than input resistance of the cell

        self.alpha_current_vec = h.Vector(current)
        self.alpha_time_vec = h.Vector(time)
        self.alpha_current_vec.play(self.icl._ref_amp1, self.alpha_time_vec)
        self.alpha_params = dict(delay_ms=delay_ms, dur_from_delay_ms=dur_from_delay_ms, amp_ns=amp_ns)

    def to_dict(self):
        allSectionsType, allSections_DistFromSoma, allSectionsLength, allSegmentsType, allSegmentsLength, \
               allSegments_DistFromSoma, allSegments_SectionDistFromSoma, allSegments_SectionInd, \
               allSegments_seg_ind_within_sec_ind =\
            self.collect_sections_from_model()

        experimentParams = {}
        experimentParams['allSectionsType'] = allSectionsType
        experimentParams['allSections_DistFromSoma'] = allSections_DistFromSoma
        experimentParams['allSectionsLength'] = allSectionsLength
        experimentParams['allSegmentsType'] = allSegmentsType
        experimentParams['allSegmentsLength'] = allSegmentsLength
        experimentParams['allSegments_DistFromSoma'] = allSegments_DistFromSoma
        experimentParams['allSegments_SecDistFromSoma'] = allSegments_SectionDistFromSoma
        experimentParams['allSegments_SectionInd'] = allSegments_SectionInd
        experimentParams['allSegments_seg_ind_within_sec_ind'] = allSegments_seg_ind_within_sec_ind
        experimentParams['basal_seg_length_um'] = self.basal_seg_length_um
        experimentParams['apical_seg_length_um'] = self.apical_seg_length_um
        if self.icl is not None:
            if hasattr(self.icl, 'delay'):
                experimentParams['icl_soma_delay_ms'] = self.icl.delay
                experimentParams['icl_soma_dur_ms'] = self.icl.dur
                experimentParams['icl_soma_amp_ns'] = self.icl.amp

        return experimentParams

    @staticmethod
    def set_field(field_name, value, sections):
        if value is None or np.isnan(value):
            return
        logging.debug(f"Change cell's {field_name}" + "from {0:.3e} to {1:.3e}".format(getattr(sections[0], field_name), value))
        for section in sections:  # todo sep places?
            if hasattr(section, field_name): # todo else err?
                setattr(section, field_name, value)
            else:
                logging.error("{section.name()} doesn't have {field_name}. cant change (set_field)")

    def set_voltage_activation_curve_of_Ih_current(self, Ih_g_bar=None, Ih_alpha_shift_v=None, Ih_tau_add=None, Ih_tau_mul=None):
        all_sections = self.BasalSectionsList + self.ApicalSectionsList + self.SomaSectionsList
        self.set_field(field_name="gIhbar_{0}".format(self.ih_type), value=Ih_g_bar, sections=all_sections)
        self.set_field(field_name="m_alpha_shift_v_{0}".format(self.ih_type), value=Ih_alpha_shift_v, sections=all_sections)
        self.set_field(field_name="m_tau_add_{0}".format(self.ih_type), value=Ih_tau_add, sections=all_sections)
        self.set_field(field_name="m_tau_mul_{0}".format(self.ih_type), value=Ih_tau_mul, sections=all_sections)

    def set_sk_multiplicative_factor(self, SKE2_mult_factor):
        if SKE2_mult_factor is None or np.isnan(SKE2_mult_factor):
            return
        for section in self.SomaSectionsList + self.AxonalSectionsList + self.ApicalSectionsList:
            orig_SKE2_g = section.gSK_E2bar_SK_E2
            section.gSK_E2bar_SK_E2 = orig_SKE2_g * SKE2_mult_factor

    @staticmethod
    def DefineSynapse_add_common(synapse, e=0):
        if e is not None:
            synapse.e = e
        synapse.Use = 1
        synapse.u0 = 0
        synapse.Dep = 0
        synapse.Fac = 0

    @staticmethod
    def DefineSynapse_AMPA(segment, gMax=0.0004, tau_r=0.3, tau_d=3.0):
        synapse = h.ProbUDFsyn2(segment)
        synapse.tau_r = tau_r
        synapse.tau_d = tau_d
        synapse.gmax = gMax
        NeuronCell.DefineSynapse_add_common(synapse)
        return synapse

    @staticmethod
    def DefineSynapse_NMDA(segment, gMax=0.0004, tau_r_AMPA=0.3, tau_d_AMPA=3.0, tau_r_NMDA=2.0, tau_d_NMDA=70.0):
        synapse = h.ProbAMPANMDA2(segment) 
        synapse.tau_r_AMPA = tau_r_AMPA
        synapse.tau_d_AMPA = tau_d_AMPA
        synapse.tau_r_NMDA = tau_r_NMDA
        synapse.tau_d_NMDA = tau_d_NMDA
        synapse.gmax = gMax
        NeuronCell.DefineSynapse_add_common(synapse)
        return synapse

    @staticmethod
    def DefineSynapse_GABA_A(segment, gMax=0.001, tau_r=0.2, tau_d=8):
        synapse = h.ProbUDFsyn2(segment)
        synapse.tau_r = tau_r
        synapse.tau_d = tau_d
        synapse.gmax = gMax
        NeuronCell.DefineSynapse_add_common(synapse, e=-80)
        return synapse

    @staticmethod
    def DefineSynapse_GABA_B(segment, gMax=0.001, tau_r=3.5, tau_d=260.9):
        synapse = h.ProbUDFsyn2(segment)
        synapse.tau_r = tau_r
        synapse.tau_d = tau_d
        synapse.gmax = gMax
        NeuronCell.DefineSynapse_add_common(synapse, e=-97)
        return synapse

    @staticmethod
    def DefineSynapse_GABA_AB(segment, gMax=0.001, tau_r_GABAA=0.2, tau_d_GABAA=8, tau_r_GABAB=3.5, tau_d_GABAB=260.9):
        """GABA A+B synapse
        :return:
        """
        synapse = h.ProbGABAAB_EMS(segment)
        synapse.tau_r_GABAA = tau_r_GABAA
        synapse.tau_d_GABAA = tau_d_GABAA
        synapse.tau_r_GABAB = tau_r_GABAB
        synapse.tau_d_GABAB = tau_d_GABAB
        synapse.gmax = gMax
        synapse.e_GABAA = -80
        synapse.e_GABAB = -97
        synapse.GABAB_ratio = 0.0
        NeuronCell.DefineSynapse_add_common(synapse, e=None)
        return synapse

    @staticmethod
    def _get_distance_between_segments(origin_segment, to_segment):
        """ (sec contains segments, equivalent to RC-circuits)
        Example: soma(0.5) origin, and a segment to connect to
        :param origin_segment:
        :param to_segment:
        :return: distance in um
        """
        h.distance(0, origin_segment.x, sec=origin_segment.sec)
        return h.distance(to_segment.x, sec=to_segment.sec)

    @staticmethod
    def __GetDistanceBetweenSections(sourceSection, destSection):
        h.distance(sec=sourceSection)
        return h.distance(0, sec=destSection)

    @staticmethod
    def __getSegmentsLength_um(SectionsList):
        result = []
        for k, section in enumerate(SectionsList):
            for currSegment in section:
                result.append(float(section.L) / section.nseg)  # todo why?
        return result

    @staticmethod
    def __getAllSegments(allSections):
        result = []
        for k, section in enumerate(allSections):
            for currSegment in section:
                result.append(currSegment)
        return result

    @staticmethod
    def sec_to_type_name(sec, map_name={'apic': 'apical', 'dend': 'basal', 'axon': 'axonal', 'soma': 'soma'}):
        curr_name = sec.name().split(".")[1].split("[")[0].replace("]", "")
        return map_name.get(curr_name, curr_name)

    def collect_sections_from_model(self):
        allSections = self.allSections
        allSectionsType = [self.sec_to_type_name(sec) for sec in allSections]
        # allSectionsType = ['basal' for x in self.BasalSectionsList] + ['apical' for x in self.ApicalSectionsList]

        assert(len(self.SomaSectionsList) == 1)
        soma = self.SomaSectionsList[0]

        allSectionsLength = []
        allSections_DistFromSoma = []
        allSegments = []
        allSegmentsLength = []
        allSegmentsType = []
        allSegments_DistFromSoma = []
        allSegments_SectionDistFromSoma = []
        allSegments_SectionInd = []
        allSegments_seg_ind_within_sec_ind = []
        # get a list of all segments
        for k, section in enumerate(allSections):
            allSectionsLength.append(section.L)
            allSections_DistFromSoma.append(NeuronCell.__GetDistanceBetweenSections(soma, section))
            for seg_ind, currSegment in enumerate(section):
                allSegments.append(currSegment)
                allSegmentsLength.append(float(section.L) / section.nseg)
                allSegmentsType.append(allSectionsType[k])
                allSegments_DistFromSoma.append(NeuronCell.__GetDistanceBetweenSections(soma, section) + float(section.L) * currSegment.x)
                allSegments_SectionDistFromSoma.append(NeuronCell.__GetDistanceBetweenSections(soma, section))
                allSegments_SectionInd.append(k)
                allSegments_seg_ind_within_sec_ind.append(seg_ind)
        return allSectionsType, allSections_DistFromSoma, allSectionsLength, allSegmentsType, allSegmentsLength, \
            allSegments_DistFromSoma, allSegments_SectionDistFromSoma, allSegments_SectionInd, allSegments_seg_ind_within_sec_ind
