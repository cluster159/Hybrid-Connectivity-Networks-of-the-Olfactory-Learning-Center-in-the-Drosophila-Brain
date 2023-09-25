import pandas as pd
from pandas import DataFrame as Df
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import generate_connection_v3 as gc
from generate_connection_v3 import ConnectionSetting
import pickle
import os
from scipy import stats
from mayavi import mlab
from tvtk.api import tvtk
import navis
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
from collections import defaultdict, Counter
import trimesh
import trimesh.proximity as proximity
from scipy.special import kl_div
import copy
import random as rd
import shutil
from scipy.stats import linregress
import pingouin as pg
from sklearn.decomposition import PCA
import platform
import trimesh
import pyvista as pv
import vtk
import analysis_tool
from analysis_tool import Anatomical_analysis

plt.rcParams['font.family'] = 'Arial'

class figure_manager():
    a = Anatomical_analysis()
    a.load_spatial_distribution_dict()
    Cluster_color_list = [(1, 0, 0), (1, 1, 0.569), (0.6, 1, 1)]
    KC_class_list = ['KCg',"KCa'b'","KCab"]
    PN_cluster_list = [1,2,3]

    def get_bouton_connection_number_ratio(self):
        '''
        fig. s1c-f
        :return:
        '''
        self.a.get_bouton_connection_number_ratio()

    def output_glomerulus_cluster_AL(self):
        '''
        The function will export glomeruli based on Cluster in /hemibrain_data/FlyEM_neuropil/
        :return:
        '''
        for Cluster in self.a.Cluster_to_Glomerulus:
            combined_mesh = trimesh.Trimesh()
            for G in a.Cluster_to_Glomerulus[Cluster]:
                if G == 'lvVM4':
                    G = "VM4"
                mesh = trimesh.load(f"{a.neuropil_path}AL-{G}(R).obj")
                # Merge the loaded mesh with the combined mesh
                combined_mesh += mesh
            output_file = f'{a.neuropil_path}AL_Cluster_{Cluster}.obj'
            combined_mesh.export(output_file)
        return

    def visualize_glomerulus_cluster_AL(self):
        '''
        fig. 1b
        The function will visualize glomeruli based on Cluster
        :return:
        '''
        self.a.visualize_neuropil(obj=["MB(R)", "AL_ALL(R)"])
        self.a.visualize_neuropil(obj=[f'AL_Cluster_1'], color=self.Cluster_color_list[0], opacity=0.3)
        self.a.visualize_neuropil(obj=[f'AL_Cluster_2'], color=self.Cluster_color_list[1], opacity=0.3)
        self.a.visualize_neuropil(obj=[f'AL_Cluster_3'], color=self.Cluster_color_list[2], opacity=0.3)
        self.a.visualize_mlab()
        return

    def visualize_spatial_distribution_density(self, Classification_type='major', data_type='synapse', neuropil='CA(R)'):
        '''
        fig. 2a, fig. 2b
        The function is used to visualize the density distribution
        :param Classification_type: major=KC class, minor =KC subclass, Glomerulus, Cluster
        :param data_type: neuron=skeleton, synapse=synapse, bouton, claw
        :param neuropil: CA(R) = calyx(R), AL_ALL(R) = all glomeruli in antennal lobe(R), LH(R) = lateral horn(R)
        :return:
        '''
        Classification = self.a.Classification_dict[Classification_type]
        cmap_list = ['Reds', 'Wistia', 'Blues', 'Purple', 'Greens', 'Oranges', 'Greys']
        cmap_list += cmap_list
        #
        print(Classification)
        for class_id, classification in enumerate(Classification[-4:]):
            if Classification_type == 'Glomerulus' or Classification_type == 'Cluster':
                keyname = f"FlyEM_{data_type}_PN_cluster_{classification}"
            else:
                keyname = f"FlyEM_{data_type}_{classification}"
            if data_type == 'synapse':
                if neuropil == 'CA(R)':
                    if 'KC' in keyname:
                        keyname += '_from_PN_CA(R)'
                    else:
                        keyname += '_to_KC_CA(R)'
                else:
                    keyname += f'_{neuropil}'
            else:
                if 'KC' in keyname:
                    keyname += f'_{neuropil}_0.8'
                else:
                    keyname += f'_{neuropil}_0'
            if keyname not in self.a.spatial_distribution_dict:
                print(keyname, 'fali')
                continue
            self.a.visualize_density_by_density(self.a.spatial_distribution_dict[keyname], obj=neuropil, cmap=cmap_list[class_id],
                                           contour_num=4)
        self.a.visualize_neuropil(obj=["CA(R)"])
        self.a.visualize_neuropil(obj=["MB(R)"])
        self.a.visualize_mlab()

    def visualize_synaptic_distribution(self):
        '''
        fig. s16
        :return:
        '''
        cmap_list = ['Reds','Wistia','Blues']
        Classification = self.a.Classification_dict['Cluster']
        for cluster_id, PN_cluster in enumerate(Classification):
            keyname1 = f'FlyEM_synapse_PN_cluster_{PN_cluster}_LH(R)_pre_0'
            keyname2 = f'FlyEM_synapse_PN_cluster_{PN_cluster}_CA(R)_pre_0'
            keyname3 = f'FlyEM_synapse_PN_cluster_{PN_cluster}_AL_ALL(R)_post_0'
            cmaps = cmap_list[cluster_id]
            self.a.visualize_density_by_density(self.a.spatial_distribution_dict[keyname1], obj='LH(R)', cmap=cmaps,
                                           contour_num=6)
            self.a.visualize_density_by_density(self.a.spatial_distribution_dict[keyname2], obj='CA(R)', cmap=cmaps,
                                           contour_num=6)
            self.a.visualize_density_by_density(self.a.spatial_distribution_dict[keyname3], obj='AL_ALL(R)', cmap=cmaps,
                                           contour_num=6)
        xyz1 = self.a.read_swc(f"hemibrain_data/FlyEM_skeleton/{self.a.Glomerulus_to_PNid['DA1'][0]}.swc")
        self.a.visualize_neuron(np.array(xyz1), color=(1, 0, 0))
        xyz2 = self.a.read_swc(f"hemibrain_data/FlyEM_skeleton/{self.a.Glomerulus_to_PNid['DM3'][0]}.swc")
        self.a.visualize_neuron(np.array(xyz2), color=(0, 0, 1))
        self.a.visualize_neuropil(obj=["LH(R)"])
        self.a.visualize_neuropil(obj=["CA(R)"])
        self.a.visualize_neuropil(obj=["MB(R)"])
        self.a.visualize_neuropil(obj=["AL_ALL(R)"])
        self.a.visualize_mlab()

    def compare_group_connection_num_weight(self,t1='Cluster', t2='major'):
        '''
        fig. 1c, fig. s3.
        This function is used to calculate the connection number, weight ratio of PN clusters with repect of each KC classes.
        The figure will be saved in 'Result/connection_related_figure/'
        :param t1:
        :param t2:
        :return:
        '''
        self.a.evaluate_PN_connection_density_bias(t1=t1,t2=t2)
        self.a.get_statistics_for_connection_pooled_individual_input_weight_ratio()
        self.a.get_statistics_for_connection_pooled_input_weight_ratio()

    def get_synapse_spatial_similarity(self):
        '''
        fig. 2d
        :return:
        '''
        self.a.plot_spatial_similarity()

    def get_DV_ratio_comparison(self, t1="Cluster"):
        '''
        fig. 2c
        The figure will be saved in 'Result\Fig2'
        :return:
        '''
        self.a.load_neuron_coordinates_in_neuropil()
        self.a.compare_neuron_in_CA_upper_and_lower(t1=t1)

    def get_synapse_distribution_similarity(self):
        self.a.calculate_JS_divergent_for_all()
        self.a.plot_spatial_similarity()

    def get_data_type(self,t,d):
        if t == 'major' or t == 'minor':
            if d == 'meso':
                d = 'claw'
        if t == 'Cluster' or t == 'Glomerulus':
            if d == 'meso':
                d = 'bouton'
        return d

    def check_p_related_to_KC(self,synapse_related_to_KC, t, neuropil, d):
        if neuropil == 'CA(R)' and d == 'synapse' and t in ['Glomerulus', "Cluster"] and synapse_related_to_KC:
            return True
        return False

    def get_spatial_distribution_preference(self, neuropil_list=['CA(R)'], t1_list=['Glomerulus'],
                                                  t2_list=['major'],d1_list=['synapse'],d2_list=['synapse'],
                                                  synapse_related_to_KC_1=True, synapse_related_to_KC_2=True,
                                                  clustermap=False):
        '''
        fig. s9, fig. s10, fig. s15, fig. s16,
        :param neuropil_list: 'AL_ALL(R)', "CA(R)", 'LH(R)'
        :param t1_list: "Glomerulus" PN glomerulus, "Cluster": PN cluster, "major": KC major class, "minor": KC subclass
        :param t2_list: "Glomerulus" PN glomerulus, "Cluster": PN cluster, "major": KC major class, "minor": KC subclass
        :param d1_list: for t1 --> "synapse": synapse, "meso": bouton for PN, claw for KC, "neuron": neurite
        :param d2_list: for t2 --> "synapse": synapse, "meso": bouton for PN, claw for KC, "neuron": neurite
        :param synapse_related_to_KC_1: --> for PN to KC analysis, synapse is limited to PN > KC synapse.
                for PN synapses in LH and AL, the synapse is not related to KC.
                if d1 is not synapse and neuropil is not CA(R), the parameter is not used
        :param synapse_related_to_KC_2: --> for PN to KC analysis, synapse is limited to PN > KC synapse.
                for PN synapses in LH and AL, the synapse is not related to KC
                if d2 is not synapse and neuropil is not CA(R), the parameter is not used

        :return:
        '''
        for neuropil in neuropil_list:
            for t1 in t1_list:
                for t2 in t2_list:
                    for d1 in d1_list:
                        for d2 in d2_list:
                            d1 = self.get_data_type(t1,d1)
                            d2 = self.get_data_type(t2,d2)
                            synapse_related_to_KC_1 = self.check_p_related_to_KC(synapse_related_to_KC_1, t1, neuropil,
                                                                                 d1)
                            synapse_related_to_KC_2 = self.check_p_related_to_KC(synapse_related_to_KC_2, t2, neuropil,
                                                                                 d2)

                            a.analyze_spatial_distribution_bar(t1=t1, d1=d1, t2=t2, d2=d2,
                                                                               neuropil=neuropil,
                                                                               datatype="JS",
                                                                               synapse_related_to_KC_1=synapse_related_to_KC_1,
                                                                               synapse_related_to_KC_2=synapse_related_to_KC_2,
                                                                               JS_clustermap=clustermap
                                                                               )

    def init_get_spatial_distribution_dict(self, re_init=False):
        '''
        The function is used to get all spatial distribution of neurite, bouton/claw, and synapse of PNs and KCs.
        Here
        1. get all coordinates of neurite, bouton/claw, synapse
        2. calculate the spatial distribution
        3. calculate JS distribution similarity
        :return:
        '''
        if os.path.isfile('init.txt') or not re_init:
            if re_init:
                print("Start to initiate. It takes time.")
            else:
                print("The initiation has been completed")
            return
        else:
            print("Start to initiate. It takes time.")
        print("Calculating spatial distribution.")
        self.a.add_PN_neuron_spatial_distribution_dict() ## get spatial distribution of neurites of PNs based on glomerulus and clusters
        self.a.add_KC_neuron_spatial_distribution_dict() ## get spatial distribution of neurites of KCs based on KC classes, and KC subclasses
        self.a.add_PN_synapse_spatial_distribution_dict() ## get spatial distribution of synapses of PNs based on glomerulus and clusters
        self.a.add_KC_synapse_spatial_distribution_dict() ## get spatial distribution of synapses of KCs based on KC classes, and KC subclasses
        self.a.add_PN_synapse_neuron_based_spatial_distribution_dict(neuropil_list=["AL_ALL(R)"], pre=False, post=True)
        self.a.add_PN_synapse_neuron_based_spatial_distribution_dict(neuropil_list=["LH(R)"], pre=True, post=False)
        self.a.add_bouton_spatial_distribution_dict() ## get spatial distribution of boutons of PNs based on glomerulus and clusters
        self.a.add_claw_spatial_distribution_dict() ## get spatial distribution of claws of KCs based on KC classes, and KC subclasses
        self.a.save_neuron_coordinates_in_neuropil()
        self.a.save_synapse_coordinate_neuropil()
        self.a.save_spatial_distribution_dict()
        self.a.calculate_JS_divergent_for_all()
        self.a.pool_shuffled_result()

        with open('init.txt','wt')as ff:
            ff.writelines('finished')
        return

    def get_relationship_spatial_connection_preference(self):
        '''
        fig. s12
        Preference is defined as the (real data - shuffled_mean)/ shuffled_std
        :return:
        '''
        self.a.analyze_spatial_distribution_connection_preference()

    def get_spatial_input_weight_ratio(self):
        '''
        fig. 2e
        :return:
        '''
        self.a.plot_spatial_input_weight_ratio()



if __name__ == '__main__':
    a = Anatomical_analysis()
    plot_result = figure_manager() ## each function indicates the related figure
    # plot_result.init_get_spatial_distribution_dict()
    # plot_result.visualize_glomerulus_cluster_AL()
    # plot_result.visualize_synaptic_distribution()
    # plot_result.visualize_spatial_distribution_density(Classification_type='Cluster',data_type='neuron',neuropil='CA(R)')
    # plot_result.compare_group_connection_num_weight()
    # plot_result.get_spatial_distribution_preference()
    # plot_result.get_DV_ratio_comparison(t1='Cluster')
    # plot_result.get_synapse_spatial_similarity()
    # plot_result.get_relationship_spatial_connection_preference()
    # plot_result.get_spatial_input_weight_ratio()
    # plot_result.get_bouton_connection_number_ratio()
