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

plt.rcParams['font.family'] = 'Arial'

class Anatomical_analysis():
    def __init__(self, root='Result/', threshold=3, neuropil_path='hemibrain_data/FlyEM_neuropil/'):
        if not os.path.isdir(root): os.mkdir(root)
        self.root = root
        self.neuropil_path = neuropil_path
        self.tmp_file_path = 'tmp_files/'
        self.swc_path = 'hemibrain_data/FlyEM_skeleton/'
        self.swc_source_path = "D:/eFlyPlotv2p1/Data/FlyEM_skeleton/"
        self.PN_to_KC_weight_threshold = threshold
        self.network = gc.load_network()
        self.Glomerulus_to_Cluster, self.Cluster_to_Glomerulus, self.PNid_to_Cluster, self.Cluster_to_PNid, \
        self.PNid_to_Glomerulus, self.Glomerulus_to_PNid, self.KCid_to_Subtype, self.Subtype_to_KCid = \
            self.network.obtain_lookup_dict_weight()
        self.connection_raw_data_path = "hemibrain_data/"
        self.spatial_distribution_dict = {}
        self.neuron_coordinate_neuropil_dict = {}
        self.synapse_coordinate_neuropil_dict = {}
        self.neuropil_space_dict = {}
        self.neuropil_coord_dict = {}
        self.bounding_box_dict = {}
        self.result_root = 'Result/'
        self.result_fig2 = f'{self.result_root}Fig2/'
        self.result_fig4 = f'{self.result_root}Fig4/'
        self.network.get_new_KC_subtype()
        self.prepare_color_dict()
        self.bouton_coordinate_neuropil_dict = {}
        self.claw_coordinate_neuropil_dict = {}
        self.path_claw_bouton = f'{self.connection_raw_data_path}PN_KC_bouton_claw_information_20230618/'
        if not os.path.isdir(self.result_root): os.mkdir(self.result_root)
        if not os.path.isdir(self.result_fig2): os.mkdir(self.result_fig2)
        if not os.path.isdir(self.result_fig4): os.mkdir(self.result_fig4)
        self.get_connection_preference()
        self.plot_order_list_dict = {'major': ['KCg', "KCa'b'", "KCab"],
                                     'minor': ["KCg-m", 'KCg-d', "KCa'b'-ap1", "KCa'b'-ap2", "KCa'b'-m", "KCab-p",
                                               "KCab-s", "KCab-m", "KCab-c"],
                                     "Cluster": [1, 2, 3],
                                     "Glomerulus": self.network.G_list
                                     }
        self.fontdict = {'label': {'fontsize': 20}, 'tick': {'fontsize': 16}}
        self.Classification_dict = {"Glomerulus": self.network.G_list, 'Cluster': [1, 2, 3],
                                    'minor': self.plot_order_list_dict['minor'], 'major': ['KCg', "KCa'b'", "KCab"]}
        self.id_to_classification_dict = {"Glomerulus": self.PNid_to_Glomerulus, "Cluster": self.PNid_to_Cluster,
                                          "major": self.KCid_to_Subtype, 'minor': self.network.id_to_new_subtype}
        # self.prepare_color_dict()
        self.fig_color_dict = self.Color_dict
        self.fig_color_dict["KCa'b'"] = 'gold'
        self.fig_color_dict[2] = 'gold'
        self.xyzi = []

    def get_bouton_connection_number_ratio(self):
        PN_to_KC_weight_threshold = 3
        network = ConnectionSetting()
        network.PN_to_KC_weight_threshold = PN_to_KC_weight_threshold
        Glomerulus_to_Cluster, Cluster_to_Glomerulus, PNid_to_Cluster, Cluster_to_PNid, PNid_to_Glomerulus, \
        Glomerulus_to_PNid, KCid_to_Subtype, Subtype_to_KCid = network.obtain_lookup_dict_weight()
        weight = network.connection_matrix_collection_dict['FlyEM'][0]
        Cluster_connect_num_dict = {}
        G_total_connect_num_dict = {}
        for i, PNid in enumerate(network.PNid_list):
            G = PNid_to_Glomerulus[PNid]
            C = PNid_to_Cluster[PNid]
            if G not in G_total_connect_num_dict:
                G_total_connect_num_dict[G] = 0
            if C not in Cluster_connect_num_dict:
                Cluster_connect_num_dict[C] = 0

            for j, KCid in enumerate(network.KCid_list):
                if weight[i][j] > 0:
                    G_total_connect_num_dict[G] += 1
                    Cluster_connect_num_dict[C] += 1

        network.transform_PN_KC_connection_to_G_KC_connection()
        fontdict = {"fontsize": 28}
        tick_fontsize = 20
        #
        path_claw_bouton = 'PN_KC_bouton_claw_information_20230618/'

        G_bouton_num_dict = {}
        G_connect_num_dict = {}
        ratio_dict = {}
        pooled_data = []
        for PN_index, PNid in enumerate(network.PNid_list):
            G = PNid_to_Glomerulus[PNid]
            if G not in G_bouton_num_dict:
                G_bouton_num_dict[G] = []
                G_connect_num_dict[G] = []
                ratio_dict[G] = []
            connect_num = np.count_nonzero(network.pre_to_post_weight[PN_index])
            bouton_num = 0
            try:
                print(f"{path_claw_bouton}{PNid}_bouton.txt")
                with open(f"{path_claw_bouton}{PNid}_bouton.txt", 'rt')as ff:
                    for line in ff:
                        if line[0] != '\n':
                            bouton_num += 1
            except:
                continue
            if bouton_num == 0:
                continue
            G_connect_num_dict[G].append(connect_num)
            G_bouton_num_dict[G].append(bouton_num)
            ratio_dict[G].append(connect_num / bouton_num)
            pooled_data.append([G, PNid_to_Cluster[PNid], connect_num, bouton_num, connect_num / bouton_num])

        pooled_data = pd.DataFrame(data=pooled_data,
                                   columns=['Glomerulus', 'Cluster', 'Connect num', 'Bouton num', 'Ratio'])
        pooled_data.to_excel("PN_bouton_connection_num.xlsx")
        pooled_data = pd.read_excel("PN_bouton_connection_num.xlsx")

        # Group by 'Glomerulus' and calculate the sum and standard error of 'Connect num'
        grouped_data = pooled_data.groupby('Glomerulus')['Connect num']
        df_grouped = grouped_data.sum().reset_index()
        df_error = grouped_data.std().reset_index()

        # Define the color_dict to map colors to each glomerulus
        color_dict = {}
        for G in network.G_list:
            if Glomerulus_to_Cluster[G] == 1:
                color_dict[G] = 'red'
            elif Glomerulus_to_Cluster[G] == 2:
                color_dict[G] = 'gold'
            else:
                color_dict[G] = 'deepskyblue'

        df_sorted = df_grouped.sort_values(by='Connect num', ascending=False)

        # Create a bar plot with error bars and custom colors for each glomerulus
        plt.figure(figsize=(15, 4.5))
        for glomerulus in df_sorted['Glomerulus']:
            glomerulus_data = df_sorted[df_sorted['Glomerulus'] == glomerulus]
            plt.bar(glomerulus_data['Glomerulus'], glomerulus_data['Connect num'], label=glomerulus, color='gray',
                    alpha=0.7)

        # Create a bar plot with error bars
        plt.xlabel('Glomerulus', fontdict=fontdict)
        plt.ylabel('KC number', fontdict=fontdict)
        plt.xticks(rotation=90, fontsize=tick_fontsize)
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5)  # X-axis
        ax.spines['left'].set_linewidth(1.5)  # Y-axis
        ax.spines['top'].set_linewidth(1.5)  # X-axis
        ax.spines['right'].set_linewidth(1.5)  # Y-axis

        G_list = df_sorted['Glomerulus'].values.tolist()
        for x_tick_index, xtick in enumerate(ax.get_xticklabels()):
            xtick.set_color(color_dict[G_list[x_tick_index]])
        plt.yticks([0, 150, 300, 450], fontsize=tick_fontsize)
        plt.ylim([-2, 500])
        plt.tight_layout()
        plt.show()

        data_collected = [[G, Glomerulus_to_Cluster[G], G_total_connect_num_dict[G]] for G in G_total_connect_num_dict]
        Df(data=np.array(data_collected), columns=['Glomerulus', 'Cluster', "Connect num"]).to_excel(
            "Glomerulus_Cluster_Connect.xlsx")
        data_collected = pd.read_excel("Glomerulus_Cluster_Connect.xlsx")
        sns.violinplot(data=data_collected, x='Cluster', y="Connect num")

        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5)  # X-axis
        ax.spines['left'].set_linewidth(1.5)  # Y-axis
        ax.spines['top'].set_linewidth(1.5)  # X-axis
        ax.spines['right'].set_linewidth(1.5)  # Y-axis

        plt.xlabel("Cluster", fontdict=fontdict)
        plt.ylabel("KC number", fontdict=fontdict)
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks([0, 300, 600], fontsize=tick_fontsize)
        plt.ylim([-2, 650])
        plt.tight_layout()
        plt.show()

        ### Ratio
        # Group by 'Glomerulus' and calculate the mean and standard error of 'ratio'
        grouped_data = pooled_data.groupby('Glomerulus')['Ratio']
        df_grouped = grouped_data.mean().reset_index()
        df_error = grouped_data.std().reset_index()
        # Sort the DataFrame in descending order based on the mean 'Ratio' values
        df_sorted = df_grouped.sort_values(by='Ratio', ascending=False)
        plt.figure(figsize=(15, 4.5))
        for glomerulus in df_sorted['Glomerulus']:
            glomerulus_data = df_sorted[df_sorted['Glomerulus'] == glomerulus]
            plt.bar(glomerulus_data['Glomerulus'], glomerulus_data['Ratio'], yerr=df_error[df_error['Glomerulus'] == glomerulus]['Ratio'], capsize=5, alpha=0.7, ecolor='black', label=glomerulus, color='gray')
        plt.xlabel('Glomerulus', fontdict=fontdict)
        plt.ylabel('KC number/bouton', fontdict=fontdict)
        plt.xticks(rotation=90, fontsize=tick_fontsize)
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5)  # X-axis
        ax.spines['left'].set_linewidth(1.5)  # Y-axis
        ax.spines['top'].set_linewidth(1.5)  # X-axis
        ax.spines['right'].set_linewidth(1.5)  # Y-axis

        G_list = df_sorted['Glomerulus'].values.tolist()
        for x_tick_index, xtick in enumerate(ax.get_xticklabels()):
            xtick.set_color(color_dict[G_list[x_tick_index]])
        plt.yticks([0, 20, 40], fontsize=tick_fontsize)
        plt.ylim([-2, 45])
        plt.tight_layout()
        plt.show()

        sns.violinplot(data=pooled_data,x='Cluster', y="Ratio")

        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5)  # X-axis
        ax.spines['left'].set_linewidth(1.5)  # Y-axis
        ax.spines['top'].set_linewidth(1.5)  # X-axis
        ax.spines['right'].set_linewidth(1.5)  # Y-axis

        plt.xlabel("Cluster", fontdict=fontdict)
        plt.ylabel("KC number/bouton", fontdict=fontdict)
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks([0, 30, 60], fontsize=tick_fontsize)
        plt.ylim([-2, 60])
        plt.tight_layout()
        plt.show()

    def get_divided_CA_along_x_z_by_x(self):
        neuropil_name = "CA(R)"
        if 'Windows' in platform.platform():
            path = 'D:/eFlyPlotv2p1/Data/'
        elif 'Linux' in platform.platform():
            path = '/mnt/DS416j/charng/eFlyPlotv2p1/Data/'
        neuropil = trimesh.load(f'{path}FlyEM_neuropil/{neuropil_name}.obj')
        x, y, z = neuropil.vertices.T
        pca = PCA(n_components=2)
        X = np.array([x, z]).transpose()
        pca.fit(X)
        X_fit = pca.transform(X)
        neuropil = trimesh.load(f'{path}FlyEM_neuropil/MB(R).obj')
        x, y, z = neuropil.vertices.T
        X_MB = pca.transform(np.array([x, z]).transpose())
        plt.plot(X_MB[:, 0], X_MB[:, 1], '.', color='gray')
        plt.plot(X_fit[:, 0], X_fit[:, 1], '.')
        max_x = np.max(X_fit[:, 0])
        min_x = np.min(X_fit[:, 0])
        x = X_fit[:, 0]
        print(x[x > np.mean(x)])
        z_max_x = np.mean(X_fit[:, 1][x > np.mean(x)])
        z_min_x = np.mean(X_fit[:, 1][x < np.mean(x)])
        plt.plot([min_x, max_x], [z_min_x, z_max_x], 'r')
        plt.axis('equal')
        plt.savefig("DV_calyx_visualization.png", dpi=500)
        plt.close()
        # z = mx + b
        line = [[min_x, z_min_x], [max_x, z_max_x]]
        line = pca.inverse_transform(np.array(line))
        m = (line[1][1] - line[0][1]) / (line[1][0] - line[0][0])
        b = line[1][1] - m * line[1][0]
        print(m, b)
        return [m, b]

    def get_coordinate_in_divided_CA(self, Node_xyz):
        m, b = self.get_divided_CA_along_x_z_by_x()
        tmp = []
        for i in range(len(Node_xyz)):
            if Node_xyz[i][0] * m + b - Node_xyz[i][2] > 0:
                tmp.append(Node_xyz[i])
        return tmp

    def compare_neuron_in_CA_upper_and_lower(self, t1='Cluster', omit_ratio=0, neuropil='CA(R)'):
        # self.neuron_coordinate_neuropil_dict[f"FlyEM_PN_cluster_{PN_cluster}_{neuropil}_{omit_ratio}"]
        ratio = []
        for classification in self.Classification_dict[t1]:
            print(self.neuron_coordinate_neuropil_dict.keys())
            if t1 == 'Cluster':
                pooled_coordinate = self.neuron_coordinate_neuropil_dict[
                    f"FlyEM_PN_cluster_{classification}_{neuropil}_{omit_ratio}"]
            elif t1 == 'major':
                omit_ratio = 0.8
                pooled_coordinate = self.neuron_coordinate_neuropil_dict[
                    f"FlyEM_{classification}_{neuropil}_{omit_ratio}"]
            total_number = len(pooled_coordinate)
            filtered_number = len(self.get_coordinate_in_divided_CA(pooled_coordinate))
            print(classification, total_number, filtered_number)
            filtered_ratio = filtered_number / total_number * 100
            print(filtered_ratio - 100 + filtered_ratio)
            ratio.append([classification, filtered_number / (total_number - filtered_number)])
        result = Df(ratio, columns=['PN cluster', "D/V ratio"])
        print(result)
        ax = sns.barplot(data=result, y="D/V ratio", x='PN cluster', palette=['r', 'gold', 'deepskyblue'])
        ax.set_ylabel("D/V ratio", fontsize=30)
        ax.set_xlabel(f"{t1}", fontsize=30)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['1', '2', '3'], fontsize=24)
        ax.set_yticks([0, 6, 12, 18])
        ax.set_yticklabels(['0', '6', '12', '18'], fontsize=24)
        plt.tight_layout()
        w = 1.5
        ax.spines['bottom'].set_linewidth(w)  # X-axis
        ax.spines['left'].set_linewidth(w)  # Y-axis
        ax.spines['top'].set_linewidth(w)  # X-axis
        ax.spines['right'].set_linewidth(w)  # Y-axis
        plt.savefig(f"{self.result_fig2}DV ratio_{t1}.svg", format='svg')
        plt.savefig(f"{self.result_fig2}DV ratio_{t1}.png",dpi=500)
        plt.show()
        plt.close()
        return

    def prepare_color_dict(self):
        self.Color_dict = {1: 'r', 2: '#e2bb56', 3: 'deepskyblue', "KCg": 'r', "KCa'b'": '#e2bb56',
                           "KCab": 'deepskyblue'}
        for G in self.Glomerulus_to_Cluster:
            self.Color_dict[G] = self.Color_dict[self.Glomerulus_to_Cluster[G]]

        rd.seed(100)
        for minor_class in self.network.New_subtype_to_id:
            self.Color_dict[minor_class] = (rd.random(), rd.random(), rd.random())

    def print_spatial_distribution_dict_parameter(self):
        print("key_format = name_type_neuropil_omit ratio")
        print("i.e.: KCab_neuron_CA(R)_0.5")

    def move_data(self):
        if not os.path.isdir(self.swc_path): os.mkdir(self.swc_path)
        file_list = [f'{KCid}.swc' for KCid in self.network.KCid_list]
        file_list += [f'{PNid}.swc' for PNid in self.network.PNid_list]
        for file_name in file_list:
            source_path = os.path.join(self.swc_source_path, file_name)
            destination_path = os.path.join(self.swc_path, file_name)
            # Copy the file from source to destination
            shutil.copy2(source_path, destination_path)

    def get_neuron_coordinates_in_neuropil(self, neuronid, neuropil='', omit_ratio=0):
        if not f"{neuronid}_ALL_{omit_ratio}" in self.neuron_coordinate_neuropil_dict:
            xyz = np.array(self.read_swc(f'{self.swc_path}{neuronid}.swc', omit_ratio=omit_ratio))
            self.neuron_coordinate_neuropil_dict[f"{neuronid}_ALL_{omit_ratio}"] = xyz
        xyz = self.neuron_coordinate_neuropil_dict[f"{neuronid}_ALL_{omit_ratio}"]
        if neuropil:
            xyz = self.filtered_coordinates(xyz, neuropil)
            self.neuron_coordinate_neuropil_dict[f"{neuronid}_{neuropil}_{omit_ratio}"] = xyz
        return xyz

    def save_neuron_coordinates_in_neuropil(self):
        with open(f"{self.tmp_file_path}neuron_coordinates_in_neuropil_dict.pickle", 'wb')as ff:
            pickle.dump(self.neuron_coordinate_neuropil_dict, ff)

    def load_neuron_coordinates_in_neuropil(self):
        with open(f"{self.tmp_file_path}neuron_coordinates_in_neuropil_dict.pickle", 'rb')as ff:
            self.neuron_coordinate_neuropil_dict = pickle.load(ff)

    def get_PN_KC_coordinates_in_neuropil(self, KC_omit_ratio=0.8):
        print("Start to get coordinates")
        for neuropil in ['CA(R)', 'PED(R)']:
            for KCid in self.network.KCid_list:
                print(KCid, neuropil)
                self.get_neuron_coordinates_in_neuropil(KCid, neuropil, KC_omit_ratio)
            print('Finished_KC')
        for neuropil in ["CA(R)", 'LH(R)']:
            for PNid in self.network.PNid_list:
                print(PNid, neuropil)
                self.get_neuron_coordinates_in_neuropil(PNid, neuropil, 0)
            print('Finished_PN')
        return

    def shuffle_neuronId(self, neuron_id_list, Classification):
        rd.shuffle(neuron_id_list)
        for Class in Classification:
            Classification[Class] = neuron_id_list[:len(Classification[Class])]
            neuron_id_list = neuron_id_list[len(Classification[Class]):]
        return Classification

    def calculate_KL_divergent_for_all(self, neuropil_list=['CA(R)', 'LH(R)', 'PED(R)']):
        result = []
        for neuropil in neuropil_list:
            for i in self.spatial_distribution_dict:
                if neuropil not in i:
                    continue
                if 'Shuffle' in i:
                    continue
                for j in self.spatial_distribution_dict:
                    if neuropil not in j:
                        continue
                    dis = sum(rel_entr(np.ravel(self.spatial_distribution_dict[i]),
                                       np.ravel(self.spatial_distribution_dict[j])))
                    result.append([i, j, dis])
        data = pd.DataFrame(data=result, columns=['Condition 1', 'Condition 2', 'KL divergence'])
        data.to_csv(f"{self.result_fig2}KL_divergence.csv")

    def calculate_JS_divergent_for_all(self, neuropil_list=['CA(R)', 'LH(R)', 'PED(R)','AL_ALL(R)']):
        result = []
        if os.path.isfile(f"{self.result_fig2}JS_similarity.csv"):
            return
        for neuropil in neuropil_list:
            for i in self.spatial_distribution_dict:
                if neuropil not in i:
                    continue
                if 'Shuffle' in i:
                    continue
                for j in self.spatial_distribution_dict:
                    if neuropil not in j:
                        continue
                    dis = 1 - jensenshannon(np.ravel(self.spatial_distribution_dict[i]),
                                            np.ravel(self.spatial_distribution_dict[j]))
                    result.append([i, j, dis])
        data = pd.DataFrame(data=result, columns=['Condition 1', 'Condition 2', 'JS similarity'])
        data.to_csv(f"{self.result_fig2}JS_similarity.csv")

    def add_PN_synapse_neuron_based_spatial_distribution_dict(self, neuropil_list=['CA(R)', 'LH(R)'], omit_ratio=0, shuffle_num=30,
                                                seed=100, pre=True, post=False):
        '''
        Here, we have single neuron, glomerulus, cluster - three levels for PN in CA(R), LH(R)
        :return:
        '''
        if pre:
            self.get_PN_synapses_in_neuropil(pre=pre, post=False, omit_ratio=omit_ratio,
                                                 neuropil_list=neuropil_list)
            rd.seed(seed)
            for Classification_original in [self.Glomerulus_to_PNid, self.Cluster_to_PNid]:
                Classification = Classification_original
                for PN_cluster in Classification:
                    for neuropil in neuropil_list:
                        tmp = []
                        for PNid in Classification[PN_cluster]:
                            tmp += self.synapse_coordinate_neuropil_dict[f"{PNid}_{neuropil}_pre_{omit_ratio}"].tolist()
                        tmp = np.array(tmp)
                        self.synapse_coordinate_neuropil_dict[
                            f"FlyEM_PN_cluster_{PN_cluster}_{neuropil}_pre_{omit_ratio}"] = tmp
                        self.spatial_distribution_dict[f'FlyEM_synapse_PN_cluster_{PN_cluster}_{neuropil}_pre_{omit_ratio}'] \
                            = self.calculate_spatial_distribution(tmp, neuropil)
                for i in range(shuffle_num):
                    Classification = self.shuffle_neuronId(copy.deepcopy(self.network.PNid_list),
                                                           copy.deepcopy(Classification))
                    for PN_cluster in Classification:
                        for neuropil in neuropil_list:
                            tmp = []
                            for PNid in Classification[PN_cluster]:
                                tmp += self.synapse_coordinate_neuropil_dict[f"{PNid}_{neuropil}_pre_{omit_ratio}"].tolist()
                            tmp = np.array(tmp)
                            self.synapse_coordinate_neuropil_dict[
                                f"Shuffled{i}_PN_cluster_{PN_cluster}_{neuropil}_pre_{omit_ratio}"] = tmp
                            self.spatial_distribution_dict[
                                f'Shuffled{i}_synapse_PN_cluster_{PN_cluster}_{neuropil}_pre_{omit_ratio}'] \
                                = self.calculate_spatial_distribution(tmp, neuropil)

        if post:
            self.get_PN_synapses_in_neuropil(pre=False, post=post, omit_ratio=omit_ratio,
                                                 neuropil_list=neuropil_list)
            rd.seed(seed)
            for Classification_original in [self.Glomerulus_to_PNid, self.Cluster_to_PNid]:
                Classification = Classification_original
                print(Classification)
                for PN_cluster in Classification:
                    for neuropil in neuropil_list:
                        tmp = []
                        for PNid in Classification[PN_cluster]:
                            tmp += self.synapse_coordinate_neuropil_dict[f"{PNid}_{neuropil}_post_{omit_ratio}"].tolist()
                        tmp = np.array(tmp)
                        self.synapse_coordinate_neuropil_dict[
                            f"FlyEM_PN_cluster_{PN_cluster}_{neuropil}_post_{omit_ratio}"] = tmp
                        self.spatial_distribution_dict[f'FlyEM_synapse_PN_cluster_{PN_cluster}_{neuropil}_post_{omit_ratio}'] \
                            = self.calculate_spatial_distribution(tmp, neuropil)
                for i in range(shuffle_num):
                    print(f"shuffling_{i}")
                    Classification = self.shuffle_neuronId(copy.deepcopy(self.network.PNid_list), copy.deepcopy(Classification))
                    for PN_cluster in Classification:
                        for neuropil in neuropil_list:
                            tmp = []
                            for PNid in Classification[PN_cluster]:
                                tmp += self.synapse_coordinate_neuropil_dict[f"{PNid}_{neuropil}_post_{omit_ratio}"].tolist()
                            tmp = np.array(tmp)
                            self.synapse_coordinate_neuropil_dict[
                                f"Shuffled{i}_PN_cluster_{PN_cluster}_{neuropil}_post_{omit_ratio}"] = tmp
                            self.spatial_distribution_dict[
                                f'Shuffled{i}_synapse_PN_cluster_{PN_cluster}_{neuropil}_post_{omit_ratio}'] \
                                = self.calculate_spatial_distribution(tmp, neuropil)
        return

    def add_PN_neuron_spatial_distribution_dict(self, neuropil_list=['CA(R)', 'LH(R)', ], omit_ratio=0, shuffle_num=30,
                                                seed=100):
        '''
        Here, we have single neuron, glomerulus, cluster - three levels for PN in CA(R), LH(R)
        :return:
        '''
        for PNid in self.network.PNid_list:
            file_name = f"{PNid}_ALL_{omit_ratio}"
            if file_name not in self.neuron_coordinate_neuropil_dict:
                self.get_neuron_coordinates_in_neuropil(PNid, omit_ratio)
            for neuropil in neuropil_list:
                file_name = f"{PNid}_{neuropil}_{omit_ratio}"
                if file_name not in self.neuron_coordinate_neuropil_dict:
                    self.get_neuron_coordinates_in_neuropil(PNid, neuropil, omit_ratio)

        rd.seed(seed)
        for Classification_original in [self.Glomerulus_to_PNid, self.Cluster_to_PNid]:
            Classification = Classification_original
            for PN_cluster in Classification:
                for neuropil in neuropil_list:
                    tmp = []
                    for PNid in Classification[PN_cluster]:
                        tmp += self.neuron_coordinate_neuropil_dict[f"{PNid}_{neuropil}_{omit_ratio}"].tolist()
                    tmp = np.array(tmp)
                    self.neuron_coordinate_neuropil_dict[f"FlyEM_PN_cluster_{PN_cluster}_{neuropil}_{omit_ratio}"] = tmp
                    self.spatial_distribution_dict[f'FlyEM_neuron_PN_cluster_{PN_cluster}_{neuropil}_{omit_ratio}'] \
                        = self.calculate_spatial_distribution(tmp, neuropil)
            for i in range(shuffle_num):
                Classification = self.shuffle_neuronId(copy.deepcopy(self.network.PNid_list),
                                                       copy.deepcopy(Classification))
                for PN_cluster in Classification:
                    for neuropil in neuropil_list:
                        tmp = []
                        for PNid in Classification[PN_cluster]:
                            tmp += self.neuron_coordinate_neuropil_dict[f"{PNid}_{neuropil}_{omit_ratio}"].tolist()
                        tmp = np.array(tmp)
                        self.neuron_coordinate_neuropil_dict[
                            f"Shuffled{i}_PN_cluster_{PN_cluster}_{neuropil}_{omit_ratio}"] = tmp
                        self.spatial_distribution_dict[
                            f'Shuffled{i}_neuron_PN_cluster_{PN_cluster}_{neuropil}_{omit_ratio}'] \
                            = self.calculate_spatial_distribution(tmp, neuropil)
        return

    def add_KC_neuron_spatial_distribution_dict(self, omit_ratio=0.8, neuropil_list=['CA(R)', 'PED(R)'], shuffle_num=30,
                                                seed=100):
        '''
        Here, we have single neuron, KC_main_class, KC_minor_class - three levels for KC
        :return:
        '''
        for KCid in self.network.KCid_list:
            file_name = f"{KCid}_ALL_{omit_ratio}"
            if file_name not in self.neuron_coordinate_neuropil_dict:
                self.get_neuron_coordinates_in_neuropil(KCid, omit_ratio)
            for neuropil in neuropil_list:
                file_name = f"{KCid}_{neuropil}_{omit_ratio}"
                if file_name not in self.neuron_coordinate_neuropil_dict:
                    self.get_neuron_coordinates_in_neuropil(KCid, neuropil, omit_ratio)

        rd.seed(seed)
        print(self.network.New_subtype_to_id.keys())
        for Classification_original in [self.network.New_subtype_to_id, self.Subtype_to_KCid]:
            Classification = Classification_original
            for KC_class in Classification:
                for neuropil in neuropil_list:
                    tmp = []
                    for KCid in Classification[KC_class]:
                        tmp += self.neuron_coordinate_neuropil_dict[f"{KCid}_{neuropil}_{omit_ratio}"].tolist()
                    tmp = np.array(tmp)
                    self.neuron_coordinate_neuropil_dict[f"FlyEM_{KC_class}_{neuropil}_{omit_ratio}"] = tmp
                    self.spatial_distribution_dict[f'FlyEM_neuron_{KC_class}_{neuropil}_{omit_ratio}'] \
                        = self.calculate_spatial_distribution(tmp, neuropil)
            for i in range(shuffle_num):
                Classification = self.shuffle_neuronId(copy.deepcopy(self.network.KCid_list),
                                                       copy.deepcopy(Classification))
                for KC_class in Classification:
                    for neuropil in neuropil_list:
                        tmp = []
                        for KCid in Classification[KC_class]:
                            tmp += self.neuron_coordinate_neuropil_dict[f"{KCid}_{neuropil}_{omit_ratio}"].tolist()
                        tmp = np.array(tmp)
                        self.neuron_coordinate_neuropil_dict[f"Shuffled{i}_{KC_class}_{neuropil}_{omit_ratio}"] = tmp
                        self.spatial_distribution_dict[f'Shuffled{i}_neuron_{KC_class}_{neuropil}_{omit_ratio}'] \
                            = self.calculate_spatial_distribution(tmp, neuropil)

    def save_synapse_coordinate_neuropil(self):
        with open(f"{self.connection_raw_data_path}synapse_coordinate_neuropil.pickle", 'wb')as ff:
            pickle.dump(self.synapse_coordinate_neuropil_dict, ff)

    def load_synapse_coordinate_neuropil(self):
        with open(f"{self.connection_raw_data_path}synapse_coordinate_neuropil.pickle", 'rb')as ff:
            self.synapse_coordinate_neuropil_dict = pickle.load(ff)

    def get_PN_synapses_in_neuropil(self,pre=True, post=False, neuropil_list=['CA(R)','LH(R)'],omit_ratio=0):
        for PNid in self.network.PNid_list:
            directory = 'PN_synapses_total/'
            if pre:
                if f"{PNid}_all_pre_{omit_ratio}" not in self.synapse_coordinate_neuropil_dict:
                    upstream_synapse_data = pd.read_excel(f"{directory}downstream of {PNid}_synapse_v1.2.1.xlsx")
                    x, y, z = upstream_synapse_data['up_syn_coordinate_x'].values, upstream_synapse_data[
                        'up_syn_coordinate_y'].values, upstream_synapse_data['up_syn_coordinate_z'].values
                    xyz = np.array([x, y, z]).transpose()
                    xyz = np.unique(xyz, axis=0)
                    print(PNid, xyz.shape)
                    # print(xyz)
                    if omit_ratio < 0 or omit_ratio > 1:
                        raise ValueError("omit ratio should vary from 0 to 1")
                    if omit_ratio > 0:
                        num_coordinates = int((1 - omit_ratio) * len(xyz))
                        # Randomly select 30% of coordinates
                        selected_coordinates = np.random.choice(len(xyz), size=num_coordinates, replace=False)
                        # Get the selected coordinates from the original array
                        xyz = xyz[selected_coordinates]
                    self.synapse_coordinate_neuropil_dict[f"{PNid}_all_pre_{omit_ratio}"] = xyz
                for neuropil in neuropil_list:
                    if f"{PNid}_{neuropil}_pre_{omit_ratio}" not in self.synapse_coordinate_neuropil_dict:
                        try:
                            filtered_xyz = self.filtered_coordinates(xyz, neuropil)
                        except:
                            filtered_xyz = np.array([])
                            print("NONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            pass
                        self.synapse_coordinate_neuropil_dict[f"{PNid}_{neuropil}_pre_{omit_ratio}"] = filtered_xyz
            if post:
                if f"{PNid}_all_post_{omit_ratio}" not in self.synapse_coordinate_neuropil_dict:
                    downstream_synapse_data = pd.read_excel(f"{directory}upstream of {PNid}_synapse_v1.2.1.xlsx")
                    x, y, z = downstream_synapse_data['down_syn_coordinate_x'].values, downstream_synapse_data[
                        'down_syn_coordinate_y'].values, downstream_synapse_data['down_syn_coordinate_z'].values
                    xyz = np.array([x, y, z]).transpose()
                    xyz = np.unique(xyz, axis=0)
                    if omit_ratio < 0 or omit_ratio > 1:
                        raise ValueError("omit ratio should vary from 0 to 1")
                    if omit_ratio > 0:
                        num_coordinates = int((1 - omit_ratio) * len(xyz))
                        # Randomly select 30% of coordinates
                        selected_coordinates = np.random.choice(len(xyz), size=num_coordinates, replace=False)
                        # Get the selected coordinates from the original array
                        xyz = xyz[selected_coordinates]
                    self.synapse_coordinate_neuropil_dict[f"{PNid}_all_post_{omit_ratio}"] = xyz
                for neuropil in neuropil_list:
                    if f"{PNid}_{neuropil}_post_{omit_ratio}" not in self.synapse_coordinate_neuropil_dict:
                        try:
                            filtered_xyz = self.filtered_coordinates(xyz, neuropil)
                        except:
                            filtered_xyz = np.array([])
                            print("NONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            pass
                        self.synapse_coordinate_neuropil_dict[f"{PNid}_{neuropil}_post_{omit_ratio}"] = filtered_xyz

    def get_PN_KC_synapses(self):
        synapse_data = pd.read_excel(f'{self.connection_raw_data_path}PN_to_KCg_synapse.xlsx').values.tolist()
        synapse_data += pd.read_excel(f"{self.connection_raw_data_path}PN_to_KCa'b'_synapse.xlsx").values.tolist()
        synapse_data += pd.read_excel(f'{self.connection_raw_data_path}PN_to_KCab_synapse.xlsx').values.tolist()
        for synapse in synapse_data:
            PNid, KCid = synapse[1], synapse[4]
            PN_xyz = synapse[7:10]
            KC_xyz = synapse[10:]
            if PNid not in self.synapse_coordinate_neuropil_dict:
                self.synapse_coordinate_neuropil_dict[PNid] = {}
            if KCid not in self.synapse_coordinate_neuropil_dict:
                self.synapse_coordinate_neuropil_dict[KCid] = {}
            if KCid not in self.synapse_coordinate_neuropil_dict[PNid]:
                self.synapse_coordinate_neuropil_dict[PNid][KCid] = []
            if PNid not in self.synapse_coordinate_neuropil_dict[KCid]:
                self.synapse_coordinate_neuropil_dict[KCid][PNid] = []
            self.synapse_coordinate_neuropil_dict[KCid][PNid].append(KC_xyz)
            self.synapse_coordinate_neuropil_dict[PNid][KCid].append(PN_xyz)

    def add_PN_synapse_spatial_distribution_dict(self, shuffle_num=30, seed=100):
        for Classification_original in [self.Glomerulus_to_PNid, self.Cluster_to_PNid]:
            Classification = Classification_original
            packed = []
            num = []
            for clusterid in Classification:
                tmp = []
                count = 0
                for PNid in Classification[clusterid]:
                    for KCid in self.synapse_coordinate_neuropil_dict[PNid]:
                        tmp += self.synapse_coordinate_neuropil_dict[PNid][KCid]
                        packed.append(self.synapse_coordinate_neuropil_dict[PNid][KCid])
                        count += 1
                tmp = np.array(tmp)
                num.append(count)
                self.synapse_coordinate_neuropil_dict[f"FlyEM_synapse_PN_cluster_{clusterid}_to_KC_CA(R)"] = tmp
                self.spatial_distribution_dict[f"FlyEM_synapse_PN_cluster_{clusterid}_to_KC_CA(R)"] \
                    = self.calculate_spatial_distribution(tmp, "CA(R)")

            rd.seed(seed)
            for i in range(shuffle_num):
                rd.shuffle(packed)
                tmp_packed = copy.deepcopy(packed)
                for clusterindex, clusterid in enumerate(Classification):
                    tmp = []
                    print(clusterindex, clusterid)
                    for j in range(num[clusterindex]):
                        tmp += tmp_packed[j]
                    tmp_packed = tmp_packed[num[clusterindex]:]
                    tmp = np.array(tmp)
                    self.synapse_coordinate_neuropil_dict[
                        f"Shuffled{i}_synapse_PN_cluster_{clusterid}_to_KC_CA(R)"] = tmp
                    self.spatial_distribution_dict[f"Shuffled{i}_synapse_PN_cluster_{clusterid}_to_KC_CA(R)"] \
                        = self.calculate_spatial_distribution(tmp, 'CA(R)')

    def add_KC_synapse_spatial_distribution_dict(self, shuffle_num=30, seed=100):
        for Classification_original in [self.Subtype_to_KCid, self.network.New_subtype_to_id]:
            Classification = Classification_original
            packed = []
            num = []
            for classid in Classification:
                tmp = []
                count = 0
                for KCid in Classification[classid]:
                    for PNid in self.synapse_coordinate_neuropil_dict[KCid]:
                        tmp += self.synapse_coordinate_neuropil_dict[KCid][PNid]
                        packed.append(self.synapse_coordinate_neuropil_dict[KCid][PNid])
                        count += 1
                num.append(count)
                tmp = np.array(tmp)
                self.synapse_coordinate_neuropil_dict[f"FlyEM_synapse_{classid}_from_PN_CA(R)"] = tmp
                self.spatial_distribution_dict[f"FlyEM_synapse_{classid}_from_PN_CA(R)"] \
                    = self.calculate_spatial_distribution(tmp, "CA(R)")
            rd.seed(seed)
            for i in range(shuffle_num):
                rd.shuffle(packed)
                tmp_packed = copy.deepcopy(packed)
                for classindex, classid in enumerate(Classification):
                    tmp = []
                    for j in range(num[classindex]):
                        tmp += tmp_packed[j]
                    tmp_packed = tmp_packed[num[classindex]:]
                    tmp = np.array(tmp)
                    self.synapse_coordinate_neuropil_dict[f"Shuffled{i}_synapse_{classid}_from_PN_CA(R)"] = tmp
                    self.spatial_distribution_dict[f"Shuffled{i}_synapse_{classid}_from_PN_CA(R)"] \
                        = self.calculate_spatial_distribution(tmp, "CA(R)")

    def add_bouton_spatial_distribution_dict(self, seed=100, shuffle_num=30):
        print("Bouton")
        for Classification_original in [self.Cluster_to_PNid, self.Glomerulus_to_PNid]:
            print(Classification_original)
            Classification = Classification_original
            packed = []
            num = []
            for clusterid in Classification:
                tmp = []
                for PNid in Classification[clusterid]:
                    try:
                        Node_xyz = self.read_bouton_claw(f'{PNid}_bouton.txt', self.path_claw_bouton)
                        ## Node_xyz = [[x1,y1,z1],[x2,y2,z2]]
                        tmp += Node_xyz
                        packed += Node_xyz
                    except:
                        continue
                num.append(len(tmp))
                tmp = np.array(tmp)
                if num[-1] <= 3:
                    ## if bouton num <= 3, it is not suitable to estimate density which needs more data points
                    continue
                self.bouton_coordinate_neuropil_dict[f"FlyEM_{clusterid}_CA(R)"] = tmp
                self.spatial_distribution_dict[f"FlyEM_bouton_PN_cluster_{clusterid}_CA(R)"] \
                    = self.calculate_spatial_distribution(tmp, "CA(R)")

            rd.seed(seed)
            for i in range(shuffle_num):
                rd.shuffle(packed)
                tmp_packed = copy.deepcopy(packed)
                for clusterindex, clusterid in enumerate(Classification):
                    tmp = tmp_packed[:num[clusterindex]]
                    tmp_packed = tmp_packed[num[clusterindex]:]
                    # print(len(tmp_packed))
                    tmp = np.array(tmp)
                    if num[clusterindex] <= 3:
                        ## if bouton num <= 3, it is not suitable to estimate density which needs more data points
                        continue
                    self.bouton_coordinate_neuropil_dict[f"Shuffled{i}_bouton_PN_cluster_{clusterid}_CA(R)"] = tmp
                    self.spatial_distribution_dict[f"Shuffled{i}_bouton_PN_cluster_{clusterid}_CA(R)"] \
                        = self.calculate_spatial_distribution(tmp, 'CA(R)')

    def add_claw_spatial_distribution_dict(self, seed=100, shuffle_num=30):
        print("Claw")
        for Classification_original in [self.Subtype_to_KCid, self.network.New_subtype_to_id]:
            print(Classification_original)
            Classification = Classification_original
            packed = []
            num = []
            for classid in Classification:
                if 'KCg-s4' in classid:
                    num.append(0)
                    continue
                tmp = []
                count = 0
                for KCid in Classification[classid]:
                    try:
                        Node_xyz = self.read_bouton_claw(f'{KCid}_claw.txt', self.path_claw_bouton)
                        # print(Node_xyz)
                        tmp += Node_xyz
                        packed += Node_xyz
                    except:
                        continue
                count = len(tmp)
                tmp = np.array(tmp)
                num.append(count)
                if num[-1] <= 3:
                    ## if bouton num <= 3, it is not suitable to estimate density which needs more data points
                    continue

                self.claw_coordinate_neuropil_dict[f"FlyEM_{classid}"] = np.array(tmp)
                self.spatial_distribution_dict[f"FlyEM_claw_{classid}_CA(R)"] \
                    = self.calculate_spatial_distribution(tmp, "CA(R)")

            rd.seed(seed)
            for i in range(shuffle_num):
                rd.shuffle(packed)
                tmp_packed = copy.deepcopy(packed)
                tmp_length = 0
                for classindex, classid in enumerate(Classification):
                    tmp_length += num[classindex]
                    print(tmp_length)
                    if 'KCg-s4' in classid:
                        continue
                    tmp = tmp_packed[:num[classindex]]
                    # if len(tmp_packed) != len(tmp):
                    tmp_packed = tmp_packed[num[classindex]:]
                    tmp = np.array(tmp)
                    if num[classindex] <= 3:
                        ## if bouton num <= 3, it is not suitable to estimate density which needs more data points
                        continue

                    self.claw_coordinate_neuropil_dict[f"Shuffled{i}_claw_{classid}_CA(R)"] = tmp
                    print(f"Shuffled_{classid}")
                    self.spatial_distribution_dict[f"Shuffled{i}_claw_{classid}_CA(R)"] \
                        = self.calculate_spatial_distribution(tmp, 'CA(R)')

    def load_spatial_distribution_dict(self):
        if os.path.isfile(f"{self.tmp_file_path}spatial_distribution_dict.pickle"):
            with open(f"{self.tmp_file_path}spatial_distribution_dict.pickle", 'rb')as ff:
                self.spatial_distribution_dict = pickle.load(ff)
        else:
            print("No spatial distribution file!")

    def save_spatial_distribution_dict(self):
        with open(f"{self.tmp_file_path}spatial_distribution_dict.pickle", 'wb')as ff:
            pickle.dump(self.spatial_distribution_dict, ff)

    def get_bounding_box_of_neuropil(self, neuropil_space):
        x, y, z = neuropil_space.vertices.T
        bounding_box = np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z)
        return bounding_box

    def filtered_coordinates_out_of_bounding_box(self, coordinate, bounding_box): ## Checked
        xmin, xmax, ymin, ymax, zmin, zmax = bounding_box
        x, y, z = coordinate.T
        mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax) & (z >= zmin) & (z <= zmax)
        return coordinate[mask]

    def filtered_coordinates(self, coordinate, neuropil): ## checked
        if neuropil not in self.neuropil_space_dict:
            neuropil_space = trimesh.load(f'{self.neuropil_path}{neuropil}.obj')
            self.neuropil_space_dict[neuropil] = neuropil_space
        neuropil_space = self.neuropil_space_dict[neuropil]
        if neuropil not in self.bounding_box_dict:
            bounding_box = self.get_bounding_box_of_neuropil(neuropil_space)
            self.bounding_box_dict[neuropil] = bounding_box
        bounding_box = self.bounding_box_dict[neuropil]
        coordinate = self.filtered_coordinates_out_of_bounding_box(coordinate, bounding_box)
        if neuropil == 'AL_ALL':
            return coordinate
        if len(coordinate) > 0:
            filtered_coordinate = coordinate[proximity.signed_distance(neuropil_space, coordinate) >= 0]
        else:
            filtered_coordinate = np.array([])
        return filtered_coordinate

    def calculate_spatial_distribution(self, coordinate, neuropil='', bounding_box=[], xyz_slice_num=[20, 20, 20]):
        if neuropil:         # If neuropil is provided, use it to define the spatial space
            if neuropil not in self.neuropil_space_dict:
                neuropil_space = trimesh.load(f'{self.neuropil_path}{neuropil}.obj')
                self.neuropil_space_dict[neuropil] = neuropil_space
            neuropil_space = self.neuropil_space_dict[neuropil]
            x, y, z = neuropil_space.vertices.T
            xmin, xmax, ymin, ymax, zmin, zmax = np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z)
        elif bounding_box:         # If bounding_box is provided, use it to define the spatial space
            xmin, xmax, ymin, ymax, zmin, zmax = bounding_box
        # Get the number of slices along each axis
        x_num, y_num, z_num = xyz_slice_num
        # Create a grid of points in 3D space
        xi, yi, zi = np.mgrid[xmin:xmax:complex(0, x_num), ymin:ymax:complex(0, y_num), zmin:zmax:complex(0, z_num)]
        # Stack the grid coordinates into a 3xN array
        coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
        if neuropil:
            self.neuropil_coord_dict[neuropil] = coords
        kde = stats.gaussian_kde(coordinate.transpose())
        # Evaluate the density at the grid points and reshape it to match the grid shape
        density = kde(coords).reshape(xyz_slice_num)
        print("Finished density evaluation")
        return density

    def visualize_multi_kde(self, values_collection, xyz_number=[10, 10, 10], show=False, obj=[],
                            cmap_collection=['Reds']):
        if len(values_collection) != len(cmap_collection):
            print('Error: cmap number is not equal to data number')
            return [], [], []
        data_path = 'D:/eFlyPlotv2p1/Data/FlyEM_neuropil/'
        for value_id, values in enumerate(values_collection):
            tmp_xmin, tmp_ymin, tmp_zmin = values.min(axis=1)
            tmp_xmax, tmp_ymax, tmp_zmax = values.max(axis=1)
            if value_id == 0:
                xmin, ymin, zmin = tmp_xmin, tmp_ymin, tmp_zmin
                xmax, ymax, zmax = tmp_xmax, tmp_ymax, tmp_zmax
            else:
                xmin, ymin, zmin = min(tmp_xmin, xmin), min(tmp_ymin, ymin), min(tmp_zmin, zmin)
                xmax, ymax, zmax = min(tmp_xmax, xmax), min(tmp_ymax, ymax), min(tmp_zmax, zmax)
        xi, yi, zi = np.mgrid[xmin:xmax:complex(0, xyz_number[0]), ymin:ymax:complex(0, xyz_number[1]),
                     zmin:zmax:complex(0, xyz_number[2])]
        coords = np.vstack([item.ravel() for item in [xi, yi, zi]])

        if show:
            v = mlab.figure()
            mlab.clf()
            v.scene.background = (1, 1, 1)
            mlab.options.backend = 'envisage'
            if obj:
                reader = tvtk.OBJImporter()
                for neuropil_file in obj:
                    neuropil = trimesh.load(f'D:/eFlyPlotv2p1/Data/FlyEM_neuropil/{neuropil_file}.obj')
                    x, y, z = neuropil.vertices.T
                    mlab.triangular_mesh(x, y, z, neuropil.faces, color=(0.9, 0.9, 0.9), opacity=0.5)
                    # values_n = get_obj_vertices(data_path + neuropil).T
                    # kde_n = stats.gaussian_kde(values_n)
                    # # Create a regular 3D grid with 50 points in each dimension
                    # xmin_n, ymin_n, zmin_n = values_n.min(axis=1)
                    # xmax_n, ymax_n, zmax_n = values_n.max(axis=1)
                    # xi_n, yi_n, zi_n = np.mgrid[xmin_n:xmax_n:np.complex(0, xyz_number[0]),
                    #                    ymin_n:ymax_n:np.complex(0, xyz_number[1]),
                    #                    zmin_n:zmax_n:np.complex(0, xyz_number[2])]
                    # # Evaluate the KDE on a regular grid...
                    # coords_n = np.vstack([item.ravel() for item in [xi_n, yi_n, zi_n]])
                    # density_n = kde_n(coords_n).reshape(xi_n.shape)
                    # mlab.contour3d(xi_n, yi_n, zi_n, density_n, opacity=0.1, color=(0.9, 0.9, 0.9))
        density_collection = []
        for values, cmap in zip(values_collection, cmap_collection):
            kde = stats.gaussian_kde(values)
            density = kde(coords).reshape(xi.shape)
            density_collection.append(density)
            # Visualize the density estimate as isosurfaces
            if show:
                # v.scene.add_actor(mlab.contour3d(xi, yi, zi, density, opacity=0.5))
                mlab.contour3d(xi, yi, zi, density, opacity=0.5, colormap=cmap)
                # mlab.axes()
        if show:
            mlab.show()
        return density_collection, [xmin, ymin, zmin], [xmax, ymax, zmax]

    def read_bouton_claw(self, file_name, path):
        Nodex_xyz = []
        # print(file_name)
        if not os.path.isfile(path + file_name):
            return Nodex_xyz
        with open(path + file_name, 'rt')as ff:
            for line in ff:
                if line[0] == "\n":
                    break
                groups = line.split(" ")
                x, y, z = float(groups[0]), float(groups[1]), float(groups[2])
                Nodex_xyz.append([x, y, z])
        # print(Nodex_xyz)
        return Nodex_xyz

    def read_swc(self, file_name, omit_ratio=0):
        xyz = []
        with open(file_name, "rt")as ff:
            for line in ff:
                if line.find("#") != -1 or rd.random() < omit_ratio:
                    continue
                groups = line[:-1].split(" ")
                x, y, z = float(groups[2]), float(groups[3]), float(groups[4])
                xyz.append([x, y, z])
        return xyz

    def rename_KC_subtype(self, string):
        '''
        This function is used to rename KC subtypes for the greek letters.
        :param string:
        :return:
        '''
        group = string.split("-")
        if "ab" in group[0]:
            group[0] = "a/b"
        elif "a'b'" in group[0]:
            group[0] = "a'/b'"
        elif 'g' in group[0]:
            group[0] = 'g'

        group[0] = group[0].replace('a', '\u03B1').replace('b', '\u03B2').replace('g', '\u03B3')
        new_string = ""
        for index, s in enumerate(group):
            if index == 0:
                new_string += s
            else:
                new_string += '-' + s
        return new_string

    def get_connection_zscore_correlation(self, zscoretype='cellular', savefig=False, visualize=True,
                                          method='complete'):
        path = f'{self.root}z_score_related/'
        if not os.path.isdir(path): os.mkdir(path)
        data = pd.read_excel(
            f"{self.connection_raw_data_path}Preference_score_threshold_{self.PN_to_KC_weight_threshold}_{zscoretype}.xlsx")
        G_list = data['glomerulus'].values.tolist()
        xyz = data.to_numpy()[:, 1:]
        xyz = xyz.tolist()
        xyz = np.array(xyz)
        corr = np.corrcoef(xyz)
        c = []
        col_colors = {}
        for G in G_list:
            col_colors[G] = self.Color_dict[self.Glomerulus_to_Cluster[G]]
            c.append(self.Color_dict[self.Glomerulus_to_Cluster[G]])
        sns.clustermap(data=corr, xticklabels=G_list, yticklabels=G_list, col_colors=c, cmap='bwr', method=method,
                       tree_kws={"linewidths": 0.})
        if savefig:
            plt.savefig(f"{path}z_score_pearson_correlation_clustermap_complete.png", dpi=500)
        if visualize:
            plt.show()
        plt.close()

    def check_cluster_subtype_synapse_num(self):
        matrix = self.network.connection_matrix_collection_dict['FlyEM'][0]
        PN_cluster_data = []
        KC_major_class_data = []
        Glomerulus_data = []
        KC_minor_class_data = []
        for i, PNid in enumerate(self.network.PNid_list):
            for j, KCid in enumerate(self.network.KCid_list):
                if matrix[i][j] > 0:
                    PN_cluster_data.append([self.PNid_to_Cluster[PNid], matrix[i][j]])
                    Glomerulus_data.append([self.PNid_to_Glomerulus[PNid], matrix[i][j]])
                    KC_major_class_data.append([self.KCid_to_Subtype[KCid], matrix[i][j]])
                    KC_minor_class_data.append([self.network.id_to_new_subtype[KCid], matrix[i][j]])
        data = pd.DataFrame(data=PN_cluster_data, columns=['Cluster', 'Weight'])
        data.to_csv("PN_cluster_weight_to_KC.csv")
        # sns.violinplot(data=data,x='Cluster',y='Weight',palette=['r','gold','deepskyblue'])
        # plt.show()
        data = pd.DataFrame(data=KC_major_class_data, columns=['Class', 'Weight'])
        data.to_csv("KC_major_class_weight_from_PN.csv")
        # sns.violinplot(data=data, x='Class', y='Weight', palette=['r', 'gold', 'deepskyblue'])
        # plt.show()
        data = pd.DataFrame(data=Glomerulus_data, columns=['Glomerulus', 'Weight'])
        data.to_csv("Glomerulus_weight_to_KC.csv")
        # sns.violinplot(data=data, x='Glomerulus', y='Weight')
        # plt.show()
        data = pd.DataFrame(data=KC_minor_class_data, columns=['Class', 'Weight'])
        data.to_csv("KC_minor_class_weight_from_PN.csv")
        # sns.violinplot(data=data, x='Class', y='Weight')
        # plt.show()

    def output_spatial_list(self):
        with open(f'{self.result_root}key_list_0711.txt', 'wt')as ff:
            for i in self.spatial_distribution_dict:
                ff.writelines(f"{i}\n")

    def pool_shuffled_result(self, result_type='JS'):
        if result_type == 'JS':
            df = pd.read_csv(f"{self.result_fig2}JS_similarity.csv")
        elif result_type == 'KL':
            df = pd.read_csv(f"{self.result_fig2}KL_divergence.csv")
        data = df.values.tolist()
        condition_based_dict = {}
        for i in range(len(data)):
            c1 = data[i][1]
            c2 = data[i][2]
            print(c1, c2)
            combined = c1[c1.find("_") + 1:] + ',' + c2[c2.find("_") + 1:]
            print(combined)
            if combined not in condition_based_dict:
                condition_based_dict[combined] = [[], []]
            if 'FlyEM' in c2:
                condition_based_dict[combined][0] = data[i][3]
            elif 'Shuffle' in c2:
                condition_based_dict[combined][1].append(data[i][3])
        pooled_data = []
        for condition in condition_based_dict:
            pooled_data.append(
                [condition, condition_based_dict[condition][0], np.mean(condition_based_dict[condition][1]),
                 np.std(condition_based_dict[condition][1])])
        pooled_data = pd.DataFrame(data=pooled_data, columns=['Condition', 'FlyEM', 'Shuffled mean', 'Shuffled STD'])
        if result_type == "JS":
            pooled_data.to_csv("JS_analysis_summary_0711.csv")
        elif result_type == "KL":
            pooled_data.to_csv("KL_analysis_summary_0711.csv")

    # def create_condition_collection(self,condition,condition_collection):
    #     if condition == 'Glomerulus' or condition == 'Cluster':
    #         condition_collection.append([True, 'PN'])
    #         if condition == 'Glomerulus':
    #
    #         for i in range(1,4):
    #         condition_collection.append([False, 'PN_cluster_1'])
    #             condition_collection.append([False, 'PN_cluster_2'])
    #             condition_collection.append([False, 'PN_cluster_3'])
    #         if condition == 'Cluster':
    #             condition_collection.append([False, 'PN_cluster_1'])
    #             condition_collection.append([False, 'PN_cluster_2'])
    #             condition_collection.append([False, 'PN_cluster_3'])
    #
    #
    #     return condition_collection

    def get_condition_neuropil(self,string):
        neuropil = ''
        if 'CA(R)' in string:
            neuropil = 'CA(R)'
        elif 'LH(R)' in string:
            neuropil = 'LH(R)'
        elif 'PED(R)' in string:
            neuropil = 'PED(R)'
        elif 'AL_ALL(R)' in string:
            neuropil = 'AL_ALL(R)'

        return neuropil

    def get_condition_level(self, string):
        level = ''
        if 'neuron' in string:
            level = 'neuron'
        elif 'bouton' in string:
            level = 'bouton'
        elif 'claw' in string:
            level = 'claw'
        elif 'synapse' in string:
            level = 'synapse'
        return level

    def get_condition_classification(self, string):
        if 'PN_cluster_1' in string or 'PN_cluster_2' in string or 'PN_cluster_3' in string:
            classification = 'Cluster'
            class_name = string[string.find("cluster"):]
            class_name = class_name[8]
        elif 'PN_cluster' in string:
            classification = "Glomerulus"
            class_name = string[string.find("cluster"):]
            class_name = class_name[class_name.find("_") + 1:]
            class_name = class_name[:class_name.find("_")]
        elif "KCab-" in string or "KCa'b'-" in string or "KCg-" in string:
            classification = 'minor'
            class_name = string[string.find("KC"):]
            class_name = class_name[:class_name.find("_")]
        elif 'KCab' in string or "KCa'b'" in string or "KCg" in string:
            classification = 'major'
            class_name = string[string.find("KC"):]
            if class_name.find("_") != -1:
                class_name = class_name[:class_name.find("_")]
        return [classification, class_name]

    def get_condition_KC_related(self, string):
        KC_related = ""
        if 'to_KC' in string or "from_KC" in string:
            KC_related = "_to_KC"
        if "from_KC" in string:
            KC_related = "_from_KC"
        return KC_related

    def summary_data_rename(self, df):
        Conditions = df['Condition'].values.tolist()
        summary = [[], [], []]
        for condition_index, condition in enumerate(Conditions):
            neuropil = self.get_condition_neuropil(condition)
            c1, c2 = condition.split(",")
            for c_index, c in enumerate([c1, c2]):
                KC_related = self.get_condition_KC_related(c)
                level = self.get_condition_level(c)
                classification, class_name = self.get_condition_classification(c)
                summary[c_index].append(f"{level}_{classification}_{class_name}{KC_related}")
            summary[2].append(neuropil)
            print(condition, summary[0][-1], summary[1][-1], summary[2][-1])
        return summary

    def analyze_spatial_distribution(self, t1='Glomerulus', t2='Glomerulus', d1='neuron', d2='neuron',
                                     neuropil='CA(R)', synapse_related_to_KC_1=False, synapse_related_to_KC_2=False):
        '''
        t1 compared with t2 and shuffled t2.
        :param t1: Glomerulus, Cluster, major class, minor class
        :param t2: Glomerulus, Cluster, major class, minor class
        :param d1: neuron, synapse
        :param d2: neuron, synapse
        :param neuropil: CA(R)
        Condition	FlyEM	Shuffled mean	Shuffled STD	z score

        :return:
        '''

        df = pd.read_csv(f"JS_analysis_summary_0711.csv")
        df['condition 1'], df['condition 2'], df['neuropil'] = self.summary_data_rename(df)
        mask1 = df['condition 1'].str.contains(t1)
        mask2 = df['condition 2'].str.contains(t2)
        mask3 = df['condition 1'].str.contains(d1)
        mask4 = df['condition 2'].str.contains(d2)
        mask5 = df['neuropil'] == neuropil
        mask6 = df['condition 1'].str.contains('to_KC')
        mask7 = df['condition 2'].str.contains('to_KC')
        if synapse_related_to_KC_1 and synapse_related_to_KC_2:
            filtered_df = df[mask1 & mask2 & mask3 & mask4 & mask5 & mask6 & mask7].values.tolist()
        elif synapse_related_to_KC_1:
            filtered_df = df[mask1 & mask2 & mask3 & mask4 & mask5 & mask6 & ~mask7].values.tolist()
        elif synapse_related_to_KC_2:
            filtered_df = df[mask1 & mask2 & mask3 & mask4 & mask5 & ~mask6 & mask7].values.tolist()
        else:
            filtered_df = df[mask1 & mask2 & mask3 & mask4 & mask5 & ~mask6 & ~mask7].values.tolist()

        Classification_dict = {"Glomerulus": self.network.G_list, 'Cluster': [1, 2, 3],
                               'minor': list(self.network.New_subtype_to_id.keys()), 'major': ['KCg', "KCa'b'", "KCab"]}
        real_matrix = np.zeros((len(Classification_dict[t1]), len(Classification_dict[t2])))
        shuffled_matrix = np.zeros(real_matrix.shape)
        z_score_matrix = np.zeros(real_matrix.shape)
        for i in range(len(filtered_df)):
            c1 = filtered_df[i][-3].split("_")[-1]
            c2 = filtered_df[i][-2].split("_")[-1]
            index_1 = Classification_dict[t1].index(c1)
            index_2 = Classification_dict[t2].index(c2)
            real_matrix[index_1][index_2] = filtered_df[i][2]
            shuffled_matrix[index_1][index_2] = filtered_df[i][3]
            z_score_matrix[index_1][index_2] = (filtered_df[i][2] - filtered_df[i][3]) / filtered_df[i][4]

        row_c = []
        for G in Classification_dict[t1]:
            row_c.append(self.Color_dict[G])

        col_c = []
        for G in Classification_dict[t2]:
            col_c.append(self.Color_dict[G])

        g = sns.clustermap(data=real_matrix, xticklabels=Classification_dict[t2],
                           yticklabels=Classification_dict[t1],
                           col_colors=col_c, row_colors=row_c, cmap='bwr', method='complete',
                           )
        g.ax_row_dendrogram.set_visible(False)
        g.ax_col_dendrogram.set_visible(False)
        plt.show()

        g = sns.clustermap(data=z_score_matrix, xticklabels=Classification_dict[t2],
                           yticklabels=Classification_dict[t1],
                           col_colors=col_c, row_colors=row_c, cmap='bwr', method='complete',
                           vmin=-2, vmax=2)
        g.ax_row_dendrogram.set_visible(False)
        g.ax_col_dendrogram.set_visible(False)
        plt.show()

    123

    def analyze_spatial_distribution_bar(self, t1='Glomerulus', t2='Glomerulus', d1='neuron', d2='neuron',
                                         neuropil='CA(R)', datatype='JS',synapse_related_to_KC_1=False,synapse_related_to_KC_2=False,
                                         JS_clustermap=False):
        '''
        t1 compared with t2 and shuffled t2.
        :param t1: Glomerulus, Cluster, major class, minor class
        :param t2: Glomerulus, Cluster, major class, minor class
        :param d1: neuron, synapse
        :param d2: neuron, synapse
        :param neuropil: CA(R)f
        Condition	FlyEM	Shuffled mean	Shuffled STD	z score

        :return:
        '''
        path = f"{self.result_fig2}{t1}_{t2}_{neuropil}/"
        if not os.path.isdir(path):
            os.mkdir(path)
        if synapse_related_to_KC_1:
            KC1 = 'T'
        else:
            KC1 = 'F'
        if synapse_related_to_KC_2:
            KC2 = 'T'
        else:
            KC2 = 'F'
        file_name = f'{t1}_{d1}_{t2}_{d2}_{KC1}_{KC2}_{neuropil}'
        if os.path.isfile('JS_analysis_summary_rename.csv'):
            df = pd.read_csv('JS_analysis_summary_rename.csv')
            df.drop('Unnamed: 0', axis=1, inplace=True)
        else:
            df = pd.read_csv(f"JS_analysis_summary_0711.csv")
            df['condition 1'], df['condition 2'], df['neuropil'] = self.summary_data_rename(df)
            df.to_csv('JS_analysis_summary_rename.csv')
        mask00 = df['condition 1'].str.contains('KCg-s4')
        mask01 = df['condition 2'].str.contains('KCg-s4')
        mask1 = df['condition 1'].str.contains(t1)
        mask2 = df['condition 2'].str.contains(t2)
        mask3 = df['condition 1'].str.contains(d1)
        mask4 = df['condition 2'].str.contains(d2)
        mask5 = df['neuropil'] == neuropil
        mask6 = df['condition 1'].str.contains('to_KC')
        mask7 = df['condition 2'].str.contains('to_KC')
        if synapse_related_to_KC_1 and synapse_related_to_KC_2:
            filtered_df = df[mask1 & mask2 & mask3 & mask4 & mask5 & mask6 & mask7 & ~mask00 & ~mask01].values.tolist()
        elif synapse_related_to_KC_1:
            filtered_df = df[mask1 & mask2 & mask3 & mask4 & mask5 & mask6 & ~mask7 & ~mask00 & ~mask01].values.tolist()
        elif synapse_related_to_KC_2:
            filtered_df = df[mask1 & mask2 & mask3 & mask4 & mask5 & ~mask6 & mask7 & ~mask00 & ~mask01].values.tolist()
        else:
            filtered_df = df[mask1 & mask2 & mask3 & mask4 & mask5 & ~mask6 & ~mask7 & ~mask00 & ~mask01].values.tolist()
        # print(t1,t2,d1,d2,synapse_related_to_KC_1,synapse_related_to_KC_2, neuropil)
        # print(filtered_df[0:3])
        # input()
        # return
        Classification_dict = self.Classification_dict

        # Classification_dict['minor'].pop(Classification_dict['minor'].index("KCg-s4"))
        real_matrix = np.zeros((len(Classification_dict[t1]), len(Classification_dict[t2])))
        shuffled_matrix = np.zeros(real_matrix.shape)
        z_score_matrix = np.zeros(real_matrix.shape)
        for i in range(len(filtered_df)):
            # print("#####")
            print(filtered_df[i])
            c1 = filtered_df[i][-3]
            c2 = filtered_df[i][-2]
            if "_to_KC" in c1:
                c1 = c1[:c1.find("_to_KC")]
            elif "_from_KC" in c1:
                c1 = c1[:c1.find("_from_KC")]
            if "_to_KC" in c2:
                c2 = c2[:c2.find("_to_KC")]
            elif "_from_KC" in c2:
                c2 = c2[:c2.find("_from_KC")]
            c1 = c1.split("_")[-1]
            c2 = c2.split("_")[-1]
            print(c1, c2)
            if t1 == 'Cluster':
                c1 = int(c1)
            if t2 == 'Cluster':
                c2 = int(c2)
            if "KCg-t"in str(c1) or 'KCg-t'in str(c2):
                continue
            index_1 = Classification_dict[t1].index(c1)
            index_2 = Classification_dict[t2].index(c2)
            real_matrix[index_1][index_2] = filtered_df[i][2]
            shuffled_matrix[index_1][index_2] = filtered_df[i][3]
            z_score_matrix[index_1][index_2] = (filtered_df[i][2] - filtered_df[i][3]) / filtered_df[i][4]

        if JS_clustermap:
            color_1 = [self.Color_dict[i] for i in Classification_dict[t1]]
            color_2 = [self.Color_dict[i] for i in Classification_dict[t2]]
            candidate_list = []
            x_tick_label = []
            for G in self.network.G_list:
                if self.Glomerulus_to_Cluster[G] == 2:
                    candidate_list.append(False)
                else:
                    candidate_list.append(True)
                    x_tick_label.append(G)
            color_1 = [self.Color_dict[i] for i in x_tick_label]
            color_2 = [self.Color_dict[i] for i in x_tick_label]
            print(candidate_list)
            fil_data = real_matrix[candidate_list,:]
            fil_data = fil_data[:,candidate_list]
            print(fil_data)
            g=sns.clustermap(data=fil_data,method='single',xticklabels=x_tick_label,
                           yticklabels=x_tick_label, col_colors=color_2,row_colors=color_1,
                           vmax=np.percentile(fil_data, 90), vmin=np.percentile(fil_data, 10),
                             metric='correlation',
                           cmap='bwr')
            g.ax_row_dendrogram.set_visible(False)
            g.ax_col_dendrogram.set_visible(False)
            plt.show()

            # g=sns.clustermap(data=real_matrix,method='ward',xticklabels=Classification_dict[t1],
            #                yticklabels=Classification_dict[t2], col_colors=color_2,row_colors=color_1,
            #                  vmax=np.percentile(real_matrix, 90), vmin=np.percentile(real_matrix, 10),
            #                  cmap='bwr')
            # g.ax_row_dendrogram.set_visible(False)
            # g.ax_col_dendrogram.set_visible(False)
            # plt.show()
            #
            # g=sns.clustermap(data=real_matrix,method='complete',xticklabels=Classification_dict[t1],
            #                yticklabels=Classification_dict[t2], col_colors=color_2,row_colors=color_1,
            #                 vmax=np.percentile(real_matrix, 90), vmin=np.percentile(real_matrix, 10),
            #                  cmap='bwr')
            # g.ax_row_dendrogram.set_visible(False)
            # g.ax_col_dendrogram.set_visible(False)
            # plt.show()

            # g=sns.clustermap(data=real_matrix,method='single',xticklabels=Classification_dict[t1],
            #                yticklabels=Classification_dict[t2], col_colors=color_2,row_colors=color_1,
            #                vmax=np.percentile(real_matrix, 90), vmin=np.percentile(real_matrix, 10),
            #                cmap='bwr')
            # g.ax_row_dendrogram.set_visible(False)
            # g.ax_col_dendrogram.set_visible(False)
            # plt.show()

            # g=sns.clustermap(data=real_matrix,method='average',xticklabels=Classification_dict[t1],
            #                yticklabels=Classification_dict[t2], col_colors=color_2,row_colors=color_1,
            #                  vmax=np.percentile(real_matrix, 90), vmin=np.percentile(real_matrix, 10),
            #                  cmap='bwr')
            # g.ax_row_dendrogram.set_visible(False)
            # g.ax_col_dendrogram.set_visible(False)
            # plt.show()



        row_c = []
        for G in Classification_dict[t1]:
            row_c.append(self.Color_dict[G])

        col_c = []
        for G in Classification_dict[t2]:
            col_c.append(self.Color_dict[G])

        # for classification_index, classification in enumerate(Classification_dict[t1]):
        #
        #     plt.bar(x=[i for i in range(len(Classification_dict[t2]))], height=z_score_matrix[classification_index])
        #     plt.title(classification)
        #     plt.show()

        if t1 == 'major':
            order_list = [0, 1, 2]
        elif t2 == 'minor':
            classification_index = Classification_dict[t2].index('KCg-m')
            data = z_score_matrix[:, classification_index]
            order_list = (-1 * data).argsort()  ## descending order
            print(order_list)
        elif t2 == 'major':
            classification_index = Classification_dict[t2].index("KCg")
            data = z_score_matrix[:, classification_index]
            order_list = (-1 * data).argsort()  ## descending order
        else:
            order_list = [i for i in range(len(z_score_matrix[:, 0]))]

        row_num = len(Classification_dict[t2])
        if len(order_list) > 20:
            fig, axes = plt.subplots(nrows=row_num, ncols=1, figsize=(12, 2 * row_num))
            # plt.figure(figsize=(9.5, 1.85))
        elif t1 == 'minor':
            fig, axes = plt.subplots(nrows=row_num, ncols=1, figsize=(8, 1.5 * row_num))
            # plt.figure(figsize=(6, 2))
        else:
            # plt.figure(figsize=(4, 1))
            fig, axes = plt.subplots(nrows=row_num, ncols=1, figsize=(5, 1.5 * row_num))
        print(len(axes), row_num)
        for classification_index, classification in enumerate(Classification_dict[t2]):
            ax = axes[classification_index]
            plt.sca(ax)
            height = z_score_matrix[:, classification_index]
            ax.bar(x=[i for i in range(len(Classification_dict[t1]))], height=height[order_list], color='k')
            xticklabel = np.array(Classification_dict[t1])
            xticklabel = xticklabel[order_list]
            # Df(data = [[xticklabel[ii], height[ii]] for ii in range(len(xticklabel))]).to_csv(f'{path}{file_name}_{classification}.csv')
            if t1 == 'major' or t1 == 'minor':
                xticklabel = [self.rename_KC_subtype(xticklabel[i]) for i in range(len(xticklabel))]
            if len(xticklabel) > 8:
                rotation = 90
            else:
                rotation = 0
            if classification_index == row_num - 1:
                if t1 == 'major' or t1 == "Cluster":
                    fontsize = self.fontdict['label']['fontsize']
                else:
                    fontsize = self.fontdict['tick']['fontsize']
                plt.xticks(ticks=[i for i in range(len(Classification_dict[t1]))], labels=xticklabel, rotation=rotation,
                           fontsize=fontsize)
                for x_tick_index, xtick in enumerate(ax.get_xticklabels()):
                    xtick.set_color(self.Color_dict[Classification_dict[t1][order_list[x_tick_index]]])
            else:
                plt.xticks(ticks=[])
            if isinstance(classification, str):
                if "KC" in classification:
                    ax.set_ylabel(self.rename_KC_subtype(classification), rotation=0, fontdict=self.fontdict['label'])
                else:
                    ax.set_ylabel(classification, rotation=0, fontdict=self.fontdict['label'])
            else:
                ax.set_ylabel(classification, rotation=0, fontdict=self.fontdict['label'])
            y_ticks = list(ax.get_yticks()) + [-3, 3]
            final_yticks = [min(y_ticks), max(y_ticks)]
            final_yticks = sorted(final_yticks)
            plt.yticks(final_yticks, fontsize=self.fontdict['tick']['fontsize'])
            ax.axhline(y=2, color='gray', linestyle='--')
            ax.axhline(y=-2, color='gray', linestyle='--')
            plt.tight_layout()

        plt.savefig(f"{path}{file_name}_summary.svg", dpi=500, format='svg')
        plt.savefig(f"{path}{file_name}_summary.png", dpi=500, format='png')
        plt.show()
        plt.close()
        data = Df(data=z_score_matrix, index=Classification_dict[t1], columns=Classification_dict[t2])
        data.to_excel(f"{path}{file_name}_zscore_summary.xlsx")
        data = Df(data=real_matrix, index=Classification_dict[t1], columns=Classification_dict[t2])
        data.to_excel(f"{path}{file_name}_JS_similarity_summary.xlsx")
        data = Df(data=shuffled_matrix, index=Classification_dict[t1], columns=Classification_dict[t2])
        data.to_excel(f"{path}{file_name}_JS_similarity_shuffled_summary.xlsx")



    # def get_spatial_spatial_correlation(self):
    #     path_1 = f"{self.result_fig2}Glomerulus_Cluster_CA(R)/"
    #     f1 = 'Glomerulus_synapse_Cluster_synapse_F_F_CA(R)_zscore_summary.xlsx'
    #     path_2 = f"{self.result_fig2}Glomerulus_Cluster_LH(R)/"
    #     f2 = 'Glomerulus_synapse_Cluster_synapse_F_F_LH(R)_zscore_summary.xlsx'
    #     file_name = f'{t1}_{d1}_{t2}_{d2}_{KC1}_{KC2}_{neuropil}'
    #     df = pd.read_csv(f"JS_analysis_summary_0711.csv")
    #     df['condition 1'], df['condition 2'], df['neuropil'] = self.summary_data_rename(df)
    #     df.to_csv('JS_analysis_summary_rename.csv')
    #     mask00 = df['condition 1'].str.contains('KCg-s4')
    #     mask01 = df['condition 2'].str.contains('KCg-s4')
    #     mask1 = df['condition 1'].str.contains(t1)
    #     mask2 = df['condition 2'].str.contains(t2)
    #     mask3 = df['condition 1'].str.contains(d1)
    #     mask4 = df['condition 2'].str.contains(d2)
    #     mask5 = df['neuropil'] == neuropil_1
    #     mask6 = df['condition 1'].str.contains('to_KC')
    #     mask7 = df['condition 2'].str.contains('to_KC')
    #     if synapse_related_to_KC_1 and synapse_related_to_KC_2:
    #         filtered_df = df[mask1 & mask2 & mask3 & mask4 & mask5 & mask6 & mask7 & ~mask00 & ~mask01].values.tolist()
    #     elif synapse_related_to_KC_1:
    #         filtered_df = df[mask1 & mask2 & mask3 & mask4 & mask5 & mask6 & ~mask7 & ~mask00 & ~mask01].values.tolist()
    #     elif synapse_related_to_KC_2:
    #         filtered_df = df[mask1 & mask2 & mask3 & mask4 & mask5 & ~mask6 & mask7 & ~mask00 & ~mask01].values.tolist()
    #     else:
    #         filtered_df = df[
    #             mask1 & mask2 & mask3 & mask4 & mask5 & ~mask6 & ~mask7 & ~mask00 & ~mask01].values.tolist()
    #
    #     Classification_dict = self.Classification_dict
    #     real_matrix = np.zeros((len(Classification_dict[t1]), len(Classification_dict[t2])))
    #     shuffled_matrix = np.zeros(real_matrix.shape)
    #     z_score_matrix = np.zeros(real_matrix.shape)
    #     for i in range(len(filtered_df)):
    #         c1 = filtered_df[i][-3].split("_")[-1]
    #         c2 = filtered_df[i][-2].split("_")[-1]
    #         if t1 == 'Cluster':
    #             c1 = int(c1)
    #         if t2 == 'Cluster':
    #             c2 = int(c2)
    #         index_1 = Classification_dict[t1].index(c1)
    #         index_2 = Classification_dict[t2].index(c2)
    #         real_matrix[index_1][index_2] = filtered_df[i][2]
    #         shuffled_matrix[index_1][index_2] = filtered_df[i][3]
    #         z_score_matrix[index_1][index_2] = (filtered_df[i][2] - filtered_df[i][3]) / filtered_df[i][4]
    #
    #     if t2 == 'minor':
    #         classification_index = Classification_dict[t2].index('KCg-m')
    #     elif t2 == 'major':
    #         classification_index = Classification_dict[t2].index("KCg")
    #     123
    #     data = []
    #     for G_index, G in enumerate(Classification_dict[t1]):
    #         data.append([z_score_matrix[G_index, classification_index], self.cellular_connection_pref_dict[G],
    #                      self.synaptic_connection_pref_dict[G]])
    #     data = np.array(data)
    #     slope, intercept, r_value, p_value, std_err = linregress(data[:, 0], data[:, 1])
    #     y_pred = intercept + slope * data[:, 0]
    #     plt.scatter(data[:, 0], data[:, 1])
    #     plt.plot(data[:, 0], y_pred, 'k-')
    #     r_squared = r_value ** 2
    #     r_squared_text = f'r\u00b2 = {r_squared:.3f}'
    #     if p_value < 0.001:
    #         p_value_text = f'p < 0.001'
    #     else:
    #         p_value_text = f'p = {p_value:.3f}'
    #     plt.text(0.05, 0.9, r_squared_text, transform=plt.gca().transAxes, fontsize=14)
    #     plt.text(0.05, 0.85, p_value_text, transform=plt.gca().transAxes, fontsize=14)
    #     plt.xticks(fontsize=18)
    #     plt.yticks(fontsize=18)
    #     plt.xlabel("Z score in spatial pref.", fontdict={'fontsize': 22})
    #     plt.ylabel("Z score in connection pref.", fontdict={'fontsize': 22})
    #     plt.tight_layout()
    #     plt.savefig(f"{self.result_fig2}synapse_spatial_connection_cellular_relationship.png", dpi=500,
    #                 transparent=True)
    #     plt.close()


    def analyze_spatial_distribution_connection_preference(self, t1='Glomerulus', t2='major', d1='synapse',
                                                           d2='synapse', synapse_related_to_KC_1=True,synapse_related_to_KC_2=False,
                                                           neuropil='CA(R)'):
        '''
        t1 compared with t2 and shuffled t2.
        :param t1: Glomerulus, Cluster, major class, minor class
        :param t2: Glomerulus, Cluster, major class, minor class
        :param d1: neuron, synapse
        :param d2: neuron, synapse
        :param neuropil: CA(R)
        Condition	FlyEM	Shuffled mean	Shuffled STD	z score

        :return:
        '''
        ############### 須注意~!!!!!!!!!!!!!!!!!!!!!!!!!!!!還需要再加入考慮有沒有KC這個字眼XD
        file_name = f'{t1}_{d1}_{t2}_{d2}_{neuropil}'
        df = pd.read_csv(f"JS_analysis_summary_0711.csv")
        df['condition 1'], df['condition 2'], df['neuropil'] = self.summary_data_rename(df)
        mask00 = df['condition 1'].str.contains('KCg-s4')
        mask01 = df['condition 2'].str.contains('KCg-s4')
        mask1 = df['condition 1'].str.contains(t1)
        mask2 = df['condition 2'].str.contains(t2)
        mask3 = df['condition 1'].str.contains(d1)
        mask4 = df['condition 2'].str.contains(d2)
        mask5 = df['neuropil'] == neuropil
        mask6 = df['condition 1'].str.contains('to_KC')
        mask7 = df['condition 2'].str.contains('to_KC')
        if synapse_related_to_KC_1 and synapse_related_to_KC_2:
            filtered_df = df[mask1 & mask2 & mask3 & mask4 & mask5 & mask6 & mask7 & ~mask00 & ~mask01]
        elif synapse_related_to_KC_1:
            filtered_df = df[mask1 & mask2 & mask3 & mask4 & mask5 & mask6 & ~mask7 & ~mask00 & ~mask01]
        elif synapse_related_to_KC_2:
            filtered_df = df[mask1 & mask2 & mask3 & mask4 & mask5 & ~mask6 & mask7 & ~mask00 & ~mask01]
        else:
            filtered_df = df[mask1 & mask2 & mask3 & mask4 & mask5 & ~mask6 & ~mask7 & ~mask00 & ~mask01]
        # filtered_df = df[mask1 & mask2 & mask3 & mask4 & mask5]
        filtered_df.to_excel("check.xlsx")
        filtered_df = filtered_df.values.tolist()
        Classification_dict = {"Glomerulus": self.network.G_list, 'Cluster': [1, 2, 3],
                               'minor': list(self.network.New_subtype_to_id.keys()), 'major': ['KCg', "KCa'b'", "KCab"]}
        real_matrix = np.zeros((len(Classification_dict[t1]), len(Classification_dict[t2])))
        shuffled_matrix = np.zeros(real_matrix.shape)
        z_score_matrix = np.zeros(real_matrix.shape)
        record_tmp = []
        for i in range(len(filtered_df)):
            c1 = filtered_df[i][-3]
            c2 = filtered_df[i][-2]
            if '_to_KC' in c1:
                c1 = c1[:c1.find("_to_KC")]
            elif '_from_PN' in c1:
                c1 = c1[:c1.find("_from_PN")]
            if '_to_KC' in c2:
                c2 = c2[:c2.find("_to_KC")]
            elif '_from_PN' in c2:
                c2 = c2[:c2.find("_from_PN")]
            c1 = c1.split("_")[-1]
            c2 = c2.split("_")[-1]
            if t1 == 'Cluster':
                c1 = int(c1)
            if t2 == 'Cluster':
                c2 = int(c2)
            index_1 = Classification_dict[t1].index(c1)
            index_2 = Classification_dict[t2].index(c2)
            real_matrix[index_1][index_2] = filtered_df[i][2]
            shuffled_matrix[index_1][index_2] = filtered_df[i][3]
            z_score_matrix[index_1][index_2] = (filtered_df[i][2] - filtered_df[i][3]) / filtered_df[i][4]
        if t2 == 'minor':
            classification_index = Classification_dict[t2].index('KCg-m')
        elif t2 == 'major':
            classification_index = Classification_dict[t2].index("KCg")
        data = []
        for G_index, G in enumerate(Classification_dict[t1]):
            data.append([z_score_matrix[G_index, classification_index], self.cellular_connection_pref_dict[G],
                         self.synaptic_connection_pref_dict[G]])
        data = np.array(data)
        Df(data=data, columns=['spatial preference','cellular connection preference', 'synaptic connection preference'],
           index=Classification_dict[t1]).to_excel('connection_spatial_tmp.xlsx')
        slope, intercept, r_value, p_value, std_err = linregress(data[:, 0], data[:, 1])
        y_pred = intercept + slope * data[:, 0]
        plt.scatter(data[:, 0], data[:, 1])
        plt.plot(data[:, 0], y_pred, 'k-')
        r_squared = r_value ** 2
        r_squared_text = f'r\u00b2 = {r_squared:.3f}'
        if p_value < 0.001:
            p_value_text = f'p < 0.001'
        else:
            p_value_text = f'p = {p_value:.3f}'
        plt.text(0.05, 0.9, r_squared_text, transform=plt.gca().transAxes, fontsize=14)
        plt.text(0.05, 0.85, p_value_text, transform=plt.gca().transAxes, fontsize=14)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel("Z score in spatial pref.", fontdict={'fontsize': 22})
        plt.ylabel("Z score in connection pref.", fontdict={'fontsize': 22})
        plt.tight_layout()
        plt.savefig(f"{self.result_fig2}synapse_spatial_connection_cellular_relationship.png", dpi=500,
                    transparent=True)
        plt.savefig(f"{self.result_fig2}synapse_spatial_connection_cellular_relationship.svg", format='svg')
        plt.show()
        plt.close()
        #
        # for classification_index, classification in enumerate(Classification_dict[t2]):
        #     if len(order_list) > 20:
        #         plt.figure(figsize=(9.5, 1.85))
        #     else:
        #         plt.figure(figsize=(4, 1))
        #     ax = plt.subplot(111)
        #     height = z_score_matrix[:, classification_index]
        #     ax.bar(x=[i for i in range(len(Classification_dict[t1]))], height=height[order_list], color='k')
        #     xticklabel = np.array(Classification_dict[t1])
        #     xticklabel = xticklabel[order_list]
        #     if t1 == 'major' or t1 == 'minor':
        #         xticklabel = [self.rename_KC_subtype(xticklabel[i]) for i in range(len(xticklabel))]
        #     if len(xticklabel) > 8:
        #         rotation = 90
        #     else:
        #         rotation = 0
        #     plt.xticks(ticks=[i for i in range(len(Classification_dict[t1]))], labels=xticklabel, rotation=rotation)
        #     if isinstance(classification, str):
        #         if "KC" in classification:
        #             plt.ylabel(self.rename_KC_subtype(classification), rotation=0)
        #         else:
        #             plt.ylabel(classification, rotation=0)
        #     else:
        #         plt.ylabel(classification, rotation=0)
        #     for x_tick_index, xtick in enumerate(ax.get_xticklabels()):
        #         xtick.set_color(self.Color_dict[Classification_dict[t1][order_list[x_tick_index]]])
        #     plt.tight_layout()
        #     plt.savefig(f"{self.result_fig2}{file_name}{classification}.png", dpi=500)
        #     plt.close()

    def check_condition(self, data, condition_list):
        for condition in condition_list:
            if condition[0] == True:
                if condition[1] not in data:
                    return False
            if condition[0] == False:
                if condition[1] in data:
                    return False
        return True

    def analyze_G_G_spatial(self, data_type='neuron'):
        df = pd.read_csv(f"{self.result_fig2}JS_similarity.csv")
        data = df.values.tolist()
        condition_1 = [True, 'FlyEM']
        condition_2 = [True, 'neuron']
        condition_3 = [True, 'PN']
        condition_4 = [True, 'LH(R)']
        condition_list = [condition_1, condition_2, condition_3, condition_4]
        filtered_data = [i[1:] for i in data if
                         self.check_condition(i[1], condition_list) & self.check_condition(i[2], condition_list)]
        G_num = len(self.Glomerulus_to_Cluster)
        glomerulus_glomerulus_matrix = np.zeros((G_num, G_num))
        col_colors = {}
        c = []
        for G in self.network.G_list:
            col_colors[G] = self.Color_dict[self.Glomerulus_to_Cluster[G]]
            c.append(self.Color_dict[self.Glomerulus_to_Cluster[G]])
        for i in range(len(filtered_data)):
            print(filtered_data[i])
            G1 = filtered_data[i][0].split("_")[4]
            G2 = filtered_data[i][1].split("_")[4]
            if G1 not in self.network.G_list or G2 not in self.network.G_list:
                continue
            s = filtered_data[i][2]
            glomerulus_glomerulus_matrix[self.network.G_list.index(G1)][self.network.G_list.index(G2)] = s
            glomerulus_glomerulus_matrix[self.network.G_list.index(G2)][self.network.G_list.index(G1)] = s

        g = sns.clustermap(data=glomerulus_glomerulus_matrix, xticklabels=self.network.G_list,
                           yticklabels=self.network.G_list, col_colors=c, row_colors=c, cmap='bwr', method='complete', )
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
        g.ax_row_dendrogram.set_visible(False)
        g.ax_col_dendrogram.set_visible(False)
        plt.show()
        # plt.savefig(f"{self.result_fig2}clustermap_Glomerulus_neuron_CA(R)_complete.png", dpi=500)
        # plt.close()

    def get_connection_preference(self):
        data = pd.read_excel(
            f'{self.connection_raw_data_path}Preference_score_threshold_3_cellular.xlsx').values.tolist()
        self.cellular_connection_pref_dict = {}
        for i in range(len(data)):
            self.cellular_connection_pref_dict[data[i][0]] = data[i][1]
        data = pd.read_excel(
            f'{self.connection_raw_data_path}Preference_score_threshold_3_synaptic.xlsx').values.tolist()
        self.synaptic_connection_pref_dict = {}
        for i in range(len(data)):
            self.synaptic_connection_pref_dict[data[i][0]] = data[i][1]

    def analyze_weight_distribution(self, wiring_style='FlyEM'):
        connection_weight = copy.deepcopy(self.network.connection_matrix_collection_dict[wiring_style][0])
        Weight_dict = {}
        for col_id, i in enumerate(self.network.KCid_list):
            for row_id, j in enumerate(self.network.PNid_list):
                if (self.PNid_to_Cluster[j], self.KCid_to_Subtype[i]) not in Weight_dict:
                    Weight_dict[(self.PNid_to_Cluster[j], self.KCid_to_Subtype[i])] = []
                if connection_weight[row_id][col_id] > 0:
                    Weight_dict[(self.PNid_to_Cluster[j], self.KCid_to_Subtype[i])].append(
                        connection_weight[row_id][col_id])
        result = []
        for cluster, subtype in Weight_dict:
            for weight in Weight_dict[(cluster, subtype)]:
                result.append([cluster, subtype, weight])
        result = Df(data=result, columns=['Cluster', 'Class', 'Weight'])
        result.to_excel(f'Weight_distribution_{wiring_style}.xlsx')
        data = pg.anova(data=result, dv='Weight', between=['Cluster', 'Class'])
        print(data)
        print(data.values.tolist())
        mask1 = result['Cluster'] == 1
        mask2 = result['Cluster'] == 2
        mask3 = result['Cluster'] == 3
        data = pg.ttest(x=result[mask1]['Weight'], y=result[mask2]['Weight'])
        data.to_excel('Cluster 1 2.xlsx')
        data = pg.ttest(x=result[mask1]['Weight'], y=result[mask3]['Weight'])
        data.to_excel('Cluster 1 3.xlsx')
        data = pg.ttest(x=result[mask2]['Weight'], y=result[mask3]['Weight'])
        data.to_excel('Cluster 2 3.xlsx')

        mask1 = result['Class'] == 'KCg'
        mask2 = result['Class'] == "KCa'b'"
        mask3 = result['Class'] == "KCab"
        data = pg.ttest(x=result[mask1]['Weight'], y=result[mask2]['Weight'])
        data.to_excel('Class g apbp.xlsx')
        data = pg.ttest(x=result[mask1]['Weight'], y=result[mask3]['Weight'])
        data.to_excel('Class g ab.xlsx')
        data = pg.ttest(x=result[mask2]['Weight'], y=result[mask3]['Weight'])
        data.to_excel('Class apbp ab.xlsx')

        data = pg.pairwise_gameshowell(dv='Weight', between='Cluster', data=result)
        print(data)
        print(data.values.tolist())
        data = pg.pairwise_gameshowell(dv='Weight', between='Class', data=result)
        print(data)
        print(data.values.tolist())
        sns.violinplot(data=result, x='Class', y='Weight', hue='Cluster')
        plt.show()

    def get_claw_bouton_num(self):
        path = 'PN_KC_bouton_claw_information_20230618/'
        PNid_bouton_dict = {}
        KCid_claw_dict = {}
        for PNid in self.network.PNid_list:
            try:
                with open(f"{path}{PNid}_bouton.txt", 'rt')as ff:
                    count = 0
                    for line in ff:
                        if len(line) > 1:
                            count += 1
                    PNid_bouton_dict[PNid] = count
            except:
                PNid_bouton_dict[PNid] = 0
        pooled_g = []
        for PNid in PNid_bouton_dict:
            pooled_g.append([self.PNid_to_Cluster[PNid], self.PNid_to_Glomerulus[PNid], PNid, PNid_bouton_dict[PNid]])
        pooled_g = pd.DataFrame(data=pooled_g, columns=['Cluster', 'Glomerulus', 'neuronId', 'bouton number'])
        pooled_g.to_csv(f"{path}PN_bouton_summary.csv")

        for KCid in self.network.KCid_list:
            try:
                with open(f"{path}{KCid}_claw.txt", 'rt')as ff:
                    count = 0
                    for line in ff:
                        if len(line) > 1:
                            count += 1
                    KCid_claw_dict[KCid] = count
            except:
                KCid_claw_dict[KCid] = 0
        pooled_KC = []
        for KCid in KCid_claw_dict:
            pooled_KC.append(
                [self.KCid_to_Subtype[KCid], self.network.id_to_new_subtype[KCid], KCid, KCid_claw_dict[KCid]])
        pooled_KC = pd.DataFrame(data=pooled_KC, columns=['Class', 'Subclass', 'neuronId', 'claw number'])
        pooled_KC.to_csv(f"{path}KC_claw_summary.csv")
        sns.barplot(data=pooled_KC, y='claw number', x='Class')
        plt.show()
        sns.barplot(data=pooled_KC, y='claw number', x='Subclass')
        plt.show()
        sns.barplot(data=pooled_g, y='bouton number', x='Cluster')
        plt.show()
        sns.barplot(data=pooled_g, y='bouton number', x='Glomerulus')
        plt.show()

    def calcualte_connection_pref_score(self):
        KC_class_list = self.plot_order_list_dict['major']
        G_KC_matrix = np.zeros((len(self.network.G_list),len(KC_class_list)))

        for i, PNid in enumerate(self.network.PNid_list):
            for j, KCid in enumerate(self.network.KCid_list):
                if self.network.connection_matrix_collection_dict['FlyEM'][0][i][j]>0:
                    G_index = self.network.G_list.index(self.PNid_to_Glomerulus[PNid])
                    KC_class_index = KC_class_list.index(self.KCid_to_Subtype[KCid])
                    G_KC_matrix[G_index][KC_class_index] += 1
        FlyEM_matrix = copy.deepcopy(G_KC_matrix.ravel())
        random_collection = []
        for random_index in range(len(self.network.connection_matrix_collection_dict['Random network'])):
            G_KC_matrix = np.zeros((len(self.network.G_list), len(KC_class_list)))
            for i, PNid in enumerate(self.network.PNid_list):
                for j, KCid in enumerate(self.network.KCid_list):
                    if self.network.connection_matrix_collection_dict['Random network'][random_index][i][j] > 0:
                        G_index = self.network.G_list.index(self.PNid_to_Glomerulus[PNid])
                        KC_class_index = KC_class_list.index(self.KCid_to_Subtype[KCid])
                        G_KC_matrix[G_index][KC_class_index] += 1
            random_collection.append(copy.deepcopy(G_KC_matrix.ravel()))
        random_collection = np.array(random_collection)
        preference = ((FlyEM_matrix - np.mean(random_collection,axis=0))/np.std(random_collection,axis=0)).reshape(G_KC_matrix.shape)
        result = []
        for i, G in enumerate(self.network.G_list):
            for j, KC in enumerate(KC_class_list):
                result.append([G,KC,preference[i][j]])
        result = Df(data=result,columns=['Glomerulus','KC class', "z score"])
        result.to_csv("Connection_pref.csv")
        f, ax = plt.subplots()
        ax.bar(x=[i for i in range(len(self.network.G_list))], height=preference[:,0], color='k')
        xticklabel = np.array(self.network.G_list)
        plt.xticks(ticks=[i for i in range(len(self.network.G_list))], labels=xticklabel, rotation=90,
                   fontsize=self.fontdict['tick']['fontsize'])
        self.prepare_color_dict()
        for x_tick_index, xtick in enumerate(ax.get_xticklabels()):
            xtick.set_color(self.Color_dict[self.network.G_list[x_tick_index]])
        plt.axhline(y=2)
        plt.axhline(y=-2)
        plt.show()

    def evaluate_PN_connection_density_bias(self, t1='Cluster', t2='major'):
        path = f'{self.result_root}connection_related_figure/'
        if not os.path.isdir(path): os.mkdir(path)
        pooled_result_input_num = []
        pooled_result_output_num = []
        pooled_result_weight = []
        pooled_result_input_ratio = []
        pooled_result_output_ratio = []
        for connection_style in ['FlyEM', 'Random network', 'Random network fix KC']:
            # connection_sum = np.zeros((len(self.plot_order_list_dict[t1]), len(self.plot_order_list_dict[t2])))
            # weight_sum = np.zeros((len(self.plot_order_list_dict[t1]), len(self.plot_order_list_dict[t2])))
            print(t1, t2)
            result_input_num = []
            result_output_num = []
            result_weight = []
            result_input_ratio = []
            result_output_ratio = []
            for network_index, weight_matrix in enumerate(
                    self.network.connection_matrix_collection_dict[connection_style]):
                total_input_weight = np.sum(weight_matrix, axis=0)
                total_output_weight = np.sum(weight_matrix, axis=1)
                for classid_1, classification_1 in enumerate(self.Classification_dict[t1]):
                    weight_index = []
                    for PN_index, PNid in enumerate(self.network.PNid_list):
                        if self.id_to_classification_dict[t1][PNid] == classification_1:
                            weight_index.append(PN_index)
                    tmp_weight_matrix = weight_matrix[weight_index, :]
                    tmp_total_output_weight = total_output_weight[weight_index]
                    for classid_2, classification_2 in enumerate(self.Classification_dict[t2]):
                        weight_index = []
                        for KC_index, KCid in enumerate(self.network.KCid_list):
                            if self.id_to_classification_dict[t2][KCid] == classification_2:
                                weight_index.append(KC_index)
                        tmp_tmp_weight_matrix = tmp_weight_matrix[:, weight_index]
                        tmp_total_input_weight = total_input_weight[weight_index]
                        result_weight += [[network_index, classification_1, classification_2, i] for i in
                                          tmp_tmp_weight_matrix[tmp_tmp_weight_matrix > 0].tolist()]
                        result_input_num += [[network_index, classification_1, classification_2, i] for i in
                                             np.count_nonzero(tmp_tmp_weight_matrix, axis=0).tolist()]
                        result_output_num += [[network_index, classification_1, classification_2, i] for i in
                                              np.count_nonzero(tmp_tmp_weight_matrix, axis=1).tolist()]

                        pooled_result_weight += [[connection_style, network_index, classification_1, classification_2, i] for i
                                          in tmp_tmp_weight_matrix[tmp_tmp_weight_matrix > 0].tolist()]
                        pooled_result_input_num += [[connection_style, network_index, classification_1, classification_2, i]
                                             for i in np.count_nonzero(tmp_tmp_weight_matrix, axis=0).tolist()]
                        pooled_result_output_num += [[connection_style, network_index, classification_1, classification_2, i]
                                              for i in np.count_nonzero(tmp_tmp_weight_matrix, axis=1).tolist()]

                        input_weight_ratio = np.sum(tmp_tmp_weight_matrix, axis=0) / tmp_total_input_weight
                        output_weight_ratio = np.sum(tmp_tmp_weight_matrix, axis=1) / tmp_total_output_weight
                        result_input_ratio += [[network_index, classification_1, classification_2, i] for i in
                                               input_weight_ratio.tolist()]
                        result_output_ratio += [[network_index, classification_1, classification_2, i] for i in
                                                output_weight_ratio.tolist()]
                        pooled_result_input_ratio += [[connection_style, network_index, classification_1, classification_2, i]
                                               for i in input_weight_ratio.tolist()]
                        pooled_result_output_ratio += [[connection_style, network_index, classification_1, classification_2, i]
                                                for i in output_weight_ratio.tolist()]
            result_input_num = Df(data=result_input_num, columns=["network_index", t1, t2, 'Connection number'])
            result_input_num.to_csv(f"{path}{t1}_{t2}_{connection_style}_input_connection_number.csv")
            result_output_num = Df(data=result_output_num, columns=["network_index",t1, t2, 'Connection number'])
            result_output_num.to_csv(f"{path}{t1}_{t2}_{connection_style}_output_connection_number.csv")
            result_weight = Df(data=result_weight, columns=["network_index",t1, t2, 'Weight'])
            result_weight.to_csv(f"{path}{t1}_{t2}_{connection_style}_weight.csv")
            result_input_ratio = Df(data=result_input_ratio, columns=["network_index",t1, t2, 'Input weight ratio'])
            result_input_ratio.to_csv(f"{path}{t1}_{t2}_{connection_style}_input_weight_ratio.csv")
            result_output_ratio = Df(data=result_output_ratio, columns=["network_index",t1, t2, 'Output weight ratio'])
            result_output_ratio.to_csv(f"{path}{t1}_{t2}_{connection_style}_output_weight_ratio.csv")

            row_num = len(self.Classification_dict[t1])
            ## connection num
            fig, axes = plt.subplots(nrows=row_num, ncols=1, figsize=(5, 3 * row_num), sharex=True)
            for classification_index, classification in enumerate(self.Classification_dict[t1]):
                ax = axes[classification_index]
                plt.sca(ax)
                mask = result_input_num[t1] == classification
                sns.violinplot(data=result_input_num[mask], y=t2, x='Connection number', palette=self.fig_color_dict,
                               orient='h', cut=0, bw=0.5, ax=ax)
                plt.yticks(fontsize=self.fontdict['tick']['fontsize'])
                ytick_labels = ax.get_yticks().tolist()
                xtick_labels = ax.get_xticks().tolist()
                ax.set_yticklabels([self.rename_KC_subtype(i) for i in self.Classification_dict[t2]],
                                   fontsize=self.fontdict['tick']['fontsize'])
                plt.ylabel("")
                plt.xlabel("")
                ax.set_xticklabels([i for i in xtick_labels],
                                   fontsize=self.fontdict['tick']['fontsize'])
                ax.spines['bottom'].set_linewidth(1.5)  # X-axis
                ax.spines['left'].set_linewidth(1.5)  # Y-axis
                ax.spines['top'].set_linewidth(1.5)  # X-axis
                ax.spines['right'].set_linewidth(1.5)  # Y-axis
            plt.xlabel("Connection number", fontdict=self.fontdict['label'])
            plt.savefig(f'{path}{t1}_{t2}_{connection_style}_input_connection_number.svg', format='svg', dpi=500)
            plt.savefig(f'{path}{t1}_{t2}_{connection_style}_input_connection_number.png', format='png', dpi=500)
            plt.close()
            ## connection weight
            fig, axes = plt.subplots(nrows=row_num, ncols=1, figsize=(5, 3 * row_num), sharex=True)
            for classification_index, classification in enumerate(self.Classification_dict[t1]):
                ax = axes[classification_index]
                plt.sca(ax)
                mask = result_weight[t1] == classification
                sns.violinplot(data=result_weight[mask], y=t2, x='Weight', palette=self.fig_color_dict,
                               orient='h', cut=0, bw=0.5, ax=ax)
                plt.yticks(fontsize=self.fontdict['tick']['fontsize'])
                ytick_labels = ax.get_yticks().tolist()
                xtick_labels = ax.get_xticks().tolist()
                ax.set_yticklabels([self.rename_KC_subtype(i) for i in self.Classification_dict[t2]],
                                   fontsize=self.fontdict['tick']['fontsize'])
                plt.ylabel("")
                plt.xlabel("")
                ax.set_xticklabels([i for i in xtick_labels],
                                   fontsize=self.fontdict['tick']['fontsize'])
                ax.spines['bottom'].set_linewidth(1.5)  # X-axis
                ax.spines['left'].set_linewidth(1.5)  # Y-axis
                ax.spines['top'].set_linewidth(1.5)  # X-axis
                ax.spines['right'].set_linewidth(1.5)  # Y-axis
            plt.xlabel("Weight", fontdict=self.fontdict['label'])
            plt.savefig(f'{path}{t1}_{t2}_{connection_style}_weight.svg', format='svg', dpi=500)
            plt.savefig(f'{path}{t1}_{t2}_{connection_style}_weight.png', format='png', dpi=500)
            plt.close()
            ### input ratio
            fig, axes = plt.subplots(nrows=row_num, ncols=1, figsize=(5, 3 * row_num), sharex=True)
            for classification_index, classification in enumerate(self.Classification_dict[t1]):
                ax = axes[classification_index]
                plt.sca(ax)
                mask = result_input_ratio[t1] == classification
                sns.violinplot(data=result_input_ratio[mask], y=t2, x='Input weight ratio', palette=self.fig_color_dict,
                               orient='h', cut=0, bw=0.5, ax=ax)
                plt.yticks(fontsize=self.fontdict['tick']['fontsize'])
                ytick_labels = ax.get_yticks().tolist()
                xtick_labels = ax.get_xticks().tolist()
                ax.set_yticklabels([self.rename_KC_subtype(i) for i in self.Classification_dict[t2]],
                                   fontsize=self.fontdict['tick']['fontsize'])
                plt.ylabel("")
                plt.xlabel("")
                plt.xticks(ticks=[0.0, 0.25, 0.50, 0.75, 1.0], fontsize=self.fontdict['tick']['fontsize'])
                # ax.set_xticklabels([i for i in xtick_labels],
                #                    fontsize=self.fontdict['tick']['fontsize'])
                ax.spines['bottom'].set_linewidth(1.5)  # X-axis
                ax.spines['left'].set_linewidth(1.5)  # Y-axis
                ax.spines['top'].set_linewidth(1.5)  # X-axis
                ax.spines['right'].set_linewidth(1.5)  # Y-axis
            plt.xlabel('Input weight ratio', fontdict=self.fontdict['label'])
            plt.savefig(f'{path}{t1}_{t2}_{connection_style}_input weight ratio.svg', format='svg', dpi=500)
            plt.savefig(f'{path}{t1}_{t2}_{connection_style}_input weight ratio.png', format='png', dpi=500)
            plt.close()

            tt = t1
            t1 = t2
            t2 = tt

            row_num = len(self.Classification_dict[t1])
            fig, axes = plt.subplots(nrows=row_num, ncols=1, figsize=(5, 3 * row_num), sharex=True)
            for classification_index, classification in enumerate(self.Classification_dict[t1]):
                ax = axes[classification_index]
                plt.sca(ax)
                mask = result_output_num[t1] == classification
                sns.violinplot(data=result_output_num[mask], y=t2, x='Connection number', palette=self.fig_color_dict,
                               orient='h', cut=0, bw=0.5, ax=ax)
                plt.yticks(fontsize=self.fontdict['tick']['fontsize'])
                ytick_labels = ax.get_yticks().tolist()
                xtick_labels = ax.get_xticks().tolist()
                ax.set_yticklabels([i for i in self.Classification_dict[t2]],
                                   fontsize=self.fontdict['tick']['fontsize'])
                plt.ylabel("")
                plt.xlabel("")
                ax.set_xticklabels([i for i in xtick_labels],
                                   fontsize=self.fontdict['tick']['fontsize'])
                ax.spines['bottom'].set_linewidth(1.5)  # X-axis
                ax.spines['left'].set_linewidth(1.5)  # Y-axis
                ax.spines['top'].set_linewidth(1.5)  # X-axis
                ax.spines['right'].set_linewidth(1.5)  # Y-axis
            plt.xlabel("Connection number", fontdict=self.fontdict['label'])
            plt.savefig(f'{path}{t1}_{t2}_{connection_style}_output_connection_number.svg', format='svg', dpi=500)
            plt.savefig(f'{path}{t1}_{t2}_{connection_style}_output_connection_number.png', format='png', dpi=500)
            plt.close()

            ### output ratio
            fig, axes = plt.subplots(nrows=row_num, ncols=1, figsize=(5, 3 * row_num), sharex=True)
            for classification_index, classification in enumerate(self.Classification_dict[t1]):
                ax = axes[classification_index]
                plt.sca(ax)
                mask = result_output_ratio[t1] == classification
                sns.violinplot(data=result_output_ratio[mask], y=t2, x='Output weight ratio',
                               palette=self.fig_color_dict,
                               orient='h', cut=0, bw=0.5, ax=ax)
                plt.yticks(fontsize=self.fontdict['tick']['fontsize'])
                ytick_labels = ax.get_yticks().tolist()
                xtick_labels = ax.get_xticks().tolist()
                ax.set_yticklabels([i for i in self.Classification_dict[t2]],
                                   fontsize=self.fontdict['tick']['fontsize'])
                plt.ylabel("")
                plt.xlabel("")
                plt.xticks(ticks=[0.0, 0.25, 0.50, 0.75, 1.0], fontsize=self.fontdict['tick']['fontsize'])
                # ax.set_xticklabels([i for i in xtick_labels],
                #                    fontsize=self.fontdict['tick']['fontsize'])
                ax.spines['bottom'].set_linewidth(1.5)  # X-axis
                ax.spines['left'].set_linewidth(1.5)  # Y-axis
                ax.spines['top'].set_linewidth(1.5)  # X-axis
                ax.spines['right'].set_linewidth(1.5)  # Y-axis
            plt.xlabel('Output weight ratio', fontdict=self.fontdict['label'])
            plt.savefig(f'{path}{t1}_{t2}_{connection_style}_output weight ratio.svg', format='svg', dpi=500)
            plt.savefig(f'{path}{t1}_{t2}_{connection_style}_output weight ratio.png', format='png', dpi=500)
            plt.close()

            tt = t1
            t1 = t2
            t2 = tt
        pooled_result_input_num = Df(data=pooled_result_input_num, columns=['Connection style',"network_index", t1, t2, 'Connection number'])
        pooled_result_input_num.to_csv(f"{path}{t1}_{t2}_ALL_input_connection_number.csv")
        pooled_result_output_num = Df(data=pooled_result_output_num, columns=['Connection style',"network_index", t1, t2, 'Connection number'])
        pooled_result_output_num.to_csv(f"{path}{t1}_{t2}_ALL_output_connection_number.csv")
        pooled_result_weight = Df(data=pooled_result_weight, columns=['Connection style',"network_index", t1, t2, 'Weight'])
        pooled_result_weight.to_csv(f"{path}{t1}_{t2}_ALL_weight.csv")
        pooled_result_input_ratio = Df(data=pooled_result_input_ratio, columns=['Connection style',"network_index", t1, t2, 'Input weight ratio'])
        pooled_result_input_ratio.to_csv(f"{path}{t1}_{t2}_ALL_input_weight_ratio.csv")
        pooled_result_output_ratio = Df(data=pooled_result_output_ratio, columns=['Connection style',"network_index", t1, t2, 'Output weight ratio'])
        pooled_result_output_ratio.to_csv(f"{path}{t1}_{t2}_ALL_output_weight_ratio.csv")

        return


    def get_statistics_for_connection_pooled_individual_input_weight_ratio(self):
        path = 'connection_related_figure/'
        data = pd.read_csv(f'{self.result_root}{path}Cluster_major_ALL_input_weight_ratio.csv')
        connection_style_list = data['Connection style'].unique().tolist()
        cluster_list = data['Cluster'].unique().tolist()
        class_list = data['major'].unique().tolist()
        result = []
        for cluster in cluster_list:
            for classification in class_list:
                single_result = []
                for connection_style in connection_style_list:
                    mask_cluster = data['Cluster'] == cluster
                    mask_class = data['major'] == classification
                    mask_connection = data['Connection style'] == connection_style
                    tmp_data = data[mask_class & mask_cluster & mask_connection]
                    if connection_style == 'FlyEM':
                        single_result = [cluster,classification,tmp_data['Input weight ratio'].mean()]
                    else:
                        network_num = len(tmp_data['network_index'].unique().tolist())
                        mean_list = []
                        for network_id in range(network_num):
                            mask_id = tmp_data['network_index'] == network_id
                            mean_list.append(tmp_data[mask_id]['Input weight ratio'].mean())
                        mean = np.mean(mean_list)
                        std = np.std(mean_list)
                        single_result += [mean,std, (single_result[2]-mean)/std]
                result.append(single_result)
        result = Df(data=result,columns=['Cluster','Class','FlyEM_mean',f'{connection_style_list[1]}_mean',f'{connection_style_list[1]}_std',f'{connection_style_list[1]}_z_score'
                                         ,f'{connection_style_list[2]}_mean',f'{connection_style_list[2]}_std',f'{connection_style_list[2]}_z_score'])
        result.to_csv(f"{self.result_root}{path}Summary_pooled_individual_Input_weight_ratio.csv")

    def get_statistics_for_connection_pooled_input_weight_ratio(self):
        path = 'connection_related_figure/'
        data = pd.read_csv(f'{self.result_root}{path}Cluster_major_ALL_weight.csv')
        connection_style_list = data['Connection style'].unique().tolist()
        cluster_list = data['Cluster'].unique().tolist()
        class_list = data['major'].unique().tolist()
        result = []
        for cluster in cluster_list:
            for classification in class_list:
                single_result = []
                for connection_style in connection_style_list:
                    mask_cluster = data['Cluster'] == cluster
                    mask_class = data['major'] == classification
                    mask_connection = data['Connection style'] == connection_style
                    tmp_data = data[mask_class & mask_cluster & mask_connection]
                    if connection_style == 'FlyEM':
                        w_data = data[mask_connection & mask_class]
                        total_weight = w_data['Weight'].sum()
                        single_result = [cluster,classification,tmp_data['Weight'].sum()/total_weight]

                    else:
                        network_num = len(tmp_data['network_index'].unique().tolist())
                        mean_list = []
                        for network_id in range(network_num):
                            mask_id = tmp_data['network_index'] == network_id
                            w_mask_id = data['network_index'] == network_id
                            w_data = data[mask_connection & mask_class & w_mask_id]
                            total_weight = w_data['Weight'].sum()
                            mean_list.append(tmp_data[mask_id]['Weight'].sum()/total_weight)
                        mean = np.mean(mean_list)
                        std = np.std(mean_list)
                        single_result += [mean,std, (single_result[2]-mean)/std]
                result.append(single_result)
        result = Df(data=result,columns=['Cluster','Class','FlyEM_mean',f'{connection_style_list[1]}_mean',f'{connection_style_list[1]}_std',f'{connection_style_list[1]}_z_score'
                                         ,f'{connection_style_list[2]}_mean',f'{connection_style_list[2]}_std',f'{connection_style_list[2]}_z_score'])
        result.to_csv(f"{self.result_root}{path}Summary_pooled_Input_weight_ratio.csv")

    def check_random_network_wegiht(self):
        weight_matrix_FlyEM = self.network.connection_matrix_collection_dict["FlyEM"][0]
        weight_matrix_shuffled = self.network.connection_matrix_collection_dict["Random network"][0]
        print(np.sum(weight_matrix_FlyEM, axis=0) - np.sum(weight_matrix_shuffled, axis=0))
        print(np.sum(weight_matrix_FlyEM, axis=1) - np.sum(weight_matrix_shuffled, axis=1))

    def plot_spatial_input_weight_ratio(self, fix = 'PN'):
        file = f'{self.result_root}connection_related_figure/Fig 2 weight spatial data.xlsx'
        data = pd.read_excel(file)
        if fix == 'PN':
            y = data['Input weight ratio difference between Random and FlyEM'].values.tolist()
            x = data['Spatial distribution difference between Random and FlyEM'].values.tolist()
        elif fix == 'KC':
            y = data['Input weight ratio difference between Random KC  fix and FlyEM'].values.tolist()
            x = data['Spatial distribution difference between Random KC fix and FlyEM'].values.tolist()
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        # Calculate R-squared value
        r_squared = r_value ** 2
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.scatter(x, y, color='blue')
        # Create regression line
        x_line = np.array([min(x), max(x)])
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, color='red',linewidth=2)
        # Add R-squared and p-value annotations
        plt.text(0.6, 0.2,  r"$r^2$ = {:.2f}".format(r_squared), transform=plt.gca().transAxes,fontdict={"fontsize":32})
        if p_value < 0.001:
            plt.text(0.6, 0.1, "p < 0.001", transform=plt.gca().transAxes,fontdict={"fontsize":32})
        elif p_value < 0.01:
            plt.text(0.6, 0.1, "p < 0.01", transform=plt.gca().transAxes,fontdict={"fontsize":32})
        elif p_value < 0.05:
            plt.text(0.6, 0.1, "p < 0.05", transform=plt.gca().transAxes,fontdict={"fontsize":32})
        else:
            plt.text(0.6, 0.1, "p > 0.05", transform=plt.gca().transAxes,fontdict={"fontsize":32})

        # Set plot labels and title
        if fix == 'KC':
            plt.xticks([-0.3, -0.2, -0.1], fontsize=32)
        elif fix == 'PN':
            plt.xticks([-0.1, 0, 0.1], fontsize=32)
        plt.yticks([-0.15,0,0.15],fontsize=32)
        plt.xlabel(r'$\Delta$Distribution Similarity',fontdict={"fontsize":40})
        plt.ylabel(r'$\Delta$Weight ratio',fontdict={"fontsize":40})
        plt.tight_layout()
        for w in [1.5]:
            ax.spines['bottom'].set_linewidth(w)  # X-axis
            ax.spines['left'].set_linewidth(w)  # Y-axis
            ax.spines['top'].set_linewidth(w)  # X-axis
            ax.spines['right'].set_linewidth(w)  # Y-axis
            plt.savefig(f'Fig 2 Spatial-Weight_fix_{fix}_w_{w}.png',dpi=500)
            plt.savefig(f'Fig 2 Spatial-Weight_fix_{fix}_w_{w}.svg',format='svg')
            plt.show()
        plt.close()

    def plot_spatial_similarity(self):
        data = pd.read_excel(f"{self.result_root}connection_related_figure/Spatial_distribution_synapse_summary.xlsx")
        random_mean_list = []
        random_std_list = []
        real_mean_list = []
        label_list = []
        for KC_class in self.Classification_dict['major']:
            for PN_cluster in self.Cluster_to_PNid:
                mask_1 = data["Class"] == KC_class
                mask_2 = data['Cluster'] == PN_cluster
                tmp_data = data[mask_1 & mask_2]
                random_height = tmp_data['Random network_mean'].values.tolist()[0]
                random_std = tmp_data['Random network_std'].values.tolist()[0]
                real_height = tmp_data['FlyEM_mean'].values.tolist()[0]
                real_mean_list.append(real_height)
                random_mean_list.append(random_height)
                random_std_list.append(random_std)
                label_list.append(f"{PN_cluster}")
            real_mean_list.append(0)
            random_mean_list.append(0)
            random_std_list.append(0)
            label_list.append("")
        real_mean_list = real_mean_list[:-1]
        random_mean_list = random_mean_list[:-1]
        random_std_list = random_std_list[:-1]
        label_list = label_list[:-1]
        fig, ax = plt.subplots()
        (_, caps, _) = ax.errorbar([i for i in range(len(real_mean_list))], random_mean_list, yerr=random_std_list, fmt='k.',
                                   capsize=10, elinewidth=3)

        for cap in caps:
            cap.set_color('black')
        ax.set_xticks([i for i in range(len(real_mean_list))])
        ax.set_xticklabels(label_list, fontsize=24)
        ax.set_yticks([0.3, 0.6,0.9])
        ax.set_yticklabels([0.3, 0.6,0.9], fontsize=24)

        plt.ylim((0.25, 0.95))
        ax.bar([i for i in range(len(real_mean_list))], real_mean_list,
               align='center',
               alpha=0.8,
               ecolor='black',
               # capsize=10,
               capsize=10,
               color=['r', 'gold', 'deepskyblue','black', 'r', 'gold', 'deepskyblue','black','r', 'gold', 'deepskyblue']
               )
        plt.xlabel("PN cluster", fontdict={'fontsize':30})
        plt.ylabel("Distribution similarity", fontdict={'fontsize':30})
        plt.tight_layout()
        w = 1.5
        ax.spines['bottom'].set_linewidth(w)  # X-axis
        ax.spines['left'].set_linewidth(w)  # Y-axis
        ax.spines['top'].set_linewidth(w)  # X-axis
        ax.spines['right'].set_linewidth(w)  # Y-axis
        plt.savefig(f"{self.result_fig2}Synapse_distribution_similarity_w_{w}.png",dpi=500)
        plt.savefig(f"{self.result_fig2}Synapse_distribution_similarity_w_{w}.svg",format='svg')
        plt.show()
        plt.close()

    def visualize_density_by_density(self, density, create_new=False, obj = 'LH(R)', cmap='Reds',contour_num=4):
        if create_new:
            mlab.clf()
            v = mlab.figure()
            v.scene.background = (1, 1, 1)
            mlab.options.backend = 'envisage'
        if obj:
            tvtk.OBJImporter()
            neuropil = trimesh.load(f'D:/eFlyPlotv2p1/Data/FlyEM_neuropil/{obj}.obj')
            x, y, z = neuropil.vertices.T
            # mesh = mlab.triangular_mesh(x, y, z, neuropil.faces, color=(0.9, 0.9, 0.9), opacity=0.5)
        x_num, y_num, z_num = density.shape
        print(x_num,y_num,z_num)
        xmin, xmax, ymin, ymax, zmin, zmax = self.get_bounding_box_of_neuropil(neuropil)
        xi, yi, zi = np.mgrid[xmin:xmax:complex(0, x_num),
                 ymin:ymax:complex(0, y_num),
                 zmin:zmax:complex(0, z_num)]
        contour = mlab.contour3d(xi, yi, zi, density, opacity=0.5, colormap=cmap, contours=contour_num)
        return contour

    def visualize_neuron(self,xyz, color):
        size = 100
        # Display the points with the specified size and color
        mlab.points3d(xyz[:,0],xyz[:,1],xyz[:,2], color=color, scale_factor=size)
        return

    def visualize_density(self, xyz, create_new=False, obj = ['MB(R)'],xyz_slice_num=[20,20,20],cmap='Reds',
                          contour_num=4, smooth=False):
        if create_new:
            mlab.clf()
            v = mlab.figure()
            v.scene.background = (1, 1, 1)
            mlab.options.backend = 'envisage'
        if obj:
            tvtk.OBJImporter()
            for neuropil_file in obj:
                neuropil = trimesh.load(f'D:/eFlyPlotv2p1/Data/FlyEM_neuropil/{neuropil_file}.obj')
                x, y, z = neuropil.vertices.T
                mesh = mlab.triangular_mesh(x, y, z, neuropil.faces, color=(0.9, 0.9, 0.9), opacity=0.5)
                # if smooth:
                #     mesh = mlab.pipeline.smooth(mesh)
        if len(self.xyzi) == 0:
            xmin, xmax, ymin, ymax, zmin, zmax = np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z)
            # Get the number of slices along each axis
            x_num, y_num, z_num = xyz_slice_num
            # Create a grid of points in 3D space
            xi, yi, zi = np.mgrid[xmin:xmax:complex(0, x_num),
                         ymin:ymax:complex(0, y_num),
                         zmin:zmax:complex(0, z_num)]
        else:
            xi, yi, zi = self.xyzi
        coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
        self.xyzi = [xi,yi,zi]
        # kde = stats.gaussian_kde(xyz.transpose())
        kde = stats.gaussian_kde(xyz,bw_method=0.1)
        density = kde(coords).reshape(xi.shape)
        contour = mlab.contour3d(xi, yi, zi, density, opacity=0.5, colormap=cmap, contours=contour_num)
        return contour

    def visualize_mlab(self, smooth=False):
        mlab.draw()
        mlab.show()

    def visualize_neuropil(self, obj = ['CA(R)', 'MB(R)'],create_new=False, smooth=False, color=(1,1,1),opacity=0.1):
        if create_new:
            mlab.clf()
            v = mlab.figure()
            v.scene.background = (1, 1, 1)
            mlab.options.backend = 'envisage'
        if obj:
            tvtk.OBJImporter()
            for neuropil_file in obj:
                neuropil = trimesh.load(f'{self.neuropil_path}{neuropil_file}.obj')
                x, y, z = neuropil.vertices.T
                mesh = mlab.triangular_mesh(x, y, z, neuropil.faces, color=color, opacity=opacity)
                # if smooth:
                #     mesh = mlab.pipeline.smooth(mesh)

    def get_statistics_for_KC_PN_synapse_fix_KC(self,xyz_max_number=30):
        PN_to_KC_weight_threshold = 3
        network = ConnectionSetting()
        network.PN_to_KC_weight_threshold = PN_to_KC_weight_threshold
        Glomerulus_to_Cluster, Cluster_to_Glomerulus, PNid_to_Cluster, Cluster_to_PNid, PNid_to_Glomerulus, Glomerulus_to_PNid, KCid_to_Subtype, Subtype_to_KCid = network.generate_simple_connection_table_weight()
        ### 從connection table的shuffle來做shuffle，也就是說A若連到B，若判定為KCg > PN1，shuffle是只將這個連結(以及所有這兩個神經的synapse)改變成其他屬性如KCab
        KC_PN_synapse_dict = {}
        KC_subtype_collection = ['KCg', "KCa'b'", "KCab"]

        Original_subtype_synapse_collection = defaultdict()
        original_coordinate = defaultdict()
        Count_for_subtype_connection = Counter()
        Count_for_cluster_connection = Counter()
        PN_cluster_coordinate = defaultdict()
        for KC_subtype in KC_subtype_collection:
            file_name = f"PN_to_{KC_subtype}_synapse.xlsx"
            data = pd.read_excel(f'{file_name}')
            xi = data['down_syn_coordinate_x'].to_numpy()
            yi = data['down_syn_coordinate_y'].to_numpy()
            zi = data['down_syn_coordinate_z'].to_numpy()
            xi_u = data['up_syn_coordinate_x'].to_numpy()
            yi_u = data['up_syn_coordinate_y'].to_numpy()
            zi_u = data['up_syn_coordinate_z'].to_numpy()
            PN_id_list = data['up.bodyId'].values.tolist()
            KC_id_list = data['down.bodyId'].values.tolist()
            for i in range(len(PN_id_list)):
                PN_id = PN_id_list[i]
                KC_id = KC_id_list[i]
                if PN_id not in PNid_to_Cluster or KC_id not in KCid_to_Subtype:
                    continue
                elif network.pre_to_post_weight[network.PNid_list.index(PN_id)][network.KCid_list.index(KC_id)] == 0:
                    continue
                coordinate = np.array([xi[i], yi[i], zi[i]])
                if (KC_id, PN_id) not in KC_PN_synapse_dict:
                    Count_for_subtype_connection[KC_subtype] += 1
                    # Count_for_cluster_connection[]
                    KC_PN_synapse_dict[(KC_id, PN_id)] = []
                KC_PN_synapse_dict[(KC_id, PN_id)] += [coordinate]
                if KC_subtype not in Original_subtype_synapse_collection:
                    Original_subtype_synapse_collection[KC_subtype] = []
                    original_coordinate[KC_subtype] = []
                Original_subtype_synapse_collection[KC_subtype] += [(KC_id, PN_id)]
                original_coordinate[KC_subtype] += [coordinate]
                cluster_id = PNid_to_Cluster[PN_id]
                if cluster_id not in PN_cluster_coordinate:
                    PN_cluster_coordinate[cluster_id] = []
                PN_coordinate = np.array([xi_u[i], yi_u[i], zi_u[i]])
                PN_cluster_coordinate[cluster_id] += [PN_coordinate]

        pooled_coordinate = []
        for KC_subtype in KC_subtype_collection:
            pooled_coordinate += original_coordinate[KC_subtype]
        for cluster_id in Cluster_to_PNid:
            pooled_coordinate += PN_cluster_coordinate[cluster_id]
        pooled_coordinate = np.array(pooled_coordinate).transpose()
        values = pooled_coordinate
        xmin, ymin, zmin = values.min(axis=1)
        xmax, ymax, zmax = values.max(axis=1)
        edge_length = max([xmax - xmin, ymax - ymin, zmax - zmin]) / xyz_max_number
        x_num, y_num, z_num = int((xmax - xmin) / edge_length), int((ymax - ymin) / edge_length), int(
            (zmax - zmin) / edge_length)
        print('edge length', edge_length)
        print('x_num,y_num,z_num', x_num, y_num, z_num)
        if x_num < 2:
            x_num = 2
        if y_num < 2:
            y_num = 2
        if z_num < 2:
            z_num = 2
        xi, yi, zi = np.mgrid[xmin:xmax:np.complex(0, x_num), ymin:ymax:np.complex(0, y_num),
                     zmin:zmax:np.complex(0, z_num)]
        # Evaluate the KDE on a regular grid...
        coords = np.vstack([item.ravel() for item in [xi, yi, zi]])

        original_record_data = []
        PN_density_collection = {}
        for cluster_id in PN_cluster_coordinate:
            kde = stats.gaussian_kde(np.array(PN_cluster_coordinate[cluster_id]).transpose())
            density = kde(coords).reshape(xi.shape)
            PN_density_collection[cluster_id] = density
        ############
        # input("STOP!!!!!!!!!!!!!!!!!!!!")
        ##############
        KC_density_collection = {}
        for subtype in KC_subtype_collection:
            kde = stats.gaussian_kde(np.array(original_coordinate[subtype]).transpose())
            density = kde(coords).reshape(xi.shape)
            KC_density_collection[subtype] = density
        for cluster_id in PN_cluster_coordinate:
            for subtype in KC_subtype_collection:
                dis = jensenshannon(np.ravel(PN_density_collection[cluster_id]),
                                    np.ravel(KC_density_collection[subtype]))
                print(cluster_id, subtype, dis)
                original_record_data.append([cluster_id, subtype, dis])

        original_record_data = Df(data=original_record_data, columns=['PN cluster', 'KC subtype', "JS divergence"])
        original_record_data.to_excel(f'Synapse_JS_divergence_weight_threshold_{3}_original.xlsx')

        ## Shuffle ##
        JS_distance_collection = []
        record_data = []
        for shuffle_id in range(100):
            print(shuffle_id)
            shuffled_coordinate = defaultdict()
            candidate_list = list(KC_PN_synapse_dict.keys())
            rd.shuffle(candidate_list)
            tmp = 0
            # for PN_cluster in Cluster_to_PNid:

            for KC_subtype in KC_subtype_collection:
                for i in range(tmp, tmp + Count_for_subtype_connection[KC_subtype]):
                    if KC_subtype not in shuffled_coordinate:
                        shuffled_coordinate[KC_subtype] = []
                    shuffled_coordinate[KC_subtype] += KC_PN_synapse_dict[candidate_list[i]]
                tmp = tmp + Count_for_subtype_connection[KC_subtype]
            KC_density_collection = {}
            for subtype in KC_subtype_collection:
                kde = stats.gaussian_kde(np.array(shuffled_coordinate[subtype]).transpose())
                density = kde(coords).reshape(xi.shape)
                KC_density_collection[subtype] = density
            for cluster_id in PN_cluster_coordinate:
                for subtype in KC_subtype_collection:
                    dis = jensenshannon(np.ravel(PN_density_collection[cluster_id]),
                                        np.ravel(KC_density_collection[subtype]))
                    record_data.append([shuffle_id, cluster_id, subtype, dis])
                    print(cluster_id, subtype, dis)
        record_data = Df(data=record_data, columns=['exp_id', 'PN cluster', 'KC subtype', "JS divergence"])
        record_data.to_excel(f'Synapse_JS_divergence_weight_threshold_{3}_shuffled.xlsx')
        return

    def get_synapse_location(self, path='', file='', up=True, mask_column="", mask_condition="", format='full'):
        if not file:
            path = 'D:/eFlyPlotv2p1/Output/Synapse_data/'
            file = "Synapse_MBON03s_downstream_of_KCa'b'-ap2_w_0_v1.2.1.xlsx"
        if file.find(".xlsx") != -1:
            data = pd.read_excel(f'{path}{file}')
            if mask_column and mask_condition:
                mask = data[mask_column] == mask_condition
                data = data[mask]
            if format == 'full':
                if up:
                    xi = data['up_syn_coordinate_x'].to_numpy()
                    yi = data['up_syn_coordinate_y'].to_numpy()
                    zi = data['up_syn_coordinate_z'].to_numpy()
                else:
                    xi = data['down_syn_coordinate_x'].to_numpy()
                    yi = data['down_syn_coordinate_y'].to_numpy()
                    zi = data['down_syn_coordinate_z'].to_numpy()
            elif format == 'xyz':
                xi = data['x'].to_numpy()
                yi = data['y'].to_numpy()
                zi = data['z'].to_numpy()
        elif file.find(".txt") != -1:
            xi, yi, zi = [], [], []
            with open(path + file, 'rt')as ff:
                for line in ff:
                    groups = line[:-1].split(" ")
                    xi.append(float(groups[0]))
                    yi.append(float(groups[1]))
                    zi.append(float(groups[2]))
        # values = np.array([xi, yi, zi])
        values = np.array([xi, yi, zi])
        return values