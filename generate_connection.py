import numpy as np
import random as rd
from matplotlib import pyplot as plt
from pandas import DataFrame
import seaborn as sn
import copy
from collections import defaultdict
import pandas as pd
from pandas import DataFrame as Df
import os
import pickle
from sklearn.decomposition import PCA
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.spatial import distance

plt.rcParams['font.family'] = 'Arial'


def cosine_similarity(mat):
    summation = np.sum(mat,axis=1)
    row_index = [i for i in range(summation.shape[0]) if summation[i] > 0]
    mat = mat[row_index]
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    normed_mat = mat / norm
    similarity = np.dot(normed_mat, normed_mat.T)
    return similarity

class ConnectionSetting:
    root = "hemibrain_data/"
    tmp_path = 'tmp_files/'
    PN_to_KC_weight_threshold = 3
    preference_type = 'cellular'
    PN_to_KC_dir = 'PN_to_KC_connection_data/'
    New_subtype_to_label = {"KCa'b'-ap1": 8, "KCa'b'-ap2": 9, "KCa'b'-m": 10, "KCab-c": 12, "KCab-m": 13,
                                 "KCab-p": 14, "KCab-s": 11, "KCg-d": 1, "KCg-m": 2, "KCg-s1": 3, "KCg-s2": 4,
                                 "KCg-s3": 5, "KCg-s4": 6, "KCg-t": 7}
    Subtype_to_label = {"KCg": 1, "KCa'b'": 2, "KCab": 3}
    if not os.path.isdir(tmp_path): os.mkdir(tmp_path)
    fontdict = {'label': {'fontsize': 20}, 'tick': {'fontsize': 16}}


    def __init__(self, PN_to_KC_weight_threshold=3,shuffle_ratio=1,fixed_ratio=1,preference_type='cellular',seed_num=101):
        self.PN_to_KC_weight_threshold = PN_to_KC_weight_threshold
        self.connection_matrix_collection_dict = {}
        self.connection_matrix_normalized_collection_dict = {}
        self.preference_type = preference_type
        rd.seed(seed_num)
        self.pre_number = 0
        self.post_number = 0
        self.pre_to_post_weight = []
        self.shuffle_ratio = shuffle_ratio
        self.fixed_ratio = fixed_ratio
        self.KC_connections_statistics = []
        self.KC_connection_weights_statistics = []
        self.PNid_list = []
        self.KCid_list = []
        self.pre_to_post_weight_g = []
        self.G_list = []
        self.pre_to_post_weight_norm = []
        self.pre_to_post_weight_g_norm = []
        self.New_subtype_to_id = defaultdict(list)
        self.All_new_subtype_to_id = defaultdict(list)
        self.id_to_new_subtype = {}
        self.KC_subtype_location = {}
        self.generate_simple_connection_table_weight()
        self.get_new_KC_subtype()
        self.KC_new_subtype_location = {}
        self.get_KC_subtype_location()
        self.KC_subtype_label = [1 for _ in range(len(self.Subtype_to_KCid["KCg"]))] + \
                       [2 for _ in range(len(self.Subtype_to_KCid["KCa'b'"]))] + \
                       [3 for _ in range(len(self.Subtype_to_KCid["KCab"]))]
        self.PN_cluster_label = [1 for _ in range(len(self.Cluster_to_PNid[1]))] + \
                       [2 for _ in range(len(self.Cluster_to_PNid[2]))] + \
                       [3 for _ in range(len(self.Cluster_to_PNid[3]))]

    @staticmethod
    def KC_from_subtype_to_main_type(KC_type) -> str:
        '''
        #### Finished
        This function get the KC main type from KC subtype
        :param KC_type:
        :return:
        '''
        if KC_type.find("KCg") != -1:
            KC_main_type = 'KCg'
        elif KC_type.find("KCab") != -1:
            KC_main_type = "KCab"
        elif KC_type.find("KCa'b'") != -1:
            KC_main_type = "KCa'b'"
        else:
            raise BaseException("Unexpected KC type!!!!!!!!!!!!!!")
        return KC_main_type

    # @classmethod
    def obtain_glomerulus_cluster(cls) -> (dict, dict):
        '''
        #### Finished
        This function obtain the pre-classified glomerulus and its cluster index.
        :return:
        '''
        data = pd.read_excel(f'{cls.root}Glomerulus_Cluster_threshold_{cls.PN_to_KC_weight_threshold}_{cls.preference_type}.xlsx')
        Cluster_to_Glomerulus = {}
        Glomerulus_to_Cluster = {}
        Cluster_to_Glomerulus[1] = [_ for _ in data['cluster1'].values.tolist() if isinstance(_, str)]
        Cluster_to_Glomerulus[2] = [_ for _ in data['cluster2'].values.tolist() if isinstance(_, str)]
        Cluster_to_Glomerulus[3] = [_ for _ in data['cluster3'].values.tolist() if isinstance(_, str)]
        G_list = []
        for i in range(1, 4):
            G_list += Cluster_to_Glomerulus[i]
        cls.G_list = G_list
        for Cluster_index in Cluster_to_Glomerulus:
            for Glomerulus in Cluster_to_Glomerulus[Cluster_index]:
                Glomerulus_to_Cluster[Glomerulus] = Cluster_index
        cls.Cluster_to_Glomerulus = Cluster_to_Glomerulus
        cls.Glomerulus_to_Cluster = Glomerulus_to_Cluster
        return Cluster_to_Glomerulus, Glomerulus_to_Cluster

    def check_PN_instances(self):
        Cluster_to_Glomerulus, Glomerulus_to_Cluster = self.obtain_glomerulus_cluster()
        data = pd.read_excel(f'{self.root}PN_to_KC_connection.xlsx')
        Cluster_to_Glomerulus, Glomerulus_to_Cluster = self.obtain_glomerulus_cluster()
        data = pd.read_excel(f'{self.root}PN_to_KC_connection.xlsx')
        PN_ids = data['up.bodyId'].values.tolist()
        KC_ids = data['down.bodyId'].values.tolist()
        PN_types = data['up.type'].values.tolist()
        KC_types = data['down.type'].values.tolist()
        weights = data['w.weight'].values.tolist()
        intance_collections = []
        for PN_id, KC_id, PN_type, KC_type, weight in zip(PN_ids, KC_ids, PN_types, KC_types, weights):
            if KC_type.find("part") != -1:
                ## This means the KC has incomplete info so that we don't take into consideration
                continue
            if weight >= self.PN_to_KC_weight_threshold:
                if PN_type not in intance_collections:
                    intance_collections.append(PN_type)
        print(intance_collections)

    # @classmethod
    def obtain_uniglomerular_PN(cls):
        '''
        #### Finished
        This function get the uniglomerular PN to KC connection filtered by connection weight.
        Notice! we don't take VP neurons into consideration!!
        :return:
        '''
        path = f"{cls.root}{cls.PN_to_KC_dir}"
        if not os.path.isdir(path): os.mkdir(path)
        Cluster_to_Glomerulus, Glomerulus_to_Cluster = cls.obtain_glomerulus_cluster()
        data = pd.read_excel(f'{cls.root}PN_to_KC_connection.xlsx')
        PN_ids = data['up.bodyId'].values.tolist()
        KC_ids = data['down.bodyId'].values.tolist()
        PN_types = data['up.type'].values.tolist()
        KC_types = data['down.type'].values.tolist()
        weights = data['w.weight'].values.tolist()
        pooled_connection_weight = {1: [], 2: [], 3: []}
        pooled_simple_type_connection_weight = {1: [], 2: [], 3: []}
        for PN_id, KC_id, PN_type, KC_type, weight in zip(PN_ids, KC_ids, PN_types, KC_types, weights):
            if KC_type.find("part") != -1:
                ## This means the KC has incomplete info so that we don't take into consideration
                continue
            if 'VM4_lv' in PN_type:
                Glomerulus = 'lvVM4'
            else:
                Glomerulus = copy.deepcopy(PN_type.split("_")[0])  ## data format is like Glomerulus_adPN
            if Glomerulus not in Glomerulus_to_Cluster or PN_type.find("+") != -1:
                ####### This means the PN is from multiglomeruli which we don't take into consideration
                continue
            if weight >= cls.PN_to_KC_weight_threshold:  ## Check weight thershold
                pooled_connection_weight[Glomerulus_to_Cluster[Glomerulus]].append(
                    [PN_id, PN_type, KC_id, KC_type, weight])
            ## obtain KC main type
            KC_main_type = cls.KC_from_subtype_to_main_type(KC_type)
            if weight >= cls.PN_to_KC_weight_threshold:
                pooled_simple_type_connection_weight[Glomerulus_to_Cluster[Glomerulus]].append(
                    [PN_id, Glomerulus, KC_id, KC_main_type, weight])
        for cluster_index in Cluster_to_Glomerulus:
            Df(data=pooled_connection_weight[cluster_index],
               columns=['PN_id', 'PN_type', 'KC_id', 'KC_type', 'weight']).to_excel(
                f"{path}UniglomerularPN_to_CompleteKC_weight{cls.PN_to_KC_weight_threshold}_cluster{cluster_index}.xlsx")
            Df(data=pooled_simple_type_connection_weight[cluster_index],
               columns=['PN_id', 'PN_type', 'KC_id', 'KC_type', 'weight']).to_excel(
                f"{path}UniglomerularPN_to_CompleteKC_weight{cls.PN_to_KC_weight_threshold}_cluster{cluster_index}_simple_type.xlsx")
        return pooled_simple_type_connection_weight

    @classmethod
    def obtain_lookup_dict_weight(cls) -> (dict,dict,dict,dict,dict,dict,dict,dict):
        '''
        #### Finished
        This function offers the dictionary to look up the corresponding neuronId, Cluster, Subtype, Glomerulus.
        :return:
        '''
        if not os.path.isfile(f"{cls.tmp_path}obtain_lookup_dict_weight{cls.PN_to_KC_weight_threshold}.pickle"):
            Glomerulus_to_Cluster, Cluster_to_Glomerulus, PNid_to_Glomerulus, PNid_to_Cluster = {}, {}, {}, {}
            Glomerulus_to_PNid, Cluster_to_PNid, Subtype_to_KCid, KCid_to_Subtype = {}, {}, {}, {}
            cluster_list = [i for i in range(1, 4)]
            for cluster in cluster_list:
                data = pd.read_excel(
                    f'{cls.root}{cls.PN_to_KC_dir}UniglomerularPN_to_CompleteKC_weight{cls.PN_to_KC_weight_threshold}_cluster{cluster}_simple_type.xlsx')
                tmp_Glomerulus_list = data['PN_type'].values.tolist()
                tmp_PN_NeuronId_list = data['PN_id'].values.tolist()
                tmp_KC_NeuronId_list = data['KC_id'].values.tolist()
                tmp_KC_subtype_list = data['KC_type'].values.tolist()
                Cluster_to_PNid[cluster] = []
                Glomerulus_list = []
                for KC_id, PN_id, Glomerulus, KC_subtype in zip(tmp_KC_NeuronId_list, tmp_PN_NeuronId_list,
                                                                tmp_Glomerulus_list, tmp_KC_subtype_list):
                    if PN_id not in PNid_to_Cluster:
                        PNid_to_Cluster[PN_id] = cluster
                        Cluster_to_PNid[cluster].append(PN_id)
                        PNid_to_Glomerulus[PN_id] = Glomerulus
                        if Glomerulus not in Glomerulus_list:
                            Glomerulus_to_PNid[Glomerulus] = []
                            Glomerulus_list.append(Glomerulus)
                        Glomerulus_to_PNid[Glomerulus].append(PN_id)
                        Glomerulus_to_Cluster[Glomerulus]=cluster
                    if KC_id not in KCid_to_Subtype:
                        KCid_to_Subtype[KC_id] = KC_subtype
                        if KC_subtype not in Subtype_to_KCid:
                            Subtype_to_KCid[KC_subtype] = []
                        Subtype_to_KCid[KC_subtype].append(KC_id)
                Cluster_to_Glomerulus[cluster] = Glomerulus_list
            pooled_data = [Glomerulus_to_Cluster, Cluster_to_Glomerulus, PNid_to_Cluster, Cluster_to_PNid,
                           PNid_to_Glomerulus, Glomerulus_to_PNid, KCid_to_Subtype, Subtype_to_KCid]
            with open(f"{cls.tmp_path}obtain_lookup_dict_weight{cls.PN_to_KC_weight_threshold}.pickle", 'wb')as ff:
                pickle.dump(pooled_data, ff)
        else:
            with open(f"{cls.tmp_path}obtain_lookup_dict_weight{cls.PN_to_KC_weight_threshold}.pickle", 'rb')as ff:
                pooled_data = pickle.load(ff)
        return pooled_data

    def get_KC_subtype_location(self):
        '''
        #### Finished
        This function offers the location of each KC subtype in the connection matrix!
        :return:
        '''
        if self.id_to_new_subtype == {}:
            self.get_new_KC_subtype()
        KC_subtype_location = {}
        KC_new_subtype_location = {}
        for KC_index,KCid in enumerate(self.KCid_list):
            if KC_index == 0:
                KC_subtype_location[self.KCid_to_Subtype[KCid]] = [0,0]
                pre_subtype = self.KCid_to_Subtype[KCid]
                KC_new_subtype_location[self.id_to_new_subtype[KCid]] = [0,0]
                pre_new_subtype = self.id_to_new_subtype[KCid]
            else:
                ptr_subtype = self.KCid_to_Subtype[KCid]
                ptr_new_subtype = self.id_to_new_subtype[KCid]
                if ptr_subtype != pre_subtype:
                    KC_subtype_location[pre_subtype][1] = KC_index - 1
                    KC_subtype_location[ptr_subtype] = [KC_index,KC_index]
                if ptr_new_subtype != pre_new_subtype:
                    KC_new_subtype_location[pre_new_subtype][1] = KC_index - 1
                    KC_new_subtype_location[ptr_new_subtype] = [KC_index,KC_index]
                pre_subtype = ptr_subtype
                pre_new_subtype = ptr_new_subtype
        KC_subtype_location[pre_subtype][1] = len(self.KCid_list)-1
        KC_new_subtype_location[pre_new_subtype][1] = len(self.KCid_list)-1
        self.KC_subtype_location = KC_subtype_location
        self.KC_new_subtype_location = KC_new_subtype_location

    def transform_PN_KC_connection_to_G_KC_connection_norm(self):
        '''
        #### Finished
        This function sum up the normalized connection weight for from same glomerular PNs to KCs.
        :return:
        '''
        self.connection_matrix_collection_dict_g_norm = {}
        for wiring_pattern in self.connection_matrix_collection_dict_g:
            self.connection_matrix_collection_dict_g_norm[wiring_pattern] = []
            for network_id, weight in enumerate(self.connection_matrix_collection_dict_g[wiring_pattern]):
                self.connection_matrix_collection_dict_g_norm[wiring_pattern].append(self.normalize_connection_weight(copy.deepcopy(weight)))

    def transform_PN_KC_connection_to_G_KC_connection(self, connection_table=[]):
        '''
        #### Finished
        This function sum up the connection weight for from same glomerular PNs to KCs.
        :return:
        '''
        if len(connection_table)==0:
            self.connection_matrix_collection_dict_g = {}
            for wiring_pattern in self.connection_matrix_collection_dict:
                self.connection_matrix_collection_dict_g[wiring_pattern] = []
                for network_id, weight in enumerate(self.connection_matrix_collection_dict[wiring_pattern]):
                    connection_matrix_g = np.zeros((len(self.G_list), len(self.KCid_list)), dtype=float)
                    for PNindex, connection in enumerate(weight):
                        G_index = self.G_list.index(self.PNid_to_Glomerulus[self.PNid_list[PNindex]])
                        for KCindex, weight in enumerate(connection):
                            connection_matrix_g[G_index][KCindex] += weight
                    self.connection_matrix_collection_dict_g[wiring_pattern].append(connection_matrix_g)
        else:
            connection_matrix_g = np.zeros((len(self.G_list), len(self.KCid_list)), dtype=float)
            for PNindex, connection in enumerate(connection_table):
                G_index = self.G_list.index(self.PNid_to_Glomerulus[self.PNid_list[PNindex]])
                for KCindex, weight in enumerate(connection):
                    connection_matrix_g[G_index][KCindex] += weight
            return connection_matrix_g

    def generate_KC_to_MBON_connection(self):
        '''
        ### Finished
        This function offers the selected KCid (filted by connection weight and degree of data completness) connects to MBON table.
        :return:
        '''
        KC_to_MBON = pd.read_excel("KC_to_MBON_connection.xlsx")
        MBON_subtypes_to_id = {}
        id_to_MBON_subtype = {}
        MBON_id_list = []
        KC_to_MBON_connections = []
        for KCid, MBONid,MBON_subtype, weight in zip(KC_to_MBON['up.bodyId'],KC_to_MBON['down.bodyId'], KC_to_MBON['down.type'], KC_to_MBON['w.weight']):
            if KCid not in self.KCid_to_Subtype:
                print(f"KC: {KCid} is not considered in the analysis.")
                continue
            KC_to_MBON_connections.append([KCid, MBONid, int(weight)])
            if MBON_subtype not in MBON_subtypes_to_id:
                MBON_subtypes_to_id[MBON_subtype] = []
            if MBONid not in id_to_MBON_subtype:
                id_to_MBON_subtype[MBONid] = MBON_subtype
                MBON_subtypes_to_id[MBON_subtype].append(MBONid)
            if MBONid not in MBON_id_list:
                MBON_id_list.append(MBONid)
        MBON_id_list = sorted(MBON_id_list, key=lambda k: int(id_to_MBON_subtype[k][4:6]))
        KC_to_MBON_connection_table = np.zeros((len(self.network.KCid_list), len(MBONid)))
        for connection in KC_to_MBON_connections:
            KC_to_MBON_connection_table[self.KCid_list.index(connection[0])][MBON_id_list.index(connection[1])] = connection[2]
        self.KC_to_MBON_connection_table = KC_to_MBON_connection_table
        self.id_to_MBON_subtype = id_to_MBON_subtype
        self.MBON_subtypes_to_id = MBON_subtypes_to_id
        self.MBON_id_list = MBON_id_list

    def analyze_KC_connections_from_PN(self):
        '''
        #### Finished
        This function analyze that each KC connection number and the weight!
        :return:
        '''
        self.KC_connections_statistics = Df(np.count_nonzero(self.pre_to_post_weight,axis=0))
        self.KC_connection_weights_statistics = Df(np.sum(self.pre_to_post_weight,axis=0))
        return self.KC_connections_statistics, self.KC_connection_weights_statistics

    def visualize_KC_wiring_diagram(self,file_name):
        '''
        #### Finished?
        This function offers the histgram and heatmap for KC connections from PN
        :param file_name:
        :return:
        '''
        plt.hist(self.KC_connections_statistics[0],
                 [i for i in range(min(self.KC_connections_statistics[0]), max(self.KC_connections_statistics[0]))])
        plt.xticks([i for i in range(min(self.KC_connections_statistics[0]), max(self.KC_connections_statistics[0]))])
        plt.title("The number of received PN inputs for KCs")
        plt.xlabel("Number of connected PN")
        plt.ylabel("Neuron number")
        plt.savefig(file_name + "_KC_connection_number.png")
        plt.close()
        plt.hist(self.KC_connection_weights_statistics[0], [i for i in range(int(min(self.KC_connection_weights_statistics[0])),
                                                                        int(max(self.KC_connection_weights_statistics[0])))])
        plt.xticks([i for i in range(int(min(self.KC_connection_weights_statistics[0])),
                                     int(max(self.KC_connection_weights_statistics[0])))])
        plt.title("The number of received PN inputs for KCs")
        plt.xlabel("Sum of connection weights")
        plt.ylabel("Neuron number")
        plt.savefig(file_name + "_KC_connection_weight.png")
        plt.close()
        sn.clustermap(self.pre_to_post_weight)
        plt.savefig(file_name + "_KC_connection_cluster_map.png")
        plt.close()
        sn.heatmap(self.pre_to_post_weight)
        plt.savefig(file_name + "_KC_connection_heat_map.png")
        plt.close()
        return

    def get_new_KC_subtype(self):
        '''
        #### Finished
        :return:
        '''
        New_subtype_to_id = defaultdict(list)
        All_new_subtype_to_id = defaultdict(list)
        id_to_new_subtype = {}
        KC_info_file = 'KC_info.xlsx'
        data = pd.read_excel(f"{self.root}{KC_info_file}")
        for bodyId, subtype in zip(data['n.bodyId'].values.tolist(), data['n.type'].values.tolist()):
            All_new_subtype_to_id[subtype].append(bodyId)
            id_to_new_subtype[bodyId] = subtype
            if bodyId not in self.KCid_list:
                continue
            New_subtype_to_id[subtype].append(bodyId)
        self.New_subtype_to_id = New_subtype_to_id
        self.All_new_subtype_to_id = All_new_subtype_to_id
        self.id_to_new_subtype = id_to_new_subtype
        return

    def generate_simple_connection_table_weight(self): #####final
        '''
        -> obtain_uniglomerular_PN -> obtain
        :return:
        '''
        pooled_simple_type_connection_weight = self.obtain_uniglomerular_PN()
        self.Glomerulus_to_Cluster, self.Cluster_to_Glomerulus, self.PNid_to_Cluster, self.Cluster_to_PNid, \
        self.PNid_to_Glomerulus, self.Glomerulus_to_PNid, self.KCid_to_Subtype, self.Subtype_to_KCid =self.obtain_lookup_dict_weight()
        KC_id_list = []
        PN_id_list = []
        for cluster_index in pooled_simple_type_connection_weight:
            for connection in pooled_simple_type_connection_weight[cluster_index]:
                if connection[0] not in PN_id_list:
                    PN_id_list.append(connection[0])
                if connection[2] not in KC_id_list:
                    KC_id_list.append(connection[2])
        self.get_new_KC_subtype()
        KC_id_list = sorted(KC_id_list, key=lambda k: self.New_subtype_to_label[self.id_to_new_subtype[k]])
        PN_id_list = sorted(PN_id_list, key=lambda k: self.PNid_to_Cluster[k])
        PN_number = len(PN_id_list)
        KC_number = len(KC_id_list)
        self.pre_number = PN_number
        self.post_number = KC_number
        pre_to_post_weight = np.array([[0.0 for _ in range(len(KC_id_list))] for __ in range(len(PN_id_list))])
        for cluster_index in pooled_simple_type_connection_weight:
            for connection in pooled_simple_type_connection_weight[cluster_index]:
                pre_to_post_weight[PN_id_list.index(connection[0])][KC_id_list.index(connection[2])] = connection[4]
        self.pre_to_post_weight = pre_to_post_weight
        self.original_pre_to_post_weight = copy.deepcopy(pre_to_post_weight)
        self.connection_matrix_collection_dict['FlyEM'] = [self.original_pre_to_post_weight]
        self.KCid_list = KC_id_list
        self.PNid_list = PN_id_list
        return

    def generate_simple_connection_table_binary(self):
        self.pre_to_post_weight_binary = copy.deepcopy(self.pre_to_post_weight)
        self.pre_to_post_weight_binary[self.pre_to_post_weight_binary>0] = 1

    def shuffle_connection_table_fix_KC(self, shuffle_times=2, seed=100, network_number=3, shuffle_ratio=1):
        self.shuffle_ratio = shuffle_ratio
        print("Shuffle the connection table!")
        self.shuffled_pre_to_post_weight_collection = []
        for shuffle_seed in range(seed, seed + network_number):
            print(shuffle_seed)
            rd.seed(shuffle_seed)
            self.pre_to_post_weight = copy.deepcopy(self.original_pre_to_post_weight)
            KC_total_number = 0
            KC_weight_list = []
            PN_wait_list = []
            for KC_index in range(self.post_number):
                KC_weight_list.append([])
                for PN_index in range(self.pre_number):
                    p = rd.random()
                    if self.pre_to_post_weight[PN_index][
                        KC_index] > 0 and p < shuffle_ratio:  # the shuffle_ratio controls the prob for shuffling
                        KC_weight_list[-1].append(self.pre_to_post_weight[PN_index][
                                                      KC_index])  ##From this list, we will know how many connections are there for that KC
                        self.pre_to_post_weight[PN_index][KC_index] = 0
                        PN_wait_list.append(
                            PN_index)  ##From this list, we will know how many connections are there for that PN
            max_KC = 0
            for i in KC_weight_list:
                KC_total_number += len(i)
                if max_KC < len(i):
                    max_KC = len(i)
            for _ in range(shuffle_times):
                rd.shuffle(PN_wait_list)
            Connection_list = []
            while 1:
                tmp_PN_wait_list = copy.deepcopy(PN_wait_list)
                error_count = 0
                for connection_index in range(max_KC, -1, -1):  ### Start from the largest connection number KC
                    '''
                    The algorithm assigns PN to KC in turn along the length.
                    For example:
                    connection_index = 100
                    if KCi connection number is larger than 100:
                    assign one PN
                    when all connection_index=100 have its connection
                    connection index becomes 99
                    assign all KCi that have 99th connection
                    Together, KCi does not get all PN directly. They take turns to get PN.
                    This helps prevent the illegal event that the KC has no legal PN to connect. (One KC can only connect a PN once)
                    '''
                    for KC_index in range(self.post_number):
                        if connection_index < len(KC_weight_list[KC_index]):
                            weight = KC_weight_list[KC_index][connection_index]
                            error_count = 0
                            while (KC_index, tmp_PN_wait_list[-1]) in Connection_list:
                                rd.shuffle(tmp_PN_wait_list)
                                error_count += 1
                                if error_count > 30:
                                    ##sometimes we generate impossible random network which means the KC cannot connect PN once, we need to restart again.
                                    break
                            if error_count > 30:
                                break
                            self.pre_to_post_weight[tmp_PN_wait_list[-1]][KC_index] = weight
                            Connection_list.append((KC_index, tmp_PN_wait_list[-1]))
                            tmp_PN_wait_list = tmp_PN_wait_list[:-1]
                    if error_count > 30:
                        break
                if error_count > 30:
                    self.pre_to_post_weight[self.pre_to_post_weight > 0] = 0
                    continue
                else:
                    break
            if len(tmp_PN_wait_list) > 0:
                print(f"length: {len(PN_wait_list)}")
                print("ERROR")
                raise BaseException("PN wait list is largern than 0!!!!!!!!!!!!!!!!!!!!!!!")
                continue
            self.shuffled_pre_to_post_weight_collection.append(copy.deepcopy(self.pre_to_post_weight))
        self.shuffled_pre_to_post_weight = self.shuffled_pre_to_post_weight_collection[0]
        if shuffle_ratio == 1:
            self.connection_matrix_collection_dict['Random network fix KC'] = self.shuffled_pre_to_post_weight_collection
        else:
            self.connection_matrix_collection_dict[
                f'Random network {shuffle_ratio} fix KC'] = self.shuffled_pre_to_post_weight_collection
        print(f"Finished! We have constructed {len(self.shuffled_pre_to_post_weight_collection)} random networks ifx KC")

    # def prepare_shuffle_matrix(self, weight_matrix, ratio):
    def plot_partial_shuffle_result(self, KC_class, binary=True):
        if KC_class != 'ALL':
            column_index_list = [i for i in range(len(self.KCid_list)) if self.KCid_to_Subtype[self.KCid_list[i]] == KC_class]
        else:
            column_index_list = [i for i in range(len(self.KCid_list))]
        weight_matrix = self.connection_matrix_collection_dict_g['FlyEM'][0][:,column_index_list]
        if binary:
            weight_matrix[weight_matrix>0] = 1
        summation = np.sum(weight_matrix, axis=1)
        row_index = [i for i in range(summation.shape[0]) if summation[i] > 0]
        tmp_connection_matrix = weight_matrix[row_index]
        correlation_matrix = np.corrcoef(tmp_connection_matrix, rowvar=True)

        if np.isnan(correlation_matrix).any() or np.isinf(correlation_matrix).any():
            print("There is nan in the correlation matrix!!!!")
            # correlation_matrix = np.nan_to_num(correlation_matrix)

        eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)

        pca = PCA()
        # Applying PCA
        data_pca = pca.fit_transform(correlation_matrix)

        # Getting the explained variance ratio
        explained_variance_ratio = pca.explained_variance_ratio_

        if binary:
            b_string = 'binary'
        else:
            b_string = 'weighted'
        data = pd.read_excel(f"Partial shuffle PC1 {KC_class}_{b_string}_pearson_100.xlsx")

        data = data.drop(columns=['Unnamed: 0'])
        x = [0]
        y = [explained_variance_ratio[0]*100]
        # y = [np.max(eigenvalues)/np.sum(eigenvalues)*100]
        std = [0]
        for p in data.columns:
            x.append(int(float(p) * 100))
            y.append(data[p].mean()*100)
            std.append(data[p].std()*100)
            print(y)
            print(std)
        plt.figure(figsize=(8, 6))
        plt.bar(x[0],y[0],4.5)
        plt.bar(x[1:],y[1:],4.5,capsize=5,yerr=std[1:], color='brown')
        # plt.axhline(y=y[-1] + std[-1],color='r',linestyle='--',label)
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5)  # X-axis
        ax.spines['left'].set_linewidth(1.5)  # Y-axis
        ax.spines['top'].set_linewidth(1.5)  # X-axis
        ax.spines['right'].set_linewidth(1.5)  # Y-axis
        plt.xlabel('Shuffled ratio (%)', fontdict={'fontsize': 20})
        plt.ylabel('Explaned variance (%)', fontdict={'fontsize': 20})
        plt.yticks([0,4,8,12],fontsize=16)
        plt.xticks([0,20,40,60,80,100],fontsize=16)
        plt.show()
        # plt.savefig(f"Partial_shuffled_{KC_class}_{b_string}.png",dpi=500)
        # plt.savefig(f"Partial_shuffled_{KC_class}_{b_string}.svg")
        # plt.close()

    def subsample_neurons(self,KC_class,neuron_number,network_number,seed=100,binary=True):
        rd.seed(seed)
        np.random.seed(seed)
        root = 'partial_shuffled_PCA_result/'
        if not os.path.isdir(root): os.mkdir(root)

        if KC_class != 'ALL':
            column_index_list = [i for i in range(len(self.KCid_list)) if
                                 self.KCid_to_Subtype[self.KCid_list[i]] == KC_class]
        else:
            column_index_list = [i for i in range(len(self.KCid_list))]
        for network_id in range(network_number):
            rd.shuffle(column_index_list)
            column_index_list = column_index_list[:neuron_number]
            weight_matrix = self.connection_matrix_collection_dict_g['FlyEM'][0][:, column_index_list]

    def partial_shuffle_connections(self, KC_class, network_number=10, shuffle_ratio=1.0, seed=100, binary=True):
        rd.seed(seed)
        np.random.seed(seed)
        root = 'partial_shuffled_PCA_result/'
        if not os.path.isdir(root): os.mkdir(root)

        if KC_class != 'ALL':
            column_index_list = [i for i in range(len(self.KCid_list)) if self.KCid_to_Subtype[self.KCid_list[i]] == KC_class]
        else:
            column_index_list = [i for i in range(len(self.KCid_list))]
        weight_matrix = self.connection_matrix_collection_dict_g['FlyEM'][0][:,column_index_list]
        original_connection_matrix = copy.deepcopy(weight_matrix)
        original_connection_matrix[original_connection_matrix > 0] = 1
        pre_num = original_connection_matrix.shape[0]
        post_num = original_connection_matrix.shape[1]
        pca_score_collection = []
        pooled_explained_ratio = []
        for network_i in range(network_number):
            while 1:
                if binary:
                    connection_matrix = copy.deepcopy(original_connection_matrix)
                else:
                    connection_matrix = copy.deepcopy(weight_matrix)
                retry = 0
                pre_weight_list = []
                post_wait_list = []
                for pre_index in range(pre_num):
                    pre_weight_list.append([])
                    for post_index in range(post_num):
                        w = connection_matrix[pre_index][post_index]
                        if w > 0 and rd.random() < shuffle_ratio:
                            ## From this list, we will know how many connections are there for that PN
                            pre_weight_list[pre_index].append(w)
                            connection_matrix[pre_index][post_index] = 0
                            ## Put post neuronId into wait list
                            post_wait_list.append(post_index)
                rd.shuffle(post_wait_list)
                Connection_list = list(map(tuple,np.argwhere(connection_matrix).tolist()))
                max_pre = max([len(k) for k in pre_weight_list])
                tmp_connection_matrix = copy.deepcopy(connection_matrix)
                tmp_post_wait_list = copy.deepcopy(post_wait_list)
                for connection_index in range(max_pre, -1, -1):  ### Start from the largest connection number PN
                    '''
                    The algorithm assigns KC to PN in turn along the length.
                    For example:
                    connection_index = 100
                    if PNi connection number is larger than 100:
                    assign one KC
                    when all connection_index=100 have its connection
                    connection index becomes 99
                    assign all PNi that have 99th connection
                    Together, PNi does not get all KC directly. They take turns to get KC.
                    This helps prevent the illegal event that the PN has no legal KC to connect. (One PN can only connect a KC once)
                    '''
                    if retry == 1:
                        break
                    for pre_index in range(pre_num):
                        if connection_index < len(pre_weight_list[pre_index]):
                            weight = pre_weight_list[pre_index][connection_index]
                            for p, post_index in enumerate(tmp_post_wait_list):
                                if (pre_index, post_index) not in Connection_list:
                                    break
                            if (pre_index, post_index) in Connection_list:
                                retry = 1
                                break
                            else:
                                tmp_connection_matrix[pre_index][post_index] = weight
                                Connection_list.append((pre_index, post_index))
                                tmp_post_wait_list.pop(p)
                if len(tmp_post_wait_list) == 0:
                    break
            if binary:
                summation_pre = np.sum(original_connection_matrix,axis=0)
                new_pre = np.sum(tmp_connection_matrix,axis=0)
                if np.count_nonzero(summation_pre-new_pre) > 0:
                    raise BaseException("The connection matrix is wrong!!")
                summation_post = np.sum(original_connection_matrix,axis=1)
                new_post = np.sum(tmp_connection_matrix,axis=1)
                if np.count_nonzero(summation_post-new_post) > 0:
                    raise BaseException("The connection matrix is wrong!!")
            pca = PCA()
            summation = np.sum(tmp_connection_matrix, axis=1)
            row_index = [i for i in range(summation.shape[0]) if summation[i] > 0]
            tmp_connection_matrix = tmp_connection_matrix[row_index]
            correlation_matrix = np.corrcoef(tmp_connection_matrix,rowvar=True)
            if np.isnan(correlation_matrix).any() or np.isinf(correlation_matrix).any():
                print("There is nan in the correlation matrix!!!!")
                continue

            # Applying PCA
            data_pca = pca.fit_transform(correlation_matrix)

            # Getting the explained variance ratio
            explained_variance_ratio = pca.explained_variance_ratio_
            pooled_explained_ratio.append(explained_variance_ratio)

            # Output the explained variance ratio
            pca_score_collection.append(explained_variance_ratio[0])
        print(np.average(pca_score_collection),np.std(pca_score_collection))
        if binary:
            Df(data=np.array(pooled_explained_ratio)).to_excel(f"{root}{KC_class}_{shuffle_ratio}.xlsx")
        else:
            Df(data=np.array(pooled_explained_ratio)).to_excel(f"{root}{KC_class}_{shuffle_ratio}_weighted.xlsx")

        # self.partial_connection_dict[]
        # for i in range(network_number):
        return pca_score_collection

    def get_partial_PCA(self):
        return

    def match_bouton_claw(self, boutons, claw):
        return np.argmin([distance.euclidean(boutons[i], claw) for i in range(len(boutons))])

    def match_claw_bouton_reconstruct_connection_table(self):
        path = 'PN_KC_bouton_claw_information_20230618/'
        PNid_bouton_dict = {}
        connection_table = np.zeros(self.original_pre_to_post_weight.shape)
        for PNid in self.PNid_list:
            PNid_bouton_dict[PNid] = []
            with open(f"{path}{PNid}_bouton.txt",'r')as ff:
                for line in ff:
                    groups = line[:-1].split(" ")[:3]
                    PNid_bouton_dict[PNid].append(list(map(float,groups)))
        self.PNid_bouton_dict = PNid_bouton_dict
        KCid_claw_dict = {}
        for KCid in self.KCid_list:
            KCid_claw_dict[KCid] = []
            with open(f"{path}{KCid}_claw.txt",'r')as ff:
                for line in ff:
                    groups = line[:-1].split(" ")[:3]
                    KCid_claw_dict[KCid].append(list(map(float,groups)))
        self.KCid_claw_dict = KCid_claw_dict
        for KCindex, KCid in enumerate(self.KCid_list):
            connected_PNids = [self.PNid_list[i] for i in np.nonzero(self.original_pre_to_post_weight[:,KCindex])[0]]
            candidate_bouton = []
            candidate_bouton_PNids = []
            for PNid in connected_PNids:
                candidate_bouton += PNid_bouton_dict[PNid]
                candidate_bouton_PNids += [PNid for _ in range(len(PNid_bouton_dict[PNid]))]
            for claw in KCid_claw_dict[KCid]:
                PNid = candidate_bouton_PNids[self.match_bouton_claw(candidate_bouton, claw)]
                connection_table[self.PNid_list.index(PNid)][KCindex] = 1
        KCid_claw_num_dict = {KCid:len(KCid_claw_dict[KCid]) for KCid in KCid_claw_dict}
        self.KCid_claw_num_dict = KCid_claw_num_dict
        # for PNindex, PNid in enumerate(self.PNid_list):
            # for KCindex, KCid in enumerate(self.KCid_list):
        connection_table_df = Df(data=connection_table,columns=[f"KC{i}" for i in range(connection_table.shape[1])], index=[self.PNid_to_Glomerulus[PNid] for PNid in self.PNid_list])
        connection_table_df.to_csv("PN_to_KC_connection_table_bouton_claw.csv")
        sn.heatmap(data=connection_table)
        plt.savefig("PN_to_KC_connection_table_bouton_claw.png")
        plt.close()
        binary_map = copy.deepcopy(self.original_pre_to_post_weight)
        binary_map[binary_map>0]=1
        sn.heatmap(data=binary_map)
        plt.savefig("PN_to_KC_connection_table.png")
        plt.close()
        sn.heatmap(data=binary_map-connection_table)
        plt.savefig("PN_to_KC_connection_table_diff_from_bouton_claw.png")
        plt.close()
        self.PN_bouton_KC_claw_connection_table = connection_table

    def get_connectivity(self, connection_table, PN_table=True, dataset='FlyEM'):
        if dataset == "FAFB":
            KC_class_list = ['KCg',"KCa'b'",'KCab','Other']
        elif dataset == 'FlyEM':
            KC_class_list = ['KCg',"KCa'b'",'KCab']
        
        connectivity = np.zeros((len(self.G_list),len(self.Subtype_to_KCid)))
        print(self.KCid_to_Subtype)
        if PN_table:
            for i in range(len(self.PNid_list)):
                for j in range(len(self.KCid_list)):
                    G_index = self.G_list.index(self.PNid_to_Glomerulus[self.PNid_list[i]])
                    Class_index = KC_class_list.index(self.KCid_to_Subtype[self.KCid_list[j]])
                    connectivity[G_index][Class_index] += int(connection_table[i][j]>0)
        # else:
        #     for

        return connectivity

    def construct_random_glomerulus_model(self, shuffled_times=1000):
        KC_class_list = ['KCg',"KCa'b'",'KCab']
        connection_table = self.PN_bouton_KC_claw_connection_table
        original_connectivity = self.get_connectivity(connection_table).transpose().ravel()
        G_PN_list = [G for G in self.G_list for _ in range(len(self.Glomerulus_to_PNid))]
        connectivity_collection = []
        for shuffled_index in range(shuffled_times):
            connectivity = np.zeros((len(self.G_list),len(KC_class_list)))
            for KCid in self.KCid_list:
                KC_class = self.KCid_to_Subtype[KCid]
                for clawid in range(len(self.KCid_claw_dict[KCid])):
                    G = rd.sample(G_PN_list,1)[0]
                    connectivity[self.G_list.index(G)][KC_class_list.index(KC_class)] += 1
            connectivity_collection.append(connectivity.transpose().ravel())
        connectivity_collection = np.array(connectivity_collection)
        z_score = (original_connectivity - np.average(connectivity_collection,axis=0))/np.std(connectivity_collection,axis=0)
        z_score = z_score.reshape((3,len(self.G_list)))
        
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9, 1.5 * 3))
        for classification_index in range(len(KC_class_list)):
            ax = axes[classification_index]
            plt.sca(ax)
            height = z_score[classification_index]
            ax.bar(x=[i for i in range(len(self.G_list))], height=height, color='k')
            plt.axhline(y=2,linestyle='--', color='gray')
            plt.axhline(y=-2,linestyle='--',color='gray')

            xticklabel = self.G_list
            if classification_index == 2:
                plt.xticks([i for i in range(len(self.G_list))],xticklabel, rotation=90)
                for i, label in enumerate(ax.get_xticklabels()):
                    if self.Glomerulus_to_Cluster[self.G_list[i]]==1:  # Color every other label red
                        label.set_color('red')
                    elif self.Glomerulus_to_Cluster[self.G_list[i]]==2:  # Color every other label red
                        label.set_color('gold')
                    elif self.Glomerulus_to_Cluster[self.G_list[i]]==3:  # Color every other label red
                        label.set_color('deepskyblue')
            else:
                plt.xticks([])
                        
        plt.savefig('Random_glomerulus_model_z_score.png',dpi=500)
        plt.savefig('Random_glomerulus_model_z_score.svg',format='svg')
        plt.close()
        result = Df(data=z_score.transpose,columns=["KCg", "KCa'b'","KCab"], index=self.G_list)
        result.to_csv("Z_score_table_random_glomerulus_model.csv")
        return

    def perform_condition_input_analysis(self, connection_table):
        matrix = np.zeros((len(self.G_list),len(self.G_list)))
        for PNindex, PNid in enumerate(self.PNid_list):
            connected_KCids = np.nonzero(connection_table[PNindex])[0]
            for KCindex in connected_KCids:
                upstream_PNids = np.nonzero(connection_table[:,KCindex])[0]
                for PNindex2 in upstream_PNids:
                    # if PNindex == PNindex2:
                    #     continue
                    G_index_1 = self.G_list.index(self.PNid_to_Glomerulus[PNid])
                    G_index_2 = self.G_list.index(self.PNid_to_Glomerulus[self.PNid_list[PNindex2]])
                    matrix[G_index_1][G_index_2] += 1
        return matrix
    
    def revisit_FAFB_claw_num(self):
        file_name = 'FAFB_bouton_claw_connection.csv'
        data = pd.read_csv(file_name)
        KC_class_list = ['KCy',"KCa'B'",'KCaB']
        for KC_class in KC_class_list:
            count_list = []
            mask = data['kc_names'].str.contains(KC_class)
            filtered_data = data[mask]
            neuronIds = filtered_data['kc_skid'].unique().tolist()
            for neuronId in neuronIds:
                maskId = filtered_data['kc_skid'] == neuronId
                count_list.append(len(filtered_data[maskId]))
            print(KC_class, np.average(count_list), np.std(count_list))
        print('total data, claw_num',len(data),len(data['claw_ids']))

    def check_FAFB_bouton(self):
        file_name = "FAFB_PN_num.xlsx"
        data = pd.read_excel(file_name)
        G_bouton_dict = {}
        for G, bouton_num in zip(data['Glomerulus'],data['total bouton num']):
            G_bouton_dict[G] = bouton_num
        self.G_bouton_num_dict_FAFB = G_bouton_dict
        # https://ars.els-cdn.com/content/image/1-s2.0-S0960982222009903-mmc2.csv, Obtained from Zheng et al., 2022
        file_name = 'FAFB_bouton_claw_connection.csv'
        data = pd.read_csv(file_name)
        PNid_list_FAFB = data['pn_skid'].unique().tolist()
        KCid_list_FAFB = data['kc_skid'].unique().tolist()
        KC_names = Df(data=np.array([i.split(" ")[0] for i in data['kc_names'].unique().tolist()]),columns=['kc_names'])['kc_names'].unique()
        G_list_FAFB = data['pn_type'].unique().tolist()
        check_list = []
        additional_list = []
        for G in self.Classification_dict['Glomerulus']:
            if G not in G_list_FAFB:
                check_list.append(G)
        for G in G_list_FAFB:
            if G not in self.Classification_dict['Glomerulus']:
                additional_list.append(G)
        print(check_list)
        print(additional_list)

    def read_FAFB_connection_csv(self):
        ## FAFB data only contains VM4 but not lvVM4. In FlyEM, both of them are in the dataset.
        file_name = "FAFB_PN_num.xlsx"
        data = pd.read_excel(file_name)
        G_bouton_dict = {}
        for G, bouton_num in zip(data['Glomerulus'],data['total bouton num']):
            G_bouton_dict[G] = bouton_num
        self.G_bouton_num_dict_FAFB = G_bouton_dict
        # https://ars.els-cdn.com/content/image/1-s2.0-S0960982222009903-mmc2.csv, Obtained from Zheng et al., 2022
        file_name = 'FAFB_bouton_claw_connection.csv'
        data = pd.read_csv(file_name)
        PNid_list_FAFB = data['pn_skid'].unique().tolist()
        KCid_list_FAFB = data['kc_skid'].unique().tolist()
        KC_names = Df(data=np.array([i.split(" ")[0] for i in data['kc_names'].unique().tolist()]),columns=['kc_names'])['kc_names'].unique()
        print(KC_names)
        print(len(KCid_list_FAFB))
        G_list_FAFB = data['pn_type'].unique().tolist()
        connection_matrix = np.zeros((len(PNid_list_FAFB),len(KCid_list_FAFB)))
        data = data.values.tolist()
        KCid_claw_num_list = [0 for _ in range(len(KCid_list_FAFB))]
        KCid_claw_num_dict = {}
        PNid_to_G_FAFB = {}
        G_to_PNid_FAFB = {}
        KCid_to_Subtype_FAFB = {}
        Subtype_to_KCid_FAFB = {}
        KC_class_list_FAFB = ['KCy',"KCa'B'","KCaB"]
        KC_class_list = ["KCg","KCa'b'","KCab",'Other']
        error_list = []

        for connection in data:
            PNid,KCid, pn_name, KC_name, G, clawId = connection
            KCindex = KCid_list_FAFB.index(KCid)
            PNindex = PNid_list_FAFB.index(PNid)
            if G not in G_to_PNid_FAFB:
                G_to_PNid_FAFB[G] = []
            if PNid not in PNid_to_G_FAFB:
                PNid_to_G_FAFB[PNid] = G
                G_to_PNid_FAFB[G].append(PNid)
            connection_matrix[PNindex][KCindex] += 1
            check = 0
            for KC_class_id, KC_class in enumerate(KC_class_list_FAFB):
                if KC_class in KC_name.split(" ")[0]:
                    check = 1
                    break
            if check == 0:
                # print(f"No such KC class: {KC_name}")
                if KCid not in error_list:
                    error_list.append(KCid)
                KC_class = 'Other'
                KC_class_id = 3
            KC_class = KC_class_list[KC_class_id]
            if KCid not in KCid_to_Subtype_FAFB:
                KCid_to_Subtype_FAFB[KCid] = KC_class
                if KC_class not in Subtype_to_KCid_FAFB:
                    Subtype_to_KCid_FAFB[KC_class] = []
                Subtype_to_KCid_FAFB[KC_class].append(KCid)
            KCid_claw_num_list[KCindex] += 1
        
        self.connection_matrix_FAFB = connection_matrix
        self.KCid_claw_num_list_FAFB = KCid_claw_num_list
        self.KCid_to_Subtype_FAFB = KCid_to_Subtype_FAFB
        self.PNid_to_Glomerulus_FAFB = PNid_to_G_FAFB
        self.Glomerulus_to_PNid_FAFB = G_to_PNid_FAFB
        self.Subtype_to_KCid_FAFB = Subtype_to_KCid_FAFB
        self.G_list_FAFB = [G for G in self.G_list if G in G_to_PNid_FAFB] + [G for G in G_to_PNid_FAFB if G not in self.G_list]
        self.KCid_list_FAFB = KCid_list_FAFB
        self.PNid_list_FAFB = PNid_list_FAFB
        self.KCid_claw_num_dict_FAFB = {KCid_list_FAFB[i]:KCid_claw_num_list[i] for i in range(len(KCid_list_FAFB))}
        Glomerulus_to_Cluster_FAFB = self.Glomerulus_to_Cluster
        for G in G_list_FAFB:
            if G not in Glomerulus_to_Cluster_FAFB:
                Glomerulus_to_Cluster_FAFB[G] = 4
        self.Glomerulus_to_Cluster_FAFB = Glomerulus_to_Cluster_FAFB

        print(len(error_list))
    
    def check_overlapped_FAFB(self):
        overlapped_list = []
        for KC_class_1 in self.Subtype_to_KCid_FAFB:
            for KC_class_2 in self.Subtype_to_KCid_FAFB:
                if KC_class_1 == KC_class_2:
                    continue
                for KCid in self.Subtype_to_KCid_FAFB[KC_class_1]:
                    if KCid in self.Subtype_to_KCid_FAFB[KC_class_2]:
                        if KCid not in overlapped_list:
                            overlapped_list.append(KCid)
        print("Overlapped_list", overlapped_list)

    def compare_FAFB_and_FlyEM_KC_num(self):
        self.check_overlapped_FAFB()
        KC_class_list =["KCg","KCa'b'","KCab"]
        result = []
        for KC_class in ["KCg","KCa'b'","KCab"]:
            KCid_list = Df(data=self.Subtype_to_KCid_FAFB[KC_class], columns=['KCid'])['KCid'].unique()
            result.append([KC_class, len(self.Subtype_to_KCid[KC_class]),'FlyEM'])
            result.append([KC_class, len(KCid_list),'FAFB'])
            print(KC_class,len(self.Subtype_to_KCid_FAFB[KC_class]))
        result = Df(data=result,columns=["KC class", 'KC num', 'Dataset'])
        result.to_excel("FAFB_FlyEM_neuron_num_comparing.xlsx")
        fig, ax = plt.subplots(ncols=1, nrows=1)
        sn.barplot(data=result,x="KC class",y="KC num", hue='Dataset',palette=['black','grey','black','grey','black','grey'])
        ax.set_ylabel("KC number", fontsize=30)
        ax.set_xlabel(f"KC class", fontsize=30)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.tight_layout()
        w = 1.5
        ax.spines['bottom'].set_linewidth(w)  # X-axis
        ax.spines['left'].set_linewidth(w)  # Y-axis
        ax.spines['top'].set_linewidth(w)  # X-axis
        ax.spines['right'].set_linewidth(w)  # Y-axis        
        plt.savefig("FlyEM_FAFB_KC_num_comparison.png")
        plt.savefig("FlyEM_FAFB_KC_num_comparison.svg")
        plt.close()

        num_ratio = []
        for KC_class in KC_class_list:
            mask = result['KC class'] == KC_class
            mask_F = result['Dataset'] == 'FAFB'
            mask_h = result['Dataset'] == 'FlyEM'
            ratio = result[mask & mask_F]['KC num'].mean()/result[mask & mask_h]['KC num'].mean()
            print(KC_class,'num ratio',ratio)
            num_ratio.append(ratio)
        self.num_ratio = num_ratio

    def check_claw_by_table(self):
        KC_class_list = ['KCg',"KCa'b'",'KCab']
        for KC_class in KC_class_list:
            KCindex_list = [self.KCid_list_FAFB.index(i) for i in self.Subtype_to_KCid_FAFB[KC_class]]
            print(KC_class,np.sum(self.connection_matrix_FAFB[:,KCindex_list]/len(KCindex_list)))

    def compare_claw_num_FAFB_FlyEM(self):
        self.check_claw_by_table()
        result = []
        KC_class_list = ['KCg',"KCa'b'",'KCab']
        for KC_class in KC_class_list:
            result_class = [[KCid, 'FlyEM', KC_class, self.KCid_claw_num_dict[KCid]] for KCid in self.Subtype_to_KCid[KC_class]]
            values = [i[3] for i in result_class]
            result += result_class
            print(KC_class,'FlyEM', np.average(values),np.std(values))
            result_class = [[KCid, 'FAFB', KC_class, self.KCid_claw_num_dict_FAFB[KCid]] for KCid in self.Subtype_to_KCid_FAFB[KC_class]]
            values = [i[3] for i in result_class]
            result += result_class
            print(KC_class,'FAFB', np.average(values),np.std(values))
        
        result = Df(data=result, columns=["KCid", "Dataset", "KC class",'Claw num'])
        result.to_excel("Claw_num_comparison_FAFB_FlyEM.xlsx")

        ax = sn.boxplot(data=result, y="Claw num", x='Dataset', hue='KC class',palette=['r', 'gold', 'deepskyblue'])
        ax.set_ylabel("Claw num", fontsize=30)
        ax.set_xlabel(f"Dataset", fontsize=30)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["hemibrain",'FAFB'], fontsize=24)
        ax.set_yticks([0, 6, 12, 18])
        ax.set_yticklabels(['0', '6', '12', '18'], fontsize=24)
        plt.tight_layout()
        w = 1.5
        ax.spines['bottom'].set_linewidth(w)  # X-axis
        ax.spines['left'].set_linewidth(w)  # Y-axis
        ax.spines['top'].set_linewidth(w)  # X-axis
        ax.spines['right'].set_linewidth(w)  # Y-axis
        plt.savefig('Claw_num_comparison_FAFB_FlyEM_cluster.png')
        plt.savefig('Claw_comparison_FAFB_FlyEM_cluster.svg')
        plt.close()

        ax = sn.boxplot(data=result, y="Claw num", x='KC class', hue='Dataset')
        ax.set_ylabel("Claw num", fontsize=30)
        ax.set_xlabel(f"KC class", fontsize=30)
        plt.xticks(fontsize=24)
        ax.set_yticks([0, 6, 12, 18])
        ax.set_yticklabels(['0', '6', '12', '18'], fontsize=24)
        plt.tight_layout()
        w = 1.5
        ax.spines['bottom'].set_linewidth(w)  # X-axis
        ax.spines['left'].set_linewidth(w)  # Y-axis
        ax.spines['top'].set_linewidth(w)  # X-axis
        ax.spines['right'].set_linewidth(w)  # Y-axis
        plt.savefig('Claw_num_comparison_FAFB_FlyEM_cluster_v2.png')
        plt.savefig('Claw_comparison_FAFB_FlyEM_cluster_v2.svg')
        plt.close()
        claw_ratio = []
        for KC_class in KC_class_list:
            mask = result['KC class'] == KC_class
            mask_F = result['Dataset'] == 'FAFB'
            mask_h = result['Dataset'] == 'FlyEM'
            ratio = result[mask & mask_F]['Claw num'].mean()/result[mask & mask_h]['Claw num'].mean()
            print(KC_class,'claw ratio',ratio)
            claw_ratio.append(ratio)
        self.claw_ratio = claw_ratio

    def compare_bouton_num_FAFB_FlyEM(self):
        result = []
        G_bouton_num_dict = {}
        for PNid in self.PNid_bouton_dict:
            G = self.PNid_to_Glomerulus[PNid]
            if G not in G_bouton_num_dict:
                G_bouton_num_dict[G] = 0
            G_bouton_num_dict[G] += len(self.PNid_bouton_dict[PNid])

        for G in self.G_list:
            if G in self.G_list_FAFB:
                bouton_num_FAFB = self.G_bouton_num_dict_FAFB[G]
                bouton_num_FlyEM = G_bouton_num_dict[G]
                result.append([G, self.Glomerulus_to_Cluster[G], bouton_num_FAFB, bouton_num_FlyEM, bouton_num_FAFB/bouton_num_FlyEM])
        result = Df(data=result, columns=["Glomerulus",'Cluster','Bouton num in FAFB','Bouton num in FlyEM', 'Ratio of bouton num (FAFB/FlyEM)'])
        result.to_excel("Bouton_num_comparison_FAFB_FlyEM.xlsx")
        sn.boxplot(data=result,x='Cluster', y='Ratio of bouton num (FAFB/FlyEM)')
        plt.savefig('Bouton_num_comparison_FAFB_FlyEM_cluster.png')
        plt.savefig('Bouton_num_comparison_FAFB_FlyEM_cluster.svg')
        plt.close()

    def compare_connectivity_FAFB_and_FlyEM(self):
        KC_class_list = ['KCg',"KCa'b'",'KCab']
        result_collection = []
        ratio_collection = []
        for G in self.G_list:
            if G in self.G_list_FAFB:
                tmp = []
                PNindex_list = [self.PNid_list_FAFB.index(PNid) for PNid in self.Glomerulus_to_PNid_FAFB[G]]
                connections = self.connection_matrix_FAFB[PNindex_list]
                connections[connections>0] = 1
                connection_num = np.sum(connections)
                result_collection.append([G,'FAFB',self.Glomerulus_to_Cluster[G], 'ALL', connection_num])
                tmp.append(connection_num)
                for KC_class in KC_class_list:
                    KCindex_list = [self.KCid_list_FAFB.index(KCid) for KCid in self.Subtype_to_KCid_FAFB[KC_class]]
                    connections_sub = connections[:,KCindex_list]
                    connection_num = np.sum(connections_sub)
                    result_collection.append([G,'FAFB',self.Glomerulus_to_Cluster[G],KC_class,connection_num])
                    tmp.append(connection_num)
                    

                PNindex_list = [self.PNid_list.index(PNid) for PNid in self.Glomerulus_to_PNid[G]]
                connections = self.PN_bouton_KC_claw_connection_table[PNindex_list]
                connections[connections>0] = 1
                connection_num = np.sum(connections)
                result_collection.append([G,'FlyEM',self.Glomerulus_to_Cluster[G], 'ALL', connection_num])
                tmp.append(connection_num)
                for KC_class in KC_class_list:
                    KCindex_list = [self.KCid_list.index(KCid) for KCid in self.Subtype_to_KCid[KC_class]]
                    connections_sub = connections[:,KCindex_list]
                    connection_num = np.sum(connections_sub)
                    result_collection.append([G,'FlyEM',self.Glomerulus_to_Cluster[G],KC_class,connection_num])
                    tmp.append(connection_num)
                
                for KC_class_index in range(4):
                    KC_class = (["ALL"] + KC_class_list)[KC_class_index]
                    ratio_collection.append([G,self.Glomerulus_to_Cluster[G],KC_class,tmp[KC_class_index]/tmp[KC_class_index+4]])
        color_dict = {1:'red',2:"gold",3:"deepskyblue",4:'purple'}
        
        ratio_collection = Df(data=ratio_collection,columns=['Glomerulus','Cluster','KC class','Ratio'])
        ratio_collection.to_excel('G_KC_connectivity_FAFB_FlyEM_bouton_claw_ratio.xlsx')
        ax = sn.barplot(data=ratio_collection,x="Glomerulus",y="Ratio",hue='KC class')
        for tick_label in ax.get_xticklabels():
            text = tick_label.get_text()
            if text in color_dict:
                tick_label.set_color(color_dict[self.Glomerulus_to_Cluster[text]])
        plt.xticks(rotation=90)
        plt.savefig(f"Compare_G_KC_connectivity_FAFB_FlyEM_bouton_claw_ratio.png")
        plt.savefig(f"Compare_G_KC_connectivity_FAFB_FlyEM_bouton_claw_ratio.svg")
        plt.close()
        ax = sn.boxplot(data=ratio_collection, x='Cluster',y='Ratio',hue='KC class')
        plt.savefig(f"Compare_Cluster_KC_connectivity_FAFB_FlyEM_bouton_claw_ratio.png")
        plt.savefig(f"Compare_Cluster_KC_connectivity_FAFB_FlyEM_bouton_claw_ratio.svg")
        plt.close()
        


        result_collection = Df(data=result_collection,columns=['Glomerulus','Dataset','Cluster','KC class','Connections'])
        result_collection.to_excel(f"Compare_G_KC_connectivity_FAFB_FlyEM_bouton_claw.xlsx")
        ptr_G_list = result_collection['Glomerulus'].unique()
        for KC_class in KC_class_list+['ALL']:
            mask = result_collection['KC class'] == KC_class            
            ax = sn.barplot(data=result_collection[mask],x="Glomerulus",y="Connections",hue='Dataset')
            plt.title(KC_class)
            for tick_label in ax.get_xticklabels():
                text = tick_label.get_text()
                if text in color_dict:
                    tick_label.set_color(color_dict[self.Glomerulus_to_Cluster[text]])
            plt.xticks(rotation=90)
            plt.savefig(f"Compare_G_KC_connectivity_FAFB_FlyEM_bouton_claw_{KC_class}.png")
            plt.savefig(f"Compare_G_KC_connectivity_FAFB_FlyEM_bouton_claw_{KC_class}.svg")
            plt.close()
            coordinate_collection = []
            color_list = []
            for G in ptr_G_list:
                mask_G = result_collection['Glomerulus'] == G
                coordinate = []
                for dataset in ["FAFB",'FlyEM']:
                    mask_d = result_collection['Dataset'] == dataset
                    coordinate.append(result_collection[mask & mask_G & mask_d]['Connections'].values[0])
                    print(coordinate)
                coordinate_collection.append(coordinate)
                color_list.append(color_dict[self.Glomerulus_to_Cluster_FAFB[G]])
            ##
            coordinate_collection = np.array(coordinate_collection)
            correlation_matrix = np.corrcoef(coordinate_collection[:,0], coordinate_collection[:,1])
            correlation_xy = correlation_matrix[0,1]
            # Linear regression
            x = coordinate_collection[:,0]
            y = coordinate_collection[:,1]
            a, b = np.polyfit(x, y, 1)

            # Create a sequence of 100 numbers from x min to x max
            x_reg = np.linspace(x.min(), x.max(), 100)
            y_reg = a * x_reg + b
            fig, ax = plt.subplots(ncols=1,nrows=1)
            plt.scatter(coordinate_collection[:,0], coordinate_collection[:,1],color=color_list)
            plt.plot([0],[0],color=color_dict[1],label='1')
            plt.plot([0],[0],color=color_dict[2],label='2')
            plt.plot([0],[0],color=color_dict[3],label='3')

            plt.plot(x_reg, y_reg, color='black')
            plt.text(10,y.max()/2,f'y = {a:.2f}x + {b:.2f}',fontdict={'fontsize':18})
            x_reg = np.linspace(x.min(), x.max(), 100)
            y_reg = (1/0.78) * x_reg + 0
            plt.plot(x_reg, y_reg, color='grey')
            plt.text(x.max()/2,30,f'y = {len(self.KCid_list)/len(self.KCid_list_FAFB):.2f}x',fontdict={'fontsize':18,'color':"grey"})

            # Adding a legend
            legend = plt.legend()
            # Setting the title of the legend
            legend.set_title('PN cluster')
            ax.set_ylabel("Connection number (hemibrain)", fontsize=20)
            ax.set_xlabel(f"Connection number (FAFB)", fontsize=20)
            # ax.set_xticks([0, 100, 200, 300, 400])
            plt.xticks(fontsize=18)
            # ax.set_yticks([0, 250, 500])
            plt.yticks(fontsize=18)
            w = 1.5
            ax.spines['bottom'].set_linewidth(w)  # X-axis
            ax.spines['left'].set_linewidth(w)  # Y-axis
            ax.spines['top'].set_linewidth(w)  # X-axis
            ax.spines['right'].set_linewidth(w)  # Y-axis
            plt.title(f'G-KC connections, {KC_class}',fontdict={'fontsize':20})
            # plt.text(x=1,y=1,str=f'r = {correlation_xy:.2f}')
            # plt.grid(True)
            plt.tight_layout()

            plt.savefig(f"Scatter G_KC connectivity FAFB_FlyEM_{KC_class}.png")
            plt.savefig(f"Scatter G_KC connectivity FAFB_FlyEM_{KC_class}.svg")
            plt.close()

    def plot_bouton_claw_connectivity(self):
        KC_class_list = ['KCg',"KCa'b'",'KCab']
        connection_table = self.PN_bouton_KC_claw_connection_table
        connection_table = self.transform_PN_KC_connection_to_G_KC_connection(connection_table)
        G_KC_class_connection_table = np.zeros((len(self.G_list),len(KC_class_list)))
        for G_index in range(len(self.G_list)):
            for KCindex in range(len(self.KCid_list)):
                G_KC_class_connection_table[G_index][KC_class_list.index(self.KCid_to_Subtype[self.KCid_list[KCindex]])] += connection_table[G_index][KCindex]
        KC_class_G_connection_table = G_KC_class_connection_table.transpose()
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9, 1.5 * 3))
        for classification_index in range(len(KC_class_list)):
            ax = axes[classification_index]
            plt.sca(ax)
            height = KC_class_G_connection_table[classification_index]
            ax.bar(x=[i for i in range(len(self.G_list))], height=height, color='k')
            # plt.axhline(y=2,linestyle='--', color='gray')
            # plt.axhline(y=-2,linestyle='--',color='gray')

            xticklabel = self.G_list
            if classification_index == 2:
                plt.xticks([i for i in range(len(self.G_list))],xticklabel, rotation=90)
                for i, label in enumerate(ax.get_xticklabels()):
                    if self.Glomerulus_to_Cluster[self.G_list[i]]==1:  # Color every other label red
                        label.set_color('red')
                    elif self.Glomerulus_to_Cluster[self.G_list[i]]==2:  # Color every other label red
                        label.set_color('gold')
                    elif self.Glomerulus_to_Cluster[self.G_list[i]]==3:  # Color every other label red
                        label.set_color('deepskyblue')
            else:
                plt.xticks([])
                        
        plt.savefig(f'Bouton_claw_subclass_connectivity.png',dpi=500)
        plt.savefig(f'Bouton_claw_subclass_connectivity.png.svg',format='svg')
        plt.close()
        result = Df(data=G_KC_class_connection_table,columns=["KCg", "KCa'b'","KCab"], index=self.G_list)
        result.to_csv(f"Bouton_claw_subclass_connectivity.csv")
        height = np.sum(KC_class_G_connection_table,axis=0)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 1.5))
        ax.bar(x=[i for i in range(len(self.G_list))], height=height, color='k')
        plt.xticks([i for i in range(len(self.G_list))], xticklabel, rotation=90)
        for i, label in enumerate(ax.get_xticklabels()):
            if self.Glomerulus_to_Cluster[self.G_list[i]]==1:  # Color every other label red
                label.set_color('red')
            elif self.Glomerulus_to_Cluster[self.G_list[i]]==2:  # Color every other label red
                label.set_color('gold')
            elif self.Glomerulus_to_Cluster[self.G_list[i]]==3:  # Color every other label red
                label.set_color('deepskyblue')
        plt.tight_layout()
        plt.savefig(f'Bouton_claw_connectivity.png',dpi=500)
        plt.savefig(f'Bouton_claw_connectivity.png.svg',format='svg')
        plt.close()
        
        return 

    def Calculate_Cluster_to_KC_specificity(self, dataset='FlyEM'):
        Cluster_KC_class_connection_num_dict = {(i, KC_class):0 for i in self.Cluster_to_Glomerulus for KC_class in self.Subtype_to_KCid}
        Glomerulus_KC_class_connection_num_dict = {(G, KC_class):0 for G in self.G_list for KC_class in self.Subtype_to_KCid}

        if dataset=='FAFB':
            print(self.G_list)
            KC_class_list = ['KCg',"KCa'b'",'KCab','Other']
            self.KCid_list = self.KCid_list_FAFB
            self.PNid_list = self.PNid_list_FAFB
            self.original_pre_to_post_weight = self.connection_matrix_FAFB
            self.Subtype_to_KCid = self.Subtype_to_KCid_FAFB
            self.KCid_to_Subtype = self.KCid_to_Subtype_FAFB
            self.PNid_to_Glomerulus = self.PNid_to_Glomerulus_FAFB
            self.Glomerulus_to_Cluster['VM4'] = self.Glomerulus_to_Cluster['lvVM4']
            self.PNid_to_Cluster = {PNid:self.Glomerulus_to_Cluster[self.PNid_to_Glomerulus[PNid]] \
                                    for PNid in self.PNid_list if self.PNid_to_Glomerulus[PNid] in self.Glomerulus_to_Cluster}
            self.Glomerulus_to_PNid = self.Glomerulus_to_PNid_FAFB
            # self.Glomerulus_to_PNid['VM4'] = self.Glomerulus_to_PNid['lvVM4']
            self.G_list = self.G_list_FAFB
            G_list = []
            G_index_list = []
            for Gindex, G in enumerate(self.G_list):
                if self.Glomerulus_to_Cluster[G] != 4:
                    G_list.append(G)
                    G_index_list.append(Gindex)
                    print(self.Glomerulus_to_Cluster[G])
            KC_list = []
            KC_index_list = []
            for KCindex, KCid in enumerate(self.KCid_list):
                if self.KCid_to_Subtype[KCid] == 'Other':
                    continue
                else:
                    KC_index_list.append(KCindex)

            self.pre_number = len(self.PNid_list_FAFB)
            self.post_number = len(self.KCid_list_FAFB)
            self.KCid_claw_num_dict = self.KCid_claw_num_dict_FAFB
            connection_table = self.connection_matrix_FAFB
            self.Glomerulus_to_Cluster = self.Glomerulus_to_Cluster_FAFB
            print(self.KCid_claw_num_list_FAFB)
        elif dataset == "FlyEM":
            KC_class_list = ['KCg',"KCa'b'",'KCab']
            connection_table = self.PN_bouton_KC_claw_connection_table
        KC_class_list = ['KCg',"KCa'b'",'KCab']
        for PNindex, PNid in enumerate(self.PNid_list):
            for KCindex, KCid in enumerate(self.KCid_list):
                if connection_table[PNindex][KCindex] > 0:
                    if 'VP' in  self.PNid_to_Glomerulus[PNid]:
                        continue
                    if "Other" in self.KCid_to_Subtype[KCid]:
                        continue
                    Cluster_KC_class_connection_num_dict[(self.PNid_to_Cluster[PNid],self.KCid_to_Subtype[KCid])] += 1
                    Glomerulus_KC_class_connection_num_dict[(self.PNid_to_Glomerulus[PNid],self.KCid_to_Subtype[KCid])] += 1
        Specificity_result = []
        Specificity_G_result = []
        Specificity_G_result_adjusted = []
        for cluster in self.Cluster_to_Glomerulus:
            tmp = [Cluster_KC_class_connection_num_dict[(cluster, KC_class)]/len(self.Subtype_to_KCid[KC_class]) for KC_class in KC_class_list]
            specificity = np.max(tmp)/np.sum(tmp)
            Specificity_result.append([cluster, specificity])
            for G in self.Cluster_to_Glomerulus[cluster]:
                tmp = [Glomerulus_KC_class_connection_num_dict[(G, KC_class)] for KC_class in KC_class_list]
                specificity = np.max(tmp)/np.sum(tmp)
                Specificity_G_result.append([cluster, G, specificity])
                tmp = [Glomerulus_KC_class_connection_num_dict[(G, KC_class)]/len(self.Subtype_to_KCid[KC_class]) for KC_class in KC_class_list]
                specificity = np.max(tmp)/np.sum(tmp)
                Specificity_G_result_adjusted.append([cluster, G, specificity])
                

        data = Df(data=Specificity_G_result, columns=['Cluster', "Glomerulus",'Specificity'])
        data.to_excel(f"Cluster_KC_specificity_{dataset}.xlsx")
        sn.boxplot(data=data,x='Cluster', y='Specificity', palette=['r','gold','deepskyblue'])
        ax = plt.gca()
        ax.set_ylabel("Specificity", fontsize=20)
        ax.set_xlabel(f"Cluster", fontsize=20)
        # ax.set_xticks([0, 100, 200, 300, 400])
        plt.xticks(fontsize=18)
        # ax.set_yticks([0, 250, 500])
        plt.yticks([0,0.2,0.4,0.6,0.8,1.0,1.2],fontsize=18)
        w = 1.5
        ax.spines['bottom'].set_linewidth(w)  # X-axis
        ax.spines['left'].set_linewidth(w)  # Y-axis
        ax.spines['top'].set_linewidth(w)  # X-axis
        ax.spines['right'].set_linewidth(w)  # Y-axis
        plt.tight_layout()
        plt.savefig(f"Cluster_KC_specificity_{dataset}.png",dpi=500)
        plt.savefig(f"Cluster_KC_specificity_{dataset}.svg")
        plt.close()

        data = Df(data=Specificity_G_result_adjusted, columns=['Cluster', "Glomerulus",'Specificity'])
        data.to_excel(f"Cluster_KC_specificity_adjusted_{dataset}.xlsx")
        sn.boxplot(data=data,x='Cluster', y='Specificity', palette=['r','gold','deepskyblue'])
        ax = plt.gca()
        ax.set_ylabel("Specificity", fontsize=20)
        ax.set_xlabel(f"Cluster", fontsize=20)
        # ax.set_xticks([0, 100, 200, 300, 400])
        plt.xticks(fontsize=18)
        # ax.set_yticks([0, 250, 500])
        plt.yticks([0,0.2,0.4,0.6,0.8,1.0,1.2],fontsize=18)
        w = 1.5
        ax.spines['bottom'].set_linewidth(w)  # X-axis
        ax.spines['left'].set_linewidth(w)  # Y-axis
        ax.spines['top'].set_linewidth(w)  # X-axis
        ax.spines['right'].set_linewidth(w)  # Y-axis
        plt.tight_layout()
        plt.savefig(f"Cluster_KC_specificity_adjusted_{dataset}.png",dpi=500)
        plt.savefig(f"Cluster_KC_specificity_adjusted_{dataset}.svg")
        plt.close()

        ## for random networks
        Specificity_result_adjusted_random_collection = []
        for shuffled_index in range(30):
            Cluster_KC_class_connection_num_dict = {(i, KC_class):0 for i in self.Cluster_to_Glomerulus for KC_class in self.Subtype_to_KCid}
            Glomerulus_KC_class_connection_num_dict = {(G, KC_class):0 for G in self.G_list for KC_class in self.Subtype_to_KCid}
            connection_table = self.shuffle_connection_table(shuffle_times=1, seed=shuffled_index, network_number=1, shuffle_ratio=1)
            Specificity_result = []
            Specificity_result_adjusted = []
        
            for PNindex, PNid in enumerate(self.PNid_list):
                for KCindex, KCid in enumerate(self.KCid_list):
                    if connection_table[PNindex][KCindex] > 0:
                        if 'VP' in  self.PNid_to_Glomerulus[PNid]:
                            continue
                        if "Other" in self.KCid_to_Subtype[KCid]:
                            continue
                        Cluster_KC_class_connection_num_dict[(self.PNid_to_Cluster[PNid],self.KCid_to_Subtype[KCid])] += 1
                        Glomerulus_KC_class_connection_num_dict[(self.PNid_to_Glomerulus[PNid],self.KCid_to_Subtype[KCid])] += 1
            Specificity_result = []
            Specificity_G_result = []
            Specificity_G_result_adjusted = []
            for cluster in self.Cluster_to_Glomerulus:
                tmp = [Cluster_KC_class_connection_num_dict[(cluster, KC_class)]/len(self.Subtype_to_KCid[KC_class]) for KC_class in KC_class_list]
                specificity = np.max(tmp)/np.sum(tmp)
                Specificity_result.append([cluster, specificity])
                for G in self.Cluster_to_Glomerulus[cluster]:
                    if dataset == "FAFB":
                        if G == 'lvVM4':
                            continue
                    tmp = [Glomerulus_KC_class_connection_num_dict[(G, KC_class)] for KC_class in KC_class_list]
                    specificity = np.max(tmp)/np.sum(tmp)
                    Specificity_G_result.append([cluster, G, specificity])
                    tmp = [Glomerulus_KC_class_connection_num_dict[(G, KC_class)]/len(self.Subtype_to_KCid[KC_class]) for KC_class in KC_class_list]
                    specificity = np.max(tmp)/np.sum(tmp)
                    Specificity_G_result_adjusted.append([cluster, G, specificity])

            result = Df(data=Specificity_G_result_adjusted, columns=['Cluster', G,'specificity'])
            for cluster in self.Cluster_to_Glomerulus:
                mask = result['Cluster'] == cluster
                Specificity_result_adjusted_random_collection.append([cluster,result[mask]['specificity'].mean()])
        Specificity_result_adjusted_random_collection = Df(data=Specificity_result_adjusted_random_collection, columns=['Cluster','specificity'])
        Specificity_result_adjusted_random_collection.to_excel(f"Cluster_KC_specificity_adjusted_{dataset}_shuffled_mean.xlsx")
        y = []
        y_err = []
        observed_y = []
        z_score = []
        for cluster_index, cluster in enumerate(self.Cluster_to_Glomerulus):
            print('check order',cluster)
            mask = Specificity_result_adjusted_random_collection['Cluster'] == cluster
            y.append(Specificity_result_adjusted_random_collection[mask]['specificity'].mean())
            y_err.append(Specificity_result_adjusted_random_collection[mask]['specificity'].std())
            mask = data['Cluster'] == cluster
            observed_y.append(data[mask]['Specificity'].mean())
            z_score.append((observed_y[cluster_index]-y[cluster_index])/y_err[cluster_index])

        plt.errorbar(x=list(self.Cluster_to_Glomerulus.keys()),y=y,yerr=y_err, fmt='o',markersize=1,capsize=3,label='Shuffled networks')
        if dataset=='FlyEM':
            label='hemibrain'
        else:
            label=dataset
        plt.plot([1,2,3],observed_y,'.', label=label)
        ax = plt.gca()
        ax.set_ylabel("Specificity", fontsize=20)
        ax.set_xlabel(f"Cluster", fontsize=20)
        # ax.set_xticks([0, 100, 200, 300, 400])
        plt.xticks(fontsize=18)
        # ax.set_yticks([0, 250, 500])
        plt.yticks(fontsize=18)
        w = 1.5
        ax.spines['bottom'].set_linewidth(w)  # X-axis
        ax.spines['left'].set_linewidth(w)  # Y-axis
        ax.spines['top'].set_linewidth(w)  # X-axis
        ax.spines['right'].set_linewidth(w)  # Y-axis
        plt.legend(fontsize='large')
        plt.tight_layout()
        plt.savefig(f"Cluster_KC_specificity_adjusted_{dataset}_shuffled_observed.png",dpi=500)
        plt.savefig(f"Cluster_KC_specificity_adjusted_{dataset}_shuffled_observed.svg")
        plt.close()
        plt.bar(list(self.Cluster_to_Glomerulus.keys()),z_score,color='k')
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5)  # X-axis
        ax.spines['left'].set_linewidth(1.5)  # Y-axis
        ax.spines['top'].set_linewidth(1.5)  # X-axis
        ax.spines['right'].set_linewidth(1.5)  # Y-axis
        plt.xlabel('Cluster', fontdict={'fontsize': 20})
        plt.ylabel('Z score', fontdict={'fontsize': 20})
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.savefig(f"Cluster_KC_specificity_adjusted_{dataset}_shuffled_observed_zscore.png",dpi=500)
        plt.savefig(f"Cluster_KC_specificity_adjusted_{dataset}_shuffled_observed_zscore.svg")
        plt.close()

    def Calculate_KC_from_Cluster_specificity(self, dataset='FlyEM'):
        Cluster_KCid_connection_num_dict = {(cluster, KCid):0 for cluster in self.Cluster_to_Glomerulus for KCid in self.KCid_list}
        ## for observed networks
        if dataset=='FAFB':

            KC_class_list = ['KCg',"KCa'b'",'KCab','Other']
            self.KCid_list = self.KCid_list_FAFB
            self.PNid_list = self.PNid_list_FAFB
            self.original_pre_to_post_weight = self.connection_matrix_FAFB
            self.Subtype_to_KCid = self.Subtype_to_KCid_FAFB
            self.KCid_to_Subtype = self.KCid_to_Subtype_FAFB
            self.PNid_to_Glomerulus = self.PNid_to_Glomerulus_FAFB
            self.Glomerulus_to_Cluster['VM4'] = self.Glomerulus_to_Cluster['lvVM4']
            self.PNid_to_Cluster = {PNid:self.Glomerulus_to_Cluster[self.PNid_to_Glomerulus[PNid]] \
                                    for PNid in self.PNid_list if self.PNid_to_Glomerulus[PNid] in self.Glomerulus_to_Cluster}
            self.Glomerulus_to_PNid = self.Glomerulus_to_PNid_FAFB
            # self.Glomerulus_to_PNid['VM4'] = self.Glomerulus_to_PNid['lvVM4']

            self.Glomerulus_to_PNid = self.Glomerulus_to_PNid_FAFB
            self.G_list = self.G_list_FAFB
            G_list = []
            G_index_list = []
            for Gindex, G in enumerate(self.G_list):
                if self.Glomerulus_to_Cluster[G] != 4:
                    G_list.append(G)
                    G_index_list.append(Gindex)
                    print(self.Glomerulus_to_Cluster[G])
            KC_list = []
            KC_index_list = []
            for KCindex, KCid in enumerate(self.KCid_list):
                if self.KCid_to_Subtype[KCid] == 'Other':
                    continue
                else:
                    KC_index_list.append(KCindex)

            self.pre_number = len(self.PNid_list_FAFB)
            self.post_number = len(self.KCid_list_FAFB)
            self.KCid_claw_num_dict = self.KCid_claw_num_dict_FAFB
            connection_table = self.connection_matrix_FAFB
            self.Glomerulus_to_Cluster = self.Glomerulus_to_Cluster_FAFB
            print(self.KCid_claw_num_list_FAFB)
        elif dataset == "FlyEM":
            KC_class_list = ['KCg',"KCa'b'",'KCab']
            connection_table = self.PN_bouton_KC_claw_connection_table
        KC_class_list = ['KCg',"KCa'b'",'KCab']
        
        
        for PNindex, PNid in enumerate(self.PNid_list):
            for KCindex, KCid in enumerate(self.KCid_list):
                if connection_table[PNindex][KCindex] > 0:
                    if 'VP' in  self.PNid_to_Glomerulus[PNid]:
                        continue
                    if "Other" in self.KCid_to_Subtype[KCid]:
                        continue
                    Cluster_KCid_connection_num_dict[(self.PNid_to_Cluster[PNid],KCid)] += 1
                    
        Specificity_result = []
        Specificity_result_adjusted = []
        for KCindex,KCid in enumerate(self.KCid_list):
            tmp = [Cluster_KCid_connection_num_dict[(cluster, KCid)] for cluster in self.Cluster_to_Glomerulus]
            if np.sum(tmp)==0:
                ## due to claw extraction, some KC may receive no input
                continue
            specificity = np.max(tmp)/np.sum(tmp)
            Specificity_result.append([self.KCid_to_Subtype[KCid], specificity])
            tmp = [Cluster_KCid_connection_num_dict[(cluster, KCid)]/len(self.Cluster_to_PNid[cluster]) for cluster in self.Cluster_to_Glomerulus]
            specificity = np.max(tmp)/np.sum(tmp)
            Specificity_result_adjusted.append([self.KCid_to_Subtype[KCid], specificity])
        
        data = Df(data=Specificity_result, columns=['KC class','Specificity'])
        data.to_excel(f"KC_cluster_specificity_{dataset}.xlsx")
        sn.boxplot(data=data,x='KC class', y='Specificity', palette=['r','gold','deepskyblue'])
        ax = plt.gca()
        ax.set_ylabel("Specificity", fontsize=20)
        ax.set_xlabel(f"KC class", fontsize=20)
        # ax.set_xticks([0, 100, 200, 300, 400])
        plt.xticks(fontsize=18)
        # ax.set_yticks([0, 250, 500])
        plt.yticks([0,0.2,0.4,0.6,0.8,1.0,1.2],fontsize=18)
        w = 1.5
        ax.spines['bottom'].set_linewidth(w)  # X-axis
        ax.spines['left'].set_linewidth(w)  # Y-axis
        ax.spines['top'].set_linewidth(w)  # X-axis
        ax.spines['right'].set_linewidth(w)  # Y-axis
        plt.tight_layout()
        plt.savefig(f"KC_class_specificity_{dataset}.png",dpi=500)
        plt.savefig(f"KC_class_specificity_{dataset}.svg")
        plt.close()

        data = Df(data=Specificity_result_adjusted, columns=['KC class','Specificity'])
        data.to_excel(f"KC_cluster_specificity_adjusted_{dataset}.xlsx")
        sn.boxplot(data=data,x='KC class', y='Specificity', palette=['r','gold','deepskyblue'])
        ax = plt.gca()
        ax.set_ylabel("Specificity", fontsize=20)
        ax.set_xlabel(f"KC class", fontsize=20)
        # ax.set_xticks([0, 100, 200, 300, 400])
        plt.xticks(fontsize=18)
        # ax.set_yticks([0, 250, 500])
        plt.yticks([0,0.2,0.4,0.6,0.8,1.0,1.2],fontsize=18)
        w = 1.5
        ax.spines['bottom'].set_linewidth(w)  # X-axis
        ax.spines['left'].set_linewidth(w)  # Y-axis
        ax.spines['top'].set_linewidth(w)  # X-axis
        ax.spines['right'].set_linewidth(w)  # Y-axis
        plt.tight_layout()
        plt.savefig(f"KC_cluster_specificity_adjusted_{dataset}.png",dpi=500)
        plt.savefig(f"KC_cluster_specificity_adjusted_{dataset}.svg")
        plt.close()

        ## for random networks
        Specificity_result_adjusted_random_collection = []
        for shuffled_index in range(30):
            Cluster_KCid_connection_num_dict = {(cluster, KCid):0 for cluster in self.Cluster_to_Glomerulus for KCid in self.KCid_list}
            connection_table = self.shuffle_connection_table(shuffle_times=1, seed=shuffled_index, network_number=1, shuffle_ratio=1)
            Specificity_result = []
            Specificity_result_adjusted = []
        
            for PNindex, PNid in enumerate(self.PNid_list):
                for KCindex, KCid in enumerate(self.KCid_list):
                    if connection_table[PNindex][KCindex] > 0:
                        if 'VP' in  self.PNid_to_Glomerulus[PNid]:
                            continue
                        if "Other" in self.KCid_to_Subtype[KCid]:
                            continue
                        Cluster_KCid_connection_num_dict[(self.PNid_to_Cluster[PNid],KCid)] += 1

            for KCindex,KCid in enumerate(self.KCid_list):
                tmp = [Cluster_KCid_connection_num_dict[(cluster, KCid)] for cluster in self.Cluster_to_Glomerulus]
                if np.sum(tmp)==0:
                    ## due to claw extraction, some KC may receive no input
                    continue
                specificity = np.max(tmp)/np.sum(tmp)
                Specificity_result.append([self.KCid_to_Subtype[KCid], specificity])
                tmp = [Cluster_KCid_connection_num_dict[(cluster, KCid)]/len(self.Cluster_to_PNid[cluster]) for cluster in self.Cluster_to_Glomerulus]
                specificity = np.max(tmp)/np.sum(tmp)
                Specificity_result_adjusted.append([self.KCid_to_Subtype[KCid], specificity])
            result = Df(data=Specificity_result_adjusted, columns=['KC class','specificity'])
            for KC_class in KC_class_list:
                mask = result['KC class'] == KC_class
                Specificity_result_adjusted_random_collection.append([KC_class,result[mask]['specificity'].mean()])
        Specificity_result_adjusted_random_collection = Df(data=Specificity_result_adjusted_random_collection, columns=['KC class','specificity'])
        Specificity_result_adjusted_random_collection.to_excel(f"KC_cluster_specificity_adjusted_{dataset}_shuffled_mean.xlsx")
        y = []
        y_err = []
        observed_y = []
        z_score = []
        for KC_class in KC_class_list:
            mask = Specificity_result_adjusted_random_collection['KC class'] == KC_class
            y.append(Specificity_result_adjusted_random_collection[mask]['specificity'].mean())
            y_err.append(Specificity_result_adjusted_random_collection[mask]['specificity'].std())
            mask = data['KC class'] == KC_class
            observed_y.append(data[mask]['Specificity'].mean())
            z_score.append((observed_y[-1]-y[-1])/y_err[-1])

        plt.errorbar(x=KC_class_list,y=y,yerr=y_err, fmt='o',markersize=1,capsize=3, label='Shuffled networks')
        if dataset=='FlyEM':
            label='hemibrain'
        else:
            label=dataset
        plt.plot([0,1,2],observed_y,'.', label=label)
        ax = plt.gca()
        ax.set_ylabel("Specificity", fontsize=20)
        ax.set_xlabel(f"KC class", fontsize=20)
        # ax.set_xticks([0, 100, 200, 300, 400])
        plt.xticks(fontsize=18)
        # ax.set_yticks([0, 250, 500])
        plt.yticks(fontsize=18)
        w = 1.5
        ax.spines['bottom'].set_linewidth(w)  # X-axis
        ax.spines['left'].set_linewidth(w)  # Y-axis
        ax.spines['top'].set_linewidth(w)  # X-axis
        ax.spines['right'].set_linewidth(w)  # Y-axis
        plt.legend(fontsize='large')
        plt.tight_layout()
        plt.savefig(f"KC_cluster_specificity_adjusted_{dataset}_shuffled_observed.png",dpi=500)
        plt.savefig(f"KC_cluster_specificity_adjusted_{dataset}_shuffled_observed.svg")
        plt.close()

        plt.bar(KC_class_list,z_score,color='k')
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5)  # X-axis
        ax.spines['left'].set_linewidth(1.5)  # Y-axis
        ax.spines['top'].set_linewidth(1.5)  # X-axis
        ax.spines['right'].set_linewidth(1.5)  # Y-axis
        plt.xlabel('KC class', fontdict={'fontsize': 20})
        plt.ylabel('Z score', fontdict={'fontsize': 20})
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.savefig(f"KC_cluster_specificity_adjusted_{dataset}_shuffled_observed_zscore.png",dpi=500)
        plt.savefig(f"KC_cluster_specificity_adjusted_{dataset}_shuffled_observed_zscore.svg")
        plt.close()

    def plot_connection_table(self, dataset='FlyEM', sampled_ratio_list=[1.0,1.0,1.0], subsampling=False,post_fix_name="",
                              claw_ratio_list=[1.0,1.0,1.0],claw_subsample=False, seed=100):
        rd.seed(seed)
        if dataset=='FAFB':
            print(self.G_list)
            KC_class_list = ['KCg',"KCa'b'",'KCab','Other']
            self.KCid_list = self.KCid_list_FAFB
            self.PNid_list = self.PNid_list_FAFB
            self.original_pre_to_post_weight = self.connection_matrix_FAFB
            self.Subtype_to_KCid = self.Subtype_to_KCid_FAFB
            self.KCid_to_Subtype = self.KCid_to_Subtype_FAFB
            self.PNid_to_Glomerulus = self.PNid_to_Glomerulus_FAFB
            self.Glomerulus_to_PNid = self.Glomerulus_to_PNid_FAFB
            self.G_list = self.G_list_FAFB
            G_list = []
            G_index_list = []
            for Gindex, G in enumerate(self.G_list):
                if self.Glomerulus_to_Cluster[G] != 4:
                    G_list.append(G)
                    G_index_list.append(Gindex)
                    print(self.Glomerulus_to_Cluster[G])
            KC_list = []
            KC_index_list = []
            for KCindex, KCid in enumerate(self.KCid_list):
                if self.KCid_to_Subtype[KCid] == 'Other':
                    continue
                else:
                    KC_index_list.append(KCindex)

            self.pre_number = len(self.PNid_list_FAFB)
            self.post_number = len(self.KCid_list_FAFB)
            self.KCid_claw_num_dict = self.KCid_claw_num_dict_FAFB
            connection_table = self.connection_matrix_FAFB
            self.Glomerulus_to_Cluster = self.Glomerulus_to_Cluster_FAFB
            # print(self.KCid_claw_num_list_FAFB)
        elif dataset == "FlyEM":
            KC_class_list = ['KCg',"KCa'b'",'KCab']
            connection_table = self.PN_bouton_KC_claw_connection_table
        if subsampling == True:
            new_KC_id_list = []
            for KC_class, sampled_ratio in zip(KC_class_list, sampled_ratio_list):
                new_KC_id_list += rd.sample(self.Subtype_to_KCid[KC_class],k=round(sampled_ratio*len(self.Subtype_to_KCid[KC_class])))
            KC_num = len(new_KC_id_list)
            self.KCid_list = new_KC_id_list
            self.post_number = len(new_KC_id_list)
            KCindex_list = [self.KCid_list.index(i) for i in self.KCid_list]
            connection_table = connection_table[:,KCindex_list]
        if claw_subsample:
            for KCindex in range(self.post_number):
                prob = claw_ratio_list[KC_class_list.index(self.KCid_to_Subtype[self.KCid_list[KCindex]])]
                for PNindex in range(connection_table.shape[0]):
                    if connection_table[PNindex][KCindex]>0 and rd.random() > prob:
                        connection_table[PNindex][KCindex] = 0

        self.original_pre_to_post_weight = connection_table
        connection_table_g = self.transform_PN_KC_connection_to_G_KC_connection(connection_table)

        col_colors = []
        color_dict = {1:'r',2:"gold",3:"deepskyblue",'KCg':'r',"KCa'b'":"gold","KCab":"deepskyblue"}
        for KC_class in ["KCg","KCa'b'","KCab"]:
            shuffled_index_list = []
            for KCid in self.Subtype_to_KCid[KC_class]:
                if KCid in self.KCid_list:
                    shuffled_index_list.append(self.KCid_list.index(KCid))
                    col_colors.append(color_dict[KC_class])

            original_index_list = copy.deepcopy(shuffled_index_list)
            rd.shuffle(shuffled_index_list)
            connection_table_g[:,original_index_list] = connection_table_g[:,shuffled_index_list]
        
        row_colors = []        
        for Cluster in [1,2,3]:
            shuffled_index_list = []
            for Gid in self.Cluster_to_Glomerulus[Cluster]:
                if dataset == 'FAFB':
                    if Gid == 'lvVM4':
                        Gid = "VM4"

                shuffled_index_list.append(self.G_list.index(Gid))
                row_colors.append(color_dict[Cluster])
            original_index_list = copy.deepcopy(shuffled_index_list)
            rd.shuffle(shuffled_index_list)
            connection_table_g[original_index_list] = connection_table_g[shuffled_index_list]
        connection_table_g[connection_table_g>0] = 1
        if dataset == 'FAFB':
            connection_table_g = connection_table_g[G_index_list]
            connection_table_g = connection_table_g[:,KC_index_list]
        sn.clustermap(data=connection_table_g,row_cluster=False,col_cluster=False,row_colors=row_colors, col_colors=col_colors, cmap='gray')
        plt.savefig(f"Original_G_KC_connection_table_{dataset}_{post_fix_name}.png")
        plt.close()

        shuffled_table = self.shuffle_connection_table(network_number=1, shuffle_times=1)
        connection_table_g = self.transform_PN_KC_connection_to_G_KC_connection(shuffled_table)

        for KC_class in ["KCg","KCa'b'","KCab"]:
            shuffled_index_list = []
            for KCid in self.Subtype_to_KCid[KC_class]:
                if KCid in self.KCid_list:
                    shuffled_index_list.append(self.KCid_list.index(KCid))
            original_index_list = copy.deepcopy(shuffled_index_list)
            rd.shuffle(shuffled_index_list)
            connection_table_g[:,original_index_list] = connection_table_g[:,shuffled_index_list]
        
        for Cluster in [1,2,3]:
            shuffled_index_list = []
            for Gid in self.Cluster_to_Glomerulus[Cluster]:
                if dataset == 'FAFB':
                    if Gid == 'lvVM4':
                        Gid = "VM4"
                shuffled_index_list.append(self.G_list.index(Gid))
            original_index_list = copy.deepcopy(shuffled_index_list)
            rd.shuffle(shuffled_index_list)
            connection_table_g[original_index_list] = connection_table_g[shuffled_index_list]
        
        connection_table_g[connection_table_g>0] = 1
        if dataset == 'FAFB':
            connection_table_g = connection_table_g[G_index_list]
            connection_table_g = connection_table_g[:,KC_index_list]
        
        sn.clustermap(data=connection_table_g,row_cluster=False,col_cluster=False,row_colors=row_colors, col_colors=col_colors,cmap='gray')
        plt.savefig(f"Original_G_KC_connection_table_shuffled_{dataset}_{post_fix_name}.png")
        plt.close()

    def output_FAFB_connection_matrix(self):
        KC_class_list = ['KCg',"KCa'b'",'KCab','Other']
        self.KCid_list = self.KCid_list_FAFB
        self.PNid_list = self.PNid_list_FAFB
        self.original_pre_to_post_weight = self.connection_matrix_FAFB
        self.Subtype_to_KCid = self.Subtype_to_KCid_FAFB
        self.KCid_to_Subtype = self.KCid_to_Subtype_FAFB
        self.PNid_to_Glomerulus = self.PNid_to_Glomerulus_FAFB
        self.Glomerulus_to_PNid = self.Glomerulus_to_PNid_FAFB
        self.G_list = self.G_list_FAFB
        self.pre_number = len(self.PNid_list_FAFB)
        self.post_number = len(self.KCid_list_FAFB)
        self.KCid_claw_num_dict = self.KCid_claw_num_dict_FAFB
        connection_table = self.connection_matrix_FAFB
        self.Glomerulus_to_Cluster = self.Glomerulus_to_Cluster_FAFB
        print(self.KCid_claw_num_list_FAFB)
        output_result = []

        connection_table_g = self.transform_PN_KC_connection_to_G_KC_connection(self.connection_matrix_FAFB)        
        col_colors = []
        color_dict = {1:'r',2:"gold",3:"deepskyblue",'KCg':'r',"KCa'b'":"gold","KCab":"deepskyblue"}
        original_index_list = []
        for KC_class in ["KCg","KCa'b'","KCab"]:
            for KCid in self.Subtype_to_KCid[KC_class]:
                if KCid in self.KCid_list:
                    original_index_list.append(self.KCid_list.index(KCid))
                    col_colors.append(color_dict[KC_class])
        connection_table_g = connection_table_g[:,original_index_list]
        new_KCid_list = []
        for i in range(len(original_index_list)):
            KCid = self.KCid_list[original_index_list[i]]
            output_result.append([i,KCid,'KC',self.KCid_to_Subtype_FAFB[KCid]])
            new_KCid_list.append(KCid)
        for i,G in enumerate(self.G_list_FAFB):
            output_result.append([i,i,'Glomerulus',G])
        Df(data=output_result, columns=['index','neuronId','NeuronType','Classification']).to_csv("FAFB_connection_profile.csv")
        Df(data=connection_table_g,columns =[self.KCid_to_Subtype_FAFB[KCid] for KCid in new_KCid_list],index=self.G_list_FAFB).to_csv("FAFB_connection_table.csv")
        sn.clustermap(data=connection_table_g.transpose(),col_cluster=False,row_cluster=False)
        plt.show()

    def construct_random_claw_model(self, shuffled_times=100, completeness=1.0, dataset='FlyEM', sampled_ratio_list = [1.0,1.0,1.0], 
                                    subsampling=False,post_fix_name='',vmax=0, claw_ratio_list=[1.0,1.0,1.0], claw_subsample=False, seed=100):
        rd.seed(seed)
        if dataset=='FAFB':
            KC_class_list = ['KCg',"KCa'b'",'KCab','Other']
            self.KCid_list = self.KCid_list_FAFB
            self.PNid_list = self.PNid_list_FAFB
            self.original_pre_to_post_weight = self.connection_matrix_FAFB
            self.Subtype_to_KCid = self.Subtype_to_KCid_FAFB
            self.KCid_to_Subtype = self.KCid_to_Subtype_FAFB
            self.PNid_to_Glomerulus = self.PNid_to_Glomerulus_FAFB
            self.Glomerulus_to_PNid = self.Glomerulus_to_PNid_FAFB
            self.G_list = self.G_list_FAFB
            self.pre_number = len(self.PNid_list_FAFB)
            self.post_number = len(self.KCid_list_FAFB)
            self.KCid_claw_num_dict = self.KCid_claw_num_dict_FAFB
            connection_table = self.connection_matrix_FAFB
            self.Glomerulus_to_Cluster = self.Glomerulus_to_Cluster_FAFB
            print(self.KCid_claw_num_list_FAFB)
        elif dataset == "FlyEM":
            KC_class_list = ['KCg',"KCa'b'",'KCab']
            connection_table = self.PN_bouton_KC_claw_connection_table
        if completeness < 1 and completeness >= 0:
            KC_num = round(completeness*len(self.KCid_list))
            self.post_number = KC_num
            sampled_list = sorted(rd.sample([i for i in range(len(self.KCid_list))],KC_num))
            self.KCid_list = [self.KCid_list[i] for i in sampled_list]
            connection_table = connection_table[:,sampled_list]
        elif subsampling == True:
            new_KC_id_list = []
            for KC_class, sampled_ratio in zip(KC_class_list, sampled_ratio_list):
                new_KC_id_list += rd.sample(self.Subtype_to_KCid[KC_class],k=round(sampled_ratio*len(self.Subtype_to_KCid[KC_class])))
            KC_num = len(new_KC_id_list)
            self.KCid_list = new_KC_id_list
            self.post_number = len(new_KC_id_list)
            KCindex_list = [self.KCid_list.index(i) for i in self.KCid_list]
            connection_table = connection_table[:,KCindex_list]
        if claw_subsample:
            for KCindex in range(self.post_number):
                prob = claw_ratio_list[KC_class_list.index(self.KCid_to_Subtype[self.KCid_list[KCindex]])]
                for PNindex in range(connection_table.shape[0]):
                    if connection_table[PNindex][KCindex]>0 and rd.random() > prob:
                        connection_table[PNindex][KCindex] = 0

        self.original_pre_to_post_weight = connection_table
        original_G_conditional = self.perform_condition_input_analysis(connection_table)
        connection_table_g = self.transform_PN_KC_connection_to_G_KC_connection(connection_table)
        original_G_correlation = np.corrcoef(connection_table_g,rowvar=True)
        original_connectivity = self.get_connectivity(connection_table, dataset=dataset).transpose().ravel()
        connectivity_collection = []
        G_correlation_collection = []
        G_conditional_collection = []
        for shuffled_index in range(shuffled_times):
            shuffled_table = self.shuffle_connection_table(seed=shuffled_index,network_number=1, shuffle_times=1)
            # print('SHAPE',shuffled_table.shape, KC_num)
            if shuffled_table is None:
                continue
            else:
                conditional_matrix = self.perform_condition_input_analysis(shuffled_table)
                G_conditional_collection.append(conditional_matrix.ravel())
                shuffled_table_g = self.transform_PN_KC_connection_to_G_KC_connection(connection_table=shuffled_table)
                G_correlation = np.corrcoef(shuffled_table_g,rowvar=True)
                G_correlation_collection.append(G_correlation.ravel())
                print(shuffled_table_g.shape)
                connectivity = np.zeros((len(self.G_list),len(KC_class_list)))
                for G_index in range(len(self.G_list)):
                    for KC_index in range(len(self.KCid_list)):
                        KC_class_index = KC_class_list.index(self.KCid_to_Subtype[self.KCid_list[KC_index]])
                        connectivity[G_index][KC_class_index] += int(shuffled_table_g[G_index][KC_index] >0)
                connectivity_collection.append(connectivity.transpose().ravel())

        G_conditional_collection = np.array(G_conditional_collection)
        z_score_G_condition = (original_G_conditional.ravel()-np.average(G_conditional_collection,axis=0))/np.std(G_conditional_collection,axis=0)
        z_score_G_condition = np.reshape(z_score_G_condition,(len(self.G_list),len(self.G_list)))
        z_score_G_condition[np.isinf(z_score_G_condition)] = 0
        z_score_G_condition[np.isnan(z_score_G_condition)] = 0
        sn.heatmap(data=np.reshape(np.average(G_conditional_collection,axis=0),(len(self.G_list),len(self.G_list))))
        plt.savefig(f"G_average_random_claw_{completeness}_{dataset}_{post_fix_name}.png")
        plt.close()
        sn.heatmap(data=np.reshape(np.std(G_conditional_collection,axis=0),(len(self.G_list),len(self.G_list))))
        plt.savefig(f"G_STD_{completeness}_{dataset}_{post_fix_name}.png")
        plt.close()
        sn.heatmap(data=z_score_G_condition)
        plt.savefig(f"G_zscore_{completeness}_{dataset}_{post_fix_name}.png")
        plt.close()
        if vmax:
            c_bar_scale = vmax
        else:
            c_bar_scale = max(np.max(z_score_G_condition),abs(np.min(z_score_G_condition)))
        g = sn.clustermap(data=z_score_G_condition,xticklabels=self.G_list,yticklabels=self.G_list,cmap='bwr', vmax=c_bar_scale, vmin = -c_bar_scale)
        color_dict = {1:'red',2:"gold",3:"deepskyblue",4:'purple'}
        for tick_label in g.ax_heatmap.axes.get_xticklabels():
            tick_text = tick_label.get_text()
            # tick_label.set_color(color_dict[self.Glomerulus_to_Cluster[self.G_list[int(tick_text)]]])
            tick_label.set_color(color_dict[self.Glomerulus_to_Cluster[tick_text]])
        for tick_label in g.ax_heatmap.axes.get_yticklabels():
            tick_text = tick_label.get_text()
            tick_label.set_color(color_dict[self.Glomerulus_to_Cluster[tick_text]])
        plt.savefig(f'Random_claw_model_z_score_{completeness}_G_condition_{dataset}_{post_fix_name}.png',dpi=500)
        plt.close()
        
        g = sn.clustermap(data=z_score_G_condition,xticklabels=self.G_list,yticklabels=self.G_list,cmap='bwr', vmax=c_bar_scale, vmin = -c_bar_scale, method='complete')
        color_dict = {1:'red',2:"gold",3:"deepskyblue",4:'purple'}
        for tick_label in g.ax_heatmap.axes.get_xticklabels():
            tick_text = tick_label.get_text()
            # tick_label.set_color(color_dict[self.Glomerulus_to_Cluster[self.G_list[int(tick_text)]]])
            tick_label.set_color(color_dict[self.Glomerulus_to_Cluster[tick_text]])
        for tick_label in g.ax_heatmap.axes.get_yticklabels():
            tick_text = tick_label.get_text()
            tick_label.set_color(color_dict[self.Glomerulus_to_Cluster[tick_text]])
        plt.savefig(f'Random_claw_model_z_score_{completeness}_G_condition_{dataset}_{post_fix_name}_complete.png',dpi=500)
        plt.close()
        

        G_sequence_Zheng = pd.read_excel('Zheng_G_matrix_order_v1.xlsx')['Glomerulus'].values.tolist()
        re_order_list = [self.G_list.index(G) for G in G_sequence_Zheng if G in self.G_list]
        re_order_list += [i for i in range(len(self.G_list)) if i not in re_order_list]
        G_reorder_list = [self.G_list[i] for i in re_order_list]
        z_score_G_condition_reorder = z_score_G_condition[re_order_list]
        z_score_G_condition_reorder = z_score_G_condition_reorder[:,re_order_list]
        g = sn.clustermap(data=z_score_G_condition_reorder,xticklabels=G_reorder_list,yticklabels=G_reorder_list,cmap='bwr', vmax=c_bar_scale, vmin = -c_bar_scale, col_cluster=False, row_cluster=False)
        for tick_label in g.ax_heatmap.axes.get_xticklabels():
            tick_text = tick_label.get_text()
            # tick_label.set_color(color_dict[self.Glomerulus_to_Cluster[self.G_list[int(tick_text)]]])
            tick_label.set_color(color_dict[self.Glomerulus_to_Cluster[tick_text]])
        for tick_label in g.ax_heatmap.axes.get_yticklabels():
            tick_text = tick_label.get_text()
            tick_label.set_color(color_dict[self.Glomerulus_to_Cluster[tick_text]])
        plt.savefig(f'Random_claw_model_z_score_{completeness}_G_condition_{dataset}_{post_fix_name}_reorder.png',dpi=500)
        plt.close()
        

        connectivity_collection = np.array(connectivity_collection)
        z_score = (original_connectivity - np.average(connectivity_collection,axis=0))/np.std(connectivity_collection,axis=0)
        z_score = z_score.reshape((len(KC_class_list),len(self.G_list)))
        
        fig, axes = plt.subplots(nrows=len(KC_class_list), ncols=1, figsize=(9, 1.5 * len(KC_class_list)))
        for classification_index in range(len(KC_class_list)):
            ax = axes[classification_index]
            plt.sca(ax)
            height = z_score[classification_index]
            ax.bar(x=[i for i in range(len(self.G_list))], height=height, color='k')
            plt.axhline(y=2,linestyle='--', color='gray')
            plt.axhline(y=-2,linestyle='--',color='gray')

            xticklabel = self.G_list
            if classification_index == len(KC_class_list)-1:
                plt.xticks([i for i in range(len(self.G_list))],xticklabel, rotation=90)
                for i, label in enumerate(ax.get_xticklabels()):
                    if self.Glomerulus_to_Cluster[self.G_list[i]]==1:  # Color every other label red
                        label.set_color('red')
                    elif self.Glomerulus_to_Cluster[self.G_list[i]]==2:  # Color every other label red
                        label.set_color('gold')
                    elif self.Glomerulus_to_Cluster[self.G_list[i]]==3:  # Color every other label red
                        label.set_color('deepskyblue')
                    elif self.Glomerulus_to_Cluster[self.G_list[i]]==4:
                        label.set_color('purple')

            else:
                plt.xticks([])
                        
        plt.savefig(f'Random_claw_model_z_score_{completeness}_{dataset}_{post_fix_name}.png',dpi=500)
        plt.savefig(f'Random_claw_z_score_{completeness}_{dataset}_{post_fix_name}.svg',format='svg')
        plt.close()
        result = Df(data=z_score.transpose,columns=["KCg", "KCa'b'","KCab"], index=self.G_list)
        result.to_csv(f"Z_score_table_random_claw_model_{dataset}_{completeness}_{post_fix_name}.csv")
        return

    def construct_random_bouton_model(self, shuffled_times=10000, completeness=1, dataset='FlyEM'):
        if dataset=='FAFB':
            KC_class_list = ['KCg',"KCa'b'",'KCab','Other']
            self.KCid_list = self.KCid_list_FAFB
            self.PNid_list = self.PNid_list_FAFB
            self.original_pre_to_post_weight = self.connection_matrix_FAFB
            self.Subtype_to_KCid = self.Subtype_to_KCid_FAFB
            self.KCid_to_Subtype = self.KCid_to_Subtype_FAFB
            self.PNid_to_Glomerulus = self.PNid_to_Glomerulus_FAFB
            self.Glomerulus_to_PNid = self.Glomerulus_to_PNid_FAFB
            self.G_list = self.G_list_FAFB
            self.pre_number = len(self.PNid_list_FAFB)
            self.post_number = len(self.KCid_list_FAFB)
            self.KCid_claw_num_dict = self.KCid_claw_num_dict_FAFB
            connection_table = self.connection_matrix_FAFB
            self.Glomerulus_to_Cluster = self.Glomerulus_to_Cluster_FAFB
            print(self.KCid_claw_num_list_FAFB)
        elif dataset == "FlyEM":
            KC_class_list = ['KCg',"KCa'b'",'KCab']
            connection_table = self.PN_bouton_KC_claw_connection_table
        # print(self.KCid_list)
        # print(self.KCid_to_Subtype)

        if completeness < 1 and completeness > 0:
            KC_num = round(completeness*len(self.KCid_list))
            self.post_number = KC_num
            sampled_list = [i for i in range(len(self.KCid_list))]
            rd.shuffle(sampled_list)
            sampled_list = sampled_list[:KC_num]
            self.KCid_list = [self.KCid_list[sampled_list[i]] for i in range(KC_num)]
            print([self.KCid_to_Subtype[i] for i in self.KCid_list[:15]])
            connection_table = connection_table[:,sampled_list]
        self.original_pre_to_post_weight = connection_table
        original_G_conditional = self.perform_condition_input_analysis(connection_table)
        connection_table_g = self.transform_PN_KC_connection_to_G_KC_connection(connection_table)
        original_G_correlation = np.corrcoef(connection_table_g,rowvar=True)
        original_connectivity = self.get_connectivity(connection_table,dataset=dataset).transpose().ravel()
        connectivity_collection = []
        G_correlation_collection = []
        G_conditional_collection = []
        G_bouton_num_dict = {}
        if dataset == 'FlyEM':
            for PNid in self.PNid_bouton_dict:
                G = self.PNid_to_Glomerulus[PNid]
                if G not in G_bouton_num_dict:
                    G_bouton_num_dict[G] = 0
                G_bouton_num_dict[G] += len(self.PNid_bouton_dict[PNid])
        elif dataset == 'FAFB':
            G_bouton_num_dict = self.G_bouton_num_dict_FAFB
        
        G_bouton_list = [G for G in G_bouton_num_dict for _ in range(G_bouton_num_dict[G])]
        connectivity_collection = []
        for shuffled_index in range(shuffled_times):
            connectivity = np.zeros((len(self.G_list),len(KC_class_list)))
            G_conditional_matrix = np.zeros((len(self.G_list),len(self.G_list)))
            for KCid in self.KCid_list:
                KC_class = self.KCid_to_Subtype[KCid]
                shared_G_list = []
                for clawid in range(self.KCid_claw_num_dict[KCid]):
                    G = rd.sample(G_bouton_list,1)[0]
                    connectivity[self.G_list.index(G)][KC_class_list.index(KC_class)] += 1
                    shared_G_list.append(G)
                for G1 in shared_G_list:
                    for G2 in shared_G_list:
                        G_conditional_matrix[self.G_list.index(G1)][self.G_list.index(G2)] += 1
            G_conditional_collection.append(G_conditional_matrix.ravel())
            connectivity_collection.append(connectivity.transpose().ravel())
        
        G_conditional_collection = np.array(G_conditional_collection)
        z_score_G_condition = (original_G_conditional.ravel()-np.average(G_conditional_collection,axis=0))/np.std(G_conditional_collection,axis=0)
        z_score_G_condition = np.reshape(z_score_G_condition,(len(self.G_list),len(self.G_list)))
        z_score_G_condition[np.isinf(z_score_G_condition)] = 0
        z_score_G_condition[np.isnan(z_score_G_condition)] = 0
        sn.heatmap(data=np.reshape(np.average(G_conditional_collection,axis=0),(len(self.G_list),len(self.G_list))))
        plt.savefig(f"G_average_random_bouton_{dataset}.png")
        plt.close()
        sn.heatmap(data=np.reshape(np.std(G_conditional_collection,axis=0),(len(self.G_list),len(self.G_list))))
        plt.savefig(f"G_STD_random_bouton_{dataset}.png")
        plt.close()
        sn.heatmap(data=z_score_G_condition)
        plt.savefig(f"G_zscore_random_bouton_{dataset}.png")
        plt.close()
        g = sn.clustermap(data=z_score_G_condition,xticklabels=self.G_list,cmap='bwr')
        color_dict = {1:'red',2:"gold",3:"deepskyblue",4:"purple"}
        for tick_label in g.ax_heatmap.axes.get_xticklabels():
            tick_text = tick_label.get_text()
            # tick_label.set_color(color_dict[self.Glomerulus_to_Cluster[self.G_list[int(tick_text)]]])
            tick_label.set_color(color_dict[self.Glomerulus_to_Cluster[tick_text]])
        plt.savefig(f'Random_bouton_model_z_score_{completeness}_G_condition_{dataset}.png',dpi=500)
        plt.close()


        connectivity_collection = np.array(connectivity_collection)
        z_score = (original_connectivity - np.average(connectivity_collection,axis=0))/np.std(connectivity_collection,axis=0)
        z_score = z_score.reshape((len(KC_class_list),len(self.G_list)))
        
        fig, axes = plt.subplots(nrows=len(KC_class_list), ncols=1, figsize=(9, 1.5 * len(KC_class_list)))
        for classification_index in range(len(KC_class_list)):
            ax = axes[classification_index]
            plt.sca(ax)
            height = z_score[classification_index]
            ax.bar(x=[i for i in range(len(self.G_list))], height=height, color='k')
            plt.axhline(y=2,linestyle='--')
            plt.axhline(y=-2,linestyle='--')

            xticklabel = self.G_list
            if classification_index == len(KC_class_list)-1:
                plt.xticks([i for i in range(len(self.G_list))],xticklabel, rotation=90)
                for i, label in enumerate(ax.get_xticklabels()):
                    if self.Glomerulus_to_Cluster[self.G_list[i]]==1:  # Color every other label red
                        label.set_color('red')
                    elif self.Glomerulus_to_Cluster[self.G_list[i]]==2:  # Color every other label red
                        label.set_color('gold')
                    elif self.Glomerulus_to_Cluster[self.G_list[i]]==3:  # Color every other label red
                        label.set_color('deepskyblue')
                    elif self.Glomerulus_to_Cluster[self.G_list[i]]==4:
                        label.set_color('purple')
            else:
                plt.xticks([])
                        
        plt.savefig(f'Random_bouton_model_z_score_{completeness}_{dataset}.png',dpi=500)
        plt.savefig(f'Random_bouton_model_z_score_{completeness}_{dataset}.svg',format='svg')
        plt.close()
        result = Df(data=z_score.transpose,columns=KC_class_list, index=self.G_list)
        result.to_csv(f"Z_score_table_random_bouton_model_{completeness}_{dataset}.csv")
        return

    def check_bouton_num_difference_with_connection_table(self):
        path = 'PN_KC_bouton_claw_information_20230618/'
        file_name = 'PN_bouton_summary.csv'
        data = pd.read_csv(f'{path}{file_name}')
        PN_original_connection_num = np.count_nonzero(self.original_pre_to_post_weight, axis=1)
        print(PN_original_connection_num)
        diff = []
        neuronId_bouton_list = data['neuronId'].values.tolist()
        # print(neuronId_claw_list)
        for i, neuronId in enumerate(self.PNid_list):
            if neuronId in neuronId_bouton_list:
                # print(neuronId, 'in list')
                diff.append(data[data['neuronId']==self.PNid_list[i]]['bouton number'].values.tolist()[0]-PN_original_connection_num[i])
            else:
                print(neuronId, 'not in list')
        print(sum([abs(i) for i in diff]))
        print(sum(diff))
        print(data['bouton number'].sum())
        print(sum(diff)/data['bouton number'].sum())

    def check_claw_num_difference_with_connection_table(self):
        path = 'PN_KC_bouton_claw_information_20230618/'
        file_name = 'KC_claw_summary.csv'
        data = pd.read_csv(f'{path}{file_name}')
        KC_original_connection_num = np.count_nonzero(self.original_pre_to_post_weight, axis=0)
        print(KC_original_connection_num)
        diff = []
        neuronId_claw_list = data['neuronId'].values.tolist()
        # print(neuronId_claw_list)
        for i, neuronId in enumerate(self.KCid_list):
            if neuronId in neuronId_claw_list:
                # print(neuronId, 'in list')
                diff.append(data[data['neuronId']==self.KCid_list[i]]['claw number'].values.tolist()[0]-KC_original_connection_num[i])
            else:
                print(neuronId, 'not in list')
        print(sum([abs(i) for i in diff]))
        print(sum(diff))
        print(data['claw number'].sum())
        print(sum(diff)/data['claw number'].sum())

    def shuffle_connection_table(self, shuffle_times=2, seed=100, network_number=2,shuffle_ratio=1):
        '''
        ###### Finished
        This function shuffles the connection from PN to downstream KC.
        For a PN connection weight and number is kept as the same.
        For a KC, the connection number is the same, but not the connection weight.
        Make a connection candidate pool and PN weight pool (or should I make KC weight pool?)

        :return:
        '''
        self.shuffle_ratio = shuffle_ratio
        print("Shuffle the connection table!")
        self.shuffled_pre_to_post_weight_collection = []
        for shuffle_seed in range(seed,seed+network_number):
            print(shuffle_seed)
            rd.seed(shuffle_seed)
            self.pre_to_post_weight = copy.deepcopy(self.original_pre_to_post_weight)
            PN_total_number = 0
            PN_weight_list = []
            KC_wait_list = []
            for PN_index in range(self.pre_number):
                PN_weight_list.append([])
                for KC_index in range(self.post_number):
                    p = rd.random()
                    if self.pre_to_post_weight[PN_index][
                        KC_index] > 0 and p < shuffle_ratio:  # the shuffle_ratio controls the prob for shuffling
                        PN_weight_list[-1].append(self.pre_to_post_weight[PN_index][KC_index])  ##From this list, we will know how many connections are there for that PN
                        self.pre_to_post_weight[PN_index][KC_index] = 0
                        KC_wait_list.append(
                            KC_index)  ##From this list, we will know how many connections are there for that KC
            max_PN = 0
            for i in PN_weight_list:
                PN_total_number += len(i)
                if max_PN < len(i):
                    max_PN = len(i)
            for _ in range(shuffle_times):
                rd.shuffle(KC_wait_list)
            Connection_list = []
            while 1:
                tmp_KC_wait_list = copy.deepcopy(KC_wait_list)
                error_count = 0
                for connection_index in range(max_PN, -1, -1):  ### Start from the largest connection number PN
                    '''
                    The algorithm assigns KC to PN in turn along the length.
                    For example:
                    connection_index = 100
                    if PNi connection number is larger than 100:
                    assign one KC
                    when all connection_index=100 have its connection
                    connection index becomes 99
                    assign all PNi that have 99th connection
                    Together, PNi does not get all KC directly. They take turns to get KC.
                    This helps prevent the illegal event that the PN has no legal KC to connect. (One PN can only connect a KC once)
                    '''
                    for PN_index in range(self.pre_number):
                        if connection_index < len(PN_weight_list[PN_index]):
                            weight = PN_weight_list[PN_index][connection_index]
                            error_count = 0
                            while (PN_index, tmp_KC_wait_list[-1]) in Connection_list:
                                rd.shuffle(tmp_KC_wait_list)
                                error_count += 1
                                if error_count > 30:
                                    ##sometimes we generate impossible random network which means the PN cannot connect KC once, we need to restart again.
                                    break
                            if error_count > 30:
                                break
                            self.pre_to_post_weight[PN_index][tmp_KC_wait_list[-1]] = weight
                            Connection_list.append((PN_index, tmp_KC_wait_list[-1]))
                            tmp_KC_wait_list = tmp_KC_wait_list[:-1]
                    if error_count > 30:
                        break
                if error_count > 30:
                    self.pre_to_post_weight[self.pre_to_post_weight > 0] = 0
                    continue
                else:
                    break
            if len(tmp_KC_wait_list) > 0:
                print(f"length: {len(KC_wait_list)}")
                print("ERROR")
                raise BaseException("KC wait list is largern than 0!!!!!!!!!!!!!!!!!!!!!!!")
            self.shuffled_pre_to_post_weight_collection.append(copy.deepcopy(self.pre_to_post_weight))
        self.shuffled_pre_to_post_weight = self.shuffled_pre_to_post_weight_collection[0]
        if shuffle_ratio == 1:
            self.connection_matrix_collection_dict['Random network'] = self.shuffled_pre_to_post_weight_collection
        else:
            self.connection_matrix_collection_dict[f'Random network {shuffle_ratio}'] = self.shuffled_pre_to_post_weight_collection
        print(f"Finished! We have constructed {len(self.shuffled_pre_to_post_weight_collection)} random networks")
        return self.shuffled_pre_to_post_weight

    def increase_stereotypy_FlyEM(self,seed=100,network_number=2):
        '''
        #### Finished
        This function will arrange the network to stereotyped network.
        However, KCab and KCa'b' cannot accomodate all the connection from PN cluster 3 and PN cluster 2, respectively.
        Thus, some PN connections will be assigned to KCg.
        We will not change the connections that originally a PN connects to the right KC.
        :return:
        '''
        print("Arrange the connection table!")
        Subtype_to_index = self.Subtype_to_label
        self.stereotyped_pre_to_post_weight_collection = []
        for shuffled_seed in range(seed,seed+network_number):
            self.pre_to_post_weight = copy.deepcopy(self.original_pre_to_post_weight)
            PN_weight_list = [[], [], []]
            KC_wait_list = [[], [], []]
            Connection_list = []
            for PN_index in range(self.pre_number):
                PN_id = self.PNid_list[PN_index]
                PN_weight_list.append([])
                for KC_index in range(self.post_number):
                    KC_id = self.KCid_list[KC_index]
                    PN_cluster_id = self.PNid_to_Cluster[PN_id]
                    KC_subtype_id = Subtype_to_index[self.KCid_to_Subtype[KC_id]]
                    if self.pre_to_post_weight[PN_index][
                        KC_index] > 0 and rd.random() < self.fixed_ratio and PN_cluster_id != KC_subtype_id:
                        PN_weight_list[PN_cluster_id - 1].append(
                            [PN_index, self.pre_to_post_weight[PN_index][KC_index]])
                        self.pre_to_post_weight[PN_index][KC_index] = 0
                        KC_wait_list[KC_subtype_id - 1].append(KC_index)
                    elif self.pre_to_post_weight[PN_index][KC_index] > 0:
                        Connection_list.append((PN_index, KC_index))
            # plt.imshow(self.original_pre_to_post_weight,aspect='auto')
            # plt.show()
            print(f"original connection number = {np.count_nonzero(self.original_pre_to_post_weight)}")
            print(
                f"wait_number = {len(PN_weight_list[0]) + len(PN_weight_list[1]) + len(PN_weight_list[2])}, {len(KC_wait_list[0]) + len(KC_wait_list[1]) + len(KC_wait_list[2])}")
            print(f"current connection number = {np.count_nonzero(self.pre_to_post_weight)}")
            print(PN_weight_list)
            print(KC_wait_list)
            PN_weight_list_final = []
            for PN_list in PN_weight_list:
                rd.shuffle(PN_list)
                PN_weight_list_final.append(PN_list)
            KC_wait_list_final = []
            for KC_list in KC_wait_list:
                rd.shuffle(KC_list)
                KC_wait_list_final.append(KC_list)
            #######################################
            left_PN_list = []
            left_KC_list = []
            for i in range(3):
                print('arrange index', i)
                '''
                Due to the min number for PN connection and KC connection is different, we can only match the connection limited by min (PN,KC)
                '''
                tmp_KC_wait_list = copy.deepcopy(KC_wait_list_final[i])
                tmp_PN_wait_list = copy.deepcopy(PN_weight_list_final[i])
                print(len(tmp_PN_wait_list))
                print(len(tmp_KC_wait_list))

                giveup_number = 0
                succeed_number = 0
                while tmp_KC_wait_list and tmp_PN_wait_list:
                    check_num = 0
                    while (tmp_PN_wait_list[0][0], tmp_KC_wait_list[0]) in Connection_list:
                        if check_num >= 200:
                            left_PN_list.append(tmp_PN_wait_list[0])
                            tmp_PN_wait_list = tmp_PN_wait_list[1:]
                            giveup_number += 1
                            break
                        rd.shuffle(tmp_KC_wait_list)
                        check_num += 1
                    if check_num < 200:
                        Connection_list.append((tmp_PN_wait_list[0][0], tmp_KC_wait_list[0]))
                        self.pre_to_post_weight[tmp_PN_wait_list[0][0]][tmp_KC_wait_list[0]] = tmp_PN_wait_list[0][1]
                        tmp_PN_wait_list = tmp_PN_wait_list[1:]
                        tmp_KC_wait_list = tmp_KC_wait_list[1:]
                        succeed_number += 1
                left_KC_list += tmp_KC_wait_list
                left_PN_list += tmp_PN_wait_list
                print(giveup_number, succeed_number)
                print(len(tmp_PN_wait_list))
                print(len(tmp_KC_wait_list))
            rd.shuffle(left_PN_list)
            rd.shuffle(left_KC_list)
            print(f"tmp connect num: {np.count_nonzero(self.pre_to_post_weight)}")
            print(f"## stage II: PN: {len(left_PN_list)},KC: {len(left_KC_list)}")
            if len(left_PN_list) != len(left_KC_list):
                print("Different!!!!")
                print(len(left_PN_list), len(left_KC_list))
            check_num = 0
            giveup_number = 0
            succeed_number = 0
            print(left_PN_list)
            print(left_KC_list)
            while left_PN_list:
                while (left_PN_list[0][0], left_KC_list[0]) in Connection_list:
                    if check_num >= 1000:
                        print("fail")
                        break
                        # raise BaseException(
                        #     "Error!! we cannot consturct legal connection table, so please restart the program again.")
                    check_num += 1
                    rd.shuffle(left_KC_list)
                if check_num < 1000:
                    self.pre_to_post_weight[left_PN_list[0][0]][left_KC_list[0]] = left_PN_list[0][1]
                    Connection_list.append((left_PN_list[0][0], left_KC_list[0]))
                    left_PN_list = left_PN_list[1:]
                    left_KC_list = left_KC_list[1:]
                    succeed_number += 1
                else:
                    giveup_number += 1
                    break

            if check_num >= 1000:
                continue
            self.stereotyped_pre_to_post_weight_collection.append(self.pre_to_post_weight)
            print(
                f"original connect num: {np.count_nonzero(self.original_pre_to_post_weight)}, arranged: {np.count_nonzero(self.pre_to_post_weight)}")
            print(
                f"total weight original: {np.sum(np.sum(self.original_pre_to_post_weight))}, arranged: {np.sum(np.sum(self.pre_to_post_weight))}")
            print(giveup_number, succeed_number)
        self.stereotyped_pre_to_post_weight = self.stereotyped_pre_to_post_weight_collection[0]
        self.connection_matrix_collection_dict['Labeled-line network'] = self.stereotyped_pre_to_post_weight_collection
        print(f"Finished!! We have constructed {len(self.stereotyped_pre_to_post_weight_collection)} stereotyped networks")
        # plt.imshow(self.pre_to_post_weight,aspect='auto')
        # plt.show()
        return

    def normalize_connection_weight(self, pre_to_post_weight):
        '''
        ### Finished!! But do I need to differentiate original or random or arranged network?
        :return:
        '''
        KC_weight_dict = {}
        for i, KCid in enumerate(self.KCid_list):
            total_weight = float(sum(pre_to_post_weight[:, i]))
            KC_weight_dict[KCid] = total_weight
            for j in range(len(pre_to_post_weight)):
                if pre_to_post_weight[j][i] > 0:
                    pre_to_post_weight[j][i] = float(pre_to_post_weight[j][i]) / total_weight
        return pre_to_post_weight

    def normalize_all_connection_in_dict(self):
        for connection_style in self.connection_matrix_collection_dict:
            self.connection_matrix_normalized_collection_dict[connection_style] = []
            for network_id in range(len(self.connection_matrix_collection_dict[connection_style])):
                self.connection_matrix_normalized_collection_dict[connection_style].append(
                    self.normalize_connection_weight(copy.deepcopy(self.connection_matrix_collection_dict[connection_style][network_id])))

    def code_check(self):
        print("####### Check shuffled/arranged network configuration ###########")
        original_KC_connect_number_collection = np.count_nonzero(self.original_pre_to_post_weight, axis=0)
        for i in range(min(len(self.stereotyped_pre_to_post_weight_collection),len(self.shuffled_pre_to_post_weight_collection))):
            shuffled_KC_connect_number_collection = np.count_nonzero(self.shuffled_pre_to_post_weight_collection[i], axis=0)
            arranged_KC_connect_number_collection = np.count_nonzero(self.stereotyped_pre_to_post_weight_collection[i], axis=0)
            print(f"KC_connection_dif_shuffled={np.count_nonzero(original_KC_connect_number_collection-shuffled_KC_connect_number_collection)}")
            print(f"KC_connection_dif_stereotyped={np.count_nonzero(original_KC_connect_number_collection-arranged_KC_connect_number_collection)}")
        print("######################")
        original_PN_connect_number_collection = np.count_nonzero(self.original_pre_to_post_weight, axis=1)
        for i in range(min(len(self.stereotyped_pre_to_post_weight_collection), len(self.shuffled_pre_to_post_weight_collection))):
            shuffled_PN_connect_number_collection = np.count_nonzero(self.shuffled_pre_to_post_weight_collection[i], axis=1)
            arranged_PN_connect_number_collection = np.count_nonzero(self.stereotyped_pre_to_post_weight_collection[i], axis=1)
            print(f"PN_con_dif_shuffled={np.count_nonzero(original_PN_connect_number_collection-shuffled_PN_connect_number_collection)}")
            print(f"PN_con_dif_stereotyped={np.count_nonzero(original_PN_connect_number_collection-arranged_PN_connect_number_collection)}")
        print("#####################")
        original_PN_connect_weight_collection = np.sum(self.original_pre_to_post_weight, axis=1)
        for i in range(min(len(self.stereotyped_pre_to_post_weight_collection),len(self.shuffled_pre_to_post_weight_collection))):
            shuffled_PN_connect_weight_collection = np.sum(self.shuffled_pre_to_post_weight_collection[i], axis=1)
            arranged_PN_connect_weight_collection = np.sum(self.stereotyped_pre_to_post_weight_collection[i], axis=1)
            print(f"PN_w_dif_shuffled={np.count_nonzero(original_PN_connect_weight_collection - shuffled_PN_connect_weight_collection)}")
            print(f"PN_w_dif_stereotyped={np.count_nonzero(original_PN_connect_weight_collection - arranged_PN_connect_weight_collection)}")
        print("####### Weight normalization ###########")
        sum_of_weight = np.sum(self.pre_to_post_weight_norm,axis=0)
        print(sum_of_weight)
        for i in range(len(self.KCid_list)):
            if abs(sum_of_weight[i]-1)>0.2:
                raise BaseException(f"The normalized weight is wrong: {sum_of_weight[i]}!")
        print(np.sum(sum_of_weight), len(self.KCid_list))

def check_connection_table_with_latest_data(network):
    data = pd.read_excel(f"{network.root}{network.PN_to_KC_dir}Connection_KCs_downstream_of_sPNs_w_3_v1.2.1.xlsx")
    difference_count = 0
    same_count = 0
    for PNid,KCid,PNtype,KCtype,weight in zip(data['up.bodyId'].values.tolist(),data['down.bodyId'].values.tolist(),data['up.type'].values.tolist(),data['down.type'].values.tolist(),data['w.weight'].values.tolist()):
        if PNid not in network.PNid_list or KCid not in network.KCid_list:
            print(PNtype,KCtype)
        elif network.original_pre_to_post_weight[network.PNid_list.index(PNid)][network.KCid_list.index(KCid)]!= weight:
            print(PNid,KCid,weight,network.original_pre_to_post_weight[network.PNid_list.index(PNid)][network.KCid_list.index(KCid)])
            difference_count += 1
        else:
            same_count += 1
    print(difference_count,same_count)
    print(np.count_nonzero(network.original_pre_to_post_weight))

def save_network(network,file_name='ConnectionSetting.pickle',path='tmp_files/'):
    with open(f"{path}{file_name}",'wb')as ff:
        pickle.dump(network,ff)

def load_network(file_name='ConnectionSetting.pickle',path='tmp_files/'):
    with open(f"{path}{file_name}", 'rb')as ff:
        network = pickle.load(ff)
    return network

def check_normalized_network(network):
    network_id=0
    for connection_style in network.connection_matrix_normalized_collection_dict:
        print(connection_style)
        print(np.sum(network.connection_matrix_normalized_collection_dict[connection_style][network_id],axis=0))
        sn.heatmap(network.connection_matrix_normalized_collection_dict[connection_style][network_id],cmap='jet')
        plt.show()

def shuffle_within_class(network, weight,upstream_type='PN'):
    rd.seed(1000)
    new_matrix = np.zeros(weight.shape)
    ptr = 0
    for subtype in network.KC_subtype_location:
        KCid_list = [i for i in range(network.KC_subtype_location[subtype][0],network.KC_subtype_location[subtype][1]+1)]
        rd.shuffle(KCid_list)
        rd.shuffle(KCid_list)
        for i in KCid_list:
            new_matrix[:,ptr] = weight[:,i]
            ptr = ptr + 1
    if upstream_type == 'PN':
        up_cluster_num = [len(network.Cluster_to_PNid[1]), len(network.Cluster_to_PNid[2]),
                          len(network.Cluster_to_PNid[3])]
        up_location = [(0, up_cluster_num[0]), (up_cluster_num[0], up_cluster_num[0] + up_cluster_num[1]),
                       (up_cluster_num[0] + up_cluster_num[1], len(network.PNid_list))]
    elif upstream_type =='Glomerulus':
        up_cluster_num = [len(network.Cluster_to_Glomerulus[1]), len(network.Cluster_to_Glomerulus[2]),
                          len(network.Cluster_to_Glomerulus[3])]
        up_location = [(0, up_cluster_num[0]), (up_cluster_num[0], up_cluster_num[0] + up_cluster_num[1]),
                       (up_cluster_num[0] + up_cluster_num[1], len(network.G_list))]
    new_matrix_2 = np.zeros(weight.shape)
    ptr = 0
    for location in up_location:
        up_list = [i for i in range(location[0],location[1])]
        rd.shuffle(up_list)
        rd.shuffle(up_list)
        for i in up_list:
            new_matrix_2[ptr,:] = new_matrix[i,:]
            ptr = ptr + 1
    return new_matrix_2

def draw_connection_table(network):
    color_dict = {"KCg": "red", 1: "red", "KCa'b'": "gold", 2: 'gold', "KCab": "deepskyblue", 3: "deepskyblue"}
    row_colors = [color_dict[network.PNid_to_Cluster[i]] for i in network.PNid_list]
    col_colors = [color_dict[network.KCid_to_Subtype[i]] for i in network.KCid_list]
    for wiring_pattern in network.connection_matrix_collection_dict:
        # network.connection_matrix_collection_dict
        fontsize=40
        for network_id, weight in enumerate(network.connection_matrix_collection_dict[wiring_pattern]):
            cg = sn.clustermap(data=weight,row_colors=row_colors,col_colors=col_colors,col_cluster=False,row_cluster=False,cmap='hot',xticklabels=False,yticklabels=False)
            cg.ax_row_dendrogram.set_visible(False)
            cg.ax_col_dendrogram.set_visible(False)
            cg.ax_heatmap.set_ylabel("PN", fontsize=fontsize)
            cg.ax_heatmap.yaxis.set_label_position("left")
            cg.ax_heatmap.set_xlabel("KC", fontsize=fontsize)
            ax_row_colors = cg.ax_row_colors
            box = ax_row_colors.get_position()
            box_heatmap = cg.ax_heatmap.get_position()
            ax_row_colors.set_position([box_heatmap.max[0], box.y0, box.width, box.height])
            cg.savefig(f"{wiring_pattern}_connection table_{network_id}.png",dpi=600)
            plt.close()
            new_weight = shuffle_within_class(network,weight)
            cg = sn.clustermap(data=new_weight,row_colors=row_colors,col_colors=col_colors,col_cluster=False,row_cluster=False,cmap='hot',xticklabels=False,yticklabels=False)
            cg.ax_row_dendrogram.set_visible(False)
            cg.ax_col_dendrogram.set_visible(False)
            cg.ax_heatmap.set_ylabel("PN", fontsize=fontsize)
            cg.ax_heatmap.yaxis.set_label_position("left")
            cg.ax_heatmap.set_xlabel("KC", fontsize=fontsize)
            ax_row_colors = cg.ax_row_colors
            box = ax_row_colors.get_position()
            box_heatmap = cg.ax_heatmap.get_position()
            ax_row_colors.set_position([box_heatmap.max[0], box.y0, box.width, box.height])
            cg.savefig(f"{wiring_pattern}_connection table_shuffled_{network_id}.png", dpi=600)
            plt.close()
            new_weight[new_weight>0] = 1
            cg = sn.clustermap(data=new_weight,row_colors=row_colors,col_colors=col_colors,col_cluster=False,row_cluster=False,cmap='hot',xticklabels=False,yticklabels=False)
            cg.ax_row_dendrogram.set_visible(False)
            cg.ax_col_dendrogram.set_visible(False)
            cg.ax_heatmap.set_ylabel("PN", fontsize=fontsize)
            cg.ax_heatmap.yaxis.set_label_position("left")
            cg.ax_heatmap.set_xlabel("KC", fontsize=fontsize)
            ax_row_colors = cg.ax_row_colors
            box = ax_row_colors.get_position()
            box_heatmap = cg.ax_heatmap.get_position()
            ax_row_colors.set_position([box_heatmap.max[0], box.y0, box.width, box.height])
            cg.savefig(f"{wiring_pattern}_connection table_shuffled_binary_{network_id}.png", dpi=600)
            plt.close()

        for network_id, weight in enumerate(network.connection_matrix_normalized_collection_dict[wiring_pattern]):
            cg = sn.clustermap(data=weight,row_colors=row_colors,col_colors=col_colors,col_cluster=False,row_cluster=False,cmap='hot',xticklabels=False,yticklabels=False)
            cg.ax_row_dendrogram.set_visible(False)
            cg.ax_col_dendrogram.set_visible(False)
            cg.ax_heatmap.set_ylabel("PN", fontsize=fontsize)
            cg.ax_heatmap.yaxis.set_label_position("left")
            cg.ax_heatmap.set_xlabel("KC", fontsize=fontsize)
            ax_row_colors = cg.ax_row_colors
            box = ax_row_colors.get_position()
            box_heatmap = cg.ax_heatmap.get_position()
            ax_row_colors.set_position([box_heatmap.max[0], box.y0, box.width, box.height])
            cg.savefig(f"{wiring_pattern}_normalized_connection table_{network_id}.png",dpi=600)
            plt.close()
            new_weight = shuffle_within_class(network,weight)
            cg = sn.clustermap(data=new_weight,row_colors=row_colors,col_colors=col_colors,col_cluster=False,row_cluster=False,cmap='hot',xticklabels=False,yticklabels=False)
            cg.ax_row_dendrogram.set_visible(False)
            cg.ax_col_dendrogram.set_visible(False)
            cg.ax_heatmap.set_ylabel("PN", fontsize=fontsize)
            cg.ax_heatmap.yaxis.set_label_position("left")
            cg.ax_heatmap.set_xlabel("KC", fontsize=fontsize)
            ax_row_colors = cg.ax_row_colors
            box = ax_row_colors.get_position()
            box_heatmap = cg.ax_heatmap.get_position()
            ax_row_colors.set_position([box_heatmap.max[0], box.y0, box.width, box.height])
            cg.savefig(f"{wiring_pattern}_normalized_connection table_shuffled_{network_id}.png", dpi=600)
            plt.close()
            new_weight[new_weight>0] = 1
            cg = sn.clustermap(data=new_weight,row_colors=row_colors,col_colors=col_colors,col_cluster=False,row_cluster=False,cmap='hot',xticklabels=False,yticklabels=False)
            cg.ax_row_dendrogram.set_visible(False)
            cg.ax_col_dendrogram.set_visible(False)
            cg.ax_heatmap.set_ylabel("PN", fontsize=fontsize)
            cg.ax_heatmap.yaxis.set_label_position("left")
            cg.ax_heatmap.set_xlabel("KC", fontsize=fontsize)
            ax_row_colors = cg.ax_row_colors
            box = ax_row_colors.get_position()
            box_heatmap = cg.ax_heatmap.get_position()
            ax_row_colors.set_position([box_heatmap.max[0], box.y0, box.width, box.height])
            cg.savefig(f"{wiring_pattern}_normalized_connection table_shuffled_binary_{network_id}.png", dpi=600)
            plt.close()

    ############# Glomerulus    ###############################################
    row_colors = [color_dict[network.Glomerulus_to_Cluster[i]] for i in network.G_list]
    col_colors = [color_dict[network.KCid_to_Subtype[i]] for i in network.KCid_list]
    for wiring_pattern in network.connection_matrix_collection_dict_g:
        # network.connection_matrix_collection_dict
        fontsize=28
        for network_id, weight in enumerate(network.connection_matrix_collection_dict_g[wiring_pattern]):
            cg = sn.clustermap(data=weight,row_colors=row_colors,col_colors=col_colors,col_cluster=False,row_cluster=False,cmap='hot',xticklabels=False,yticklabels=False)
            cg.ax_row_dendrogram.set_visible(False)
            cg.ax_col_dendrogram.set_visible(False)
            cg.ax_heatmap.set_ylabel("Glomerulus", fontsize=fontsize)
            cg.ax_heatmap.yaxis.set_label_position("left")
            cg.ax_heatmap.set_xlabel("KC", fontsize=fontsize)
            ax_row_colors = cg.ax_row_colors
            box = ax_row_colors.get_position()
            box_heatmap = cg.ax_heatmap.get_position()
            ax_row_colors.set_position([box_heatmap.max[0], box.y0, box.width, box.height])
            cg.savefig(f"{wiring_pattern}_connection table_g_{network_id}.png",dpi=600)
            plt.close()
            new_weight = shuffle_within_class(network,weight,upstream_type='Glomerulus')
            cg = sn.clustermap(data=new_weight,row_colors=row_colors,col_colors=col_colors,col_cluster=False,row_cluster=False,cmap='hot',xticklabels=False,yticklabels=False)
            cg.ax_row_dendrogram.set_visible(False)
            cg.ax_col_dendrogram.set_visible(False)
            cg.ax_heatmap.set_ylabel("Glomerulus", fontsize=fontsize)
            cg.ax_heatmap.yaxis.set_label_position("left")
            cg.ax_heatmap.set_xlabel("KC", fontsize=fontsize)
            ax_row_colors = cg.ax_row_colors
            box = ax_row_colors.get_position()
            box_heatmap = cg.ax_heatmap.get_position()
            ax_row_colors.set_position([box_heatmap.max[0], box.y0, box.width, box.height])
            cg.savefig(f"{wiring_pattern}_connection table_shuffled_g_{network_id}.png", dpi=600)
            plt.close()
            new_weight[new_weight>0] = 1
            cg = sn.clustermap(data=new_weight,row_colors=row_colors,col_colors=col_colors,col_cluster=False,row_cluster=False,cmap='hot',xticklabels=False,yticklabels=False)
            cg.ax_row_dendrogram.set_visible(False)
            cg.ax_col_dendrogram.set_visible(False)
            cg.ax_heatmap.set_ylabel("Glomerulus", fontsize=fontsize)
            cg.ax_heatmap.yaxis.set_label_position("left")
            cg.ax_heatmap.set_xlabel("KC", fontsize=fontsize)
            ax_row_colors = cg.ax_row_colors
            box = ax_row_colors.get_position()
            box_heatmap = cg.ax_heatmap.get_position()
            ax_row_colors.set_position([box_heatmap.max[0], box.y0, box.width, box.height])
            cg.savefig(f"{wiring_pattern}_connection table_shuffled_binary_g_{network_id}.png", dpi=600)
            plt.close()

        for network_id, weight in enumerate(network.connection_matrix_collection_dict_g_norm[wiring_pattern]):
            cg = sn.clustermap(data=weight,row_colors=row_colors,col_colors=col_colors,col_cluster=False,row_cluster=False,cmap='hot',xticklabels=False,yticklabels=False)
            cg.ax_row_dendrogram.set_visible(False)
            cg.ax_col_dendrogram.set_visible(False)
            cg.ax_heatmap.set_ylabel("Glomerulus", fontsize=fontsize)
            cg.ax_heatmap.yaxis.set_label_position("left")
            cg.ax_heatmap.set_xlabel("KC", fontsize=fontsize)
            ax_row_colors = cg.ax_row_colors
            box = ax_row_colors.get_position()
            box_heatmap = cg.ax_heatmap.get_position()
            ax_row_colors.set_position([box_heatmap.max[0], box.y0, box.width, box.height])
            cg.savefig(f"{wiring_pattern}_normalized_connection table_g_{network_id}.png",dpi=600)
            plt.close()
            new_weight = shuffle_within_class(network,weight,upstream_type='Glomerulus')
            cg = sn.clustermap(data=new_weight,row_colors=row_colors,col_colors=col_colors,col_cluster=False,row_cluster=False,cmap='hot',xticklabels=False,yticklabels=False)
            cg.ax_row_dendrogram.set_visible(False)
            cg.ax_col_dendrogram.set_visible(False)
            cg.ax_heatmap.set_ylabel("Glomerulus", fontsize=fontsize)
            cg.ax_heatmap.yaxis.set_label_position("left")
            cg.ax_heatmap.set_xlabel("KC", fontsize=fontsize)
            ax_row_colors = cg.ax_row_colors
            box = ax_row_colors.get_position()
            box_heatmap = cg.ax_heatmap.get_position()
            ax_row_colors.set_position([box_heatmap.max[0], box.y0, box.width, box.height])
            cg.savefig(f"{wiring_pattern}_normalized_connection table_shuffled_g_{network_id}.png", dpi=600)
            plt.close()
            new_weight[new_weight>0] = 1
            cg = sn.clustermap(data=new_weight,row_colors=row_colors,col_colors=col_colors,col_cluster=False,row_cluster=False,cmap='hot',xticklabels=False,yticklabels=False)
            cg.ax_row_dendrogram.set_visible(False)
            cg.ax_col_dendrogram.set_visible(False)
            cg.ax_heatmap.set_ylabel("Glomerulus", fontsize=fontsize)
            cg.ax_heatmap.yaxis.set_label_position("left")
            cg.ax_heatmap.set_xlabel("KC", fontsize=fontsize)
            ax_row_colors = cg.ax_row_colors
            box = ax_row_colors.get_position()
            box_heatmap = cg.ax_heatmap.get_position()
            ax_row_colors.set_position([box_heatmap.max[0], box.y0, box.width, box.height])
            cg.savefig(f"{wiring_pattern}_normalized_connection table_shuffled_binary_g_{network_id}.png", dpi=600)
            plt.close()

def get_partial_statistics(file_name):
    df = pd.read_excel(file_name)
    df = df.drop(columns=[df.columns[0]])
    print(df)
    # input()
    f_stat, p_value = f_oneway(*[df[col] for col in df.columns])
    print(f"F-statistic: {f_stat}\nP-value: {p_value}")

    # Post-hoc Testing
    if p_value < 0.05:
        # Reshape the dataframe for the post-hoc test
        df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=df.columns)
        # print(df_melt['value'].apply(type).value_counts())
        # print(df_melt['variable'].apply(type).value_counts())
        df_melt['value'] = pd.to_numeric(df_melt['value'], errors='coerce')
        df_melt['variable'] = df_melt['variable'].astype(str)
        # Perform Tukey's HSD
        posthoc = pairwise_tukeyhsd(df_melt['value'], df_melt['variable'], alpha=0.05)
        summary = posthoc.summary()
        headers = summary.data[0]
        results_data = summary.data[1:]
        results_df = pd.DataFrame(results_data, columns=headers)

        # Exporting to Excel
        results_df.to_excel(f'{file_name}tukeyHSD_results.xlsx', index=False)
        print("Performing Tukey's HSD...")
        print(posthoc)
    else:
        print("No significant difference found, may not proceed with post-hoc test.")

def plot_connection_matrix_according_to_function():
    path = 'Final_figures_summary/'
    G_list = [
    'VA7l', 'VA1v', 'VM4', 'VM1', 'VA1d', 'VC4', 'VC3m', 'D', 'VL2a', 'DL1',
    'VA5', 'VC1', 'VA7m', 'VL2p', 'VA6', 'DL3', 'DC3', 'DL5', 'VA3', 'VM7d',
    'VA2', 'DM1', 'DM4', 'DL4', 'VA4', 'VM7v', 'DM6', 'DM5', 'VC3l', 'VM3',
    'VM2', 'DA4l', 'DA3', 'VC2', 'DC1', 'DM2', 'DA4m', 'DM3', 'VM5v', 'DC2', 'VM5d']
    rd.seed(1000)
    np.random.seed(1000)
    # Transform and process the network
        # Transform and process the network
    network = ConnectionSetting()
    network.transform_PN_KC_connection_to_G_KC_connection()
    weight_matrix = shuffle_within_class(network, network.connection_matrix_collection_dict_g['FlyEM'][0], 'Glomerulus')
    # weight_matrix[weight_matrix > 0] = 1
    correlation_matrix = np.corrcoef(weight_matrix, rowvar=True)
    color_dict = {1:'red',2:'gold',3:'deepskyblue'}
    color_list = [color_dict[network.Glomerulus_to_Cluster[G]] for G in network.G_list]
    cg = sn.clustermap(data=correlation_matrix,col_colors=color_list,cmap='bwr',vmax=0.1,vmin=-0.1, method='complete')
    # Access the x-axis of the heatmap
    ax = cg.ax_heatmap
    ax.set_xticks(np.arange(len(network.G_list)))
    ax.set_xticklabels(network.G_list, rotation=90, ha='right')
    plt.show()


    print(len(network.Subtype_to_KCid['KCg']))
    print(len(network.Subtype_to_KCid["KCa'b'"]))
    print(len(network.Subtype_to_KCid["KCab"]))
    Gid_list = [network.G_list.index(G) for G in G_list]
    weight_matrix=weight_matrix[Gid_list,:]

    # Plot heatmap
    ax = sn.heatmap(weight_matrix.transpose(), cbar=False)
        # Define row and column separation indices
    row_sep_indices = [
        len(network.Subtype_to_KCid['KCg']),
        len(network.Subtype_to_KCid['KCg']) + len(network.Subtype_to_KCid["KCa'b'"])
    ]
    # Add horizontal separation lines using ax.axhline
    for row in row_sep_indices:
        ax.axhline(row, color='red', linewidth=2)
    

    # Show the plot
    plt.savefig(f'{path}Supplementary fig connection matrix according to function.png',dpi=500)
    plt.close()

    correlation_matrix = np.corrcoef(weight_matrix, rowvar=True)
    pooled = copy.deepcopy(correlation_matrix).ravel()
    vmin = np.min(pooled)
    pooled[pooled==1]=-1000
    vmax = np.max(pooled)
    sn.heatmap(correlation_matrix,cmap='bwr',vmax=0.1,vmin=-0.1)
    plt.show()
    color_dict = {1:'red',2:'gold',3:'deepskyblue'}
    color_list = [color_dict[network.Glomerulus_to_Cluster[G]] for G in G_list]
    sn.clustermap(data=correlation_matrix,col_cluster=False,row_cluster=False,col_colors=color_list,cmap='bwr',vmax=0.1,vmin=-0.1)
    plt.show()



def plot_Lshape():
    path = 'Final_figures_summary/'

    rd.seed(1000)
    np.random.seed(1000)
    # Transform and process the network
        # Transform and process the network
    network = ConnectionSetting()
    network.transform_PN_KC_connection_to_G_KC_connection()
    weight_matrix = shuffle_within_class(network, network.connection_matrix_collection_dict_g['FlyEM'][0], 'Glomerulus')
    weight_matrix[weight_matrix > 0] = 1
    print(len(network.Subtype_to_KCid['KCg']))
    print(len(network.Subtype_to_KCid["KCa'b'"]))
    print(len(network.Subtype_to_KCid["KCab"]))
          

    # Plot heatmap
    ax = sn.heatmap(weight_matrix.transpose(), cbar=False)
    plt.xticks([i for i in range(len(network.G_list))], network.G_list, rotation=90)
    
    # Show the plot
    plt.savefig(f'{path}Fig 1A_no_line.png',dpi=500)
    plt.close()

def get_network_info():
    file_name = 'ConnectionSetting.pickle'
    path = 'tmp_files/'
    c = ConnectionSetting()
    network = load_network(file_name=file_name,path=path)
    print(f"Total G num={len(network.G_list)}")
    for cluster in network.Cluster_to_Glomerulus:
        print(f"cluster {cluster} num = {len(network.Cluster_to_Glomerulus[cluster])}")
        print(network.Cluster_to_Glomerulus[cluster])
    print(f"Total KC num={len(network.KCid_list)}")
    for subtype in network.Subtype_to_KCid:
        print(f"{subtype}={len(network.Subtype_to_KCid[subtype])}")
    hemibrain_path = 'hemibrain_data/'
    cell_pref_file = 'Preference_score_threshold_3_cellular.xlsx'
    data = pd.read_excel(f"{hemibrain_path}{cell_pref_file}")
    for KC in network.Subtype_to_KCid:
        col = KC[2:]
        count_pref = data[(data[col] >= 2)].shape[0]
        count_dis = data[(data[col] <= -2)].shape[0]
        print(KC, 'Preferred:', count_pref, 'Dislike:', count_dis)

    both_pref_dis = []
    pref_list = []
    dis_list = []
    no_pref_dis_list = []

    for G in network.G_list:
        mask = data["glomerulus"] == G
        row_G = data[mask]
        count_pref = (row_G.iloc[0, 1:] > 2).sum()
        count_dis = (row_G.iloc[0, 1:] < -2).sum()
        if count_pref>0 and count_dis>0:
            both_pref_dis.append(G)
        elif count_pref>0:
            pref_list.append(G)
        elif count_dis>0:
            dis_list.append(G)
        else:
            no_pref_dis_list.append(G)
    print(f"Pref+Dis_num={len(both_pref_dis)}",both_pref_dis)
    print(f"Pref_num={len(pref_list)}",pref_list)
    print(f"Dis_num={len(dis_list)}",dis_list)
    print(f"no_pref_no_dis_num={len(no_pref_dis_list)}",no_pref_dis_list)
    pooled_list = []
    for G in network.G_list:
        for KC in network.Subtype_to_KCid:
            pooled_list.append([G,KC,data[data['glomerulus']==G][KC[2:]].values.tolist()[0]])
    print(pooled_list)
    DataFrame(data=pooled_list,columns=["Glomerulus",'KC class','z score']).to_csv("Connection_pref_final.csv")




        
    




if __name__=='__main__':
    file_name = 'ConnectionSetting.pickle'
    path = 'tmp_files/'
    Load = True
    c = ConnectionSetting()
    network = load_network(file_name=file_name,path=path)
    c.Calculate_Cluster_to_KC_specificity()
    c.Calculate_KC_from_Cluster_specificity()
    c.plot_connection_table('FlyEM')
    # c.plot_connection_table('FlyEM',subsampling=True,sampled_ratio_list=[0.95,0.75,0.8],post_fix_name='subsampled_ratio_neuron_claw_num_as_Zheng', claw_subsample=True, claw_ratio_list=[0.6,1,1])
    # c.revisit_FAFB_claw_num()

    # c.plot_bouton_claw_connectivity()
    # c.construct_random_glomerulus_model()
    # c.construct_random_bouton_model(completeness=1, shuffled_times=1000)
    # c.construct_random_bouton_model(completeness=0.75, shuffled_times=1000)
    # c.construct_random_bouton_model(completeness=0.5, shuffled_times=1000)
    # c.construct_random_bouton_model(dataset='FAFB',shuffled_times=1000)
    # c.construct_random_claw_model(shuffled_times=100,dataset='FlyEM',vmax=6.5)
    # c.construct_random_claw_model(shuffled_times=100,subsampling=True,sampled_ratio_list=[0.95,0.75,0.8],dataset='FlyEM',post_fix_name='subsampled_ratio_neuron_claw_num_as_Zheng', claw_subsample=True, claw_ratio_list=[0.6,1,1], vmax=6.5)

    # c.construct_random_claw_model(shuffled_times=100,subsampling=True,sampled_ratio_list=[0.95,0.75,0.6],dataset='FlyEM',post_fix_name='subsampled_ratio_as_Zheng')
    # c.construct_random_claw_model(shuffled_times=100,subsampling=True,sampled_ratio_list=[0.8,0.8,0.8],dataset='FlyEM',post_fix_name='averagly_subsampled_0.8')
    # c.construct_random_claw_model(shuffled_times=100,subsampling=True,completeness=0.8,dataset='FlyEM',post_fix_name='averagly_subsampled_0.8')
    c.read_FAFB_connection_csv()
    c.output_FAFB_connection_matrix()

    # input('over')
 
    c.plot_connection_table("FAFB")
    input()
    # c.compare_connectivity_FAFB_and_FlyEM()
    # c.compare_bouton_num_FAFB_FlyEM()
    c.compare_claw_num_FAFB_FlyEM()
    # c.construct_random_claw_model(dataset='FAFB',shuffled_times=100,vmax=6.5)

    c.compare_FAFB_and_FlyEM_KC_num()
    # c.construct_random_claw_model(shuffled_times=100,subsampling=True,sampled_ratio_list=c.num_ratio, 
                                #   claw_subsample=True, claw_ratio_list=c.claw_ratio,dataset='FlyEM',
                                #   post_fix_name='subsampled_ratio_as_Zheng_2',seed=102)

    # for i in range(10):
    #     c.plot_connection_table('FlyEM',subsampling=True,sampled_ratio_list=c.num_ratio,
    #                             post_fix_name=f'subsampled_ratio_neuron_claw_num_as_Zheng_{i}', 
    #                             claw_subsample=True, claw_ratio_list=c.claw_ratio, seed=i+100)
    # c.construct_random_claw_model(completeness=1, shuffled_times)
    # c.construct_random_claw_model(completeness=0.85, shuffled_times=50)
    # c.construct_random_claw_model(completeness=0.8, shuffled_times=50)

    input("OVERRRRRR")
    weight_matrix = network.connection_matrix_collection_dict_g['FlyEM'][0]
    weight_matrix[weight_matrix>0] = 1
    original = copy.deepcopy(weight_matrix)
    colors = {1:'red',2:'gold',3:'deepskyblue',"KCa'b'":'gold', 'KCab':"deepskyblue", 'KCg':"red"}
    row_colors = [colors[c.Glomerulus_to_Cluster[i]] for i in c.G_list]
    col_colors = [colors[c.KCid_to_Subtype[i]] for i in c.KCid_list]
    row_index_list = [[i  for i in range(len(c.Cluster_to_Glomerulus[1]))]]
    row_index_list += [[i+len(c.Cluster_to_Glomerulus[1]) for i in range(len(c.Cluster_to_Glomerulus[2]))]]
    row_index_list += [[i+len(c.Cluster_to_Glomerulus[1])+len(c.Cluster_to_Glomerulus[2]) for i in range(len(c.Cluster_to_Glomerulus[3]))]]
    rd.shuffle(row_index_list[0])
    rd.shuffle(row_index_list[1])
    rd.shuffle(row_index_list[2])
    l1 = len(c.Cluster_to_Glomerulus[1])
    l2 = len(c.Cluster_to_Glomerulus[2])
    weight_matrix[:l1] = weight_matrix[row_index_list[0]]
    weight_matrix[l1:l2+l1] = weight_matrix[row_index_list[1]]
    weight_matrix[l1+l2:] = weight_matrix[row_index_list[2]]

    col_index_list = [[i  for i in range(len(c.Subtype_to_KCid["KCg"]))]]
    col_index_list += [[i+len(c.Subtype_to_KCid["KCg"]) for i in range(len(c.Subtype_to_KCid["KCa'b'"]))]]
    col_index_list += [[i+len(c.Subtype_to_KCid["KCg"])+len(c.Subtype_to_KCid["KCa'b'"]) for i in range(len(c.Subtype_to_KCid["KCab"]))]]
    rd.shuffle(col_index_list[0])
    rd.shuffle(col_index_list[1])
    rd.shuffle(col_index_list[2])
    l1 = len(c.Subtype_to_KCid["KCg"])
    l2 = len(c.Subtype_to_KCid["KCa'b'"])

    weight_matrix = weight_matrix.T
    weight_matrix[:l1] = weight_matrix[col_index_list[0]]
    weight_matrix[l1:l2 + l1] = weight_matrix[col_index_list[1]]
    weight_matrix[l1+l2:] = weight_matrix[col_index_list[2]]
    # sn.clustermap(data=weight_matrix)
    weight_matrix=weight_matrix.T
    # sn.clustermap(data=weight_matrix,row_cluster=False,col_cluster=False,row_colors=row_colors,col_colors=col_colors)
    # plt.savefig('G-to-KC_connection matrix.png',dpi=600)
    # plt.close()
    print(np.count_nonzero(np.sum(original)-np.sum(weight_matrix)))
    input()
    # data = np.corrcoef(weight_matrix,rowvar=False)
    # pca = PCA()
    # data_pca = pca.fit_transform(data)
    # # Getting the explained variance ratio
    # explained_variance_ratio = pca.explained_variance_ratio_
    # # Output the explained variance ratio
    # print(explained_variance_ratio[0] / np.sum(explained_variance_ratio))
    # column_list = [i for i in range(weight_matrix.shape[1])]
    # rd.shuffle(column_list)
    # new_matrix = weight_matrix[:,column_list]
    # sn.heatmap(data = weight_matrix)
    # plt.show()
    # sn.heatmap(data = new_matrix)
    # plt.show()
    # data = np.corrcoef(new_matrix,rowvar=False)
    # pca = PCA()
    # data_pca = pca.fit_transform(data)
    # # Getting the explained variance ratio
    # explained_variance_ratio = pca.explained_variance_ratio_
    # # Output the explained variance ratio
    # print(explained_variance_ratio[0] / np.sum(explained_variance_ratio))
    #
    # # [0.05 * (i + 1) for i in range(20)]
    # input()
    for KC_class in ['KCab','KCg',"KCa'b'",'ALL']:
        # results = []
        # for ratio in [0.05 * (i + 1) for i in range(20)]:
        #     print(ratio)
        #     results.append(network.partial_shuffle_connections(KC_class, network_number=100, shuffle_ratio=ratio, seed=300, binary=False))
        # Df(data=np.array(results).T,columns=[0.05 * (i + 1) for i in range(20)]).to_excel(f"Partial shuffle PC1 {KC_class}_binary_pearson_100.xlsx")
        # Df(data=np.array(results).T, columns=[0.05 * (i + 1) for i in range(20)]).to_excel(
        #     f"Partial shuffle PC1 {KC_class}_weighted_pearson_100.xlsx")
        # get_partial_statistics(f"Partial shuffle PC1 {KC_class}_binary_pearson_100.xlsx")
        # get_partial_statistics(f"Partial shuffle PC1 {KC_class}_weighted_pearson_100.xlsx")
        network.plot_partial_shuffle_result(KC_class)
        network.plot_partial_shuffle_result(KC_class,binary=False)


    # network.shuffle_connection_table_fix_KC(network_number=10)
    # print(np.count_nonzero(np.sum(network.connection_matrix_collection_dict['Random network fix KC'][0], axis=0) - np.sum(
    #     network.connection_matrix_collection_dict['FlyEM'][0], axis=0)))
    # print(np.sum(np.sum(network.connection_matrix_collection_dict['Random network fix KC'][0], axis=1) - np.sum(
    #     network.connection_matrix_collection_dict['FlyEM'][0], axis=1)))
    # network.normalize_all_connection_in_dict()
    # save_network(network=network)
    # if os.path.isfile(f"{path}{file_name}") and Load:
    #     network = load_network(file_name=file_name,path=path)
    #     print(network.connection_matrix_collection_dict["FlyEM"][0].shape)
    # else:
    #     network = ConnectionSetting()
    #     for shuffle_ratio in [1]:
    #         network.shuffle_connection_table(shuffle_ratio=shuffle_ratio, network_number=50)
    #     network.increase_stereotypy_FlyEM(network_number=5)
    #     network.normalize_all_connection_in_dict()
    #     network.transform_PN_KC_connection_to_G_KC_connection()
    #     network.transform_PN_KC_connection_to_G_KC_connection_norm()
    #     save_network(network=network)
    # # draw_connection_table(network)

