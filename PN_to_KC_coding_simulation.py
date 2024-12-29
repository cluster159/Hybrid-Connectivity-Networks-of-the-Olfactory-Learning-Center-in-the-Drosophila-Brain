import numpy as np
from matplotlib import pyplot as plt
import random as rd
import os
import pandas as pd
import seaborn as sn
from pandas import DataFrame as Df
import copy
from sklearn.decomposition import PCA
# from evaluation_result import *
from generate_connection import ConnectionSetting
import generate_connection as gc
import simulation_process as sim
from simulation_process import Artificial_Odor
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
# import data_visualization
from scipy.spatial import distance
from scipy.cluster import hierarchy
import plotly.express as px
from sklearn import manifold
from joblib import Parallel, delayed
import pickle
import time

class simulation_experiment():
    KC_activity_threshold = 1
    PN_to_KC_weight_threshold = 3
    seed_number = 100
    activity_scaling = 3
    odor_number = 1000
    concentration_list = [0.8,0.9,1.0]
    Activity_type = ['Activity', 'Activation number', 'Activation ratio']

    def __init__(self):
        self.network = gc.load_network()
        self.odor_generator = sim.load_artificial_odor()
        self.thread_num = os.cpu_count()
        self.activity_dict = {}
        rd.seed(self.seed_number)
        self.record_pathway = f"simulation_result_threshold_{self.KC_activity_threshold}_weight_{self.PN_to_KC_weight_threshold}/"
        if not os.path.isdir(self.record_pathway): os.mkdir(self.record_pathway)
        self.more_glomerulus = False
        self.more_glomerulus_num = 2
        self.more_glomerulus_intensity = 0.2
        self.KC_reaction_map = False

    def transform_PN_activity_to_KC(self, weight, PN_activity):
        return PN_activity.dot(weight) * self.activity_scaling

    def calculate_activity_activation(self, response_matrix,  subtype_location_dict):
        '''

        :param response_matrix:
        :param subtype_location_dict:
        :return:
        '''
        # plt.subplot()
        Subtype_response_profile_dict = {}
        for subtype in subtype_location_dict:
            Subtype_response_profile_dict[subtype] = {'Activity':[],'Activation number':[],'Activation ratio':[]}
            for odor_id in range(response_matrix.shape[0]):
                response = \
                    response_matrix[odor_id][subtype_location_dict[subtype][0]:subtype_location_dict[subtype][1]+1]
                Subtype_response_profile_dict[subtype]['Activity'].append(np.sum(response))
                Subtype_response_profile_dict[subtype]['Activation number'].append(np.count_nonzero(response))
                Subtype_response_profile_dict[subtype]['Activation ratio'].append(
                    float(Subtype_response_profile_dict[subtype]['Activation number'][-1]) / (
                            subtype_location_dict[subtype][1] - subtype_location_dict[subtype][0] + 1))
        # print(Subtype_response_profile_dict)
        return Subtype_response_profile_dict

    def calculate_activation_ratio_for_a_KC(self, response_matrix, subtype_location_dict):
        response_for_a_KC_dict = {}
        activated_ratio_result = np.count_nonzero(response_matrix, axis=0) ##count along the column which is a KC
        for subtype in subtype_location_dict:
            response_for_a_KC_dict[subtype] = activated_ratio_result[subtype_location_dict[subtype][0]:subtype_location_dict[subtype][1]+1]
        return response_for_a_KC_dict

    def calculate_coresponding_pair_frequency(self, response_matrix, subtype_location_dict):
        activated_ratio_result = np.count_nonzero(response_matrix, axis=0)  ##count along the column which is a KC
        coresponding_pair_for_nonzero_dict = {}
        response_binary_matrix = copy.deepcopy(response_matrix)
        response_binary_matrix[response_binary_matrix>0] = 1
        for subtype in subtype_location_dict:
            coresponding_pair_for_nonzero_dict[subtype] = []
            for i in range(subtype_location_dict[subtype][0], subtype_location_dict[subtype][1]):
                for j in range(i+1, subtype_location_dict[subtype][1]+1):
                    if activated_ratio_result[i] != 0 and activated_ratio_result[j] != 0:
                        pair_result = response_binary_matrix[:, i] + response_binary_matrix[:, j]
                        pair_result[pair_result<2] = 0
                        coresponding_time = np.count_nonzero(pair_result)
                        coresponding_pair_for_nonzero_dict[subtype].append(coresponding_time)
        return coresponding_pair_for_nonzero_dict

    def calculate_dimension(self,pca):
        dimension = 0
        s = 0
        m = 0
        # print(pca.explained_variance_)
        for i in pca.explained_variance_:
            s = s + i
            m = m + i ** 2
        dimension = s ** 2 / m
        return dimension

    def calculate_KC_dimensionality(self, response_matrix, subtype_location_dict):
        #[:,:2]
        dimensionality_dict = {}
        response_binary_matrix = copy.deepcopy(response_matrix)
        for subtype in subtype_location_dict:
            sub_response_matrix = response_binary_matrix[:,subtype_location_dict[subtype][0]:subtype_location_dict[subtype][1]+1]
            if min(sub_response_matrix.shape) == 0:
                continue
            if np.sum(sub_response_matrix) == 0:
                continue
            pca = PCA(n_components=min(sub_response_matrix.shape))
            pca.fit(sub_response_matrix)
            dimensionality_dict[subtype] = self.calculate_dimension(pca)
        return dimensionality_dict

    def calculate_KC_dimensionality_ratio(self, dimensionality_dict, subtype_location_dict):
        dimensionality_ratio_dict = {}
        for subtype in subtype_location_dict:
            if subtype not in dimensionality_dict:
                continue
            dimensionality_ratio_dict[subtype] = float(dimensionality_dict[subtype]) / (
                    subtype_location_dict[subtype][1] - subtype_location_dict[subtype][0] + 1)
        return dimensionality_ratio_dict

    def execute_parallel_simulation(self,setting_parameter):
        connection_style, network_id, odor_type, class_id, concentration, activated_glomerulus_num, connection_type = setting_parameter
        print(setting_parameter)
        response_pooled_dict = {}
        if self.more_glomerulus == True and concentration == 1.2: ############add same cluster glomerulus################
            print("COME")
            odor_collection = copy.deepcopy(self.odor_generator.Odor_collection_dict[odor_type][activated_glomerulus_num][class_id])
            G_list = self.network.Cluster_to_Glomerulus[class_id+1]
            k = np.argwhere(odor_collection > 0)
            exist_G_list = [[] for _ in range(odor_collection.shape[0])]
            for non_zero_place in k:
                Odor_id, PNindex = non_zero_place
                G = self.network.PNid_to_Glomerulus[self.network.PNid_list[PNindex]]
                if G not in exist_G_list[Odor_id]:
                    exist_G_list[Odor_id].append(G)
            candidate_G_list = [[G for G in G_list if G not in exist_G_list[odor_id]] for odor_id in
                                range(odor_collection.shape[0])]
            for odor_id in range(odor_collection.shape[0]):
                rd.shuffle(candidate_G_list[odor_id])
                candidate_G_list[odor_id] = candidate_G_list[odor_id][:self.more_glomerulus_num]
                for G in candidate_G_list[odor_id]:
                    for PNid in self.network.Glomerulus_to_PNid[G]:
                        odor_collection[odor_id][self.network.PNid_list.index(PNid)] = 0.2
            KC_class_activity_collection = \
                odor_collection.dot(
                    self.network.connection_matrix_normalized_collection_dict[connection_style][network_id])

        else:
            print("HHH")
            KC_class_activity_collection = \
                self.odor_generator.Odor_collection_dict[odor_type][activated_glomerulus_num][class_id].dot(
                    self.network.connection_matrix_normalized_collection_dict[connection_style][network_id])
            print("AAA")
        file_name = \
            f'{self.record_pathway}{connection_style}_{network_id}_{odor_type}_{class_id}_{concentration}_{activated_glomerulus_num}_{connection_type}_KC_response.xlsx'
        result_dict = {}
        scailing_factor = self.activity_scaling * concentration
        final_KC_activity_collection = \
            scailing_factor * KC_class_activity_collection - self.KC_activity_threshold
        final_KC_activity_collection[final_KC_activity_collection < 0] = 0
        if self.KC_reaction_map == True:
            print("HEY iam here")
            sn.heatmap(data=final_KC_activity_collection,cmap='hot')
            plt.xticks([])
            plt.yticks([])
            plt.xlabel("KC",fontdict={"fontsize":20})
            plt.ylabel("Odor id",fontdict={"fontsize":20})
            plt.savefig(file_name[:-4] + 'png',dpi=600)
            plt.close()
            tmp = copy.deepcopy(final_KC_activity_collection)
            tmp[tmp>0]=1
            sn.heatmap(data=tmp,cmap='hot')
            plt.xticks([])
            plt.yticks([])
            plt.xlabel("KC",fontdict={"fontsize":20})
            plt.ylabel("Odor id",fontdict={"fontsize":20})
            plt.savefig(file_name[:-5]+'_binary.png',dpi=600)
            plt.close()
        response_pooled_dict[setting_parameter] = final_KC_activity_collection
        Df(data=final_KC_activity_collection,
           columns=[f"{self.network.id_to_new_subtype[self.network.KCid_list[i]]} {i}"
                    for i in range(self.network.post_number)],
           index=[f"Odor {i}" for i in range(len(final_KC_activity_collection))]).to_excel(file_name)
        ## for an odor, what is the activation ratio of KC
        KC_response_profile_dict = self.calculate_activity_activation(
            final_KC_activity_collection, {"KC": (0, self.network.post_number - 1)})
        KC_response_profile_dict.update(self.calculate_activity_activation(
            final_KC_activity_collection, self.network.KC_subtype_location))
        KC_response_profile_dict.update(self.calculate_activity_activation(
            final_KC_activity_collection, self.network.KC_new_subtype_location))
        result_dict['KC response profile'] = KC_response_profile_dict
        # ## for single KC activated ratio to odors
        # single_KC_activated_frequency_dict = self.calculate_activation_ratio_for_a_KC(
        #     final_KC_activity_collection, {"KC": (0, self.network.post_number - 1)})
        # single_KC_activated_frequency_dict.update(self.calculate_activation_ratio_for_a_KC(
        #     final_KC_activity_collection, self.network.KC_subtype_location))
        # single_KC_activated_frequency_dict.update(self.calculate_activation_ratio_for_a_KC(
        #     final_KC_activity_collection, self.network.KC_new_subtype_location))
        # result_dict['single KC activated frequency'] = single_KC_activated_frequency_dict
        # # print(single_KC_activated_frequency_dict['KC'])
        # ## for co-responding KC pair
        # co_responding_KC_dict = self.calculate_coresponding_pair_frequency(
        #     final_KC_activity_collection, {"KC": (0, self.network.post_number - 1)})
        # co_responding_KC_dict.update(self.calculate_coresponding_pair_frequency(
        #     final_KC_activity_collection, self.network.KC_subtype_location))
        # co_responding_KC_dict.update(self.calculate_coresponding_pair_frequency(
        #     final_KC_activity_collection, self.network.KC_new_subtype_location))
        # result_dict['co-responding KC'] = co_responding_KC_dict
        # print(co_responding_KC_dict['KC'])
        # # ## calculate dimensionality
        # dimensionality_KC_dict = self.calculate_KC_dimensionality(
        #     final_KC_activity_collection, {"KC": (0, self.network.post_number - 1)})
        # dimensionality_KC_dict.update(self.calculate_KC_dimensionality(
        #     final_KC_activity_collection, self.network.KC_subtype_location))
        # # # dimensionality_KC_dict.update(self.calculate_KC_dimensionality(
        # # #     final_KC_activity_collection, self.network.KC_new_subtype_location))
        # result_dict['Dimensionality'] = dimensionality_KC_dict
        # # # # print(dimensionality_KC_dict['KC'])
        # # # ## calculate dimensionality ratio
        # # # '''
        # # # Due to the KC number difference between subtypes,
        # # # we should not directly compare different subtype dimensionality ratio.
        # # # We don't have enough odor number due to the expansion from PN to KC.
        # # # '''
        # dimensionality_ratio_KC_dict = self.calculate_KC_dimensionality_ratio(
        #     dimensionality_KC_dict, {"KC": (0, self.network.post_number - 1)})
        # dimensionality_ratio_KC_dict.update(self.calculate_KC_dimensionality_ratio(
        #     dimensionality_KC_dict, self.network.KC_subtype_location))
        # # # dimensionality_ratio_KC_dict.update(self.calculate_KC_dimensionality_ratio(
        # # #     dimensionality_KC_dict, self.network.KC_new_subtype_location))
        # result_dict['Dimensionality ratio'] = dimensionality_ratio_KC_dict
        # # # print(dimensionality_ratio_KC_dict['KC'])
        # # end_time = time.time()
        # # print(end_time - start_time)

        return [setting_parameter, result_dict, response_pooled_dict]

    def weight_binarization(self,w):
        w[w>0] = 1
        w = w/np.sum(w,axis=0)
        return w


    def parallel_simulation(self,connection_type='original_weight', target_connection_style='all', cpu_num=4):
        '''

        :param connection_type: original weight, binary
        :return:
        '''
        ## deal with connection configuration
        if connection_type == 'binary':
            for connection_style in self.network.connection_matrix_normalized_collection_dict:
                print(connection_style)
                for network_id in range(len(self.network.connection_matrix_normalized_collection_dict[connection_style])):
                    w = self.network.connection_matrix_normalized_collection_dict[connection_style][network_id]
                    w_new = self.weight_binarization(w)
                    self.network.connection_matrix_normalized_collection_dict[connection_style][network_id] = w_new
                    print(network_id)
        weight_normalized_dict = {}
        weight_dict = {}
        if target_connection_style!='all':
            for connection_style in self.network.connection_matrix_normalized_collection_dict:
                if connection_style == target_connection_style:
                    weight_dict[target_connection_style] = self.network.connection_matrix_collection_dict[target_connection_style]
                    weight_normalized_dict[target_connection_style] = self.network.connection_matrix_normalized_collection_dict[target_connection_style]
            self.network.connection_matrix_normalized_collection_dict = weight_normalized_dict
            self.network.connection_matrix_collection_dict = weight_dict
        
        for connection_style in self.network.connection_matrix_normalized_collection_dict:
            parameters = [(connection_style, network_id, odor_type, class_id, concentration, activated_glomerulus_num, connection_type)
                      # for connection_style in self.network.connection_matrix_normalized_collection_dict
                      for network_id in range(min(len(self.network.connection_matrix_normalized_collection_dict[connection_style]),1)) ##only take atmost 10 networks
                      for odor_type in self.odor_generator.Odor_collection_dict
                      for class_id in range(len(self.odor_generator.Odor_collection_dict[odor_type][7]))
                      for activated_glomerulus_num in [7]
                      for concentration_id, concentration in enumerate(self.concentration_list)
                      ]
            if os.path.isfile(f"{self.record_pathway}result_dict_scaling_{self.activity_scaling}_{connection_style}_{connection_type}.pickle"):
                continue
            result = Parallel(n_jobs=cpu_num)(delayed(self.execute_parallel_simulation) (parameter) for parameter in parameters)
            result_dict = {}
            response_pooled_dict = {}
            for i in range(len(result)):
                result_dict[result[i][0]] = result[i][1]
                response_pooled_dict[result[i][0]] = result[i][2]
            with open(f"{self.record_pathway}result_dict_scaling_{self.activity_scaling}_{connection_style}_{connection_type}.pickle",'wb')as ff:
                pickle.dump(result_dict,ff)
            with open(f"{self.record_pathway}KC_response_pooled_dict_scaling_{self.activity_scaling}_{connection_style}_{connection_type}.pickle", 'wb')as ff:
                pickle.dump(response_pooled_dict, ff)
            del result, response_pooled_dict, result_dict


    def start_simulation(self):
        result_dict = {}
        for connection_style in self.network.connection_matrix_normalized_collection_dict:
            for network_id in range(len(self.network.connection_matrix_normalized_collection_dict[connection_style])):
                for odor_type in self.odor_generator.Odor_collection_dict:
                    for class_id in range(len(self.odor_generator.Odor_collection_dict[odor_type])):
                        KC_class_activity_collection = \
                            self.odor_generator.Odor_collection_dict[odor_type][class_id].dot(
                                self.network.connection_matrix_normalized_collection_dict[connection_style][network_id])
                        for concentration_id, concentration in enumerate(self.concentration_list):
                            if self.more_glomerulus == True and concentration == 1.2:
                                print("COME")
                                odor_collection = self.odor_generator.Odor_collection_dict[odor_type][class_id]
                                G_list = self.network.Cluster_to_Glomerulus[class_id]
                                k = np.argwhere(odor_collection > 0)
                                exist_G_list = [[] for _ in range(odor_collection.shape[0])]
                                for non_zero_place in k:
                                    Odor_id,PNindex = non_zero_place
                                    G = self.network.PNid_to_Glomerulus(self.network.PNid_list[PNindex])
                                    if G not in exist_G_list[Odor_id]:
                                        exist_G_list[Odor_id].append(G)
                                candidate_G_list = [[G for G in G_list if G not in exist_G_list[odor_id]] for odor_id in odor_collection.shape[0]]
                                for odor_id in range(odor_collection.shape[0]):
                                    rd.shuffle(candidate_G_list[odor_id])
                                    candidate_G_list[odor_id] = candidate_G_list[odor_id][:self.more_glomerulus_num]
                                    for PNid in self.network.Glomerulus_to_PNid:
                                        odor_collection[odor_id][self.network.PNid_list.index(PNid)] = self.more_glomerulus_intensity
                                KC_class_activity_collection = \
                                    odor_collection.dot(
                                        self.network.connection_matrix_normalized_collection_dict[connection_style][network_id])
                            # start_time = time.time()
                            file_name = \
                                f'{self.record_pathway}{connection_style}_{network_id}_{odor_type}_{class_id}_{concentration}_KC_response.xlsx'
                            setting_parameter = (connection_style, network_id, odor_type, class_id, concentration)
                            print(setting_parameter)
                            result_dict[setting_parameter] = {}
                            scailing_factor = self.activity_scaling * concentration
                            final_KC_activity_collection = \
                                scailing_factor * KC_class_activity_collection - self.KC_activity_threshold
                            final_KC_activity_collection[final_KC_activity_collection < 0] = 0
                            if self.KC_reaction_map == True:
                                print("HEY")
                                sn.heatmap(data=final_KC_activity_collection)
                                plt.savefig(file_name[:-4]+'png')
                                plt.close()
                            Df(data=final_KC_activity_collection,
                               columns=[f"{self.network.id_to_new_subtype[self.network.KCid_list[i]]} {i}"
                                        for i in range(self.network.post_number)],
                               index=[f"Odor {i}" for i in range(len(final_KC_activity_collection))]).to_excel(file_name)
                            ## for an odor, what is the activation ratio of KC
                            KC_response_profile_dict = self.calculate_activity_activation(
                                final_KC_activity_collection, {"KC":(0, self.network.post_number-1)})
                            KC_response_profile_dict.update(self.calculate_activity_activation(
                                final_KC_activity_collection, self.network.KC_subtype_location))
                            KC_response_profile_dict.update(self.calculate_activity_activation(
                                final_KC_activity_collection, self.network.KC_new_subtype_location))
                            result_dict[setting_parameter]['KC response profile'] = KC_response_profile_dict
                            # print(KC_response_profile_dict['KC'])
                            ## for single KC activated ratio to odors
                            single_KC_activated_frequency_dict = self.calculate_activation_ratio_for_a_KC(
                                final_KC_activity_collection, {"KC": (0, self.network.post_number - 1)})
                            single_KC_activated_frequency_dict.update(self.calculate_activation_ratio_for_a_KC(
                                final_KC_activity_collection, self.network.KC_subtype_location))
                            single_KC_activated_frequency_dict.update(self.calculate_activation_ratio_for_a_KC(
                                final_KC_activity_collection, self.network.KC_new_subtype_location))
                            result_dict[setting_parameter]['single KC activated frequency'] = single_KC_activated_frequency_dict
                            # print(single_KC_activated_frequency_dict['KC'])
                            ## for co-responding KC pair
                            co_responding_KC_dict = self.calculate_coresponding_pair_frequency(
                                final_KC_activity_collection, {"KC":(0, self.network.post_number-1)})
                            co_responding_KC_dict.update(self.calculate_coresponding_pair_frequency(
                                final_KC_activity_collection, self.network.KC_subtype_location))
                            co_responding_KC_dict.update(self.calculate_coresponding_pair_frequency(
                                final_KC_activity_collection, self.network.KC_new_subtype_location))
                            result_dict[setting_parameter]['co-responding KC'] = co_responding_KC_dict
                            # print(co_responding_KC_dict['KC'])
                            # ## calculate dimensionality
                            # dimensionality_KC_dict = self.calculate_KC_dimensionality(
                            #     final_KC_activity_collection, {"KC":(0, self.network.post_number-1)})
                            # dimensionality_KC_dict.update(self.calculate_KC_dimensionality(
                            #     final_KC_activity_collection, self.network.KC_subtype_location))
                            # dimensionality_KC_dict.update(self.calculate_KC_dimensionality(
                            #     final_KC_activity_collection, self.network.KC_new_subtype_location))
                            # result_dict[setting_parameter]['Dimensionality'] = dimensionality_KC_dict
                            # # print(dimensionality_KC_dict['KC'])
                            # ## calculate dimensionality ratio
                            # '''
                            # Due to the KC number difference between subtypes,
                            # we should not directly compare different subtype dimensionality ratio.
                            # We don't have enough odor number due to the expansion from PN to KC.
                            # '''
                            # dimensionality_ratio_KC_dict = self.calculate_KC_dimensionality_ratio(
                            #     dimensionality_KC_dict, {"KC":(0, self.network.post_number-1)})
                            # dimensionality_ratio_KC_dict.update(self.calculate_KC_dimensionality_ratio(
                            #     dimensionality_KC_dict, self.network.KC_subtype_location))
                            # dimensionality_ratio_KC_dict.update(self.calculate_KC_dimensionality_ratio(
                            #     dimensionality_KC_dict, self.network.KC_new_subtype_location))
                            # result_dict[setting_parameter]['Dimensionality ratio'] = dimensionality_ratio_KC_dict
                            # # print(dimensionality_ratio_KC_dict['KC'])
                            # end_time = time.time()
                            # print(end_time-start_time)
        with open(f"{self.record_pathway}result_dict_scaling_{self.activity_scaling}.pickle",'wb')as ff:
            pickle.dump(result_dict,ff)

if __name__=='__main__':
    exp = simulation_experiment()
    print(len(exp.network.G_list))
    # exp.more_glomerulus = True
    exp.KC_reaction_map = False
    exp.parallel_simulation(connection_type='binary',target_connection_style='FlyEM')

