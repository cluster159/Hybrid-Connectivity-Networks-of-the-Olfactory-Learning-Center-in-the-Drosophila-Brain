import random as rd
import numpy as np
import copy as cp
from generate_connection import ConnectionSetting
import generate_connection as gc
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
import os
from pandas import DataFrame as Df

class Artificial_Odor():
    def __init__(self):
        self.Odor_collection_dict = {}
        self.draw_odor_PN = False
        self.root = 'Artificial_odor/'
        if not os.path.isdir(self.root): os.mkdir(self.root)

    def get_PN_activity_artificial_single_cluster_glomerulus(self,network, activated_glomerulus_number=5, strength=1,
                                                             odor_number=1000, seed_number=100, noise=0, group_number=5):
        '''
        This function activate within single cluster glomeruli to mimic odor-sensing.
        :param network:
        :param activated_glomerulus_number:
        :param strength:
        :param odor_number:
        :param seed_number:
        :param noise:
        :return:
        '''
        print(f'Start to generate single odor!--{str(activated_glomerulus_number)}')
        rd.seed(seed_number)
        for group_index in range(group_number):
            PN_activity_class_collection = []
            G_code_collection = []
            for cluster_id in network.Cluster_to_Glomerulus:
                PN_activity_collection = []
                i = 0
                while i < odor_number:
                    candidate_G_list = cp.deepcopy(network.Cluster_to_Glomerulus[cluster_id])
                    rd.shuffle(candidate_G_list)
                    candidate_G_list = candidate_G_list[:activated_glomerulus_number]
                    candidate_G_list = sorted(candidate_G_list)
                    if str(candidate_G_list) in G_code_collection:
                        # print(f'Duplicate code detection!\n{str(candidate_G_list)}')
                        continue
                    G_code_collection.append(str(candidate_G_list))
                    i = i + 1
                    PN_activity = np.zeros(network.pre_number, dtype=float)
                    for G in candidate_G_list:
                        for PNid in network.Glomerulus_to_PNid[G]:
                            PN_index = network.PNid_list.index(PNid)
                            response = strength + rd.gauss(0.0, noise)
                            PN_activity[PN_index] = response

                    PN_activity_collection.append(PN_activity)
                PN_activity_class_collection.append(PN_activity_collection)
            if f'single_{group_index}' not in self.Odor_collection_dict:
                self.Odor_collection_dict[f'single_{group_index}'] = {}
            self.Odor_collection_dict[f'single_{group_index}'][activated_glomerulus_number] = np.array(PN_activity_class_collection)
            Df(data=self.Odor_collection_dict[f'single_{group_index}'][activated_glomerulus_number][0],
               columns=[f"PN {i}" for i in range(network.pre_number)],
               index=[f"Odor {i}" for i in range(odor_number)]).to_excel(
                f"{self.root}single_{group_index}_{activated_glomerulus_number}_PN_activity_class_1.xlsx")
            Df(data=self.Odor_collection_dict[f'single_{group_index}'][activated_glomerulus_number][1],
               columns=[f"PN {i}" for i in range(network.pre_number)],
               index=[f"Odor {i}" for i in range(odor_number)]).to_excel(
                f"{self.root}single_{group_index}_{activated_glomerulus_number}_PN_activity_class_2.xlsx")
            Df(data=self.Odor_collection_dict[f'single_{group_index}'][activated_glomerulus_number][2],
               columns=[f"PN {i}" for i in range(network.pre_number)],
               index=[f"Odor {i}" for i in range(odor_number)]).to_excel(
                f"{self.root}single_{group_index}_{activated_glomerulus_number}_PN_activity_class_3.xlsx")
            if self.draw_odor_PN:
                figure = plt.subplots(1, 3)
                for i in range(1, 4):
                    ax = plt.subplot(1, 3, i)
                    sns.heatmap(PN_activity_class_collection[i - 1], cmap='hot', cbar=False)
                    plt.title(f"Class {i}", fontdict={'fontsize': 20})
                    if i == 1:
                        plt.ylabel("Odor id", fontdict={'fontsize': 20})
                    plt.xlabel("PN", fontdict={'fontsize': 20})
                    plt.yticks([])
                    plt.xticks([])
                    if i > 1:
                        plt.yticks([])
                # plt.show()
                plt.savefig(f'{self.root}odor_single_{group_index}_{activated_glomerulus_number}.png')
                plt.close()

        return PN_activity_class_collection

    def get_PN_activity_artificial_random_odor_glomerulus(self,network, activated_glomerulus_number=5, strength=1,
                                                          odor_number=1000, seed_number=100, noise=0, group_number=5):
        '''
        This function activate within single cluster glomeruli to mimic odor-sensing.
        :param network:
        :param activated_glomerulus_number:
        :param strength:
        :param odor_number:
        :param seed_number:
        :param noise:
        :return:
        '''
        print(f'Start to generate random odor (shuffle G)!--{str(activated_glomerulus_number)}')
        rd.seed(seed_number)
        for group_index in range(group_number):
            PN_activity_class_collection = []
            G_code_collection = []

            G_list = cp.deepcopy([G for G in network.Glomerulus_to_PNid])
            rd.shuffle(G_list)
            shuffled_G_list = []
            i = 0
            for cluster_id in network.Cluster_to_Glomerulus:
                shuffled_G_list.append(cp.deepcopy(G_list))
            for cluster_id in network.Cluster_to_Glomerulus:
                PN_activity_collection = []
                i = 0
                while i < odor_number:
                    candidate_G_list = cp.deepcopy(shuffled_G_list[cluster_id - 1])
                    rd.shuffle(candidate_G_list)
                    candidate_G_list = candidate_G_list[:activated_glomerulus_number]
                    candidate_G_list = sorted(candidate_G_list)
                    if str(candidate_G_list) in G_code_collection:
                        # print(f'Duplicate code detection!\n{str(candidate_G_list)}')
                        continue
                    G_code_collection.append(str(candidate_G_list))
                    i = i + 1
                    PN_activity = np.zeros(network.pre_number, dtype=float)
                    for G in candidate_G_list:
                        for PNid in network.Glomerulus_to_PNid[G]:
                            PN_index = network.PNid_list.index(PNid)
                            response = strength + rd.gauss(0.0, noise)
                            PN_activity[PN_index] = response
                    PN_activity_collection.append(PN_activity)
                PN_activity_class_collection.append(PN_activity_collection)
            if f'random_{group_index}' not in self.Odor_collection_dict:
                self.Odor_collection_dict[f'random_{group_index}'] = {}
            self.Odor_collection_dict[f'random_{group_index}'][activated_glomerulus_number] = np.array(PN_activity_class_collection)
            Df(data=self.Odor_collection_dict[f'random_{group_index}'][activated_glomerulus_number][0],
               columns=[f"PN {i}" for i in range(network.pre_number)],
               index=[f"Odor {i}" for i in range(odor_number)]).to_excel(
                f"{self.root}random_{group_index}_{activated_glomerulus_number}_PN_activity_class_1.xlsx")
            Df(data=self.Odor_collection_dict[f'random_{group_index}'][activated_glomerulus_number][1],
               columns=[f"PN {i}" for i in range(network.pre_number)],
               index=[f"Odor {i}" for i in range(odor_number)]).to_excel(
                f"{self.root}random_{group_index}_{activated_glomerulus_number}_PN_activity_class_2.xlsx")
            Df(data=self.Odor_collection_dict[f'random_{group_index}'][activated_glomerulus_number][2],
               columns=[f"PN {i}" for i in range(network.pre_number)],
               index=[f"Odor {i}" for i in range(odor_number)]).to_excel(
                f"{self.root}random_{group_index}_{activated_glomerulus_number}_PN_activity_class_3.xlsx")
            if self.draw_odor_PN:
                figure = plt.subplots(1, 3)
                for i in range(1, 4):
                    ax = plt.subplot(1, 3, i)
                    sns.heatmap(PN_activity_class_collection[i - 1], cmap='hot', cbar=False)
                    plt.title(f"Class {i}", fontdict={'fontsize': 20})
                    if i == 1:
                        plt.ylabel("Odor id", fontdict={'fontsize': 20})
                    plt.xlabel("PN", fontdict={'fontsize': 20})
                    plt.yticks([])
                    plt.xticks([])
                    if i > 1:
                        plt.yticks([])
                # plt.show()
                plt.savefig(f'{self.root}odor_random_{group_index}_{activated_glomerulus_number}.png')
                plt.close()
        return PN_activity_class_collection

    def get_PN_activity_artificial_shuffled_odor_glomerulus(self,network, activated_glomerulus_number=5, strength=1,
                                                          odor_number=1000, seed_number=100, noise=0, shuffled_time=3):
        '''
        This function activate within single cluster glomeruli to mimic odor-sensing.
        :param network:
        :param activated_glomerulus_number:
        :param strength:
        :param odor_number:
        :param seed_number:
        :param noise:
        :return:
        '''
        print(f'Start to generate shuffled odor (shuffle G)!--{str(activated_glomerulus_number)}')
        for shuffle_index in range(shuffled_time):
            rd.seed(seed_number+shuffle_index)
            G_list = cp.deepcopy([G for G in network.Glomerulus_to_PNid])
            PN_activity_class_collection = []
            G_code_collection = []
            rd.shuffle(G_list)
            shuffled_G_list = []
            i = 0
            for cluster_id in network.Cluster_to_Glomerulus:
                shuffled_G_list.append([])
                for j in range(len(network.Cluster_to_Glomerulus[cluster_id])):
                    shuffled_G_list[-1].append(G_list[i])
                    i = i + 1
            for cluster_id in network.Cluster_to_Glomerulus:
                PN_activity_collection = []
                i = 0
                whole_candidate_G_list = cp.deepcopy(shuffled_G_list[cluster_id - 1])
                while i < odor_number:
                    rd.shuffle(whole_candidate_G_list)
                    candidate_G_list = whole_candidate_G_list[:activated_glomerulus_number]
                    candidate_G_list = sorted(candidate_G_list)
                    if str(candidate_G_list) in G_code_collection:
                        # print(f'Duplicate code detection!\n{str(candidate_G_list)}')
                        continue
                    G_code_collection.append(str(candidate_G_list))
                    i = i + 1
                    PN_activity = np.zeros(network.pre_number, dtype=float)
                    for G in candidate_G_list:
                        for PNid in network.Glomerulus_to_PNid[G]:
                            PN_index = network.PNid_list.index(PNid)
                            response = strength + rd.gauss(0.0, noise)
                            PN_activity[PN_index] = response
                    PN_activity_collection.append(PN_activity)
                PN_activity_class_collection.append(PN_activity_collection)
            if f'shuffled_{shuffle_index}' not in self.Odor_collection_dict:
                self.Odor_collection_dict[f'shuffled_{shuffle_index}'] = {}
            self.Odor_collection_dict[f'shuffled_{shuffle_index}'][activated_glomerulus_number] = np.array(PN_activity_class_collection)
            Df(data=self.Odor_collection_dict[f'shuffled_{shuffle_index}'][activated_glomerulus_number][0],
               columns=[f"PN {i}" for i in range(network.pre_number)],
               index=[f"Odor {i}" for i in range(odor_number)]).to_excel(
                f"{self.root}shuffled_{shuffle_index}_{activated_glomerulus_number}_PN_activity_class_1.xlsx")
            Df(data=self.Odor_collection_dict[f'shuffled_{shuffle_index}'][activated_glomerulus_number][1],
               columns=[f"PN {i}" for i in range(network.pre_number)],
               index=[f"Odor {i}" for i in range(odor_number)]).to_excel(
                f"{self.root}shuffled_{shuffle_index}_{activated_glomerulus_number}_PN_activity_class_2.xlsx")
            Df(data=self.Odor_collection_dict[f'shuffled_{shuffle_index}'][activated_glomerulus_number][2],
               columns=[f"PN {i}" for i in range(network.pre_number)],
               index=[f"Odor {i}" for i in range(odor_number)]).to_excel(
                f"{self.root}shuffled_{shuffle_index}_{activated_glomerulus_number}_PN_activity_class_3.xlsx")
            if self.draw_odor_PN:
                figure = plt.subplots(1, 3)
                for i in range(1, 4):
                    ax = plt.subplot(1, 3, i)
                    sns.heatmap(PN_activity_class_collection[i - 1], cmap='hot',cbar=False)
                    plt.title(f"Class {i}", fontdict={'fontsize': 20})
                    if i == 1:
                        plt.ylabel("Odor id", fontdict={'fontsize': 20})
                    plt.xlabel("PN", fontdict={'fontsize': 20})
                    plt.yticks([])
                    plt.xticks([])
                    if i > 1:
                        plt.yticks([])
                plt.savefig(f'{self.root}odor_shuffled_{shuffle_index}_{activated_glomerulus_number}.png')
                plt.close()
        return PN_activity_class_collection
    
    def get_PN_activity_artificial_ingroup_preference(self,network, activated_glomerulus_number=4,ingroup_ratio=0.75, strength=1,
                                                             odor_number=1000, seed_number=100, noise=0, group_number=5):
        print(f'Start to generate ingroup-outgroup odor!--{str(activated_glomerulus_number)}')
        total_G_number = activated_glomerulus_number
        rd.seed(seed_number)
        for group_index in range(group_number):
            PN_activity_class_collection = []
            G_code_collection = []
            for cluster_id in network.Cluster_to_Glomerulus:
                PN_activity_collection = []
                i = 0
                cluster_list_tmp = cp.deepcopy(list(network.Cluster_to_Glomerulus.keys()))
                ingroup_Glomerulus_list = network.Cluster_to_Glomerulus[cluster_id]
                cluster_list_tmp.pop(cluster_list_tmp.index(cluster_id))
                outgroup_Glomerulus_list = network.Cluster_to_Glomerulus[cluster_list_tmp[0]] \
                                           + network.Cluster_to_Glomerulus[cluster_list_tmp[1]]
                ingroup_number = round(activated_glomerulus_number*ingroup_ratio)
                outgroup_number = activated_glomerulus_number - ingroup_number
                while i < odor_number:
                    rd.shuffle(ingroup_Glomerulus_list)
                    rd.shuffle(outgroup_Glomerulus_list)
                    tmp_candidate_G_list = ingroup_Glomerulus_list[:ingroup_number] \
                                           + outgroup_Glomerulus_list[:outgroup_number]
                    candidate_G_list = tmp_candidate_G_list
                    candidate_G_list = sorted(candidate_G_list)
                    if str(candidate_G_list) in G_code_collection:
                        # print(f'Duplicate code detection!\n{str(candidate_G_list)}')
                        continue
                    G_code_collection.append(str(candidate_G_list))
                    i = i + 1
                    # print(f"prefereed: cluster_id = {cluster_id}", f"Activated glomuruli:{candidate_G_list}",
                    #       f"corresponded cluster:{[network.Glomerulus_to_Cluster[g] for g in candidate_G_list]}")
                    PN_activity = np.zeros(network.pre_number, dtype=float)
                    for G in candidate_G_list:
                        for PNid in network.Glomerulus_to_PNid[G]:
                            PN_index = network.PNid_list.index(PNid)
                            response = strength + rd.gauss(0.0, noise)
                            PN_activity[PN_index] = response
                    PN_activity_collection.append(PN_activity)

                PN_activity_class_collection.append(PN_activity_collection)
            if f'ingroup_{ingroup_ratio}_{group_index}' not in self.Odor_collection_dict:
                self.Odor_collection_dict[f'ingroup_{ingroup_ratio}_{group_index}'] = {}
            self.Odor_collection_dict[f'ingroup_{ingroup_ratio}_{group_index}'][total_G_number] = np.array(
                PN_activity_class_collection)
            Df(data=self.Odor_collection_dict[f'ingroup_{ingroup_ratio}_{group_index}'][total_G_number][0],
               columns=[f"PN {i}" for i in range(network.pre_number)],
               index=[f"Odor {i}" for i in range(odor_number)]).to_excel(
                f"{self.root}ingroup_{ingroup_ratio}_{group_index}_{total_G_number}_PN_activity_class_1.xlsx")
            Df(data=self.Odor_collection_dict[f'ingroup_{ingroup_ratio}_{group_index}'][total_G_number][1],
               columns=[f"PN {i}" for i in range(network.pre_number)],
               index=[f"Odor {i}" for i in range(odor_number)]).to_excel(
                f"{self.root}ingroup_{ingroup_ratio}_{group_index}_{total_G_number}_PN_activity_class_2.xlsx")
            Df(data=self.Odor_collection_dict[f'ingroup_{ingroup_ratio}_{group_index}'][total_G_number][2],
               columns=[f"PN {i}" for i in range(network.pre_number)],
               index=[f"Odor {i}" for i in range(odor_number)]).to_excel(
                f"{self.root}ingroup_{ingroup_ratio}_{group_index}_{total_G_number}_PN_activity_class_3.xlsx")
            if self.draw_odor_PN:
                figure = plt.subplots(1, 3)
                for i in range(1, 4):
                    ax = plt.subplot(1, 3, i)
                    sns.heatmap(PN_activity_class_collection[i - 1], cmap='hot', cbar=False)
                    plt.title(f"Class {i}", fontdict={'fontsize': 20})
                    if i == 1:
                        plt.ylabel("Odor id", fontdict={'fontsize': 20})
                    plt.xlabel("PN", fontdict={'fontsize': 20})
                    plt.yticks([])
                    plt.xticks([])
                    if i > 1:
                        plt.yticks([])
                # plt.show()
                plt.savefig(f'{self.root}odor_ingroup_{ingroup_ratio}_{group_index}_{total_G_number}.png')
                plt.close()
        return PN_activity_class_collection

    def get_PN_activity_artificial_biased_cluster_glomerulus(self,network, activated_glomerulus_number=[3, 1], strength=1,
                                                             odor_number=1000, seed_number=100, noise=0, group_number=5):
        '''
        This function activate more selected cluster glomeruli but still activate non-preferred cluster glomeruli to mimic odor-sensing.
        :param network:
        :param activated_glomerulus_number: [number of preferred glomerulus, number of non-preferred glomerulus
        :param strength:
        :param odor_number:
        :param seed_number:
        :param noise:
        :return:
        '''
        print(f'Start to generate biased odor!--{str(activated_glomerulus_number)}')
        total_G_number = activated_glomerulus_number[0] + 2*activated_glomerulus_number[1]
        rd.seed(seed_number)
        for group_index in range(group_number):
            PN_activity_class_collection = []
            G_code_collection = []
            for cluster_id in network.Cluster_to_Glomerulus:
                PN_activity_collection = []
                i = 0
                while i < odor_number:
                    tmp_candidate_G_list = []
                    for current_cluster_id in network.Cluster_to_Glomerulus:
                        candidate_G_list = cp.deepcopy(network.Cluster_to_Glomerulus[current_cluster_id])
                        rd.shuffle(candidate_G_list)
                        if current_cluster_id == cluster_id:
                            tmp_candidate_G_list = tmp_candidate_G_list + candidate_G_list[
                                                                          :activated_glomerulus_number[0]]
                        else:
                            tmp_candidate_G_list = tmp_candidate_G_list + candidate_G_list[
                                                                          :activated_glomerulus_number[1]]
                    candidate_G_list = tmp_candidate_G_list
                    candidate_G_list = sorted(candidate_G_list)
                    if str(candidate_G_list) in G_code_collection:
                        # print(f'Duplicate code detection!\n{str(candidate_G_list)}')
                        continue
                    G_code_collection.append(str(candidate_G_list))
                    i = i + 1
                    # print(f"prefereed: cluster_id = {cluster_id}", f"Activated glomuruli:{candidate_G_list}",
                    #       f"corresponded cluster:{[network.Glomerulus_to_Cluster[g] for g in candidate_G_list]}")
                    PN_activity = np.zeros(network.pre_number, dtype=float)
                    for G in candidate_G_list:
                        for PNid in network.Glomerulus_to_PNid[G]:
                            PN_index = network.PNid_list.index(PNid)
                            response = strength + rd.gauss(0.0, noise)
                            PN_activity[PN_index] = response
                    PN_activity_collection.append(PN_activity)

                PN_activity_class_collection.append(PN_activity_collection)
            if f'preferred_{group_index}' not in self.Odor_collection_dict:
                self.Odor_collection_dict[f'preferred_{group_index}'] = {}
            self.Odor_collection_dict[f'preferred_{group_index}'][total_G_number] = np.array(PN_activity_class_collection)
            Df(data=self.Odor_collection_dict[f'preferred_{group_index}'][total_G_number][0],
               columns=[f"PN {i}" for i in range(network.pre_number)],
               index=[f"Odor {i}" for i in range(odor_number)]).to_excel(
                f"{self.root}preferred_{group_index}_{total_G_number}_PN_activity_class_1.xlsx")
            Df(data=self.Odor_collection_dict[f'preferred_{group_index}'][total_G_number][1],
               columns=[f"PN {i}" for i in range(network.pre_number)],
               index=[f"Odor {i}" for i in range(odor_number)]).to_excel(
                f"{self.root}preferred_{group_index}_{total_G_number}_PN_activity_class_2.xlsx")
            Df(data=self.Odor_collection_dict[f'preferred_{group_index}'][total_G_number][2],
               columns=[f"PN {i}" for i in range(network.pre_number)],
               index=[f"Odor {i}" for i in range(odor_number)]).to_excel(
                f"{self.root}preferred_{group_index}_{total_G_number}_PN_activity_class_3.xlsx")
            if self.draw_odor_PN:
                figure = plt.subplots(1, 3)
                for i in range(1, 4):
                    ax = plt.subplot(1, 3, i)
                    sns.heatmap(PN_activity_class_collection[i - 1], cmap='hot', cbar=False)
                    plt.title(f"Class {i}", fontdict={'fontsize': 20})
                    if i == 1:
                        plt.ylabel("Odor id", fontdict={'fontsize': 20})
                    plt.xlabel("PN", fontdict={'fontsize': 20})
                    plt.yticks([])
                    plt.xticks([])
                    if i > 1:
                        plt.yticks([])
                # plt.show()
                plt.savefig(f'{self.root}odor_preferred_{group_index}_{total_G_number}.png')
                plt.close()
        return PN_activity_class_collection

def save_artificial_odor(odor_stimulation,file_name='ArtificialOdor.pickle',path='tmp_files/'):
    with open(f"{path}{file_name}",'wb')as ff:
        pickle.dump(odor_stimulation,ff)

def load_artificial_odor(file_name='ArtificialOdor.pickle',path='tmp_files/'):
    if os.path.isfile(f"{path}{file_name}"):
        with open(f"{path}{file_name}",'rb')as ff:
            odor_stimulation = pickle.load(ff)
    else:
        network = gc.load_network()
        Artificial_Odor.draw_odor_PN = True
        odor_stimulation = Artificial_Odor()
        odor_stimulation.get_PN_activity_artificial_biased_cluster_glomerulus(network)
        odor_stimulation.get_PN_activity_artificial_single_cluster_glomerulus(network)
        odor_stimulation.get_PN_activity_artificial_random_odor_glomerulus(network)
        save_artificial_odor(odor_stimulation)
    return odor_stimulation


if __name__=='__main__':
    network = gc.load_network()
    odor_stimulation = Artificial_Odor()
    odor_stimulation.draw_odor_PN = False
    for activated_glomerulus_number in [4,5,7]:
        odor_stimulation.get_PN_activity_artificial_shuffled_odor_glomerulus(network,activated_glomerulus_number=activated_glomerulus_number)
        for ratio in [0.75,0.5]:
            odor_stimulation.get_PN_activity_artificial_ingroup_preference(network,activated_glomerulus_number=activated_glomerulus_number,group_number=5,ingroup_ratio=ratio)
        odor_stimulation.get_PN_activity_artificial_random_odor_glomerulus(network,activated_glomerulus_number=activated_glomerulus_number,group_number=5)
        odor_stimulation.get_PN_activity_artificial_biased_cluster_glomerulus(network,activated_glomerulus_number=[activated_glomerulus_number-2,1],group_number=5)
        odor_stimulation.get_PN_activity_artificial_single_cluster_glomerulus(network,activated_glomerulus_number=activated_glomerulus_number,group_number=5)
    save_artificial_odor(odor_stimulation)
    print(odor_stimulation.Odor_collection_dict['single'])
    odor_generator = load_artificial_odor()
    print(odor_generator.Odor_collection_dict)
    #
    # print(odor_generator.Odor_collection_dict['single'][4][0])
    # print(len(odor_generator.Odor_collection_dict['single'][4][0]))
    # print(len(odor_generator.Odor_collection_dict['single'][4][1]))
    # print(len(odor_generator.Odor_collection_dict['single'][4][2]))