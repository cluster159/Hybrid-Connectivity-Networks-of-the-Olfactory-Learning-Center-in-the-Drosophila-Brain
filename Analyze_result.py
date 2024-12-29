import pandas as pd
from pandas import DataFrame as Df
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from PN_to_KC_coding_simulation import simulation_experiment
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import distance
from scipy.cluster import hierarchy
from sklearn import manifold
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg
from generate_connection import ConnectionSetting
import generate_connection as gc
import simulation_process as sim
from simulation_process import Artificial_Odor
from scipy.stats import ks_2samp
from scipy.stats import friedmanchisquare
import copy
from scipy.spatial.distance import euclidean

plt.rcParams['font.family'] = 'Arial'


def rename_KC_subtype(string):
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
    group[0] = group[0].replace('a', '\u03B1').replace('b', '\u03B2').replace('g', '\u03B3')
    new_string = ""
    for index, s in enumerate(group):
        if index == 0:
            new_string += s
        else:
            new_string += '-'+s
    return new_string


class linear_regression_model:
    def __init__(self):
        self.mse = np.nan
        self.r_squared = np.nan
        self.adj_r_squared = np.nan
        self.slope = np.nan
        self.intercept = np.nan
        self.model = LinearRegression(fit_intercept=True)

    def model_fitting(self, x, y):
        x = x[:, np.newaxis]
        model = LinearRegression(fit_intercept=True)
        model.fit(x, y)
        # print(model.intercept_)
        mse = np.mean((model.predict(x) - y) ** 2)
        r_squared = model.score(x, y)
        adj_r_squared = r_squared - (1 - r_squared) * (x.shape[1] / (x.shape[0] - x.shape[1] - 1))
        self.mse = mse
        self.r_squared = r_squared
        self.adj_r_squared = adj_r_squared
        self.slope = model.coef_[0]
        self.intercept = model.intercept_
        self.model = model
        return

    def plot(self, x, y, show=False, file_name=''):
        xfit = np.linspace(min(x), max(x), len(x) * 10)
        yfit = self.model.predict(xfit[:, np.newaxis])
        plt.scatter(x, y)
        plt.plot(xfit, yfit)
        # print('here')
        if len(file_name) != 0:
            print('save')
            plt.text(0.4, 0.1, f"r_squard = {self.r_squared}", fontsize=14)
            plt.savefig(f"{file_name}.png")
        if show == True:
            plt.show()
        plt.close()
        return


class Result_Analyzer(simulation_experiment):
    color_dict = {"KC": 'k', "KCab": 'deepskyblue', 'KCg': 'red', "KCa'b'": "gold", "Odor class 1": 'red',
                  "Odor class 2": "gold", "Odor class 3": "deepskyblue"}
    # connection_style, network_id, odor_type, class_id, concentration
    root = 'Analysis_result/'
    if not os.path.isdir(root): os.mkdir(root)
    fontdict = {'Title': {'fontsize': 36}, 'Label': {'fontsize': 28}, 'Legend': {'fontsize': 20},
                'Tick': {'fontsize': 20}}

    def calculate_cluster_centroid_distance(self, m1, m2, record):
        center_m1 = np.average(m1, axis=0)
        center_m2 = np.average(m2, axis=0)
        intra_dist_m1 = 0
        inter_dist_m1 = 0
        for i in range(m1.shape[0]):
            intra_dist_m1 += euclidean(m1[i], center_m1)
            inter_dist_m1 += euclidean(m1[i], center_m2)
        intra_dist_m1 = intra_dist_m1 / m1.shape[0]
        inter_dist_m1 = inter_dist_m1 / m1.shape[0]

        intra_dist_m2 = 0
        inter_dist_m2 = 0
        for i in range(m2.shape[0]):
            intra_dist_m2 += euclidean(m2[i], center_m2)
            inter_dist_m2 += euclidean(m2[i], center_m1)
        intra_dist_m2 = intra_dist_m2 / m2.shape[0]
        inter_dist_m2 = inter_dist_m2 / m2.shape[0]
        record[0] = intra_dist_m1
        record[1] = inter_dist_m1
        record[2] = intra_dist_m2
        record[3] = inter_dist_m2
        return record

    def fast_draw_PCA(self):
        path = f"{self.root}KC code distance/"
        if not os.path.isdir(path): os.mkdir(path)
        for connection_style in self.network.connection_matrix_normalized_collection_dict:
            if connection_style != 'FlyEM':
                continue
            self.load_simulation_data(connection_style=connection_style)
            parameter_list = []
            odor_class_collection = []
            for setting_parameter in self.response_pooled_dict:
                c_style, network_id, odor_type, class_id, concentration, activated_glomerulus_num = setting_parameter
                if [connection_style, network_id, odor_type, 0, concentration,
                    activated_glomerulus_num] not in parameter_list:
                    parameter_list.append(
                        [connection_style, network_id, odor_type, 0, concentration, activated_glomerulus_num])
                if class_id not in odor_class_collection:
                    odor_class_collection.append(class_id)
            for setting_parameter in parameter_list:
                response_collection = []
                for class_id in odor_class_collection:
                    setting_parameter[3] = class_id
                    # print(type(self.response_pooled_dict[tuple(setting_parameter)][tuple(setting_parameter)]))
                    response_collection += list(
                        self.response_pooled_dict[tuple(setting_parameter)][tuple(setting_parameter)])
                response_collection = np.array(response_collection)
                pca = PCA(n_components=2)
                X = pca.fit_transform(X=response_collection)
                color = ['r', 'gold', 'deepskyblue']
                l = 1000
                for i in range(3):
                    plt.plot(X[l * i:l * (i + 1), 0], X[l * i:l * (i + 1), 1], '.', color=color[i])
                file_name = ''
                for n in setting_parameter:
                    file_name += str(n) + "_"
                plt.savefig(f"{path}{file_name[:-1]}.png")
                plt.close()

    def fast_draw_tsne(self):
        path = f"{self.root}KC code distance/"
        if not os.path.isdir(path): os.mkdir(path)
        for connection_style in self.network.connection_matrix_normalized_collection_dict:
            if connection_style != 'FlyEM':
                continue
            self.load_simulation_data(connection_style=connection_style)
            parameter_list = []
            odor_class_collection = []
            for setting_parameter in self.response_pooled_dict:
                c_style, network_id, odor_type, class_id, concentration, activated_glomerulus_num = setting_parameter
                if [connection_style, network_id, odor_type, 0, concentration,
                    activated_glomerulus_num] not in parameter_list:
                    parameter_list.append(
                        [connection_style, network_id, odor_type, 0, concentration, activated_glomerulus_num])
                if class_id not in odor_class_collection:
                    odor_class_collection.append(class_id)
            for setting_parameter in parameter_list:
                response_collection = []
                for class_id in odor_class_collection:
                    setting_parameter[3] = class_id
                    # print(type(self.response_pooled_dict[tuple(setting_parameter)][tuple(setting_parameter)]))
                    response_collection += list(
                        self.response_pooled_dict[tuple(setting_parameter)][tuple(setting_parameter)])
                response_collection = np.array(response_collection)
                pca = TSNE(n_components=2)
                X = pca.fit_transform(X=response_collection)
                color = ['r', 'gold', 'deepskyblue']
                l = 1000
                for i in range(3):
                    plt.plot(X[l * i:l * (i + 1), 0], X[l * i:l * (i + 1), 1], '.', color=color[i])
                file_name = ''
                for n in setting_parameter:
                    file_name += str(n) + "_"
                plt.savefig(f"{path}{file_name[:-1]}_tsne.png")
                plt.close()

    def analyze_cluster_distance(self):
        path = f"{self.root}KC code distance/"
        if not os.path.isdir(path): os.mkdir(path)
        pooled_data = []
        for connection_style in self.network.connection_matrix_normalized_collection_dict:
            self.load_simulation_data(connection_style=connection_style)
            parameter_list = []
            odor_class_collection = []
            for setting_parameter in self.response_pooled_dict:
                c_style, network_id, odor_type, class_id, concentration, activated_glomerulus_num = setting_parameter
                if [connection_style, network_id, odor_type, 0, concentration,
                    activated_glomerulus_num] not in parameter_list:
                    parameter_list.append(
                        [connection_style, network_id, odor_type, 0, concentration, activated_glomerulus_num])
                if class_id not in odor_class_collection:
                    odor_class_collection.append(class_id)
            for setting_parameter in parameter_list:
                response_collection = []
                for class_id in odor_class_collection:
                    setting_parameter[3] = class_id
                    # print(type(self.response_pooled_dict[tuple(setting_parameter)][tuple(setting_parameter)]))
                    response_collection.append(
                        self.response_pooled_dict[tuple(setting_parameter)][tuple(setting_parameter)])
                for i, mi in enumerate(response_collection):
                    for j, mj in enumerate(response_collection):
                        if j <= i:
                            continue
                        record = self.calculate_cluster_centroid_distance(mi, mj, np.zeros(4, dtype=float))
                        c_style, network_id, odor_type, class_id, concentration, activated_glomerulus_num = setting_parameter
                        pooled_data.append(
                            [c_style, network_id, odor_type, concentration, activated_glomerulus_num, i,
                             j] + record.tolist())
                        print(record)
            ###################################################################################################################
        pooled_data_df = Df(data=np.array(pooled_data), columns=["Wiring pattern", "Network id", 'Stimulation type',
                                                                 'Concentration', 'Glomerulus number',
                                                                 'Odor class id 1', 'Odor class id 2',
                                                                 'Intra-distance 1', 'Inter-distance 1 2',
                                                                 'Intra-distance 2', 'Inter-distance 2 1'])
        pooled_data_df.to_csv(f"{path}pooled_result.csv")

    def load_simulation_data(self, connection_style='FlyEM', connection_type='binary'):
        '''
        result_dict_scaling_3_FlyEM_binary.pickle
        '''
        print(connection_type)
        if connection_style:
            connection_style = f"_{connection_style}"
        print(f"{self.record_pathway}result_dict_scaling_{self.activity_scaling}{connection_style}.pickle")
        if connection_type:
            connection_type = f"_{connection_type}"
        print(f"{self.record_pathway}result_dict_scaling_{self.activity_scaling}{connection_style}{connection_type}.pickle")
        with open(f"{self.record_pathway}result_dict_scaling_{self.activity_scaling}{connection_style}{connection_type}.pickle",
                  'rb')as ff:
            self.result_dict = pickle.load(ff)
        with open(
                f"{self.record_pathway}KC_response_pooled_dict_scaling_{self.activity_scaling}{connection_style}{connection_type}.pickle",
                'rb')as ff:
            self.response_pooled_dict = pickle.load(ff)

    def analyze_activation_ratio(self, KC_type_collection, connection_type='binary'):
        path = f"{self.root}Activation ratio/"
        if not os.path.isdir(path): os.mkdir(path)
        fitting_linear_regression = linear_regression_model()
        for connection_style in self.network.connection_matrix_normalized_collection_dict:
            try:
                self.load_simulation_data(connection_style=connection_style, connection_type=connection_type)
            except:
                print(f'no {connection_style} simulation data')
                continue
            pooled_data = [list(parameter) + [subtype, odor_id, activation_ratio]
                           for parameter in self.result_dict
                           for subtype in self.network.Subtype_to_KCid
                           for odor_id, activation_ratio in
                           enumerate(self.result_dict[parameter]['KC response profile'][subtype]['Activation ratio'])
                           ]
            for i in range(len(pooled_data)):
                pooled_data[i][3] = f"Odor class {pooled_data[i][3] + 1}"
            # print(pooled_data)
            pooled_data_df = Df(data=pooled_data,
                                columns=["Wiring pattern", "Network id", 'Stimulation type', 'Odor class',
                                         'Concentration', 'Glomerulus number','Connection type', "KC class", 'Odor id',
                                         "Activation ratio"])
            pooled_data_df.to_csv(f"{path}Activation ratio of KC major subtype.csv")
            pooled_data_df = pd.read_csv(f"{path}Activation ratio of KC major subtype.csv")
            for G_num in [4,7]:
                condition_4 = pooled_data_df['Glomerulus number'] == G_num
                wiring_pattern = connection_style
                condition_1 = pooled_data_df['Wiring pattern'] == wiring_pattern
                for stimulation_type in self.odor_generator.Odor_collection_dict:
                    condition_2 = pooled_data_df['Stimulation type'] == stimulation_type
                    Acuity_result = []
                    for subtype in KC_type_collection:
                        condition_3 = pooled_data_df["KC class"] == subtype
                        tmp_data = pooled_data_df[condition_1 & condition_2 & condition_3 & condition_4]
                        network_id_list = tmp_data['Network id'].drop_duplicates()
                        for network_id in network_id_list:
                            mask_5 = tmp_data['Network id'] == network_id
                            file_name = f'{wiring_pattern}_{stimulation_type}_{subtype}_{G_num}_{network_id}'
                            data = tmp_data[mask_5]
                            sns.lineplot(data=data,
                                         x='Concentration', y='Activation ratio', hue='Odor class',
                                         palette=self.color_dict)
                            plt.title(f"{rename_KC_subtype(subtype)}".replace("KC", ""),
                                      fontdict=self.fontdict['Title'])
                            plt.xlabel('Concentration', fontdict=self.fontdict['Label'])
                            plt.ylabel('Activation ratio', fontdict=self.fontdict['Label'])
                            plt.legend(fontsize=self.fontdict['Legend']['fontsize'], loc='upper right')
                            plt.xticks(fontsize=self.fontdict['Tick']['fontsize'])
                            plt.yticks([0.0, 0.2, 0.4], fontsize=self.fontdict['Tick']['fontsize'])
                            plt.tight_layout()
                            plt.savefig(f"{path}{file_name}_Activation ratio.png", dpi=600)
                            plt.close()
                            # get acuity
                            # if wiring_pattern =='FlyEM':
                            if stimulation_type:
                                for odor_class in ["Odor class 1", "Odor class 2", "Odor class 3"]:
                                    mask_odor_class = data['Odor class'] == odor_class
                                    odor_class_data = data[mask_odor_class]
                                    for odor_id in range(
                                            len(self.odor_generator.Odor_collection_dict['single_1'][G_num][0])):
                                        mask_odor_id = odor_class_data['Odor id'] == odor_id
                                        single_odor_data = odor_class_data[mask_odor_id]
                                        if len(single_odor_data) == 0:
                                            continue
                                        concentration_list = []
                                        activation_ratio_list = []
                                        for concentration, activation_ratio in zip(single_odor_data['Concentration'],
                                                                                   single_odor_data[
                                                                                       'Activation ratio']):
                                            # print('#',concentration,activation_ratio)
                                            concentration_list.append(concentration)
                                            activation_ratio_list.append(activation_ratio)
                                        print(odor_class, stimulation_type, subtype, wiring_pattern, odor_id)
                                        print(concentration_list)
                                        print(activation_ratio_list)
                                        fitting_linear_regression.model_fitting(x=np.array(concentration_list),
                                                                                y=np.array(activation_ratio_list))
                                        Acuity_result.append(
                                            [subtype, network_id, odor_class, odor_id, fitting_linear_regression.slope])
                    # if stimulation_type == 'random':
                    #     continue
                    Acuity_result_df = Df(data=Acuity_result,
                                          columns=['KC class','Network id', "Odor class", "Odor id", "Acuity"])
                    Acuity_result_df.to_csv(f'{path}{wiring_pattern}_{stimulation_type}_{G_num}_acuity.csv')
                    Acuity_result_df = pd.read_csv(f'{path}{wiring_pattern}_{stimulation_type}_{G_num}_acuity.csv')
                    ax = sns.barplot(data=Acuity_result_df, x='KC class', y='Acuity', hue='Odor class',
                                     palette=self.color_dict)
                    plt.ylabel("Acuity", fontdict=self.fontdict['Label'])
                    plt.xlabel("KC Class", fontdict=self.fontdict['Label'])
                    ax.set_xticklabels(
                        [f"{rename_KC_subtype(string.get_text())}" for string in ax.get_xticklabels()],
                        fontdict=self.fontdict['Tick'])
                    # plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=self.fontdict['Tick']['fontsize'])
                    # plt.legend(fontsize=self.fontdict['Legend']['fontsize'], loc='upper right')
                    # plt.tight_layout()
                    # plt.savefig(f'{path}{wiring_pattern}_{stimulation_type}_{G_num}_acuity.png', dpi=600)
                    plt.show()
                    # plt.close()
                    # if wiring_pattern == 'Labeled-line network':
                    #     continue

                    # aov = pg.rm_anova(data=Acuity_result_df, subject='Odor id', dv='Acuity',
                    #                   within=['KC class', 'Odor class'])
                    # pt = pg.pairwise_ttests(data=Acuity_result_df, subject='Odor id', dv='Acuity',
                    #                         within=['KC class', 'Odor class'])
                    # aov.to_excel(f'{path}{wiring_pattern}_{stimulation_type}_{G_num}_acuity_rmanova.xlsx')
                    # pt.to_excel(f'{path}{wiring_pattern}_{stimulation_type}_{G_num}_acuity_posthoc.xlsx')

                    # if wiring_pattern == 'FlyEM':
                    #     print(file_name)
                    #     aov = pg.rm_anova(data=data,subject='Odor id', dv='Activation ratio', within=['Concentration', 'Odor class'])
                    #     pt = pg.pairwise_ttests(data=data,subject='Odor id', dv='Activation ratio', within=['Concentration', 'Odor class'])
                    #     aov.to_excel(f"{path}{file_name}_Activation ratio_rmanova.xlsx")
                    #     pt.to_excel(f"{path}{file_name}_Activation ratio_posthoc.xlsx")

                    # aovrm2way = AnovaRM(data=data, depvar='rt', subject='Sub_id', within=['Concentration', 'Odor class'])
                    # res2way = aovrm2way.fit()
                    # print(res2way)
                    # tukey = pairwise_tukeyhsd(endog=df['count'], groups=df['spray'], alpha=0.05)
                    # print(tukey)

    def analyze_single_KC_activated_ratio(self, KC_classification='Major'):
        '''
        K-S test
        This is used to compare independent KC populations.
        That is to compare same condition but for different KC population (KCab vs KCg)
        We may need to check whether to develop several k-sample K-S test
        https://www.jstor.org/stable/pdf/2238210.pdf?refreqid=excelsior%3Af7f37b9b45f404b3829159c0321a9c4e&ab_segments=&origin=

        (Related-Samples) Friedman's Two-way Analysis of Variance by Ranks
        This is used to compare activation times of same KC population with different conditions (dependent data)
        The package can compare more than two conditions so that different concentration is fine.
        :return:
        '''
        # connection_style, network_id, odor_type, class_id, concentration
        ## start to construct Dataframe
        if KC_classification == 'Major':
            KC_types = self.network.Subtype_to_KCid
        elif KC_classification == 'All':
            KC_types = ['KC']
        elif KC_classification == 'Minor':
            KC_types = self.network.New_subtype_to_id
        path = f"{self.root}Single_KC_Activation ratio/"
        if not os.path.isdir(path): os.mkdir(path)

        pooled_data = [list(parameter) + [subtype, KC_id, activation_times]
                       for parameter in self.result_dict
                       for subtype in KC_types
                       for KC_id, activation_times in
                       enumerate(self.result_dict[parameter]['single KC activated frequency'][subtype])
                       ]
        for i in range(len(pooled_data)):
            pooled_data[i][3] = f"Odor class {pooled_data[i][3] + 1}"
        pooled_data_df = Df(data=pooled_data,
                            columns=["Wiring pattern", "Network id", 'Stimulation type', 'Odor class', 'Concentration',
                                     "KC class", 'KC id', "Activation times"])
        pooled_data_df.to_csv(f"{path}Activation frequency of KC {KC_classification} subtype.csv")
        pooled_data_df = pd.read_csv(f"{path}Activation frequency of KC {KC_classification} subtype.csv")
        '''
        1. plot FlyEM with three odor class (single)
        2. plot three network structure with pooled odor(single)

        '''
        ## 1. Compare network with pooled odor
        mask_concentration = pooled_data_df['Concentration'] == 1.0
        mask_networkid = pooled_data_df['Network id'] == 0
        for subtype in KC_types:
            for stimulation_type in ['single', 'preferred', 'random']:
                mask_stimulationtype = pooled_data_df['Stimulation type'] == stimulation_type
                mask_subtype = pooled_data_df['KC class'] == subtype
                tmp = pooled_data_df[mask_concentration & mask_networkid & mask_subtype & mask_stimulationtype]
                sns.kdeplot(data=tmp,
                            x='Activation times',
                            hue="Wiring pattern",
                            palette={'FlyEM': 'k', 'Labeled-line network': 'darkorange', 'Random network': 'royalblue'},
                            cut=0,
                            # bw_adjust=.5
                            )
                plt.ylabel("Density", fontdict=self.fontdict['Label'])
                plt.yticks(fontsize=self.fontdict['Tick']['fontsize'])
                plt.xlabel("Activation times", fontdict=self.fontdict['Label'])
                plt.xticks([0, 250, 500], fontsize=self.fontdict['Tick']['fontsize'])
                # plt.legend(fontsize=self.fontdict['Legend']['fontsize'], loc='upper right')
                plt.tight_layout()
                plt.savefig(f"{path}network_{subtype}_{stimulation_type}_pooled_odor_concen_1.0.png")
                plt.close()

        ## 2. Compare stimulation type with pooled odor
        for subtype in KC_types:
            for wiring_pattern in self.network.connection_matrix_normalized_collection_dict:
                mask_subtype = pooled_data_df['KC class'] == subtype
                mask_wiringpattern = pooled_data_df['Wiring pattern'] == wiring_pattern
                tmp = pooled_data_df[mask_concentration & mask_networkid & mask_subtype & mask_wiringpattern]
                sns.kdeplot(data=tmp,
                            x='Activation times',
                            hue="Stimulation type",
                            palette={'single': 'k', 'preferred': 'darkorange', 'random': 'royalblue'},
                            cut=0,
                            # bw_adjust=.5
                            )
                plt.ylabel("Density", fontdict=self.fontdict['Label'])
                plt.yticks(fontsize=self.fontdict['Tick']['fontsize'])
                plt.xlabel("Activation times", fontdict=self.fontdict['Label'])
                plt.xticks([0, 250, 500], fontsize=self.fontdict['Tick']['fontsize'])
                # plt.legend(fontsize=self.fontdict['Legend']['fontsize'], loc='upper right')
                plt.tight_layout()
                plt.savefig(f"{path}stimulation_{subtype}_{wiring_pattern}_pooled_odor_concen_1.0.png")
                plt.close()

        ## 3. Compare odor group
        for subtype in KC_types:
            for wiring_pattern in self.network.connection_matrix_normalized_collection_dict:
                for stimulation_type in ['single', 'preferred']:
                    mask_subtype = pooled_data_df['KC class'] == subtype
                    mask_wiringpattern = pooled_data_df['Wiring pattern'] == wiring_pattern
                    mask_stimulationtype = pooled_data_df['Stimulation type'] == stimulation_type
                    tmp = pooled_data_df[
                        mask_concentration & mask_networkid & mask_subtype & mask_wiringpattern & mask_stimulationtype]
                    sns.kdeplot(data=tmp,
                                x='Activation times',
                                hue="Odor class",
                                palette={'Odor class 1': 'r', 'Odor class 2': 'gold', 'Odor class 3': 'deepskyblue'},
                                cut=0,
                                # bw_adjust=.5
                                )
                    plt.ylabel("Density", fontdict=self.fontdict['Label'])
                    plt.yticks(fontsize=self.fontdict['Tick']['fontsize'])
                    plt.xlabel("Activation times", fontdict=self.fontdict['Label'])
                    plt.xticks([0, 250, 500], fontsize=self.fontdict['Tick']['fontsize'])
                    # plt.legend(fontsize=self.fontdict['Legend']['fontsize'], loc='upper right')
                    plt.tight_layout()
                    plt.savefig(f"{path}Odor class_{subtype}_{wiring_pattern}_{stimulation_type}_concen_1.0.png")
                    plt.close()
        ## 4. Compare concentration
        color_list = ['Reds', 'Wistia', 'Blues']
        for subtype in KC_types:
            for wiring_pattern in self.network.connection_matrix_normalized_collection_dict:
                for stimulation_type in ['single', 'preferred']:
                    for odor_class_id, odor_class in enumerate(['Odor class 1', 'Odor class 2', 'Odor class 3']):
                        mask_subtype = pooled_data_df['KC class'] == subtype
                        mask_wiringpattern = pooled_data_df['Wiring pattern'] == wiring_pattern
                        mask_stimulationtype = pooled_data_df['Stimulation type'] == stimulation_type
                        mask_odorclass = pooled_data_df['Odor class'] == odor_class
                        tmp = pooled_data_df[
                            mask_odorclass & mask_networkid & mask_subtype & mask_wiringpattern & mask_stimulationtype]
                        palette = color_list[odor_class_id]
                        sns.kdeplot(data=tmp,
                                    x='Activation times',
                                    hue="Concentration",
                                    palette=palette,
                                    cut=0,
                                    # bw_adjust=.5
                                    )
                        plt.ylabel("Density", fontdict=self.fontdict['Label'])
                        plt.yticks(fontsize=self.fontdict['Tick']['fontsize'])
                        plt.xlabel("Activation times", fontdict=self.fontdict['Label'])
                        plt.xticks([0, 250, 500], fontsize=self.fontdict['Tick']['fontsize'])
                        # plt.legend(fontsize=self.fontdict['Legend']['fontsize'], loc='upper right')
                        plt.tight_layout()
                        plt.savefig(
                            f"{path}Concentration_{subtype}_{wiring_pattern}_{stimulation_type}_{odor_class}.png")
                        plt.close()
        return

    def analyze_single_KC_activated_ratio_without_zero(self, KC_classification='Major', concentration=1.0):
        '''
        K-S test
        This is used to compare independent KC populations.
        That is to compare same condition but for different KC population (KCab vs KCg)
        We may need to check whether to develop several k-sample K-S test
        https://www.jstor.org/stable/pdf/2238210.pdf?refreqid=excelsior%3Af7f37b9b45f404b3829159c0321a9c4e&ab_segments=&origin=

        (Related-Samples) Friedman's Two-way Analysis of Variance by Ranks
        This is used to compare activation times of same KC population with different conditions (dependent data)
        The package can compare more than two conditions so that different concentration is fine.
        :return:
        '''
        # connection_style, network_id, odor_type, class_id, concentration
        ## start to construct Dataframe
        if KC_classification == 'Major':
            KC_types = self.network.Subtype_to_KCid
        elif KC_classification == 'All':
            KC_types = ['KC']
        elif KC_classification == 'Minor':
            KC_types = self.network.New_subtype_to_id
        path = f"{self.root}Single_KC_Activation ratio_without_zero/"
        if not os.path.isdir(path): os.mkdir(path)

        pooled_data = [list(parameter) + [subtype, KC_id, activation_times]
                       for parameter in self.result_dict
                       for subtype in KC_types
                       for KC_id, activation_times in
                       enumerate(self.result_dict[parameter]['single KC activated frequency'][subtype])
                       if activation_times != 0
                       ]
        for i in range(len(pooled_data)):
            pooled_data[i][3] = f"Odor class {pooled_data[i][3] + 1}"
        pooled_data_df = Df(data=pooled_data,
                            columns=["Wiring pattern", "Network id", 'Stimulation type', 'Odor class', 'Concentration',
                                     "KC class", 'KC id', "Activation times"])
        pooled_data_df.to_csv(f"{path}Activation frequency of KC {KC_classification} subtype.csv")
        pooled_data_df = pd.read_csv(f"{path}Activation frequency of KC {KC_classification} subtype.csv")
        '''
        1. plot FlyEM with three odor class (single)
        2. plot three network structure with pooled odor(single)

        '''
        ## 1. Compare network with pooled odor
        mask_concentration = pooled_data_df['Concentration'] == concentration
        mask_networkid = pooled_data_df['Network id'] == 0
        for subtype in KC_types:
            for stimulation_type in ['single', 'preferred', 'random']:
                mask_stimulationtype = pooled_data_df['Stimulation type'] == stimulation_type
                mask_subtype = pooled_data_df['KC class'] == subtype
                tmp = pooled_data_df[mask_concentration & mask_networkid & mask_subtype & mask_stimulationtype]
                sns.kdeplot(data=tmp,
                            x='Activation times',
                            hue="Wiring pattern",
                            palette={'FlyEM': 'k', 'Labeled-line network': 'darkorange', 'Random network': 'royalblue'},
                            cut=0,
                            bw_adjust=.5
                            )
                plt.ylabel("Density", fontdict=self.fontdict['Label'])
                plt.yticks(fontsize=self.fontdict['Tick']['fontsize'])
                plt.xlabel("Activation times", fontdict=self.fontdict['Label'])
                plt.xticks([0, 250, 500], fontsize=self.fontdict['Tick']['fontsize'])
                # plt.legend(fontsize=self.fontdict['Legend']['fontsize'], loc='upper right')
                plt.tight_layout()
                plt.savefig(f"{path}network_{subtype}_{stimulation_type}_pooled_odor_concen_{concentration}.png")
                plt.close()

        ## 2. Compare stimulation type with pooled odor
        for subtype in KC_types:
            for wiring_pattern in self.network.connection_matrix_normalized_collection_dict:
                mask_subtype = pooled_data_df['KC class'] == subtype
                mask_wiringpattern = pooled_data_df['Wiring pattern'] == wiring_pattern
                tmp = pooled_data_df[mask_concentration & mask_networkid & mask_subtype & mask_wiringpattern]
                sns.kdeplot(data=tmp,
                            x='Activation times',
                            hue="Stimulation type",
                            palette={'single': 'k', 'preferred': 'darkorange', 'random': 'royalblue'},
                            cut=0,
                            bw_adjust=.5
                            )
                plt.ylabel("Density", fontdict=self.fontdict['Label'])
                plt.yticks(fontsize=self.fontdict['Tick']['fontsize'])
                plt.xlabel("Activation times", fontdict=self.fontdict['Label'])
                plt.xticks([0, 250, 500], fontsize=self.fontdict['Tick']['fontsize'])
                # plt.legend(fontsize=self.fontdict['Legend']['fontsize'], loc='upper right')
                plt.tight_layout()
                plt.savefig(f"{path}stimulation_{subtype}_{wiring_pattern}_pooled_odor_concen_{concentration}.png")
                plt.close()

        ## 3. Compare odor group
        for subtype in KC_types:
            for wiring_pattern in self.network.connection_matrix_normalized_collection_dict:
                for stimulation_type in ['single', 'preferred']:
                    mask_subtype = pooled_data_df['KC class'] == subtype
                    mask_wiringpattern = pooled_data_df['Wiring pattern'] == wiring_pattern
                    mask_stimulationtype = pooled_data_df['Stimulation type'] == stimulation_type
                    tmp = pooled_data_df[
                        mask_concentration & mask_networkid & mask_subtype & mask_wiringpattern & mask_stimulationtype]
                    sns.kdeplot(data=tmp,
                                x='Activation times',
                                hue="Odor class",
                                palette={'Odor class 1': 'r', 'Odor class 2': 'gold', 'Odor class 3': 'deepskyblue'},
                                cut=0,
                                bw_adjust=.5
                                )
                    plt.ylabel("Density", fontdict=self.fontdict['Label'])
                    plt.yticks(fontsize=self.fontdict['Tick']['fontsize'])
                    plt.xlabel("Activation times", fontdict=self.fontdict['Label'])
                    plt.xticks([0, 250, 500], fontsize=self.fontdict['Tick']['fontsize'])
                    # plt.legend(fontsize=self.fontdict['Legend']['fontsize'], loc='upper right')
                    plt.tight_layout()
                    plt.savefig(
                        f"{path}Odor class_{subtype}_{wiring_pattern}_{stimulation_type}_concen_{concentration}.png")
                    plt.close()
        ## 4. Compare concentration
        color_list = ['Reds', 'Wistia', 'Blues']
        for subtype in KC_types:
            for wiring_pattern in self.network.connection_matrix_normalized_collection_dict:
                for stimulation_type in ['single', 'preferred']:
                    for odor_class_id, odor_class in enumerate(['Odor class 1', 'Odor class 2', 'Odor class 3']):
                        mask_subtype = pooled_data_df['KC class'] == subtype
                        mask_wiringpattern = pooled_data_df['Wiring pattern'] == wiring_pattern
                        mask_stimulationtype = pooled_data_df['Stimulation type'] == stimulation_type
                        mask_odorclass = pooled_data_df['Odor class'] == odor_class
                        tmp = pooled_data_df[
                            mask_odorclass & mask_networkid & mask_subtype & mask_wiringpattern & mask_stimulationtype]
                        palette = color_list[odor_class_id]
                        sns.kdeplot(data=tmp,
                                    x='Activation times',
                                    hue="Concentration",
                                    palette=palette,
                                    cut=0,
                                    bw_adjust=.5
                                    )
                        plt.ylabel("Density", fontdict=self.fontdict['Label'])
                        plt.yticks(fontsize=self.fontdict['Tick']['fontsize'])
                        plt.xlabel("Activation times", fontdict=self.fontdict['Label'])
                        plt.xticks([0, 250, 500], fontsize=self.fontdict['Tick']['fontsize'])
                        # plt.legend(fontsize=self.fontdict['Legend']['fontsize'], loc='upper right')
                        plt.tight_layout()
                        plt.savefig(
                            f"{path}Concentration_{subtype}_{wiring_pattern}_{stimulation_type}_{odor_class}.png")
                        plt.close()
        return

    def analyze_coresponding_frequency(self):
        123
        return

    def calculate_dimension(self, pca):
        dimension = 0
        s = 0
        m = 0
        for i in pca.explained_variance_:
            s = s + i
            m = m + i ** 2
        dimension = s ** 2 / m
        return dimension

    #     def analyze_dimensionality(self,setting_parameters,colors,file_name,subtype_analysis="",heatmap=False):
    # #        (connection_style, network_id, odor_type, class_id, concentration, activated_glomerulus_num)
    #         path = f"{self.root}PCA_code/"
    #         if not os.path.isdir(path): os.mkdir(path)
    #         pooled_result = []
    #         color_list = []
    #         for parameter,color in zip(setting_parameters,colors):
    #             tmp_data = self.response_pooled_dict[parameter][parameter].tolist()
    #             pooled_result += tmp_data
    #             color_list += [color for _ in range(len(tmp_data))]
    #         pca = PCA(n_components=2)
    #         pooled_result = np.array(np.array(pooled_result))
    #         if heatmap == True:
    #             sns.heatmap(data=pooled_result)
    #             plt.savefig(file_name+'_heatmap.png')
    #             plt.close()
    #         X_pca = pca.fit_transform(pooled_result)
    #         # Data Visualization
    #         x_min, x_max = X_pca.min(0), X_pca.max(0)
    #         X_norm = (X_pca - x_min) / (x_max - x_min)  # Normalize
    #         plt.figure(figsize=(8, 8))
    #         plt.scatter(X_norm[:,0],X_norm[:,1],color=color_list)
    #         plt.savefig(f'{path}{file_name}.png',dpi=600)
    #         plt.close()
    #         pca = PCA(n_components=np.min(pooled_result.shape))
    #         pca.fit_transform(pooled_result)
    #         pooled_dimensionality = self.calculate_dimension(pca)
    #         if subtype_analysis == "main class":
    #             for subtype in self.network.Subtype_to_KCid:
    #                 pooled_result = []
    #                 for i in range(3):
    #                     pooled_result += self.response_pooled_dict[('FlyEM', 0, 'single', i, 1.0)][
    #                                          ('FlyEM', 0, 'single', i, 1.0)][:,
    #                                      self.network.KC_subtype_location[subtype][0]:
    #                                      self.network.KC_subtype_location[subtype][1] + 1].tolist()
    #                 color_list = ['r' for i in range(1000)] + [
    #                     'gold' for i in range(1000)] + [
    #                                  'deepskyblue' for i in range(1000)]
    #                 pca = PCA(n_components=2)
    #                 X_pca = pca.fit_transform(np.array(pooled_result))
    #                 # Data Visualization
    #                 x_min, x_max = X_pca.min(0), X_pca.max(0)
    #                 X_norm = (X_pca - x_min) / (x_max - x_min)  # Normalize
    #                 plt.figure(figsize=(8, 8))
    #                 plt.scatter(X_norm[:, 0], X_norm[:, 1], color=color_list)
    #                 plt.savefig(f'{path}FlyEM_{subtype}_single_pooled_{concentration}.png', dpi=600)
    #                 plt.close()
    #         return pooled_dimensionality
    def get_Dimensionality(self, dir, wiring_pattern, stimulation_type, network_id,
                           concentration_list=[0.8, 0.9, 1.0], glomerulus_number_list=[4, 7],
                           cluster_list=[0, 1, 2],connection_type='original_weight'):
        collection_dict = {}
        code_merge_dict = {}
        KCg_num = 607
        KCapbp_num = 335
        KCab_num = 803
        total_KC = 67 + 335 + 803
        Dimensionality_pooled_data = []
        for G_num in glomerulus_number_list:
            for class_id in cluster_list:
                for concentration in concentration_list:
                    # wiring_pattern = 'Random network'
                    # data = pd.read_excel(f"{dir}FlyEM_0_single_{class_id}_{concentration}_4_KC_response.xlsx")
                    try:
                        data = pd.read_excel(
                            f"{dir}{wiring_pattern}_{network_id}_{stimulation_type}_{class_id}_{concentration}_{G_num}_{connection_type}_KC_response.xlsx")
                    except:
                        continue
                    data = data.to_numpy()[:, 1:]
                    # print(data)
                    data[data > 0] = 1
                    print(data.shape)
                    activated_ratio = np.sum(data, axis=1) / total_KC
                    plt.hist(x=activated_ratio)
                    plt.title(f"Concentration {concentration}")
                    plt.xlabel("Activated ratio")
                    plt.savefig(
                        f"{dir}{wiring_pattern}_{network_id}_{stimulation_type}_{class_id}_{concentration}_{G_num}_total_activated_ratio.png")
                    plt.close()
                    KCg_activated_ratio = np.sum(data[:, :KCg_num], axis=1) / KCg_num
                    KCapbp_activated_ratio = np.sum(data[:, KCg_num:KCg_num + KCapbp_num], axis=1) / KCapbp_num
                    KCab_activated_ratio = np.sum(data[:, KCg_num + KCapbp_num:], axis=1) / KCab_num
                    plt.subplots(ncols=1, nrows=3)
                    plt.subplot(311)
                    plt.hist(x=KCg_activated_ratio)
                    plt.xticks([])
                    plt.title(f"Concentration {concentration}", fontdict={"fontsize": 20})
                    # plt.xlabel("Activated ratio")
                    plt.subplot(312)
                    plt.hist(x=KCapbp_activated_ratio)
                    plt.xticks([])
                    # plt.title(f"{concentration}")
                    # plt.xlabel("Activated ratio")
                    plt.subplot(313)
                    plt.hist(x=KCab_activated_ratio)
                    # plt.title(f"{concentration}")
                    plt.xlabel("Activated ratio", fontdict={"fontsize": 14})
                    plt.savefig(
                        f"{dir}{wiring_pattern}_{network_id}_{stimulation_type}_{class_id}_{concentration}_{G_num}_Subtype_activated_ratio.png")
                    # plt.show()
                    plt.close()
                    collection_dict[(class_id, concentration)] = data
                    pca = PCA()
                    pca.fit_transform(np.array(data))
                    dimension = self.calculate_dimension(pca)
                    print(dimension)
                    Dimensionality_pooled_data.append(
                        [G_num, network_id, class_id, concentration, 'Pooled', dimension])  ##Pooled KC class

            data_pooled = []
            print(collection_dict.keys())
            # input("#FRQ")
            # for concentration in [0.8,0.9,1.0]:
            for concentration in [0.8]:
                code_merge_dict["KCg"] = list(collection_dict[(0, concentration)][:, :KCg_num]) + list(
                    collection_dict[(1, concentration)][:, :KCg_num]) + \
                                         list(collection_dict[(2, concentration)][:, :KCg_num])
                code_merge_dict["KCa'b'"] = list(
                    collection_dict[(0, concentration)][:, KCg_num:KCg_num + KCapbp_num]) + list(
                    collection_dict[(1, concentration)][:, KCg_num:KCg_num + KCapbp_num]) + \
                                            list(collection_dict[(2, concentration)][:, KCg_num:KCg_num + KCapbp_num])
                code_merge_dict["KCab"] = list(collection_dict[(0, concentration)][:, KCg_num + KCapbp_num:]) + list(
                    collection_dict[(1, concentration)][:, KCg_num + KCapbp_num:]) + \
                                          list(collection_dict[(2, concentration)][:, KCg_num + KCapbp_num:])
                print("Dimension")
                pca = PCA()
                try:
                    pca.fit_transform(np.array(code_merge_dict["KCg"]))
                    dimension = self.calculate_dimension(pca)
                except:
                    dimension = 0
                print(dimension)
                Dimensionality_pooled_data.append([G_num, network_id, "Pooled", concentration, 'KCg', dimension])
                try:
                    pca.fit_transform(np.array(code_merge_dict["KCa'b'"]))
                    dimension = self.calculate_dimension(pca)
                except:
                    dimension = 0
                Dimensionality_pooled_data.append([G_num, network_id, "Pooled", concentration, "KCa'b'", dimension])
                print(dimension)
                try:
                    pca.fit_transform(np.array(code_merge_dict["KCab"]))
                    dimension = self.calculate_dimension(pca)
                except:
                    dimension = 0
                Dimensionality_pooled_data.append([G_num, network_id, "Pooled", concentration, "KCab", dimension])
                print(dimension)

                ###
                print("KCg")
                try:
                    pca.fit_transform(np.array(code_merge_dict["KCg"][0:999]))
                    dimension = self.calculate_dimension(pca)
                except:
                    dimension = 0
                Dimensionality_pooled_data.append([G_num, network_id, 1, concentration, 'KCg', dimension])
                print(dimension)
                try:
                    pca.fit_transform(np.array(code_merge_dict["KCg"][1000:1999]))
                    dimension = self.calculate_dimension(pca)
                except:
                    dimension = 0
                Dimensionality_pooled_data.append([G_num, network_id, 2, concentration, 'KCg', dimension])
                print(dimension)
                try:
                    pca.fit_transform(np.array(code_merge_dict["KCg"][2000:2999]))
                    dimension = self.calculate_dimension(pca)
                except:
                    dimension = 0
                Dimensionality_pooled_data.append([G_num, network_id, 3, concentration, 'KCg', dimension])
                print(dimension)
                ####
                print("KCa'b'")
                try:
                    pca.fit_transform(np.array(code_merge_dict["KCa'b'"][0:999]))
                    dimension = self.calculate_dimension(pca)
                except:
                    dimension = 0
                Dimensionality_pooled_data.append([G_num, network_id, 1, concentration, "KCa'b'", dimension])
                print(dimension)
                try:
                    pca.fit_transform(np.array(code_merge_dict["KCa'b'"][1000:1999]))
                    dimension = self.calculate_dimension(pca)
                except:
                    dimension = 0
                Dimensionality_pooled_data.append([G_num, network_id, 2, concentration, "KCa'b'", dimension])
                print(dimension)
                try:
                    pca.fit_transform(np.array(code_merge_dict["KCa'b'"][2000:2999]))
                    dimension = self.calculate_dimension(pca)
                except:
                    dimension = 0
                Dimensionality_pooled_data.append([G_num, network_id, 3, concentration, "KCa'b'", dimension])
                print(dimension)
                ####
                print("KCab")
                try:
                    pca.fit_transform(np.array(code_merge_dict["KCab"][0:999]))
                    dimension = self.calculate_dimension(pca)
                except:
                    dimension = 0
                Dimensionality_pooled_data.append([G_num, network_id, 1, concentration, "KCab", dimension])
                print(dimension)
                try:
                    pca.fit_transform(np.array(code_merge_dict["KCab"][1000:1999]))
                    dimension = self.calculate_dimension(pca)
                except:
                    dimension = 0
                Dimensionality_pooled_data.append([G_num, network_id, 2, concentration, "KCab", dimension])
                print(dimension)
                try:
                    pca.fit_transform(np.array(code_merge_dict["KCab"][2000:2999]))
                    dimension = self.calculate_dimension(pca)
                except:
                    dimension = 0
                Dimensionality_pooled_data.append([G_num, network_id, 3, concentration, "KCab", dimension])
                print(dimension)
            #     code_difference = (collection_dict[(0, 1.2)] - collection_dict[(0, 1.0)])
            #     print("odor 0")
            #
            #     KCg_difference = np.sum(code_difference[:, :KCg_num], axis=1)
            #     KCab_difference = np.sum(code_difference[:, KCg_num + KCapbp_num:], axis=1)
            #     KCapbp_difference = np.sum(code_difference[:, KCg_num:KCg_num + KCapbp_num], axis=1)
            #     # print(KCg_difference)
            #     for i in range(KCg_difference.shape[0]):
            #         data_pooled.append(["KCg", 1, KCg_difference[i]])
            #     for i in range(KCab_difference.shape[0]):
            #         data_pooled.append(["KCab", 1, KCab_difference[i]])
            #     for i in range(KCapbp_difference.shape[0]):
            #         data_pooled.append(["KCa'b'", 1, KCab_difference[i]])
            #
            #     code_difference = (collection_dict[(1, 1.2)] - collection_dict[(1, 1.0)])
            #     print("odor 1")
            #     KCg_difference = np.sum(code_difference[:, :KCg_num], axis=1)
            #     KCab_difference = np.sum(code_difference[:, KCg_num + KCapbp_num:], axis=1)
            #     KCapbp_difference = np.sum(code_difference[:, KCg_num:KCg_num + KCapbp_num], axis=1)
            #     # print(KCg_difference)
            #     for i in range(KCg_difference.shape[0]):
            #         data_pooled.append(["KCg", 2, KCg_difference[i]])
            #     for i in range(KCab_difference.shape[0]):
            #         data_pooled.append(["KCab", 2, KCab_difference[i]])
            #     for i in range(KCapbp_difference.shape[0]):
            #         data_pooled.append(["KCa'b'", 2, KCab_difference[i]])
            #
            #     code_difference = (collection_dict[(2, 1.2)] - collection_dict[(2, 1.0)])
            #     KCg_difference = np.sum(code_difference[:, :KCg_num], axis=1)
            #     KCab_difference = np.sum(code_difference[:, KCg_num + KCapbp_num:], axis=1)
            #     KCapbp_difference = np.sum(code_difference[:, KCg_num:KCg_num + KCapbp_num], axis=1)
            #
            #     for i in range(KCg_difference.shape[0]):
            #         data_pooled.append(["KCg", 3, KCg_difference[i]])
            #     for i in range(KCab_difference.shape[0]):
            #         data_pooled.append(["KCab", 3, KCab_difference[i]])
            #     for i in range(KCapbp_difference.shape[0]):
            #         data_pooled.append(["KCa'b'", 3, KCab_difference[i]])
            #
            #     data_pooled = Df(data=np.array(data_pooled), columns=['KC class', 'Odor class', 'Hamming distance'])
            #     data_pooled.to_excel("pooled_data.xlsx")
            #     data_pooled = pd.read_excel("pooled_data.xlsx")
            #     sns.boxplot(data=data_pooled, x='Odor class', y='Hamming distance', hue='KC class',
            #                 hue_order=['KCg', "KCa'b'", "KCab"])
            #     plt.show()
            #
            #
        Dimensionality_pooled_data = Df(data=np.array(Dimensionality_pooled_data),
                                        columns=["Glomerulus num", "Network id", "Odor class", "Concentration",
                                                 "KC class",
                                                 "Dimensionality"])
        Dimensionality_pooled_data.to_excel(f"{dir}Pooled_data_{network_id}_{wiring_pattern}_{stimulation_type}.xlsx")
        return

    def get_dimension_normalized_data(self, tmp_data):
        mask_wiring = tmp_data['Wiring pattern'] == 'Random'
        normalization_dict = {}
        for KC_class in [r"$\alpha'\beta'$", r"$\alpha\beta$", r"$\gamma$"]:
            mask_KC = tmp_data['KC class'] == KC_class
            normalization_dict[KC_class] = np.average(tmp_data[mask_wiring & mask_KC]["Representation dimension"])

    def analyze_dimensionality(self, dir='simulation_result_threshold_1_weight_3/',connection_type='original_weight', new_calculation=False):
        stimulation_protocol = sim.load_artificial_odor()
        network = gc.load_network()
        palette = {r"$\gamma$": 'red', r"$\alpha'\beta'$": "gold", r"$\alpha\beta$": "deepskyblue", "Pooled": 'black'}
        if not os.path.isfile(f"{dir}Pooled_data_All.csv") or new_calculation:
            pooled_data = []
            for stimulation_type in stimulation_protocol.Odor_collection_dict:
                for wiring_pattern in network.connection_matrix_normalized_collection_dict:
                    for network_id in range(len(network.connection_matrix_normalized_collection_dict[wiring_pattern])):
                        if not os.path.isfile(
                                f"{dir}Pooled_data_{wiring_pattern}_{stimulation_type}.xlsx") or new_calculation:
                            try:
                                self.get_Dimensionality(
                                    dir, wiring_pattern, stimulation_type, network_id, 
                                    concentration_list=[0.8,0.9,1.0], glomerulus_number_list=[7],
                                    connection_type=connection_type)
                            except:
                                print(f'No {stimulation_type}_{wiring_pattern}')
                        try:
                            Dimensionality_pooled_data = pd.read_excel(
                                f"{dir}Pooled_data_{network_id}_{wiring_pattern}_{stimulation_type}.xlsx")
                            ######
                            KC_label = Dimensionality_pooled_data['KC class'].values.tolist()
                            new_label = []
                            for i in KC_label:
                                if "a'b'" in i:
                                    new_label.append(r"$\alpha'\beta'$")
                                elif "ab" in i:
                                    new_label.append(r"$\alpha\beta$")
                                elif "g" in i:
                                    new_label.append(r"$\gamma$")
                                else:
                                    new_label.append(i)
                            Dimensionality_pooled_data['KC class'] = new_label
                            palette = {r"$\gamma$": 'red', r"$\alpha'\beta'$": "gold", r"$\alpha\beta$": "deepskyblue"}
                            # sns.catplot(data=Dimensionality_pooled_data[mask2], x='Glomerulus num', y="Dimensionality", hue="KC class",
                            #             kind='bar', ci='sd', row='Concentration',palette=palette)
                            # plt.savefig(f"{dir}{wiring_pattern}ALL_Dimension.png")
                            # plt.close()
                            alias = wiring_pattern.replace("network", "")
                            pooled_data += [i[1:] + [alias] + [stimulation_type] for i in
                                            Dimensionality_pooled_data.values.tolist()]
                        except:
                            pass
            pooled_data = Df(data=np.array(pooled_data),
                             columns=["Glomerulus num", "Network id", "Odor class", "Concentration", "KC class",
                                      "Dimensionality", "Wiring pattern", "Stimulation type"])
            pooled_data.to_csv(f"{dir}Pooled_data_All.csv")
        pooled_data = pd.read_csv(f"{dir}Pooled_data_All.csv")
        pooled_data.rename(columns={'Dimensionality': "Representation dimension"}, inplace=True)
        # for stimulation_type in stimulation_protocol.Odor_collection_dict:
        #     for g_num in [4, 7]:
        #         for concentration in [0.8, 0.9, 1.0]:
        #             for odor_class in [1, 2, 3, "Pooled"]:
        #                 mask = pooled_data['Odor class'] != odor_class
        #                 mask_G = pooled_data["Glomerulus num"] == g_num
        #                 mask_C = pooled_data["Concentration"] == concentration
        #                 mask_S = pooled_data['Stimulation type'] == stimulation_type
        #                 tmp_data = copy.deepcopy(pooled_data[mask_C & mask_G & mask & mask_S])
        #                 tmp_data = self.get_dimension_normalized_data(tmp_data)

        #                 sns.set(font_scale=1.75)
        #                 sns.set_style("ticks")
        #                 # sns.set_palette(sns.color_palette('bright'))
        #                 g = sns.barplot(data=pooled_data[mask & mask_G & mask_C], x='Wiring pattern',
        #                                 y="Representation dimension",
        #                                 hue="KC class", ci='sd', palette=palette,
        #                                 order=['Random', 'FlyEM', 'Labeled-line'])
        #                 plt.legend(loc='upper right', fontsize=self.fontdict['Legend']['fontsize'])
        #                 # plt.title(f"{rename_KC_subtype(subtype)}", fontdict=self.fontdict['Title'])
        #                 plt.xlabel('Wiring pattern', fontdict=self.fontdict['Label'])
        #                 plt.ylabel('Representation dimension',
        #                            fontdict={'fontsize': self.fontdict['Label']['fontsize'] * 0.8})
        #                 plt.legend(fontsize=self.fontdict['Legend']['fontsize'], loc='upper right')
        #                 plt.xticks(ticks=[0, 1, 2], labels=['Random', 'FlyEM', 'Labeled-line'],
        #                            fontsize=self.fontdict['Tick']['fontsize'])
        #                 plt.yticks([0, 40, 80, 120], fontsize=self.fontdict['Tick']['fontsize'])

        #                 # g.set_xticklabels(g.get_xticklabels(), rotation=10)
        #                 plt.tight_layout()
        #                 plt.show()

    def dimensionality_random_odor_shuffle_ratio(self, dir='simulation_result_threshold_1_weight_3/'):
        stimulation_protocol = sim.load_artificial_odor()
        network = gc.load_network()
        pooled_data = pd.read_csv(f"{dir}Pooled_data_All.csv")
        pooled_data.rename(columns={'Dimensionality': "Representation dimension"}, inplace=True)
        stimulation_collection = []
        stimulation_type_collection = []
        for i in pooled_data['Stimulation type']:
            if 'random' in i:
                stimulation_collection.append('random')
            else:
                stimulation_collection.append(i[:-2])
        for i in stimulation_collection:
            if i not in stimulation_type_collection:
                stimulation_type_collection.append(i)
        pooled_data['Stimulation type'] = stimulation_collection
        wiring_pattern = []
        for i in pooled_data['Wiring pattern']:
            if 'Label' in i:
                wiring_pattern.append(-100)
            elif 'Fly' in i:
                wiring_pattern.append(0)
            elif i == "Random ":
                wiring_pattern.append(100)
            else:
                percent = int(round(float(i.split(" ")[-1]), 2) * 100)
                wiring_pattern.append(percent)

        pooled_data['Wiring pattern'] = wiring_pattern
        mask = pooled_data["Wiring pattern"] != -100
        data = pooled_data[mask]
        # data = pooled_data

        mask_s1 = data["Stimulation type"] == "random"
        mask_s2 = data["Stimulation type"] == "preferred"
        mask_s3 = data["Stimulation type"] == "single"

        # print(data['Wiring pattern'])
        # for stimulation_type in stimulation_type_collection:
        # for stimulation_type in ['random']:
        #     for g_num in [4, 7]:
        #         for concentration in [0.8, 0.9, 1.0]:
        #             mask1 = data['Stimulation type'] == stimulation_type
        #             mask2 = data['Concentration'] == concentration
        #             mask3 = data['Glomerulus num'] == g_num
        #             mask4 = data["Odor class"] == "Pooled"
        #             tmp_data = data[mask1 & mask2 & mask3 & mask_s]
        #             print(stimulation_type,concentration,g_num)
        #             sns.lineplot(data=tmp_data[mask4],x ='Wiring pattern',y="Representation dimension",hue='KC class',marker="o",style='')
        #             plt.title(f"{stimulation_type} {concentration} {g_num}")
        #             plt.show()
        KC_subtype_collection = []
        for i in data['KC class']:
            if i not in KC_subtype_collection:
                KC_subtype_collection.append(i)
        print(i)
        for g_num in [7]:
            for concentration in [0.8, 0.9, 1.0]:
                # mask1 = data['Stimulation type'] == stimulation_type
                mask2 = data['Concentration'] == concentration
                mask3 = data['Glomerulus num'] == g_num
                mask4 = data["Odor class"] == "Pooled"
                sns.lineplot(data=data[mask2 & mask3 & mask4 & (mask_s1 | mask_s2 | mask_s3)], x='Wiring pattern',
                             y="Representation dimension", hue='KC class',
                             marker="o", style='Stimulation type')
                # sns.lineplot(data=data[mask2 & mask3 & mask4 ], x='Wiring pattern', y="Representation dimension", hue='KC class',
                #              marker="o", style='Stimulation type')
                plt.show()
                for i in KC_subtype_collection:
                    mask5 = data["KC class"] == i
                    mask7 = data['Stimulation type'] != 'single'
                    sns.lineplot(data=data[mask2 & mask3 & mask4 & mask5 & mask7], x='Wiring pattern',
                                 y="Representation dimension",
                                 marker="o", hue='Stimulation type')
                    plt.title(f"{concentration} {g_num}_{i}")
                    plt.show()

    #######################################
    def analyze_block_activate_experiment(self, root, path, file_list, post_condition=''):
        record_block_dir = 'block_analysis/'
        if not os.path.isdir(record_block_dir): os.mkdir(record_block_dir)
        record_block_dir = record_block_dir + root
        if not os.path.isdir(record_block_dir): os.mkdir(record_block_dir)
        record_block_dir = record_block_dir + path + post_condition
        if not os.path.isdir(record_block_dir): os.mkdir(record_block_dir)
        path = root + path
        if not os.path.isfile(f'{path}Pooled{post_condition}.xlsx'):
            for file_index, file in enumerate(file_list):
                if file_index == 0:
                    data = pd.read_excel(path + file)
                else:
                    data = pd.merge(data, pd.read_excel(path + file), how="outer")
            data.to_excel(f"{path}Pooled{post_condition}.xlsx")
        else:
            data = pd.read_excel(f"{path}Pooled{post_condition}.xlsx")
        # mask = data['Glomerulus number'] == 4
        # data = data[mask]
        palette_dict = {1: 'red', 2: "gold", 3: "deepskyblue"}
        data['Stimulation type'][data['Stimulation type'] == 'shuffled_0'] = 'shuffled'
        data['Stimulation type'][data['Stimulation type'] == 'shuffled_1'] = 'shuffled'
        data['Stimulation type'][data['Stimulation type'] == 'shuffled_2'] = 'shuffled'

        for Concentration in [0.8, 1.0, 1.2]:
            for Stimulation_type in ['single', 'preferred', 'random', 'shuffled']:
                for Wiring_pattern in ['FlyEM', 'Random network', 'Labeled-line network']:
                    for Glomerulus_num in [4, 5]:
                        mask1 = data['Concentration'] == Concentration
                        mask2 = data['Stimulation type'] == Stimulation_type
                        mask3 = data['Wiring pattern'] == Wiring_pattern
                        mask4 = data['Glomerulus number'] == Glomerulus_num
                        plt.figure(figsize=(40, 20))
                        sns.barplot(data=data[mask1 & mask2 & mask3 & mask4], hue='Exp', y='Accuracy', x='Odor class',
                                    ci='sd')
                        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                        plt.savefig(
                            f"{record_block_dir}{Concentration}_{Stimulation_type}_{Wiring_pattern}_{Glomerulus_num}_odor_class.png")
                        # plt.show()
                        plt.close()

    def analyze_interclass_generalization_result(self, data, postfix=''):
        data.rename({'Accuracy': "Generalization score", "Test Odor class": "Test odor class",
                     "Train Odor class": "Train odor class"}, axis='columns', inplace=True)
        data.loc[data['Stimulation type'] == 'shuffled_0', 'Stimulation type'] = 'shuffled'
        data.loc[data['Stimulation type'] == 'shuffled_1', 'Stimulation type'] = 'shuffled'
        data.loc[data['Stimulation type'] == 'shuffled_2', 'Stimulation type'] = 'shuffled'
        print("We need check Exp id")
        data['Exp id'] = [i % 100 for i in range(len(data))]
        palette_dict = {1: 'red', 2: "gold", 3: "deepskyblue"}
        path = 'Generalization/'
        if not os.path.isdir(f"{self.root}{path}"): os.mkdir(f"{self.root}{path}")
        for Concentration in [0.8, 1.0, 1.2]:
            for Stimulation_type in ['single', 'preferred', 'random', 'shuffled']:
                for Wiring_pattern in ['FlyEM', 'Random network', 'Labeled-line network']:
                    mask1 = data['Concentration'] == Concentration
                    mask2 = data['Stimulation type'] == Stimulation_type
                    mask3 = data['Wiring pattern'] == Wiring_pattern
                    mask4 = data['Glomerulus number'] == 4
                    data_analysis = data[mask1 & mask2 & mask3 & mask4]
                    ax = sns.barplot(data=data_analysis, x='Test odor class', y='Generalization score',
                                     hue='Train odor class', palette=palette_dict, ci='sd')
                    plt.ylabel("Generalization score", fontdict={'fontsize': self.fontdict['Label']['fontsize'] * 0.9})
                    plt.xlabel("Test odor class", fontdict={'fontsize': self.fontdict['Label']['fontsize'] * 0.9})
                    ax.set_xticklabels([f"{string.get_text()}" for string in ax.get_xticklabels()],
                                       fontdict={'fontsize': 0.8 * self.fontdict['Tick']['fontsize']})
                    plt.yticks([0.0, 0.5, 1.0], fontsize=self.fontdict['Tick']['fontsize'] * 0.9)
                    # new_title = 'Test odor class'
                    # plt.legend(title='Test odor class', loc='upper left',title_fontsize=20*0.8,fontsize=self.fontdict['Legend']['fontsize']*0.8, ncol=3)
                    plt.legend(title='Train odor class', loc='upper left', title_fontsize=20 * 0.8,
                               fontsize=self.fontdict['Legend']['fontsize'] * 0.9)
                    plt.tight_layout()
                    plt.savefig(
                        f'{self.root}{path}{postfix}{Wiring_pattern}_{Stimulation_type}_{4}_{Concentration}.png',
                        dpi=600)
                    plt.close()
                    for test_odor_class in [1, 2, 3]:
                        mask = data_analysis['Test odor class'] == test_odor_class
                        data_analysis_inference = data_analysis[mask]
                        homo = pg.homoscedasticity(data=data_analysis_inference, dv='Generalization score')
                        aov = pg.rm_anova(data=data_analysis_inference, subject='Exp id', dv='Generalization score',
                                          within='Train odor class')
                        pt = pg.pairwise_ttests(data=data_analysis_inference, subject='Exp id',
                                                dv='Generalization score',
                                                within='Train odor class')

                        aov.to_excel(
                            f'{self.root}{path}{postfix}{Wiring_pattern}_{Stimulation_type}_{4}_{Concentration}_{test_odor_class}_rmanova.xlsx')
                        pt.to_excel(
                            f'{self.root}{path}{postfix}{Wiring_pattern}_{Stimulation_type}_{4}_{Concentration}_{test_odor_class}_posthoc.xlsx')

        # plt.close()
        # sns.catplot(data=data[mask1 & mask4], col='Wiring pattern', y='Generalization score', hue='Test Odor class',
        #            palette=palette_dict, ci='sd', x='Train Odor class', kind="bar", row='Stimulation type')
        # # sns.catplot(data=data[mask1  & mask4], col='Wiring pattern', y='Performance Index', hue='Test Odor class',
        # #            palette=palette_dict, ci='sd', x='Train Odor class',kind="bar",row='Stimulation type')
        #
        # # plt.ylim(0.4,1.2)
        # # plt.yticks([0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        # # plt.subplot(122)
        # # sns.boxplot(data=data[mask1 & mask2 & mask4], x='Wiring pattern', y='Generalization score', hue='Odor class', palette=palette_dict)
        # plt.show()
        #
        # # palette_dict = {1: 'red', 2: "gold", 3: "deepskyblue"}
        # # plt.subplots(1,2)
        # # plt.subplot(121)
        # # sns.catplot(data=data[mask1 &mask3 & mask4],x='Stimulation type', y='Generalization score', hue='Test Odor class', palette=palette_dict,ci='sd',col='Train Odor class',kind="bar")
        # # plt.subplot(122)
        # # sns.boxplot(data=data[mask1 & mask3 & mask4], x='Stimulation type', y='Generalization score', hue='Odor class', palette=palette_dict)
        # # plt.show()
        #
        # # palette_dict = {1: 'red', 2: "gold", 3: "deepskyblue"}
        # # sns.boxplot(data=data[mask2 & mask3 & mask4], x='Concentration', y='Generalization score', hue='Odor class', palette=palette_dict)
        # # plt.show()
        # #
        # # sns.boxplot(data=data[mask2 & mask3 & mask1], x='Glomerulus number', y='Generalization score', hue='Odor class',
        # #            palette=palette_dict)
        # # plt.show()

    def analyze_discrimination_result(self, data):
        mask1 = data['Training iteration'] == 100
        mask2 = data['Odor number'] == 100
        mask3 = data['Stimulation type'] == 'random'
        new_data = data[mask1 & mask2 & mask3]
        ax = sns.barplot(data=new_data, ci='sd', x='Wiring pattern', y='Accuracy', palette='Set2')
        plt.yticks([0.4, 0.7, 1.0], fontsize=self.fontdict['Tick']['fontsize'])
        plt.ylim([0.35, 1.05])
        plt.ylabel("Decode accuracy", fontdict=self.fontdict['Label'])
        plt.xlabel("Wiring pattern", fontdict=self.fontdict['Label'])
        xticklabels = []
        for string in ax.get_xticklabels():
            s = str(string.get_text())
            s = s.split(" ")[0]
            xticklabels.append(s)
        ax.set_xticklabels(xticklabels, fontdict={'fontsize': self.fontdict['Tick']['fontsize']})
        # plt.legend(fontsize=self.fontdict['Legend']['fontsize'], loc='upper right')
        plt.tight_layout()
        plt.savefig("Random odor discrimination.png", dpi=600)
        plt.close()
        nom = pg.normality(data=new_data, dv='Accuracy', group='Wiring pattern')
        nom.to_excel("Random odor discrimination normality test.xlsx")
        homo = pg.homoscedasticity(data=new_data, dv='Accuracy', group='Wiring pattern')
        homo.to_excel("Random odor discrimination variance test.xlsx")
        aov = pg.kruskal(data=new_data, dv='Accuracy', between='Wiring pattern')
        aov.to_excel("Random odor discrimination kruskal.xlsx")
        pt = pg.pairwise_ttests(data=new_data, dv='Accuracy', between='Wiring pattern')
        pt.to_excel("Random odor discrimination post-hoc.xlsx")

        mask3 = data['Stimulation type'] == 'single'
        new_data = data[mask1 & mask2 & mask3]

        ax = sns.barplot(data=new_data, ci='sd', hue='Wiring pattern', x='Odor class', y='Accuracy', palette='Set2')
        plt.yticks([0.4, 0.7, 1.0], fontsize=self.fontdict['Tick']['fontsize'])
        plt.ylim([0.35, 1.05])
        plt.ylabel("Decode accuracy", fontdict=self.fontdict['Label'])
        plt.xlabel("Odor class", fontdict=self.fontdict['Label'])
        ax.set_xticklabels([f"{string.get_text()}" for string in ax.get_xticklabels()],
                           fontdict=self.fontdict['Tick'])
        plt.legend(fontsize=self.fontdict['Legend']['fontsize'], loc='upper right')
        plt.tight_layout()
        plt.savefig("Single odor discrimination.png", dpi=600)
        plt.close()
        new_data['group'] = [f"{wiring_pattern} {odor_class}" for wiring_pattern, odor_class in
                             zip(new_data['Wiring pattern'], new_data['Odor class'])]
        nom = pg.normality(data=new_data, dv='Accuracy', group='group')
        nom.to_excel("Single odor discrimination normality test.xlsx")
        homo = pg.homoscedasticity(data=new_data, dv='Accuracy', group='group')
        homo.to_excel("Single odor discrimination variance test.xlsx")
        aov = pg.kruskal(data=new_data, dv='Accuracy', between=['Wiring pattern', 'Odor class'])
        aov.to_excel("Single odor discrimination kruskal.xlsx")
        pt = pg.pairwise_ttests(data=new_data, dv='Accuracy', between=['Wiring pattern', 'Odor class'])
        pt.to_excel("Single odor discrimination post-hoc.xlsx")

        mask3 = data['Stimulation type'] == 'preferred'
        new_data = data[mask1 & mask2 & mask3]
        ax = sns.barplot(data=new_data, ci='sd', hue='Wiring pattern', x='Odor class', y='Accuracy', palette='Set2')
        plt.yticks([0.4, 0.7, 1.0], fontsize=self.fontdict['Tick']['fontsize'])
        plt.ylim([0.35, 1.05])
        plt.ylabel("Decode accuracy", fontdict=self.fontdict['Label'])
        plt.xlabel("Odor class", fontdict=self.fontdict['Label'])
        ax.set_xticklabels([f"{string.get_text()}" for string in ax.get_xticklabels()],
                           fontdict=self.fontdict['Tick'])
        plt.legend(fontsize=self.fontdict['Legend']['fontsize'], loc='upper right')
        plt.tight_layout()
        plt.savefig("Preferred odor discrimination.png", dpi=600)
        plt.close()
        new_data['group'] = [f"{wiring_pattern} {odor_class}" for wiring_pattern, odor_class in
                             zip(new_data['Wiring pattern'], new_data['Odor class'])]
        nom = pg.normality(data=new_data, dv='Accuracy', group='group')
        nom.to_excel("Preferred odor discrimination normality test.xlsx")
        homo = pg.homoscedasticity(data=new_data, dv='Accuracy', group='group')
        homo.to_excel("Preferred odor discrimination variance test.xlsx")
        aov = pg.kruskal(data=new_data, dv='Accuracy', between=['Wiring pattern', 'Odor class'])
        aov.to_excel("Preferred odor discrimination kruskal.xlsx")
        pt = pg.pairwise_ttests(data=new_data, dv='Accuracy', between=['Wiring pattern', 'Odor class'])
        pt.to_excel("Preferred odor discrimination post-hoc.xlsx")

    def analyze_generalization_result(self, data):
        mask1 = data['Concentration'] == 1.2
        mask2 = data['Stimulation type'] == 'single'
        mask3 = data['Wiring pattern'] == 'FlyEM'
        mask4 = data['Glomerulus number'] == 4
        data.rename({'Accuracy': "Generalization score"}, axis='columns', inplace=True)

        # data["Generalization score"] = (data["Generalization score"] - 0.5)*2
        palette_dict = {1: 'red', 2: "gold", 3: "deepskyblue"}
        # for odor_class_id in [1,2,3]:
        #     mask5 = data['Odor class'] == odor_class_id
        #     print(odor_class_id)
        #     with open(f"{odor_class_id}_non_overlap.txt",'wt')as ff:
        #         for d in data[mask1 & mask2 & mask4 & mask3 & mask5]["Generalization score"].values.tolist():
        #             ff.writelines(f"{d}\n")
        # plt.subplots(1,2)
        # plt.subplot(121)
        sns.barplot(data=data[mask1 & mask2 & mask4], x='Wiring pattern', y='Generalization score', hue='Odor class',
                    palette=palette_dict, ci='sd')
        # plt.ylim(0.4,1.2)
        # plt.yticks([0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        # plt.subplot(122)
        # sns.boxplot(data=data[mask1 & mask2 & mask4], x='Wiring pattern', y='Generalization score', hue='Odor class', palette=palette_dict)
        plt.show()

        palette_dict = {1: 'red', 2: "gold", 3: "deepskyblue"}
        # plt.subplots(1,2)
        # plt.subplot(121)
        sns.barplot(data=data[mask1 & mask3 & mask4], x='Stimulation type', y='Generalization score', hue='Odor class',
                    palette=palette_dict, ci='sd')
        # plt.subplot(122)
        # sns.boxplot(data=data[mask1 & mask3 & mask4], x='Stimulation type', y='Generalization score', hue='Odor class', palette=palette_dict)
        plt.show()

        palette_dict = {1: 'red', 2: "gold", 3: "deepskyblue"}
        sns.boxplot(data=data[mask2 & mask3 & mask4], x='Concentration', y='Generalization score', hue='Odor class',
                    palette=palette_dict)
        plt.show()

        sns.boxplot(data=data[mask2 & mask3 & mask1], x='Glomerulus number', y='Generalization score', hue='Odor class',
                    palette=palette_dict)
        plt.show()

    def analyze_result(self, data):
        mask1 = data['Concentration'] == 1.0
        mask2 = data['Stimulation type'] == 'single'
        mask3 = data['Wiring pattern'] == 'FlyEM'
        mask4 = data['Glomerulus number'] == 7
        data.rename({'Accuracy': "Performance Index"}, axis='columns', inplace=True)
        data["Performance Index"] = (data["Performance Index"] - (1 - data["Performance Index"]))
        data = data[mask1 & mask2 & mask3 & mask4]
        data.to_excel("quick.xlsx")
        # data['Stimulation type'][data['Stimulation type']=='shuffled_0'] = 'shuffled'
        # data['Stimulation type'][data['Stimulation type']=='shuffled_1'] = 'shuffled'
        # data['Stimulation type'][data['Stimulation type']=='shuffled_2'] = 'shuffled'
        #
        #
        # palette_dict = {1: 'red', 2: "gold", 3: "deepskyblue"}
        # sns.catplot(data=data[mask4], col='Concentration', y='Performance Index', hue='Odor class',
        #            palette=palette_dict, ci='sd', x='Wiring pattern', kind="bar", row='Stimulation type')
        #
        # # sns.barplot(data=data[mask1 & mask2 & mask4], x='Wiring pattern', y='Performance Index', hue='Odor class', palette=palette_dict)
        # # plt.yticks([0,0.25,0.5,0.75,1])
        # # plt.ylim([0,1.1])
        # plt.show()

        # palette_dict = {1: 'red', 2: "gold", 3: "deepskyblue"}
        # sns.barplot(data=data[mask1 & mask3 & mask4], x='Stimulation type', y='Performance Index', hue='Odor class', palette=palette_dict)
        # plt.show()
        #
        # palette_dict = {1: 'red', 2: "gold", 3: "deepskyblue"}
        # sns.barplot(data=data[mask2 & mask3 & mask4], x='Concentration', y='Performance Index', hue='Odor class', palette=palette_dict)
        # plt.show()
        #
        # palette_dict = {1: 'red', 2: "gold", 3: "deepskyblue"}
        # sns.barplot(data=data[mask2 & mask3 & mask1], x='Glomerulus number', y='Performance Index', hue='Odor class', palette=palette_dict)
        # plt.show()



if __name__ == '__main__':
    result_analyzer = Result_Analyzer()
    result_analyzer.load_simulation_data()
    result_analyzer.analyze_activation_ratio(result_analyzer.network.Subtype_to_KCid.keys(),connection_type='binary')
    result_analyzer.analyze_dimensionality(connection_type='binary')
