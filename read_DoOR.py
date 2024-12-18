import pyreadr
import pandas as pd
import os
from pandas import DataFrame as Df
import numpy as np
import pickle

##### deal with DoOR data############################################
def transform_DoOR_data_to_excel(): ##transform .Rdata to .xlsx
    path = 'C:/Users/clust/Downloads/DoOR.data-v2.0.0/Dahaniel-DoOR.data-6436660/data/'
    record_path = 'DoOR_data/'
    if not os.path.isdir(record_path):
        os.mkdir(record_path)
    file_list = [_ for _ in os.listdir(path) if _.find("RData") != -1]
    for file in file_list:
        result = pyreadr.read_r(path + file)  # also works for Rds
        for key in result.keys():
            result[key].to_excel(f'{record_path}{key}.xlsx')
    return

def construct_OR_glomerulus_table():
    path =  'DoOR_data/'
    file = 'DoOR.mappings.xlsx'
    data = pd.read_excel(path+file)
    OR_to_G = {}
    G_to_OR = {}
    Gs = data['glomerulus'].values.tolist()
    ORs = data['receptor'].values.tolist()
    for OR,G in zip(ORs,Gs):
        if not isinstance(G, str):
            continue
        if OR.find("?")!=-1 or G.find("?")!=-1:
            continue
        groups = G.split("+")
        for group in groups:
            if OR not in OR_to_G:
                OR_to_G[OR] = []
            OR_to_G[OR].append(group)
            if group not in G_to_OR:
                G_to_OR[group] = []
            G_to_OR[group].append(OR)

    return OR_to_G, G_to_OR

def lookup_odor_to_Glomerulus():
    if not os.path.isfile("lookup_odor_to_G.pickle"):
        ############## get InChIKey -> odor ###########
        path = 'DoOR_data/'
        file = 'data.format.xlsx'
        data = pd.read_excel(path + file)
        Class_to_InChIKey = {}
        InChIKey_to_Class = {}
        InChIKey_to_Name = {}
        Name_to_InChIKey = {}
        Class = data['Class'].values.tolist()
        InChIKey = data['InChIKey'].values.tolist()
        Name = data['Name'].values.tolist()
        for name, class_che, ChIKey in zip(Name, Class, InChIKey):
            if isinstance(class_che, str):
                if class_che not in Class_to_InChIKey:
                    Class_to_InChIKey[class_che] = []
                Class_to_InChIKey[class_che].append(ChIKey)
                InChIKey_to_Class[ChIKey] = class_che
                InChIKey_to_Name[ChIKey] = name
                Name_to_InChIKey[name] = ChIKey

        ############## get odor -> ORN response ###########
        file = 'response.matrix_transpose.xlsx'
        data = pd.read_excel(path + file)
        receptors = data['receptor'].values.tolist()
        InChIKey = data.head()[1:]
        OR_to_G, G_to_OR = construct_OR_glomerulus_table()
        '''
        condition:
        1. The odorant has been annotated its Class
        2. The odorant should activate more than 2 receptors which has been logged in the dictionary
        '''
        Odor_to_Glomerulus_activity = {}
        for odor in InChIKey:
            if odor not in InChIKey_to_Class:
                continue
            responses = data[odor].values.tolist()
            Odor_to_Glomerulus_activity[odor] = {}
            for receptor, response in zip(receptors, responses):
                if not isinstance(response, float):
                    continue
                if np.isnan(response):
                    continue
                if receptor not in OR_to_G:
                    continue
                for G in OR_to_G[receptor]:
                    Odor_to_Glomerulus_activity[odor][G] = response
        pooled_data = [Odor_to_Glomerulus_activity,Class_to_InChIKey, InChIKey_to_Class, InChIKey_to_Name, Name_to_InChIKey]
        with open("lookup_odor_to_G.pickle",'wb')as ff:
            pickle.dump(pooled_data,ff)
    else:
        with open("lookup_odor_to_G.pickle", 'rb')as ff:
            pooled_data = pickle.load(ff)
        Odor_to_Glomerulus_activity, Class_to_InChIKey, InChIKey_to_Class, InChIKey_to_Name, Name_to_InChIKey = pooled_data

    return Odor_to_Glomerulus_activity,Class_to_InChIKey, InChIKey_to_Class, InChIKey_to_Name, Name_to_InChIKey



if __name__=='__main__':
    # obtain_uniglomerular_PN()
    # Glomerulus_to_Cluster, Cluster_to_Glomerulus, PNid_to_Cluster, Cluster_to_PNid, PNid_to_Glomerulus, Glomerulus_to_PNid, KCid_to_Subtype, Subtype_to_KCid = obtain_lookup_dict_weight()
    # OR_to_G, G_to_OR = construct_OR_glomerulus_table()
    # # print(OR_to_G)
    # for subtype in Subtype_to_KCid:
    #     print(f"{subtype}: {len(Subtype_to_KCid[subtype])}")
    # # for G in G_to_OR.keys():
    # #     if G not in Glomerulus_to_Cluster:
    # #         print(G)
    Odor_to_Glomerulus_activity,Class_to_InChIKey, InChIKey_to_Class, InChIKey_to_Name, Name_to_InChIKey = lookup_odor_to_Glomerulus()
    print(InChIKey_to_Name['UAHWPYUMFXYFJY-UHFFFAOYSA-N'])
    print(InChIKey_to_Name['SFR'])

    # for odor in Odor_to_Glomerulus_activity:
    #     if len(Odor_to_Glomerulus_activity[odor])>35:
    #         print(InChIKey_to_Name[odor],len(Odor_to_Glomerulus_activity[odor]))
    #         activity = [Odor_to_Glomerulus_activity[odor][i] for i in Odor_to_Glomerulus_activity[odor]]
    #         print(max(activity))
    # # with open("Odor_to_Glomerulus_activity.pickle",'wb')as ff:
    # #     pickle.dump(Odor_to_Glomerulus_activity,ff)
    # # print(Odor_to_Glomerulus_activity)


    G_record = []
    odor_list = ['geranyl acetate','4-methylcyclohexanol','methyl salicylate','3-octanol']
    Odor_to_Glomerulus_activity,Class_to_InChIKey, InChIKey_to_Class, InChIKey_to_Name, Name_to_InChIKey = lookup_odor_to_Glomerulus()
    for odor in odor_list:
        for G in Odor_to_Glomerulus_activity[Name_to_InChIKey[odor]]:
            if G not in G_record:
                G_record.append(G)
    data = np.zeros((len(G_record),len(odor_list)))
    for odor_id,odor in enumerate(odor_list):
        for G in Odor_to_Glomerulus_activity[Name_to_InChIKey[odor]]:
            data[G_record.index(G)][odor_id] = Odor_to_Glomerulus_activity[Name_to_InChIKey[odor]][G]
    Df(data=data,columns=odor_list,index=G_record).to_excel("20241129_Odor_G_resopnse_Zhiyuan.xlsx")

