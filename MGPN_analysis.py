import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import generate_connection as gc
from generate_connection import ConnectionSetting
from pandas import DataFrame as Df
from collections import Counter


def quantify_MGPN_connected_with_KC_synapses():
    df_M_to_KC = pd.read_excel(f"hemibrain_data/Connection_M_s_upstream_of_KCs_w_3_v1.2.1.xlsx")
    MPNid_list = df_M_to_KC['up.bodyId'].unique().tolist()
    if os.path.isfile('hemibrain_data/MGPN_connected_with_KC_synapse_neuropil.csv'):
        data = pd.read_csv('hemibrain_data/MGPN_connected_with_KC_synapse_neuropil.csv')
    else:
        df = pd.read_excel('hemibrain_data/Synapse_downstream_of_M_s_w_0_v1.2.1.xlsx')
        df = df[df['up.bodyId'].isin(MPNid_list)]
        xyz = df[['up_syn_coordinate_x', 'up_syn_coordinate_y', 'up_syn_coordinate_z']].to_numpy()
        xyz = np.unique(xyz, axis=0)
        synapse_distribution_result = []
        synapse_distribution_result.append([len(xyz),'All'])
        for neuropil in ['CA(R)','LH(R)']:
            filtered_xyz = a.filtered_coordinates(xyz, neuropil)
            np.save(f"hemibrain_data/MGPN_connected_with_KC_synapse_{neuropil}.npz",filtered_xyz)
            synapse_distribution_result.append([len(filtered_xyz),neuropil])
        data=pd.DataFrame(data=synapse_distribution_result, columns=['Presynapse number','Neuropil'])
        data.to_csv('hemibrain_data/MGPN_connected_with_KC_synapse_neuropil.csv')
    sns.barplot(data=data,x='Neuropil',y='Presynapse number',palette=['black','white','grey'],edgecolor='black')
    plt.xlabel('Neuropil', fontdict={'fontsize': 20})
    plt.ylabel('Presynapse number', fontdict={'fontsize': 20})
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)  # X-axis
    ax.spines['left'].set_linewidth(1.5)  # Y-axis
    ax.spines['top'].set_linewidth(1.5)  # X-axis
    ax.spines['right'].set_linewidth(1.5)  # Y-axis
    plt.title('MGPN connected with KC output synapse')
    plt.show()

def get_MGPN_number():
    df_M_to_KC = pd.read_excel(f"hemibrain_data/Connection_M_s_upstream_of_KCs_w_3_v1.2.1.xlsx")
    MPNid_list_KC = df_M_to_KC['up.bodyId'].unique().tolist()
    df = pd.read_excel('hemibrain_data/Synapse_downstream_of_M_s_w_0_v1.2.1.xlsx')
    MPNid_list_All = df['up.bodyId'].unique().tolist()
    sns.barplot(data=pd.DataFrame([['All', len(MPNid_list_All)],['Connected to KC',len(MPNid_list_KC)]], columns=['Neuropil','PN number']),x='Neuropil',y='PN number',palette=['black','white'],edgecolor='black')
    plt.xlabel('', fontdict={'fontsize': 20})
    plt.ylabel('PN number', fontdict={'fontsize': 20})
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)  # X-axis
    ax.spines['left'].set_linewidth(1.5)  # Y-axis
    ax.spines['top'].set_linewidth(1.5)  # X-axis
    ax.spines['right'].set_linewidth(1.5)  # Y-axis
    plt.show()

def quantify_MGPN_synapses():
    if os.path.isfile('hemibrain_data/MGPN_synapse_neuropil.csv'):
        data = pd.read_csv('hemibrain_data/MGPN_synapse_neuropil.csv')
    else:
        df = pd.read_excel('hemibrain_data/Synapse_downstream_of_M_s_w_0_v1.2.1.xlsx')
        xyz = df[['up_syn_coordinate_x', 'up_syn_coordinate_y', 'up_syn_coordinate_z']].to_numpy()
        xyz = np.unique(xyz, axis=0)
        synapse_distribution_result = []
        synapse_distribution_result.append([len(xyz),'All'])
        for neuropil in ['CA(R)','LH(R)']:
            filtered_xyz = a.filtered_coordinates(xyz, neuropil)
            np.save(f"hemibrain_data/MGPN_synapse_{neuropil}.npz",filtered_xyz)
            synapse_distribution_result.append([len(filtered_xyz),neuropil])
        data=pd.DataFrame(data=synapse_distribution_result, columns=['Presynapse number','Neuropil'])
        data.to_csv('hemibrain_data/MGPN_synapse_neuropil.csv')
    sns.barplot(data=data,x='Neuropil',y='Presynapse number',palette=['black','white','grey'],edgecolor='black')
    plt.xlabel('Neuropil', fontdict={'fontsize': 20})
    plt.ylabel('Presynapse number', fontdict={'fontsize': 20})
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)  # X-axis
    ax.spines['left'].set_linewidth(1.5)  # Y-axis
    ax.spines['top'].set_linewidth(1.5)  # X-axis
    ax.spines['right'].set_linewidth(1.5)  # Y-axis
    plt.title('MGPN output synapse')
    plt.show()

def analysis_for_KC_connected_MGPN_ORN_PN():
    ## get type
    path = 'hemibrain_data/'
    file = 'Connection_upstream_of_M_s_w_3_v1.2.1.xlsx'
    connection_data = pd.read_excel(f"{path}{file}")
    color_dict = {1:"red",2:"gold",3:"deepskyblue",'KCg':'red',"KCa'b'":'gold',"KCab":"deepskyblue"}
    c = ConnectionSetting()
    network = gc.load_network()
    
    MGPN_ID_list_ALL = connection_data['down.bodyId'].unique().tolist()
    connection_data['down.type'] = [i.split("_")[1].split("PN")[0] for i in connection_data['down.type']]
    MGPNtype_to_MGPNID_dict = (connection_data.groupby('down.type')['down.bodyId'].apply(lambda x: list(set(x))).to_dict())
    MGPNID_to_MGPNtype_dict = connection_data.set_index('down.bodyId')['down.type'].to_dict()
    ## to KC
    df_M_to_KC = pd.read_excel(f"hemibrain_data/Connection_M_s_upstream_of_KCs_w_3_v1.2.1.xlsx")
    MGPN_connected_to_KC_list = df_M_to_KC['up.bodyId'].unique().tolist()
    KCid_list = []
    df_M_to_KC['down.type'] = [i.split("-")[0] for i in df_M_to_KC['down.type']]
    KCclass_to_KCid_dict = (df_M_to_KC.groupby('down.type')['down.bodyId'].apply(lambda x: list(set(x))).to_dict())
    KCid_to_KCclass_dict = df_M_to_KC.set_index('down.bodyId')['down.type'].to_dict()
    KC_class_list = ['KCg',"KCa'b'","KCab"]
    for KC_class in KC_class_list:
        KCid_list += KCclass_to_KCid_dict[KC_class]
    MGPN_to_KCclass_connected_freq = np.zeros((len(MGPN_connected_to_KC_list),len(KC_class_list)))
    for connection in df_M_to_KC.values.tolist():    
        upId,downId,w = connection[1],connection[4],connection[7]
        MGPN_to_KCclass_connected_freq[MGPN_connected_to_KC_list.index(upId)][KC_class_list.index(KCid_to_KCclass_dict[downId])] += 1
    fig, ax = plt.subplots()
    plt.bar([0,1,2],[len(KCclass_to_KCid_dict[KC_class]) for KC_class in KC_class_list],color='black')
    plt.xticks([0,1,2],['KCg',"KCa'b'","KCab"])
    plt.xlabel('', fontdict={'fontsize': 20})
    plt.ylabel('Connected KC number', fontdict={'fontsize': 20})
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    ax.spines['bottom'].set_linewidth(1.5)  # X-axis
    ax.spines['left'].set_linewidth(1.5)  # Y-axis
    ax.spines['top'].set_linewidth(1.5)  # X-axis
    ax.spines['right'].set_linewidth(1.5)  # Y-axis
    plt.title("Connected KC number")
    plt.show()

    sns.clustermap(MGPN_to_KCclass_connected_freq, col_cluster=False)
    plt.title('Connected number of MGPN-to-KC')
    plt.show()

    MGPN_correlation_by_KC = np.corrcoef(MGPN_to_KCclass_connected_freq,rowvar=True)
    cluster = sns.clustermap(data=MGPN_correlation_by_KC,cmap='bwr')
    plt.title("Correlation of MGPN based on MGPN-to-KC")
    plt.show()

    ### There are three clusters
    MGPN_classification_list = [[MGPN_connected_to_KC_list[6],MGPN_connected_to_KC_list[9]],[MGPN_connected_to_KC_list[0],MGPN_connected_to_KC_list[12]]]
    MGPN_classification_list.append([MGPN_connected_to_KC_list[i] for i in range(len(MGPN_connected_to_KC_list)) if i not in [0,6,9,12]])
    MGPN_classification_index_list = [[6,9],[0,12]]
    MGPN_classification_index_list.append([i for i in range(len(MGPN_connected_to_KC_list)) if i not in [0,6,9,12]])
    MGPN_classification_list.reverse()
    MGPN_classification_index_list.reverse()
    MGPNtype_classification_list = [[MGPNID_to_MGPNtype_dict[i] for i in k] for k in MGPN_classification_list]
    results = []
    for group, MGPNid_list in zip(["A",'B','C'],MGPN_classification_list):
        for MGPNid in MGPNid_list:
            results.append([MGPNid,group])
    Df(data=results,columns=['neuronId','Type']).to_csv("MGPN_group.csv")

    MGPNid_to_Classification_dict = {}
    Classification_to_MGPNid_dict = {}
    for i in range(len(MGPN_classification_list)):
        Classification_to_MGPNid_dict[i] = []
        for PNid in MGPN_classification_list[i]:
            MGPNid_to_Classification_dict[PNid] = i
            Classification_to_MGPNid_dict[i].append(PNid)
    MGPN_classification_color = ['red','green','purple']
    MGPN_colors = [MGPN_classification_color[MGPNid_to_Classification_dict[MGPNid]] for MGPNid in MGPN_connected_to_KC_list]
    for classification in range(len(MGPN_classification_list)):
        connected_freq = np.sum(MGPN_to_KCclass_connected_freq[MGPN_classification_index_list[classification]],axis=0).tolist()
        fig, ax = plt.subplots()
        plt.bar([0,1,2],connected_freq,color='black')
        plt.xticks([0,1,2],['KCg',"KCa'b'","KCab"])
        plt.xlabel('', fontdict={'fontsize': 20})
        plt.ylabel('Connected KC number', fontdict={'fontsize': 20})
        plt.title(f'{classification}')
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        ax.spines['bottom'].set_linewidth(1.5)  # X-axis
        ax.spines['left'].set_linewidth(1.5)  # Y-axis
        ax.spines['top'].set_linewidth(1.5)  # X-axis
        ax.spines['right'].set_linewidth(1.5)  # Y-axis
        plt.show()

    connection_data = pd.read_excel(f"{path}{file}")
    connection_data['up.type'] = connection_data['up.type'].fillna("")
    MGPN_ID_list = MGPN_connected_to_KC_list
    weight_collection = []
    for neuronId in MGPN_ID_list:
            # Mask for connections involving the current neuron ID
        mask = connection_data['down.bodyId'] == neuronId
        total_upstream_weight = connection_data.loc[mask, 'w.weight'].sum()

        # Mask for ORN inputs
        mask_ORN = mask & connection_data['up.type'].str.contains("ORN_")
        total_up_ORN_weight = connection_data.loc[mask_ORN, 'w.weight'].sum() / total_upstream_weight
        

        # Mask for uni-PN inputs
        mask_uni_PN = mask & connection_data['up.type'].str.endswith("PN")
        total_up_uniPN_weight = connection_data.loc[mask_uni_PN, 'w.weight'].sum() / total_upstream_weight

        # Append the weights to the collection
        weight_collection.append([total_up_ORN_weight, total_up_uniPN_weight])

    weight_collection = np.array(weight_collection)

    fig, ax = plt.subplots()
    ax.scatter(weight_collection[:, 0], weight_collection[:, 1], c='black', s=80)
    plt.xlabel('Input ratio from ORN', fontdict={'fontsize': 20})
    plt.ylabel('Input ratio from uniglomerular PN', fontdict={'fontsize': 20})
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    ax.spines['bottom'].set_linewidth(1.5)  # X-axis
    ax.spines['left'].set_linewidth(1.5)  # Y-axis
    ax.spines['top'].set_linewidth(1.5)  # X-axis
    ax.spines['right'].set_linewidth(1.5)  # Y-axis
    plt.title("Input of MGPNs")
    plt.show()    

    fig, ax = plt.subplots()
    ax.scatter(weight_collection[:, 0], weight_collection[:, 1], c=MGPN_colors, s=80)
    plt.xlabel('Input ratio from ORN', fontdict={'fontsize': 20})
    plt.ylabel('Input ratio from uniglomerular PN', fontdict={'fontsize': 20})
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    ax.spines['bottom'].set_linewidth(1.5)  # X-axis
    ax.spines['left'].set_linewidth(1.5)  # Y-axis
    ax.spines['top'].set_linewidth(1.5)  # X-axis
    ax.spines['right'].set_linewidth(1.5)  # Y-axis
    plt.title("Input of MGPNs")
    plt.show()    


    ## ORN
    print("ORN")
    connection_data = pd.read_excel(f"{path}{file}")
    connection_data['up.type'] = connection_data['up.type'].fillna("")
    mask_ORN = connection_data['up.type'].str.contains("ORN_")
    connection_data = connection_data[mask_ORN]
    connection_data['up.type'] = connection_data['up.type'].str.replace('ORN_', '', regex=False)
    Glomerulus_to_ORNid_dict = connection_data.groupby('up.type')['up.bodyId'].apply(lambda x: list(set(x))).to_dict()
    ORNid_to_Glomerulus_dict = connection_data.set_index('up.bodyId')['up.type'].to_dict()
    MPNid_list = [i for i in connection_data['down.bodyId'].unique().tolist() if i in MGPN_connected_to_KC_list]
    ORNid_list = []
    for G in network.G_list:
        if G in Glomerulus_to_ORNid_dict:
            ORNid_list = ORNid_list + Glomerulus_to_ORNid_dict[G]
    ORN_to_MPN_weight = np.zeros((len(ORNid_list),len(MPNid_list)))
    w_list = []
    for connection in connection_data.values.tolist():
        upId,downId,w = connection[1],connection[4],connection[7]
        if downId in MPNid_list:
            ORN_to_MPN_weight[ORNid_list.index(upId)][MPNid_list.index(downId)] = w
            w_list.append(w)
    Glomerulus_to_MPN_weight = np.zeros((len(network.G_list),len(MPNid_list)))
    for ORNindex, ORNid in enumerate(ORNid_list):
        # for KCindex, KCid in enumerate
        for PNindex, PNid in enumerate(MPNid_list):
            Glomerulus_to_MPN_weight[network.G_list.index(ORNid_to_Glomerulus_dict[ORNid])][PNindex] += ORN_to_MPN_weight[ORNindex][PNindex]            
    row_sum = np.sum(Glomerulus_to_MPN_weight,axis=1)
    column_sum = np.sum(Glomerulus_to_MPN_weight,axis=0)
    Glomerulus_to_MPN_weight = Glomerulus_to_MPN_weight[row_sum>0,:]
    Glomerulus_to_MPN_weight = Glomerulus_to_MPN_weight[:,column_sum>0]
    Connected_Glomerulus_num = np.count_nonzero(Glomerulus_to_MPN_weight,axis=0)
    plt.hist(Connected_Glomerulus_num)
    plt.xlabel('Connected Glomerulus (ORN) number', fontdict={'fontsize': 20})
    plt.ylabel('multi-glomerular PN number', fontdict={'fontsize': 20})
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    ax.spines['bottom'].set_linewidth(1.5)  # X-axis
    ax.spines['left'].set_linewidth(1.5)  # Y-axis
    ax.spines['top'].set_linewidth(1.5)  # X-axis
    ax.spines['right'].set_linewidth(1.5)  # Y-axis
    plt.show()
    G_new_list = np.array(network.G_list)
    G_new_list_ORN = G_new_list[row_sum>0].tolist()
    colors = [color_dict[network.Glomerulus_to_Cluster[G]] for G in G_new_list_ORN]
    # Create clustermap
    cluster = sns.clustermap(
        data=Glomerulus_to_MPN_weight,
        row_cluster=False,
        row_colors=colors,
        yticklabels=G_new_list_ORN,
        method='complete'
    )
    plt.title("ORN-MGPN connection matrix")
    plt.show()
    G_G_ORN_correlation = np.corrcoef(Glomerulus_to_MPN_weight,rowvar=True)
    cluster = sns.clustermap(
        data=G_G_ORN_correlation,
        row_colors=colors,
        yticklabels = G_new_list_ORN,
        method='complete'
    )
    plt.title("Glomerulus correlation based on ORN-MGPN connectivity")
    plt.show()
    
    ## PN
    print('PN')
    connection_data = pd.read_excel(f"{path}{file}")
    connection_data['up.type'] = connection_data['up.type'].fillna("")
    mask_uniPN = connection_data['up.type'].str.endswith("PN")
    connection_data = connection_data[mask_uniPN]
    connection_data['up.type'] = [PN_type.split("_")[0] for PN_type in connection_data['up.type']]
    Glomerulus_to_uniPNid_dict = connection_data.groupby('up.type')['up.bodyId'].apply(lambda x: list(set(x))).to_dict()
    uniPNid_to_Glomerulus_dict = connection_data.set_index('up.bodyId')['up.type'].to_dict()
    MPNid_list = [i for i in connection_data['down.bodyId'].unique().tolist() if i in MGPN_connected_to_KC_list]
    uniPNid_list = []
    for G in network.G_list:
        if G in Glomerulus_to_uniPNid_dict:
            uniPNid_list = uniPNid_list + Glomerulus_to_uniPNid_dict[G]
    uniPN_to_MPN_weight = np.zeros((len(uniPNid_list),len(MPNid_list)))
    w_list = []
    for connection in connection_data.values.tolist():
        upId,downId,w = connection[1],connection[4],connection[7]
        if upId in uniPNid_list and downId in MPNid_list:
            uniPN_to_MPN_weight[uniPNid_list.index(upId)][MPNid_list.index(downId)] = w
            w_list.append(w)
    Glomerulus_to_MPN_weight = np.zeros((len(network.G_list),len(MPNid_list)))
    for uniPNindex, uniPNid in enumerate(uniPNid_list):
        # for KCindex, KCid in enumerate
        for PNindex, PNid in enumerate(MPNid_list):
            Glomerulus_to_MPN_weight[network.G_list.index(uniPNid_to_Glomerulus_dict[uniPNid])][PNindex] += uniPN_to_MPN_weight[uniPNindex][PNindex]            
    # Glomerulus_to_MPN_weight_PN = copy.deepcopy(Glomerulus_to_MPN_weight)
    row_sum = np.sum(Glomerulus_to_MPN_weight,axis=1)
    column_sum = np.sum(Glomerulus_to_MPN_weight,axis=0)
    Glomerulus_to_MPN_weight = Glomerulus_to_MPN_weight[row_sum>0,:]
    Glomerulus_to_MPN_weight = Glomerulus_to_MPN_weight[:,column_sum>0]
    Connected_Glomerulus_num = np.count_nonzero(Glomerulus_to_MPN_weight,axis=0)
    plt.hist(Connected_Glomerulus_num)
    plt.title("Number of Glomerulus connected to MGPNs")
    plt.show()
    G_new_list = np.array(network.G_list)
    G_new_list_uniPN = G_new_list[row_sum>0].tolist()
    colors = [color_dict[network.Glomerulus_to_Cluster[G]] for G in G_new_list_uniPN]
    # Create clustermap
    cluster = sns.clustermap(
        data=Glomerulus_to_MPN_weight,
        row_cluster=False,
        row_colors=colors,
        yticklabels=G_new_list_uniPN,
        method='complete'
    )
    plt.title("G-to-PN connection matrix")
    plt.show()
    G_G_uniPN_correlation = np.corrcoef(Glomerulus_to_MPN_weight,rowvar=True)
    cluster = sns.clustermap(
        data=G_G_uniPN_correlation,
        row_colors=colors,
        yticklabels = G_new_list_uniPN,
        method='complete'
    )
    plt.title("Glomerulus correlation based on uniPN-to-MGPN connectivity")
    plt.show()

    ## ORN, PN
    file = 'Connection_upstream_of_M_s_w_3_v1.2.1.xlsx'
    connection_data = pd.read_excel(f"{path}{file}")
    connection_data['up.type'] = connection_data['up.type'].fillna("")
    non_excluded_shared_G_list = [G for G in network.G_list if G in G_new_list_uniPN or G in G_new_list_ORN]
    Glomerulus_to_MPN_weight_pooled = np.zeros((len(non_excluded_shared_G_list), len(MGPN_ID_list)))
    
    for connection in connection_data.values.tolist():
        upId,downId,w = connection[1],connection[4],connection[7]
        if upId in ORNid_list:
            G = ORNid_to_Glomerulus_dict[upId]
        elif upId in uniPNid_list:
            G = uniPNid_to_Glomerulus_dict[upId]
        if G in non_excluded_shared_G_list and downId in MGPN_ID_list:
            Glomerulus_to_MPN_weight_pooled[non_excluded_shared_G_list.index(G)][MGPN_ID_list.index(downId)] += w
    
    Connected_Glomerulus_num_ORN_PN = np.count_nonzero(Glomerulus_to_MPN_weight_pooled,axis=0)
    print("All-shared_G")
    print(Connected_Glomerulus_num_ORN_PN)
    plt.hist(Connected_Glomerulus_num_ORN_PN)
    plt.xlabel('Connected Glomerulus (ORN+PN) number', fontdict={'fontsize': 20})
    plt.ylabel('multi-glomerular PN number', fontdict={'fontsize': 20})
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    ax.spines['bottom'].set_linewidth(1.5)  # X-axis
    ax.spines['left'].set_linewidth(1.5)  # Y-axis
    ax.spines['top'].set_linewidth(1.5)  # X-axis
    ax.spines['right'].set_linewidth(1.5)  # Y-axis
    plt.show()
    
    shared_G_list = [G for G in network.G_list if G in G_new_list_uniPN or G in G_new_list_ORN]
    Glomerulus_to_MPN_weight_pooled = np.zeros((len(shared_G_list), len(MGPN_ID_list)))
    for connection in connection_data.values.tolist():
        upId,downId,w = connection[1],connection[4],connection[7]
        if upId in ORNid_list:
            G = ORNid_to_Glomerulus_dict[upId]
        elif upId in uniPNid_list:
            G = uniPNid_to_Glomerulus_dict[upId]
        if G in shared_G_list and downId in MGPN_ID_list:
            Glomerulus_to_MPN_weight_pooled[shared_G_list.index(G)][MGPN_ID_list.index(downId)] += w

    Connected_Glomerulus_num_ORN_PN = np.count_nonzero(Glomerulus_to_MPN_weight_pooled,axis=0)
    plt.hist(Connected_Glomerulus_num_ORN_PN)
    plt.xlabel('Connected Glomerulus (ORN+PN) number', fontdict={'fontsize': 20})
    plt.ylabel('multi-glomerular PN number', fontdict={'fontsize': 20})
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    ax.spines['bottom'].set_linewidth(1.5)  # X-axis
    ax.spines['left'].set_linewidth(1.5)  # Y-axis
    ax.spines['top'].set_linewidth(1.5)  # X-axis
    ax.spines['right'].set_linewidth(1.5)  # Y-axis
    plt.show()
    
    G_G_pooled_correlation = np.corrcoef(Glomerulus_to_MPN_weight_pooled,rowvar=True)
    colors = [color_dict[network.Glomerulus_to_Cluster[G]] for G in shared_G_list]
    cluster = sns.clustermap(
        data=G_G_pooled_correlation,
        row_colors=colors,        
        yticklabels = shared_G_list,
        method='complete',
        cmap='bwr'
    )
    plt.title("Glomerulus correlation based on ORN+uniPN -to-MGPN connectivity")
    plt.show()

    fig, ax = plt.subplots()
    cluster_count = Counter([network.Glomerulus_to_Cluster[G] for G in shared_G_list])
    plt.bar([0,1,2],[cluster_count[Cluster] for Cluster in [1,2,3]],color='black')
    plt.xticks([0,1,2],[f"Cluster {i}" for i in [1,2,3]])
    plt.xlabel('', fontdict={'fontsize': 20})
    plt.ylabel('Connected Glomerulus number', fontdict={'fontsize': 20})
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    ax.spines['bottom'].set_linewidth(1.5)  # X-axis
    ax.spines['left'].set_linewidth(1.5)  # Y-axis
    ax.spines['top'].set_linewidth(1.5)  # X-axis
    ax.spines['right'].set_linewidth(1.5)  # Y-axis

    plt.show()

    fig, ax = plt.subplots()
    cluster_count = Counter([network.Glomerulus_to_Cluster[G] for G in shared_G_list])
    plt.bar([0,1,2],[cluster_count[Cluster]/len(network.Cluster_to_Glomerulus[Cluster]) for Cluster in [1,2,3]],color='black')
    plt.xticks([0,1,2],[f"Cluster {i}" for i in [1,2,3]])
    plt.xlabel('', fontdict={'fontsize': 20})
    plt.ylabel('Ratio of Connected Glomerulus', fontdict={'fontsize': 20})
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    ax.spines['bottom'].set_linewidth(1.5)  # X-axis
    ax.spines['left'].set_linewidth(1.5)  # Y-axis
    ax.spines['top'].set_linewidth(1.5)  # X-axis
    ax.spines['right'].set_linewidth(1.5)  # Y-axis
    plt.show()

    ## ORN, PN
    for classification, MGPN_ID_list in enumerate(MGPN_classification_list):
        shared_G_list = [G for G in network.G_list if G in G_new_list_uniPN or G in G_new_list_ORN]
        Glomerulus_to_MPN_weight_pooled = np.zeros((len(shared_G_list), len(MGPN_ID_list)))
        for connection in connection_data.values.tolist():
            upId,downId,w = connection[1],connection[4],connection[7]
            if upId in ORNid_list:
                G = ORNid_to_Glomerulus_dict[upId]
            elif upId in uniPNid_list:
                G = uniPNid_to_Glomerulus_dict[upId]
            if G in shared_G_list and downId in MGPN_ID_list:
                Glomerulus_to_MPN_weight_pooled[shared_G_list.index(G)][MGPN_ID_list.index(downId)] += w
        shared_G_list = np.array(shared_G_list)
        shared_G_list = shared_G_list[np.nonzero(np.sum(Glomerulus_to_MPN_weight_pooled, axis=1))[0].tolist()].tolist()
        print(len(shared_G_list))
        print(classification,shared_G_list)
        fig, ax = plt.subplots()
        cluster_count = Counter([network.Glomerulus_to_Cluster[G] for G in shared_G_list])
        plt.bar([0,1,2],[cluster_count[Cluster] for Cluster in [1,2,3]],color='black')
        plt.title(classification)
        plt.xticks([0,1,2],[f"Cluster {i}" for i in [1,2,3]])
        plt.xlabel('', fontdict={'fontsize': 20})
        plt.ylabel('Connected Glomerulus number', fontdict={'fontsize': 20})
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        ax.spines['bottom'].set_linewidth(1.5)  # X-axis
        ax.spines['left'].set_linewidth(1.5)  # Y-axis
        ax.spines['top'].set_linewidth(1.5)  # X-axis
        ax.spines['right'].set_linewidth(1.5)  # Y-axis
        plt.show()

        fig, ax = plt.subplots()
        cluster_count = Counter([network.Glomerulus_to_Cluster[G] for G in shared_G_list])
        plt.bar([0,1,2],[cluster_count[Cluster]/len(network.Cluster_to_Glomerulus[Cluster]) for Cluster in [1,2,3]],color='black')
        plt.title(classification)
        plt.xticks([0,1,2],[f"Cluster {i}" for i in [1,2,3]])
        plt.xlabel('', fontdict={'fontsize': 20})
        plt.ylabel('Ratio of Connected Glomerulus', fontdict={'fontsize': 20})
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        ax.spines['bottom'].set_linewidth(1.5)  # X-axis
        ax.spines['left'].set_linewidth(1.5)  # Y-axis
        ax.spines['top'].set_linewidth(1.5)  # X-axis
        ax.spines['right'].set_linewidth(1.5)  # Y-axis
        plt.show()



def analysis_for_all_MGPN_ORN_PN():
    path = 'hemibrain_data/'
    file = 'Connection_upstream_of_M_s_w_3_v1.2.1.xlsx'
    color_dict = {1:"red",2:"gold",3:"deepskyblue",'KCg':'red',"KCa'b'":'gold',"KCab":"deepskyblue"}
    c = ConnectionSetting()
    network = gc.load_network()
    connection_data = pd.read_excel(f"{path}{file}")
    connection_data['up.type'] = connection_data['up.type'].fillna("")
    MGPN_ID_list = connection_data['down.bodyId'].unique().tolist()
    weight_collection = []
    for neuronId in MGPN_ID_list:
            # Mask for connections involving the current neuron ID
        mask = connection_data['down.bodyId'] == neuronId
        total_upstream_weight = connection_data.loc[mask, 'w.weight'].sum()

        # Mask for ORN inputs
        mask_ORN = mask & connection_data['up.type'].str.contains("ORN_")
        total_up_ORN_weight = connection_data.loc[mask_ORN, 'w.weight'].sum() / total_upstream_weight
        

        # Mask for uni-PN inputs
        mask_uni_PN = mask & connection_data['up.type'].str.endswith("PN")
        total_up_uniPN_weight = connection_data.loc[mask_uni_PN, 'w.weight'].sum() / total_upstream_weight

        # Append the weights to the collection
        weight_collection.append([total_up_ORN_weight, total_up_uniPN_weight])

    weight_collection = np.array(weight_collection)
    fig, ax = plt.subplots()
    ax.scatter(weight_collection[:, 0], weight_collection[:, 1], c='black', s=20)
    plt.xlabel('Input ratio from ORN', fontdict={'fontsize': 20})
    plt.ylabel('Input ratio from uniglomerular PN', fontdict={'fontsize': 20})
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    ax.spines['bottom'].set_linewidth(1.5)  # X-axis
    ax.spines['left'].set_linewidth(1.5)  # Y-axis
    ax.spines['top'].set_linewidth(1.5)  # X-axis
    ax.spines['right'].set_linewidth(1.5)  # Y-axis
    plt.show()    
