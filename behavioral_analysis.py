import generate_connection as gc
from generate_connection import ConnectionSetting
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import kruskal
from matplotlib import pyplot as plt
from pandas import DataFrame as Df
import seaborn as sns
import read_DoOR
import random as rd
from collections import Counter
from scipy import stats
from scikit_posthocs import posthoc_dunn
import os
import scikit_posthocs as sp
import copy

def get_odor_list(OdorType):
    ## derived from Knaden, M., Strutz, A., Ahsan, J., Sachse, S., & Hansson, B. S. (2012). Spatial representation of odorant valence in an insect brain. Cell reports, 1(4), 392-399.
    if OdorType == 'attractive':
        candidates = ["gamma-butyrolactone", "formic acid", "2,3-butanedione", "hexanoic acid", "pentanoic acid",
                "       3-methylthio-1-propanol", "propionic acid", "4-ethylguaiacol"]
    elif OdorType == 'aversive':
        candidates = ['1-octen-3-ol','acetophenone','linalool','2-methylphenol','benzaldehyde','1-octanol']
    elif OdorType == 'neutral':
        candidates = [
            "ethyl hexanoate", "beta-citronellol", "ammonium hydroxide", "E2-thyl butenoate", "pentanal", "ethanol",
            "acetaldehyde", "nerol", "linoleic acid", "E3-hexenol", "2-propenal", "E2-hexenol", "terpinolene", "(-)-E-caryophylene",
            "nonanoic acid", "E2-hexenal", "gamma-octalactone"
            ]
    Odor_to_Glomerulus_activity,Class_to_InChIKey, InChIKey_to_Class, InChIKey_to_Name, Name_to_InChIKey = read_DoOR.lookup_odor_to_Glomerulus()
    candidates = [i for i in candidates if i in Name_to_InChIKey if len(Odor_to_Glomerulus_activity[Name_to_InChIKey[i]])>10]
    return candidates

def get_ranks(lst):
    # Get the sorted indices of the list
    sorted_indices = sorted(range(len(lst)), key=lambda x: lst[x])
    # Create a list to store the ranks
    ranks = [0] * len(lst)
    # Assign ranks based on the sorted order
    for rank, index in enumerate(sorted_indices):
        ranks[index] = rank + 1  # Adding 1 to make ranks 1-based
    return ranks

def get_thresholding(KC_activity, network, classification='KC class'):
    if classification=='KC class':
        KC_class_list = ['KCg',"KCa'b'","KCab"]
        KCid_to_class_dict = network.KCid_to_Subtype
        Class_to_KCid_dict = network.Subtype_to_KCid
    else:
        KC_class_list = [i for i in list(network.New_subtype_to_id.keys())]
        KCid_to_class_dict = network.id_to_new_subtype
        Class_to_KCid_dict = network.New_subtype_to_id

    # # Process each row independently
    KC_class_activity_collection = []
    updated_activity_collection = []
    for i in range(KC_activity.shape[0]):
        KC_class_activity = [0 for _ in range(len(KC_class_list))]
        row = KC_activity[i, :]
        threshold = np.percentile(row, 90)  # 90th percentile threshold
        row[row<threshold] = 0
        updated_activity_collection.append(row)
        for KCindex, KCid in enumerate(network.KCid_list):
            KC_class_activity[KC_class_list.index(KCid_to_class_dict[KCid])] += row[KCindex]
        KC_class_activity_collection.append(KC_class_activity)
    KC_class_activity_collection = np.array(KC_class_activity_collection)
    return KC_class_activity_collection

def get_post_fix(classification, normalized, zero_replaced, post_fix, connectivity_model):
    if zero_replaced:
        zero_replaced_str='_zero'
    else:
        zero_replaced_str = '_nonzero'
    if classification == 'Glomerulus':
        if normalized:
            normalized_str = '_normalized'
        else:
            normalized_str = '_nonnormalized'
        post_fix = post_fix + normalized_str + zero_replaced_str
    elif classification == 'Cluster':
        if normalized:
            normalized_str = '_normalized'
        else:
            normalized_str = '_nonnormalized'
        post_fix = post_fix + normalized_str + zero_replaced_str
    elif classification == 'KC subclass':
        post_fix = post_fix + f'_{connectivity_model}'+ zero_replaced_str
    elif classification == 'KC class':
        post_fix = post_fix + f'_{connectivity_model}'+ zero_replaced_str
    return post_fix

def analyze_behavior_olfactory_correlation(Candidate_collections, Odor_activation_matrix, network, normalized=False, 
                                           classification='Glomerulus', connectivity_model='_FlyEM', zero_replaced=False,post_fix = ''):
    candidates, candidate_rankings, candidate_valences = Candidate_collections
    if classification == 'Glomerulus':
        classification_list = network.G_list
        if normalized:
            row_sums_G = np.sum(Odor_activation_matrix, axis=1)
            Odor_activation_matrix = (Odor_activation_matrix.T / row_sums_G).T
    elif classification == 'Cluster':
        classification_list = [1,2,3]
        if normalized:
            row_sums_G = np.sum(Odor_activation_matrix, axis=1)
            Odor_activation_matrix = (Odor_activation_matrix.T / row_sums_G).T
    elif classification == 'KC subclass':
        classification_list = list(network.New_subtype_to_id.keys())
    elif classification == 'KC class':
        classification_list = ['KCg',"KCa'b'",'KCab']
    post_fix = get_post_fix(classification,normalized,zero_replaced,post_fix,connectivity_model)
    Pooled = []
    for row_id, candidate in enumerate(candidates):
        Pooled.append(Odor_activation_matrix[row_id].tolist() + [candidate_rankings[row_id],candidate, candidate_valences[row_id]])
    Pooled = Df(data=Pooled, columns=classification_list+['Valence ranking','Odor','Valence'])
    Pooled.to_csv(f"Behavior_Odor_{classification}_activation_matrix{post_fix}.csv")
    p_list, r_list = [],[]
    nonused_G_list = []
    for G in classification_list:
        r,p = spearmanr(Pooled[G],Pooled['Valence ranking'])
        if np.isnan(r):
            nonused_G_list.append(G)
        p_list.append(p)
        r_list.append(r)
    df = pd.DataFrame({
    classification: classification_list,
    'p_value': p_list,
    'r_value': r_list
    })
    print("## original spearman's correlation ##")
    print(classification, post_fix[1:])
    print(df)
    ## Get spearman adjusted ratio
    adjusted_Odor_activation_matrix = copy.deepcopy(Odor_activation_matrix)
    for col_id, G in enumerate(classification_list):
        if G in nonused_G_list:
            continue
        adjusted_Odor_activation_matrix[:,col_id] = Odor_activation_matrix[:,col_id] * r_list[col_id]
    r,p = spearmanr(np.sum(adjusted_Odor_activation_matrix,axis=1),Pooled['Valence ranking'])
    print(f'Adjusted {classification} with valence, r, p',r,p)
    
    ############ make weighted activity ranking plot
    plt.plot(get_ranks(Pooled['Valence ranking'].values.tolist()),get_ranks(np.sum(adjusted_Odor_activation_matrix,axis=1).tolist()),'k.')
    plt.plot((0,len(Pooled['Valence ranking'])),(0,len(Pooled['Valence ranking'])),'r')
    plt.text(plt.xlim()[0]+0.5,plt.ylim()[0]+0.5,f'rs={r:.4f}, p={p:.4f}')
    plt.title(f"{classification}{post_fix}")
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)  # X-axis
    ax.spines['left'].set_linewidth(1.5)  # Y-axis
    ax.spines['top'].set_linewidth(1.5)  # X-axis
    ax.spines['right'].set_linewidth(1.5)  # Y-axis
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.set_ylabel("Weighted activity ranking",fontdict={'fontsize': 20})
    ax.set_xlabel("Preference ranking",fontdict={'fontsize': 20})
    plt.savefig(f'{classification}_adjusted_Spearman_scatter{post_fix}.png',dpi=500)
    plt.savefig(f'{classification}_adjusted_Spearman_scatter{post_fix}.svg')
    plt.show()
    plt.close()
    with open(f'Correlation of Adjusted {classification} and valence{post_fix}.txt','wt')as ff:
        ff.write(f'r={r}, p={p}')
    Pooled = []
    for row_id, candidate in enumerate(candidates):
        Pooled.append(adjusted_Odor_activation_matrix[row_id].tolist() + [candidate_rankings[row_id],candidate, candidate_valences[row_id]])
    Pooled = Df(data=Pooled, columns=classification_list+['Valence ranking','Odor','Valence'])
    Pooled.to_csv(f"Behavior_Odor_{classification}_activation_matrix_adjusted_by_spearson{post_fix}.csv")
    
    ### make barplot for spearman r
    df_sorted = df.sort_values(by='r_value', ascending=True)
    df_sorted.to_csv(f"{classification}_valence_r_p{post_fix}.csv")
    # Extract the sorted order
    if len(nonused_G_list) > 0:
        G_sorted = df_sorted[classification].values[:-len(nonused_G_list)]
        r_sorted = df_sorted['r_value'].values[:-len(nonused_G_list)]
        p_sorted = df_sorted['p_value'].values[:-len(nonused_G_list)]
    else:
        G_sorted = df_sorted[classification].values
        r_sorted = df_sorted['r_value'].values
        p_sorted = df_sorted['p_value'].values
    # Create a barplot of r-values
    if classification == 'Glomerulus' or classification == 'KC subclass':
        fig, ax = plt.subplots(figsize=(12.8,2.4))
    elif classification == 'Cluster' or classification == 'KC class':
        fig, ax = plt.subplots(figsize=(2.4,2.4))
    bars = ax.bar(range(len(G_sorted)), r_sorted, color='gray', edgecolor='black')
    ax.spines['bottom'].set_linewidth(1.5)  # X-axis
    ax.spines['left'].set_linewidth(1.5)  # Y-axis
    ax.spines['top'].set_linewidth(1.5)  # X-axis
    ax.spines['right'].set_linewidth(1.5)  # Y-axis
    ax.set_xticks(range(len(G_sorted)))
    ax.set_xticklabels(G_sorted, rotation=90)
    plt.yticks(fontsize=16)
    ax.set_ylabel("Spearman's r",fontdict={'fontsize': 20})
    ax.set_xlabel(classification,fontdict={'fontsize': 20})
    ax.set_title("Spearman's correlation with Valence")
    color_dict = {'aversive':'red','neutral':'gold','attractive':'deepskyblue',1:'red',2:'gold',3:'deepskyblue'}
    # Change the color of the xtick labels based on their cluster
    if classification == 'Glomerulus':
        for tick_label, g in zip(ax.get_xticklabels(), G_sorted):
            cluster_id = network.Glomerulus_to_Cluster[g]  # get cluster number for this glomerulus
            tick_label.set_color(color_dict[cluster_id])    
    plt.savefig(f'Behavior_{classification}_correlation{post_fix}.png',dpi=500)
    plt.savefig(f'Behavior_{classification}_correlation{post_fix}.svg')
    plt.close()
    if classification == 'Glomerulus' or classification == 'KC subclass':
        fig, ax = plt.subplots(figsize=(12.8,2.4))
    elif classification == 'Cluster' or classification == 'KC class':
        fig, ax = plt.subplots(figsize=(2.4,2.4))
    for row_id, candidate in enumerate(candidates):
        Pooled.append(Odor_activation_matrix[row_id].tolist() + [candidate_rankings[row_id],candidate, candidate_valences[row_id]])
    Pooled = Df(data=Pooled, columns=classification_list+[ 'Valence ranking','Odor', 'Valence'])
    Pooled.to_csv(f"Behavior_Odor_{classification}_activation_matrix{post_fix}.csv")

    Pooled_result = []
    for i in range(len(candidates)):
        for cluster_id, cluster in enumerate(classification_list):
            Pooled_result.append([candidates[i], candidate_valences[i] ,candidate_rankings[i], cluster, Odor_activation_matrix[i][cluster_id]])
    Pooled_result = Df(data=Pooled_result,columns=['Odor', 'Valence', 'Valence ranking',classification,'Activity'])
    Pooled_result.to_csv(f'Behavior_Odor_{classification}_activity_valence{post_fix}.csv')
    mask = Pooled_result['Valence'] == 'neutral'
    sns.boxplot(data=Pooled_result[~mask],x=classification,y='Activity',hue='Valence',palette=color_dict,hue_order=['aversive','attractive'],order=classification_list)
    ax.spines['bottom'].set_linewidth(1.5)  # X-axis
    ax.spines['left'].set_linewidth(1.5)  # Y-axis
    ax.spines['top'].set_linewidth(1.5)  # X-axis
    ax.spines['right'].set_linewidth(1.5)  # Y-axis
    plt.xticks(fontsize=16)
    if classification == 'KC class' or classification == 'KC subclass':
        plt.yticks([0,20,40,60],fontsize=16)
    elif classification == 'Cluster':
        plt.yticks([0,1,2,3],fontsize=16)
    else:
        plt.yticks([0,0.3,0.6,0.9],fontsize=16)
    ax.set_ylabel("Total activity",fontdict={'fontsize': 20})
    ax.set_xlabel(classification,fontdict={'fontsize': 20})
    ax.set_title(post_fix)
    plt.savefig(f'Behavior_Odor_valenceType_{classification}_boxplot{post_fix}.png',dpi=500)
    plt.savefig(f'Behavior_Odor_valenceType_{classification}_boxplot{post_fix}.svg')
    plt.close()
    # Perform Kruskal-Wallis H test for each 'KC class'
    results = []
    posthoc_results = {}
    if classification == 'Glomerulus':
        return
    Pooled_result = Pooled_result[~mask]
    for neuron_classification in classification_list:
        subset = Pooled_result[Pooled_result[classification] == neuron_classification]
        grouped_data = [subset[subset['Valence'] == odor_type]['Activity'].values for odor_type in subset['Valence'].unique()]

        # Perform Kruskal-Wallis H test
        stat, p_value = kruskal(*grouped_data)
        results.append({classification: neuron_classification, 'Statistic': stat, 'P-value': p_value})

        # Post-hoc Dunn's test if significant
        if p_value < 0.05:
            posthoc = sp.posthoc_dunn(subset, val_col='Activity', group_col='Valence', p_adjust='bonferroni')
            posthoc_results[neuron_classification] = posthoc
            print(neuron_classification,posthoc_results)
    # Create results DataFrame for Kruskal-Wallis H test
    results_df = pd.DataFrame(results)
    print(classification, post_fix, 'aversive vs attractive')
    print('kruskal wallis')
    print(results_df)
    return 

def get_G_cluster_matrix(candidates, network, zero_replaced, baseline='oil'):
    Odor_to_Glomerulus_activity,Class_to_InChIKey, InChIKey_to_Class, InChIKey_to_Name, Name_to_InChIKey = read_DoOR.lookup_odor_to_Glomerulus()
    if baseline:
        if baseline in Name_to_InChIKey:
            baseline_G_activity = Odor_to_Glomerulus_activity[Name_to_InChIKey[baseline]]
            if 'DL2d/v'in baseline_G_activity:
                baseline_G_activity['DL2d'] = baseline_G_activity['DL2d/v']
                baseline_G_activity['DL2v'] = baseline_G_activity['DL2d/v']
        else:
            baseline_G_activity = {}
    else:
        baseline_G_activity = {}
    Odor_G_matrix = np.zeros((len(candidates),len(network.G_list)))
    Odor_cluster_matrix = np.zeros((len(candidates),3))

    for row_id, candidate in enumerate(candidates):
        # Get the InChIKey for the candidate odor name
        inchi_key = Name_to_InChIKey[candidate]
        # Retrieve glomerulus activity for this odor
        glomerulus_activity = Odor_to_Glomerulus_activity[inchi_key]
        if 'DL2d/v'in glomerulus_activity:
            glomerulus_activity['DL2d'] = glomerulus_activity['DL2d/v']
            glomerulus_activity['DL2v'] = glomerulus_activity['DL2d/v']
        top_glomeruli = sorted(glomerulus_activity.items(), key=lambda x: x[1], reverse=True)
        for G, activity in top_glomeruli:
            if G in network.G_list:
                col_id = network.G_list.index(G)
                if G in baseline_G_activity:
                    activity = (activity - baseline_G_activity[G])
                    if activity < 0 and zero_replaced:
                        activity = 0
                Odor_G_matrix[row_id][col_id] = activity
                col_id = network.Glomerulus_to_Cluster[G]-1
                Odor_cluster_matrix[row_id][col_id] += activity
    return Odor_G_matrix, Odor_cluster_matrix

def from_G_to_KC(Odor_G_matrix, candidates, network, classification='KC class', connectivity_model='FlyEM', network_index=0, zero_replaced=False):
    if classification=='KC class':
        KC_class_list = ['KCg',"KCa'b'","KCab"]
        KCid_to_class_dict = network.KCid_to_Subtype
        Class_to_KCid_dict = network.Subtype_to_KCid
    else:
        KC_class_list = [i for i in list(network.New_subtype_to_id.keys())]
        KCid_to_class_dict = network.id_to_new_subtype
        Class_to_KCid_dict = network.New_subtype_to_id
    network.transform_PN_KC_connection_to_G_KC_connection_norm()
    weight = network.connection_matrix_collection_dict_g_norm[connectivity_model][network_index]
    KC_activity = np.matmul(Odor_G_matrix,weight)
    original_class_activity_matrix = np.zeros((len(candidates), len(KC_class_list)))
    for row_id in range(len(candidates)):
        for KCindex,KCid in enumerate(network.KCid_list):
            original_class_activity_matrix[row_id][KC_class_list.index(KCid_to_class_dict[KCid])] += KC_activity[row_id][KCindex]
    original_class_activity_df = Df(data=original_class_activity_matrix, columns=KC_class_list)
    if zero_replaced:
        zero_replaced = 'zero'
    else:
        zero_replaced = 'nonzero'
    original_class_activity_df.to_csv(f"{classification}_original_KC_activity_{connectivity_model}_{zero_replaced}.csv")
    return KC_activity,original_class_activity_matrix

def analyze_ranking_correlation_behavior_activity():
    network = gc.load_network()
    network.get_new_KC_subtype()
    file = 'Result/odor_AI.xlsx'
    data= pd.read_excel(file)
    odor_AI_dict = data.set_index('Odor')['AI'].to_dict()
    odor_Rank_dict = data.set_index('Odor')['Rank'].to_dict()
    odor_Valence_dict = data.set_index('Odor')['Valence'].to_dict()
    Odor_to_Glomerulus_activity,Class_to_InChIKey, InChIKey_to_Class, InChIKey_to_Name, Name_to_InChIKey = read_DoOR.lookup_odor_to_Glomerulus()
    candidates = [odor for odor in odor_AI_dict if odor in Name_to_InChIKey if len(Odor_to_Glomerulus_activity[Name_to_InChIKey[odor]])>10]
    candidate_rankings = [odor_Rank_dict[odor] for odor in candidates]
    candidate_valences = [odor_Valence_dict[odor] for odor in candidates]
    Candidate_collection = [candidates, candidate_rankings, candidate_valences]

    for zero_replaced in [False]:
        for normalized in [False]:
            Odor_G_matrix, Odor_cluster_matrix = get_G_cluster_matrix(candidates, network, zero_replaced=zero_replaced, baseline='oil')
            analyze_behavior_olfactory_correlation(Candidate_collection, Odor_G_matrix, network, normalized=normalized, classification='Glomerulus', zero_replaced=zero_replaced)
            analyze_behavior_olfactory_correlation(Candidate_collection, Odor_cluster_matrix, network, normalized=normalized,classification='Cluster',zero_replaced=zero_replaced)
            for connectivity_model in ['FlyEM','Random network']:
                for classification in ['KC class','KC subclass']:
                    KC_activity, original_class_activity = from_G_to_KC(Odor_G_matrix, candidates, network, classification=classification, connectivity_model=connectivity_model, zero_replaced=zero_replaced)
                    analyze_behavior_olfactory_correlation(Candidate_collection, original_class_activity, network, False, classification=classification, connectivity_model=connectivity_model)
                    thresholding_class_activity = get_thresholding(KC_activity, network, classification=classification)
                    analyze_behavior_olfactory_correlation(Candidate_collection, thresholding_class_activity, network, False, classification=classification, post_fix='_thresholding',connectivity_model=connectivity_model,zero_replaced=zero_replaced)

def compare_cluster_activity_for_aversive_and_attractive_odors():
    seed = 100
    rd.seed(seed)
    np.random.seed(seed)
    network = gc.load_network()
    candidates = get_odor_list('aversive') + get_odor_list('attractive')
    candidate_labels = ['aversive' for _ in get_odor_list('aversive')] + ['attractive' for _ in get_odor_list('attractive')]
    Odor_to_Glomerulus_activity,Class_to_InChIKey, InChIKey_to_Class, InChIKey_to_Name, Name_to_InChIKey = read_DoOR.lookup_odor_to_Glomerulus()
    baseline_G_activity = Odor_to_Glomerulus_activity[Name_to_InChIKey['oil']]
    if 'DL2d/v'in baseline_G_activity:
        baseline_G_activity['DL2d'] = baseline_G_activity['DL2d/v']
        baseline_G_activity['DL2v'] = baseline_G_activity['DL2d/v']
    Odor_G_matrix = np.zeros((len(candidates),len(network.G_list)))
    Odor_cluster_matrix = np.zeros((len(candidates),3))
    for row_id, candidate in enumerate(candidates):
        # Get the InChIKey for the candidate odor name
        inchi_key = Name_to_InChIKey[candidate]
        # Retrieve glomerulus activity for this odor
        glomerulus_activity = Odor_to_Glomerulus_activity[inchi_key]
        if 'DL2d/v'in glomerulus_activity:
            glomerulus_activity['DL2d'] = glomerulus_activity['DL2d/v']
            glomerulus_activity['DL2v'] = glomerulus_activity['DL2d/v']
        top_glomeruli = sorted(glomerulus_activity.items(), key=lambda x: x[1], reverse=True)
        for G, activity in top_glomeruli:
            if G in network.G_list:
                col_id = network.G_list.index(G)
                if G in baseline_G_activity:
                    activity = (activity - baseline_G_activity[G])
                Odor_G_matrix[row_id][col_id] = activity
            else:
                print(G, 'not in network')
        for G, activity in top_glomeruli:
            if G in network.G_list:
                if G in baseline_G_activity:
                    activity = activity - baseline_G_activity[G]
                col_id = network.Glomerulus_to_Cluster[G]-1
                Odor_cluster_matrix[row_id][col_id] += activity
                # Count_cluster[col_id] += 1
        Activity_sum = np.sum(Odor_cluster_matrix[row_id])
        if Activity_sum == 0:
            raise ValueError(f"Activity should not be zero for {odor}")
        for col_id in range(len(Odor_cluster_matrix[0])):
            Odor_cluster_matrix[row_id][col_id] = Odor_cluster_matrix[row_id][col_id]/Activity_sum
    G_cluster_list = [1,2,3]    
    Pooled_result = []
    for i in range(len(candidates)):
        for cluster_id, cluster in enumerate(G_cluster_list):
            Pooled_result.append([candidates[i], candidate_labels[i], cluster, Odor_cluster_matrix[i][cluster_id]])
    Pooled_result = Df(data=Pooled_result,columns=['Odor','Type','Cluster','Normalized activity'])
    Pooled_result.to_csv(f'Behavior_Odor_cluster_activity.csv')

    # Perform Kruskal-Wallis H test for each Cluster
    results = []
    for cluster in Pooled_result['Cluster'].unique():
        subset = Pooled_result[Pooled_result['Cluster'] == cluster]
        grouped_data = [subset[subset['Type'] == odor_type]['Normalized activity'].values for odor_type in subset['Type'].unique()]

        # Perform Kruskal-Wallis H test
        stat, p_value = kruskal(*grouped_data)
        results.append({
            'Cluster': cluster,
            'Statistic': stat,
            'P-value': p_value
        })

    # Create a DataFrame to display the results
    results_df = pd.DataFrame(results)
    print(results_df)
    sns.boxplot(data=Pooled_result, x='Cluster',y='Normalized activity',hue='Type')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)  # X-axis
    ax.spines['left'].set_linewidth(1.5)  # Y-axis
    ax.spines['top'].set_linewidth(1.5)  # X-axis
    ax.spines['right'].set_linewidth(1.5)  # Y-axis
    plt.ylabel("Normalized activity",fontdict={'fontsize': 20})
    plt.xlabel("Cluster",fontdict={'fontsize': 20})
    plt.savefig("Aversive_attractive_cluster_activity.png",dpi=500)
    plt.savefig("Aversive_attractive_cluster_activity.svg")
    plt.show()
    plt.close()

def compare_glomerulus_across_ranking(rank_max=5):
    odor_type_list = ['aversive', 'neutral', 'attractive']
    Pooled = pd.DataFrame()
    network = gc.load_network()
    connection_pref = pd.read_csv("Result/Connection_pref_final.csv")
    no_pref_G = []
    for G in network.G_list:
        mask = connection_pref['Glomerulus'] == G
        filtered_data = connection_pref[mask]
        if (filtered_data['z score'].abs() < 2).all():
            no_pref_G.append(G)
    pref_G = [G for G in network.G_list if G not in no_pref_G]
    Pref_G_dict = {'Pref':pref_G, 'No pref':no_pref_G}

    for odor_type in odor_type_list:
        for rank_num in [i+1 for i in range(rank_max)]:
            file = f'{odor_type} odor_bias_cluster_top{rank_num}.csv'
            if not os.path.isfile(file):
                continue
            data = pd.read_csv(file)
            data['Odor type'] = [odor_type if model == 'Real odor' else 'random' for model in data["Model"]]
            data['Ranking'] = rank_num
            data['Normalized count'] = [count/len(network.Cluster_to_Glomerulus[cluster]) for count, cluster in zip(data['Count'],data['Cluster'])] 
            
            if mask.sum() == 0:
                print(f"No data for 'Real odor' in file {file}")
            else:
                Pooled = pd.concat([Pooled, data], ignore_index=True)

    if Pooled.empty:
        print("No data to process.")
        return
    Pooled.to_csv(f"cross_compare_cluster_pooled.csv", index=False)
    results = []
    kruskal_results = []
    posthoc_results = []

    for odor_type in odor_type_list:
        for rank_num in [i + 1 for i in range(rank_max)]:
            # Filter data
            mask_odortype = Pooled['Odor type'] == odor_type
            mask_ranknum = Pooled["Ranking"] == rank_num
            mask_model = Pooled['Model'] == 'Real odor'
            filtered = Pooled[mask_odortype & mask_ranknum & mask_model]

            for odor in filtered['Odor'].unique().tolist():
                # Filter for Cluster == 1 and Cluster == 3
                count_1 = filtered.loc[(filtered['Odor'] == odor) & (filtered['Cluster'] == 1), 'Count'].sum()
                count_3 = filtered.loc[(filtered['Odor'] == odor) & (filtered['Cluster'] == 3), 'Count'].sum()

                # Compute the difference
                count_diff = count_1 - count_3
                results.append([odor, odor_type, rank_num, count_diff])

    # Convert results to DataFrame
    results_df = pd.DataFrame(results, columns=['Odor', 'Odor type', 'Ranking', 'N1-N3'])

    # Perform Kruskal-Wallis test and Dunn's test for each ranking
    for rank_num in results_df['Ranking'].unique():
        subset = results_df[results_df['Ranking'] == rank_num]
        groups = [subset[subset['Odor type'] == odor_type]['N1-N3'] for odor_type in odor_type_list]

        # Ensure there are sufficient data points for comparison
        if all(len(group) > 1 for group in groups):
            # Kruskal-Wallis test
            stat, p_value = kruskal(*groups)
            kruskal_results.append({
                'Ranking': rank_num,
                'Statistic': stat,
                'p-value': p_value
            })

            # Prepare data for post-hoc test
            subset['Group'] = subset['Odor type']
            posthoc = posthoc_dunn(subset, val_col='N1-N3', group_col='Group', p_adjust='bonferroni')
            
            # Store post-hoc results
            posthoc['Ranking'] = rank_num
            posthoc_results.append(posthoc)

    # Save Kruskal-Wallis results
    kruskal_results_df = pd.DataFrame(kruskal_results)
    kruskal_results_df.to_csv("Kruskal_Wallis_Results.csv", index=False)

    # Combine post-hoc results into a single DataFrame
    posthoc_combined = pd.concat(posthoc_results)
    posthoc_combined.to_csv("Dunn_Posthoc_Results.csv", index=False)
    print(kruskal_results_df)
    print(posthoc_combined)
    # Save results to CSV and plot
    results_df.to_csv("cross_compare_diff.csv", index=False)
    sns.boxplot(data=results_df, x='Ranking', y='N1-N3', hue='Odor type', hue_order=['aversive', 'neutral', 'attractive'])
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)  # X-axis
    ax.spines['left'].set_linewidth(1.5)  # Y-axis
    ax.spines['top'].set_linewidth(1.5)  # X-axis
    ax.spines['right'].set_linewidth(1.5)  # Y-axis
    plt.xlabel('Ranking', fontdict={'fontsize': 20})
    plt.ylabel('N1 - N3', fontdict={'fontsize': 20})
    plt.yticks([-4,0,4],fontsize=16)
    plt.xticks(fontsize=16)
    plt.savefig(f"Valence N1 - N3along ranking.png",dpi=500)
    plt.savefig(f"Valence N1 - N3along ranking.svg",dpi=500)
    plt.show()

    for odor_type in odor_type_list:
        mask = Pooled['Odor type'] == odor_type
        sns.lineplot(data=Pooled[mask], x='Ranking',y='Normalized count',hue='Cluster',palette={1:'red',2:'goldenrod',3:'deepskyblue'},
                     linewidth=3, alpha=0.6)
        plt.title(odor_type)
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5)  # X-axis
        ax.spines['left'].set_linewidth(1.5)  # Y-axis
        ax.spines['top'].set_linewidth(1.5)  # X-axis
        ax.spines['right'].set_linewidth(1.5)  # Y-axis
        plt.xlabel('Ranking', fontdict={'fontsize': 20})
        plt.ylabel('Normalized count', fontdict={'fontsize': 20})
        plt.yticks(fontsize=16)
        plt.xticks([1,2,3,4,5], fontsize=16)
        plt.savefig(f"{odor_type} cluster along ranking.png",dpi=500)
        plt.savefig(f"{odor_type} cluster along ranking.svg")
        plt.show()
        plt.close()

    for cluster in [1,2,3]:
        mask = Pooled['Cluster'] == cluster
        sns.lineplot(data=Pooled[mask], x='Ranking',y='Normalized count',hue='Odor type',palette={'aversive':'red','neutral':'goldenrod','attractive':'deepskyblue','random':'gray'},
                     linewidth=3,alpha=0.6)
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5)  # X-axis
        ax.spines['left'].set_linewidth(1.5)  # Y-axis
        ax.spines['top'].set_linewidth(1.5)  # X-axis
        ax.spines['right'].set_linewidth(1.5)  # Y-axis
        plt.xlabel('Ranking', fontdict={'fontsize': 20})
        plt.ylabel('Normalized count', fontdict={'fontsize': 20})
        plt.yticks(fontsize=16)
        plt.xticks([1, 2, 3, 4, 5], fontsize=16)
        plt.title(cluster)
        plt.savefig(f"Cluster {cluster} valence along ranking.png",dpi=500)
        plt.savefig(f"Cluster {cluster} valence along ranking.svg")
        plt.show()
        plt.close()

    
    Pooled = pd.DataFrame()
    for odor_type in odor_type_list:
        for rank_num in [i+1 for i in range(rank_max)]:
            file = f'{odor_type} odor_bias_pref_nopref_top{rank_num}.csv'
            if not os.path.isfile(file):
                continue
            data = pd.read_csv(file)
            data['Odor type'] = odor_type
            data['Ranking'] = rank_num
            data['Normalized count'] = [count/len(Pref_G_dict[classification]) for count, classification in zip(data['Count'],data['Classification'])] 
            
            if mask.sum() == 0:
                print(f"No data for 'Real odor' in file {file}")
            else:
                Pooled = pd.concat([Pooled, data], ignore_index=True)

    if Pooled.empty:
        print("No data to process.")
        return
    
    Pooled.to_csv(f"cross_compare_pref_pooled.csv", index=False)
    Pooled = pd.read_csv(f"cross_compare_pref_pooled.csv")
    mask = Pooled['Classification'] == 'No pref' 
    odor_type_list_df = []
    for odor_type, model in zip(Pooled['Odor type'],Pooled['Model']):
        if odor_type == 'neutral' and model=='Real odor':
            odor_type_list_df.append('neutral')
        elif model == 'Real odor':
            odor_type_list_df.append('non-neutral')
        else:
            odor_type_list_df.append('random')
    Pooled['Odor type'] = odor_type_list_df
    # sns.lineplot(data=Pooled[mask], x='Ranking',y='Normalized count',hue='Odor type',palette={'aversive':'red','neutral':'gold','attractive':'deepskyblue', 'non-neutral':'black'},
                #  alpha=0.3)
    sns.lineplot(data=Pooled[mask], x='Ranking',y='Normalized count',hue='Odor type',palette={'aversive':'red','neutral':'gold','attractive':'deepskyblue', 'non-neutral':'black','random':'gray'},
                 alpha=0.3,marker="o",  # Marker for scatter points
                err_style="bars")
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)  # X-axis
    ax.spines['left'].set_linewidth(1.5)  # Y-axis
    ax.spines['top'].set_linewidth(1.5)  # X-axis
    ax.spines['right'].set_linewidth(1.5)  # Y-axis
    plt.xlabel('Ranking', fontdict={'fontsize': 20})
    plt.ylabel('Normalized count', fontdict={'fontsize': 20})
    plt.yticks(fontsize=16)
    plt.xticks([1, 2, 3,4,5], fontsize=16)
    plt.title("No pref")
    plt.savefig("No pref along ranking.png",dpi=500)
    plt.savefig("No pref along ranking.svg")
    plt.close()
    Pooled_no_pref = Pooled[Pooled['Classification'] == 'No pref']
    # odor_types = ['aversive', 'neutral', 'attractive']
    odor_types = ['non-neutral', 'neutral','random']

    for ranking, group in Pooled_no_pref.groupby('Ranking'):
        data_groups = [group.loc[group['Odor type'] == ot, 'Normalized count'] for ot in odor_types]

        # Perform Kruskal-Wallis if each group is non-empty
        if all(len(d) > 1 for d in data_groups):
            stat, p_val = kruskal(*data_groups)
            print(f"Ranking: {ranking}, Kruskal-Wallis p-value: {p_val}")

            if p_val < 0.05:
                # Dunn's test requires a list of arrays
                posthoc = sp.posthoc_dunn(data_groups, p_adjust='bonferroni')
                # The rows and columns correspond to groups in the order you provided them
                # i.e., 0 = aversive, 1 = neutral, 2 = attractive
                posthoc.columns = odor_types
                posthoc.index = odor_types
                print(f"Post-hoc Dunn test for Ranking {ranking}:\n", posthoc)

def get_behavior_with_odor_glomerulus_ranking(OdorType='neutral', rank_num=3):
    seed = 100
    rd.seed(seed)
    np.random.seed(seed)
    figure_path = 'Final_figures_summary/'
    if not os.path.isdir(figure_path): os.mkdir(figure_path)
    Odor_to_Glomerulus_activity,Class_to_InChIKey, InChIKey_to_Class, InChIKey_to_Name, Name_to_InChIKey = read_DoOR.lookup_odor_to_Glomerulus()
    candidates = get_odor_list(OdorType)
    candidates = [i for i in candidates if i in Name_to_InChIKey]
    baseline_G_activity = Odor_to_Glomerulus_activity[Name_to_InChIKey['oil']]
    if 'DL2d/v'in baseline_G_activity:
        baseline_G_activity['DL2d'] = baseline_G_activity['DL2d/v']
        baseline_G_activity['DL2v'] = baseline_G_activity['DL2d/v']
    network = gc.load_network()
    connection_pref = pd.read_csv("Result/Connection_pref_final.csv")
    no_pref_G = []
    for G in network.G_list:
        mask = connection_pref['Glomerulus'] == G
        filtered_data = connection_pref[mask]
        if (filtered_data['z score'].abs() < 2).all():
            no_pref_G.append(G)
    pref_G = [G for G in network.G_list if G not in no_pref_G]
    Pref_G_dict = {'Pref':pref_G, 'No pref':no_pref_G}
    top_glomeruli_counter = Counter()
    top_cluster_counter = Counter()
    Pooled = []
    Pooled_G = []
    Pooled_pref = []
    for candidate in candidates:
        # Get the InChIKey for the candidate odor name
        inchi_key = Name_to_InChIKey[candidate]
        # Retrieve glomerulus activity for this odor
        glomerulus_activity = Odor_to_Glomerulus_activity[inchi_key]
        if 'DL2d/v'in glomerulus_activity:
            glomerulus_activity['DL2d'] = glomerulus_activity['DL2d/v']
            glomerulus_activity['DL2v'] = glomerulus_activity['DL2d/v']
        ## Baseline
        for G in glomerulus_activity:
            activity = glomerulus_activity[G]
            if G in network.G_list:
                if G in baseline_G_activity:
                    activity = (activity - baseline_G_activity[G])
                glomerulus_activity[G] = activity
        top_glomeruli = sorted(glomerulus_activity.items(), key=lambda x: x[1], reverse=True)
        top_glomeruli = top_glomeruli[:rank_num]
        k = [KK[0] for KK in top_glomeruli]
        for G in network.G_list:
            Pooled_G.append([candidate, G, Counter(k)[G],'Real odor'])        
        top_cluster = [network.Glomerulus_to_Cluster[G[0]] for G in top_glomeruli if G[0] in network.Glomerulus_to_Cluster]
        top_pref = ['Pref' if G[0] in pref_G else 'No pref' for G in top_glomeruli ]
        for cluster in [1,2,3]:
            count = Counter(top_cluster)[cluster]
            weighted_count = 0
            for r,k in enumerate(top_cluster):
                if k == cluster:
                    weighted_count += (rank_num+1-r)
            normalized_count = count/len(network.Cluster_to_Glomerulus[cluster])
            Pooled.append([candidate, cluster, count, normalized_count,'Real odor',weighted_count])
        for classification in Pref_G_dict:
            count = Counter(top_pref)[classification]
            weighted_count = 0
            for r,k in enumerate(top_pref):
                if k == classification:
                    weighted_count += (rank_num+1-r)
            normalized_count = count/len(Pref_G_dict[classification])
            Pooled_pref.append([candidate, classification,count, normalized_count, 'Real odor', weighted_count])
        # Update counter with glomeruli in the top
        top_glomeruli_counter.update([g[0] for g in top_glomeruli])
        top_cluster_counter.update([cluster for cluster in top_cluster])
    for exp_index in range(100):
        top_glomeruli = rd.sample(network.G_list, rank_num)
        for G in network.G_list:
            Pooled_G.append([f'random', G, Counter(top_glomeruli)[G],'Random model'])
        top_cluster = [network.Glomerulus_to_Cluster[G] for G in top_glomeruli if G in network.Glomerulus_to_Cluster]
        top_pref = ['Pref' if G[0] in pref_G else 'No pref' for G in top_glomeruli ]
        for cluster in [1,2,3]:
            count = Counter(top_cluster)[cluster]
            weighted_count = 0
            for r,k in enumerate(top_cluster):
                if k == cluster:
                    weighted_count += (rank_num+1-r)
            normalized_count = count/len(network.Cluster_to_Glomerulus[cluster])
            Pooled.append([f'random', cluster, count, normalized_count,'Random model', weighted_count])
        for classification in Pref_G_dict:
            count = Counter(top_pref)[classification]
            print(count)
            weighted_count = 0
            for r,k in enumerate(top_pref):
                if k == classification:
                    weighted_count += (rank_num+1-r)
            normalized_count = count/len(Pref_G_dict[classification])
            Pooled_pref.append(['random', classification,count, normalized_count, 'Random model', weighted_count])
            # input()
    ## cluster
    Pooled = Df(data=Pooled, columns = ["Odor",'Cluster','Count','Normalized count','Model','Weighted count'])
    Pooled.to_csv(f"{OdorType} odor_bias_cluster_top{rank_num}.csv")
    for cluster in network.Cluster_to_Glomerulus:
        real_odor_counts = Pooled[(Pooled['Cluster'] == cluster) & (Pooled['Model'] == 'Real odor')]['Count']
        random_model_counts = Pooled[(Pooled['Cluster'] == cluster) & (Pooled['Model'] == 'Random model')]['Count']
        # Mann-Whitney U test
        if not real_odor_counts.empty and not random_model_counts.empty:  # Ensure there is data in both groups
            stat, p_value = stats.mannwhitneyu(real_odor_counts, random_model_counts, alternative='two-sided')
            print(f"Mann-Whitney U test for Cluster {cluster}: U-statistic = {stat}, p-value = {p_value}")
        else:
            print(f"Insufficient data for Cluster {cluster} to perform the test.")

    ## pref_no_pref
    Pooled_pref = Df(data=Pooled_pref, columns = ["Odor",'Classification','Count', 'Normalized count','Model', 'Weighted count'])
    Pooled_pref.to_csv(f"{OdorType} odor_bias_pref_nopref_top{rank_num}.csv")

    for classification in ['Pref', "No pref"]:
        real_odor_counts = Pooled_pref[(Pooled_pref['Classification'] == classification) & (Pooled_pref['Model'] == 'Real odor')]['Count']
        random_model_counts = Pooled_pref[(Pooled_pref['Classification'] == classification) & (Pooled_pref['Model'] == 'Random model')]['Count']
        # Mann-Whitney U test
        if not real_odor_counts.empty and not random_model_counts.empty:  # Ensure there is data in both groups
            stat, p_value = stats.mannwhitneyu(real_odor_counts, random_model_counts, alternative='two-sided')
            print(f"Mann-Whitney U test for Classification {classification}: U-statistic = {stat}, p-value = {p_value}")
        else:
            print(f"Insufficient data for Classification {classification} to perform the test.")

    ### Glomerulus
    Pooled_G = Df(data=Pooled_G, columns = ["Odor",'Glomerulus','Count','Model'])
    plt.figure(figsize=(19.2, 4.8))
    sns.barplot(data=Pooled_G,x='Glomerulus',y='Count',hue='Model', palette={'Real odor':'white','Random model':'black'},edgecolor='black', order=network.G_list)
    plt.xlabel('Glomerulus')
    plt.ylabel(f'Top {rank_num} count', fontdict={'fontsize': 20})
    plt.yticks( fontsize=16)
    plt.xticks(fontsize=16,rotation=90)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)  # X-axis
    ax.spines['left'].set_linewidth(1.5)  # Y-axis
    ax.spines['top'].set_linewidth(1.5)  # X-axis
    ax.spines['right'].set_linewidth(1.5)  # Y-axis
    plt.savefig(f"{figure_path}glomerulus count of top{rank_num} for {OdorType} odors.png",dpi=500)
    plt.savefig(f"{figure_path}glomerulus count of top{rank_num} for {OdorType} odors.svg")
    plt.close()
    # plt.show()
    Pooled_G.to_csv(f"{OdorType} odor_bias_G_top{rank_num}.csv")
    odor_pref_G = []
    for G in network.G_list:

        real_odor_counts = Pooled_G[(Pooled_G['Glomerulus'] == G) & (Pooled_G['Model'] == 'Real odor')]['Count']
        random_model_counts = Pooled_G[(Pooled_G['Glomerulus'] == G) & (Pooled_G['Model'] == 'Random model')]['Count']
        
        # Mann-Whitney U test
        if not real_odor_counts.empty and not random_model_counts.empty:  # Ensure there is data in both groups
            stat, p_value = stats.mannwhitneyu(real_odor_counts, random_model_counts, alternative='two-sided')
            if p_value <0.05:
                print(f"Mann-Whitney U test for Glomerulus {G}: U-statistic = {stat}, p-value = {p_value}")
                odor_pref_G.append(G)
        else:
            print(f"Insufficient data for Glomerulus {G} to perform the test.")
    print(odor_pref_G)
    total_G = len(network.G_list)
    total_count_top = sum(top_cluster_counter.values())
    original_cluster_ratio_counter = Counter([network.Glomerulus_to_Cluster[G] for G in network.G_list])
    print(original_cluster_ratio_counter)
    result = [top_cluster_counter[cluster]/len(candidates) for cluster in [1,2,3]]
    plt.bar([1,2,3],result,color=['red','gold','deepskyblue'])
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)  # X-axis
    ax.spines['left'].set_linewidth(1.5)  # Y-axis
    ax.spines['top'].set_linewidth(1.5)  # X-axis
    ax.spines['right'].set_linewidth(1.5)  # Y-axis
    plt.xlabel('Cluster', fontdict={'fontsize': 20})
    plt.ylabel('Averaged Top Cluster Count', fontdict={'fontsize': 20})
    plt.yticks([0, 0.5, 1, 1.5], fontsize=16)
    plt.xticks([1, 2, 3], fontsize=16)
    plt.savefig(f"{figure_path}average count of top{rank_num} for {OdorType} odors.png",dpi=500)
    plt.savefig(f"{figure_path}average count of top{rank_num} for {OdorType} odors.svg")
    plt.close()

    # network.Glomerulus_to_Cluster[G]/total_G*total_count_top
    compared_result = [(top_cluster_counter[cluster]-((original_cluster_ratio_counter[cluster]/total_G)*total_count_top))/len(candidates) for cluster in [1,2,3]]
    plt.bar([1,2,3],compared_result,color=['red','gold','deepskyblue'])
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)  # X-axis
    ax.spines['left'].set_linewidth(1.5)  # Y-axis
    ax.spines['top'].set_linewidth(1.5)  # X-axis
    ax.spines['right'].set_linewidth(1.5)  # Y-axis
    plt.xlabel('Cluster', fontdict={'fontsize': 20})
    plt.ylabel(f'Normalized Top{rank_num} ount', fontdict={'fontsize': 20})
    plt.yticks([-0.5, 0, 0.5], fontsize=16)
    plt.xticks([1, 2, 3], fontsize=16)
    plt.savefig(f"{figure_path}bias of top{rank_num} for {OdorType} odors.png",dpi=500)
    plt.savefig(f"{figure_path}bias count of top{rank_num} for {OdorType} odors.svg")
    plt.close()