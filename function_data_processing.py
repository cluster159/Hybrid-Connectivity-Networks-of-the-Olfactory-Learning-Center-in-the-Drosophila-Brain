from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import copy
from scipy import stats
from matplotlib import rcParams
from scipy.stats import pearsonr
from scipy.stats import mannwhitneyu
import random as rd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare
from pandas import DataFrame as Df

rcParams['font.family'] = 'Arial'

def compare_subregions():
    # Load data
    path = 'Functional exp/'
    file = "Different_subregion_max.xlsx"
    data = pd.read_excel(path + file)
    
    # Extract unique odors and subregions
    odor_list = data.iloc[:, 0].unique().tolist()  # Assuming odor is the first column
    subregion_list = data.iloc[:, -1].unique().tolist()  # Assuming subregion is the last column
    subregion_list.reverse()
    
    # Create dictionary for subregion responses
    subregion_response_dict = {}
    for i, row in data.iterrows():
        subregion = row.iloc[-1]
        odor = row.iloc[0]
        if subregion not in subregion_response_dict:
            subregion_response_dict[subregion] = {}
        subregion_response_dict[subregion][odor] = np.array(row.iloc[1:-1].values, dtype=float)  # Assuming responses are in middle columns
    Pooled_dataframe = []

    # Perform Friedman test for each odor
    all_odor_friedman_material = [[] for _ in range(len(subregion_list))]
    for odor in odor_list:
        friedman_material = []
        for s1id, subregion in enumerate(subregion_list):
            responses = subregion_response_dict[subregion].get(odor, []).tolist()
            for flyid,r in enumerate(responses):
                Pooled_dataframe.append([subregion, odor, r, flyid])
                Pooled_dataframe.append([subregion, 'Pooled', r, flyid])

            friedman_material.append(responses)
            all_odor_friedman_material[s1id] += responses

        # Perform the Friedman test
        stat, p_value = friedmanchisquare(*friedman_material)
        print(f"Odor: {odor}")
        print("Friedman test statistic:", stat)
        print("p-value:", p_value)

        # Post-hoc analysis
        if p_value < 0.05:
            posthoc = sp.posthoc_conover_friedman(np.array(friedman_material).T, p_adjust='holm')
            print(f"Post-hoc results for {odor}:")
            print(posthoc)
    Pooled_dataframe = Df(data=Pooled_dataframe,columns=['Sub-region','Odor', 'Response','FlyId'])
    print("\nPooled Analysis")
    stat, p_value = friedmanchisquare(*all_odor_friedman_material)
    print("Friedman test statistic:", stat)
    print("p-value:", p_value)
    # Post-hoc analysis for pooled data
    if p_value < 0.05:
        pooled_posthoc = sp.posthoc_conover_friedman(np.array(all_odor_friedman_material).T, p_adjust='holm')
        print("Post-hoc results for pooled data:")
        print(pooled_posthoc)


    import distinctipy
    colors = distinctipy.get_colors(len(subregion_list))
    hue_dict = {subregion:color for subregion, color in zip(subregion_list,colors)}
    order_list = ['Pooled','PA','BA','EP','cVA','EtOH'] 
    sns.violinplot(data=Pooled_dataframe,x='Odor',y='Response',hue='Sub-region',palette=hue_dict, order=order_list)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)  # X-axis
    ax.spines['left'].set_linewidth(1.5)  # Y-axis
    ax.spines['top'].set_linewidth(1.5)  # X-axis
    ax.spines['right'].set_linewidth(1.5)  # Y-axis
    plt.xlabel('Odor', fontdict={'fontsize': 20})
    plt.ylabel('Response strength', fontdict={'fontsize': 20})
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    plt.show()
    ###### inter-fly difference
    subject_response_list = []
    shuffled_response_list = []
    for flyId in range(8):
        mask = Pooled_dataframe['FlyId'] == flyId
        subject_response_list.append(Pooled_dataframe[mask]['Response'].values.tolist())
        shuffled_response_list.append(np.random.permutation(Pooled_dataframe[mask]['Response'].values.tolist()))
    subject_corr = np.corrcoef(subject_response_list,rowvar=True)
    sns.clustermap(data=subject_corr, vmin=-1, vmax=1, cmap='bwr')
    plt.show()
    shuffled_corr = np.corrcoef(shuffled_response_list,rowvar=True)
    sns.clustermap(data=shuffled_corr, vmin=-1, vmax=1, cmap='bwr')
    plt.show()

    subject_corr_list = []
    for i in range(len(subject_corr)):
        for j in range(i, len(subject_corr)):
            if i==j:
                continue
            subject_corr_list.append([subject_corr[i][j],'fly'])
            subject_corr_list.append([shuffled_corr[i][j],'shuffled model'])
    
    sns.violinplot(data=Df(data=subject_corr_list, columns=['Inter-fly correlation','Model']), x='Model', y='Inter-fly correlation')
    plt.show()
    ## get the correlation between 

    for subregion in subregion_list:
        subject_response_list = []
        shuffled_response_list = []

        for flyId in range(8):
            mask = Pooled_dataframe['FlyId'] == flyId
            mask_sub = Pooled_dataframe['Sub-region'] == subregion
            subject_response_list.append(Pooled_dataframe[mask & mask_sub]['Response'].values.tolist())
            shuffled_response_list.append(np.random.permutation(Pooled_dataframe[mask & mask_sub]['Response'].values.tolist()))
        subject_corr = np.corrcoef(subject_response_list,rowvar=True)
        # sns.clustermap(data=subject_corr, vmin=-1, vmax=1, cmap='bwr')
        # plt.show()
        # shuffled_corr = np.corrcoef(shuffled_response_list,rowvar=True)
        # sns.clustermap(data=shuffled_corr, vmin=-1, vmax=1, cmap='bwr')
        # plt.show()

        subject_corr_list = []
        for i in range(len(subject_corr)):
            for j in range(i, len(subject_corr)):
                if i==j:
                    continue
                subject_corr_list.append([subject_corr[i][j],'fly'])
                subject_corr_list.append([shuffled_corr[i][j],'shuffled model'])
        
        # sns.violinplot(data=Df(data=subject_corr_list, columns=['Inter-fly correlation','Model']), x='Model', y='Inter-fly correlation')
        # plt.title(subregion)
        # plt.show()

    for odor in odor_list:
        pooled_odor_result = []
        for s1id, subregion_1 in enumerate(subregion_list):
            odor_result = []
            for s2id, subregion_2 in enumerate(subregion_list):
                # Retrieve responses
                if s2id < s1id:
                    continue
                response_1 = subregion_response_dict[subregion_1].get(odor)
                response_2 = subregion_response_dict[subregion_2].get(odor)
                # plt.scatter(response_1,response)
                # Check if both responses are valid arrays
                if isinstance(response_1, np.ndarray) and isinstance(response_2, np.ndarray):
                    # Calculate correlation coefficient
                    correlation = np.corrcoef([response_1], [response_2])[0, 1]
                
                odor_result.append(correlation)
                shuffled_corr = np.corrcoef(np.random.permutation(response_1),np.random.permutation(response_2))
                # results.append([subregion_1,subregion_2,odor,'fly',correlation])
                # results.append([subregion_1,subregion_2,odor,'shuffled model',shuffled_corr])
            pooled_odor_result.append(odor_result)
        
        # results_df = pd.DataFrame(data=results, columns=['sub-region 1','sub-region 2','Odor','Model','Correlation'])
        # Pooled_results = results_df
        
        # Convert results to DataFrame for heatmap
        correlation_df = pd.DataFrame(pooled_odor_result, index=subregion_list, columns=subregion_list)
        sns.heatmap(correlation_df, annot=True, cmap="Reds", cbar=True, vmax=1,vmin=0, annot_kws={"fontsize": 22})
        plt.tick_params(axis='both', which='major', labelsize=26)  # Change the font size here
        plt.title(f"{odor}",fontdict={'fontsize':30})
        plt.savefig(f"{path}gamma_subregion_{odor}.png", dpi=500)
        plt.savefig(f"{path}gamma_subregion_{odor}.svg")
        plt.close()

    # Pool all responses across odors for each subregion
    pooled_responses = {subregion: [] for subregion in subregion_list}
    for subregion in subregion_list:
        for odor in odor_list:
            response = subregion_response_dict[subregion].get(odor)
            pooled_responses[subregion].extend(response)
    
    # Calculate correlation across pooled responses
    pooled_data_matrix = []
    for subregion in subregion_list:
        pooled_data_matrix.append(pooled_responses[subregion])
    plt.tick_params(axis='both', which='major', labelsize=26)  # Change the font size here

    plt.plot(pooled_responses[subregion_list[1]],pooled_responses[subregion_list[2]],'.')
    plt.xlabel(r'$\gamma$2', fontdict={'fontsize':30})
    plt.ylabel(r'$\gamma$3', fontdict={'fontsize':30})
    
    plt.show()

    pooled_data_matrix = np.array(pooled_data_matrix)
    
    # Compute correlation matrix across pooled responses
    pooled_correlation_matrix = np.corrcoef(pooled_data_matrix)
    
    # Convert pooled correlation matrix to DataFrame for heatmap
    pooled_correlation_df = pd.DataFrame(pooled_correlation_matrix, index=subregion_list, columns=subregion_list)
    sns.heatmap(pooled_correlation_df, annot=True, cmap="Reds", cbar=True,vmax=1,vmin=0, annot_kws={"fontsize": 22})
    plt.title("All Odors",fontdict={'fontsize':30})
    plt.tick_params(axis='both', which='major', labelsize=26)  # Change the font size here
    plt.savefig(f"{path}pooled_correlation_across_odors.png", dpi=500)
    plt.savefig(f"{path}pooled_correlation_across_odors.svg")
    plt.close()

def compare_activity_difference():
    marker_dict = {'EP':'o','IA':'v','OA':'^','PA':'H','BA':'D','ANS':'8','PYR':'s','EtOH':'p'}
    rd.seed(100)
    np.random.seed(100)

    path = 'Functional exp/'

    m_ap_file = 'm_ap_exp_collection.xlsx'
    data = pd.read_excel(f'{path}{m_ap_file}')
    odor_list = data['Odor'].unique().tolist()
    subject_list = [i+1 for i in range(10)]
    for odor in odor_list:
        mask = data['Odor'] == odor
        x = data[mask]['m'].values.tolist()
        y = data[mask]['ap'].values.tolist()
        correlation, p_value = pearsonr(x, y)
        print(odor)
        print("Correlation coefficient:", correlation)
        print("P-value:", p_value)
        plt.scatter(x,y,s=10,c=subject_list, cmap='rainbow')
        plt.text(x[1],y[1],f'r = {correlation}')
        plt.title(odor)
        # plt.show()
        plt.close()

    pooled_sub_diff = []
    for subject in subject_list:
        mask = data['exp_id'] == subject
        x = data[mask]['m'].values
        y = data[mask]['ap'].values
        pooled_sub_diff.append((x-y).tolist())
    corr = np.corrcoef(np.array(pooled_sub_diff),rowvar=True)
    sns.clustermap(data=corr,cmap='bwr',vmax=1,vmin=-1, method='complete')
    plt.title("Subject Diff. m vs. ap")
    plt.show()
    ###########
    real_corr = []
    for i in range(len(subject_list)):
        for j in range(i, len(subject_list)):
            if i==j:
                continue
            real_corr.append(corr[i][j])

    pooled_sub_diff_shuffled = []
    shuffled_corr = []
    for t in range(30):
        for subject in subject_list:
            mask = data['exp_id'] == subject
            x =  np.random.permutation(data[mask]['m'].values)
            y = np.random.permutation(data[mask]['ap'].values)
            pooled_sub_diff_shuffled.append((x-y).tolist())
        corr = np.corrcoef(np.array(pooled_sub_diff_shuffled),rowvar=True)
        for i in range(len(subject_list)):
            for j in range(i, len(subject_list)):
                if i==j:
                    continue
                shuffled_corr.append(corr[i][j])

    results = []
    for i in range(len(real_corr)):
        results.append(['fly',real_corr[i],'m-ap'])
    for i in range(len(shuffled_corr)):
        results.append(['shuffled model', shuffled_corr[i],'m-ap'])
    results_df = pd.DataFrame(data=results, columns=['Model','Correlation','Inter-class'])
    Pooled_results = results_df
    model1 = 'fly'
    model2 = 'shuffled model'
    data1 = results_df[results_df["Model"] == model1]["Correlation"]
    data2 = results_df[results_df["Model"] == model2]["Correlation"]
    stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
    print('m-ap: fly, shuffled',p_value)

    b_g_file = 'b_g_exp_collection.xlsx'
    data = pd.read_excel(f'{path}{b_g_file}')
    odor_list = data['Odor'].unique().tolist()
    pooled_sub_diff = []
    for subject in subject_list:
        mask = data['exp_id'] == subject
        x = data[mask]['b'].values
        y = data[mask]['g'].values
        pooled_sub_diff.append((x-y).tolist())
    corr = np.corrcoef(np.array(pooled_sub_diff),rowvar=True)
    sns.clustermap(data=corr,cmap='bwr',vmax=1,vmin=-1, method='complete')
    plt.title("Subject Diff. b vs. g")
    plt.show()
    ###########
    real_corr = []
    for i in range(len(subject_list)):
        for j in range(i, len(subject_list)):
            if i==j:
                continue
            real_corr.append(corr[i][j])
    pooled_sub_diff_shuffled = []
    shuffled_corr = []
    for t in range(30):
        for subject in subject_list:
            mask = data['exp_id'] == subject
            x =  np.random.permutation(data[mask]['b'].values)
            y = np.random.permutation(data[mask]['g'].values)
            pooled_sub_diff_shuffled.append((x-y).tolist())
        corr = np.corrcoef(np.array(pooled_sub_diff_shuffled),rowvar=True)
        for i in range(len(subject_list)):
            for j in range(i, len(subject_list)):
                if i==j:
                    continue
                shuffled_corr.append(corr[i][j])


    results = []
    for i in range(len(real_corr)):
        results.append(['fly',real_corr[i],'b-g'])
    for i in range(len(shuffled_corr)):
        results.append(['shuffled model', shuffled_corr[i],'b-g'])
    results_df = pd.DataFrame(data=results, columns=['Model','Correlation','Inter-class'])
    Pooled_results = pd.concat([Pooled_results,results_df])
    model1 = 'fly'
    model2 = 'shuffled model'
    data1 = results_df[results_df["Model"] == model1]["Correlation"]
    data2 = results_df[results_df["Model"] == model2]["Correlation"]
    stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
    print('b-g, fly vs shuffled',p_value)
    sns.violinplot(data=Pooled_results, x='Inter-class',y='Correlation', hue='Model', cut=1, bw=0.3)

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)  # X-axis
    ax.spines['left'].set_linewidth(1.5)  # Y-axis
    ax.spines['top'].set_linewidth(1.5)  # X-axis
    ax.spines['right'].set_linewidth(1.5)  # Y-axis
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(-1.2,1.2)
    plt.ylabel("Correlation of diff between b and g",fontdict={'fontsize': 20})
    plt.xlabel("")
    plt.show()

    sim_type_list = ['FlyEM','shuffled']
    for sim_type in sim_type_list:
        file_list = [f'b_g_sim_collection_{sim_type}.xlsx', 'b_g_exp_collection.xlsx',
                    f'm_ap_sim_collection_{sim_type}.xlsx', 'm_ap_exp_collection.xlsx']
        result_collection = []
        for file in file_list:
            data = pd.read_excel(f"{path}{file}")
            if 'sim' in file:
                exp_type = 'Sim'
            else:
                exp_type = 'Exp'
            if 'b_g' in file:
                c1 = 'b'
                c2 = 'g'
                t1 = c1
                t2 = c2
            else:
                c1 = 'm'
                c2 = 'ap'
                t1 = "b'-m"
                t2 = "b'-ap"
            for odor, class1, class2, subject_id in zip(data['Odor'], data[c1], data[c2], data['exp_id']):
                result_collection.append([odor, t1, class1, exp_type,subject_id])
                result_collection.append([odor, t2, class2, exp_type,subject_id])
        data = pd.DataFrame(data=result_collection, columns=['Odor', 'KC', 'dF/F', 'Approach','subject_id'])
        data['Class_approach'] = [f"{i} {j}" for i, j in zip(data['KC'], data['Approach'])]
        data.to_excel(f"{path}Final_function_summary_{sim_type}.xlsx")
        pooled_observed_result = []
        pooled_predicted_result = []
        pooled_marker_list = []
        pooled_fillstyle_list = []
        pooled_color_list = []

        for KC_class in ['major', 'minor']:
            if KC_class == 'major':
                class_list = ['b', 'g']
                odor_num = 4
                color_list = ['black','gray']
                fill_style = 'full'
                color_marker = 'black'
            else:
                class_list = ["b'-m","b'-ap"]
                odor_num = 8
                color_list = [(0,0.502,0.502),(0.502,0,0)]
                fill_style = 'full'
                color_marker = 'orange'
            fig, axes = plt.subplots(2, 1, figsize=(2 * odor_num, 14))
            ax = axes[0]
            mask = data['Class_approach'].str.contains("Exp")
            mask_1 = data['KC'] == class_list[0]
            mask_2 = data['KC'] == class_list[1]
            print(data[mask & (mask_1 | mask_2)])
            plt.sca(ax)
            filtered_data = data[~mask & (mask_1 | mask_2)]
            filtered_odor_list = filtered_data['Odor'].unique().tolist()
            for odorId in range(len(filtered_odor_list)):
                mask_odor = data['Odor'] == filtered_odor_list[odorId]
                y1 = data[mask & mask_1 & mask_odor]['dF/F'].values.tolist()
                y2 = data[mask & mask_2 & mask_odor]['dF/F'].values.tolist()
                x1 = [odorId*3+0 for _ in range(len(y1))]
                x2 = [odorId*3+1 for _ in range(len(y2))]
                mean_1 = np.mean(y1)
                mean_2 = np.mean(y2)
                std_1 = np.std(y1)
                std_2 = np.std(y2)
                plt.bar([x1[0], x2[0]], [mean_1, mean_2],yerr=[std_1,std_2], color=color_list, width=1)
                for exp_id in range(len(x1)):
                    plt.plot([x1[exp_id],x2[exp_id]],[y1[exp_id], y2[exp_id]],
                    color='lightgray', alpha=0.7, linestyle='-', marker='o', markersize=5,
                    )
                pooled_observed_result.append(mean_1-mean_2)
                pooled_marker_list.append(marker_dict[filtered_odor_list[odorId]])
                pooled_fillstyle_list.append(fill_style)
                pooled_color_list.append(color_marker)

            plt.xticks([i*3+1 for i in range(len(filtered_odor_list))],filtered_odor_list, fontsize=36)
            plt.xlim((-1,len(filtered_odor_list)*3-1))

            plt.ylabel("dF/F", fontdict={'fontsize': 30})
            plt.yticks([0, 30, 60], fontsize=36)
            plt.ylim((-0.5, 70))
            ax.spines['bottom'].set_linewidth(3)  # X-axis
            ax.spines['left'].set_linewidth(3)  # Y-axis
            ax.spines['top'].set_linewidth(3)  # X-axis
            ax.spines['right'].set_linewidth(3)  # Y-axis

            ax = axes[1]
            plt.sca(ax)
            mask = data['Class_approach'].str.contains("Sim")
            filtered_data = data[mask & (mask_1 | mask_2)]
            filtered_odor_list = filtered_data['Odor'].unique().tolist()
            for odorId in range(len(filtered_odor_list)):
                mask_odor = data['Odor'] == filtered_odor_list[odorId]
                y1 = data[mask & mask_1 & mask_odor]['dF/F'].values.tolist()
                y2 = data[mask & mask_2 & mask_odor]['dF/F'].values.tolist()
                x1 = [odorId*3+0 for _ in range(len(y1))]
                x2 = [odorId*3+1 for _ in range(len(y2))]
                mean_1 = np.mean(y1)
                mean_2 = np.mean(y2)
                plt.bar([x1[0], x2[0]], [mean_1, mean_2], color=color_list, width=1)
                pooled_predicted_result.append(mean_1-mean_2)
            ax = plt.gca()
            plt.ylabel("dF/F", fontdict={'fontsize': 30})
            plt.yticks([0, 30, 60], fontsize=36)
            plt.ylim((-0.5, 70))
            ax.spines['bottom'].set_linewidth(3)  # X-axis
            ax.spines['left'].set_linewidth(3)  # Y-axis
            ax.spines['top'].set_linewidth(3)  # X-axis
            ax.spines['right'].set_linewidth(3)  # Y-axis
            plt.xticks([i*3+1 for i in range(len(filtered_odor_list))],filtered_odor_list, fontsize=36)
            plt.xlim((-1,len(filtered_odor_list)*3-1))
            plt.savefig(f"{path}Fig3_{KC_class}_function_{sim_type}.png", dpi=500)
            plt.savefig(f"{path}Fig3_{KC_class}_function_{sim_type}.svg", format='svg')
            plt.close()
        
        fig, ax = plt.subplots()
        for i in range(len(pooled_observed_result)):
            if KC_class == 'minor':        
                plt.plot(pooled_predicted_result[i],pooled_observed_result[i],marker=pooled_marker_list[i],fillstyle=pooled_fillstyle_list[i], 
                        color=pooled_color_list[i], markersize=15)
            else:
                plt.plot(pooled_predicted_result[i],pooled_observed_result[i],marker=pooled_marker_list[i],fillstyle=pooled_fillstyle_list[i],
                        color=pooled_color_list[i], markersize=15)
        xlim = plt.xlim()
        for odor in filtered_odor_list:
            plt.plot([10000],[0],marker=marker_dict[odor],label=odor, markersize=10, 
                    color='black', fillstyle='none', linewidth=0,
                    )
        plt.xlim(xlim)
        slope, intercept, r_value, p_value, std_err = stats.linregress(pooled_predicted_result, pooled_observed_result)
        x_line = np.array([min(pooled_predicted_result), max(pooled_predicted_result)])
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, linestyle='dashed', color='black')

        plt.ylabel("Diff. in Exp.", fontdict={'fontsize': 30})
        plt.xlabel("Diff. in Pred.", fontdict={'fontsize': 30})
        plt.yticks(fontsize=24)
        ax.spines['bottom'].set_linewidth(3)  # X-axis
        ax.spines['left'].set_linewidth(3)  # Y-axis
        ax.spines['top'].set_linewidth(3)  # X-axis
        ax.spines['right'].set_linewidth(3)  # Y-axis
        plt.xticks(fontsize=24)
        plt.legend()    
        plt.savefig(f"{path}Diff between Pred_Exp_{sim_type}.svg")
        plt.show()
        plt.close()
