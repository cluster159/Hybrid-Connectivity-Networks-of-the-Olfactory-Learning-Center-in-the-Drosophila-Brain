from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import copy
from scipy.stats import linregress
from scipy import stats
from matplotlib import rcParams
from scipy.stats import pearsonr
from scipy.stats import mannwhitneyu
import random as rd
rcParams['font.family'] = 'Arial'
marker_dict = {'EP':'o','IA':'v','OA':'^','PA':'H','BA':'D','ANS':'8','PYR':'s','EtOH':'p'}

rd.seed(100)
np.random.seed(100)

path = 'Functional exp_final/'

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


# print("##############################")
# individual_m_list = []
# individual_ap_list = []
# shuffled_m_list = []
# shuffled_ap_list = []
# for flyId in data['exp_id'].unique():
#     mask = data['exp_id'] == flyId
#     individual_m_list.append(data[mask]['m'].values.tolist())
#     individual_ap_list.append(data[mask]['ap'].values.tolist())
#     shuffled_m_list.append(np.random.permutation(data[mask]['m'].values).tolist())
#     shuffled_ap_list.append(np.random.permutation(data[mask]['ap'].values).tolist())

# corr_m = np.corrcoef(individual_m_list,rowvar=True)
# sns.clustermap(data=corr_m,vmin=-1,vmax=1,cmap='bwr')
# plt.title("m")
# plt.show()

# shuffled_m = np.corrcoef(shuffled_m_list,rowvar=True)
# sns.clustermap(data=shuffled_m,vmin=-1,vmax=1,cmap='bwr')
# plt.title("shuffled_m")
# plt.show()

# corr_ap = np.corrcoef(individual_ap_list,rowvar=True)
# sns.clustermap(data=corr_ap,vmin=-1,vmax=1,cmap='bwr')
# plt.title("ap")
# plt.show()

# shuffled_ap = np.corrcoef(shuffled_ap_list,rowvar=True)
# sns.clustermap(data=shuffled_ap,vmin=-1,vmax=1,cmap='bwr')
# plt.title("shuffled_ap")
# plt.show()

# pooled_individual_corr = []
# for i in range(len(corr_m)):
#     for j in range(i, len(corr_m)):
#         if i == j :
#             continue
#         pooled_individual_corr.append(['m','fly',corr_m[i][j]])
#         pooled_individual_corr.append(['ap','fly',corr_ap[i][j]])
#         pooled_individual_corr.append(['m','shuffled model',shuffled_m[i][j]])
#         pooled_individual_corr.append(['ap','shuffled model',shuffled_ap[i][j]])

# pooled_individual_corr_df = pd.DataFrame(data=pooled_individual_corr,columns=['KC class','Model','Inter-fly correlation'])
# sns.violinplot(data=pooled_individual_corr_df,x='KC class',y='Inter-fly correlation',hue='Model')
# plt.show()

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

# sns.violinplot(data=results_df, x='Model',y='Correlation')
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(1.5)  # X-axis
# ax.spines['left'].set_linewidth(1.5)  # Y-axis
# ax.spines['top'].set_linewidth(1.5)  # X-axis
# ax.spines['right'].set_linewidth(1.5)  # Y-axis
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.ylabel("Correlation of diff between ap and m",fontdict={'fontsize': 20})
# plt.xlabel("")
# plt.title("Across fly")
# plt.text(plt.xlim()[0],plt.ylim()[0],f'p={p_value}')
# plt.show()

# for odor in odor_list:
#     mask = data['Odor'] == odor
#     mask_sub = data['exp_id'] == 1
#     x = data[mask]['m'].values.tolist()
#     y = data[mask]['ap'].values.tolist()
#     correlation, p_value = pearsonr(x, y)
#     print("Correlation coefficient:", correlation)
#     print("P-value:", p_value)
#     plt.scatter(x,y,s=10,c=subject_list[:-1], cmap='rainbow')
#     plt.text(10,10,f'r = {correlation}')
#     plt.title(odor)
#     # plt.show()
#     plt.close()


b_g_file = 'b_g_exp_collection.xlsx'
data = pd.read_excel(f'{path}{b_g_file}')
odor_list = data['Odor'].unique().tolist()
# for odor in odor_list:
#     mask = data['Odor'] == odor
#     x = data[mask]['b'].values.tolist()
#     y = data[mask]['g'].values.tolist()
#     correlation, p_value = pearsonr(x, y)
#     print("Correlation coefficient:", correlation)
#     print("P-value:", p_value)
#     plt.scatter(x,y,s=10,c=subject_list, cmap='rainbow')
#     plt.text(10,10,f'r = {correlation}')
#     plt.title(odor)
#     # plt.show()
#     plt.close()
# print("##############################")
# individual_g_list = []
# individual_b_list = []
# shuffled_g_list = []
# shuffled_b_list = []
# for flyId in data['exp_id'].unique():
#     mask = data['exp_id'] == flyId
#     individual_g_list.append(data[mask]['g'].values.tolist())
#     individual_b_list.append(data[mask]['b'].values.tolist())
#     print(flyId)
#     print(individual_g_list[-1])
#     shuffled_g_list.append(np.random.permutation(data[mask]['g'].values).tolist())
#     shuffled_b_list.append(np.random.permutation(data[mask]['b'].values).tolist())

# corr_g = np.corrcoef(individual_g_list, rowvar=True)
# sns.clustermap(data=corr_g, vmin=-1, vmax=1, cmap='bwr')
# plt.title("g")
# plt.show()

# shuffled_g = np.corrcoef(shuffled_g_list, rowvar=True)
# sns.clustermap(data=shuffled_g, vmin=-1, vmax=1, cmap='bwr')
# plt.title("shuffled_g")
# plt.show()

# corr_b = np.corrcoef(individual_b_list, rowvar=True)
# sns.clustermap(data=corr_b, vmin=-1, vmax=1, cmap='bwr')
# plt.title("b")
# plt.show()

# shuffled_b = np.corrcoef(shuffled_b_list, rowvar=True)
# sns.clustermap(data=shuffled_b, vmin=-1, vmax=1, cmap='bwr')
# plt.title("shuffled_b")
# plt.show()

# pooled_individual_corr = []
# for i in range(len(corr_g)):
#     for j in range(i, len(corr_g)):
#         if i == j:
#             continue
#         pooled_individual_corr.append(['g', 'fly', corr_g[i][j]])
#         pooled_individual_corr.append(['b', 'fly', corr_b[i][j]])
#         pooled_individual_corr.append(['g', 'shuffled model', shuffled_g[i][j]])
#         pooled_individual_corr.append(['b', 'shuffled model', shuffled_b[i][j]])

# pooled_individual_corr_df = pd.DataFrame(data=pooled_individual_corr, columns=['KC class', 'Model', 'Inter-fly correlation'])
# sns.violinplot(data=pooled_individual_corr_df, x='KC class', y='Inter-fly correlation', hue='Model')
# plt.show()
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

# sns.violinplot(data=results_df, x='Model',y='Correlation')
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(1.5)  # X-axis
# ax.spines['left'].set_linewidth(1.5)  # Y-axis
# ax.spines['top'].set_linewidth(1.5)  # X-axis
# ax.spines['right'].set_linewidth(1.5)  # Y-axis
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.ylabel("Correlation of diff between b and g",fontdict={'fontsize': 20})
# plt.xlabel("")
# plt.title("Across fly")
# plt.text(plt.xlim()[0],plt.ylim()[0],f'p={p_value}')
# plt.show()


# sns.boxplot(data=Pooled_results, x='Inter-class',y='Correlation', hue='Model')
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
# plt.title("Across fly")
# plt.text(plt.xlim()[0],plt.ylim()[0],f'p={p_value}')
plt.show()


input()


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
    plt.close()

    # result = []
    # result_dict = {}
    # for odor in data['Odor'].unique().tolist():
    #     for KC in data['KC'].unique().tolist():
    #         for approach in data['Approach'].unique().tolist():
    #             mask_odor = data['Odor'] == odor
    #             mask_KC = data['KC'] == KC
    #             mask_approach = data['Approach'] == approach
    #             tmp = data[mask_odor & mask_KC & mask_approach]
    #             if len(tmp) == 0:
    #                 continue
    #             mean = tmp['dF/F'].mean()
    #             result.append([odor, KC, approach, mean])
    #             result_dict[(odor, KC, approach)] = mean

    # # approach_list = ['Exp', 'Sim_observed', 'Sim_random']
    # approach_list = ['Exp', 'Sim']

    # real_result = []
    # shuffled_result = []
    # KC_class_dict = {'b': 'major', 'g': 'major', "b'-ap": 'minor', "b'-m": 'minor'}
    # marker_color_dict = {}
    # for odor in data['Odor'].unique().tolist():
    #     for KC in data['KC'].unique().tolist():
    #         try:
    #             real_result.append(
    #                 [result_dict[(odor, KC, approach_list[0])], result_dict[(odor, KC, approach_list[1])]])
    #         except:
    #             continue
    #         # shuffled_result.append([result_dict[(odor,KC,approach_list[0])],result_dict[(odor,KC,approach_list[2])]])
    #         if KC_class_dict[KC] == 'major':
    #             geo = '^'
    #         else:
    #             geo = 'o'
    #         if (geo, 'black') not in marker_color_dict:
    #             marker_color_dict[(geo, 'black')] = []
    #             # marker_color_dict[(geo, 'red')] = []

    #         marker_color_dict[(geo, 'black')].append(
    #             [result_dict[(odor, KC, approach_list[0])], result_dict[(odor, KC, approach_list[1])]])
    #         # marker_color_dict[(geo, 'red')].append(
    #         #     [result_dict[(odor, KC, approach_list[0])], result_dict[(odor, KC, approach_list[2])]])

    # real_result = []
    # for geo, color in marker_color_dict:
    #     if color == 'black':
    #         real_result += marker_color_dict[(geo, color)]
    #     marker_color_dict[(geo, color)] = np.array(marker_color_dict[(geo, color)])
    #     plt.scatter(marker_color_dict[(geo, color)][:, 0], marker_color_dict[(geo, color)][:, 1], marker=geo,
    #                 color=color)

    # real_result = np.array(real_result)
    # print(real_result)
    # x = real_result[:, 0]
    # y = real_result[:, 1]
    # # Calculate regression line
    # slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # print(r_value, r_value ** 2, p_value)
    # x_line = np.array([min(x), max(x)])
    # y_line = slope * x_line + intercept
    # plt.plot(x_line, y_line, color='black')
    # plt.show()
