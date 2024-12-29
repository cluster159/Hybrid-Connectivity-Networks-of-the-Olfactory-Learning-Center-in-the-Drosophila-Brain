import pandas as pd
from pandas import DataFrame
import numpy as np
import re
import pickle
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import random as rd
from sklearn.cluster import DBSCAN
# eps = 400
# MinPts = 8
eps = 200
MinPts = 3

colors = [(0.5,0.5,0.5)]

for i in range(1000):
    r,g,b=rd.random(),rd.random(),rd.random()
    pr,pg,pb=colors[-1]
    color_d = ((r-pr)**2+(b-pb)**2+(g-pg)**2)**0.5
    color_abs=r+g+b
    while(color_d<0.4 or color_abs<0.5):
        r, g, b = rd.random(), rd.random(), rd.random()
        color_d = ((r-pr)**2+(b-pb)**2+(g-pg)**2)**0.5
        color_abs=r+g+b
    colors.append((r,g,b))


# path_skeleton="em_skeletons_bridge_so_v1_1/"
path_recored="PN_KC_bouton_claw_information_20230618/"
# ff=open("MB_roi.pickle","rb")
# MB=pickle.load(ff)
# finished_PN=os.listdir(path_skeleton)

if not os.path.isdir(path_recored):
    os.mkdir(path_recored)

pattern = re.compile(r'\D+(\d+)\D+(\d+)\D+(\d+)\D+(\d+)\D+(\d+)\D+(\d+)')

def get_bouton_parameter(claw_xyz):
    mean_radius_claw = {}
    for claw_id in range(len(claw_xyz)):
        center = [0, 0, 0]
        count = 0
        for c_xyz in claw_xyz[claw_id]:
            center = [center[0] + c_xyz[0], center[1] + c_xyz[1], center[2] + c_xyz[2]]
        count = len(claw_xyz[claw_id])
        if count < 4:
            print("bouton deletion")
        else:
            center_xyz = [center[0] / count, center[1] / count, center[2] / count]
            sigma = 0
            all_r = 0
            for c_xyz in claw_xyz[claw_id]:
                sigma = (c_xyz[0] - center_xyz[0]) ** 2 + (c_xyz[1] - center_xyz[1]) ** 2 + (
                            c_xyz[2] - center_xyz[2]) ** 2
                all_r = ((c_xyz[0] - center_xyz[0]) ** 2 + (c_xyz[1] - center_xyz[1]) ** 2 + (
                            c_xyz[2] - center_xyz[2]) ** 2) ** 0.5 + all_r
            mean_radius = all_r / count
            d_radius = (sigma / (count - 1)) ** 0.5 * 2
            mean_radius_claw[claw_id] = [center_xyz,mean_radius,d_radius]
    return mean_radius_claw


def merge_area_bouton(old_claw_xyz):
    mean_radius_claw = {}
    claw_xyz={}
    for claw_id in old_claw_xyz:
        #####split##########
        bouton_xyz = np.array(old_claw_xyz[claw_id])
        db = DBSCAN(eps, MinPts).fit(bouton_xyz)
        labels = db.labels_
        labels = np.array(labels)
        for i in range(len(bouton_xyz)):
            if labels[i] != -1:
                # print(labels)
                if str(claw_id)+ "_"+str(labels[i]) not in claw_xyz:
                    claw_xyz[str(claw_id)+ "_"+str(labels[i])] = [bouton_xyz[i]]
                else:
                    claw_xyz[str(claw_id)+ "_"+str(labels[i])].append(bouton_xyz[i])
            else:
                print("not enough points")
            #     claw_xyz[str(claw_id)+"_0"]=old_claw_xyz[claw_id]

        ############
    for claw_id in claw_xyz:

        center = [0, 0, 0]
        count = 0
        for c_xyz in claw_xyz[claw_id]:
            center = [center[0] + c_xyz[0], center[1] + c_xyz[1], center[2] + c_xyz[2]]
        count = len(claw_xyz[claw_id])
        if count < 4:
            print("bouton deletion")
        else:
            center_xyz = [center[0] / count, center[1] / count, center[2] / count]
            sigma = 0
            all_r = 0
            for c_xyz in claw_xyz[claw_id]:
                sigma = (c_xyz[0] - center_xyz[0]) ** 2 + (c_xyz[1] - center_xyz[1]) ** 2 + (
                            c_xyz[2] - center_xyz[2]) ** 2
                all_r = ((c_xyz[0] - center_xyz[0]) ** 2 + (c_xyz[1] - center_xyz[1]) ** 2 + (
                            c_xyz[2] - center_xyz[2]) ** 2) ** 0.5 + all_r
            mean_radius = all_r / count
            d_radius = (sigma / (count - 1)) ** 0.5 * 2
            mean_radius_claw[claw_id] = [center_xyz, d_radius, mean_radius]

    # print(mean_radius_claw)
    # print(claw_xyz)
    new_claw_xyz = []
    new_claw_collection = {}
    # new_merge_label={}
    # new_merge_xyz=[]
    count = 0
    new_claw_label = {}
    for claw_id in mean_radius_claw:
        new_claw_label[claw_id] = count
        count = count + 1
    for claw_id in mean_radius_claw:
        for claw_id_2 in mean_radius_claw:
            if new_claw_label[claw_id] == new_claw_label[claw_id_2]:
                # print("skip1", claw_id, claw_id_2)
                continue
            dis_threshold = (mean_radius_claw[claw_id][2] + mean_radius_claw[claw_id_2][2]) / 2
            dis_std = (mean_radius_claw[claw_id][1] + mean_radius_claw[claw_id_2][1]) / 2
            x1, y1, z1 = mean_radius_claw[claw_id][0]
            x2, y2, z2 = mean_radius_claw[claw_id_2][0]
            center_dis = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5
            # print(dis_threshold, dis_std * 2, center_dis)
            # if center_dis < dis_threshold + 3 * dis_std and center_dis < 2000:
            if center_dis < 350:

                # print()
                ##merge!!
                # print("merge", claw_id, claw_id_2)
                label = min(new_claw_label[claw_id], new_claw_label[claw_id_2])
                new_claw_label[claw_id] = label
                new_claw_label[claw_id_2] = label
                # new_merge_label
            # else:
            #     print("not merge", claw_id, claw_id_2)
                # new_claw_collection[claw_id] = count
                # count = count + 1
                # new_claw_collection[claw_id_2] = count
                # count = count + 1
                # new_claw_xyz.append(claw_xyz[claw_id])
                # new_claw_xyz.append(claw_xyz[claw_id_2])
    for claw_id in new_claw_label:
        if new_claw_label[claw_id] not in new_claw_collection:
            new_claw_collection[new_claw_label[claw_id]] = []
        new_claw_collection[new_claw_label[claw_id]].append(claw_id)
    # print(new_claw_label)
    # print(new_claw_collection)
    for label in new_claw_collection:
        xyzs = []
        for claw_id in new_claw_collection[label]:
            xyzs = list(claw_xyz[claw_id]) + xyzs
        new_claw_xyz.append(xyzs)
    # print(claw_xyz)
    # print(new_claw_xyz)

    return new_claw_xyz,new_claw_collection,new_claw_label


root = 'hemibrain_data/'
PN_KC_synapse_df_ab = pd.read_excel(f"{root}PN_to_KCab_synapse.xlsx")
PN_KC_synapse_df_apbp = pd.read_excel(f"{root}PN_to_KCa'b'_synapse.xlsx")
PN_KC_synapse_df_g = pd.read_excel(f"{root}PN_to_KCg_synapse.xlsx")
pooled = pd.merge(pd.merge(PN_KC_synapse_df_ab,PN_KC_synapse_df_g,how='outer'),PN_KC_synapse_df_apbp,how='outer')
PNid_synapse_dict = {}
KCid_synapse_dict = {}
PNids = pooled['up.bodyId'].values.tolist()
KCids = pooled['down.bodyId'].values.tolist()
xs,ys,zs = pooled['up_syn_coordinate_x'].values.tolist(),pooled['up_syn_coordinate_y'].values.tolist(),pooled['up_syn_coordinate_z'].values.tolist()

for i in range(len(PNids)):
    PNid = PNids[i]
    KCid = KCids[i]
    if PNid not in PNid_synapse_dict:
        PNid_synapse_dict[PNid] = {}
    if KCid not in PNid_synapse_dict[PNid]:
        PNid_synapse_dict[PNid][KCid] = []
    if KCid not in KCid_synapse_dict:
        KCid_synapse_dict[KCid] = {}
    if PNid not in KCid_synapse_dict[KCid]:
        KCid_synapse_dict[KCid][PNid] = []
    PNid_synapse_dict[PNid][KCid].append([xs[i], ys[i], zs[i]])
    KCid_synapse_dict[KCid][PNid].append([xs[i], ys[i], zs[i]])

PN_KC_synapse_dict=PNid_synapse_dict
KC_PN_synapse_dict=KCid_synapse_dict

for i,KC in enumerate(KC_PN_synapse_dict):
    print("KC",i)
    new_bouton_xyz, new_bouton_collection, new_bouton_label = merge_area_bouton(KC_PN_synapse_dict[KC])
    bouton_parameters=get_bouton_parameter(new_bouton_xyz)
    with open(f"{path_recored}{KC}_claw.txt","wt")as ff:
        for bouton in bouton_parameters:
            x,y,z=bouton_parameters[bouton][0]
            ff.writelines(str(x)+" "+str(y)+" "+str(z)+" "+str(bouton_parameters[bouton][1])+" "+str(bouton_parameters[bouton][2])+"\n")

for i,PN in enumerate(PN_KC_synapse_dict):
    print("PN",i)
    new_bouton_xyz, new_bouton_collection, new_bouton_label = merge_area_bouton(PN_KC_synapse_dict[PN])
    bouton_parameters=get_bouton_parameter(new_bouton_xyz)
    with open(f"{path_recored}{PN}_bouton.txt","wt")as ff:
        for bouton in bouton_parameters:
            x,y,z=bouton_parameters[bouton][0]
            ff.writelines(str(x)+" "+str(y)+" "+str(z)+" "+str(bouton_parameters[bouton][1])+" "+str(bouton_parameters[bouton][2])+"\n")

