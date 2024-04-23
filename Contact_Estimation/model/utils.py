import numpy as np
import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import torch
from datasets import ContactForceDataSet
from sklearn.metrics import average_precision_score
from sklearn.decomposition import PCA
# from scipy import sparse
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def feature_dim_reduction_viaPCA(data):
    pca = PCA()
    Xt = pca.fit_transform(data)
    print("Explained_variance_ratio: ", pca.explained_variance_ratio_)
    print("Sum of all variance: ", np.sum(pca.explained_variance_ratio_))

    return Xt

def cal_adjacency_matrix(feats):
    # Assuming the feats are normalized
    feats = feats.astype('float16')
    # feats = sparse.csr_matrix(feats)
    sim = feats.dot(feats.transpose())
    return sim

def cal_mAP_loc_wise(hall_signals, heat_maps):
    num_contacts_y = (heat_maps > 0).sum(axis=1)
    single_probe_samples = (num_contacts_y == 1)
    inds = single_probe_samples.nonzero()
    hall_signals = hall_signals[inds]
    heat_maps = heat_maps[inds]

    # Downsampling amount of factors to reduce computational cost
    ds_factor  = max(1,round(len(hall_signals) / 80000))
    selected_inds = np.arange(0,len(hall_signals),ds_factor)
    hall_signals = hall_signals[selected_inds]
    heat_maps = heat_maps[selected_inds]

    print("PCA dimension reduction ... ")
    hall_signals = feature_dim_reduction_viaPCA(hall_signals)

    feats = hall_signals
    labels = np.multiply(np.argmax(heat_maps,axis=1), (np.sum(heat_maps, axis=1) > 0))
    sim = cal_adjacency_matrix(feats)
    num_instances = len(labels)
    aps = []
    for i in range(num_instances):
        y_true = labels == labels[i]
        y_true = y_true.astype('int')
        ap = average_precision_score(y_true, sim[i,:])
        aps.append(ap)
    aps = np.array(aps)
    heat_maps2d = taxel_projection(heat_maps)
    mAP_loc_wise = np.zeros(heat_maps2d.shape[1:])
    for j in range(mAP_loc_wise.shape[0]):
        for k in range(mAP_loc_wise.shape[1]):
            curr_inds = heat_maps2d[:,j,k].nonzero()
            curr_aps = aps[curr_inds]
            mAP_loc_wise[j,k] = curr_aps.mean()

    mAP = np.mean(aps)
    print("Feature mAP: ", mAP)
    return mAP, mAP_loc_wise

def data_preprocessing_real_test_non_contact(data, data_cfgs, i=0):
    force_inds = data_cfgs["force_inds"]
    hall_ind0s = data_cfgs["hall_ind0s"]
    hall_ind1s = data_cfgs["hall_ind1s"]

    hall_ind0 = hall_ind0s[i]
    hall_ind1 = hall_ind1s[i]
    force1 = np.array(data.iloc[:, force_inds[0]].dropna())
    force2 = np.array(data.iloc[:, force_inds[1]].dropna())
    # faces = np.array(data.iloc[:, face_ind].dropna())

    # Calibrate the data to all negative
    # force = np.array(force)
    # force_max = np.max(force)
    # force = force - force_max

    hall_signal = data.iloc[:, hall_ind0:(hall_ind1+1)].dropna()
    hall_signal = np.array(hall_signal)
    selected_force = force1
    selected_hall_signal = hall_signal
    # selected_face_inds = faces
    selected_heat_map = np.zeros((selected_force.shape[0],100))

    if "cal_zero_ref" in data_cfgs:
        if data_cfgs["cal_zero_ref"]:
            # zero_inds = np.logical_and(force1 < 0.1, force2 < 0.1).nonzero()
            zero_inds = np.logical_and(force1 < 0.5, force2 < 0.5).nonzero()
            zero_refs = selected_hall_signal[zero_inds]
            zero_refs_avg = np.mean(zero_refs, axis=0)
            for i in range(selected_hall_signal.shape[0]):
                selected_hall_signal[i] = selected_hall_signal[i] - zero_refs_avg
            print("zero_refs_avg: ", zero_refs_avg)

    # # Calibrate too small force to zero
    # for iii in range(selected_heat_map.shape[0]):
    #     cal_inds = (selected_heat_map.sum(axis=1) < 0.12).nonzero()
    #     selected_heat_map[cal_inds].fill(0)
    
    # Plot raw force data    
    # fig = plt.figure()
    # plt.plot(force)
    
    return selected_hall_signal, selected_heat_map

def data_preprocessing_real_test(data, data_cfgs, i=0):

    force_inds = data_cfgs["force_inds"]
    loc_inds = data_cfgs["loc_inds"]
    hall_ind0s = data_cfgs["hall_ind0s"]
    hall_ind1s = data_cfgs["hall_ind1s"]

    force_ind = force_inds[i]
    loc_ind = loc_inds[i]
    hall_ind0 = hall_ind0s[i]
    hall_ind1 = hall_ind1s[i]

    force = np.array(data.iloc[:, force_ind].dropna())
    # faces = np.array(data.iloc[:, face_ind].dropna())

    # Calibrate the data to all negative
    # force = np.array(force)
    # force_max = np.max(force)
    # force = force - force_max

    hall_signal = data.iloc[:, hall_ind0:(hall_ind1+1)].dropna()
    hall_signal = np.array(hall_signal)
    selected_force = force
    selected_hall_signal = hall_signal
    # selected_face_inds = faces
    selected_heat_map = np.zeros((selected_force.shape[0],101))
    for ii in range(selected_force.shape[0]):
        curr_force = selected_force[ii]
        # curr_force = curr_force * 9.8e-3
        # print(f"Sample {ii}", curr_force, selected_hall_signal[ii])
        # if curr_force < 0.1:
        if curr_force < 0.5:
            curr_force = 0
        else:
            curr_force = curr_force / len(loc_ind) # use the absolute force values for our application scenarios
        curr_force = curr_force * 9.8e-3
        for j in range(len(loc_ind)):
            curr_pos = data.iloc[ii, loc_ind[j]]
            if curr_pos in [0,101]:
                curr_ind = 0
            else:
                curr_ind = curr_pos
            selected_heat_map[ii,curr_ind] = curr_force
    selected_heat_map = selected_heat_map[:,1:]

    if "cal_zero_ref" in data_cfgs:
        if data_cfgs["cal_zero_ref"]:
            zero_inds = (selected_heat_map.sum(axis=1) == 0).nonzero()
            zero_refs = selected_hall_signal[zero_inds]
            zero_refs_avg = np.mean(zero_refs, axis=0)
            for i in range(selected_hall_signal.shape[0]):
                selected_hall_signal[i] = selected_hall_signal[i] - zero_refs_avg
            print("zero_refs_avg: ", zero_refs_avg)

    # # Calibrate too small force to zero
    # for iii in range(selected_heat_map.shape[0]):
    #     cal_inds = (selected_heat_map.sum(axis=1) < 0.12).nonzero()
    #     selected_heat_map[cal_inds].fill(0)
    
    # Plot raw force data    
    # fig = plt.figure()
    # plt.plot(force)
    
    return selected_hall_signal, selected_heat_map

def data_preprocessing(data, data_cfgs):
    force_ind = data_cfgs["force_ind"]
    face_ind = data_cfgs["face_ind"]
    loc_inds = data_cfgs["loc_inds"]
    hall_ind0 = data_cfgs["hall_ind0"]
    hall_ind1 = data_cfgs["hall_ind1"]
    case_ind = data_cfgs["case_ind"]
    faces = data_cfgs["faces"]
#     position = (data.iloc[:, 0].dropna()-1)*10 + (data.iloc[:, 1].dropna() % 10)
    force = np.array(data.iloc[:, force_ind].dropna())
    case_inds = np.array(data.iloc[:, case_ind].dropna())
    face_inds = np.array(data.iloc[:, face_ind].dropna())
    selected_inds = np.array([i for i in range(len(face_inds)) if face_inds[i] in faces])

    # Temporally reduce the amount of data
    if "ds_factor" in data_cfgs:
        ds_factor = data_cfgs["ds_factor"]
        selected_inds = np.arange(0,len(selected_inds),ds_factor)

    # Calibrate the data to all negative
    # force = np.array(force)
    force_max = np.max(force)
    force = force - force_max

    hall_signal = data.iloc[:, hall_ind0:hall_ind1].dropna()
    hall_signal = np.array(hall_signal)
    selected_force = force[selected_inds]
    selected_hall_signal = hall_signal[selected_inds]
    selected_face_inds = face_inds[selected_inds]
    selected_case_inds = case_inds[selected_inds]
    selected_heat_map = np.zeros((selected_force.shape[0],len(faces)*100+1))
    for i in range(selected_force.shape[0]):
        curr_face = selected_face_inds[i]
        curr_face = faces.index(curr_face)
        curr_force = abs(selected_force[i]) / len(loc_inds) # use the absolute force values for our application scenarios
        for j in range(len(loc_inds)):
            curr_pos = data.iloc[i, loc_inds[j]]
            if curr_pos in [0,101] or selected_case_inds[i] in [0,101]:
                curr_ind = 0
            else:
                curr_ind = 100 * curr_face + curr_pos
            selected_heat_map[i,curr_ind] = curr_force
    selected_heat_map = selected_heat_map[:,1:]

    if "cal_zero_ref" in data_cfgs:
        if data_cfgs["cal_zero_ref"]:
            zero_inds = (selected_heat_map.sum(axis=1) == 0).nonzero()
            zero_refs = selected_hall_signal[zero_inds]
            zero_refs_avg = np.mean(zero_refs, axis=0)
            for i in range(selected_hall_signal.shape[0]):
                selected_hall_signal[i] = selected_hall_signal[i] - zero_refs_avg
            print("zero_refs_avg: ", zero_refs_avg)
    
    return selected_hall_signal, selected_heat_map, selected_case_inds

def taxel_projection(data):
    out = data.copy()
    out = out.reshape((-1,10,10))
    data = out.copy()
    for i in range(out.shape[0]):
        out[i,1::2] = data[i,1::2,::-1]
    out = out.transpose((0,2,1))   
    return out

def location_wise_hall_signal_range(all_hall_signals, all_hall_signals_scaled, all_heat_maps):
    # Only evaluate those with a single probe
    num_contacts_y = (all_heat_maps > 0).sum(axis=1)
    single_probe_samples = (num_contacts_y == 1)
    inds = single_probe_samples.nonzero()
    all_hall_signals = all_hall_signals[inds]
    all_hall_signals_scaled = all_hall_signals_scaled[inds]
    all_heat_maps = all_heat_maps[inds]
    # Project to heatmaps
    proj_heat_maps = taxel_projection(all_heat_maps)

    # Calculate errors
    loc_wise_hall_max = np.zeros(proj_heat_maps.shape[1:]+all_hall_signals.shape[1:])
    loc_wise_hall_min = np.zeros(proj_heat_maps.shape[1:]+all_hall_signals.shape[1:])
    loc_wise_hall_range = np.zeros(proj_heat_maps.shape[1:]+all_hall_signals.shape[1:])
    loc_wise_hall_QHull_vol = np.zeros(proj_heat_maps.shape[1:])
    loc_wise_hall_scaled_QHull_vol = np.zeros(proj_heat_maps.shape[1:])
    for j in range(proj_heat_maps.shape[1]):
        for k in range(proj_heat_maps.shape[2]):
            curr_inds = proj_heat_maps[:,j,k].nonzero()
            if len(curr_inds[0]) > 0:
                curr_hall_signals = all_hall_signals[curr_inds]
                curr_hall_signals_scaled = all_hall_signals_scaled[curr_inds]
                loc_wise_hall_min[j,k] = curr_hall_signals.min(axis=0)
                loc_wise_hall_max[j,k] = curr_hall_signals.max(axis=0)
                loc_wise_hall_range[j,k] = loc_wise_hall_max[j,k] - loc_wise_hall_min[j,k]
                if len(curr_inds[0]) >= 10:
                    loc_wise_hall_QHull_vol[j,k] = ConvexHull(curr_hall_signals).volume
                    loc_wise_hall_scaled_QHull_vol[j,k] = ConvexHull(curr_hall_signals_scaled).volume

    return [loc_wise_hall_min, loc_wise_hall_max, loc_wise_hall_range, loc_wise_hall_QHull_vol, loc_wise_hall_scaled_QHull_vol]


def load_data_real_test(cfgs):
    all_data_cfgs = cfgs["dataset"]
    faces = cfgs["faces"]
    non_cfgs = cfgs["non_contact_info"]
    non_faces = non_cfgs["faces"]
    all_faces = faces + non_faces
    n_faces = len(all_faces)

    whether_non_contact = []
    test_data_all_faces = []
    y_mins = []
    input_sizes = []
    output_sizes = []
    force_magnitudes = []
    force_magnitudes_max = []
    force_magnitudes_min = []
    all_all_heat_maps = []
    all_all_hall_signals = []
    all_all_hall_signals_scaled = []
    for ii in range(n_faces):
        curr_face = all_faces[ii]
        if curr_face in faces:
            whether_non_contact.append(False)
        elif curr_face in non_faces:
            whether_non_contact.append(True)
        # Load all data files
        hall_signals, heat_maps = [], []
        dataset_names = []
        for i in range(len(all_data_cfgs)):
            data_cfgs = all_data_cfgs[i]
            data_path = data_cfgs["csv_path"]
            dataset_name = data_cfgs["name"]
            dataset_names.append(dataset_name)
            data = pd.read_csv(data_path)

            if whether_non_contact[ii]:
                hall_signal, heat_map = data_preprocessing_real_test_non_contact(data, non_cfgs, ii-len(faces))
            else:
                hall_signal, heat_map = data_preprocessing_real_test(data, data_cfgs, ii)
                # print(heat_map)

            hall_signals.append(hall_signal)
            heat_maps.append(heat_map)

        all_hall_signals = np.concatenate(hall_signals, axis=0)
        all_heat_maps = np.concatenate(heat_maps, axis=0)

        # Regularize data
        # Loading the scaler and dataset statistics
        result_path = cfgs["result"]["path"]
        face = all_faces[ii]
        scaler_fn = f"{result_path}/training_hist_face{face}_data_scalers.pkl"
        file = open(scaler_fn, 'rb')
        data = pickle.load(file)
        y_scaler = data[1]
        X_scaler = data[0]
        y_min = data[-1]
        
        all_heat_maps_scaled = y_scaler.transform(all_heat_maps)
        all_hall_signals_scaled = X_scaler.transform(all_hall_signals)
        force_magnitude_scaled = np.max(all_heat_maps_scaled,axis=1)
        # non_zero_inds = (np.sum(all_heat_maps_scaled,axis=1)>0).nonzero()
        force_magnitudes_min.append(np.min(np.where(all_heat_maps_scaled==0, all_heat_maps_scaled.max(), all_heat_maps_scaled), axis=1))
        force_magnitudes_max.append(np.max(np.where(all_heat_maps_scaled==0, all_heat_maps_scaled.min(), all_heat_maps_scaled), axis=1))

        # force_magnitudes_max.append(all_heat_maps_scaled.max(axis=1))
        # force_magnitudes_min.append(all_heat_maps_scaled.min(axis=1))
        # This should be min max, not just max
        # force_magnitude_scaled = np.sum(all_heat_maps_scaled,axis=1)
        # print("Max force magnitude_scaled: ", force_magnitude_scaled.max())

        all_data = ContactForceDataSet(all_hall_signals_scaled, all_heat_maps_scaled)
        print(f"Total samples loaded for Face{all_faces[ii]}: ", len(all_data))

        test_data_all_faces.append(all_data)
        input_sizes.append(all_hall_signals_scaled.shape[1])
        output_sizes.append(all_heat_maps_scaled.shape[1])
        y_mins.append(y_min)
        force_magnitudes.append(force_magnitude_scaled)
        all_all_heat_maps.append(all_heat_maps)
        all_all_hall_signals.append(all_hall_signals)
        all_all_hall_signals_scaled.append(all_hall_signals_scaled)

    all_hall_statistics = []
    selected_inds = []
    for ii in range(n_faces):
        curr_face = all_faces[ii]
        other_faces = all_faces[:ii] + all_faces[ii+1:]
        all_heat_maps = all_all_heat_maps[ii]
        all_hall_signals = all_all_hall_signals[ii]
        all_hall_signals_scaled = all_all_hall_signals_scaled[ii]
        
        # Filter out the samples not in the training range
        y_min = y_mins[ii]
        # print(y_min)
        lower_bound = y_min
        # upper_bound = 0.3
        upper_bound1 = 0.3
        upper_bound2 = 0.4
        upper_bound3 = 0.6
        upper_bound4 = 0.8
        upper_bound5 = 1.0

        # too_small_self_bool = np.logical_and(force_magnitudes[ii] < lower_bound, force_magnitudes[ii] > 0)
        # too_large_self_bool = (force_magnitudes[ii] > 1.0)
        too_small_self_bool = np.logical_and(force_magnitudes_min[ii] < lower_bound, force_magnitudes_min[ii] > 0)
        # too_small_self_bool = np.logical_and(force_magnitudes_min[ii] < lower_bound, force_magnitudes_max[ii] > 0)
        too_large_self_bool = (force_magnitudes_max[ii] > 1.0)
        out_of_range_inds = np.logical_or(too_small_self_bool, too_large_self_bool).nonzero()[0]

        # too_small_bool = None
        too_large_bool1 = None
        too_large_bool2 = None
        too_large_bool3 = None
        too_large_bool4 = None
        too_large_bool5 = None
        for jj in range(len(other_faces)):
            data_ind = all_faces.index(other_faces[jj])
            # print(force_magnitudes[data_ind])
            # if too_small_bool is None:
            #     too_small_bool = np.logical_and(force_magnitudes[data_ind] < lower_bound, force_magnitudes[data_ind] > 0)
            #     # print(too_small_bool)
            # else:
            #     tmp = np.logical_and(force_magnitudes[data_ind] < lower_bound, force_magnitudes[data_ind] > 0)
            #     too_small_bool = np.logical_or(too_small_bool, tmp)

            if too_large_bool1 is None:
                too_large_bool1 = (force_magnitudes[data_ind] > upper_bound1)
                too_large_bool2 = (force_magnitudes[data_ind] > upper_bound2)
                too_large_bool3 = (force_magnitudes[data_ind] > upper_bound3)
                too_large_bool4 = (force_magnitudes[data_ind] > upper_bound4)
                too_large_bool5 = (force_magnitudes[data_ind] > upper_bound5)
            else:
                too_large_bool1 = np.logical_or(too_large_bool1, force_magnitudes[data_ind] > upper_bound1)
                too_large_bool2 = np.logical_or(too_large_bool2, force_magnitudes[data_ind] > upper_bound2)
                too_large_bool3 = np.logical_or(too_large_bool3, force_magnitudes[data_ind] > upper_bound3)
                too_large_bool4 = np.logical_or(too_large_bool4, force_magnitudes[data_ind] > upper_bound4)
                too_large_bool5 = np.logical_or(too_large_bool5, force_magnitudes[data_ind] > upper_bound5)

        too_large_inds1 = too_large_bool1.nonzero()[0]
        too_large_inds2 = too_large_bool2.nonzero()[0]
        too_large_inds3 = too_large_bool3.nonzero()[0]
        too_large_inds4 = too_large_bool4.nonzero()[0]
        too_large_inds5 = too_large_bool5.nonzero()[0]
 
        selected_ind1 = []
        selected_ind2 = []
        selected_ind3 = []
        selected_ind4 = []
        selected_ind5 = []
        for ind in range(force_magnitudes[data_ind].shape[0]):
            if ind not in out_of_range_inds:
                if ind not in too_large_inds1:
                    selected_ind1.append(ind)
                if ind not in too_large_inds2:
                    selected_ind2.append(ind)
                if ind not in too_large_inds3:
                    selected_ind3.append(ind)
                if ind not in too_large_inds4:
                    selected_ind4.append(ind)
                if ind not in too_large_inds5:
                    selected_ind5.append(ind)
        selected_ind1 = np.array(selected_ind1)
        selected_ind2 = np.array(selected_ind2)
        selected_ind3 = np.array(selected_ind3)
        selected_ind4 = np.array(selected_ind4)
        selected_ind5 = np.array(selected_ind5)
        # print(f">>> Samples with a out-of-range force: {out_of_range_inds}")
        # print(f">>> Samples with a too strong force: {too_large_inds1}")
        # print(f">>> Samples with a too strong force: {too_large_inds2}")
        # print(f">>> Samples with a too strong force: {too_large_inds3}")
        # print(f">>> Samples with a too strong force: {too_large_inds4}")
        # print(f">>> Samples with a too strong force: {too_large_inds5}")
        # print(f">>> Selected {len(selected_ind1)} samples for selected_ind1 {selected_ind1}")
        # print(f">>> Selected {len(selected_ind2)} samples for selected_ind2 {selected_ind2}")
        # print(f">>> Selected {len(selected_ind3)} samples for selected_ind3 {selected_ind3}")
        # print(f">>> Selected {len(selected_ind4)} samples for selected_ind4 {selected_ind4}")
        # print(f">>> Selected {len(selected_ind5)} samples for selected_ind5 {selected_ind5}")
        selected_ind = [selected_ind1, selected_ind2, selected_ind3, selected_ind4, selected_ind5]
        
        # Analyze raw signals
        # print(selected_ind1)
        hall_statistics1 = location_wise_hall_signal_range(all_hall_signals[selected_ind1], all_hall_signals_scaled[selected_ind1], all_heat_maps[selected_ind1])
        hall_statistics2 = location_wise_hall_signal_range(all_hall_signals[selected_ind2], all_hall_signals_scaled[selected_ind2], all_heat_maps[selected_ind2])
        hall_statistics3 = location_wise_hall_signal_range(all_hall_signals[selected_ind3], all_hall_signals_scaled[selected_ind3], all_heat_maps[selected_ind3])
        hall_statistics4 = location_wise_hall_signal_range(all_hall_signals[selected_ind4], all_hall_signals_scaled[selected_ind4], all_heat_maps[selected_ind4])
        hall_statistics5 = location_wise_hall_signal_range(all_hall_signals[selected_ind5], all_hall_signals_scaled[selected_ind5], all_heat_maps[selected_ind5])
        hall_statistics = [hall_statistics1, hall_statistics2, hall_statistics3, hall_statistics4, hall_statistics5]
        # mAP_loc_wise = None
        # mAP, mAP_loc_wise = cal_mAP_loc_wise(all_hall_signals_scaled, all_heat_maps_scaled)
    
        selected_inds.append(selected_ind)
        all_hall_statistics.append(hall_statistics)

    return test_data_all_faces, input_sizes, output_sizes, y_mins, all_hall_statistics, selected_inds, whether_non_contact

# Plot curves with distribution for similarity and force error changes with respect different amount of training data
def vis_raw_signals(hall_data,force_data,fn):
    fig, axs = plt.subplots(2, 1, figsize=[3, 3], gridspec_kw={'height_ratios': [2, 1]})
    fig.tight_layout()
    for i in range(hall_data.shape[1]):
        axs[0].plot(hall_data[:,i])

    axs[0].set_ylabel('Hall signal '+r'$s_{1:9}$')
    axs[1].plot(force_data, color='blue')
    axs[1].set_ylabel('Force (N)')
    axs[1].set_xlabel('Frame')
    plt.savefig(fn, dpi=400, bbox_inches='tight')
    plt.close()

# Add a preprocessing function for the single face situations
def load_data(cfgs):
    all_data_cfgs = cfgs["dataset"]

    # Load all data files
    hall_signals, heat_maps, case_labels = [], [], []
    dataset_names, novel_inds, rest_inds = [], [], []
    for i in range(len(all_data_cfgs)):
        data_cfgs = all_data_cfgs[i]
        data_path = data_cfgs["csv_path"]
        dataset_name = data_cfgs["name"]
        dataset_names.append(dataset_name)
        data = pd.read_csv(data_path)

        hall_signal, heat_map, case_label = data_preprocessing(data, data_cfgs)
        # Save examplary hall signals and force measures for visualization
        vis_inds = np.where(case_label==(max(case_label)//2))[0]
        vis_force = np.sum(heat_map[vis_inds],axis=1)
        vis_hall_signal = hall_signal[vis_inds]
        result_path = cfgs["result"]["path"]
        faces = cfgs["dataset"][0]["faces"]
        faces = map(str, faces)
        faces = ''.join(faces)
        vis_raw_signal_data_fn = f"{result_path}/vis_raw_signal_data_{faces}_{dataset_name}.pkl"
        file = open(vis_raw_signal_data_fn, 'wb')
        pickle.dump([vis_hall_signal, vis_force], file)
        file.close()
        vis_raw_signal_fig_fn = f"{result_path}/vis_raw_signal_data_{faces}_{dataset_name}.png"
        vis_raw_signals(vis_hall_signal,vis_force,vis_raw_signal_fig_fn)

        hall_signals.append(hall_signal)
        heat_maps.append(heat_map)
        case_labels.append(case_label)

        novel_case_inds = data_cfgs["novel_inds"]
        novel_ind = np.array([i for i in range(len(case_label)) if case_label[i] in novel_case_inds])
        rest_ind = np.array([i for i in range(len(case_label)) if case_label[i] not in novel_case_inds])
        novel_inds.append(novel_ind)
        rest_inds.append(rest_ind)
    all_hall_signals = np.concatenate(hall_signals, axis=0)
    all_heat_maps = np.concatenate(heat_maps, axis=0)
    print("Total samples loaded: ", len(all_hall_signals))

    # Regularize data
    if "scaler_fn" in data_cfgs:
        # Loading the scaler and dataset statistics
        scaler_fn = cfgs["scaler_fn"]
        # faces = cfgs["dataset"][0]["faces"]
        # faces = map(str, faces)
        # faces = ''.join(faces)
        # scaler_fn = f"{result_path}/training_hist_face{faces}_data_scalers.pkl"
        file = open(scaler_fn, 'rb')
        data = pickle.load(file)
        y_scaler = data[1]
        X_scaler = data[0]
    else:
        X_scaler = preprocessing.MinMaxScaler().fit(all_hall_signals)
        y_scaler = preprocessing.MinMaxScaler().fit(all_heat_maps)
    all_heat_maps_scaled = y_scaler.transform(all_heat_maps)
    all_hall_signals_scaled = X_scaler.transform(all_hall_signals)

    # Analyze raw signals
    # hall_statistics = location_wise_hall_signal_range(all_hall_signals, all_hall_signals_scaled, all_heat_maps)
    hall_statistics = None
    mAP_loc_wise = None
    # mAP, mAP_loc_wise = cal_mAP_loc_wise(all_hall_signals_scaled, all_heat_maps_scaled)

    # Divide data into training, validation, test_seen, and test_unseen datasets
    X_test_novels, Y_test_novels, len_test_novels = [], [], []
    X_test_knowns, Y_test_knowns, len_test_knowns = [], [], []
    X_trains, Y_trains = [], [] 
    X_vals, Y_vals = [], []
    for i in range(len(hall_signals)):
        hall_signals_scaled = X_scaler.transform(hall_signals[i])
        heat_maps_scaled = y_scaler.transform(heat_maps[i])

        if len(novel_inds[i]) > 0:
            X_test_novel = hall_signals_scaled[novel_inds[i]]
            Y_test_novel = heat_maps_scaled[novel_inds[i]]
            X_test_novels.append(X_test_novel)
            Y_test_novels.append(Y_test_novel)
            len_test_novels.append(len(Y_test_novel))

        X_rest = hall_signals_scaled[rest_inds[i]]
        Y_rest = heat_maps_scaled[rest_inds[i]]
        X_train, X_test_known, Y_train, Y_test_known = train_test_split(X_rest, Y_rest, 
                                                            test_size=0.2, random_state=2)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, 
                                                            test_size=0.25, random_state=2)
        
        # X_train, X_test_known, Y_train, Y_test_known = train_test_split(X_rest, Y_rest, 
        #                                                     test_size=0.2, random_state=2, shuffle=False)
        # X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, 
        #                                                     test_size=0.25, random_state=2, shuffle=False)
        
        # Temporally reduce the amount of training data
        if "ds_factor2" in all_data_cfgs[i]:
            ds_factor = all_data_cfgs[i]["ds_factor2"]
            selected_inds = np.arange(0,len(X_train),ds_factor)
            X_train = X_train[selected_inds]
            Y_train = Y_train[selected_inds]

        X_test_knowns.append(X_test_known)
        Y_test_knowns.append(Y_test_known)
        len_test_knowns.append(len(Y_test_known))
        X_trains.append(X_train)
        Y_trains.append(Y_train)
        X_vals.append(X_val)
        Y_vals.append(Y_val)

    X_train_all = np.concatenate(X_trains, axis=0)
    Y_train_all = np.concatenate(Y_trains, axis=0)
    X_val_all = np.concatenate(X_vals, axis=0)
    Y_val_all = np.concatenate(Y_vals, axis=0)
    X_test_known_all = np.concatenate(X_test_knowns, axis=0)
    Y_test_known_all = np.concatenate(Y_test_knowns, axis=0)
    
    training_data = ContactForceDataSet(X_train_all, Y_train_all)
    val_data = ContactForceDataSet(X_val_all, Y_val_all)
    test_data_known = ContactForceDataSet(X_test_known_all, Y_test_known_all)

    print("Training dataset size: ", len(training_data))
    print("Validation dataset size: ", len(val_data))
    print("Test (known) dataset size: ", len(test_data_known), len_test_knowns)
    # mAP, mAP_loc_wise = cal_mAP_loc_wise(X_train_all, Y_train_all)

    if len(X_test_novels) > 0:
        X_test_novel_all = np.concatenate(X_test_novels, axis=0)
        Y_test_novel_all = np.concatenate(Y_test_novels, axis=0)
        test_data_novel = ContactForceDataSet(X_test_novel_all, Y_test_novel_all)
        print("Test (novel) dataset size: ", len(test_data_novel), len_test_novels)
    else:
        test_data_novel = None
   
    y_max = np.max(all_heat_maps_scaled, axis=1)
    y_min = np.min(y_max[np.nonzero(y_max)])
    print(y_max.min(),y_max.max())

    # Saving the scaler and dataset statistics
    if "scaler_fn" not in data_cfgs:
        result_path = cfgs["result"]["path"]
        faces = cfgs["dataset"][0]["faces"]
        faces = map(str, faces)
        faces = ''.join(faces)
        scaler_fn = f"{result_path}/training_hist_face{faces}_data_scalers.pkl"
        file = open(scaler_fn, 'wb')
        pickle.dump([X_scaler, y_scaler, len_test_knowns, len_test_novels, y_min], file)
        file.close()

    return training_data, val_data, test_data_known, test_data_novel, X_train_all.shape[1], Y_train_all.shape[1], y_min, hall_statistics, mAP_loc_wise

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)