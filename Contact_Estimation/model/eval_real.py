#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from models import MLP
from utils import load_data_real_test, taxel_projection
import argparse
import yaml

import argparse
import os
import gc
import pickle
from scipy import signal

    
def visualize_sample(pred, gt, projected=False):
    if not projected:
        pred = taxel_projection(pred)[0]
        gt = taxel_projection(gt)[0]  
    
    fig = plt.figure(figsize=[9, 3])
    ax = fig.add_subplot(1, 2, 1)
#     fig, ax = plt.subplots()
    # im = ax.imshow(f1_diff, aspect='auto')
    # im = ax.imshow(pred, aspect='auto', cmap='YlOrRd')
    im = ax.imshow(pred, aspect='auto', cmap='hot')
    plt.colorbar(im, ax=ax)  
    plt.title('Predicted heatmap',y=-0.12)
    ax.set_xticks(np.arange(pred.shape[1]), labels=np.arange(1,pred.shape[1]+1))
    ax.set_yticks(np.arange(pred.shape[0]), labels=np.arange(1,pred.shape[0]+1))
    ax.xaxis.set_ticks_position('top')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.xaxis.set_label_position('top') 
    
    ax = fig.add_subplot(1, 2, 2)
#     fig, ax = plt.subplots()
    # im = ax.imshow(f1_diff, aspect='auto')
    im = ax.imshow(gt, aspect='auto', cmap='hot')
    plt.colorbar(im, ax=ax)  
    plt.title('Ground-truth heatmap',y=-0.12)
    ax.set_xticks(np.arange(gt.shape[1]), labels=np.arange(1,gt.shape[1]+1))
    ax.set_yticks(np.arange(gt.shape[0]), labels=np.arange(1,gt.shape[0]+1))
    ax.xaxis.set_ticks_position('top')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.xaxis.set_label_position('top') 

# Measure contact location statistics
def contact_location_accuracy_single_case(pred, gt, thres):
    pred_bool = pred > thres
    y_bool = gt > thres
    pred_sum = np.sum(pred_bool.astype('int'),axis=0)
    y_sum = np.sum(y_bool.astype('int'),axis=0)

    return pred_sum, y_sum

def location_wise_contact_accuracy(pred, gt, force_range2d):
     # Obatain the subset of true positives in terms of number of contact locations
    # pred_bool = pred > thres
    # y_bool = gt > thres
    # # num_contacts_pred = pred_bool.sum(axis=1)
    # num_contacts_y = y_bool.sum(axis=1)
    # true_positive_samples = (num_contacts_y > 0)
    # inds = true_positive_samples.nonzero()
    # pred = pred[inds]
    # gt = gt[inds]

    # Project to heatmaps
    proj_pred = taxel_projection(pred)
    proj_gt =taxel_projection(gt)

    # Calculate errors
    avg_v_errors = np.zeros(proj_gt.shape[1:])
    avg_h_errors = np.zeros(proj_gt.shape[1:])
    avg_eucl_errors = np.zeros(proj_gt.shape[1:])
    avg_cc_max_values = np.zeros(proj_gt.shape[1:])
    avg_force_errors = np.zeros(proj_gt.shape[1:])
    avg_force_errors_scaled = np.zeros(proj_gt.shape[1:])
    force_errors_scaled_globally = []
    force_errors_scaled_locally = []
    for j in range(proj_gt.shape[1]):
        for k in range(proj_gt.shape[2]):
            curr_inds = proj_gt[:,j,k].nonzero()
            curr_preds = proj_pred[curr_inds]
            curr_gts = proj_gt[curr_inds]

            v_errors = []
            h_errors = []
            cc_max_values = []
            force_errors = []
            force_errors_scaled = []
            num_abnormal_samples = 0
            for i in range(len(curr_gts)):
                curr_pred = curr_preds[i]
                curr_gt = curr_gts[i]
                # cc_matrix = signal.correlate2d((curr_pred-curr_pred.mean())/curr_pred.std(), (curr_gt-curr_gt.mean())/curr_gt.std(), boundary='fill', mode='same')
                cc_matrix = signal.correlate2d((curr_pred-curr_pred.mean())/curr_pred.std(), (curr_gt-curr_gt.mean())/curr_gt.std(), boundary='fill', mode='full')
                cc_matrix = cc_matrix/100.0
                cc_max = cc_matrix.max()
                max_locs = np.argwhere(cc_matrix == cc_max)
                if len(max_locs) > 1:
                    print(max_locs)
                    visualize_sample(curr_pred, curr_gt, projected=True)
                    plt.show()
                centre_loc = 9
                max_offset = max_locs[0] - centre_loc
                v_error = max_offset[0]
                h_error = max_offset[1]
                gt_inds = curr_gt.nonzero()
                pred_inds0 = gt_inds[0] + v_error
                pred_inds1 = gt_inds[1] + h_error
                pred_inds = (pred_inds0, pred_inds1)

                if pred_inds0.max() >= 10 or pred_inds1.max() >=10 or cc_max < 0.3:
                    num_abnormal_samples = num_abnormal_samples + 1
                    # print("abnormal offsets", max_offset)
                    # print(cc_max, max_locs)
                    # print(cc_max, pred_inds)
                    # print(pred_inds)
                    # print(curr_gt, curr_pred)
                    # print(cc_matrix) 
                    # visualize_sample(curr_pred, curr_gt, projected=True)
                    # plt.show()
                else:
                    force_error = curr_pred[pred_inds] - curr_gt[gt_inds]
                    force_error_scaled = np.divide(force_error,curr_gt[gt_inds])
                    # print(v_error, h_error, cc_max)
                    v_errors.append(v_error)
                    h_errors.append(h_error)
                    force_errors.extend(force_error.tolist())
                    force_errors_scaled.extend(force_error_scaled.tolist())
                cc_max_values.append(cc_max)

                # force_error = curr_pred[pred_inds] - curr_gt[gt_inds]    
            
                # v_errors.append(v_error)
                # h_errors.append(h_error)
                # cc_max_values.append(cc_max)
                # force_errors.extend(force_error.tolist())
            
            v_errors = np.array(v_errors)
            h_errors = np.array(h_errors)
            # eucl_errors = np.linalg.norm(np.concatenate((v_errors,h_errors), axis=1), axis=1)
            eucl_errors = np.linalg.norm(np.vstack((v_errors,h_errors)).T, axis=1)
            cc_max_values = np.array(cc_max_values)
            force_errors = np.array(force_errors)
            force_errors_scaled = np.array(force_errors_scaled)
            force_errors_scaled_globally.extend((force_errors/force_range2d[j,k]).tolist())
            force_errors_scaled_locally.extend(force_errors_scaled.tolist())
            
            avg_v_errors[j,k] = np.abs(v_errors).mean()
            avg_h_errors[j,k] = np.abs(h_errors).mean()
            avg_eucl_errors[j,k] = eucl_errors.mean()
            avg_cc_max_values[j,k] = cc_max_values.mean()
            avg_force_errors[j,k] = np.abs(force_errors).mean()
            avg_force_errors_scaled[j,k] = np.abs(force_errors_scaled).mean()
            
            if num_abnormal_samples > 0:
                print(f"Num of abnormal samples for (x={k},y={j}): {num_abnormal_samples}")

    return avg_v_errors, avg_h_errors, avg_eucl_errors, avg_cc_max_values, avg_force_errors, avg_force_errors_scaled, np.array(force_errors_scaled_globally), np.array(force_errors_scaled_locally)


# Measure location accuracy
def contact_location_accuracy(pred, gt, thres, result_path):
    # Obatain the subset of true positives in terms of number of contact locations
    # pred_bool = pred > thres
    y_bool = gt > thres
    # num_contacts_pred = pred_bool.sum(axis=1)
    num_contacts_y = y_bool.sum(axis=1)
    # true_positive_samples = np.logical_and((num_contacts_pred == num_contacts_y), (num_contacts_y > 0))
    true_positive_samples = (num_contacts_y > 0)
    inds = true_positive_samples.nonzero()
    pred = pred[inds]
    gt = gt[inds]

    # Project to heatmaps
    proj_pred = taxel_projection(pred)
    proj_gt =taxel_projection(gt)

    # Calculate errors
    v_errors = []
    h_errors = []
    cc_max_values = []
    force_errors = []
    bad_samples1, bad_samples2, bad_samples3 = [], [], []
    num_abnormal_samples = 0
    min_sim1, min_sim2, min_sim3 = 1, 1, 1
    max_sim1, max_sim2, max_sim3 = 0, 0, 0
    worst_sample1, worst_sample2, worst_sample3 = None, None, None
    best_sample1, best_sample2, best_sample3 = None, None, None
    for i in range(len(gt)):
        curr_pred = proj_pred[i]
        curr_gt = proj_gt[i]

        # cc_matrix = signal.correlate2d((curr_pred-curr_pred.mean())/curr_pred.std(), (curr_gt-curr_gt.mean())/curr_gt.std(), boundary='fill', mode='same')
        cc_matrix = signal.correlate2d((curr_pred-curr_pred.mean())/curr_pred.std(), (curr_gt-curr_gt.mean())/curr_gt.std(), boundary='fill', mode='full')
        # cc_matrix = signal.correlate2d(curr_pred/curr_pred.std(), curr_gt/curr_gt.std(), boundary='fill', mode='same')
        cc_matrix = cc_matrix/100.0
        cc_max = cc_matrix.max()
        max_locs = np.argwhere(cc_matrix == cc_max)
        if len(max_locs) > 1:
            print(max_locs)
            visualize_sample(curr_pred, curr_gt, projected=True)
            plt.show()
        centre_loc = 9
        max_offset = max_locs[0] - centre_loc
        v_error = max_offset[0]
        h_error = max_offset[1]
        gt_inds = curr_gt.nonzero()
        pred_inds0 = gt_inds[0] + v_error
        pred_inds1 = gt_inds[1] + h_error
        pred_inds = (pred_inds0, pred_inds1)
        if pred_inds0.max() >= 10 or pred_inds1.max() >=10 or cc_max < 0.3:
            num_abnormal_samples = num_abnormal_samples + 1
            # print("abnormal offsets", max_offset)
            # print(cc_max, max_locs)
            # print(cc_max, pred_inds)
            # print(pred_inds)
            # print(curr_gt, curr_pred)
            # print(cc_matrix) 
            # visualize_sample(curr_pred, curr_gt, projected=True)
            # plt.show()
        else:
            force_error = curr_pred[pred_inds] - curr_gt[gt_inds]
            # print(v_error, h_error, cc_max)
            v_errors.append(v_error)
            h_errors.append(h_error)
            force_errors.extend(force_error.tolist())
        cc_max_values.append(cc_max)

        # if abs(v_error) > 0 or abs(h_error) > 0:
        src = curr_gt.nonzero()
        if cc_max < 0.9:
            # print("Max Cross Correlation (Zero Normalized): ", cc_max)
            # print(dst_res[dst_coordinate])
            # visualize_sample(curr_pred, curr_gt, projected=True)
            # plt.show()
            if len(src[0]) == 1 and len(bad_samples1) < 20:
                bad_samples1.append((curr_pred, curr_gt, cc_max))
            if len(src[0]) == 2 and len(bad_samples2) < 20:
                bad_samples2.append((curr_pred, curr_gt, cc_max))
            if len(src[0]) == 3 and len(bad_samples3) < 20:
                bad_samples3.append((curr_pred, curr_gt, cc_max))
            # Add support for the visualization of the worst case (minimum maximum cross correlation)

        if len(src[0]) == 1 and cc_max <= min_sim1:
            worst_sample1 = (curr_pred, curr_gt, cc_max)
            min_sim1 = cc_max
        if len(src[0]) == 2 and cc_max <= min_sim2:
            worst_sample2 = (curr_pred, curr_gt, cc_max)
            min_sim2 = cc_max
        if len(src[0]) == 3 and cc_max <= min_sim3:
            worst_sample3 = (curr_pred, curr_gt, cc_max)
            min_sim3 = cc_max
        if len(src[0]) == 1 and cc_max >= max_sim1:
            best_sample1 = (curr_pred, curr_gt, cc_max)
            max_sim1 = cc_max
        if len(src[0]) == 2 and cc_max >= max_sim2:
            best_sample2 = (curr_pred, curr_gt, cc_max)
            max_sim2 = cc_max
        if len(src[0]) == 3 and cc_max >= max_sim3:
            best_sample3 = (curr_pred, curr_gt, cc_max)
            max_sim3 = cc_max

    for i in range(len(bad_samples1)):
        curr_pred = bad_samples1[i][0]
        curr_gt = bad_samples1[i][1]
        visualize_sample(curr_pred, curr_gt, projected=True)
        plt.draw()
        plt.savefig(f"{result_path}/bad_sample1_{i}.png", dpi=400, bbox_inches='tight')
        plt.close()      
    for i in range(len(bad_samples2)):
        curr_pred = bad_samples2[i][0]
        curr_gt = bad_samples2[i][1]
        # print(curr_gt)
        visualize_sample(curr_pred, curr_gt, projected=True)
        plt.draw()
        plt.savefig(f"{result_path}/bad_sample2_{i}.png", dpi=400, bbox_inches='tight')
        plt.close()
    for i in range(len(bad_samples3)):
        curr_pred = bad_samples3[i][0]
        curr_gt = bad_samples3[i][1]
        visualize_sample(curr_pred, curr_gt, projected=True)
        plt.draw()
        plt.savefig(f"{result_path}/bad_sample3_{i}.png", dpi=400, bbox_inches='tight')
        plt.close()   

    if worst_sample1 is not None:
        visualize_sample(worst_sample1[0], worst_sample1[1], projected=True)
        plt.draw()
        plt.savefig(f"{result_path}/worst_sample1_{round(worst_sample1[2],3)}.png", dpi=400, bbox_inches='tight')
        plt.close()

    if worst_sample2 is not None:
        visualize_sample(worst_sample2[0], worst_sample2[1], projected=True)
        plt.draw()
        plt.savefig(f"{result_path}/worst_sample2_{round(worst_sample2[2],3)}.png", dpi=400, bbox_inches='tight')
        plt.close()  

    if worst_sample3 is not None:
        visualize_sample(worst_sample3[0], worst_sample3[1], projected=True)
        plt.draw()
        plt.savefig(f"{result_path}/worst_sample3_{round(worst_sample3[2],3)}.png", dpi=400, bbox_inches='tight')
        plt.close()       

    if best_sample1 is not None:
        visualize_sample(best_sample1[0], best_sample1[1], projected=True)
        plt.draw()
        plt.savefig(f"{result_path}/best_sample1_{round(best_sample1[2],3)}.png", dpi=400, bbox_inches='tight')
        plt.close()

    if best_sample2 is not None:
        visualize_sample(best_sample2[0], best_sample2[1], projected=True)
        plt.draw()
        plt.savefig(f"{result_path}/best_sample2_{round(best_sample2[2],3)}.png", dpi=400, bbox_inches='tight')
        plt.close()  

    if best_sample3 is not None:
        visualize_sample(best_sample3[0], best_sample3[1], projected=True)
        plt.draw()
        plt.savefig(f"{result_path}/best_sample3_{round(best_sample3[2],3)}.png", dpi=400, bbox_inches='tight')
        plt.close() 


    v_errors = np.array(v_errors)
    h_errors = np.array(h_errors)
    # eucl_errors = np.linalg.norm(np.concatenate((v_errors,h_errors), axis=1), axis=1)
    eucl_errors = np.linalg.norm(np.vstack((v_errors,h_errors)).T, axis=1)
    cc_max_values = np.array(cc_max_values)
    force_errors = np.array(force_errors)
    if num_abnormal_samples > 0:
        print("Num of abnormal samples: ", num_abnormal_samples)

    return v_errors, h_errors, eucl_errors, cc_max_values, force_errors


def contact_location_estimate_dist(pred, gt, thres, result_path):

    # extract case information, organize different samples in the case dimension
    y_bool = gt > thres
    cases, case_inds = np.unique(y_bool, return_inverse=True, axis=0)

    # Visualization
    for i in range(len(cases)):
        curr_case =  cases[i]
        inds = np.where(case_inds == i)[0]
        curr_pred = pred[inds]
        curr_gt = gt[inds]
        pred_sum, y_sum = contact_location_accuracy_single_case(curr_pred, curr_gt, thres)
        visualize_sample(pred_sum, y_sum)
        plt.draw()
        curr_case = np.nonzero(curr_case)[0]+1
        if len(curr_case) == 0:
            curr_case = "0"
        else:
            # print(curr_case)
            curr_case = map(str, curr_case.tolist())
            curr_case = '_'.join(curr_case)
        # print(curr_case)
        plt.savefig(f"{result_path}/case_{curr_case}.png", dpi=400, bbox_inches='tight')
        plt.close()


# Measure the accuracy of binary estimation on contact/non-contact
def contact_accuracy(pred, gt, thres):
    pred_bool = pred > thres
    y_bool = gt > thres
    contact_acc = (pred_bool == y_bool).sum()/y_bool.size

    return contact_acc

# Measure the accuracy of number of contacts
def non_contact_accuracy(pred, gt, thres):
    pred_bool = pred > thres
    y_bool = gt > thres
    num_contacts_pred = pred_bool.sum(axis=1)
    num_contacts_y = y_bool.sum(axis=1)
    inds = (num_contacts_y == 0).nonzero()
    num_contacts_pred = num_contacts_pred[inds]
    num_contacts_y = num_contacts_y[inds]
    num_contact_acc = (num_contacts_pred == num_contacts_y).sum()/num_contacts_y.size

    return num_contact_acc

# Measure force estimation accuracy on true positive samples
def force_accuracy_on_tps(pred, gt, thres, force_scaler):
    pred_bool = pred > thres
    y_bool = gt > thres

    true_positive_samples = np.logical_and((pred_bool == y_bool), y_bool)
    inds = true_positive_samples.nonzero()

    pred_real = force_scaler.inverse_transform(pred)
    gt_real = force_scaler.inverse_transform(gt)

    # force_err = pred[inds] - gt[inds]
    force_err = abs(pred_real[inds] - gt_real[inds])
    if len(force_err) > 0:
        force_err_max = force_err.max()
        force_err_avg = np.average(force_err)
        force_err_med = np.median(force_err)
    else:
        force_err_max, force_err_avg, force_err_med = None, None, None
    
    return force_err, force_err_max, force_err_avg, force_err_med

def plot_histgram3d(x, y, fn):
    fig = plt.figure(figsize=[7,4])          #create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')

    #make histogram stuff - set bins - I choose 20x20 because I have a lot of data
    hist, xedges, yedges = np.histogram2d(x, y, bins=(5,5))
    hist = hist.T
    xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])

    xpos = xpos.flatten()/2.
    ypos = ypos.flatten()/2.
    zpos = np.zeros_like (xpos)

    dx = xedges [1] - xedges [0]
    dy = yedges [1] - yedges [0]
    dz = hist.flatten()

    # cmap = plt.get_cmap('jet') # Get desired colormap - you can change this!
    # cmap = plt.get_cmap('Spectral') # Get desired colormap - you can change this!
    cmap = plt.get_cmap('rainbow') # Get desired colormap - you can change this!
    max_height = np.max(dz)   # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    ax.set_xlabel('X axis error')
    ax.set_ylabel('Y axis error')
    ax.set_zlabel('No. of samples')
    plt.draw()
    plt.savefig(fn, dpi=400, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_histgram2d(x, y, fn):
    hist, xedges, yedges = np.histogram2d(x, y, bins=(5,5))
    hist = hist.T
    xedges = np.around(xedges, 3)
    yedges = np.around(yedges, 3)
    # print(hist, xedges, yedges)
    fig, ax = plt.subplots(figsize=[2, 1.5])
    # im = ax.imshow(hist, aspect='auto', cmap='GnBu')
    im = ax.imshow(hist, aspect='auto', cmap='Blues_r')
    # plt.colorbar(im, ax=ax)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(xedges))-0.5, labels=xedges)
    ax.set_yticks(np.arange(len(yedges))-0.5, labels=yedges)
    ax.xaxis.set_ticks_position('top')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.xaxis.set_label_position('top') 
    
    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")

    # data = np.around(data, 3)
    color_thr = 0.5 * (hist.max() + hist.min())

    # Loop over data dimensions and create text annotations.
    for i in range(len(yedges)-1):
        for j in range(len(xedges)-1):
            txt = np.array2string(hist[i,j])
            if hist[i,j] < color_thr:
                txt_color = "w"
            else:
                txt_color = "black"
            text = ax.text(j, i, txt,
                        ha="center", va="center", color=txt_color)

    # ax.set_title(title)
    fig.tight_layout()
    fig.set_figheight(5)
    fig.set_figwidth(6)
    plt.savefig(fn, dpi=400, bbox_inches='tight')
    plt.close()

def vis_matrix(data, xticks, yticks, title, fn):
    fig, ax = plt.subplots()
    im = ax.imshow(data, aspect='auto')
    # plt.colorbar(im, ax=ax)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(xticks)), labels=xticks)
    ax.set_yticks(np.arange(len(yticks)), labels=yticks)
    ax.xaxis.set_ticks_position('top')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.xaxis.set_label_position('top') 
    

    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")

    data = np.around(data, 3)
    color_thr = 0.5 * (data.max() + data.min())

    # Loop over data dimensions and create text annotations.
    for i in range(len(yticks)):
        for j in range(len(xticks)):
            txt = np.array2string(data[i,j])
            if data[i,j] < color_thr:
                txt_color = "w"
            else:
                txt_color = "black"
            text = ax.text(j, i, txt,
                        ha="center", va="center", color=txt_color)

    # ax.set_title(title)
    fig.tight_layout()
    fig.set_figheight(5)
    fig.set_figwidth(6)
    plt.savefig(fn, dpi=400, bbox_inches='tight')
    plt.close()

def location_wise_force_error(pred, gt):
    pred_2d = taxel_projection(pred)
    gt_2d = taxel_projection(gt)

    err_force_2d = np.zeros(pred_2d.shape[1:])
    for i in range(gt_2d.shape[1]):
        for j in range(gt_2d.shape[2]):
            curr_pred = pred_2d[:,i,j]
            curr_gt = gt_2d[:,i,j]
            nz_inds = curr_gt.nonzero()
            pred_nz = curr_pred[nz_inds]
            gt_nz = curr_gt[nz_inds]
            curr_err = abs(pred_nz - gt_nz)
            err_force_2d[i,j] = np.average(curr_err)

    return err_force_2d

# TODO: Think about how to evaluate the performance with different numbers of probes on other faces?????
def eval_non_contact_performance(pred, gt, result_path, thres, force_scaler, selected_ind, probes=[0,1]):
    # Add support for evaluating different probe situations
    # y_bool = gt > 0
    # num_contacts_y = y_bool.sum(axis=1)
    # inds = []
    # for probe in probes:
    #     inds.append((num_contacts_y == probe).nonzero()[0])

    # inds = np.concatenate(inds, axis=0)
    # pred = pred[inds]
    # gt = gt[inds]
    pred_real = force_scaler.inverse_transform(pred)
    gt_real = force_scaler.inverse_transform(gt)

    for i in range(len(selected_ind)):
        curr_selected_ind = selected_ind[i]
        curr_result_path = f"{result_path}/selected_ind{i+1}"
        if not os.path.exists(curr_result_path):
            os.makedirs(curr_result_path)
        print(f">>>>>>> Evaluating non-contact performance of the subset for selected_ind{i+1}")

        # selected_ind_mapped = []
        # for i in range(len(inds)):
        #     if inds[i] in curr_selected_ind:
        #         selected_ind_mapped.append(i)
        # selected_ind_mapped = np.array(selected_ind_mapped)
        # print(selected_ind_mapped)
        selected_ind_mapped = curr_selected_ind

        curr_pred = pred[selected_ind_mapped]
        curr_gt = gt[selected_ind_mapped]
        curr_pred_real = pred_real[selected_ind_mapped]
        curr_gt_real = gt_real[selected_ind_mapped]

        print("Max selected force (ground-truth): ", np.absolute(curr_gt_real).max(axis=1).mean())
        print("Max selected force (predicted): ", np.absolute(curr_pred_real).max(axis=1).mean())

        # Save all evaluation results
        result_pkl_fn = f"{curr_result_path}/non_contact_errors.pkl"
        file = open(result_pkl_fn, 'wb')
        pickle.dump([curr_pred, curr_pred_real], file)
        file.close()

        # Measure binary contact accuracy
        contact_acc = contact_accuracy(curr_pred, curr_gt, thres)
        print(">> Contact Accuracy: ", contact_acc)
        non_contact_acc = non_contact_accuracy(curr_pred, curr_gt, thres)
        print(">> Non-contact Accuracy: ", non_contact_acc)



def eval_performance(pred, gt, result_path, thres, force_scaler, selected_ind, whether_non_contact=False):    
    # curr_result_path = f"{result_path}/1probe"
    # if not os.path.exists(curr_result_path):
    #     os.makedirs(curr_result_path)
    # if whether_non_contact:
    #     eval_non_contact_performance(pred, gt, result_path, thres, force_scaler, selected_ind, probes=[0,1])
    # else:
    #     eval_performance_single_probe_situation(pred, gt, curr_result_path, thres, force_scaler, selected_ind, probes=[0,1])
    
    if whether_non_contact:
        eval_non_contact_performance(pred, gt, result_path, thres, force_scaler, selected_ind, probes=[0,2])
        gc.collect()
    else:
        curr_result_path = f"{result_path}/1probe"
        if not os.path.exists(curr_result_path):
            os.makedirs(curr_result_path)
        eval_performance_single_probe_situation(pred, gt, curr_result_path, thres, force_scaler, selected_ind, probes=[0,1])
        gc.collect()
        
        curr_result_path = f"{result_path}/2probe"
        if not os.path.exists(curr_result_path):
            os.makedirs(curr_result_path)
        eval_performance_single_probe_situation(pred, gt, curr_result_path, thres, force_scaler, selected_ind, probes=[0,2])
        gc.collect()

        curr_result_path = f"{result_path}/3probe"
        if not os.path.exists(curr_result_path):
            os.makedirs(curr_result_path)
        eval_performance_single_probe_situation(pred, gt, curr_result_path, thres, force_scaler, selected_ind, probes=[0,3])
        gc.collect()

        curr_result_path = f"{result_path}/all_probe"
        if not os.path.exists(curr_result_path):
            os.makedirs(curr_result_path)
        eval_performance_single_probe_situation(pred, gt, curr_result_path, thres, force_scaler, selected_ind, probes=[0,1,2,3])
        gc.collect()

def eval_performance_single_probe_situation(pred, gt, result_path, thres, force_scaler, selected_ind, probes):
    for i in range(len(selected_ind)):
        curr_selected_ind = selected_ind[i]
        curr_result_path = f"{result_path}/selected_ind{i+1}"
        if not os.path.exists(curr_result_path):
            os.makedirs(curr_result_path)
        print(f">>>>>>> Evaluating the subset for selected_ind{i+1}")
        eval_performance_single_probe_situation_single_subset(pred, gt, curr_result_path, thres, force_scaler, curr_selected_ind, probes)
        gc.collect()


def eval_performance_single_probe_situation_single_subset(pred, gt, result_path, thres, force_scaler, selected_ind, probes=[1,2,3]):
    # Add support for evaluating different probe situations
    y_bool = gt > thres
    # y_bool = gt > 0
    num_contacts_y = y_bool.sum(axis=1)
    inds = []
    for probe in probes:
        inds.append((num_contacts_y == probe).nonzero()[0])

    # print(probes)
    # print(inds)
    inds = np.concatenate(inds, axis=0)
    pred = pred[inds]
    gt = gt[inds]
    pred_real = force_scaler.inverse_transform(pred)
    gt_real = force_scaler.inverse_transform(gt)

    # for i in range(len(gt_real)):
    #     visualize_sample(pred_real[i], gt_real[i])
    #     plt.draw()
    #     plt.savefig(f"{result_path}/sample_{inds[i]}.png", dpi=400, bbox_inches='tight')
    #     plt.close()

    selected_ind_mapped = []
    for i in range(len(inds)):
        if inds[i] in selected_ind:
            selected_ind_mapped.append(i)

    selected_ind_mapped = np.array(selected_ind_mapped)
    # print(selected_ind_mapped)

    pred = pred[selected_ind_mapped]
    gt = gt[selected_ind_mapped]
    pred_real = pred_real[selected_ind_mapped]
    gt_real = gt_real[selected_ind_mapped]

    if len(gt_real) == 0:
        print("!!!!!!! No selected samples in the subset !!!!!!!")
        return

    print("Max selected force (ground-truth): ", gt_real.max())
    print("Max selected force (predicted): ", pred_real.max())

    # for i in range(len(gt_real)):
    #     visualize_sample(pred_real[i], gt_real[i])
    #     plt.draw()
    #     plt.savefig(f"{result_path}/sample_{inds[selected_ind_mapped[i]]}.png", dpi=400, bbox_inches='tight')
    #     plt.close()

    # Measure binary contact accuracy
    contact_acc = contact_accuracy(pred, gt, thres)
    print(">> Contact Accuracy: ", contact_acc)
    non_contact_acc = non_contact_accuracy(pred, gt, thres)
    print(">> Non-contact Accuracy: ", non_contact_acc)

    if gt_real.max() == 0:
        print("!!!!!!! No contacting samples in the subset !!!!!!!")
        return

    # Measure contact location accuracy
    loc_acc_path = f"{result_path}/loc_acc"
    if not os.path.exists(loc_acc_path):
        os.makedirs(loc_acc_path)
    contact_location_estimate_dist(pred, gt, thres, loc_acc_path)
    # v_errors, h_errors, eucl_errors, cc_max_values = contact_location_accuracy(pred, gt, thres, result_path)
    v_errors, h_errors, eucl_errors, cc_max_values, force_errors = contact_location_accuracy(pred_real, gt_real, thres, result_path)
    if len(eucl_errors) >= 10:
        plot_histgram3d(h_errors, v_errors, f"{result_path}/contact_location_errors_hist3d.png")
        plot_histgram2d(h_errors, v_errors, f"{result_path}/contact_location_errors_hist2d.png")
        fig = plt.figure(figsize=[2,3])
        # plt.boxplot(cc_max_values)
        plt.hist(eucl_errors, 10, color='blue',alpha=0.8)
        plt.xlabel('Euclidean errors')
        plt.ylabel('No. of samples')
        plt.draw()
        plt.savefig(f"{result_path}/eucl_errors_hist.png", dpi=400, bbox_inches='tight')
        plt.close()
        fig = plt.figure(figsize=[2,3])
        # plt.boxplot(cc_max_values)
        plt.hist(cc_max_values, 10, color='blue',alpha=0.8)
        plt.xlabel('Heat map similarity')
        plt.ylabel('No. of samples')
        plt.draw()
        plt.savefig(f"{result_path}/cc_max_hist.png", dpi=400, bbox_inches='tight')
        plt.close()
        abs_v_errors = np.absolute(v_errors)
        abs_h_errors = np.absolute(h_errors)
        print(f"====== Contact Location Accuracy {result_path} ======")
        print("Vertical errors: max:{}, min:{}, median:{}, avg:{}".format(abs_v_errors.max(), abs_v_errors.min(), np.median(abs_v_errors), abs_v_errors.mean()))
        print("Horizontal errors: max:{}, min:{}, median:{}, avg:{}".format(abs_h_errors.max(), abs_h_errors.min(), np.median(abs_h_errors), abs_h_errors.mean()))
        print("Euclidean errors: max:{}, min:{}, median:{}, avg:{}".format(eucl_errors.max(), eucl_errors.min(), np.median(eucl_errors), eucl_errors.mean()))
        print("CC Max: max:{}, min:{}, median:{}, avg:{}".format(cc_max_values.max(), cc_max_values.min(), np.median(cc_max_values), cc_max_values.mean()))
    else:
        print("!!!! Not enough samples for histogram plotting !!!!", len(eucl_errors))

    # avg_v_errors, avg_h_errors, avg_eucl_errors, avg_cc_max_values = location_wise_contact_accuracy(pred, gt)
    force_range2d = taxel_projection(force_scaler.data_range_)[0]
    # avg_v_errors, avg_h_errors, avg_eucl_errors, avg_cc_max_values, avg_force_errors, force_errors_scaled = location_wise_contact_accuracy(pred_real, gt_real, force_range2d)
    avg_v_errors, avg_h_errors, avg_eucl_errors, avg_cc_max_values, avg_force_errors, avg_force_errors_scaled_locally, force_errors_scaled_globally, force_errors_scaled_locally = location_wise_contact_accuracy(pred_real, gt_real, force_range2d)

    if len(force_errors_scaled_globally) > 0:
        avg_force_errors_scaled_globally = np.divide(avg_force_errors, force_range2d)
        vis_matrix(avg_v_errors, range(1,11), range(1,11), "Location-wise average contact location errors (Y axis)", f"{result_path}/loc_err_location_wise_y.png")
        vis_matrix(avg_h_errors, range(1,11), range(1,11), "Location-wise average contact location errors (X axis)", f"{result_path}/loc_err_location_wise_x.png")
        vis_matrix(avg_eucl_errors, range(1,11), range(1,11), "Location-wise average contact location errors (Euclidean)", f"{result_path}/loc_err_location_wise_eucl.png")
        vis_matrix(avg_cc_max_values, range(1,11), range(1,11), "Location-wise average heatmap similarity", f"{result_path}/similarity_location_wise.png")
        vis_matrix(avg_force_errors, range(1,11), range(1,11), "Location-wise average force errors (N)", f"{result_path}/force_error_location_wise.png")
        vis_matrix(100*avg_force_errors_scaled_globally, range(1,11), range(1,11), "Location-wise average force errors (%)", f"{result_path}/force_error_location_wise_scaled_globally.png")
        vis_matrix(100*avg_force_errors_scaled_locally, range(1,11), range(1,11), "Location-wise average force errors (%)", f"{result_path}/force_error_location_wise_scaled_locally.png")


        # Measure force accuracy for true positive samples
        force_err, force_err_max, force_err_avg, force_err_med = force_accuracy_on_tps(pred, gt, thres, force_scaler)
        print(f"====== Force Accuracy on True Positives {result_path} ======")
        print(f">> Max error: {force_err_max}N")
        print(f">> Avg error: {force_err_avg}N")
        print(f">> Med error: {force_err_med}N")
        fig = plt.figure(figsize=[2,3])
        # plt.boxplot(force_err)
        plt.hist(force_err,10,color='blue',alpha=0.8)
        plt.xlabel('Force error (N)')
        plt.ylabel('No. of samples')
        plt.draw()
        plt.savefig(f"{result_path}/force_err_tps.png", dpi=400, bbox_inches='tight')
        plt.close()

        #  Measure force accuracy for all cases/non-zero cases
        # err_matrix = abs(pred - gt)
        # gt_true = gt > thres
        # gt_true_inds = gt_true.nonzero()
        # pred_real = force_scaler.inverse_transform(pred)
        # gt_real = force_scaler.inverse_transform(gt)
        err_matrix_real = abs(pred_real - gt_real) 
        err_matrix_real_nonzero = abs(force_errors)
        err_matrix_real_nonzero_scaled_globally = abs(force_errors_scaled_globally)
        err_matrix_real_nonzero_scaled_locally = abs(force_errors_scaled_locally)
        # err_matrix_real_nonzero = abs(pred_real[gt_true_inds] - gt_real[gt_true_inds]) 
        print(f"====== Real Force Accuracy on All Nonzero Taxels {result_path} ======")
        print(f">> Max error: {err_matrix_real_nonzero.max()}N")
        print(f">> Avg error: {np.average(err_matrix_real_nonzero)}N")
        print(f">> Med error: {np.median(err_matrix_real_nonzero)}N")
        print(f"====== Globally Scaled Force Accuracy on All Nonzero Taxels {result_path} ======")
        print(f">> Max error: {err_matrix_real_nonzero_scaled_globally.max()}%")
        print(f">> Avg error: {np.average(err_matrix_real_nonzero_scaled_globally)}%")
        print(f">> Med error: {np.median(err_matrix_real_nonzero_scaled_globally)}%")
        print(f"====== Locally Scaled Force Accuracy on All Nonzero Taxels {result_path} ======")
        print(f">> Max error: {err_matrix_real_nonzero_scaled_locally.max()}%")
        print(f">> Avg error: {np.average(err_matrix_real_nonzero_scaled_locally)}%")
        print(f">> Med error: {np.median(err_matrix_real_nonzero_scaled_locally)}%")
        print(f"====== Force Accuracy on All Taxels {result_path} ======")
        print(f">> Max error: {err_matrix_real.max()}N")
        print(f">> Avg error: {np.average(err_matrix_real)}N")
        print(f">> Med error: {np.median(err_matrix_real)}N")

        fig = plt.figure(figsize=[2,3])
        # fig = plt.figure()
        # plt.boxplot(err_matrix_real)
        plt.hist(err_matrix_real.flatten(),10,color='blue',alpha=0.8)
        plt.xlabel('Force error (N)')
        plt.ylabel('No. of samples')
        plt.draw()
        plt.savefig(f"{result_path}/force_err_all.png", dpi=400, bbox_inches='tight')
        plt.close()
        # fig = plt.figure(figsize=[50,10])
        fig = plt.figure(figsize=[2,3])
        # plt.boxplot(err_matrix_real_nonzero)
        plt.hist(err_matrix_real_nonzero,10,color='blue',alpha=0.8)
        plt.xlabel('Force error (N)')
        plt.ylabel('No. of samples')
        plt.draw()
        plt.savefig(f"{result_path}/force_err_all_nonzero.png", dpi=400, bbox_inches='tight')
        plt.close()
        fig = plt.figure(figsize=[2,3])
        # plt.boxplot(err_matrix_real_nonzero)
        plt.hist(abs(force_errors_scaled_globally*100),10,color='blue',alpha=0.8)
        plt.xlabel('Force error (%)')
        plt.ylabel('No. of samples')
        plt.draw()
        plt.savefig(f"{result_path}/force_err_all_nonzero_scaled_globally.png", dpi=400, bbox_inches='tight')
        plt.close()
        fig = plt.figure(figsize=[2,3])
        # plt.boxplot(err_matrix_real_nonzero)
        plt.hist(abs(force_errors_scaled_locally*100),10,color='blue',alpha=0.8)
        plt.xlabel('Force error (%)')
        plt.ylabel('No. of samples')
        plt.draw()
        plt.savefig(f"{result_path}/force_err_all_nonzero_scaled_locally.png", dpi=400, bbox_inches='tight')
        plt.close()
    
    else:
        print("!!! Not enough force error samples !!!", len(force_errors_scaled_globally))

    # Average error map
    err_nz_avg = location_wise_force_error(pred_real, gt_real)
    vis_matrix(err_nz_avg, range(1,11), range(1,11), "Location-wise average force errors (N)", f"{result_path}/force_err_location_wise_nz.png")

    # Save all evaluation results
    result_pkl_fn = f"{result_path}/all_eval_results.pkl"
    file = open(result_pkl_fn, 'wb')
    pickle.dump([v_errors, h_errors, eucl_errors, cc_max_values, force_errors, #real-range errors for all samples
                 avg_v_errors, avg_h_errors, avg_eucl_errors, avg_cc_max_values, avg_force_errors, #real-range average location wise errors
                 err_matrix_real, err_matrix_real_nonzero, force_err, force_errors_scaled_globally, 
                 avg_force_errors_scaled_globally, avg_force_errors_scaled_locally, force_errors_scaled_locally #Added after 18/07/2023
                 ], file)
    file.close()

def infer_with_minibatch(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              device: torch.device):
    # Put model in eval mode
    model.eval() 

    preds = []
    gts = []
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X = X.to(device, non_blocking=True)
            # y.to(device, non_blocking=True)

            curr_pred = model(X)
            curr_pred = curr_pred.detach().cpu().numpy()
            curr_gt = y.detach().cpu().numpy()

            preds.append(curr_pred)
            gts.append(curr_gt)

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)

    print(">>> Inferred {} samples.".format(preds.shape[0]))

    return preds, gts

def draw_learning_curve(result_path, model_results, faces):
    plt.figure()
    # fig = plt.figure()
    plt.plot(np.array(model_results["test_loss"]), label="test")
    plt.plot(np.array(model_results["train_loss"]), label="train")
    plt.legend()
    plt.draw()
    plt.savefig(f"{result_path}/learning_curve_loss_face{faces}.png", dpi=400, bbox_inches='tight')
    plt.close()

    plt.figure()
    # fig = plt.figure()
    plt.plot(np.array(model_results["test_acc"]), label="test")
    plt.plot(np.array(model_results["train_acc"]), label="train")
    plt.legend()
    plt.draw()
    plt.savefig(f"{result_path}/learning_curve_acc_face{faces}.png", dpi=400, bbox_inches='tight')
    plt.close()

def analyze_raw_hall_and_force_signals(hall_statistics, mAP_loc_wise, hall_scaler, force_scaler, result_path):
    for i in range(len(hall_statistics)):
        curr_hall_statistics = hall_statistics[i]
        curr_result_path = f"{result_path}/selected_ind{i+1}"
        if not os.path.exists(curr_result_path):
            os.makedirs(curr_result_path)
        hall_range2d_scaled_max, force_range2d = analyze_raw_hall_and_force_signals_single_subset(curr_hall_statistics, mAP_loc_wise, hall_scaler, force_scaler, curr_result_path)
        gc.collect()
    # curr_hall_statistics = hall_statistics[1]
    # curr_result_path = result_path
    # hall_range2d_scaled_max, force_range2d = analyze_raw_hall_and_force_signals_single_subset(curr_hall_statistics, mAP_loc_wise, hall_scaler, force_scaler, curr_result_path)

    return hall_range2d_scaled_max, force_range2d

def analyze_raw_hall_and_force_signals_single_subset(hall_statistics, mAP_loc_wise, hall_scaler, force_scaler, result_path):

    print(">>> Hall signals statistics: min({}), max({}), range({})".format(hall_scaler.data_min_, hall_scaler.data_max_, hall_scaler.data_range_))
    print(">>> Force statistics: min({}), max({}), range({})".format(force_scaler.data_min_, force_scaler.data_max_, force_scaler.data_range_))

    # print(hall_statistics)
    [hall_min2d, hall_max2d, hall_range2d, loc_wise_hall_QHull_vol, loc_wise_hall_scaled_QHull_vol] = hall_statistics

    hall_range2d_scaled = np.zeros(hall_range2d.shape)
    for i in range(hall_range2d.shape[0]):
        for j in range(hall_range2d.shape[1]):
            hall_min2d_scaled = hall_scaler.transform(np.expand_dims(hall_min2d[i,j],axis=0))
            hall_max2d_scaled = hall_scaler.transform(np.expand_dims(hall_max2d[i,j],axis=0))
            hall_range2d_scaled[i,j] = hall_max2d_scaled - hall_min2d_scaled
    for i in range(hall_range2d.shape[-1]):
        curr_hall_range2d = hall_range2d[:,:,i]
        curr_hall_range2d_scaled = hall_range2d_scaled[:,:,i]
        vis_matrix(curr_hall_range2d, range(1,11), range(1,11), "Location-wise hall signal range", f"{result_path}/hall_range_loc_wise{i}.png")
        vis_matrix(curr_hall_range2d_scaled, range(1,11), range(1,11), "Location-wise hall signal range", f"{result_path}/hall_range_loc_wise{i}_scaled.png")

    hall_range2d_max = hall_range2d.max(axis=-1)
    hall_range2d_scaled_max = hall_range2d_scaled.max(axis=-1)
    vis_matrix(hall_range2d_max, range(1,11), range(1,11), "Location-wise hall signal range", f"{result_path}/hall_range_loc_wise_max.png")
    vis_matrix(hall_range2d_scaled_max, range(1,11), range(1,11), "Location-wise hall signal range", f"{result_path}/hall_range_loc_wise_scaled_max.png")

    hall_range2d_sum = hall_range2d.sum(axis=-1)
    hall_range2d_scaled_sum = hall_range2d_scaled.sum(axis=-1)
    vis_matrix(hall_range2d_sum, range(1,11), range(1,11), "Location-wise hall signal range", f"{result_path}/hall_range_loc_wise_sum.png")
    vis_matrix(hall_range2d_scaled_sum, range(1,11), range(1,11), "Location-wise hall signal range", f"{result_path}/hall_range_loc_wise_scaled_sum.png")

    hall_range2d_eucl = np.linalg.norm(hall_range2d, axis=-1)
    hall_range2d_scaled_eucl = np.linalg.norm(hall_range2d_scaled, axis=-1)
    vis_matrix(hall_range2d_eucl, range(1,11), range(1,11), "Location-wise hall signal range", f"{result_path}/hall_range_loc_wise_eucl.png")
    vis_matrix(hall_range2d_scaled_eucl, range(1,11), range(1,11), "Location-wise hall signal range", f"{result_path}/hall_range_loc_wise_scaled_eucl.png")

    # hall_range2d_vol = np.prod(hall_range2d, axis=-1, where=(hall_range2d!=0))
    # hall_range2d_scaled_vol = np.prod(hall_range2d_scaled, axis=-1, where=(hall_range2d_scaled!=0))
    vis_matrix(loc_wise_hall_QHull_vol/1e16, range(1,11), range(1,11), "Location-wise hall signal range", f"{result_path}/hall_range_loc_wise_QHullVol.png")
    vis_matrix(loc_wise_hall_scaled_QHull_vol/1e-15, range(1,11), range(1,11), "Location-wise hall signal range", f"{result_path}/hall_range_loc_wise_scaled_QHullVol.png")
    np.save(f"{result_path}/hall_range_loc_wise_QHullVol", loc_wise_hall_QHull_vol)
    np.save(f"{result_path}/hall_range_loc_wise_scaled_QHullVol", loc_wise_hall_scaled_QHull_vol)

    force_range2d = taxel_projection(force_scaler.data_range_)[0]
    # force_range_scaled = force_scaler.transform(np.expand_dims(force_scaler.data_range_, axis=0))
    # force_range_scaled2d = taxel_projection(force_range_scaled)[0]
    vis_matrix(force_range2d, range(1,11), range(1,11), "Location-wise force signal range (N)", f"{result_path}/force_range_loc_wise.png")
    # vis_matrix(force_range_scaled2d, range(1,11), range(1,11), "Location-wise force signal range (N)", f"{result_path}/force_range_loc_wise_scaled.png")

    if mAP_loc_wise is not None:
        np.save(f"{result_path}/mAP_loc_wise", mAP_loc_wise)
        vis_matrix(mAP_loc_wise, range(1,11), range(1,11), "Location-wise mAP", f"{result_path}/mAP_loc_wise.png")

    return hall_range2d_scaled_max, force_range2d

def evaluate_model_single(cfgs, all_data, input_size, output_size, hidden_size, batch_size, device, face, y_min, hall_statistics, selected_ind, curr_whether_non_contact):
    model = MLP(input_size = input_size, hidden_size = hidden_size,
                output_size = output_size).to(device)

    test_data_known_dataloader = DataLoader(all_data, batch_size=batch_size, drop_last=False, shuffle=False)

    result_path = cfgs["result"]["path"]
    # face = map(str, face)
    print(f">>>>>>>> Evaluating Face{face} >>>>>>>>")
    filename = f"{result_path}/finalized_model_face{face}.pt"
    print(">>> Evluating model: ", filename)

    model.load_state_dict(torch.load(filename))
    model.eval()

    pred_test_known, Y_test_known = infer_with_minibatch(model, 
            test_data_known_dataloader, 
            device)

    evaluate_result_path = f"{result_path}/realtest_{face}"
    if not os.path.exists(evaluate_result_path):
        os.makedirs(evaluate_result_path)

    thres = 0.9*y_min

    # Loading the scaler and dataset statistics
    scaler_fn = f"{result_path}/training_hist_face{face}_data_scalers.pkl"
    file = open(scaler_fn, 'rb')
    data = pickle.load(file)
    force_scaler = data[1]
    hall_scaler = data[0]
    mAP_loc_wise = None
    hall_range2d_scaled_max, force_range2d = analyze_raw_hall_and_force_signals(hall_statistics, mAP_loc_wise, hall_scaler, force_scaler, evaluate_result_path)
    # pickle.load([X_scaler, y_scaler, len_test_knowns, len_test_novels], file)
    file.close()

    # # Loading model results
    # training_hist_fn = f"{result_path}/training_hist_face{faces}.pkl"
    # file = open(training_hist_fn, 'rb')
    # model_results = pickle.load(file)
    # # pickle.load([X_scaler, y_scaler, len_test_knowns, len_test_novels], file)
    # file.close()
    # draw_learning_curve(result_path, model_results, faces)

    testset_path = f"{evaluate_result_path}/panda_gripper"
    if not os.path.exists(testset_path):
        os.makedirs(testset_path)

    selected_ind_fn = f"{testset_path}/selected_ind_face{face}.pkl"
    file = open(selected_ind_fn, 'wb')
    pickle.dump(selected_ind, file)
    file.close()
    for j in range(len(selected_ind)):
        print(f">>> Selected {len(selected_ind[j])} samples for selected_ind{j+1}")
    
    # if curr_whether_non_contact:
    #     eval_non_contact_performance(pred_test_known, Y_test_known, testset_path, thres, force_scaler, selected_ind)
    # else:
    #     eval_performance(pred_test_known, Y_test_known, testset_path, thres, force_scaler, selected_ind)

    eval_performance(pred_test_known, Y_test_known, testset_path, thres, force_scaler, selected_ind, curr_whether_non_contact)



def evaluate_models(cfgs):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    # device = "cpu"
    print(f"Using device: {device}") 

    test_data_all_faces, input_sizes, output_sizes, y_mins, all_hall_statistics, selected_inds, whether_non_contact = load_data_real_test(cfgs)

    hidden_size = cfgs["model"]["hidden_size"]
    batch_size = cfgs["evaluate"]["batch_size"]

    faces = cfgs["faces"]
    non_cfgs = cfgs["non_contact_info"]
    non_faces = non_cfgs["faces"]
    all_faces = faces + non_faces
    n_faces = len(all_faces)
    for ii in range(n_faces):
        all_data = test_data_all_faces[ii]
        input_size = input_sizes[ii]
        output_size = output_sizes[ii]
        y_min = y_mins[ii]
        hall_statistics = all_hall_statistics[ii]
        selected_ind = selected_inds[ii]
        curr_whether_non_contact = whether_non_contact[ii]
        face = all_faces[ii]

        evaluate_model_single(cfgs, all_data, input_size, output_size, hidden_size, batch_size, device, face, y_min, hall_statistics, selected_ind, curr_whether_non_contact)
        gc.collect()

    print("!!! Evaluation Done !!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', '-s', default='./configs/default.yaml', type=str, help='config file')
    args = parser.parse_args()

    config_file_fn = args.config_file
    cfgs = yaml.load(open(config_file_fn), Loader=yaml.FullLoader)
    print(cfgs)

    evaluate_models(cfgs)