#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
from tqdm.auto import tqdm
import yaml
import pickle
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import MLP, LinearReg
from utils import load_data

y_min = 0.0001
def eval_heat_map_acc(pred, y):
    global y_min
    # Extract position information from heatmap
    pred = pred.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    pred_bool = pred > (0.9*y_min)
    y_bool = y > (0.9*y_min)
    train_acc = (pred_bool == y_bool).sum()/y_bool.size
    
    return train_acc

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        train_acc += eval_heat_map_acc(y_pred, y)
        
    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device):
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            test_pred = model(X)
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_acc += eval_heat_map_acc(test_pred, y)
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          cfgs):
    
    print(f"[INFO] Training model {model.__class__.__name__} on device '{device}' for {epochs} epochs...")
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    result_path = cfgs["result"]["path"]
    faces = cfgs["dataset"][0]["faces"]
    faces = map(str, faces)
    faces = ''.join(faces)
    hist_model_path = f"{result_path}/hist_models_face{faces}"
    if not os.path.exists(hist_model_path):
        os.makedirs(hist_model_path)
    
    # 3. Loop through training and testing steps for a number of epochs
    #     for epoch in range(epochs):
    for epoch in tqdm(range(epochs)):
        # Do eval before training (to see if there's any errors)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device)
        
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if epoch % 200 == 0:
            filename = f"{hist_model_path}/model_face{faces}_{epoch}.pt"
            torch.save(model.state_dict(), filename)

    filename = f"{result_path}/finalized_model_face{faces}.pt"
    print(">>> Saving final model >>> ", filename)
    torch.save(model.state_dict(), filename)
    return results


def print_train_time(start, end, device=None, machine="Mac Pro M1"):
    """Prints difference between start and end time.
    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    if device:  
        print(f"\nTrain time on {machine} using PyTorch device {device}: {total_time:.3f} seconds\n")
    else:
        print(f"\nTrain time: {total_time:.3f} seconds\n")
    return round(total_time, 3)


def run(cfgs):
    global y_min

    # Create device list
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    # device = "cpu"
    print(f"Using device: {device}") 

    # Create result folder
    result_path = cfgs["result"]["path"]
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    faces = cfgs["dataset"][0]["faces"]
    faces = map(str, faces)
    faces = ''.join(faces)
    training_data, val_data, test_data_known, test_data_novel, input_size, output_size, y_min, hall_statistics, mAP_loc_wise = load_data(cfgs)

    # Set random seed
    seed = cfgs["optimizer"]["seed"]
    torch.manual_seed(seed)

    hidden_size = cfgs["model"]["hidden_size"]
    num_epoch = cfgs["train"]["epochs"]
    batch_size = cfgs["train"]["batch_size"]
    learning_rate = cfgs["optimizer"]["initial_lr"]   

    train_dataloader = DataLoader(training_data, batch_size=batch_size, drop_last=False, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, drop_last=False, shuffle=True)  
    
    model = LinearReg(input_size = input_size, output_size = output_size).to(device)
    
    # Setup loss function and optimizer
    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(params=model.parameters(), lr = learning_rate, weight_decay=1e-4)

    loss_fn = loss_fn.to(device)

    # Start the timer
    from timeit import default_timer as timer 
    start_time = timer()

    # Train model
    print(">>> Fitting model >>> ")
    model_results = train(model=model, 
                        train_dataloader=train_dataloader,
                        test_dataloader=val_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=num_epoch,
                        device=device,
                        cfgs=cfgs)

    # End the timer
    end_time = timer()

    # Print out timer and results
    total_train_time = print_train_time(start=start_time,
                                        end=end_time,
                                        device=device)
    
    # Create results dict
    results = {
        "machine": "Mac Pro M1",
        "device": device,
        "epochs": num_epoch,
        "batch_size": batch_size,
        "input_dim": training_data.shape[1],
        "num_train_samples": len(training_data),
        "num_test_samples": len(val_data),
        "total_train_time": round(total_train_time, 3),
        "time_per_epoch": round(total_train_time/num_epoch, 3),
        "model": model.__class__.__name__,
        "test_accuracy": model_results["test_acc"][-1]
        }
    
    # Write CSV to file
    results_df = pd.DataFrame(results, index=[0])
    results_df.to_csv(f"{result_path}/mac_pro_m1_{device}_face{faces}.csv", 
                    index=False)

    # Saving training history
    training_hist_fn = f"{result_path}/training_hist_face{faces}.pkl"
    file = open(training_hist_fn, 'wb')
    pickle.dump(model_results, file)
    file.close()

    print("!!! Training Done !!!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', '-s', default='./configs/default.yaml', type=str, help='config file')
    args = parser.parse_args()

    config_file_fn = args.config_file
    cfgs = yaml.load(open(config_file_fn), Loader=yaml.FullLoader)
    print(cfgs)

    run(cfgs)
