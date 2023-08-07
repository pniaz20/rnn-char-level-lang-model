# models.py
import sys
import os
import warnings
from pathlib import Path
import math
import json
from timeit import default_timer as timer
from datetime import datetime
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
from torch.nn.init import xavier_uniform_, zeros_, orthogonal_
from sklearn.metrics import r2_score
import numpy as np
import collections

########################################################################################################################
# Global variables, functions, and classes
########################################################################################################################




def make_path(path:str):
    ''' Make a path if it doesn't exist.'''
    Path.mkdir(Path(path).parent, parents=True, exist_ok=True)
    return path


def autoname(name):
    """
    Genereate a unique name for a file, based on the current time and the given name.
    Gets the `name` as a string and adds the time stamp to the end of it before returning it.
    """
    return name + "_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def equalize_string(string, seqlen):
    return (string + " " * (seqlen - len(string))) if len(string) < seqlen else string[:seqlen]


def get_tensor_from_exs(exs, seqlen, vocab_index):
    return torch.tensor([[vocab_index.index_of(ch) for ch in equalize_string(sentence,seqlen)] \
        for sentence in exs], dtype=torch.long, requires_grad=False)


actdict_pytorch = {
    'relu':nn.ReLU, 'leakyrelu':nn.LeakyReLU, 'sigmoid':nn.Sigmoid, 'tanh':nn.Tanh,
    'softmax':nn.Softmax, 'logsoftmax':nn.LogSoftmax,
    'softplus':nn.Softplus, 'softshrink':nn.Softshrink,
    'elu':nn.ELU, 'selu':nn.SELU, 'softsign':nn.Softsign, 'softmin':nn.Softmin,
    'softmax2d':nn.Softmax2d}

lossdict_pytorch = {
    "mse":nn.MSELoss, "crossentropy":nn.CrossEntropyLoss, "binary_crossentropy":nn.BCELoss,
    "categorical_crossentropy":nn.CrossEntropyLoss, "nll":nn.NLLLoss, "poisson":nn.PoissonNLLLoss,
    "kld":nn.KLDivLoss, "hinge":nn.HingeEmbeddingLoss, "l1":nn.L1Loss,
    "mae": nn.L1Loss, "l2":nn.MSELoss, "smoothl1":nn.SmoothL1Loss, "bce_with_logits":nn.BCEWithLogitsLoss
}
optdict_pytorch = {'adam':optim.Adam, 'sgd':optim.SGD, 'rmsprop':optim.RMSprop}


def generate_sample_batch(model):
    x = np.random.rand(*model.batch_input_shape).astype(np.float32)
    y = np.random.rand(*model.batch_output_shape).astype(np.float32)
    return (x,y)


def test_pytorch_model(model_class):
    print("Constructing model...\n")
    model = model_class()
    print("Summary of model:")
    print(model)
    print("\nGenerating random dataset...\n")
    (x_train, y_train) = generate_sample_batch(model)
    (x_val, y_val) = generate_sample_batch(model)
    x_train_t = torch.Tensor(x_train)
    y_train_t = torch.Tensor(y_train)
    x_val_t = torch.Tensor(x_val)
    y_val_t = torch.Tensor(y_val)
    trainset = TensorDataset(x_train_t, y_train_t)
    validset = TensorDataset(x_val_t, y_val_t)
    dataset = (trainset, validset)
    print("\nTraining model...\n")
    model.train_model(dataset, verbose=True, script_before_save=True, saveto="dummy_%s.pt"%model.hparams["model_name"])
    print("\nEvaluating model...\n")
    print("Done.")
    
    

def train_pytorch_model(model, dataset, batch_size:int, loss_str:str, optimizer_str:str, 
    optimizer_params:dict=None, loss_function_params:dict=None, learnrate:float=0.001, 
    learnrate_decay_gamma:float=None, epochs:int=10, validation_patience:int=10000, validation_data:float=0.1, 
    verbose:bool=True, script_before_save:bool=True, saveto:str=None, num_workers=0, permute_before_loss=None):
    """Train a Pytorch model, given some hyperparameters.

    ### Args:
        - `model` (`torch.nn`): A torch.nn model
        - `dataset` (`torch.utils.data.Dataset`): Dataset object to be used
        - `batch_size` (int): Batch size
        - `loss_str` (str): Loss function to be used. Options are "mse", "mae", "crossentropy", etc.
        - `optimizer_str` (str): Optimizer to be used. Options are "sgd", "adam", "rmsprop", etc.
        - `optimizer_params` (dict, optional): Parameters for the optimizer.
        - `loss_function_params` (dict, optional): Parameters for the loss function.
        - `learnrate` (float, optional): Learning rate. Defaults to 0.001.
        - `learnrate_decay_gamma` (float, optional): Learning rate exponential decay rate. Defaults to None.
        - `epochs` (int, optional): Number of epochs. Defaults to 10.
        - `validation_patience` (int, optional): Number of epochs to wait before stopping training. Defaults to 10000.
        - `validation_data` (float, optional): Fraction of the dataset to be used for validation. Defaults to 0.1.
        - `verbose` (bool, optional): Whether to print progress. Defaults to True.
        - `script_before_save` (bool, optional): Use TorchScript for serializing the model. Defaults to True.
        - `saveto` (str, optional): Save PyTorch model in path. Defaults to None.
        - `num_workers` (int, optional): Number of workers for the dataloader. Defaults to 0.
        

    ### Returns:
        - `model`: Trained PyTorch-compatible model
        - `history`: PyTorch model history dictionary, containing the following keys:
            - `training_loss`: List containing training loss values of epochs.
            - `validation_loss`: List containing validation loss values of epochs.
            - `learning_rate`: List containing learning rate values of epochs.
    """

    hist_training_loss = []
    hist_validation_loss = []
    hist_learning_rate = []
    hist_trn_metric = []
    hist_val_metric = []

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    if "list" in type(dataset).__name__ or "tuple" in type(dataset).__name__:
        assert len(dataset)==2, "If dataset is a tuple, it must have only two elements, "+\
            "the training dataset and the validation dataset."
        trainset, valset = dataset
        num_val_data = int(len(valset))
        num_train_data = int(len(trainset))
        num_all_data = num_train_data + num_val_data
    else:
        num_all_data = len(dataset)
        num_val_data = int(validation_data*num_all_data)
        num_train_data = num_all_data - num_val_data
        (trainset, valset) = random_split(dataset, (num_train_data, num_val_data), 
            generator=torch.Generator().manual_seed(SEED))

    
    if verbose:
        print("Total number of data points:      %d"%num_all_data)
        print("Number of training data points:   %d"%num_train_data)
        print("Number of validation data points: %d"%num_val_data)
        
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    
    
    if verbose:
        print("Number of training batches:    %d"%len(trainloader))
        print("Number of validation batches:  %d"%len(validloader))
        print("Batch size:                    %d"%batch_size)
        for x,y in trainloader:
            print("Shape of x_train:", x.shape)
            print("Shape of y_train:", y.shape)
            break
        for x,y in validloader:
            print("Shape of x_val:", x.shape)
            print("Shape of y_val:", y.shape)
            break

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("selected device: ", device)
    
    model.to(device)
    
    loss_func = lossdict_pytorch[loss_str]
    if loss_function_params:
        criterion = loss_func(**loss_function_params)
    else:
        criterion = loss_func()
        
    
    optimizer_func = optdict_pytorch[optimizer_str]
    if optimizer_params:
        optimizer = optimizer_func(model.parameters(), lr=learnrate, **optimizer_params)
    else:
        optimizer = optimizer_func(model.parameters(), lr=learnrate)

    
    if learnrate_decay_gamma:
        if verbose:
            print("The learning rate has an exponential decay rate of %.5f."%learnrate_decay_gamma)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=learnrate_decay_gamma)
        lr_sch = True
    else:
        lr_sch = False
    
    # Find out if we're going to display any metric along with the loss or not.
    display_metrics = True
    classification = False
    regression = False
    if loss_str in ["binary_crossentropy", "bce_with_logits", "nll", "crossentropy", 
                    "categorical_crossentropy"]:
        classification = True
        regression = False
        trn_metric_name = "Acc"
        val_metric_name = "Val Acc"
    elif loss_str in ["mse", "l1", "l2", "mae"]:
        classification = False
        regression = True
        trn_metric_name = "R2"
        val_metric_name = "Val R2"
    else:
        classification = False
        regression = False
        display_metrics = False
    if verbose:
        if classification:
            print("Classification problem detected. We will look at accuracies.")
        elif regression:
            print("Regression problem detected. We will look at R2 scores.")
        else:
            print("We have neither classification nor regression problem. No metric will be displayed.")
    
                    
    # Preparing training loop
    num_training_batches = len(trainloader)
    num_validation_batches = len(validloader)
    
    
    progress_bar_size = 40
    ch = "█"
    intvl = num_training_batches/progress_bar_size;
    valtol = validation_patience if validation_patience else 10000
    minvalerr = 1000000.0
    badvalcount = 0
    
    # Commencing training loop
    tStart = timer()
    for epoch in range(epochs):
        
        tEpochStart = timer()
        epoch_loss_training = 0.0
        epoch_loss_validation = 0.0
        newnum = 0
        oldnum = 0
        trn_metric = 0.0
        val_metric = 0.0
        num_train_logits = 0
        num_val_logits = 0
    
        if verbose and epoch > 0: print("Epoch %3d/%3d ["%(epoch+1, epochs), end="")
        if verbose and epoch ==0: print("First epoch ...")
        
        ##########################################################################
        # Training
        if verbose and epoch==0: print("\nTraining phase ...")
        model.train()
        for i, data in enumerate(trainloader):
            
            # Fetching data
            seqs, targets = data[0].to(device), data[1].to(device)
            
            # Forward propagation
            predictions = model(seqs)
            
            # Testing shapes
            if classification:
                assert len(predictions.shape) == len(targets.shape) or len(predictions.shape)==len(targets.shape)+1, \
                    "Target dimensionality must be equal to or one less than the prediction dimensionality.\n"+\
                    "Target shape: %s, Prediction shape: %s\n"%(str(targets.shape), str(predictions.shape))+\
                    "If targets are class indices, they must be of shape (N,), (N,1) or (N, d1, ..., dm). "+\
                    "Otherwise, they must be (N, K) or (N, K, d1, ..., dm). "+\
                    "Predictions must be (N, K) or (N, K, d1, ..., dm).\n"+\
                    "N is batch size, K is number of classes and d1 to dm are dimensionalities of classification."
                if len(predictions.shape) > 1:
                    assert predictions.shape[0] == targets.shape[0], \
                        "Batch size of targets and predictions must be the same.\n"+\
                        "Target shape: %s, Prediction shape: %s\n"%(str(targets.shape), str(predictions.shape))
                    if len(targets.shape) > 1 and len(predictions.shape) == len(targets.shape):
                        assert predictions.shape[1] == targets.shape[1] or targets.shape[1] == 1, \
                            "Number of classes of targets and predictions must be the same.\n"+\
                            "Target shape: %s, Prediction shape: %s\n"%(str(targets.shape), str(predictions.shape))
                    if len(targets.shape) > 2 and len(predictions.shape) == len(targets.shape):
                        assert predictions.shape[2:] == targets.shape[2:], \
                            "Dimensionality of targets and predictions must be the same beyond the second dim.\n"+\
                            "Target shape: %s, Prediction shape: %s\n"%(str(targets.shape), str(predictions.shape))
            else:
                assert predictions.shape == targets.shape, \
                    "Target shape must be equal to the prediction shape.\n"+\
                    "Target shape: %s, Prediction shape: %s\n"%(str(targets.shape), str(predictions.shape))
                    
            
            # Loss calculation and accumulation
            loss = criterion(predictions, targets)
            epoch_loss_training += loss.item()
            
            # Backpropagation and optimizer update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics calculation
            if display_metrics:
                with torch.no_grad():
                    if loss_str == "binary_crossentropy":
                        # Output layer already includes sigmoid.
                        class_predictions = (predictions > 0.5).int()
                    elif loss_str == "bce_with_logits":
                        # Output layer does not include sigmoid. Sigmoid is a part of the loss function.
                        class_predictions = (torch.sigmoid(predictions) > 0.5).int()
                    elif loss_str == "nll":
                        # Output layer already includes log_softmax.
                        class_predictions = torch.argmax(predictions, dim=1)
                    elif loss_str in ["crossentropy", "categorical_crossentropy"]:
                        # Output layer does not have log_softmax. It is implemented as a part of the loss function.
                        class_predictions = torch.argmax(torch.log_softmax(predictions, dim=1), dim=1)#, keepdim=True)   

                    if classification:
                        if verbose and i==0 and epoch ==0: 
                            print("Shape of model outputs:     ", predictions.shape)
                            print("Shape of class predictions: ", class_predictions.shape)
                            print("Shape of targets:           ", targets.shape)
                        
                        # Calculate accuracy
                        correct = (class_predictions == targets).int().sum().item()
                        num_train_logits += targets.numel()
                        trn_metric += correct
                        if verbose and epoch==0: 
                            print("Number of correct answers (this batch - total): %10d - %10d"%(correct, trn_metric))
                        
                        # Calculate F1 score
                        # f1 = f1_score(targets.cpu().numpy(), class_predictions.cpu().numpy(), average="macro")
                        # Calculate ROC AUC
                        # auc = roc_auc_score(targets.cpu().numpy(), class_predictions.cpu().numpy(), average="macro")
                    elif regression:
                        if verbose and i==0 and epoch==0: 
                            print("Shape of predictions: ", predictions.shape)
                            print("Shape of targets:     ", targets.shape)
                        # Calculate r2_score
                        trn_metric += r2_score(targets.cpu().numpy(), predictions.cpu().numpy())
                    
                    
            # Visualization of progressbar
            if verbose and epoch > 0:
                newnum = int(i/intvl)
                if newnum > oldnum:
                    print((newnum-oldnum)*ch, end="")
                    oldnum = newnum 
        
        # Update learning rate if necessary
        if lr_sch:
            scheduler.step()
        
        # Calculate training epoch loss
        epoch_loss_training /= num_training_batches
        if verbose and epoch==0: print("Epoch loss (training): %.5f"%epoch_loss_training)
        hist_training_loss.append(epoch_loss_training)
        
        # Calculate training epoch metric (accuracy or r2-score)
        if display_metrics:
            if classification:
                trn_metric /= num_train_logits
            else:
                trn_metric /= num_training_batches
            if verbose and epoch==0: print("Epoch metric (training): %.5f"%trn_metric)
            hist_trn_metric.append(trn_metric)
        
        
        if verbose and epoch > 0: print("] ", end="")
        
           
        ##########################################################################
        # Validation
        if verbose and epoch==0: print("\nValidation phase ...")
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(validloader):
                
                seqs, targets = data[0].to(device), data[1].to(device)
                predictions = model(seqs)
                loss = criterion(predictions, targets)
                epoch_loss_validation += loss.item()
                
                # Do prediction for metrics
                if display_metrics:
                    if loss_str == "binary_crossentropy":
                        # Output layer already includes sigmoid.
                        class_predictions = (predictions > 0.5).int()
                    elif loss_str == "bce_with_logits":
                        # Output layer does not include sigmoid. Sigmoid is a part of the loss function.
                        class_predictions = (torch.sigmoid(predictions) > 0.5).int()
                    elif loss_str == "nll":
                        # Output layer already includes log_softmax.
                        class_predictions = torch.argmax(predictions, dim=1)
                    elif loss_str in ["crossentropy", "categorical_crossentropy"]:
                        # Output layer does not have log_softmax. It is implemented as a part of the loss function.
                        class_predictions = \
                            torch.argmax(torch.log_softmax(predictions, dim=1), dim=1)#, keepdim=True).float()    
                
                    if classification:
                        if verbose and i==0 and epoch ==0: 
                            print("Shape of model outputs:     ", predictions.shape)
                            print("Shape of class predictions: ", class_predictions.shape)
                            print("Shape of targets:           ", targets.shape)
                        # Calculate accuracy
                        correct = (class_predictions == targets).int().sum().item()
                        num_val_logits += targets.numel()
                        val_metric += correct
                        if verbose and epoch==0: 
                            print("Number of correct answers (this batch - total): %10d - %10d"%(correct, val_metric))
                        # Calculate F1 score
                        # f1 = f1_score(targets.cpu().numpy(), class_predictions.cpu().numpy(), average="macro")
                        # Calculate ROC AUC
                        # auc = roc_auc_score(targets.cpu().numpy(), class_predictions.cpu().numpy(), average="macro")
                    elif regression:
                        if verbose and i==0 and epoch==0: 
                            print("Shape of predictions: ", predictions.shape)
                            print("Shape of targets:     ", targets.shape)
                        # Calculate r2_score
                        val_metric += r2_score(targets.cpu().numpy(), predictions.cpu().numpy())
        
        # Calculate epoch validation loss            
        epoch_loss_validation /= num_validation_batches
        if verbose and epoch==0: print("Epoch loss (validation): %.5f"%epoch_loss_validation)
        hist_validation_loss.append(epoch_loss_validation)
        
        # Calculate epoch validation metric (accuracy or r2-score)        
        if display_metrics:
            if classification:
                val_metric /= num_val_logits
            else:
                val_metric /= num_validation_batches
            if verbose and epoch==0: print("Epoch metric (validation): %.5f"%val_metric)
            hist_val_metric.append(val_metric)
        
        # Log the learning rate, if there is any scheduling.
        if lr_sch:
            hist_learning_rate.append(scheduler.get_last_lr())
        else:
            hist_learning_rate.append(learnrate)
        
        
        ##########################################################################
        # Post Processing Training Loop            
        tEpochEnd = timer()
        if verbose:
            if display_metrics:
                print("Loss: %5.4f |Val Loss: %5.4f |%s: %5.4f |%s: %5.4f | %6.3f s" % (
                    epoch_loss_training, epoch_loss_validation, trn_metric_name, trn_metric,
                    val_metric_name, val_metric, tEpochEnd-tEpochStart))
            else:
                print("Loss: %5.4f |Val Loss: %5.4f | %6.3f s" % (
                    epoch_loss_training, 
                    epoch_loss_validation, tEpochEnd-tEpochStart))
        

        # Checking for early stopping
        if epoch_loss_validation < minvalerr:
            minvalerr = epoch_loss_validation
            badvalcount = 0
        else:
            badvalcount += 1
            if badvalcount > valtol:
                if verbose:
                    print("Validation loss not improved for more than %d epochs."%badvalcount)
                    print("Early stopping criterion with validation loss has been reached. " + 
                        "Stopping training at %d epochs..."%epoch)
                break
    # End for loop
    
    
    ##########################################################################
    # Epilogue
    tFinish = timer()
    if verbose:        
        print('Finished Training.')
        print("Training process took %.2f seconds."%(tFinish-tStart))
    if saveto:
        try:
            if verbose: print("Saving model...")
            if script_before_save:
                example,_ = next(iter(trainloader))
                example = example[0,:].unsqueeze(0)
                model.cpu()
                with torch.no_grad():
                    traced = torch.jit.trace(model, example)
                    traced.save(saveto)
            else:
                with torch.no_grad():
                    torch.save(model, saveto)
        except Exception as e:
            if verbose:
                print(e)
                print("Failed to save the model.")
        if verbose: print("Done Saving.")
        
    torch.cuda.empty_cache()
    
    history = {
        'training_loss':hist_training_loss, 
        'validation_loss':hist_validation_loss, 
        'learning_rate':hist_learning_rate}
    if display_metrics:
        history["training_metrics"] = hist_trn_metric
        history["validation_metrics"] = hist_val_metric
    if verbose: print("Done training.")
    
    return history








def evaluate_pytorch_model(model, dataset, loss_str:str, loss_function_params:dict=None,
    batch_size:int=16, device_str:str="cuda", verbose:bool=True):
    """
    Evaluates a PyTorch model on a dataset.
    
    ### Parameters
    
    `model` (`torch.nn.Module`): The model to evaluate.
    `dataset` (`torch.utils.data.Dataset`): The dataset to evaluate the model on.
    `loss_str` (str): The loss function to use when evaluating the model.
    `loss_function_params` (dict, optional) : Parameters to pass to the loss function.
    `batch_size` (int, optional) : The batch size to use when evaluating the model. Defaults to 16.
    `device_str` (str, optional) : The device to use when evaluating the model. Defaults to "cuda".
    `verbose` (bool, optional) : Whether to print out the evaluation metrics. Defaults to True.
    
    
    ### Returns
    
    A dictionary containing the evaluation metrics, including "loss" and "metrics" in case any metric is available.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if verbose: print("Preparing data...")
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    num_batches = len(testloader)
    num_val_data = int(len(dataset))
    
    if "cuda" in device_str:
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    print("selected device: ", device)
    model.to(device)
    
    loss_func = lossdict_pytorch[loss_str]
    if loss_function_params:
        criterion = loss_func(**loss_function_params)
    else:
        criterion = loss_func()
    
    
    display_metrics = True
    classification = False
    regression = False
    if loss_str in ["binary_crossentropy", "bce_with_logits", "nll", "crossentropy", "categorical_crossentropy"]:
        classification = True
        regression = False
        metric_name = "Accuracy"
    elif loss_str in ["mse", "l1", "l2", "mae"]:
        classification = False
        regression = True
        metric_name = "R2-Score"
    else:
        classification = False
        regression = False
        display_metrics = False
        
    progress_bar_size = 20
    ch = "█"
    intvl = num_batches/progress_bar_size;
    if verbose: print("Evaluating model...")
    model.eval()
    newnum = 0
    oldnum = 0
    totloss = 0.0
    if verbose: print("[", end="")
    val_metric = 0.0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs, targets = data[0].to(device), data[1].to(device)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            totloss += loss.item()
            
            # Do prediction for metrics
            if display_metrics:
                if loss_str == "binary_crossentropy":
                    # Output layer already includes sigmoid.
                    class_predictions = (predictions > 0.5).float()
                elif loss_str == "bce_with_logits":
                    # Output layer does not include sigmoid. Sigmoid is a part of the loss function.
                    class_predictions = (torch.sigmoid(predictions) > 0.5).float()
                elif loss_str == "nll":
                    # Output layer already includes log_softmax.
                    class_predictions = torch.argmax(predictions, dim=1, keepdim=True).float()
                elif loss_str in ["crossentropy", "categorical_crossentropy"]:
                    # Output layer does not have log_softmax. It is implemented as a part of the loss function.
                    class_predictions = \
                        torch.argmax(torch.log_softmax(predictions, dim=1), dim=1)#, keepdim=True).float()    
            
                if classification:
                    # Calculate accuracy
                    correct = (class_predictions == targets).float().sum().item()
                    val_metric += correct
                    # Calculate F1 score
                    # f1 = f1_score(targets.cpu().numpy(), class_predictions.cpu().numpy(), average="macro")
                    # Calculate ROC AUC
                    # auc = roc_auc_score(targets.cpu().numpy(), class_predictions.cpu().numpy(), average="macro")
                elif regression:
                    # Calculate r2_score
                    val_metric += r2_score(targets.cpu().numpy(), predictions.cpu().numpy())
                    
            # Visualization of progressbar
            if verbose:
                newnum = int(i/intvl)
                if newnum > oldnum:
                    print((newnum-oldnum)*ch, end="")
                    oldnum = newnum 
    
    
    if display_metrics:
            if classification:
                val_metric /= num_val_data
            else:
                val_metric /= num_batches
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
                        
    if verbose: print("] ", end="")
    
                
    totloss /= num_batches
    if verbose:
        if display_metrics:
            print("Loss: %5.4f | %s: %5.4f" % (totloss, metric_name, val_metric))
        else:
            print("Loss: %5.4f" % totloss)
            
    if verbose: print("Done.")
    
    d = {"loss":totloss}
    if display_metrics:
        d["metrics"] = val_metric
    
    return d
            
            

################################
def predict_pytorch_model(model, dataset, loss_str:str, batch_size:int=16, device_str:str="cuda", 
    return_in_batches:bool=True, return_inputs:bool=False, return_raw_predictions:bool=False, verbose:bool=True):
    """
    Predicts the output of a pytorch model on a given dataset.

    ### Args:
        - `model` (`torch.nn.Module`): The PyTorch model to use.
        - `dataset` (`torch.utils.data.Dataset`): Dataset containing the input data
        - `loss_str` (str): Loss function used when training. 
            Used only for determining whether a classification or a regression model is used.
        - `batch_size` (int, optional): Batch size to use when evaluating the model. Defaults to 16.
        - `device_str` (str, optional): Device to use when performing inference. Defaults to "cuda".
        - `return_in_batches` (bool, optional): Whether the predictions should be batch-separated. Defaults to True.
        - `return_inputs` (bool, optional): Whether the output should include the inputs as well. Defaults to False.
        - `return_raw_predictions` (bool, optional): Whether raw predictions should also be returned. Defaults to False.
        - `verbose` (bool, optional): Verbosity of the function. Defaults to True.

    ### Returns:
        List: A List containing the output predictions, and optionally, the inputs and raw predictions.
        
    ### Notes:
        - If `return_in_batches` is True, the output will be a list of lists. output[i] contains the i'th batch.
        - If `return_inputs` is true, the first element of the output information will be the inputs.
        - If `return_raw_predictions` is true, the second element of the output information will be the raw predictions.
            Please note that this is only meaningful for classification problems. Otherwise, predictions will only
            include raw predictions. For classification problems, if this setting is True, the third element of the
            output information will be the class predictions.
        - "output information" here is a list containing [input, raw_predictions, class_predictions].
            For non-classification problems, "output information" will only contain [input, raw_predictions].
            If `return_inputs` is False, the first element of the output information will be omitted; [raw_predictions].
            If `return_in_batches` is True, the output will be a list of "output information" for every batch.
            Otherwise, the output will be one "output information" for the whole dataset.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if verbose: print("Preparing data...")
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    num_batches = len(testloader)
    
    if "cuda" in device_str:
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    print("selected device: ", device)
    model.to(device)
    
    
    if loss_str in ["binary_crossentropy", "bce_with_logits", "nll", "crossentropy", "categorical_crossentropy"]:
        classification = True
    else:
        classification = False
    
    
    
    output_list = []
    
        
    progress_bar_size = 20
    ch = "█"
    intvl = num_batches/progress_bar_size;
    if verbose: print("Performing Prediction...")
    model.eval()
    newnum = 0
    oldnum = 0
    if verbose: print("[", end="")
    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs = data[0].to(device)
            predictions = model(inputs)
            
            # Do prediction
            if classification:
                if loss_str == "binary_crossentropy":
                    # Output layer already includes sigmoid.
                    class_predictions = (predictions > 0.5).float()
                elif loss_str == "bce_with_logits":
                    # Output layer does not include sigmoid. Sigmoid is a part of the loss function.
                    class_predictions = (torch.sigmoid(predictions) > 0.5).float()
                elif loss_str == "nll":
                    # Output layer already includes log_softmax.
                    class_predictions = torch.argmax(predictions, dim=1, keepdim=True).float()
                elif loss_str in ["crossentropy", "categorical_crossentropy"]:
                    # Output layer does not have log_softmax. It is implemented as a part of the loss function.
                    class_predictions = \
                        torch.argmax(torch.log_softmax(predictions, dim=1), dim=1)#, keepdim=True).float()    
            
            
            # Add batch predictions to output dataset
            obatch = []
            if return_inputs:
                obatch.append(inputs.cpu().numpy())
            if classification:
                if return_raw_predictions:
                    obatch.append(predictions.cpu().numpy())
                obatch.append(class_predictions.cpu().numpy())
            else:
                obatch.append(predictions.cpu().numpy())
                
            if return_in_batches:
                output_list.append(obatch)
            elif i==0:
                output_array = obatch
            else:
                for j in range(len(obatch)):
                    output_array[j] = np.append(output_array[j], obatch[j], axis=0)
            
              
            # Visualization of progressbar
            if verbose:
                newnum = int(i/intvl)
                if newnum > oldnum:
                    print((newnum-oldnum)*ch, end="")
                    oldnum = newnum 
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
                        
    if verbose: print("] ")
    
    if return_in_batches:
        return output_list
    else:
        return output_array
        
                

class Pytorch_Seq2Dense(nn.Module):
    
    sample_hparams = {
        'model_name': 'dummy_Pytorch_Seq2Dense',
        'in_features': 10,
        'out_features': 3,
        'in_seq_len': 13,
        'out_seq_len': 6,
        'rnn_type': 'LSTM',
        'rnn_hidden_sizes': 8,
        'rnn_bidirectional': False,
        'rnn_depth': 2,
        'rnn_dropout': 0.1,
        'lstm_projsize': None,
        'final_rnn_return_sequences': False,
        'apply_dense_for_each_time_step': True,
        'dense_width': 16,
        'dense_depth': 2,
        'dense_dropout': 0.2,
        'dense_activation': 'relu',
        'dense_activation_params': None,
        'output_activation': None,
        'output_activation_params': None,
        'batchnorm': "before",
        'batchnorm_params': None,
        'l2_reg': 0.0001,
        'batch_size': 16,
        'epochs': 2,
        'validation_data': 0.1,
        'validation_tolerance_epochs': 10,
        'learning_rate': 0.0001,
        'learning_rate_decay_gamma': 0.99,
        'loss_function': 'mse',
        'loss_function_params': None,
        'optimizer': 'adam',
        'optimizer_params': {'eps': 1e-07}
    }
    
    
    def __init__(self, hparams:dict=None):
        """Sequence to Dense network with RNN for time-series classification, regression, and forecasting.
        This network uses any RNN layers as encoders to extract information from input sequences, and fully-connected 
        multilayer perceptrons (Dense) to decode the sequence into an output, which can be class probabilitites 
        (timeseries classification), a continuous number (regression), or an unfolded sequence (forecasting) of a 
        target timeseries.

        ### Usage

        `net = Pytorch_Seq2Dense(hparams)` where `hparams` is dictionary of hyperparameters containing the following:

            - `rnn_type` (str): RNN type, options are "LSTM", "GRU", "RNN".

            - `in_seq_len` (int): Input sequence length, in number of timesteps
            - `out_seq_len` (int): Output sequence length, in number of timesteps, assuming output is also a sequence.
            Use 1 for when the output is not a sequence, or do not supply this key.

            - `in_features` (int): Number of features of the input.
            - `out_features` (int): Number of features of the output.

            - `rnn_hidden_sizes` ("auto"|int|list|array): RNN layer sizes. "auto" decides automatically, 
                a number sets them all the same, and a list/array sets each RNN layer differently (stacked RNN).
            - `rnn_bidirectional` (bool): Whether the RNN layers are bidirectional or not.
            - `rnn_depth` (int): Number of stacked RNN layers. Default is 1.

            - `rnn_dropout` (float): Dropout rate, if any, of the RNN layers. 
                PyTorch ignores this if there is only one RNN layer.

            - `lstm_projsize` (int): Projected output size of the LSTM (PyTorch only, LSTM only), if any. 
                Must be less than the hidden size of the LSTM.

            - `final_rnn_return_sequences` (bool): Whether the final RNN layer returns sequences of hidden state. 
                **NOTE** Setting this to True will make the model much, much larger.
            
            - `apply_dense_for_each_time_step` (bool): Whether to apply the Dense network to each time step of the 
                RNN output. If False, the Dense network is applied to the last time step only if 
                `final_rnn_retrurn_sequences` is False, or applied to a flattened version of the output sequence
                otherwise (the dimensionality of the input feature space to the dense network will be multiplied
                by the sequence length. PLEASE NOTE that this only works if the entered sequence is exactly as long
                as the priorly defined sequence length according to the hyperparameters).

            - `dense_width` ("auto"|int|list|array): Width of the hidden layers of the Dense network. 
                "auto", a number (for all of them) or a vector holding width of each hidden layer.
            - `dense_depth` (int): Depth (number of hidden layers) of the Dense network.

            - `dense_activation` (str): Activation function for hidden layers of the Dense network.
                Supported activations are the lower-case names of their torch modules,
                e.g. "relu", "sigmoid", "softmax", "logsoftmax", "tanh", "leakyrelu", "elu", "selu", "softplus", etc.
            - `dense_activation_params` (dict): Dictionary of parameters for the activation func of the Dense network.
            - `output_activation` (str): Activation function for the output layer of the Dense network, if any.
                **NOTE** If the loss function is cross entropy, then no output activation is erquired, 
                though during inference you (may) have to manually add a logsoftmax.
                This is not required for prediction, as argmax(y) and argmax(logsoftmax(y)) yield the same result.
                However, if the loss function is nll (negative loglikelihood), 
                then you must specify an output activation as in "logsoftmax".
            - `output_activation_params` (dict): Dictionary of parameters for the activation func of the output layer.

            - `batchnorm` (str): Whether the batch normalization layer (if any) 
                should come before or after the activation of each hidden layer in the dense network.
                For activation functions such as **ReLU** and **sigmoid**, `"before"` is usually a better option. 
                For **tanh** and **LeakyReLU**, `"after"` is usually a better option.
            - `batchnorm_params` (dict): Dictionary of parameters for the batch normalization layer.
                
            - `dense_dropout` (float): Dropout rate (if any) for the hidden layers of the Dense network.

            - `batch_size` (int): Minibatch size, the expected input size of the network.
            - `learning_rate` (float): Initial learning rate of training.
            - `learning_rate_decay_gamma` (float): Exponential decay rate gamma for learning rate, if any.
            
            - `optimizer` (str): Optimizer, options are "sgd" and "adam" for now.
            - `optimizer_params` (dict): Additional parameters of the optimizer, if any.
            
            - `epochs` (int): Maximum number of epochs for training.
            - `validation_tolerance_epochs` (int): Epochs to tolerate unimproved val loss, before early stopping.
            - `l2_reg` (float): L2 regularization parameter.
            
            - `loss_function` (str): Loss function, options are "mse", "mae", "binary_crossentropy", 
                "categorical_crossentropy", "crossentropy", "kldiv", "nll".
            - `loss_function_params` (dict): Additional parameters for the loss function, if any.


        
        ### Returns

        Returns a `torch.nn.Module` object that can be trained and used accordingly.
        Run `print(net)` afterwards to see what you have inside the network.

        **NOTE** RNN kernel initializers and Dense kernel initializers have been modified to be Glorot Uniform.
        **NOTE** RNN recurrent initializer is modified to be Orthogonal.
        **NOTE** All bias initializers have been modified to be zeros.
        """
        super(Pytorch_Seq2Dense, self).__init__()
        if not hparams: hparams = self.sample_hparams
        self.hparams = hparams
        self._rnndict = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}
        self._rnntype = hparams["rnn_type"].upper()
        self._rnn = self._rnndict[self._rnntype]
        self._denseactivation = \
            actdict_pytorch[hparams["dense_activation"] if hparams.get("dense_activation") else "relu"]
        self._denseactivation_params = hparams.get("dense_activation_params")
        self._outactivation = \
            actdict_pytorch[hparams.get("output_activation")] if hparams.get("output_activation") else None
        self._outactivation_params = hparams.get("output_activation_params")
        self._batchnorm = hparams.get("batchnorm")
        self._batchnorm_params = hparams.get("batchnorm_params")
        self._infeatures = hparams["in_features"] if hparams.get("in_features") else 1
        self._outfeatures = hparams["out_features"] if hparams.get("out_features") else 1
        self._rnnhidsizes = hparams["rnn_hidden_sizes"] if hparams.get("rnn_hidden_sizes") else "auto"
        self._densehidsizes = hparams["dense_width"] if hparams.get("dense_width") else "auto"
        self._densedepth = hparams["dense_depth"] if hparams.get("dense_depth") else 0
        self._rnndepth = hparams["rnn_depth"] if hparams.get("rnn_depth") else 1
        self._bidirectional = True if hparams.get("rnn_bidirectional") else False
        self._rnndropout = hparams["rnn_dropout"] if hparams.get("rnn_dropout") else 0
        self._densedropout = hparams["dense_dropout"] if hparams.get("dense_dropout") else 0
        self._final_rnn_return_sequences = True if hparams.get("final_rnn_return_sequences") else False
        self._apply_dense_for_each_timestep = True if hparams.get("apply_dense_for_each_timestep") else False
        self._N = int(hparams["batch_size"])
        self._batchsize = int(hparams["batch_size"])
        self._L_in = int(hparams["in_seq_len"])
        self._L_out = int(hparams["out_seq_len"]) if hparams.get("out_seq_len") else 1
        self._D = int(2 if self._bidirectional else 1)
        self._lstmprojsize = hparams["lstm_projsize"] if hparams.get("lstm_projsize") and \
            hparams.get("lstm_projsize") > 0 else (0 if self._rnntype=="LSTM" else None)
        self._H_in = int(self._infeatures)
        if self._rnnhidsizes == "auto":
            self._H_cell = int(2**(np.round(math.log2(self._H_in*self._L_in))))
        else:
            self._H_cell = int(self._rnnhidsizes)
        self._H_out = int(self._lstmprojsize if self._lstmprojsize and self._lstmprojsize > 0 else self._H_cell)
        self._rnn_output = torch.zeros((self._N, self._L_in, self._D*self._H_out))
        self._rnn_output_flattened = self._rnn_output[:,-1,:]
        self._densesizevec = []
        self._denselayers = []
        self._loss = hparams.get("loss_function")
        self._lossfunctionparams = hparams.get("loss_function_params")
        self._optimizer = hparams.get("optimizer")
        self._optimizerparams = hparams.get("optimizer_params")
        self._earlystop = hparams.get("validation_tolerance_epochs")
        self._learnrate = hparams.get("learning_rate")
        self._learnrate_decay_gamma = hparams.get("learning_rate_decay_gamma")
        self._validation_data = hparams.get("validation_data")
        self._epochs = hparams.get("epochs")
        self._l2_reg = hparams.get("l2_reg") if hparams.get("l2_reg") else 0.0
        self.history = None
        self.batch_input_shape = (self._N, self._L_in, self._H_in)
        if self._final_rnn_return_sequences and self._apply_dense_for_each_timestep:
            self.batch_output_shape = (self._N, self._L_out, self._outfeatures)
        else:
            self.batch_output_shape = (self._N, self._L_out * self._outfeatures)
        if self._l2_reg > 0.0:
            if self._optimizerparams is not None:
                self._optimizerparams["weight_decay"] = self._l2_reg
            else:
                self._optimizerparams = {"weight_decay": self._l2_reg}
        
        
        # Constructing RNN layers
        if self._rnntype == "LSTM" and self._lstmprojsize is not None and self._lstmprojsize > 0:
            self.rnn = nn.LSTM(
                input_size=self._H_in,
                hidden_size=self._H_cell,
                num_layers=self._rnndepth,
                batch_first=True,
                dropout=self._rnndropout,
                bidirectional=self._bidirectional,
                proj_size=self._lstmprojsize)
        else:
            self.rnn = self._rnn(
                input_size=self._H_in,
                hidden_size=self._H_cell,
                num_layers=self._rnndepth,
                batch_first=True,
                dropout=self._rnndropout,
                bidirectional=self._bidirectional)
        for attrib in dir(self.rnn):
            if attrib.startswith("weight_ih"):
                xavier_uniform_(self.rnn.__getattr__(attrib))
            elif attrib.startswith("weight_hh"):
                orthogonal_(self.rnn.__getattr__(attrib))
            elif attrib.startswith("bias_"):
                zeros_(self.rnn.__getattr__(attrib))
        
        # Calculating Dense layers widths
        cf = self._L_in if (self._final_rnn_return_sequences and not self._apply_dense_for_each_timestep) else 1 
        self._dense_input_size = self._H_out * self._D * cf
        if self._densedepth > 0:
            if "list" in type(self._densehidsizes).__name__ or "numpy" in type(self._densehidsizes).__name__:
                self._densesizevec = self._densehidsizes
            elif self._densehidsizes == "auto":
                if self._final_rnn_return_sequences:
                    old = self._dense_input_size
                else:
                    old = self._H_cell
                for i in range(self._densedepth):
                    new = old//2 if old//2 > self._outfeatures*self._L_out else old
                    old = new
                    self._densesizevec.append(new)
            else:
                self._densesizevec = [self._densehidsizes for _ in range(self._densedepth)]
        else:
            self._densesizevec = []
            self._densehidsizes = []


        # Constructing Dense layers
        old = self._dense_input_size
        if self._densedepth > 0:
            for size in self._densesizevec:
                l = nn.Linear(old,size)
                xavier_uniform_(l.weight)
                zeros_(l.bias)
                self._denselayers.append(l)
                if self._batchnorm == "before":
                    if self._batchnorm_params:
                        self._denselayers.append(nn.BatchNorm1d(size, **self._batchnorm_params))
                    else:
                        self._denselayers.append(nn.BatchNorm1d(size))
                if self._denseactivation_params:
                    self._denselayers.append(self._denseactivation(**self._denseactivation_params))
                else:
                    self._denselayers.append(self._denseactivation())
                if self._batchnorm == "after":
                    if self._batchnorm_params:
                        self._denselayers.append(nn.BatchNorm1d(size, **self._batchnorm_params))
                    else:
                        self._denselayers.append(nn.BatchNorm1d(size))
                if self._densedropout:
                    self._denselayers.append(nn.Dropout(self._densedropout))
                old = size
        if self._final_rnn_return_sequences and not self._apply_dense_for_each_timestep:
            self._dense_output_size = int(self._L_out*self._outfeatures)
        else:
            self._dense_output_size = int(self._outfeatures)
        l = nn.Linear(old, int(self._dense_output_size))
        xavier_uniform_(l.weight)
        zeros_(l.bias)
        self._denselayers.append(l)
        if self._outactivation:
            if self._outactivation_params:
                self._denselayers.append(self._outactivation(**self._outactivation_params))
            else:
                self._denselayers.append(self._outactivation())
        self.decoder = nn.Sequential(*self._denselayers)
        self.rnn.flatten_parameters()
    
    
    def forward(self, x):
        # self._rnn_output, (self._rnn_final_hidden_states, self._lstm_final_cell_states) = self.rnn(x)
        self._rnn_output, _ = self.rnn(x)
        if self._final_rnn_return_sequences and not self._apply_dense_for_each_timestep:
            self._rnn_output_flattened = self._rnn_output.view(self._N, self._dense_input_size)
        else:
            self._rnn_output_flattened = self._rnn_output[:,-1,:]
        out = self.decoder(self._rnn_output_flattened)
        return out
    
    
    def train_model(self, dataset, verbose:bool=True, script_before_save:bool=False, saveto:str=None, **kwargs):
        self.history = train_pytorch_model(self, dataset, self._batchsize, self._loss, self._optimizer, 
            self._optimizerparams, self._lossfunctionparams, self._learnrate, 
            self._learnrate_decay_gamma, self._epochs, self._earlystop, self._validation_data, 
            verbose, script_before_save, saveto, **kwargs)
        return self.history
    
    def evaluate_model(self, dataset, verbose:bool=True, **kwargs):
        return evaluate_pytorch_model(self, dataset, loss_str=self._loss, loss_function_params=self._lossfunctionparams,
            batch_size=self._batchsize, verbose=verbose, **kwargs)
    
    def predict_model(self, dataset, 
        return_in_batches:bool=True, return_inputs:bool=False, return_raw_predictions:bool=False, verbose:bool=True,
        **kwargs):
        return predict_pytorch_model(self, dataset, self._loss, self._batchsize, 
                                     return_in_batches=return_in_batches, return_inputs=return_inputs, 
                                     return_raw_predictions=return_raw_predictions, verbose=verbose, **kwargs)




########################################################################################################################
# MODELS FOR PART 1 #
########################################################################################################################

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1



class RNN_Language_Network(Pytorch_Seq2Dense):
    def __init__(self, hparams:dict=None):
        super(RNN_Language_Network, self).__init__(hparams)
        self.embed_dim = hparams['embedding_dim'] if hparams.get('embedding_dim') else hparams['in_features']
        self.vocab_size = hparams['vocab_size'] if hparams.get('vocab_size') else 27
        assert self.embed_dim == self._infeatures, \
            "Embedding dim (%d) must be equal to input feature dim (%d)."%(self.embed_dim, self._infeatures)
        self.embed_layer = nn.Embedding(self.vocab_size, self.embed_dim)
        self._embed_output = None
        self.permute_output = hparams['permute_output'] if hparams.get('permute_output') else False
        self.batch_input_shape = [hparams['batch_size'], hparams['in_seq_len']]
        if self.permute_output:
            self.batch_output_shape = [hparams['batch_size'], hparams['out_features'], hparams['out_seq_len']]
        else:
            self.batch_output_shape = [hparams['batch_size'], hparams['out_seq_len'], hparams['out_features']]
        
    def forward(self, x):
        # TODO: Transfer the "permute_output" logic to the base class as well.
        # self._rnn_output, (self._rnn_final_hidden_states, self._lstm_final_cell_states) = self.rnn(x)
        # Shape of x should be: [N, L]
        self._embed_output = self.embed_layer(x)    # [N, L, embed_dim]
        self._rnn_output, _ = self.rnn(self._embed_output)
        if self._final_rnn_return_sequences:
            if self._apply_dense_for_each_timestep:
                self._rnn_output_flattened = self._rnn_output
            else:
                self._rnn_output_flattened = self._rnn_output.view(self._rnn_output.shape[0], -1)
        else:
            # RNN output is of shape  (N, L, D * H_out)
            self._rnn_output_flattened = self._rnn_output[:,-1,:]
        out = self.decoder(self._rnn_output_flattened)
        if self.permute_output:
            return out.permute(self.permute_output)
        else:
            return out
    
    def test(self):
        print("------------------------------------------------------------------")
        print("Testing RNN_Language_Network")
        print("Constructing random inputs and outputs ...")
        print("Batch size:              %d"%self._batchsize)
        print("Input sequence length:   %d"%self._L_in)
        print("Output sequence length:  %d"%self._L_out)
        print("Input feature dimension: %d"%self._infeatures)
        print("Construjcting random torch.long tensor for input ...")
        x = torch.randint(0, self.vocab_size, self.batch_input_shape, dtype=torch.long)
        print("Input shape:  %s"%str(x.shape))
        print("Constructing random torch.float tensor for output ...")
        y_true = torch.rand(size=self.batch_output_shape)
        print("Output shape from truth: %s"%str(y_true.shape))
        print("Calling the forward method ...")
        y_pred = self.forward(x)
        print("Output shape from preds: %s"%str(y_pred.shape))
        assert y_true.shape == y_pred.shape, \
            "Output shape (%s) does not match expected shape (%s)"%(str(y_pred.shape), str(y_true.shape))
        print("Testing complete. Output shape matches expected shape.")
        print("------------------------------------------------------------------")

        
        



class RNNClassifier(ConsonantVowelClassifier):
    def __init__(self, hparams:dict, vocab_index):
        self.model = RNN_Language_Network(hparams)
        self.hparams = hparams
        self.vocab_index = vocab_index
        self.loss_str = self.model._loss
        self.history = None
    
    def train_model(self, dataset, verbose:bool=True, saveto:str=None, **kwargs):
        self.history = train_pytorch_model(self.model, dataset, self.model._batchsize, self.model._loss, 
            self.model._optimizer, 
            self.model._optimizerparams, self.model._lossfunctionparams, self.model._learnrate, 
            self.model._learnrate_decay_gamma, self.model._epochs, self.model._earlystop, self.model._validation_data, 
            verbose, saveto=saveto, **kwargs)
        
    def predict(self, context): # Will be used on a single example
        indices = [self.vocab_index.index_of(letter) for letter in context]     # shape = [L]
        input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)     # shape = [1, L]
        self.model.eval()
        self.model.cpu()
        with torch.no_grad():
            prediction = self.model.forward(input_tensor).squeeze()             # shape = [1]
            if self.loss_str == "binary_crossentropy":
                # Output layer already includes sigmoid.
                class_prediction = (prediction > 0.5).int().item()
            elif self.loss_str == "bce_with_logits":
                # Output layer does not include sigmoid. Sigmoid is a part of the loss function.
                class_prediction = (torch.sigmoid(prediction) > 0.5).int().item()
        return class_prediction
    
    def __str__(self):
        return str(self.model)
            


def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """
    # Determine sequence length from training data
    seqlen = max([len(ex) for ex in (train_cons_exs + train_vowel_exs)])
    
    # Determine hyperparameters
    hparams = {
        'model_name': autoname("rnn_classifier"),
        'in_features': 8,
        'embedding_dim': 8,
        'vocab_size': 27,
        'out_features': 1,
        'in_seq_len': seqlen,
        'out_seq_len': 1,
        'rnn_type': 'LSTM',
        'rnn_hidden_sizes': 32,
        'rnn_bidirectional': True,
        'rnn_depth': 1,
        'rnn_dropout': None,
        'dense_width': 32,
        'dense_depth': 1,
        'dense_dropout': None,
        'dense_activation': 'relu',
        'output_activation': 'sigmoid',
        'batchnorm': "before",
        'batchnorm_params': None,
        'l2_reg': 0.0001,
        'batch_size': 128,
        'epochs': 60,
        'validation_data': 0.2,
        'validation_tolerance_epochs': 10000,
        'learning_rate': 0.1,
        'learning_rate_decay_gamma': 0.8,
        'loss_function': 'binary_crossentropy',
        'loss_function_params': None,
        'optimizer': 'adam',
        'optimizer_params': {'eps': 1e-6, 'betas': (0.9, 0.999)}
    }
    
    # Construct data tensors
    train_cons = get_tensor_from_exs(train_cons_exs, seqlen, vocab_index)
    train_vowl = get_tensor_from_exs(train_vowel_exs, seqlen, vocab_index)
    dev_cons = get_tensor_from_exs(dev_cons_exs, seqlen, vocab_index)
    dev_vowl = get_tensor_from_exs(dev_vowel_exs, seqlen, vocab_index)
    
    print("Training set for consonants has a shape of: {}".format(train_cons.shape))
    print("Training set for vowels has a shape of:     {}".format(train_vowl.shape))
    print("Dev set for consonants has a shape of:      {}".format(dev_cons.shape))
    print("Dev set for vowels has a shape of:          {}".format(dev_vowl.shape))
    
    # Construct input and output data
    train_cons_labels = torch.zeros([train_cons.shape[0],1], dtype=torch.float32)
    train_vowl_labels = torch.ones([train_vowl.shape[0],1], dtype=torch.float32)
    dev_cons_labels = torch.zeros([dev_cons.shape[0],1], dtype=torch.float32)
    dec_vowl_labels = torch.ones([dev_vowl.shape[0],1], dtype=torch.float32)
    train_data = torch.cat([train_cons, train_vowl], dim=0)
    train_labels = torch.cat([train_cons_labels, train_vowl_labels], dim=0)
    dev_data = torch.cat([dev_cons, dev_vowl], dim=0)
    dev_labels = torch.cat([dev_cons_labels, dec_vowl_labels], dim=0)
    
    # Construct datasets
    train_dataset = TensorDataset(train_data, train_labels)
    dev_dataset = TensorDataset(dev_data, dev_labels)
    
    
    # Construct model
    model = RNNClassifier(hparams, vocab_index)
    print(model)
    
    # Train model
    model.train_model([train_dataset, dev_dataset], verbose=True,
                    #   saveto=make_path('assignment3/starter_code/models/'+hparams['model_name']+'.pt'),
                      script_before_save=False
                      )
    return model
    





########################################################################################################################
# MODELS FOR PART 2 #
########################################################################################################################


class LanguageModel(object):

    def get_next_char_log_probs(self, context):# -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e
        :param context: a single character to score
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context):# -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")



class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel):
    def __init__(self, hparams:dict, vocab_index):
        self.model = RNN_Language_Network(hparams)
        self.hparams = hparams
        self.vocab_index = vocab_index
        self.loss_str = self.model._loss
        self.history = None
    
    def train_model(self, dataset, verbose:bool=True, saveto:str=None, **kwargs):
        self.history = train_pytorch_model(
            self.model, dataset, self.model._batchsize, self.model._loss, self.model._optimizer, 
            self.model._optimizerparams, self.model._lossfunctionparams, self.model._learnrate, 
            self.model._learnrate_decay_gamma, self.model._epochs, self.model._earlystop, self.model._validation_data, 
            verbose, saveto=saveto, **kwargs)
    
    def plot_training_progress(self, saveto:str=None):
        plt.figure(figsize=(8,8))
        plt.subplot(2,1,1)
        plt.grid(True)
        plt.plot(self.history['training_loss'], label='train')
        plt.plot(self.history['validation_loss'], label='dev')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.subplot(2,1,2)
        plt.grid(True)
        plt.plot(self.history['training_metrics'], label='train')
        plt.plot(self.history['validation_metrics'], label='dev')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        if saveto is not None:
            plt.savefig(saveto, dpi=600)
        plt.show()
        

    def get_next_char_log_probs(self, context):
        pred = self.predict(context)
        return pred[0,:,-1]

    def get_log_prob_sequence(self, next_chars, context):
        pred = self.predict(context+next_chars)
        pred = pred[0,:,len(context):]
        # print("----------------- log_prob_sequence is called -----------------")
        # print("Trying my method: ")
        # lp = 0
        # for i in range(len(next_chars)):
        #     value = pred[self.vocab_index.index_of(next_chars[i]),i]
        #     print("Value being added at index {} is: {}".format(i, value))
        #     lp += value
        # print("The log probability of the sequence is: ", lp)
        # print("\nTrying the one-by-one method of the assignment: ")
        log_prob_from_single_probs = 0.0
        for i in range(0, len(next_chars)):
            # print(repr(next_seq[0:i]))
            # print(repr(next_seq[i]))
            next_char_log_probs = self.get_next_char_log_probs(context + next_chars[0:i])
            value = next_char_log_probs[self.vocab_index.index_of(next_chars[i])]
            # print("Value being added at index {} is: {}".format(i, value))
            # print(repr(next_char_log_probs))
            log_prob_from_single_probs += value
        # print("The log probability of the sequence is: ", log_prob_from_single_probs)
        # print("\n")
        return log_prob_from_single_probs
        
    def predict(self, context):
        indices = [self.vocab_index.index_of(letter) for letter in context]     # shape=[L]
        input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)     # shape=[1,L]
        # print("Shape of input_tensor: ", input_tensor.shape)
        self.model.eval()
        self.model.cpu()
        with torch.no_grad():
            prediction = self.model(input_tensor)            # shape=[1, K, L] assuming it is already permuted
            # print("Shape of prediction: ", prediction.shape)
            if self.loss_str == "nll":
                # Output layer already includes log_softmax.
                sequence_class_log_probs = prediction
                # sequence_class_predictions = torch.argmax(prediction, dim=0, keepdim=True).float()
            elif self.loss_str in ["crossentropy", "categorical_crossentropy"]:
                # Output layer does not have log_softmax. It is implemented as a part of the loss function.
                sequence_class_lop_probs = torch.log_softmax(prediction, dim=1)
                # class_predictions = \
                    # torch.argmax(torch.log_softmax(predictions, dim=0), dim=0)#, keepdim=True).float()
        return sequence_class_log_probs.numpy()
    
    def classify(self, context):
        pred = self.predict(context)
        class_pred = np.argmax(pred.squeeze(), axis=0)
        return class_pred
    
    def evaluate(self, context):
        if context[0] != ' ': context = ' ' + context       # Take care of the <SOS> token
        if context[-1] != ' ': context = context + ' '      # Take care of the <EOS> token
        class_target = np.array([self.vocab_index.index_of(letter) for letter in context[1:]], dtype=int)
        class_pred = self.classify(context[:-1])
        correct = (class_target == class_pred).astype(int).sum()
        accuracy = correct / len(context)
        return accuracy
        
    
    def __str__(self):
        return str(self.model)





def train_lm_with_scale(args, train_text, dev_text, vocab_index, scale='small'):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :param scale: 'small' or 'large'. 'small' uses individual words as sequences, using space as <SOS> amd <EOS>.
        'large' uses running windows of multiple words as sequences. Again, space character is used similarly.
    :return: an RNNLanguageModel instance trained on the given data
    """
    
    LARGE_SCALE_SEQLEN_WORDS = 10
    
    train_exs = train_text.split()
    dev_exs = dev_text.split()
    if scale == 'small':
        train_exs_aug = [(' ' + word + ' ') for word in train_exs]
        dev_exs_aug = [(' ' + word + ' ') for word in dev_exs]
    else:
        train_exs_aug = []
        for i in range(len(train_exs)):
            end = (i+LARGE_SCALE_SEQLEN_WORDS) if (i+LARGE_SCALE_SEQLEN_WORDS) < len(train_exs) else (len(train_exs)-1)
            train_exs_aug.append(' ' + ' '.join(train_exs[i:end]) + ' ')
        dev_exs_aug = []
        for i in range(len(dev_exs)):
            end = (i+LARGE_SCALE_SEQLEN_WORDS) if (i+LARGE_SCALE_SEQLEN_WORDS) < len(dev_exs) else (len(dev_exs)-1)
            dev_exs_aug.append(' ' + ' '.join(dev_exs[i:end]) + ' ')
            
    
    # Determine sequence length from training data
    if scale == 'small':
        # Here, we define sequence length based on the length of the longest word (using space as <SOS> and <EOS>)
        _seqlen = max([len(word) for word in train_exs]) + 2
        _seqlen = int(2**(np.ceil(np.log2(_seqlen)))) + 1
    else:
        _seqlen = max([len(seq) for seq in train_exs_aug])
        _seqlen = int(2**(np.ceil(np.log2(_seqlen)))) + 1
        
    print("\nSequence length: ", _seqlen)
    
    seqlen = _seqlen - 1
    # Determine hyperparameters
    hparams = {
        'model_name': autoname("rnn_language_model"),
        'in_features': 8,
        'embedding_dim': 8,
        'vocab_size': 27,
        'out_features': 27,
        'in_seq_len': seqlen,
        'out_seq_len': seqlen,
        'rnn_type': 'LSTM',
        'rnn_hidden_sizes': 64,
        'rnn_bidirectional': False,     # Inapplicable with this kind of causal language model with single RNN layer.
        'rnn_depth': 1,
        'rnn_dropout': None,
        'lstm_projsize': None,
        'final_rnn_return_sequences': True,
        'apply_dense_for_each_timestep': True,
        'permute_output': [0,2,1],      # RNN+Dense output is [N, L, K]. Loss function wants [N, K, L]
        'dense_width': 64,
        'dense_depth': 0,
        'dense_dropout': None,
        'dense_activation': 'relu',
        'output_activation': 'logsoftmax',
        'output_activation_params': {'dim': 2}, # Output of RNN+Dense is [N, L, K] where K is number of classes
        'batchnorm': None,                  # Batchnorm is not supported (for now) for RNNs with return sequences
        'batchnorm_params': None,
        'l2_reg': 0.0001,
        'batch_size': 128,
        'epochs': 200,
        'validation_data': 0.2,
        'validation_tolerance_epochs': 20,
        'learning_rate': 1.0,
        'learning_rate_decay_gamma': 0.9,
        'loss_function': 'nll',
        'loss_function_params': None,
        'optimizer': 'adam',
        'optimizer_params': {'eps': 1e-6, 'betas': (0.9, 0.999)}
    }
    
    # Construct data tensors
    train_tensor = get_tensor_from_exs(train_exs_aug, _seqlen, vocab_index)
    dev_tensor = get_tensor_from_exs(dev_exs_aug, _seqlen, vocab_index)
    train_inputs = train_tensor[:, :-1]
    train_outputs = train_tensor[:, 1:]#.to(torch.float32)
    dev_inputs = dev_tensor[:, :-1]
    dev_outputs = dev_tensor[:, 1:]#.to(torch.float32)
    print("Training inputs shape:  ", train_inputs.shape)
    print("Training outputs shape: ", train_outputs.shape)
    print("Dev inputs shape:       ", dev_inputs.shape)
    print("Dev outputs shape:      ", dev_outputs.shape)
    print("Sample train input: ")
    print(train_inputs[0])
    print("Sample train output: ")
    print(train_outputs[0])
    print("Sample dev input: ")
    print(dev_inputs[0])
    print("Sample dev output: ")
    print(dev_outputs[0])
    
    # Construct datasets
    train_dataset = TensorDataset(train_inputs, train_outputs)
    dev_dataset = TensorDataset(dev_inputs, dev_outputs)
    
    # Construct model
    model = RNNLanguageModel(hparams, vocab_index)
    print(model)
    
    # Train model
    model.train_model([train_dataset, dev_dataset], verbose=True,
        saveto=make_path('assignment3/starter_code/models/'+hparams['model_name']+'.pt'),
        script_before_save=False
        )
    
    # Plot training progress curve
    model.plot_training_progress(
        saveto="assignment3/starter_code/models/"+hparams['model_name']+".png"
        )
    
    # Test model on the entire devset text
    print("\nTesting the model on the entire devset text ...")
    accuracy = model.evaluate(dev_text)
    print("Accuracy: ", accuracy)
    print("The model is trained and tested. Returning.")
    return model







SCALE = 'large'
def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    return train_lm_with_scale(args, train_text, dev_text, vocab_index, SCALE)
    
    
    













if __name__ == '__main__':
    print("Current working directory: {}".format(os.getcwd()))
    print("Testing RNN_Language_Model ...")
    hparams = {
        'model_name': autoname("rnn_language_model"),
        'in_features': 8,
        'embedding_dim': 8,
        'vocab_size': 27,
        'out_features': 27,
        'in_seq_len': 32,
        'out_seq_len': 32,
        'rnn_type': 'LSTM',
        'rnn_hidden_sizes': 64,
        'rnn_bidirectional': True,
        'rnn_depth': 1,
        'rnn_dropout': None,
        'lstm_projsize': None,
        'final_rnn_return_sequences': True,
        'apply_dense_for_each_timestep': True,
        'permute_output': [0,2,1],
        'dense_width': 32,
        'dense_depth': 0,
        'dense_dropout': None,
        'dense_activation': 'relu',
        'output_activation': 'logsoftmax',
        'batchnorm': None,
        'batchnorm_params': None,
        'l2_reg': 0.0001,
        'batch_size': 128,
        'epochs': 60,
        'validation_data': 0.2,
        'validation_tolerance_epochs': 10000,
        'learning_rate': 0.1,
        'learning_rate_decay_gamma': 0.8,
        'loss_function': 'nll',
        'loss_function_params': None,
        'optimizer': 'adam',
        'optimizer_params': {'eps': 1e-6, 'betas': (0.9, 0.999)}
    }
    model = RNN_Language_Network(hparams)
    print(model)
    model.test()
    