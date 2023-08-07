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
from torch.nn.init import xavier_uniform_, zeros_, orthogonal_, calculate_gain
# Note: Custom random initializations should NOT be implemented without the gain values from calculate_gain.
from sklearn.metrics import r2_score
import numpy as np
import collections
from tqdm import tqdm

########################################################################################################################
# Global variables, functions, and classes
########################################################################################################################

# Set random seed
SEED = 42

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
    

def _update_metrics_for_batch(
    predictions:torch.Tensor, targets:torch.Tensor, loss_str:str, classification:bool, regression:bool, 
    verbose:int, batch_num:int, epoch:int, metric:float, num_logits:int):
    if loss_str == "binary_crossentropy":
        # Output layer already includes sigmoid.
        class_predictions = (predictions > 0.5).int()
    elif loss_str == "bce_with_logits":
        # Output layer does not include sigmoid. Sigmoid is a part of the loss function.
        class_predictions = (torch.sigmoid(predictions) > 0.5).int()
    elif loss_str in ["nll", "crossentropy", "categorical_crossentropy"]:
        # nll -> Output layer already includes log_softmax.
        # others -> Output layer has no log_softmax. It's implemented as a part of the loss function.
        class_predictions = torch.argmax(predictions, dim=1)   

    if classification:
        if verbose==2 and batch_num==0 and epoch ==0: 
            print("Shape of model outputs:     ", predictions.shape)
            print("Shape of class predictions: ", class_predictions.shape)
            print("Shape of targets:           ", targets.shape)
        # Calculate accuracy
        correct = (class_predictions == targets).int().sum().item()
        num_logits += targets.numel()
        metric += correct
        if verbose==2 and epoch==0: 
            print("Number of correct answers (this batch - total): %10d - %10d"%(correct, metric))
        # Calculate F1 score
        # f1 = f1_score(targets.cpu().numpy(), class_predictions.cpu().numpy(), average="macro")
    elif regression:
        if verbose==2 and batch_num==0 and epoch==0: 
            print("Shape of predictions: ", predictions.shape)
            print("Shape of targets:     ", targets.shape)
        # Calculate r2_score
        metric += r2_score(targets.cpu().numpy(), predictions.cpu().numpy())
    
    return metric, num_logits



def _test_shapes(predictions:torch.Tensor, targets:torch.Tensor, classification:bool):
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




def _calculate_epoch_loss_and_metrics(
    cumulative_epoch_loss:float, num_batches:int, verbose:int, epoch:int, 
    hist_loss:dict, hist_metric:dict, display_metrics:bool, cumulative_metric:float, metric_denominator:int):
    # Calculate training epoch loss
    epoch_loss = cumulative_epoch_loss / num_batches
    if verbose==2 and epoch==0: print("Epoch loss (training): %.5f"%epoch_loss)
    if hist_loss is not None: hist_loss.append(epoch_loss)
    # Calculate training epoch metric (accuracy or r2-score)
    if display_metrics:
        epoch_metric = cumulative_metric / metric_denominator
        if verbose==2 and epoch==0: print("Epoch metric: %.5f"%epoch_metric)
        if hist_metric is not None: hist_metric.append(epoch_metric)
    return epoch_loss, epoch_metric, hist_loss, hist_metric



def save_pytorch_model(model:torch.nn.Module, saveto:str, dataloader, script_before_save:bool=True, verbose:int=1):
    try:
        if verbose > 0: print("Saving model...")
        if script_before_save:
            example,_ = next(iter(dataloader))
            example = example[0,:].unsqueeze(0)
            model.cpu()
            with torch.no_grad():
                traced = torch.jit.trace(model, example)
                traced.save(saveto)
        else:
            with torch.no_grad():
                torch.save(model, saveto)
    except Exception as e:
        if verbose > 0:
            print(e)
            print("Failed to save the model.")
    if verbose > 0: print("Done Saving.")
    
    
    

def train_pytorch_model(model, dataset, batch_size:int, loss_str:str, optimizer_str:str, 
    optimizer_params:dict=None, loss_function_params:dict=None, learnrate:float=0.001, 
    learnrate_decay_gamma:float=None, epochs:int=10, validation_patience:int=10000, validation_data:float=0.1, 
    verbose:int=1, script_before_save:bool=True, saveto:str=None, num_workers=0):
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
        - `verbose` (int, optional): Logging the progress. Defaults to 1. 0 prints nothing, 2 prints everything.
        - `script_before_save` (bool, optional): Use TorchScript for serializing the model. Defaults to True.
        - `saveto` (str, optional): Save PyTorch model in path. Defaults to None.
        - `num_workers` (int, optional): Number of workers for the dataloader. Defaults to 0.
        
    ### Returns:
        - `model`: Trained PyTorch-compatible model
        - `history`: PyTorch model history dictionary, containing the following keys:
            - `training_loss`: List containing training loss values of epochs.
            - `validation_loss`: List containing validation loss values of epochs.
            - `learning_rate`: List containing learning rate values of epochs.
            - `training_metrics`: List containing training metric values of epochs.
            - `validation_metrics`: List containing validation metric values of epochs.
    """
    # Initialize necessary lists
    hist_training_loss = []
    hist_validation_loss = []
    hist_learning_rate = []
    hist_trn_metric = []
    hist_val_metric = []
    
    # Empty CUDA cache
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # Check if validation data is provided or not, and calculate number of training and validation data
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

    if verbose > 0:
        print("Total number of data points:      %d"%num_all_data)
        print("Number of training data points:   %d"%num_train_data)
        print("Number of validation data points: %d"%num_val_data)
    
    # Generate training and validation dataloaders    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    if verbose > 0:
        print("Number of training batches:    %d"%len(trainloader))
        print("Number of validation batches:  %d"%len(validloader))
        print("Batch size:                    %d"%batch_size)
        for x,y in trainloader:
            print("Shape of training input from the dataloader:  ", x.shape)
            print("Shape of training output from the dataloader: ", y.shape)
            break
        for x,y in validloader:
            print("Shape of validation input from the dataloader:  ", x.shape)
            print("Shape of validation output from the dataloader: ", y.shape)
            break
    
    # Select the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if verbose > 0: print("Selected device: ", device)
    model.to(device)
    
    # Instantiate the loss function
    loss_func = lossdict_pytorch[loss_str]
    if loss_function_params: criterion = loss_func(**loss_function_params)
    else: criterion = loss_func()
        
    # Instantiate the optimizer
    optimizer_func = optdict_pytorch[optimizer_str]
    if optimizer_params: optimizer = optimizer_func(model.parameters(), lr=learnrate, **optimizer_params)
    else: optimizer = optimizer_func(model.parameters(), lr=learnrate)

    # Defining learning rate scheduling
    if learnrate_decay_gamma:
        if verbose > 0: print("The learning rate has an exponential decay rate of %.5f."%learnrate_decay_gamma)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=learnrate_decay_gamma)
        lr_sch = True
    else:
        lr_sch = False
    
    # Find out if we will display any metric along with the loss.
    display_metrics = True
    classification = False
    regression = False
    if loss_str in ["binary_crossentropy", "bce_with_logits", "nll", "crossentropy", "categorical_crossentropy"]:
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
    if verbose > 0:
        if classification: print("Classification problem detected. We will look at accuracies.")
        elif regression: print("Regression problem detected. We will look at R2 scores.")
        else: print("We have neither classification nor regression problem. No metric will be displayed.")
    
                    
    # Calculating number of training and validation batches
    num_training_batches = len(trainloader)
    num_validation_batches = len(validloader)
    
    # Preparing progress bar
    progress_bar_size = 40
    ch = "█"
    intvl = num_training_batches/progress_bar_size;
    valtol = validation_patience if validation_patience else 10000
    minvalerr = 1000000.0
    badvalcount = 0
    
    # Commencing training loop
    tStart = timer()
    loop = tqdm(range(epochs), desc='Training Progress', ncols=100) if verbose==1 else range(epochs)
    for epoch in loop:
        
        # Initialize per-epoch variables
        tEpochStart = timer()
        epoch_loss_training = 0.0
        epoch_loss_validation = 0.0
        newnum = 0
        oldnum = 0
        trn_metric = 0.0
        val_metric = 0.0
        num_train_logits = 0
        num_val_logits = 0
    
        if verbose==2 and epoch > 0: print("Epoch %3d/%3d ["%(epoch+1, epochs), end="")
        if verbose==2 and epoch ==0: print("First epoch ...")
        
        ##########################################################################
        # Training
        if verbose==2 and epoch==0: print("\nTraining phase ...")
        model.train()
        for i, data in enumerate(trainloader):
            # Fetch data
            seqs, targets = data[0].to(device), data[1].to(device)
            # Forward propagation
            predictions = model(seqs)
            # Test shapes
            _test_shapes(predictions, targets, classification)
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
                    trn_metric, num_train_logits = _update_metrics_for_batch(
                        predictions, targets, loss_str, classification, regression, 
                        verbose, i, epoch, trn_metric, num_train_logits)
                    
            # Visualization of progressbar within the batch
            if verbose==2 and epoch > 0:
                newnum = int(i/intvl)
                if newnum > oldnum:
                    print((newnum-oldnum)*ch, end="")
                    oldnum = newnum 
        
        # Update learning rate if necessary
        if lr_sch: scheduler.step()
        
        # Calculate epoch loss and metrics
        epoch_loss_training, trn_metric, hist_training_loss, hist_trn_metric = _calculate_epoch_loss_and_metrics(
            epoch_loss_training, num_training_batches, verbose, epoch, 
            hist_training_loss, hist_trn_metric, display_metrics, trn_metric, 
            (num_train_logits if classification else num_training_batches))
            
        if verbose==2 and epoch > 0: print("] ", end="")
        
        ##########################################################################
        # Validation
        if verbose==2 and epoch==0: print("\nValidation phase ...")
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(validloader):
                seqs, targets = data[0].to(device), data[1].to(device)
                predictions = model(seqs)
                loss = criterion(predictions, targets)
                epoch_loss_validation += loss.item()
                # Do prediction for metrics
                if display_metrics:
                    val_metric, num_val_logits = _update_metrics_for_batch(
                        predictions, targets, loss_str, classification, regression, 
                        verbose, i, epoch, val_metric, num_val_logits)
        # Calculate epoch loss and metrics
        epoch_loss_validation, val_metric, hist_validation_loss, hist_val_metric = _calculate_epoch_loss_and_metrics(
            epoch_loss_validation, num_validation_batches, verbose, epoch, 
            hist_validation_loss, hist_val_metric, display_metrics, val_metric, 
            (num_val_logits if classification else num_validation_batches))
        
        # Log the learning rate, if there is any scheduling.
        if lr_sch: hist_learning_rate.append(scheduler.get_last_lr())
        else: hist_learning_rate.append(learnrate)
        
        ##########################################################################
        # Post Processing Training Loop            
        tEpochEnd = timer()
        if verbose==2:
            if display_metrics:
                print("Loss: %5.4f |Val Loss: %5.4f |%s: %5.4f |%s: %5.4f | %6.3f s" % (
                    epoch_loss_training, epoch_loss_validation, trn_metric_name, trn_metric,
                    val_metric_name, val_metric, tEpochEnd-tEpochStart))
            else:
                print("Loss: %5.4f |Val Loss: %5.4f | %6.3f s" % (
                    epoch_loss_training, epoch_loss_validation, tEpochEnd-tEpochStart))
        
        # Checking for early stopping
        if epoch_loss_validation < minvalerr:
            minvalerr = epoch_loss_validation
            badvalcount = 0
        else:
            badvalcount += 1
            if badvalcount > valtol:
                if verbose > 0:
                    print("Validation loss not improved for more than %d epochs."%badvalcount)
                    print("Early stopping criterion with validation loss has been reached. " + 
                        "Stopping training at %d epochs..."%epoch)
                break
    # End for loop
    model.eval()
    ##########################################################################
    # Epilogue
    tFinish = timer()
    if verbose > 0:        
        print('Finished Training.')
        print("Training process took %.2f seconds."%(tFinish-tStart))
    if saveto:
       save_pytorch_model(model, saveto, trainloader, script_before_save, verbose)
    # Clear CUDA cache    
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    # Generate output dictionaries
    history = {
        'training_loss':hist_training_loss, 
        'validation_loss':hist_validation_loss, 
        'learning_rate':hist_learning_rate}
    if display_metrics:
        history["training_metrics"] = hist_trn_metric
        history["validation_metrics"] = hist_val_metric
    if verbose > 0: print("Done training.")
    
    return history




def evaluate_pytorch_model(model, dataset, loss_str:str, loss_function_params:dict=None,
    batch_size:int=16, device_str:str="cuda", verbose:bool=True, num_workers:int=0):
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
    `num_workers` (int, optional) : The number of workers to use when making dataloader. Defaults to 0.
    
    
    ### Returns
    
    A dictionary containing the evaluation metrics, including "loss" and "metrics" in case any metric is available.
    """
    # Clear CUDA cache
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    if verbose: print("Preparing data...")
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    num_batches = len(testloader)
        
    if "cuda" in device_str:
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    print("selected device: ", device)
    model.eval()
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
        
    progress_bar_size = 40
    ch = "█"
    intvl = num_batches/progress_bar_size;
    if verbose: print("Evaluating model...")
    model.eval()
    newnum = 0
    oldnum = 0
    totloss = 0.0
    if verbose: print("[", end="")
    val_metric = 0.0
    num_val_logits = 0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs, targets = data[0].to(device), data[1].to(device)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            totloss += loss.item()
            
            # Do prediction for metrics
            if display_metrics:
                val_metric, num_val_logits = _update_metrics_for_batch(
                        predictions, targets, loss_str, classification, regression, 
                        verbose, i, 0, val_metric, num_val_logits)
                    
            # Visualization of progressbar
            if verbose:
                newnum = int(i/intvl)
                if newnum > oldnum:
                    print((newnum-oldnum)*ch, end="")
                    oldnum = newnum 
    
    totloss, val_metric, _, _ = _calculate_epoch_loss_and_metrics(
            totloss, num_batches, verbose, 0, 
            None, None, display_metrics, val_metric, 
            (num_val_logits if classification else num_batches))
        
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if verbose: print("] ", end="") 
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
    return_in_batches:bool=True, return_inputs:bool=False, return_raw_predictions:bool=False, verbose:bool=True,
    num_workers:int=0):
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
        - `num_workers` (int, optional): Number of workers to use when making dataloader. Defaults to 0.

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
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if verbose: print("Preparing data...")
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    num_batches = len(testloader)
    if "cuda" in device_str:
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    if verbose: print("selected device: ", device)
    model.to(device)
    if loss_str in ["binary_crossentropy", "bce_with_logits", "nll", "crossentropy", "categorical_crossentropy"]:
        classification = True
    else:
        classification = False
    output_list = []
    progress_bar_size = 40
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
                elif loss_str in ["nll", "crossentropy", "categorical_crossentropy"]:
                    class_predictions = torch.argmax(predictions, dim=1).float()
            
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
        
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if verbose: print("] ")
    if return_in_batches:
        return output_list
    else:
        return output_array
        

########################################################################################################################
########################################################################################################################
########################################################################################################################

class PyTorchSmartModule(nn.Module):
    
    sample_hparams = {
        'model_name': 'dummy_Pytorch_Smart_Module',
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
        """
        Base class for smart, trainable pytorch modules. All hyperparameters are contained within the `hparams`
        dictionary. Some training-related hyperparameters are common across almost all kinds of PyTorch modules,
        which can be overloaded by the child class. The module includes functions for training, evaluation, and
        prediction. These functions cane be modified or overloaded by any child subclass.

        ### Usage

        `net = PyTorchSmartModule(hparams)` where `hparams` is dictionary of hyperparameters containing the following:
            - `model_name` (str): Name of the model.
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
        
        ### Notes:
        - The `self.batch_input_shape` attribute must be set in the `__init__` method.
        - The `self.batch_output_shape` attribute must be set in the `__init__` method.
        """
        super(PyTorchSmartModule, self).__init__()
        if not hparams: hparams = self.sample_hparams
        self.hparams = hparams
        self._batchsize = int(hparams["batch_size"])
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
        self.batch_input_shape = (self._batchsize, 1)
        self.batch_output_shape = (self._batchsize, 1)
        if self._l2_reg > 0.0:
            if self._optimizerparams is not None:
                self._optimizerparams["weight_decay"] = self._l2_reg
            else:
                self._optimizerparams = {"weight_decay": self._l2_reg}
    
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

class Seq2Dense(PyTorchSmartModule):
    
    sample_hparams = {
        'model_name': 'dummy_Seq2Dense',
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
        'permute_output': False,
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

        `net = Seq2Dense(hparams)` where `hparams` is dictionary of hyperparameters containing the following:

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
            - `permute_output` (bool): Whether to permute the output sequence to be (N, D*H_out, L_out)
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
        super(Seq2Dense, self).__init__(hparams)
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
        self._permute_output = True if hparams.get("permute_output") else False
        self._N = int(hparams["batch_size"])
        self._batchsize = int(hparams["batch_size"])
        self._L_in = int(hparams["in_seq_len"])
        self._L_out = int(hparams["out_seq_len"]) if hparams.get("out_seq_len") else 1
        self._D = int(2 if self._bidirectional else 1)
        self._lstmprojsize = hparams["lstm_projsize"] if hparams.get("lstm_projsize") and \
            hparams.get("lstm_projsize") > 0 else (0 if self._rnntype=="LSTM" else None)
        self._H_in = int(self._infeatures)
        if self._rnnhidsizes == "auto": self._H_cell = int(2**(np.round(math.log2(self._H_in*self._L_in))))
        else: self._H_cell = int(self._rnnhidsizes)
        self._H_out = int(self._lstmprojsize if self._lstmprojsize and self._lstmprojsize > 0 else self._H_cell)
        self._rnn_output = torch.zeros((self._N, self._L_in, self._D*self._H_out))
        self._rnn_output_flattened = self._rnn_output[:,-1,:]
        self._densesizevec = []
        self._denselayers = []
        self.batch_input_shape = (self._N, self._L_in, self._H_in)
        if self._final_rnn_return_sequences and self._apply_dense_for_each_timestep:
            if self._permute_output: self.batch_output_shape = (self._N, self._outfeatures, self._L_out)
            else: self.batch_output_shape = (self._N, self._L_out, self._outfeatures)
        else: self.batch_output_shape = (self._N, self._L_out * self._outfeatures)
        
        # Constructing RNN layers
        if self._rnntype == "LSTM" and self._lstmprojsize is not None and self._lstmprojsize > 0:
            self.rnn = nn.LSTM(input_size=self._H_in, hidden_size=self._H_cell, num_layers=self._rnndepth,
                batch_first=True, dropout=self._rnndropout, bidirectional=self._bidirectional, 
                proj_size=self._lstmprojsize)
        else:
            self.rnn = self._rnn(input_size=self._H_in, hidden_size=self._H_cell, num_layers=self._rnndepth,
                batch_first=True, dropout=self._rnndropout, bidirectional=self._bidirectional)
        # for attrib in dir(self.rnn):
        #     if attrib.startswith("weight_ih"): xavier_uniform_(self.rnn.__getattr__(attrib))
        #     elif attrib.startswith("weight_hh"): orthogonal_(self.rnn.__getattr__(attrib))
        #     elif attrib.startswith("bias_"): zeros_(self.rnn.__getattr__(attrib))
        
        # Calculating Dense layers widths
        cf = self._L_in if (self._final_rnn_return_sequences and not self._apply_dense_for_each_timestep) else 1 
        self._dense_input_size = self._H_out * self._D * cf
        if self._densedepth > 0:
            if "list" in type(self._densehidsizes).__name__ or "numpy" in type(self._densehidsizes).__name__:
                self._densesizevec = self._densehidsizes
            elif self._densehidsizes == "auto":
                if self._final_rnn_return_sequences: old = self._dense_input_size
                else: old = self._H_cell
                for _ in range(self._densedepth):
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
                # xavier_uniform_(l.weight)
                # zeros_(l.bias)
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
        # xavier_uniform_(l.weight)
        # zeros_(l.bias)
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
        if self._final_rnn_return_sequences:
            if self._apply_dense_for_each_timestep: self._rnn_output_flattened = self._rnn_output
            else: self._rnn_output_flattened = self._rnn_output.view(self._rnn_output.shape[0], -1)
        else:
            # RNN output is of shape  (N, L, D * H_out)
            self._rnn_output_flattened = self._rnn_output[:,-1,:]
        out = self.decoder(self._rnn_output_flattened)
        if self.permute_output:
            return out.permute(0,2,1)
        else:
            return out



########################################################################################################################

class LanguageModel(Seq2Dense):
    def __init__(self, hparams:dict=None):
        super(LanguageModel, self).__init__(hparams)
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


########################################################################################################################


class ANN(PyTorchSmartModule):
    
    sample_hparams = {
        "model_name": "ANN",
        "input_size": 10,
        "output_size": 3,
        "width": "auto",
        "depth": 2,
        "hidden_activation": "relu",
        "hidden_activation_params": None,
        "output_activation": None,
        "output_activation_params": None,
        "batchnorm": "before",
        "batchnorm_params": None,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "learning_rate_decay_gamma": 0.99,
        "optimizer": "adam",
        "optimizer_params": {"eps": 1e-08},
        "batch_size": 32,
        "epochs": 2,
        "validation_tolerance_epochs": 2,
        "l2_reg": 0.0001,
        "loss_function": "categorical_crossentropy",
        "loss_function_params": None
    }
    
    
    def __init__(self, hparams:dict=None):
        """Typical Artificial Neural Network class, also known as multilayer perceptron.
        This class will create a fully connected feedforward artificial neural network.
        It can be used for classification, regression, etc.
        It basically encompasses enough options to build all kinds of ANNs with any number of 
        inputs, outputs, layers with custom or arbitrary width or depth, etc.
        Supports multiple activation functions for hidden layers and the output layer,
        but the activation function of the hidden layers are all the same.
        
        ### Usage
        `net = ANN(hparams)` where `hparams` is the dictionary of hyperparameters.

        It can include the following keys:
            - `input_size` (int): number of inputs to the ANN, i.e. size of the input layer.
            - `output_size` (int): number of outputs to predict, i.e. size of the output layer.
            - `width` ("auto"|int|list|array): hidden layer width. "auto" decides automatically, 
                a number sets them all the same, and a list/array sets each hidden layer according to the list.
                If "auto", hidden layer widths will be set in such a way that the first half of the network will be the 
                encoder and the other half will be the decoder.
                Therefore, the first hidden layer will be twice as large as the input layer, 
                and every layer of the encoder will be twice as large as the previous one.
                In the decoder half, layer width will be halved until the output layer. 
                Layer widths will be powers of two.
            - `depth` (int): Specifies the depth of the network (number of hidden layers).
                It must be specified unless `width` is provided as a list. Then the depth will be inferred form it.
            - `hidden_activation` (str): Activation of the hidden layers.
                Supported activations are lowercase module names, e.g. "relu", "logsoftmax", etc.
            - `hidden_activation_params` (dict): Parameters for the hidden activation function, if any.
            - `output_activation` (str): Activation of the output layer, if any.
                **Note**: For classification problems, you may want to choose "sigmoid", "softmax" or "logsoftmax".
                That being said, you usually don't need to specify an activation for the output layer at all.
                Some loss functions in PyTorch have the classification activation functions embedded in them.
                **Note**: For regression problems, no activation is needed. It is by default linear, 
                unless you want to manually specify an activation.
            - `output_activation_params` (dict): Parameters for the output activation function, if any.
            - `batchnorm` (str): If given, specifies where the batch normalization layer should be included: 
                `"before"` the activation, or `"after"` it.
                For activation functions such as **ReLU** and **sigmoid**, `"before"` is usually a better option. 
                For **tanh** and **LeakyReLU**, `"after"` is usually a better option.
            - `batchnorm_params` (dict): Dictionary of parameters for the batch normalization layer.
            - `dropout` (float): If given, specifies the dropout rate after every 
                hidden layer. It should be a probability value between 0 and 1.
            - `learning_rate` (float): Initial learning rate of training.
            - `learning_rate_decay_gamma` (float): Exponential decay rate gamma for learning rate, if any.
            - `optimizer` (str): Optimizer, options are "sgd" and "adam" for now.
            - `optimizer_params` (dict): Additional parameters of the optimizer, if any.
            - `batch_size` (int): Minibatch size for training.
            - `epochs` (int): Maximum number of epochs for training.
            - `validation_tolerance_epochs` (int): Epochs to tolerate unimproved val loss, before early stopping.
            - `l2_reg` (float): L2 regularization parameter.
            - `loss_function` (str): Loss function, options are "mse", "mae", "binary_crossentropy", 
                "categorical_crossentropy", "crossentropy", "kldiv", "nll".
            - `loss_function_params` (dict): Additional parameters for the loss function, if any.
            - `metrics` (list): List of metrics to be evaluated during training (so far unused).

        ### Returns
        It returns a `torch.nn.Module` object that corresponds with an ANN model.
        run `print(net)` afterwards to see what the ANN holds.
        """
        super(ANN, self).__init__(hparams)
        self.layers = []
        self._sizevec = []
        self._insize = hparams["input_size"]
        self._outsize = hparams["output_size"]
        self._dropout = hparams.get("dropout")
        self._width = hparams.get("width")
        self._depth = hparams.get("depth")
        self._denseactivation = actdict_pytorch[hparams["hidden_activation"]]
        self._denseactivation_params = hparams.get("hidden_activation_params")
        self._outactivation = \
            actdict_pytorch[hparams.get("output_activation")] if hparams.get("output_activation") else None
        self._outactivation_params = hparams.get("output_activation_params")
        self._batchnorm = hparams.get("batchnorm")
        self._batchnorm_params = hparams.get("batchnorm_params")
        self._batchsize = hparams.get("batch_size")
        self.batch_input_shape = (self._batchsize, self._insize)
        self.batch_output_shape = (self._batchsize, self._outsize)
        
        # Constructing the layer size vector (does not include input and output layers)
        if "list" in type(self._width).__name__ or "numpy" in type(self._width).__name__:
            self._sizevec = self._width
        elif self._width == "auto":
            old = int(2**np.ceil(math.log2(self._insize)))
            for i in range(self._depth):
                new = int((2 if i < np.ceil(self._depth/2) else 0.5)*2**np.round(math.log2(old)))
                old = new
                self._sizevec.append(new)
        elif self._width is not None:
            self._sizevec = [self._width for _ in range(self._depth)]
        else:
            self._width = self._insize
            self._sizevec = [self._width for _ in range(self._depth)]
        
        # Constructing layers
        old = self._insize
        new = self._sizevec[0]
        for width in self._sizevec:
            new = width
            l = nn.Linear(old, new)
            # xavier_uniform_(l.weight)
            # zeros_(l.bias)
            self.layers.append(l)
            if self._batchnorm == "before":
                if self._batchnorm_params:
                    self.layers.append(nn.BatchNorm1d(new, **self._batchnorm_params))
                else:
                    self.layers.append(nn.BatchNorm1d(new))
            if self._denseactivation_params:
                self.layers.append(self._denseactivation(**self._denseactivation_params))
            else:
                self.layers.append(self._denseactivation())
            if self._batchnorm == "after":
                if self._batchnorm_params:
                    self.layers.append(nn.BatchNorm1d(new, **self._batchnorm_params))
                else:
                    self.layers.append(nn.BatchNorm1d(new))
            if self._dropout:
                self.layers.append(nn.Dropout(self._dropout))
            old = new
        self.layers.append(nn.Linear(old, self._outsize))
        if self._outactivation:
            if self._outactivation_params:
                self.layers.append(self._outactivation(**self._outactivation_params))
            else:
                self.layers.append(self._outactivation())
        
        # Sequentiating the layers
        self.net = nn.Sequential(*self.layers)


    def forward(self, x):
        return self.net(x)


########################################################################################################################

class Image2Dense1D(PyTorchSmartModule):
    sample_hparams = {
        "model_name": "Image2Dense1D",
        # General hyperparameters
        "in_features": 6,
        "out_features": 3,
        # Sequence hyperparameters
        "in_seq_len": 32,
        "out_seq_len": 1,
        # Convolution hyperparameters
        "num_conv_blocks": 2,
        "conv_channels": "auto",
        "conv_kernel_size": 4,
        "conv_padding": "valid",
        "conv_activation": "relu",
        "conv_activation_params": None,
        "conv_batchnorm": "after",
        "conv_batchnorm_params": None,
        "conv_stride": 1,
        "conv_dropout": 0.1,
        "pool_padding": "valid",
        "pool_kernel_size": 8,
        "pool_stride": 1,
        "min_image_size": 4,
        # Dense hyperparameters
        "dense_width": "auto",
        "dense_depth": 2,
        "dense_dropout": 0.2,
        "dense_activation": "relu",
        "dense_activation_params": None,
        "output_activation": "softmax",
        "output_activation_params": None,
        "dense_batchnorm": "before",
        "dense_batchnorm_params": None,
        # Training hyperparameters
        "l2_reg": 0.0001,
        "batch_size": 32,
        "epochs": 4,
        "validation_data": 0.2,
        "validation_tolerance_epochs": 2,
        "learning_rate": 0.0001,
        "learning_rate_decay_gamma": 0.99,
        "loss_function": "categorical_crossentropy",
        "metrics": ["accuracy"],
        "optimizer": "adam",
        "optimizer_params": {"eps": 1e-07}
    }
    
    
    def __init__(self, hparams:dict=None):
        """1D Image to Dense network with CNN for time-series classification, regression, and forecasting.
        This network uses CNN layers as encoders to extract information from input sequences, and fully-connected 
        multilayer perceptrons (Dense) to decode the sequence into an output, which can be class probabilitites 
        (timeseries classification), a continuous number (regression), or an unfolded sequence (forecasting) of a 
        target timeseries. Please note that unlike RNN, a 1D CNN accepts inputs as (N, C_in, L) and gives outputs as
        (N, C_out, L).

        ### Usage

        `model = Image2Dense1D(hparams)` where `hparams` is dictionary of hyperparameters containing following:

            - `in_seq_len` (int): Input sequence length, in number of timesteps
            - `out_seq_len` (int): Output sequence length, in number of timesteps, assuming output is also a sequence.
            Use 1 for when the output is not a sequence, or do not supply this key.
            - `in_features` (int): Number of features of the input.
            - `out_features` (int): Number of features of the output.
            - `num_conv_blocks` (int): Number of convolutional blocks.
            - `conv_kernel_size` (int): Kernel size of the convolutional layers.
            - `pool_kernel_size` (int): Kernel size of the pooling layers.
            - `conv_padding` (int|str): Padding of the convolutional layers.
            - `pool_padding` (int|str): Padding of the pooling layers.
            - `conv_activation` (str): Activation function of the convolutional layers.
            - `conv_activation_params` (dict): Dictionary of parameters for the convolution activation function.
            - `conv_batchnorm` ("before"|"after"|None): Using batch normalization in the convolutional layers.
            - `conv_batchnorm_params` (dict): Dictionary of parameters for the convolution batch normalization layer.
            - `conv_stride` (int): Stride of the convolutional layers.
            - `pool_stride` (int): Stride of the pooling layers.
            - `conv_dropout` (float|None): Dropout rate of the convolutional layers.
            - `min_image_size` (int): Minimum size of the image to be reduced to in convolutions and poolings.
                After this point, the padding and striding will be chosen such that image size does not 
                decrease further.
            - `dense_width` ("auto"|int|list|array): Width of the hidden layers of the Dense network. 
                "auto", a number (for all of them) or a vector holding width of each hidden layer.
            - `dense_depth` (int): Depth (number of hidden layers) of the Dense network.
            - `dense_activation` (str): Activation function for hidden layers of the Dense network.
                Supported activations are the lower-case names of their torch modules,
                e.g. "relu", "sigmoid", "softmax", "logsoftmax", "tanh", "leakyrelu", "elu", "selu", "softplus", etc.
            - `dense_activation_params` (dict): Dictionary of parameters for the dense activation function.
            - `output_activation` (str): Activation function for the output layer of the Dense network, if any.
                **NOTE** If the loss function is cross entropy, then no output activation is erquired, 
                though during inference you have to manually add a logsoftmax.
                However, if the loss function is nll (negative loglikelihood), 
                then you must specify an output activation as in "logsoftmax".
            - `output_activation_params` (dict): Dictionary of parameters for the output activation function.
            - `dense_batchnorm` (str): Whether the batch normalization layer (if any) 
                should come before or after the activation of each hidden layer in the dense network.
                For activation functions such as **ReLU** and **sigmoid**, `"before"` is usually a better option. 
                For **tanh** and **LeakyReLU**, `"after"` is usually a better option.
            - `dense_batchnorm_params` (dict): Dictionary of parameters for the dense batch normalization layer.
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
            - `metrics` (list): List of metrics to be evaluated during training. Currently unused.
        
        ### Returns
        
        - Returns a `nn.Module` object that can be trained and used accordingly.
        - Run `print(net)` afterwards to see what you have inside the network.
        - **NOTE** CNN kernel initializers and Dense kernel initializers are Glorot Uniform.
        - **NOTE** All bias initializers are zeros.
        """
        super(Image2Dense1D, self).__init__(hparams)
        # General parameters
        self._infeatures = hparams["in_features"]
        self._outfeatures = hparams["out_features"]
        
        # Convolutional layers hyperparameters
        self._num_conv_blocks = hparams.get("num_conv_blocks")    
        self._conv_channels = hparams.get("conv_channels")
        self._conv_channels_vec = []
        self._min_image_size = hparams["min_image_size"] if hparams.get("min_image_size") else 1
        self._img_size = hparams["in_seq_len"]
        self._img_size_list = [[self._img_size, self._infeatures]]
        
        if "list" in type(self._conv_channels).__name__.lower() or\
            "numpy" in type(self._conv_channels).__name__.lower():
            self._conv_channels_vec = self._conv_channels
        elif self._conv_channels == "auto":
            old = int(2**math.ceil(math.log2(self._infeatures)))
            for i in range(self._num_conv_blocks):
                #new = old*2 if i <= self._num_conv_blocks//2 else old//2
                new = old * 2
                old = new
                self._conv_channels_vec.append(new)
        elif "int" in type(self._conv_channels).__name__.lower():
            assert self._num_conv_blocks is not None, \
                "If `conv_channels` is an int, then `num_conv_blocks` must be specified."
            self._conv_channels_vec = [self._conv_channels]*self._num_conv_blocks
        else:
            raise ValueError("Invalid type for `conv_channels` key of hparams, or it does not exist.")
        
        self._conv_kernel_size = hparams.get("conv_kernel_size")
        self._pool_kernel_size = hparams.get("pool_kernel_size")
        self._conv_padding = hparams["conv_padding"] if hparams.get("conv_padding") else "same"
        self._pool_padding = hparams["pool_padding"] if hparams.get("pool_padding") else "same"
        self._conv_activation = hparams["conv_activation"] if hparams.get("conv_activation") else "relu"
        self._conv_activation_params = hparams.get("conv_activation_params")
        self._conv_batchnorm = hparams.get("conv_batchnorm")
        self._conv_batchnorm_params = hparams.get("conv_batchnorm_params")
        self._conv_stride = hparams["conv_stride"] if hparams.get("conv_stride") else 1
        self._pool_stride = hparams["pool_stride"] if hparams.get("pool_stride") else 1
        self._conv_dropout = hparams.get("conv_dropout")
        
        # Update hparams for later use
        self.hparams["conv_padding"] = self._conv_padding
        self.hparams["pool_padding"] = self._pool_padding
        self.hparams["conv_stride"] = self._conv_stride
        self.hparams["pool_stride"] = self._pool_stride
        
        # Dense layers hyperparameters
        self._denseactivation = hparams["dense_activation"] if hparams.get("dense_activation") else "relu"
        self._denseactivation_params = hparams.get("dense_activation_params")
        self._outactivation = hparams.get("output_activation") if hparams.get("output_activation") else None
        self._outactivation_params = hparams.get("output_activation_params")
        self._densebatchnorm = hparams.get("dense_batchnorm")
        self._densebatchnormparams = hparams.get("dense_batchnorm_params")
        self._densehidsizes = hparams["dense_width"]
        self._densedepth = hparams["dense_depth"]
        self._densedropout = hparams["dense_dropout"] if hparams.get("dense_dropout") else 0
        
        # Configuring sizes
        self._N = int(hparams["batch_size"])
        self._L_in = int(hparams["in_seq_len"])
        self._l = self._L_in
        self._L_out = int(hparams["out_seq_len"])
        self.batch_input_shape = (self._N, self._infeatures, self._L_in)
        self.batch_output_shape = (self._N, self._L_out * self._outfeatures)
        
        # Some empty initializations
        self._densesizevec = []
        self._layers_vec = []
        
        # Constructing the encoder (convolutional blocks)
        in_channels = self._infeatures
        for i in range(self._num_conv_blocks):
            out_channels = self._conv_channels_vec[i]
            self._layers_vec.append(nn.Conv1d(in_channels, out_channels, kernel_size=self._conv_kernel_size,
                stride=self._conv_stride, padding=self._conv_padding if self._l >= self._min_image_size else "same"))
            self._update_image_size(out_channels, 'conv')
            in_channels = out_channels
            if self._conv_batchnorm=='before':
                if self._conv_batchnorm_params:
                    self._layers_vec.append(nn.BatchNorm1d(**self._conv_batchnorm_params))
                else:
                    self._layers_vec.append(nn.BatchNorm1d())
            if self._conv_activation_params:
                self._layers_vec.append(actdict_pytorch[self._conv_activation](**self._conv_activation_params))
            else:
                self._layers_vec.append(actdict_pytorch[self._conv_activation]())
            if self._conv_batchnorm=='after':
                if self._conv_batchnorm_params:
                    self._layers_vec.append(nn.BatchNorm1d(**self._conv_batchnorm_params))
                else:
                    self._layers_vec.append(nn.BatchNorm1d())
            self._layers_vec.append(nn.MaxPool1d(kernel_size=self._pool_kernel_size, stride=self._pool_stride,
                padding=self._pool_padding if self._l >= self._min_image_size else "same"))
            self._update_image_size(out_channels, 'pool')
            if self._conv_dropout:
                self._layers_vec.append(nn.Dropout1d(self._conv_dropout))
            
        # Flattening (Image embedding)
        self._layers_vec.append(nn.Flatten())
        
        # Calculating Dense layers widths
        if "list" in type(self._densehidsizes).__name__.lower() or \
            "numpy" in type(self._densehidsizes).__name__.lower():
            self._densesizevec = self._densehidsizes
        elif self._densehidsizes == "auto":
            old = self._l * self._conv_channels_vec[-1]
            for i in range(self._densedepth):
                new = old//2 if old//2 > self._outfeatures*self._L_out else old
                old = new
                self._densesizevec.append(new)
        elif "int" in type(self._densehidsizes).__name__:
            self._densesizevec = [self._densehidsizes for _ in range(self._densedepth)]
        else:
            raise ValueError("Dense layer widths must be either a list of integers, an integer or 'auto'.")   
        
        # Constructing Dense layers
        oldsize = self._l * self._conv_channels_vec[-1]
        for size in self._densesizevec:
            self._layers_vec.append(nn.Linear(oldsize, size))
            oldsize = size
            if self._densebatchnorm == "before":
                if self._densebatchnormparams:
                    self._layers_vec.append(nn.BatchNorm1d(**self._densebatchnormparams))
                else:
                    self._layers_vec.append(nn.BatchNorm1d())
            if self._denseactivation_params:
                self._layers_vec.append(actdict_pytorch[self._denseactivation](**self._denseactivation_params))
            else:
                self._layers_vec.append(actdict_pytorch[self._denseactivation]())
            if self._densebatchnorm == "after":
                if self._densebatchnormparams:
                    self._layers_vec.append(nn.BatchNorm1d(**self._densebatchnormparams))
                else:
                    self._layers_vec.append(nn.BatchNorm1d())
            if self._densedropout:
                self._layers_vec.append(nn.Dropout(self._densedropout))
        self._layers_vec.append(nn.Linear(oldsize,int(self._L_out*self._outfeatures)))
        if self._outactivation:
            if self._outactivation_params:
                self._layers_vec.append(actdict_pytorch[self._outactivation](**self._outactivation_params))
            else:
                self._layers_vec.append(actdict_pytorch[self._outactivation]())

    def _calc_size(self, size_in:int, padding:int, kernel_size:int, stride:int):
        if padding == 'valid':
            padding = 0
        if padding == 'same':
            return size_in
        else:
            return math.floor((size_in + 2*padding - (kernel_size-1) - 1)/stride + 1)
    
    def _update_image_size(self, out_channels, ops:str='conv'):
        self._l = self._calc_size(
            self._l, 
            self.hparams[ops+'_padding'], 
            self.hparams[ops+'_kernel_size'], 
            self.hparams[ops+'_stride'])
        self._img_size = [self._l, out_channels]
        #print("new size: ",self._img_size)
        self._img_size_list.append(self._img_size)
    
    def forward(self, inputs):
        x = inputs
        for layer in self._layers_vec:
            x = layer(x)
        return x



















########################################################################################################################

class Dummy(PyTorchSmartModule):
    sample_hparams = {
        'model_name': 'dummy_Pytorch_Smart_Module',
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
        'optimizer_params': {'eps': 1e-07},
        'some_new_feature': True
    }
    def __init__(self, hparams:dict=None):
        super(Dummy, self).__init__(hparams)
        self._some_new_feature = self.hparams.get("some_new_feature")
    def forward(self, x):
        return x
    
########################################################################################################################


if __name__ == '__main__':
    
    # ----------------------------------
    model = Dummy()
    print(model.hparams)
    print(model._some_new_feature)
    print(model._optimizerparams)
    # ----------------------------------
    