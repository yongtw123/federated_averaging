import torch
import numpy as np
import math
import copy
from collections import defaultdict
from typing import Dict, Any
from syft.workers.base import BaseWorker
from syft.frameworks.torch.fl import BaseDataset, FederatedDataset

__MUTUAL_DERANGEMENTS_10 = [
    [2, 9, 6, 4, 0, 3, 1, 7, 8, 5],
    [9, 5, 3, 0, 8, 4, 2, 1, 6, 7],
    [5, 7, 1, 8, 9, 0, 6, 2, 4, 3],
    [7, 6, 5, 2, 1, 8, 3, 4, 0, 9], 
    [6, 3, 7, 5, 4, 1, 0, 9, 2, 8], 
    [8, 1, 4, 3, 6, 2, 9, 5, 7, 0], 
    [3, 2, 0, 6, 7, 9, 5, 8, 1, 4], 
    [1, 8, 2, 9, 3, 7, 4, 0, 5, 6], 
    [0, 4, 9, 7, 2, 5, 8, 6, 3, 1], 
    [4, 0, 8, 1, 5, 6, 7, 3, 9, 2]
]

__MUTUAL_DERANGEMENTS_5 = [
    [2, 9, 6, 4, 0], 
    [3, 1, 7, 8, 5], 
    [9, 5, 3, 0, 8],
    [4, 2, 1, 6, 7], 
    [7, 6, 5, 2, 1], 
    [0, 8, 9, 3, 4], 
    [1, 0, 4, 9, 3], 
    [6, 7, 8, 5, 2], 
    [8, 4, 2, 7, 6], 
    [5, 3, 0, 1, 9]
]

__MUTUAL_DERANGEMENTS_2 = [
    [2, 3], 
    [9, 1], 
    [6, 7], 
    [4, 8], 
    [0, 5], 
    [5, 4], 
    [1, 2], 
    [8, 9], 
    [3, 6], 
    [7, 0]
]

def copy_model(dst_model, src_model, const=0.0):
    """Copies tensor contents (.data) from src_model.named_parameters to dst_model.named_parameters.
       Modifies dst_model.
    """
    
    params_dst = dst_model.named_parameters()
    params_src = src_model.named_parameters()
    dict_params_dst = dict(params_dst)
    with torch.no_grad():
        for name, param in params_src:
            if name in dict_params_dst:
                # NOTE: Must add a dummy float otherwise only setting 'reference' to old param.data
                dict_params_dst[name].set_(param.data + const)


def add_model(dst_model, src_model, scale=1.0):
    """Add the parameters of two models.
    Args:
        dst_model (torch.nn.Module): the model to which the src_model will be added.
        src_model (torch.nn.Module): the model to be added to dst_model.
        scale (int): scaling of src_model
    """

    params1 = src_model.named_parameters()
    params2 = dst_model.named_parameters()
    dict_params2 = dict(params2)
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].set_(dict_params2[name1].data + param1.data * scale)


def scale_model(model, scale):
    """Scale the parameters of a model.
    Args:
        model (torch.nn.Module): the models whose parameters will be scaled.
        scale (float): the scaling factor.
    """
    params = model.named_parameters()
    dict_params = dict(params)
    with torch.no_grad():
        for name, param in dict_params.items():
            dict_params[name].set_(dict_params[name].data * scale)


def fedavg(version, models, n, global_model=None):
    if version == 1:
        federated_average_v1(models, n, global_model)
    elif version == 2:
        federated_average_v2(models, n, global_model)
    elif version == 3:
        federated_average_v3(models, global_model, n)
    else:
        raise ValueError("Bad version %d given for fedavg." % version)
        

def federated_average_v1(models: Dict[Any, torch.nn.Module], n:Dict[Any, int], global_model=None):
    """Calculate the federated average of a dictionary containing models.
       Modifies models in-place.
    Args:
        models (Dict[Any, torch.nn.Module]): a dictionary of models
        for which the federated average is calculated.
        n (Dict[Any, int]): a dictionary containing number of inputs for model;
        corresponding to keys in models
    """
    nr_models = len(models)
    assert nr_models > 0
    
    model_keys = list(models.keys())
    for k in model_keys:
        scale_model(models[k], n[k])
        
    pivot_key = model_keys.pop()
    for k in model_keys:
        add_model(models[pivot_key], models[k])
    scale_model(models[pivot_key], 1.0 / sum(n.values()))
    for k in model_keys:
        copy_model(models[k], models[pivot_key])
    
    if global_model is not None:
        copy_model(global_model, models[pivot_key])
    

def federated_average_v2(models: Dict[Any, torch.nn.Module], n:Dict[Any, int], global_model=None):
    """Uses state_dict instead of (named_)parameters"""
    nr_models = len(models)
    assert nr_models > 0
    
    model_keys = list(models.keys())
    avg_dict = None
    pivot_key = None
    
    for k in model_keys:
        if not avg_dict:
            pivot_key = k
            avg_dict = copy.deepcopy(models[pivot_key].state_dict())
            for name in avg_dict.keys():
                avg_dict[name] *= n[pivot_key]
            continue
        temp_dict = models[k].state_dict()
        for name in avg_dict.keys():
            avg_dict[name] += temp_dict[name] * n[k]
    for name in avg_dict.keys():
        avg_dict[name] /= sum(n.values())
    for k in model_keys:
        models[k].load_state_dict(avg_dict)
    
    if global_model is not None:
        global_model.load_state_dict(avg_dict)


def federated_average_v3(models, global_model, n):
    """Modifies local models [models] and global model [global_model]"""
    assert isinstance(global_model, torch.nn.Module), "federated_average_v3 expects a Torch model for global_model"
    nr_models = len(models)
    assert nr_models > 0
    
    model_keys = list(models.keys())
    avg_dict = None
    pivot_key = model_keys.pop()
    
    avg_dict = copy.deepcopy(models[pivot_key].state_dict()) # Initialize global model
    for name in avg_dict.keys():
        avg_dict[name] *= n[pivot_key] # Scale local model that was added first
    scale_model(models[pivot_key], n[pivot_key]) # Scale pivot local model in-place
    
    for k in model_keys:
        temp_dict = models[k].state_dict()
        for name in avg_dict.keys():
            avg_dict[name] += temp_dict[name] * n[k] # Scale and add local model to global model
        scale_model(models[k], n[k]) # Scale local model in-place ...
        add_model(models[pivot_key], models[k]) # and add it to pivot local model
        
    for name in avg_dict.keys():
        avg_dict[name] /= sum(n.values()) # Average global model
    global_model.load_state_dict(avg_dict) # Update global model
    
    scale_model(models[pivot_key], 1.0 / sum(n.values())) # Average pivot local model...
    for k in model_keys:
        copy_model(models[k], models[pivot_key]) # then copy pivot local model to all other models


def fedadam(optimizers: Dict[Any, torch.optim.Optimizer], n:Dict[Any, int]):
    """Averages 1st and 2nd moments of optimizers"""
    nr_optims = len(optimizers)
    assert nr_optims > 0
    
    optim_keys = list(optimizers.keys())
    avg_dict = None
    avg_dict_keys = []
    pivot_key = None
    
    for k in optim_keys:
        # Pick a pivot
        if not avg_dict:
            pivot_key = k
            avg_dict = copy.deepcopy(optimizers[pivot_key].state_dict()['state'])
            avg_dict_keys = avg_dict.keys() # Respect PyTorch insertion order
            for id in avg_dict_keys:
                avg_dict[id]['exp_avg'] *= n[pivot_key]
                avg_dict[id]['exp_avg_sq'] *= n[pivot_key]
            continue
        
        # Sum at pivot
        temp_dict = optimizers[k].state_dict()['state']
        for id1,id2 in zip(avg_dict_keys, temp_dict.keys()):
            assert avg_dict[id1]['exp_avg'].shape == temp_dict[id2]['exp_avg'].shape
            avg_dict[id1]['exp_avg'] += temp_dict[id2]['exp_avg'] * n[k]
            
            assert avg_dict[id1]['exp_avg_sq'].shape == temp_dict[id2]['exp_avg_sq'].shape
            avg_dict[id1]['exp_avg_sq'] += temp_dict[id2]['exp_avg_sq'] * n[k]
    
    # Divide pivot by N
    for id in avg_dict_keys:
        avg_dict[id]['exp_avg'] /= sum(n.values())
        avg_dict[id]['exp_avg_sq'] /= sum(n.values())
        
    # Copy averaged moments
    for k in optim_keys:
        optim_state = copy.deepcopy(optimizers[k].state_dict())
        for id1,id2 in zip(optim_state['state'].keys(), avg_dict_keys):
            assert optim_state['state'][id1]['exp_avg'].shape == avg_dict[id2]['exp_avg'].shape
            optim_state['state'][id1]['exp_avg'] = avg_dict[id2]['exp_avg'] + 0.
            
            assert optim_state['state'][id1]['exp_avg_sq'].shape == avg_dict[id2]['exp_avg_sq'].shape
            optim_state['state'][id1]['exp_avg_sq'] = avg_dict[id2]['exp_avg_sq'] + 0.
        optimizers[k].load_state_dict(optim_state)


def federate_dataset(dataset, workers, classnum, scheme, 
                     class_per_worker=1, uniqueness_threshold=0, custom_mapping=None, use_pysyft=True):
    """Adapted from https://github.com/OpenMined/PySyft/blob/master/syft/frameworks/torch/fl/dataset.py
    dataset: pytorch Dataset
    workers: List[Any], can be of pysyft virtualworkers
    Assert shard_per_class * len(classes) == class_per_worker * len(workers)
    classnum: int, number of classes
    scheme: can be 'naive', 'permuted', 'choose-unique', 'custom'
    custom_mapping: when scheme is 'custom', provide Dict[worker, List[int]
    Example configurations:
    1) 1 worker to 1 *full* class, implies len(classes)==len(workers)  (1,1)
    2) 1 worker to 2 *full* classes, implies len(classes)==2*len(workers) (1,2)
    3) 1 worker to 1 class, a class has 2 shards, implies 2*len(classes)==len(workers) (2,1)
    4) 1 worker to 1 class, a class has n shards, implies n*len(classes)==len(workers) (n,1)
    5) 1 worker to 2 classes, a class has 2 shards, implies len(classes)==len(workers) but a worker
       should not have the same 2 classes (2,2)
    6) ... vice versa
    TODO: this implementation requires len(classes) be divisible by class_per_worker...
    
    Returns a Pysyft FederatedDataset or a Dict {worker_id:int : [x:Tensor, y:Tensor]}
      AND a meta-data object
    """
    
    assert (class_per_worker * len(workers)) % classnum == 0, \
           "class per worker (%d) * number of workers (%d) must be a multiple of number of classes (%d)" \
           % (class_per_worker, len(workers), classnum)
    assert classnum % class_per_worker == 0, \
           "Limitation: requires number of classes (%d) be a multiple of class per worker (%d)" \
           % (classnum, class_per_worker)
    
    shard_per_class = int(class_per_worker * len(workers) / classnum)
    meta_data = {}
    with torch.no_grad():
        # Prepare the shards
        gen = dataset.enumerate_by_class() #input:tensor, target:int, output:tensor
        class_shards = defaultdict(list)
        class_index_map = []
        for data in gen:
            class_idx = data[0]['target']
            class_index_map.append(class_idx)
            x = torch.stack(list(map(lambda d: d['input'], data)), 0)
            y = torch.stack(list(map(lambda d: d['output'], data)), 0)
            shard_size = math.ceil(len(x) / shard_per_class)
            x = torch.split(x, shard_size)
            y = torch.split(y, shard_size) # tuple
            for shard in zip(x, y):
                class_shards[class_idx].append((shard[0], shard[1]))
        
        # Distribute and send shards
        class_indices = list(class_shards.keys())
        range_temp = None
        fed_datasets= {}
        
        for idx, worker in enumerate(workers):            
            # Determine classes for worker
            if scheme == 'naive':
                offset = math.floor(idx / shard_per_class) * class_per_worker
                class_indices_range = range(offset, offset+class_per_worker)
            elif scheme == 'permuted':
                if not range_temp:
                    range_temp = np.split(np.random.permutation(classnum), int(classnum / class_per_worker))
                class_indices_range = range_temp.pop()
            elif scheme == 'choose-unique':
                # NOTE: careful of upper bound for # workers = {classnum} C {class_per_worker}
                # NOTE: if parameters too stringent can loop forever... need mathematical upper bound?
                if not range_temp:
                    range_temp = {'history': [], 'buffer': []}
                if not range_temp['buffer']:
                    range_temp['buffer'] = list(range(classnum))
                    
                candidate = []
                while len(candidate) == 0 or \
                      set(candidate) in range_temp['history'] or \
                      any([len(set(candidate)&set(h)) > uniqueness_threshold for h in range_temp['history']]):
                    candidate = np.random.choice(range_temp['buffer'], class_per_worker, replace=False)
                    
                range_temp['buffer'] = [e for e in range_temp['buffer'] if e not in candidate]
                range_temp['history'].append(set(candidate))
                class_indices_range = candidate
            elif scheme == 'custom':
                assert custom_mapping is not None
                map_index = worker.id if isinstance(worker, BaseWorker) else worker
                assert map_index in custom_mapping, "Worker id %s not in custom mapping %s" % (str(map_index), str(list(custom_mapping.keys())))
                class_indices_range = [class_index_map.index(cls) for cls in custom_mapping[map_index]]
            else:
                raise ValueError('Bad scheme provided for federate_dataset')
        
            # Collect shards
            xs = []
            ys = []
            for class_index in class_indices_range:
                shard = class_shards[class_index_map[class_index]].pop()
                xs.append(shard[0])
                ys.append(shard[1])
            xs = torch.cat(xs, 0)
            ys = torch.cat(ys, 0)
            
            fed_datasets[worker] = [xs, ys]
            
            # Get some meta-data
            meta_data[worker] = {
                'xshape': list(fed_datasets[worker][0].shape),
                'yshape': list(fed_datasets[worker][1].shape),
                'yset': torch.unique(fed_datasets[worker][1], sorted=True).int().tolist()
            }
        # End looping over workers
    # End torch.no_grad
    
    
    
    
    # Send shards if Pysyft
    if use_pysyft:
        for worker in fed_datasets:
            print("Sending data to worker %s ..." % worker.id)
            fed_datasets[worker][0] = fed_datasets[worker][0].send(worker)
            fed_datasets[worker][1] = fed_datasets[worker][1].send(worker)
        return FederatedDataset([BaseDataset(data[0], data[1]) for data in fed_datasets.values()]), meta_data
    else:
        return fed_datasets, meta_data
