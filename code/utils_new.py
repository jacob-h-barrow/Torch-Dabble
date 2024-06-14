import torch
import torch.nn as nn
import numpy as np
import json
import uuid

from torch_codecs import NumpyEncoder, NumpyDecoder
from models.basic_nn_module import Weights, Model
from dataclasses import dataclass
from path import Path

import sys

sys.path.append('/home/parzival/Desktop/DL_2023/')

from proc_gpu import GpuResourceUse

@dataclass
class Epoch:
    amount: int
    loss: float
    resources_consumed: GpuResourceUse
    state_dict: str # json

@dataclass
class ModelRun:
    epochs: List[Epoch]
    total_energy_consumed: float
    
    def __post_init__(self):
        # Convert later
        pass

@dataclass
class Model:
    _type: str
    code: Path
    id: uuid.UUID = field(init=True, default_factory=uuid.uuid4)

##################################
########### ChatGPT ##############
##################################
def export_state_dict_to_numpy(model):
    state_dict = model.state_dict()
    state_dict_cpu = {key: value.cpu() for key, value in state_dict.items()}
    state_dict_np = {key: value.numpy() for key, value in state_dict_cpu.items()}
    
    return state_dict_np
    
def export_state_dict_to_json(state_dict_np):
    res = {}
    
    for i, (layer, weights) in enumerate(state_dict_np.items()):
        res[i] = (layer, json.dumps(weights, cls=NumpyEncoder))
    
    return res
    
def load_state_dict_from_json(state_dict_np_json):
    state_dict_np = {}
    
    for i, (layer, weights_json) in state_dict_np_json.items():
        weights = json.loads(weights_json, cls=NumpyDecoder)
        state_dict_np[layer] = weights

    return state_dict_np
    
def test_state_dict_eq(state_dict_np_json, model):
    state_dict_np_from_json = load_state_dict_from_json(state_dict_np_json)
    state_dict_np_from_model = export_state_dict_to_numpy(model)
    
    for layer in state_dict_np_from_model:
        if not np.array_equal(state_dict_np_from_model[layer], state_dict_np_from_json[layer]):
            return False
            
    return True
##################################
########### ChatGPT ##############
##################################
