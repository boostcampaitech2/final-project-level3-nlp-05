from collections import defaultdict
from typing import Union, List, Dict, NoReturn
import timeit

import torch
import torch.nn as nn

def collate_fn(
    batched_samples: List[Dict[str, List[int]]],
    pad_token_idx: int,
    pad_keys: List[str] = ["input_ids", "labels"],
    sort_by_length: bool = True
) -> Dict[str, torch.Tensor]:
    
    if sort_by_length:
        batched_samples = sorted(batched_samples, key=lambda x: len(x["input_ids"]), reverse=True)

    keys = batched_samples[0].keys()
    outputs = defaultdict(list)

    for key in keys:
        for sample in batched_samples:
            if sample[key] is not None:
                if not isinstance(sample[key], torch.Tensor):
                    sample[key] = torch.tensor(sample[key])
                outputs[key].append(sample[key])
            else:
                outputs[key] = None
        PAD = pad_token_idx if key in pad_keys else 0
        PAD = -1 if key == "answers" else PAD
        
        if outputs[key] is not None:
            outputs[key] = torch.nn.utils.rnn.pad_sequence(outputs[key], padding_value=PAD, batch_first=True)

    return dict(outputs)

def freeze(
    model: nn.Module,
    name: Union[str, List[str]],
    exact: bool = False,
) -> List[str]:
    """Freeze layers whose names correspond to the `name` parameter given.
    Args:
        model (nn.Module)
        name (str or List[str])
        exact (bool): (default: False)
    Returns:
        List[str] - list of frozen layers including previously frozen ones.
    """
    def _freeze_exact(model, name):
        for n, p in model.named_parameters():
            if n == name:
                p.requires_grad = False
    
    def _freeze(model, name):
        for n, p in model.named_parameters():
            if n.count(name):
                p.requires_grad = False
    
    if not isinstance(name, list):
        name = [name]

    for n in name:
        if exact:
            _freeze_exact(model, n)
        else:
            _freeze(model, n)

    return [n for n, p in model.named_parameters() if not p.requires_grad]

def unfreeze_all(model: nn.Module) -> NoReturn:
    for p in model.parameters():
        p.requires_grad = True
        
def cal_rouge():
    pass

class PrintInfo:
    
    def __init__(self):

        self.time_step = timeit.default_timer()
        self.accumulation = 0
    
    def SECTION(self, section: str, simple: bool = False):
        
        if not simple: print("\n" + "*" * 10)
        print("{} // before_step: {}ms // total: {}s".format(section, round(self._reset_time()), round(self.accumulation, 2)))
        if not simple: print("*" * 10 + "\n")

    def _reset_time(self):
        temp = self.time_step
        self.time_step = timeit.default_timer()
        diff = self.time_step - temp
        self.accumulation += diff
        return diff * 1000