from collections import defaultdict
from typing import List, Dict

import torch

def collate_fn(
    batched_samples: List[Dict[str, List[int]]],
    pad_token_idx: int,
    pad_keys: List[str] = ["input_ids", "labels"],
) -> Dict[str, torch.Tensor]:

    batched_samples = sorted(batched_samples, key=lambda x: len(x["input_ids"]), reverse=True)
    keys = batched_samples[0].keys()
    outputs = defaultdict(list)

    for key in keys:
        for sample in batched_samples:
            outputs[key].append(torch.tensor(sample[key]))
        
        # TODO: label 이름을 무엇으로 할 것인가..
        PAD = pad_token_idx if key in pad_keys else 0
        PAD = -1 if key == "answers" else PAD
        
        outputs[key] = torch.nn.utils.rnn.pad_sequence(outputs[key], padding_value=PAD, batch_first=True)
    
    return dict(outputs)