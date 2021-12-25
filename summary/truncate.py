from typing import Tuple, Union, Optional, Iterable

import torch
from torch.types import Number

def truncate(x: torch.Tensor, l: int, dim: int = -1) -> Tuple[torch.Tensor]:
    """Returns a truncated tensor with length given and a remaining tensor.
    If a provided `tensor` is shorter than the `length` given, it returns the original tensor and `None`.
    It copies the original tensor. 
    """
    if x.size(dim) <= l:
        return x, None
    else:
        a_tensor = x[:l]
        b_tensor = x[l:]
        return a_tensor, b_tensor


def max_eq_pos(x: torch.Tensor, other: Union[torch.Tensor, Number]) -> int:
    eq = torch.eq(x, other).nonzero()
    if len(eq) == 0:
        # no corresponding element in the given tensor
        raise ValueError("`x` does not contain the corresponding element.")
    else:
        # if there is an element(s), then it returns the last element
        return eq[-1]

def min_eq_pos(x: torch.Tensor, other: Union[torch.Tensor, Number]) -> int:
    eq = torch.eq(x, other).nonzero()
    if len(eq) == 0:
        # no corresponding element in the given tensor
        raise ValueError("`x` does not contain the corresponding element.")
    else:
        # if there is an element(s), then it returns the last element
        return eq[0]


def truncate_with_eq(
    x: torch.Tensor, 
    l: int, 
    sep: Union[torch.Tensor, Number] = 0, 
    dim: int = -1, 
    overflow: bool = False, 
    eos_value: Optional[int] = None
):
    """Returns a truncated tensor shorter-than-or-equal-to the length given and a remaining tensor.
    If a provided `tensor` is shorter than the `length` given, it returns the original tensor and `None`.
    """
    if x.size(dim) <= l:
        return x, None
    else:
        try:
            to = max_eq_pos(x[0:l], sep) + 1
            a_tensor = x[:to]
            b_tensor = x[to:]
        except:
            try:
                to = min_eq_pos(x, sep) + 1
                if not overflow:
                    a_tensor = x[:l]
                    if eos_value is not None:
                        a_tensor[-1] = eos_value
                else:
                    a_tensor = x[:to]
                # TODO: should return None when b_tensor.size(0) == 0
                b_tensor = x[to:]
            except:
                if overflow:
                    return x, None
                else:
                    a_tensor = x[:l]
                    if eos_value is not None:
                        a_tensor[-1] = eos_value
                    return a_tensor, None
        
        return a_tensor, b_tensor


def batch_truncate(x: torch.Tensor, l: int, dim: int = -1, padding_value: int = 0):
    _a_batch = []
    _b_batch = []
    for i in range(x.size(0)):
        _a, _b = truncate(x[i], l, dim)
        _a_batch.append(_a)
        _b_batch.append(_b)
    a_batch = torch.nn.utils.rnn.pad_sequence(_a_batch, batch_first=True, padding_value=padding_value)
    if len(_b_batch) > 0:
        b_batch = torch.nn.utils.rnn.pad_sequence(_b_batch, batch_first=True, padding_value=padding_value)
    else:
        b_batch = None
    return a_batch, b_batch


def batch_truncate_with_eq(
    x: torch.Tensor, 
    l: int, 
    sep: Union[torch.Tensor, Number] = 0, 
    dim: int = -1, 
    padding_value: int = 0, 
    overflow: bool = False,
    eos_value: Optional[int] = None,
    return_mapping: bool = True,
):
    _a_batch = []
    _b_batch = []
    mapping = []

    for i in range(x.size(0)):
        _a, _b = truncate_with_eq(x[i], l, sep, dim, overflow, eos_value)
        _a_batch.append(_a)
        if _b is not None:
            _b_batch.append(_b)
            mapping.append(i)
        elif not return_mapping:
            _b_batch.append(torch.tensor([], dtype=x.dtype))
    
    a_batch = torch.nn.utils.rnn.pad_sequence(_a_batch, batch_first=True, padding_value=padding_value)
    if len(_b_batch) > 0:
        b_batch = torch.nn.utils.rnn.pad_sequence(_b_batch, batch_first=True, padding_value=padding_value)
    else:
        b_batch = None

    if not return_mapping:
        return a_batch, b_batch
    else:
        return a_batch, b_batch, mapping


def batch_truncate_with_len(
    x: torch.Tensor,
    l: Iterable[int],
    dim: int = -1,
    padding_value: int = 0,
):
    assert len(x) == len(l), "`l` must contain the same number of elements as batch_size"

    _a_batch = []
    _b_batch = []

    for i in range(x.size(0)):
        _a_batch.append(x[i, :l[i]])
        _b_batch.append(x[i, l[i]:])

    a_batch = torch.nn.utils.rnn.pad_sequence(_a_batch, batch_first=True, padding_value=padding_value)
    if len(_b_batch) > 0:
        b_batch = torch.nn.utils.rnn.pad_sequence(_b_batch, batch_first=True, padding_value=padding_value)
    else:
        b_batch = None

    return a_batch, b_batch


def gather_lengths(x: torch.Tensor, padding_value: int = 0):
    l = []
    for i in range(x.size(0)):
        eq = torch.nonzero(x[i] == padding_value, as_tuple=False)
        if len(eq) == 0:
            l.append(int(x.size(1)))
        else:
            l.append(eq[0].item())
    return l


def concat_sentences(x: torch.Tensor, y: torch.Tensor, padding_value: int = 0):
    assert len(x) == len(y)

    merged = []

    for i in range(x.size(0)):
        
        _x = x[i]
        _x = _x[_x != padding_value]

        _y = y[i]
        _y = _y[_y != padding_value]
        # _y = _y[1:] # excluding bos token

        merged.append(torch.cat([_x, _y], dim=0))
    
    return torch.nn.utils.rnn.pad_sequence(merged, batch_first=True, padding_value=padding_value)