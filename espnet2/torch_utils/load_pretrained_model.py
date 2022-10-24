from typing import Any
from typing import Dict
from typing import Union

import logging
import torch
import torch.nn
import torch.optim
from torch.nn.parameter import Parameter

def filter_state_dict(
    dst_state: Dict[str, Union[float, torch.Tensor]],
    src_state: Dict[str, Union[float, torch.Tensor]],
):
    """Filter name, size mismatch instances between dicts.

    Args:
        dst_state: reference state dict for filtering
        src_state: target state dict for filtering

    """
    match_state = {}
    for key, value in src_state.items():
        if key in dst_state and (dst_state[key].size() == src_state[key].size()):
            match_state[key] = value
        else:
            if key not in dst_state:
                logging.warning(
                    f"Filter out {key} from pretrained dict"
                    + " because of name not found in target dict"
                )
            else:
                logging.warning(
                    f"Filter out {key} from pretrained dict"
                    + " because of size mismatch"
                    + f"({dst_state[key].size()}-{src_state[key].size()})"
                )
    return match_state


def load_pretrained_model(
    init_param: str,
    model: torch.nn.Module,
    ignore_init_mismatch: bool,
    map_location: str = "cpu",
):
    """Load a model state and set it to the model.

    Args:
        init_param: <file_path>:<src_key>:<dst_key>:<exclude_Keys>

    Examples:
        >>> load_pretrained_model("somewhere/model.pth", model)
        >>> load_pretrained_model("somewhere/model.pth:decoder:decoder", model)
        >>> load_pretrained_model("somewhere/model.pth:decoder:decoder:", model)
        >>> load_pretrained_model(
        ...     "somewhere/model.pth:decoder:decoder:decoder.embed", model
        ... )
        >>> load_pretrained_model("somewhere/decoder.pth::decoder", model)
    """
    sps = init_param.split(":", 4)
    if len(sps) == 4:
        path, src_key, dst_key, excludes = sps
    elif len(sps) == 3:
        path, src_key, dst_key = sps
        excludes = None
    elif len(sps) == 2:
        path, src_key = sps
        dst_key, excludes = None, None
    else:
        (path,) = sps
        src_key, dst_key, excludes = None, None, None
    if src_key == "":
        src_key = None
    if dst_key == "":
        dst_key = None

    if dst_key is None:
        obj = model
    else:

        def get_attr(obj: Any, key: str):
            """Get an nested attribute.

            >>> class A(torch.nn.Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.linear = torch.nn.Linear(10, 10)
            >>> a = A()
            >>> assert A.linear.weight is get_attr(A, 'linear.weight')

            """
            if key.strip() == "":
                return obj
            for k in key.split("."):
                obj = getattr(obj, k)
            return obj

        obj = get_attr(model, dst_key)

    src_state = torch.load(path, map_location=map_location)


    dst_state = obj.state_dict()

    checkpoint_temp1 ={k:v for k,v in src_state.items()}


    with open("params_change_embedding11") as f:
        for line in f:
            line=line.strip()
            checkpoint_temp1[line]=Parameter(src_state[line])
    with open("params_change_embedding22") as f:
        for line in f:
            line=line.strip()
            checkpoint_temp1[line]=Parameter(src_state[line])
    with open("params_change_embedding33") as f:
        for line in f:
            line=line.strip()
            checkpoint_temp1[line]=Parameter(src_state[line])

    checkpoint_temp1["norm11.weight"]=Parameter(src_state["norm11.weight"])
    checkpoint_temp1["norm11.bias"]=Parameter(src_state["norm11.bias"])
    checkpoint_temp1["norm22.weight"]=Parameter(src_state["norm22.weight"])
    checkpoint_temp1["norm22.bias"]=Parameter(src_state["norm22.bias"])
    checkpoint_temp1["norm33.weight"]=Parameter(src_state["norm33.weight"])
    checkpoint_temp1["norm33.bias"]=Parameter(src_state["norm33.bias"])
    checkpoint_temp1["phone_embedding2.weight"]=Parameter(src_state["phone_embedding1.weight"])
    
    dst_state.update(checkpoint_temp1)
    obj.load_state_dict(src_state)



def load_pretrained_model2(
    init_param: str,
    model: torch.nn.Module,
    first_train : bool,
    ignore_init_mismatch: bool,
    map_location: str = "cpu",
):
    """Load a model state and set it to the model.

    Args:
        init_param: <file_path>:<src_key>:<dst_key>:<exclude_Keys>

    Examples:
        >>> load_pretrained_model("somewhere/model.pth", model)
        >>> load_pretrained_model("somewhere/model.pth:decoder:decoder", model)
        >>> load_pretrained_model("somewhere/model.pth:decoder:decoder:", model)
        >>> load_pretrained_model(
        ...     "somewhere/model.pth:decoder:decoder:decoder.embed", model
        ... )
        >>> load_pretrained_model("somewhere/decoder.pth::decoder", model)
    """
    if first_train:
        sps = init_param.split(":", 4)
        if len(sps) == 4:
            path, src_key, dst_key, excludes = sps
        elif len(sps) == 3:
            path, src_key, dst_key = sps
            excludes = None
        elif len(sps) == 2:
            path, src_key = sps
            dst_key, excludes = None, None
        else:
            (path,) = sps
            src_key, dst_key, excludes = None, None, None
        if src_key == "":
            src_key = None
        if dst_key == "":
            dst_key = None

        if dst_key is None:
            obj = model
        else:

            def get_attr(obj: Any, key: str):
                """Get an nested attribute.

                >>> class A(torch.nn.Module):
                ...     def __init__(self):
                ...         super().__init__()
                ...         self.linear = torch.nn.Linear(10, 10)
                >>> a = A()
                >>> assert A.linear.weight is get_attr(A, 'linear.weight')

                """
                if key.strip() == "":
                    return obj
                for k in key.split("."):
                    obj = getattr(obj, k)
                return obj

            obj = get_attr(model, dst_key)

        #src_state = torch.load(path, map_location=map_location)


        dst_state = obj.state_dict()

        checkpoint_encode77 = torch.load("../letter_bid_fine_encoder/exp/asr_conformer_lr2e-3_warmup15k_amp_nondeterministic_right2_encoder/valid.acc.ave_10best.pth")
        checkpoint_encode88 = torch.load("../letter_bid_fine_decoder/exp/asr_conformer_lr2e-3_warmup15k_amp_nondeterministic_right_decoder/valid.acc.ave_4best.pth")

        checkpoint_temp1 ={}

        with open("c7") as f:
            for line in f:
                line=line.strip()
                if line.endswith("num_batches_tracked"):
                    checkpoint_temp1[line]=checkpoint_encode77[line]
                else:
                    checkpoint_temp1[line]=Parameter(checkpoint_encode77[line])

        with open("c8") as f:
            for line in f:
                line=line.strip()
                if line.endswith("num_batches_tracked"):
                    checkpoint_temp1[line]=checkpoint_encode88[line]
                else:
                    checkpoint_temp1[line]=Parameter(checkpoint_encode88[line])
        


        dst_state.update(checkpoint_temp1)
        obj.load_state_dict(dst_state)

#####################################

        # checkpoint="exp/asr_conformer_lr2e-3_warmup15k_amp_nondeterministic_right2_encoder/checkpoint.pth"
        # states = torch.load(
        #     checkpoint,
        #     map_location=f"cuda:{torch.cuda.current_device()}" if 1 > 0 else "cpu",
        # )
        #model.load_state_dict(states)

        # obj.load_state_dict(states["model"])









    else:
        sps = init_param.split(":", 4)
        if len(sps) == 4:
            path, src_key, dst_key, excludes = sps
        elif len(sps) == 3:
            path, src_key, dst_key = sps
            excludes = None
        elif len(sps) == 2:
            path, src_key = sps
            dst_key, excludes = None, None
        else:
            (path,) = sps
            src_key, dst_key, excludes = None, None, None
        if src_key == "":
            src_key = None
        if dst_key == "":
            dst_key = None

        if dst_key is None:
            obj = model
        else:

            def get_attr(obj: Any, key: str):
                """Get an nested attribute.

                >>> class A(torch.nn.Module):
                ...     def __init__(self):
                ...         super().__init__()
                ...         self.linear = torch.nn.Linear(10, 10)
                >>> a = A()
                >>> assert A.linear.weight is get_attr(A, 'linear.weight')

                """
                if key.strip() == "":
                    return obj
                for k in key.split("."):
                    obj = getattr(obj, k)
                return obj

            obj = get_attr(model, dst_key)

        src_state = torch.load(path, map_location=map_location)
        if excludes is not None:
            for e in excludes.split(","):
                src_state = {k: v for k, v in src_state.items() if not k.startswith(e)}

        if src_key is not None:
            src_state = {
                k[len(src_key) + 1 :]: v
                for k, v in src_state.items()
                if k.startswith(src_key)
            }

        dst_state = obj.state_dict()

        dst_state.update(src_state)
        obj.load_state_dict(dst_state)