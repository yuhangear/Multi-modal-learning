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

    # src_state["cif.conv.weight"]=dst_state["cif.conv.weight"]
    # src_state["cif.conv.bias"]=dst_state["cif.conv.bias"]
    # src_state["cif.weight_proj.weight"]=dst_state["cif.weight_proj.weight"]
    # src_state["cif.weight_proj.bias"]=dst_state["cif.weight_proj.bias"]

    # src_state["batch_norm.weight"]=dst_state["batch_norm.weight"]
    # src_state["batch_norm.bias"]=dst_state["batch_norm.bias"]
    # src_state["batch_norm.running_mean"]=dst_state["batch_norm.running_mean"]
    # src_state["batch_norm.running_var"]=dst_state["batch_norm.running_var"]

    dst_state.update(src_state)
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


        src_state = torch.load(path, map_location=map_location)


        dst_state = obj.state_dict()

        #checkpoint_encode88 = torch.load("../lm_pre_letter/19params_with_transformer8.pth")


        checkpoint_temp1 ={k:v for k,v in src_state.items() if k in dst_state.keys()}

        dst_state.update(checkpoint_temp1)
        obj.load_state_dict(dst_state)


    #     src_state = torch.load(path, map_location=map_location)


    #     dst_state = obj.state_dict()

    #     checkpoint_encode88 = torch.load("../lm_pre/params_with_transformer2.pth")
  

    #     checkpoint_temp1 ={k:v for k,v in src_state.items() if k in dst_state.keys()}
    #     checkpoint_temp1['phone_embedding1.weight']=Parameter(checkpoint_encode88["embeddings.weight"])
    #     checkpoint_temp1['phone_embedding2.weight']=Parameter(checkpoint_encode88["embeddings.weight"])
    #     checkpoint_temp1['phone_embedding3.weight']=Parameter(checkpoint_encode88["embeddings.weight"])
    #     checkpoint_temp1['phone_embedding4.weight']=Parameter(checkpoint_encode88["embeddings.weight"])
    #     checkpoint_temp1['phone_embedding5.weight']=Parameter(checkpoint_encode88["embeddings.weight"])
    #     checkpoint_temp1['phone_embedding6.weight']=Parameter(checkpoint_encode88["embeddings.weight"])
    #     checkpoint_temp1['phone_embedding7.weight']=Parameter(checkpoint_encode88["embeddings.weight"])
    #     checkpoint_temp1['phone_embedding8.weight']=Parameter(checkpoint_encode88["embeddings.weight"])

    #     checkpoint_temp1['line_emb1.weight']=Parameter(src_state["ctc_phone.ctc_lo.weight"])
    #     checkpoint_temp1['line_emb1.bias']=Parameter(src_state["ctc_phone.ctc_lo.bias"])
    #     checkpoint_temp1['line_emb2.weight']=Parameter(src_state["ctc_phone.ctc_lo.weight"])
    #     checkpoint_temp1['line_emb2.bias']=Parameter(src_state["ctc_phone.ctc_lo.bias"])
    #     checkpoint_temp1['line_emb3.weight']=Parameter(src_state["ctc_phone.ctc_lo.weight"])
    #     checkpoint_temp1['line_emb3.bias']=Parameter(src_state["ctc_phone.ctc_lo.bias"])

    #     checkpoint_temp1['line_emb4.weight']=Parameter(src_state["ctc_phone.ctc_lo.weight"])
    #     checkpoint_temp1['line_emb4.bias']=Parameter(src_state["ctc_phone.ctc_lo.bias"])

    #     checkpoint_temp1['line_emb5.weight']=Parameter(src_state["ctc_phone.ctc_lo.weight"])
    #     checkpoint_temp1['line_emb5.bias']=Parameter(src_state["ctc_phone.ctc_lo.bias"])

    #     checkpoint_temp1['line_emb6.weight']=Parameter(src_state["ctc_phone.ctc_lo.weight"])
    #     checkpoint_temp1['line_emb6.bias']=Parameter(src_state["ctc_phone.ctc_lo.bias"])

    #     checkpoint_temp1['line_emb7.weight']=Parameter(src_state["ctc_phone.ctc_lo.weight"])
    #     checkpoint_temp1['line_emb7.bias']=Parameter(src_state["ctc_phone.ctc_lo.bias"])

    #     checkpoint_temp1['line_emb8.weight']=Parameter(src_state["ctc_phone.ctc_lo.weight"])
    #     checkpoint_temp1['line_emb8.bias']=Parameter(src_state["ctc_phone.ctc_lo.bias"])

    #     checkpoint_temp1['ctc_phone_other.ctc_lo.weight']=Parameter(src_state["ctc_phone.ctc_lo.weight"])
    #     checkpoint_temp1['ctc_phone_other.ctc_lo.bias']=Parameter(src_state["ctc_phone.ctc_lo.bias"])

    #     checkpoint_temp1['line_phone.weight']=Parameter(checkpoint_encode88["line_phone.weight"])
    #     checkpoint_temp1['line_phone.bias']=Parameter(checkpoint_encode88["line_phone.bias"])

    #     with open("params_change_embedding") as f:
    #         for line in f:
    #             line=line.strip()
    #             checkpoint_temp1[line]=Parameter(checkpoint_encode88[line])
    #     checkpoint_temp1["norm1.weight"]=Parameter(checkpoint_encode88["after_norm.weight"])
    #     checkpoint_temp1["norm1.bias"]=Parameter(checkpoint_encode88["after_norm.bias"])

    #     checkpoint_temp1["after_norm.weight"]=Parameter(src_state["encoder.after_norm.weight"])
    #     checkpoint_temp1["after_norm.bias"]=Parameter(src_state["encoder.after_norm.bias"])



    #     dst_state.update(checkpoint_temp1)
    #     obj.load_state_dict(dst_state)
    # else:
    #     sps = init_param.split(":", 4)
    #     if len(sps) == 4:
    #         path, src_key, dst_key, excludes = sps
    #     elif len(sps) == 3:
    #         path, src_key, dst_key = sps
    #         excludes = None
    #     elif len(sps) == 2:
    #         path, src_key = sps
    #         dst_key, excludes = None, None
    #     else:
    #         (path,) = sps
    #         src_key, dst_key, excludes = None, None, None
    #     if src_key == "":
    #         src_key = None
    #     if dst_key == "":
    #         dst_key = None

    #     if dst_key is None:
    #         obj = model
    #     else:

    #         def get_attr(obj: Any, key: str):
    #             """Get an nested attribute.

    #             >>> class A(torch.nn.Module):
    #             ...     def __init__(self):
    #             ...         super().__init__()
    #             ...         self.linear = torch.nn.Linear(10, 10)
    #             >>> a = A()
    #             >>> assert A.linear.weight is get_attr(A, 'linear.weight')

    #             """
    #             if key.strip() == "":
    #                 return obj
    #             for k in key.split("."):
    #                 obj = getattr(obj, k)
    #             return obj

    #         obj = get_attr(model, dst_key)

    #     src_state = torch.load(path, map_location=map_location)
    #     if excludes is not None:
    #         for e in excludes.split(","):
    #             src_state = {k: v for k, v in src_state.items() if not k.startswith(e)}

    #     if src_key is not None:
    #         src_state = {
    #             k[len(src_key) + 1 :]: v
    #             for k, v in src_state.items()
    #             if k.startswith(src_key)
    #         }

    #     dst_state = obj.state_dict()

    #     dst_state.update(src_state)
    #     obj.load_state_dict(dst_state)