import numpy as np
import torch
import torchvision
import torch.nn as nn
import random
from collections import OrderedDict
import pandas as pd

def summary(model, x, *args, **kwargs):
    """Summarize the given input model.
    Summarized information are 1) output shape, 2) kernel shape,
    3) number of the parameters and 4) operations (Mult-Adds)
    Args:
        model (Module): Model to summarize
        x (Tensor): Input tensor of the model with [N, C, H, W] shape
                    dtype and device have to match to the model
        args, kwargs: Other argument used in `model.forward` function
    """
    def register_hook(module):
        def hook(module, inputs, outputs):
            cls_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            # Lookup name in a dict that includes parents
            for name, item in module_names.items():
                if item == module:
                    key = "{}_{}".format(module_idx, name)

            info = OrderedDict()
            info["id"] = id(module)
            if isinstance(outputs, (list, tuple)):
                try:
                    info["out"] = list(outputs[0].size())
                except AttributeError:
                    # pack_padded_seq and pad_packed_seq store feature into data attribute
                    info["out"] = list(outputs[0].data.size())
            else:
                info["out"] = list(outputs.size())

            info["ksize"] = "-"
            info["inner"] = OrderedDict()
            info["params_nt"], info["params"], info["macs"] = 0, 0, 0
            for name, param in module.named_parameters():
                info["params"] += param.nelement() * param.requires_grad
                info["params_nt"] += param.nelement() * (not param.requires_grad)

                if name == "weight":
                    ksize = list(param.size())
                    # to make [in_shape, out_shape, ksize, ksize]
                    if len(ksize) > 1:
                        ksize[0], ksize[1] = ksize[1], ksize[0]
                    info["ksize"] = ksize

                    # ignore N, C when calculate Mult-Adds in ConvNd
                    if "Conv" in cls_name:
                        info["macs"] += int(param.nelement() * np.prod(info["out"][2:]))
                    else:
                        info["macs"] += param.nelement()

                # RNN modules have inner weights such as weight_ih_l0
                elif "weight" in name:
                    info["inner"][name] = list(param.size())
                    info["macs"] += param.nelement()

            # if the current module is already-used, mark as "(recursive)"
            # check if this module has params
            if list(module.named_parameters()):
                for v in summary.values():
                    if info["id"] == v["id"]:
                        info["params"] = "(recursive)"

            if info["params"] == 0:
                info["params"], info["macs"] = "-", "-"

            summary[key] = info

        # ignore Sequential and ModuleList
        if not module._modules:
            hooks.append(module.register_forward_hook(hook))

    module_names = get_names_dict(model)

    hooks = []
    summary = OrderedDict()

    model.apply(register_hook)
    try:
        with torch.no_grad():
            model(x) if not (kwargs or args) else model(x, *args, **kwargs)
    finally:
        for hook in hooks:
            hook.remove()

    # Use pandas to align the columns
    df = pd.DataFrame(summary).T

    df["Mult-Adds"] = pd.to_numeric(df["macs"], errors="coerce")
    df["Params"] = pd.to_numeric(df["params"], errors="coerce")
    df["Non-trainable params"] = pd.to_numeric(df["params_nt"], errors="coerce")
    df = df.rename(columns=dict(
        ksize="Kernel Shape",
        out="Output Shape",
    ))
    # df_sum = df.sum()
    df.index.name = "Layer"

    df = df[["Kernel Shape", "Output Shape", "Params", "Mult-Adds"]]

    return df


def get_names_dict(model):
    """Recursive walk to get names including path."""
    names = {}

    def _get_names(module, parent_name=""):
        for key, m in module.named_children():
            cls_name = str(m.__class__).split(".")[-1].split("'")[0]
            num_named_children = len(list(m.named_children()))
            if num_named_children > 0:
                name = parent_name + "." + key if parent_name else key
            else:
                name = parent_name + "." + cls_name + "_"+ key if parent_name else key
            names[name] = m

            if isinstance(m, torch.nn.Module):
                _get_names(m, parent_name=name)

    _get_names(model)
    return names

def compute_MACs_params(model, data):
    df = summary(model.eval(), data)
    df_sum = df.sum(numeric_only=True)
    MACs = df_sum['Mult-Adds'] / 1e9
    params = df_sum['Params'] / 1e6
    return MACs, params

def compute_block_MACs(df):
    mac = df['Mult-Adds']
    stat = dict()
    for key in mac.keys():
        value = mac[key]
        items = key.split('.')
        stat_key = ''
        if 'features' in key:
            stat_key = f'features.{items[1]}'
        elif 'layer' in key:
            stat_key = f'layer{items[0][-1]}.{items[1]}'
        if stat_key and value > 0:
            n = stat.get(stat_key, 0)
            n += value
            stat[stat_key] = n 
    for key in stat:
        value = stat[key] / 1e6
        print(f'{key} -> {value} M')


if __name__ == '__main__':
    import os
    from models import resnet, resnet_pruned
    from models import mobilenetv2
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    rm_blocks = ['layer1.1', 'layer2.1', 'layer3.1']
    #model = resnet.resnet34_rm_blocks(rm_blocks, pretrained=True, num_classes=1000).cuda()
    # model = resnet.resnet50(num_classes=1000).cuda()
    r34 = resnet.resnet34()
    data = torch.randn(1, 3, 224, 224)
    mac, params = compute_MACs_params(r34, data)
    mn = mobilenetv2.mobilenet_v2()
    df = summary(mn.eval(), data)
    print(f'MACs: {mac}, Params: {params}')
    import IPython
    IPython.embed()

