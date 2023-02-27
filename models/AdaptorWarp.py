import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
try:
    from torchvision.models.utils import load_state_dict_from_url
except:
    from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional

class AdaptorWarp(nn.Module):
    def __init__(self, model) -> None:
        super(AdaptorWarp, self).__init__()
        self.model = model

        self.named_modules_dict = dict(model.named_modules())
        self.module2preconvs = dict()
        self.name2preconvs = dict()
        self.prehandles = dict()
        self.module2afterconvs = dict()
        self.name2afterconvs = dict()
        self.afterhandles = dict()


    def add_preconv_for_conv(self, name, convmodule=None, preconv=None):
        assert name in self.named_modules_dict
        module = self.named_modules_dict[name]
        if convmodule is None:
            convmodule = module
        else:
            assert module == convmodule
        assert isinstance(convmodule, torch.nn.Conv2d)
        if name in self.name2preconvs:
            print("Not insert because {} have defined pre-conv".format(name))
            return None
        if preconv is None:
            v = convmodule.in_channels
            preconv = nn.Conv2d(v, v, kernel_size=1, stride=1, padding=0, bias=False).cuda()
            preconv.weight.data.copy_(torch.eye(v).view(v, v, 1, 1))
        self.module2preconvs[convmodule] = preconv
        self.name2preconvs[name] = preconv
        def hook(module, input):
            preconv = self.module2preconvs[module]
            return preconv(input[0])
        self.prehandles[convmodule] = convmodule.register_forward_pre_hook(hook)
        print("=> add pre-conv on {}".format(name))
        return preconv

    def reset_preconv_for_conv(self, name, convmodule=None):
        assert name in self.named_modules_dict
        module = self.named_modules_dict[name]
        if convmodule is None:
            convmodule = module
        else:
            assert module == convmodule
        assert isinstance(convmodule, torch.nn.Conv2d)
        if name not in self.name2preconvs:
            print("WARN: Can not reset, {} have not defined pre-conv".format(name))
            return
        preconv = self.name2preconvs[name]
        v = convmodule.in_channels
        preconv.weight.data.copy_(torch.eye(v).view(v, v, 1, 1))
        return preconv


    def remove_preconv_for_conv(self, name, convmodule=None, absorb=False):
        assert name in self.named_modules_dict
        module = self.named_modules_dict[name]
        if convmodule is None:
            convmodule = module
        else:
            assert module == convmodule
        assert isinstance(convmodule, torch.nn.Conv2d)
        if name not in self.name2preconvs.keys():
            if absorb:
                print("WARN: cannot absorb because not find pre-conv on {}".format(name))
            return
        if absorb:
            print("=> absorb pre-conv on {}".format(name))
            pw = self.name2preconvs[name].weight.data
            weight = convmodule.weight.data
            w = weight.permute(2, 3, 0, 1)
            # w: 3 x 3 x out x in 
            # use double type
            new_weight = torch.matmul(w.double(), pw.squeeze().double())
            new_weight = new_weight.float().permute(2, 3, 0, 1)
            convmodule.weight.data.copy_(new_weight)
        # remove the hook
        self.prehandles[convmodule].remove()
        self.name2preconvs.pop(name)
        print("=> remove pre-conv on {}".format(name))


    def add_all_preconv(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.add_preconv_for_conv(name, module)

    def reset_all_preconv(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.reset_preconv_for_conv(name, module)

    def absorb_all_preconv(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.remove_preconv_for_conv(name, module, absorb=True)

    def remove_all_preconv(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.remove_preconv_for_conv(name, module)
                

    def add_afterconv_for_conv(self, name, convmodule=None, afterconv=None):
        assert name in self.named_modules_dict, "{} not in {}".format(
            name, self.named_modules_dict.keys())
        module = self.named_modules_dict[name]
        if convmodule is None:
            convmodule = module
        else:
            assert module == convmodule
        assert isinstance(convmodule, torch.nn.Conv2d)
        if name in self.name2afterconvs:
            print("Not insert because {} have defined after-conv".format(name))
            return None
        if afterconv is None:
            v = convmodule.out_channels
            afterconv = nn.Conv2d(v, v, kernel_size=1, stride=1, padding=0, bias=False).cuda()
            afterconv.weight.data.copy_(torch.eye(v).view(v, v, 1, 1))
        self.module2afterconvs[convmodule] = afterconv
        self.name2afterconvs[name] = afterconv
        def hook(module, input, output):
            afterconv = self.module2afterconvs[module]
            return afterconv(output)
        self.afterhandles[convmodule] = convmodule.register_forward_hook(hook)
        print("=> add after-conv on {}".format(name))
        return afterconv

    def reset_afterconv_for_conv(self, name, convmodule=None):
        assert name in self.named_modules_dict
        module = self.named_modules_dict[name]
        if convmodule is None:
            convmodule = module
        else:
            assert module == convmodule
        assert isinstance(convmodule, torch.nn.Conv2d)
        if name not in self.name2afterconvs:
            print("WARN: Can not reset, {} have not defined after-conv".format(name))
            return
        afterconv = self.name2afterconvs[name]
        v = convmodule.out_channels
        afterconv.weight.data.copy_(torch.eye(v).view(v, v, 1, 1))
        return afterconv

    def remove_afterconv_for_conv(self, name, convmodule=None, absorb=False):
        assert name in self.named_modules_dict, "{} not in {}".format(name, self.named_modules_dict.keys())
        module = self.named_modules_dict[name]
        if convmodule is None:
            convmodule = module
        else:
            assert module == convmodule
        assert isinstance(convmodule, torch.nn.Conv2d)
        if name not in self.name2afterconvs.keys():
            if absorb:
                print("WARN: cannot absorb because not find after-conv on {}".format(name))
            return
        if absorb:
            print("=> absorb after-conv on {}".format(name))
            pw = self.name2afterconvs[name].weight.data
            weight = convmodule.weight.data
            w = weight.permute(2, 3, 0, 1)
            # w: 3 x 3 x out x in 
            # use double type
            new_weight = torch.matmul(pw.squeeze().double(), w.double())
            new_weight = new_weight.float().permute(2, 3, 0, 1)
            convmodule.weight.data.copy_(new_weight)
            if convmodule.bias is not None:
                new_bias = torch.matmul(pw.double().squeeze(), convmodule.bias.data.double().unsqueeze(1))
                convmodule.bias.data.copy_(new_bias.float().flatten())
        # remove the hook
        self.afterhandles[convmodule].remove()
        self.name2afterconvs.pop(name)
        print("=> remove after-conv on {}".format(name))

    def add_all_afterconv(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.add_afterconv_for_conv(name, module)

    def reset_all_afterconv(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.reset_afterconv_for_conv(name, module)

    def absorb_all_afterconv(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.remove_afterconv_for_conv(name, module, absorb=True)

    def remove_all_afterconv(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.remove_afterconv_for_conv(name, module)



    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        return self.model(x)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def freeze_classifier(self):
        self.model.freeze_classifier()


    def cuda(self):
        for key in self.name2preconvs.keys():
            self.name2preconvs[key].cuda()
        for key in self.name2afterconvs.keys():
            self.name2afterconvs[key].cuda()
        return super(AdaptorWarp, self).cuda()