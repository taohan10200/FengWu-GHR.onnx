#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
# Some of the code is borrowed from https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple, Union
from torch import Tensor
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple

class LinearLoraGroups(nn.Linear):
    """
    LoRA (Low-Rank Adaptation) implemented in a dense layer with support for multiple 
    groups and step-wise adaptation.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        total_aug_steps: int = 8,
        lora_alpha: int = 1,
        lora_bias: bool = False,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,  # Set to True if the layer uses (fan_in, fan_out) weights
        merge_weights: bool = True,
        **kwargs
    ):
        super().__init__(in_features, out_features, **kwargs)
        
        self.r = r
        self.scale = lora_alpha / r if r > 0 else 1
        self.fan_in_fan_out = fan_in_fan_out
        self.total_aug_steps = total_aug_steps
        self.lora_bias = lora_bias
        self.current_step = 0


        self.lora_A = nn.Parameter(torch.ones_like(self.weight)*self.weight.data*0.01)
        # self.lora_A = nn.Parameter(torch.ones_like(r, in_features)*0.01)
        # self.lora_B = nn.Parameter(torch.ones(out_features, r)*0.01)
        if lora_bias:
            self.lora_Bias = nn.Parameter(torch.ones(out_features)*0.01)

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)


    def get_merged_weights(self, step: int):
        """Merge weights for the given step."""
        # import pdb
        # pdb.set_trace()
        # if self.weight.dtype == torch.float16:
        #     weight = (
        #         self.lora_B.to(torch.float32) @ 
        #         self.lora_A.to(torch.float32) * self.scale *torch.randn_like(self.weight, device=self.weight.device) +
        #         self.weight.data.to(torch.float32)
        #     )
        #     return weight.to(torch.float16)
        # else:
        #     return (
        #         self.lora_B @ 
        #         self.lora_A * self.scale *  torch.randn_like(self.weight, device=self.weight.device) + 
        #         self.weight.data
        #     )
        if self.weight.dtype == torch.float16:
            weight = (
                
                self.lora_A.to(torch.float32) *torch.randn_like(self.weight, device=self.weight.device) +
                self.weight.data.to(torch.float32)
            )
            return weight.to(torch.float16)
        else:
            return (
               
                self.lora_A * torch.randn_like(self.weight, device=self.weight.device) + 
                self.weight.data
            )
    def forward(self, x: torch.Tensor, step: int = 0, **args):
        """Forward pass with optional LoRA adaptation."""
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if step > 0:
            return F.linear(x, self.get_merged_weights(step), bias=self.bias)
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

    def train(self, mode: bool = True):
        """Set the module in training mode."""
        if not isinstance(mode, bool):
            raise ValueError("Training mode must be a boolean.")
        self.training = mode
        super().train(False)  # Disable training for the base Linear layer
        return self

    def eval(self):
        """Set the module in evaluation mode."""
        return self.train(False)

    def extra_repr(self) -> str:
        """Extra representation for module printing."""
        return f"in_features={self.in_features}, lora_rank={self.r}, out_features={self.out_features}, scale={self.scale}"
    
    
class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = True
        
    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            after_A = F.embedding(
                x, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)

class LoraLayer(nn.Module):
    def __init__(self, 
                in_features,
                out_features,
                lora_alpha: int = 1, 
                r: int = 0,
                lora_dropout: float = 0.,
                merge_weights: bool = True,
                ):
        super(LoraLayer, self).__init__()
        self.r = r
        # 定义模块的参数
        self.in_features = in_features
        self.out_features = out_features
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        self.scale = lora_alpha / self.r
        
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)    
        
    def forward(self, x: torch.Tensor,H:int=0, W: int=0):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        result = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scale
        # print(self.lora_B )
        return result
    
    def extra_repr(self) -> str:
        return 'in_features={}, lora_rank={}, out_features={}, scale={}'.format(
            self.in_features, self.r, self.out_features, self.scale is not None
        )

class ConvLayer(nn.Module):
    def __init__(self, 
                in_features,
                out_features,
                lora_alpha: int = 1, 
                hidden_dim: int = 128,
                lora_dropout: float = 0.,
                merge_weights: bool = True,
                ):
        super(ConvLayer, self).__init__()
        self.hidden_dim = hidden_dim
        # 定义模块的参数
        # self.lora_A = nn.Parameter(weight.new_zeros((r, in_features)))
        # self.lora_B = nn.Parameter(weight.new_zeros((out_features, r)))
        self.lora_A = nn.Conv2d(in_features, hidden_dim,  3, 1,padding=1)
        self.lora_B = nn.Conv2d(hidden_dim,  out_features, 3, 1, padding=1)
        self.scaling = lora_alpha / self.hidden_dim
        self.activate =  nn.GELU()
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)    
        
    def forward(self, x: torch.Tensor, H:int=32, W: int=64):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        B, N, C = x.shape
        # import pdb
        # pdb.set_trace()
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2) #B,C,H, W
        x = self.lora_A(x) #self.scaling
        # x = self.activate(x)
        x = self.lora_B(x)
        B,C,H, W = x.shape
        # result = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        # print(self.lora_B.weight)
        x = x.permute(0, 2, 3, 1).reshape(B, N, C)
        return x
class Masked_Lora_Linear(nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        total_aug_steps: int = 8,
        lora_alpha: int = 1, 
        lora_bias: bool = False,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        
        self.r = r
        self.scale = lora_alpha / r
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        self.total_aug_steps = total_aug_steps
        self.lora_bias = lora_bias
        self.mask = nn.Parameter(torch.eye(self.total_aug_steps), requires_grad=False)
        if total_aug_steps > 1:
            self.lora_A = nn.Parameter(torch.empty(self.r*total_aug_steps, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, self.r*total_aug_steps))
            
            if self.lora_bias:
                self.lora_Bias_groups.append(nn.Parameter(torch.empty(out_features)))
                
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        if hasattr(self, 'lora_B'):
            nn.init.zeros_(self.lora_B) 
        if hasattr(self, 'lora_Bias'):
            nn.init.zeros_(self.lora_Bias) 
                
    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, False)
        
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        # for module in self.lora_A_groups:
        #     module.train(mode)
        # for module in self.lora_B_groups:
        #     module.train(mode)
        # for module in self.lora_Bias_groups:
        #     module.train(mode)
        return self
        # if mode:
       
    def eval(self):
        r"""Sets the module in evaluation mode.
        Returns:
            Module: self
        """
        return self.train(False)
        
    def forward(self, 
                x: torch.Tensor, 
                step: torch.Tensor.int=0):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        # mask = F.one_hot(torch.tensor(step), num_classes=self.total_aug_steps).to(x.device)
        mask = self.mask[step]
        mask = mask.repeat_interleave(self.r, dim=0)[:,None]
        
        if self.weight.dtype == torch.float16:
            weight = (self.lora_B.to(torch.float32)*mask.transpose(0,1)) @ \
                    (self.lora_A.to(torch.float32)*mask) * self.scale \
                    + self.weight.to(torch.float32)
            weight = weight.to(torch.float16)
        else:
            weight = (self.lora_B*mask.transpose(0,1)) @ (self.lora_A*mask) * self.scale + self.weight
        if self.lora_bias:
            return F.linear(x, weight, bias=self.bias+self.lora_Bias_groups[step-1] if self.bias is not None else self.lora_Bias_groups[step-1])
        else:
            return F.linear(x, weight, bias=self.bias)
    
        
    def extra_repr(self) -> str:
        return 'in_features={}, lora_rank={}, out_features={}, scale={}'.format(
            self.in_features, self.r, self.out_features, self.scale is not None
        )




class Linear_Head_Groups(nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        total_aug_steps: int = 8,
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        
        self.r = r
        self.scale = lora_alpha / r
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        self.total_aug_steps = total_aug_steps

        if total_aug_steps > 0:
            self.lora_A_groups = nn.ParameterList()
            self.lora_B_groups = nn.ParameterList()
            self.lora_Bias_groups = nn.ParameterList()
            for i in range(total_aug_steps-1):
                    # if r==16:
                    extra_fea = out_features//r
                    # self.lora_A_groups.append(nn.Parameter(torch.empty(tmp_r, in_features)))
                    self.lora_B_groups.append(nn.Parameter(torch.empty(extra_fea, in_features)))
                    self.lora_Bias_groups.append(nn.Parameter(torch.empty(extra_fea)))
                
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
        self.current_step = 0
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A_groups'):
            # initialize A the same way as the default for nn.Linear and B to zero
            for lora_A in self.lora_A_groups:
                nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
        if hasattr(self, 'lora_B_groups'):
            for lora_B in self.lora_B_groups:
                nn.init.zeros_(lora_B) 
        if hasattr(self, 'lora_Bias_groups'):
            for lora_Bias in self.lora_Bias_groups:
                nn.init.zeros_(lora_Bias) 
                
    def defrozen_lora_groups(self, step):
        for idx, param in enumerate(self.lora_A_groups):
            if idx == step:
                param.requires_grad = True
            else:
                param.requires_grad = False
       
        for idx, param in enumerate(self.lora_B_groups):
            if idx == step:
                param.requires_grad = True
            else:
                param.requires_grad = False
    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, False)
        
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.lora_groups:
            module.train(mode)
        return self
        # if mode:
        #     if self.merge_weights and self.merged:
        #         # Make sure that the weights are not merged
        #         if self.r > 0:
        #             self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
        #         self.merged = False
        # else:
        #     if self.merge_weights and not self.merged:
        #         # Merge the weights and mark it
        #         if self.r > 0:
        #             self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
        #         self.merged = True       
    def eval(self):
        r"""Sets the module in evaluation mode.
        Returns:
            Module: self
        """
        return self.train(False)
    def merged_weights(self, step):

        # if self.weight.dtype == torch.float16:
        #     weight = self.lora_B_groups[step].to(torch.float32) @ \
        #              self.lora_A_groups[step].to(torch.float32)  * self.scale \
        #              + self.weight.data.to(torch.float32)
        #     return  weight.to(torch.float16)
        # else:
        
        st, et= step*self.r, (step+1)*self.r 
        self.weight[:,st:et] = self.weight[:,st:et]+self.lora_B_groups[step] 
            
        
    def forward(self, x: torch.Tensor, step: int=0, H:int=0, W: int=0):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        

        if step>0:
            out1= F.linear(x, T(self.weight), bias=self.bias)
            out2= F.linear(x, T(self.lora_B_groups[step-1]), bias=self.lora_Bias_groups[step-1])
            pos = (step-1)%self.r
            # import pdb
            # pdb.set_trace()
            # print(self.lora_B_groups[step-1])
            out1[:,:, pos::self.r] += out2[:,:,0::1]
            
            return out1
        return F.linear(x, T(self.weight), bias=self.bias)
        
    def extra_repr(self) -> str:
        return 'in_features={}, lora_rank={}, out_features={}, scale={}'.format(
            self.in_features, self.r, self.out_features, self.scale is not None
        )

class Conv2D_Lora_Groups(nn.Conv2d):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        r=0, 
        lora_alpha=1, 
        lora_dropout=0., 
        merge_weights=True, 
        total_aug_steps: int = 8,
        **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        self.r = r 
        # Actual trainable parameters
        self.total_aug_steps = total_aug_steps
        if r>0:
            self.scaling = lora_alpha/r
        if total_aug_steps > 0:
            self.lora_A_groups = nn.ParameterList()
            self.lora_B_groups = nn.ParameterList()
            for i in range(total_aug_steps-1):
                tmp_r = r #+i
                self.lora_A_groups.append(nn.Parameter(torch.empty((tmp_r * kernel_size[0], in_channels * kernel_size[0]))))
                # self.lora_A_groups.append(nn.Parameter(torch.empty((tmp_r * kernel_size, in_channels * kernel_size))))
                self.lora_B_groups.append(nn.Parameter(torch.empty((out_channels*kernel_size[1], tmp_r*kernel_size[0]))))
        
        self.current_step = 0
        self.reset_parameters()            
    def reset_parameters(self):
        # self.conv.reset_parameters()
        if hasattr(self, 'lora_A_groups'):
            # initialize A the same way as the default for nn.Linear and B to zero
            for lora_A in self.lora_A_groups:
                nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
            for lora_B in self.lora_B_groups:
                nn.init.zeros_(lora_B)

    def defrozen_lora_groups(self, step):
        for i in range(len(self.lora_groups)):            
            for name, param in self.lora_groups[i].named_parameters():
                if i == step:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                        
    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, False)
        
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        # for module in self.lora_groups:
        #     module.train(mode)
        return self
  
    def eval(self):
        r"""Sets the module in evaluation mode.
        Returns:
            Module: self
        """
        return self.train(False)
    def forward(self, x: torch.Tensor, step: int=0):

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if step>0:
            # import pdb
            # pdb.set_trace()
            weight  = self.weight.data + (self.lora_B_groups[step-1] @ self.lora_A_groups[step-1]).view(self.weight.shape) * self.scaling              
            return self._conv_forward(x, weight, self.bias)
           
        else:
            with torch.no_grad():
                return self._conv_forward(x, self.weight, self.bias)
            
class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    
    def reset_parameters(self):
        # nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        delta_w = F.conv1d(
            self.lora_A.unsqueeze(0), 
            self.lora_B.unsqueeze(-1), 
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True        

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling
            return result


# class Conv2d(ConvLoRA):
#     def __init__(self, *args, **kwargs):
#         super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)

# class Conv1d(ConvLoRA):
#     def __init__(self, *args, **kwargs):
#         super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)

# # Can Extend to other ones like this

# class Conv3d(ConvLoRA):
#     def __init__(self, *args, **kwargs):
#         super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)