import inspect
import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import os
from torch import nn

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.utils import deprecate, logging
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_torch_version, maybe_allow_in_graph

from diffusers.models.attention_processor import Attention

class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.timestep = 0
        self.output_dir = "kv_cache_output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.kv_cache = defaultdict(lambda: defaultdict(list))

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # save kv cache in a dict for plot
        self.print_kv_cache(key, value, attn)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        self.timestep += 1
        return hidden_states

    def print_kv_cache(self, key, value, attn):
        layer_name = attn.__class__.__name__
        self.kv_cache[layer_name]['key'].append(key.detach().cpu().numpy())
        self.kv_cache[layer_name]['value'].append(value.detach().cpu().numpy())
        

    def plot_kv_diff(self, layer_num: int):
        print(f"Plotting KV diff and stats for layer {layer_num}")
        print(f"KV cache contains {len(self.kv_cache)} layers")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for layer_name, kv_data in self.kv_cache.items():
            print(f"Processing layer: {layer_name}")
            print(f"Number of timesteps for key: {len(kv_data['key'])}")
            print(f"Number of timesteps for value: {len(kv_data['value'])}")

            if len(kv_data['key']) < 2:
                print(f"Not enough timesteps for layer {layer_name}. Skipping.")
                continue

            key_diff_means, key_diff_vars = [], []
            value_diff_means, value_diff_vars = [], []
            key_means, key_vars = [], []
            value_means, value_vars = [], []
            
            for i in range(len(kv_data['key'])):
                key = kv_data['key'][i]
                value = kv_data['value'][i]
                
                # 计算均值和方差
                key_gpu = torch.tensor(key, device=device)
                value_gpu = torch.tensor(value, device=device)
                
                key_means.append(torch.mean(torch.abs(key_gpu)).item())
                key_vars.append(torch.var(key_gpu).item())
                value_means.append(torch.mean(torch.abs(value_gpu)).item())
                value_vars.append(torch.var(value_gpu).item())
                
                if i > 0:
                    prev_key = kv_data['key'][i-1]
                    prev_value = kv_data['value'][i-1]
                    
                    # 计算差异
                    key_diff_gpu = key_gpu - torch.tensor(prev_key, device=device)
                    value_diff_gpu = value_gpu - torch.tensor(prev_value, device=device)
                    
                    key_diff_means.append(torch.mean(torch.abs(key_diff_gpu)).item())
                    key_diff_vars.append(torch.var(key_diff_gpu).item())
                    value_diff_means.append(torch.mean(torch.abs(value_diff_gpu)).item())
                    value_diff_vars.append(torch.var(value_diff_gpu).item())

            timesteps = range(len(key_means))
            diff_timesteps = range(1, len(key_diff_means) + 1)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
            
            # Plot differences with error bars
            ax1.errorbar(diff_timesteps, key_diff_means, yerr=np.sqrt(key_diff_vars), 
                         label='Key Diff', color='blue', capsize=5)
            ax1.errorbar(diff_timesteps, value_diff_means, yerr=np.sqrt(value_diff_vars), 
                         label='Value Diff', color='red', capsize=5)
            ax1.set_xlabel('Timestep')
            ax1.set_ylabel('Mean of Absolute Differences')
            ax1.legend()
            ax1.set_title(f'KV Cache Differences for {layer_name}')
            
            # Plot actual key and value statistics with error bars
            ax2.errorbar(timesteps, key_means, yerr=np.sqrt(key_vars), 
                         label='Key', color='blue', capsize=5)
            ax2.errorbar(timesteps, value_means, yerr=np.sqrt(value_vars), 
                         label='Value', color='red', capsize=5)
            ax2.set_xlabel('Timestep')
            ax2.set_ylabel('Mean of Absolute Values')
            ax2.legend()
            ax2.set_title(f'KV Cache Statistics for {layer_name}')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{layer_name}_{layer_num}_kv_stats.png'))
            plt.close()

        print("Finished plotting KV diff and stats")
