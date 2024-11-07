from typing import Optional
from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F

import numpy as np
from collections import defaultdict

from diffusers.utils import deprecate
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb

class AttnProcessorExperimentBase(metaclass=ABCMeta):
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.timestep = 0
        self.kv_cache = defaultdict(lambda: defaultdict(list))
        self.activation_cache = defaultdict(lambda: defaultdict(list))
        self.previous_step_cache = {
            'k': None, 'v': None, 'a': None, 'ek': None, 'ev': None, 'ea': None
        }
        self.info = {
            'means': {
                'k': [], 'v': [], 'a': [], 'ek': [], 'ev': [], 'ea': []
            },
            'vars': {
                'k': [], 'v': [], 'a': [], 'ek': [], 'ev': [], 'ea': []
            }
        }

    @abstractmethod
    def __call__():
        pass

    def update_cache(self, key, value):
        if self.previous_step_cache[key] is None:
            self.previous_step_cache[key] = value
        else:
            diff = torch.abs(value - self.previous_step_cache[key])
            means = torch.mean(diff).item()
            vars = torch.var(diff).item()
            self.info['means'][key].append(means)
            self.info['vars'][key].append(vars)
            self.previous_step_cache[key] = value

    def reset_cache(self):
        self.previous_step_cache = {
            'k': None, 'v': None, 'a': None, 'ek': None, 'ev': None, 'ea': None
        }
        self.info = {
            'means': {
                'k': [], 'v': [], 'a': [], 'ek': [], 'ev': [], 'ea': []
            },
            'vars': {
                'k': [], 'v': [], 'a': [], 'ek': [], 'ev': [], 'ea': []
            }
        }

    def save_activation_cache(self, activation, attn):
        layer_name = attn.__class__.__name__
        self.activation_cache[layer_name]['activation'].append(activation.detach().cpu().numpy())

    def plot_kv_diff(self, layer_num: int, ax1, num_columns: int, relative: bool = False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for layer_name, kv_data in self.kv_cache.items():
            if len(kv_data['key']) < 2:
                print(f"Not enough timesteps for layer {layer_name}. Skipping.")
                continue

            key_diff_means, key_diff_vars = [], []
            value_diff_means, value_diff_vars = [], []
            pre_key_gpu, cur_key_gpu = None, None
            pre_value_gpu, cur_value_gpu = None, None
            
            for i in range(len(kv_data['key'])):
                cur_key = kv_data['key'][i]
                cur_value = kv_data['value'][i]
                cur_key_gpu = torch.abs(torch.tensor(cur_key, device=device))
                cur_value_gpu = torch.abs(torch.tensor(cur_value, device=device))
                cur_key_gpu[cur_key_gpu == 0] = 1e-4
                cur_value_gpu[cur_value_gpu == 0] = 1e-4

                if i == 0:
                    if relative:
                        key_diff_gpu = torch.ones_like(cur_key_gpu, device=device)
                        value_diff_gpu = torch.ones_like(cur_value_gpu, device=device)
                    else:
                        key_diff_gpu = cur_key_gpu
                        value_diff_gpu = cur_value_gpu
                    
                else:
                    if relative:
                        key_diff_gpu = torch.abs(cur_key_gpu - pre_key_gpu) / (cur_key_gpu + pre_key_gpu)
                        value_diff_gpu = torch.abs(cur_value_gpu - pre_value_gpu) / (cur_value_gpu + pre_value_gpu)
                        
                    else:
                        key_diff_gpu = cur_key_gpu - pre_key_gpu
                        value_diff_gpu = cur_value_gpu - pre_value_gpu

                pre_key_gpu, cur_key_gpu = cur_key_gpu, None
                pre_value_gpu, cur_value_gpu = cur_value_gpu, None

                key_diff_means.append(torch.mean(torch.abs(key_diff_gpu)).item())
                key_diff_vars.append(torch.var(key_diff_gpu).item())
                value_diff_means.append(torch.mean(torch.abs(value_diff_gpu)).item())
                value_diff_vars.append(torch.var(value_diff_gpu).item())
                    

            timesteps = range(len(kv_data['key']))
            
            # Plot differences with error bars
            row, column = layer_num//num_columns, layer_num%num_columns
            ax1[row, column].errorbar(timesteps, key_diff_means, yerr=np.sqrt(key_diff_vars), 
                            label='Key Diff', color='blue', capsize=5)
            ax1[row, column].errorbar(timesteps, value_diff_means, yerr=np.sqrt(value_diff_vars), 
                            label='Value Diff', color='red', capsize=5)
            ax1[row, column].set_xticks(range(0, len(kv_data['key']), len(kv_data['key']) // 10))
            ax1[row, column].set_xlabel('Timestep')
            if relative:
                ax1[row, column].set_ylabel('Mean of Relative Differences')
            else:
                ax1[row, column].set_ylabel('Mean of Absolute Differences')
            ax1[row, column].legend()
            ax1[row, column].set_title(f'{layer_name} {layer_num} KV Diff')

        self.kv_cache.clear()

    def plot_activation_diff(self, layer_num: int, ax1, num_columns: int, relative: bool = False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for layer_name, activaton_data in self.activation_cache.items():
            if len(activaton_data['activation']) < 2:
                print(f"Not enough timesteps for layer {layer_name}. Skipping.")
                continue

            activation_diff_means, activation_diff_vars = [], []
            pre_activation_gpu, cur_activation_gpu = None, None
            
            for i in range(len(activaton_data['activation'])):
                cur_activation = activaton_data['activation'][i]
                cur_activation_gpu = torch.abs(torch.tensor(cur_activation, device=device))
                cur_activation_gpu[cur_activation_gpu == 0] = 1e-4
                
                if i == 0:
                    if relative:
                        activation_diff_gpu = torch.ones_like(cur_activation_gpu, device=device)
                    else:
                        activation_diff_gpu = cur_activation_gpu
                    
                else:
                    if relative:
                        activation_diff_gpu = torch.abs(cur_activation_gpu - pre_activation_gpu) / (cur_activation_gpu + pre_activation_gpu)
                        
                    else:
                        activation_diff_gpu = cur_activation_gpu - pre_activation_gpu

                pre_activation_gpu, cur_activation_gpu = cur_activation_gpu, None

                activation_diff_means.append(torch.mean(torch.abs(activation_diff_gpu)).item())
                activation_diff_vars.append(torch.var(activation_diff_gpu).item())
                    

            timesteps = range(len(activaton_data['activation']))
            
            # Plot differences with error bars
            row, column = layer_num//num_columns, layer_num%num_columns
            ax1[row, column].errorbar(timesteps, activation_diff_means, yerr=np.sqrt(activation_diff_vars), 
                            label='Activation Diff', color='blue', capsize=5)
            ax1[row, column].set_xticks(range(0, len(activaton_data['activation']), len(activaton_data['activation']) // 10))
            ax1[row, column].set_xlabel('Timestep')
            if relative:
                ax1[row, column].set_ylabel('Mean of Relative Differences')
            else:
                ax1[row, column].set_ylabel('Mean of Absolute Differences')
            ax1[row, column].legend()
            ax1[row, column].set_title(f'{layer_name} {layer_num} Activation Diff')
            
        self.activation_cache.clear()


class AttnProcessor2_0(AttnProcessorExperimentBase):
    def __init__(self):
        super().__init__()

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

        self.update_cache('k', key)
        self.update_cache('v', value)

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

        self.update_cache('a', hidden_states)

        self.timestep += 1
        return hidden_states


class xFuserCogVideoXAttnProcessor2_0(AttnProcessorExperimentBase):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        latent_seq_length = hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        self.update_cache('k', key[:,:,text_seq_length:])
        self.update_cache('v', value[:,:,text_seq_length:])
        self.update_cache('ek', key[:,:,:text_seq_length])
        self.update_cache('ev', value[:,:,:text_seq_length])

        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )

        assert text_seq_length + latent_seq_length == hidden_states.shape[1]
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, latent_seq_length], dim=1
        )
        self.update_cache('a', hidden_states)
        self.update_cache('ea', encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class xFuserJointAttnProcessor2_0(AttnProcessorExperimentBase):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        self.update_cache('k', key)
        self.update_cache('v', value)
        self.update_cache('ek', encoder_hidden_states_key_proj)
        self.update_cache('ev', encoder_hidden_states_value_proj)

        # attention
        query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        # Split the attention outputs.
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        self.update_cache('a', hidden_states)
        self.update_cache('ea', encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states


class FluxAttnProcessor2_0(AttnProcessorExperimentBase):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            encoder_size = encoder_hidden_states_query_proj.shape[2]
            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        if encoder_hidden_states is not None:
            self.update_cache('k', key[:,:,encoder_size:])
            self.update_cache('v', value[:,:,encoder_size:])
            self.update_cache('ek', key[:,:,:encoder_size])
            self.update_cache('ev', value[:,:,:encoder_size])
        else:
            self.update_cache('k', key)
            self.update_cache('v', value)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            self.update_cache('a', hidden_states)
            self.update_cache('ea', encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            self.update_cache('a', hidden_states)

            return hidden_states
