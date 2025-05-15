# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2022, Tri Dao.

import copy
import logging
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (MaskedLMOutput,
                                           SequenceClassifierOutput)
from transformers.models.bert.modeling_bert import BertPreTrainedModel

from .bert_padding import (index_first_axis,
                           index_put_first_axis, pad_input,
                           unpad_input, unpad_input_only)

logger = logging.getLogger(__name__)


class CNNEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.kernel_sizes = config.kernel_sizes
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    config.vocab_size - 1,
                    config.kernel_num,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False
                )
                for kernel_size in self.kernel_sizes
            ])
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        output = torch.nn.functional.one_hot(input_ids, num_classes=self.config.vocab_size)
        output = output[:, :, :self.config.vocab_size - 1].permute(0, 2, 1).to(dtype=torch.float32)
        output = torch.cat([conv(output) for conv in self.conv_layers], dim=1)
        output = output * attention_mask.unsqueeze(1)
        output = output.transpose(1, 2)
        output = self.LayerNorm(output)
        output = self.dropout(output)
        return output


class BertUnpadSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
                config, 'embedding_size'):
            raise ValueError(
                f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention '
                f'heads ({config.num_attention_heads})')

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /
                                       config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.p_dropout = config.attention_probs_dropout_prob
        self.Wqkv = nn.Linear(self.all_head_size, 3 * config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor,
                max_seqlen_in_batch: int, indices: torch.Tensor,
                attn_mask: torch.Tensor, bias: torch.Tensor,
                output_attentions) -> torch.Tensor:

        qkv = self.Wqkv(hidden_states)
        qkv = pad_input(qkv, indices, cu_seqlens.shape[0] - 1,
                        max_seqlen_in_batch)  # batch, max_seqlen_in_batch, thd
        qkv = rearrange(qkv,
                        'b s (t h d) -> b s t h d',
                        t=3,
                        h=self.num_attention_heads)
        # if we have nonzero attention dropout (e.g. during fine-tuning) or no Triton, compute attention in PyTorch
        q = qkv[:, :, 0, :, :].permute(0, 2, 1, 3)  # b h s d
        k = qkv[:, :, 1, :, :].permute(0, 2, 3, 1)  # b h d s
        v = qkv[:, :, 2, :, :].permute(0, 2, 1, 3)  # b h s d
        attention_scores = torch.matmul(q, k) / math.sqrt(
            self.attention_head_size)
        attention_scores = attention_scores + bias
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        attention = torch.matmul(attention_probs, v).permute(0, 2, 1,
                                                             3)  # b s h d
        # attn_mask is 1 for attend and 0 for don't
        attention = unpad_input_only(attention, torch.squeeze(attn_mask) == 1)
        if output_attentions:
            return rearrange(attention, 'nnz h d -> nnz (h d)'), attention_probs
        else:
            return rearrange(attention, 'nnz h d -> nnz (h d)'), None


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor,
                input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertUnpadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertUnpadSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
            self,
            input_tensor: torch.Tensor,
            cu_seqlens: torch.Tensor,
            max_s: int,
            subset_idx: Optional[torch.Tensor] = None,
            indices: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            bias: Optional[torch.Tensor] = None,
            output_attentions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self_output, attentions = self.self(input_tensor, cu_seqlens, max_s, indices,
                                            attn_mask, bias, output_attentions=output_attentions)
        if subset_idx is not None:
            output = self.output(index_first_axis(self_output, subset_idx),
                                 index_first_axis(input_tensor, subset_idx))
        else:
            output = self.output(self_output, input_tensor)
        return output, attentions


class BertGatedLinearUnitMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gated_layers = nn.Linear(config.hidden_size,
                                      config.intermediate_size * 2,
                                      bias=False)
        self.act = nn.GELU(approximate='none')
        self.wo = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layernorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual_connection = hidden_states
        # compute the activation
        hidden_states = self.gated_layers(hidden_states)
        gated = hidden_states[:, :self.config.intermediate_size]
        non_gated = hidden_states[:, self.config.intermediate_size:]
        hidden_states = self.act(gated) * non_gated
        hidden_states = self.dropout(hidden_states)
        # multiply by the second matrix
        hidden_states = self.wo(hidden_states)
        # add the residual connection and post-LN
        hidden_states = self.layernorm(hidden_states + residual_connection)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertUnpadAttention(config)
        self.mlp = BertGatedLinearUnitMLP(config)
        self.num_attention_heads = config.num_attention_heads
        self.r1 = nn.Parameter(torch.normal(mean=1.0, std=0.1, size=(self.num_attention_heads, 1, 1)))
        self.r2 = nn.Parameter(torch.normal(mean=1.0, std=0.1, size=(self.num_attention_heads, 1, 1)))
        self.r3 = nn.Parameter(torch.normal(mean=1.0, std=0.1, size=(self.num_attention_heads, 1, 1)))

    def forward(
            self,
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            seqlen: int,
            subset_idx: Optional[torch.Tensor] = None,
            indices: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            kerple_bias: Optional[torch.Tensor] = None,
            attn_bias: Optional[torch.Tensor] = None,
            output_attentions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r1 = torch.clamp(self.r1, 1e-7)
        r2 = torch.clamp(self.r2, 1e-7)
        r3 = torch.clamp(self.r3, 1e-7)
        kerple_bias = -r1 * torch.log(1 + r2 * kerple_bias ** r3)

        kerple_attn_mask = attn_bias + kerple_bias
        attention_output, attentions = self.attention(hidden_states, cu_seqlens, seqlen,
                                                      subset_idx, indices, attn_mask, kerple_attn_mask,
                                                      output_attentions=output_attentions)
        layer_output = self.mlp(attention_output)
        return layer_output, attentions


class BertEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        layer = BertLayer(config)
        self.num_attention_heads = config.num_attention_heads
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

        self._current_kerple_size = int(config.kerple_starting_size)
        self.kerple = torch.zeros(
            (1, self.num_attention_heads, self._current_kerple_size,
             self._current_kerple_size))

        self.rebuild_kerple_tensor(size=config.kerple_starting_size)

    def rebuild_kerple_tensor(self,
                              size: int,
                              device: Optional[Union[torch.device, str]] = None):

        context_position = torch.arange(size, device=device)[:, None]
        memory_position = torch.arange(size, device=device)[None, :]
        relative_position = torch.abs(memory_position - context_position)
        # [n_heads, max_token_length, max_token_length]
        relative_position = relative_position.unsqueeze(0).expand(
            self.num_attention_heads, -1, -1)

        # [1, n_heads, max_token_length, max_token_length]
        assert relative_position.shape == torch.Size([self.num_attention_heads, size, size])
        self._current_kerple_size = size
        self.kerple = relative_position.unsqueeze(0)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            output_all_encoded_layers: Optional[bool] = True,
            subset_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
    ) -> List[torch.Tensor]:

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=torch.float32)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        attention_mask_bool = attention_mask.bool()
        batch, seqlen = hidden_states.shape[:2]
        hidden_states, indices, cu_seqlens, _ = unpad_input(
            hidden_states, attention_mask_bool)

        if self._current_kerple_size < seqlen:
            warnings.warn(
                f'Increasing kerple size from {self._current_kerple_size} to {seqlen}'
            )
            self.rebuild_kerple_tensor(size=seqlen, device=hidden_states.device)
        if self.kerple.device != hidden_states.device:
            # Device catch-up
            self.kerple = self.kerple.to(hidden_states.device)
        kerple_bias = self.kerple[:, :, :seqlen, :seqlen]
        attn_bias = extended_attention_mask[:, :, :seqlen, :seqlen]
        all_encoder_layers = []
        all_self_attentions = () if output_attentions else None
        if subset_mask is None:
            for layer_module in self.layer:
                hidden_states, attentions = layer_module(hidden_states,
                                                         cu_seqlens,
                                                         seqlen,
                                                         None,
                                                         indices,
                                                         attn_mask=attention_mask,
                                                         kerple_bias=kerple_bias,
                                                         attn_bias=attn_bias,
                                                         output_attentions=output_attentions)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
                if output_attentions:
                    all_self_attentions = all_self_attentions + (attentions,)
            hidden_states = pad_input(hidden_states, indices, batch, seqlen)
        else:
            for i in range(len(self.layer) - 1):
                layer_module = self.layer[i]
                hidden_states, attentions = layer_module(hidden_states,
                                                         cu_seqlens,
                                                         seqlen,
                                                         None,
                                                         indices,
                                                         attn_mask=attention_mask,
                                                         kerple_bias=kerple_bias,
                                                         attn_bias=attn_bias, )
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
                if output_attentions:
                    all_self_attentions = all_self_attentions + (attentions,)
            subset_idx = torch.nonzero(subset_mask[attention_mask_bool],
                                       as_tuple=False).flatten()
            hidden_states, attentions = self.layer[-1](hidden_states,
                                                       cu_seqlens,
                                                       seqlen,
                                                       subset_idx=subset_idx,
                                                       indices=indices,
                                                       attn_mask=attention_mask,
                                                       kerple_bias=kerple_bias,
                                                       attn_bias=attn_bias, )
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_self_attentions


class BertPooler(nn.Module):

    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.config = config

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        classify_tensor = hidden_states * attention_mask.unsqueeze(2)
        classify_tensor = classify_tensor.sum(dim=1) / attention_mask.unsqueeze(2).sum(dim=1)
        return classify_tensor


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertModel(BertPreTrainedModel):

    def __init__(self, config, add_pooling_layer=True):
        super(BertModel, self).__init__(config)
        self.embeddings = CNNEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.post_init()
        self.config = config

    def forward(
            self,
            input_ids: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_all_encoded_layers: Optional[bool] = False,
            masked_tokens_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            **kwargs
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], Optional[torch.Tensor]]:

        if self.config.use_embedding:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids).to(input_ids.device)
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids).to(input_ids.device)
            embedding_output = self.embeddings(input_ids=input_ids, attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids)
        else:
            if attention_mask is None:
                attention_mask = torch.ones((input_ids.shape[0], input_ids.shape[1])).to(input_ids.device)
            token_type_ids = None
            embedding_output = input_ids

        subset_mask = []
        first_col_mask = []

        if masked_tokens_mask is None:
            subset_mask = None
        else:
            first_col_mask = torch.zeros_like(masked_tokens_mask)
            first_col_mask[:, 0] = True
            subset_mask = masked_tokens_mask | first_col_mask

        encoder_outputs, attentions = self.encoder(
            embedding_output,
            attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            subset_mask=subset_mask,
            output_attentions=output_attentions
        )

        if masked_tokens_mask is None:
            sequence_output = encoder_outputs[-1]
            pooled_output = self.pooler(
                sequence_output,
                attention_mask=attention_mask) if self.pooler is not None else None

        else:
            # TD [2022-03-01]: the indexing here is very tricky.
            attention_mask_bool = attention_mask.bool()
            subset_idx = subset_mask[attention_mask_bool]  # type: ignore
            sequence_output = encoder_outputs[-1][masked_tokens_mask[attention_mask_bool][subset_idx]]
            if self.pooler is not None:
                pool_input = encoder_outputs[-1][
                    first_col_mask[attention_mask_bool][subset_idx]]
                pooled_output = self.pooler(pool_input, attention_mask=attention_mask)
            else:
                pooled_output = None
        if not output_all_encoded_layers:
            encoder_outputs = sequence_output
        if self.pooler is not None:
            return encoder_outputs, pooled_output, attentions
        return encoder_outputs, None, attentions


###################
# Bert Heads
###################
class BertLMPredictionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.decoder = nn.Linear(768, 4)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertForMaskedLM(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            warnings.warn(
                'If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for '
                'bi-directional self-attention.')

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:

        if (input_ids is not None) == (inputs_embeds is not None):
            raise ValueError('Must specify either input_ids or input_embeds!')
        if labels is None:
            masked_tokens_mask = None
        else:
            masked_tokens_mask = labels > 0

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            masked_tokens_mask=masked_tokens_mask,
        )
        sequence_output = outputs[0]

        prediction_scores = self.cls(sequence_output)

        loss = None
        if labels is not None:
            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            masked_token_idx = torch.nonzero(labels.flatten() > 0,
                                             as_tuple=False).flatten()
            loss = loss_fct(prediction_scores,
                            labels.flatten()[masked_token_idx])
            assert input_ids is not None, 'Coding error; please open an issue'
            batch, seqlen = input_ids.shape[:2]
            prediction_scores = rearrange(index_put_first_axis(
                prediction_scores, masked_token_idx, batch * seqlen),
                '(b s) d -> b s d',
                b=batch)

        return MaskedLMOutput(
            loss=loss,
        )


class BertForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.decoder = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        classifier_dropout = (config.classifier_dropout
                              if config.classifier_dropout is not None else
                              config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            input_ids2: Optional[torch.Tensor] = None,
            attention_mask2: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        outputs2 = self.decoder(
            input_ids2,
            attention_mask=attention_mask2,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output2 = outputs2[1]
        pooled_output2 = self.dropout(pooled_output2)

        logits = self.dense(pooled_output + pooled_output2)
        logits = self.activation(logits)
        logits = self.classifier(logits)
        logits = torch.clamp(logits, 0, 1)
        loss = None
        if labels is not None:
            # Compute loss
            loss_fct = nn.MSELoss()
            if self.num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)

        if not self.config.return_loss:
            return logits

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs[0],
            attentions=None,
        )
