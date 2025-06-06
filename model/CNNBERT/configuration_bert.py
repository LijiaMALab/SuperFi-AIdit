# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from transformers import BertConfig as TransformersBertConfig


class BertConfig(TransformersBertConfig):

    def __init__(
        self,
        kerple_starting_size: int = 512,
        attention_probs_dropout_prob: float = 0.0,
        **kwargs,
    ):
        """Configuration class for MosaicBert.

        Args:
            alibi_starting_size (int): Use `alibi_starting_size` to determine how large of an alibi tensor to
                create when initializing the model. You should be able to ignore this parameter in most cases.
                Defaults to 512.
            attention_probs_dropout_prob (float): By default, turn off attention dropout in Mosaic BERT2
                (otherwise, Flash Attention will be off by default). Defaults to 0.0.
        """
        super().__init__(
            attention_probs_dropout_prob=attention_probs_dropout_prob, **kwargs)
        self.kerple_starting_size = kerple_starting_size
