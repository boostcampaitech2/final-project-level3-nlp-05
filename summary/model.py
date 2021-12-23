from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

import transformers
from transformers.models.bart.modeling_bart import BartClassificationHead, BartForConditionalGeneration, BartConfig
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput
from transformers.file_utils import ModelOutput


@dataclass
class SentenceClassifierOutput(ModelOutput):
    # TODO: Rewrite Docs
    """
    Base class for outputs of sentence-level classification models.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_max_tokens)`):
            Classification scores (before Sigmoid).
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2 tensors
            of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
            shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


class BartSummaryModel(BartForConditionalGeneration):
    """This original implementation uses Bart's encoder for sentence-level classification task.
    However, there exist several problems including
        1. It only uses the features extracted from the encoder layer. 
            The assumption behind this decision was the encoder is in charge of embedding input sentences, 
            while the decoder is taking care of generating sentences conditioned on encoder's final hidden states. 
            However, compared to encoder-only models such as BERT and RoBERTa, the BART base model actually lacks 6 encoders, 
            and 6 transformer architectures are used in the decoder. Therefore, in order to fully utilize
            the language-modeling potentials of a BART model, we decided to use both the encoder and the decoder.
        
        2. BCELoss value is too high. It is mainly because the loss is calculated using full-size matrices and
            aggreated using summation (calculating mean is not feasible because it reduces the loss significantly due to many zeros).
            In other words, although elements unrelated for sentence-level classification task of the output logit matrix are masked with `-1e9` (minus infinity) 
            and relavent elements whose corresponding inputs are `<bos>` (or `<eos>`) tokens only remain in the matrix,
            small amount of positive errors might cause undesirable behaviours. Also, the calculation overload exists. 
            It is better to use `torch.gather` function to selectively choose logits which correspond to `<bos>` (or `<eos>`) tokens.

    Also, there are some important differences between creating a classification model using decoder architectures and encoder architectures.
        3. `<eos>` tokens must be used for decoder-classification (or seq2seq) models. 
            Since only can a decoder refer to the previous output values of itself (not future values) when generating outputs, 
            it is more appropriate and reasonable to use hidden states at the end of each sentence rather than
            ones at the beginning. We noticed that HuggingFace's implementation of `BartForSequenceClassification` uses
            `<eos>` token's hidden states and feed them into `self.classification_head`, which is different from BERT and RoBERTa models
            that use `<cls>` or `<bos>` tokens for classification. Therefore, we need to change the model architecture as well as
            the input transformation. 

        4. `labels` argument must be used to indicate the gold sentence indicies. This is to follow the conventions of HuggingFace 
            and, therefore, make it easier to implement custom trainer class inherited from HuggingFace's original implementation.
            HuggingFace's Trainer recognizes `labels` as a valid input to the model. In our first implementation, we added
            a new argument called `bos_positions`, but now it is removed. 
    """
    def __init__(self, config: BartConfig):
        super(BartSummaryModel, self).__init__(config)

        # configurable hyperparameters
        hidden_size = 128
        num_layers = 2
        dropout = 0.2
        
        self.lstm = nn.LSTM(
            input_size=config.d_model, 
            hidden_size=hidden_size, 
            bidirectional=True, 
            batch_first=True, 
            num_layers=num_layers, 
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2*hidden_size, 1)
    
    def classify(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        last_hidden_state = encoder_outputs[0]
        h_n, _ = self.lstm(last_hidden_state)
        h_n = self.dropout(h_n)
        logits = self.classifier(h_n) # shape: [B, L, 1]
        logits = logits.squeeze(-1) # shape: [B, L]

        mask = torch.ones_like(logits)
        # TODO: fix errors when creating answer matrix
        if labels is not None:
            for i, pos in enumerate(labels):
                mask[i].index_fill_(0, pos, 0)
        else:
            mask = mask.masked_fill(input_ids == self.config.bos_token_id, 0)
        
        logits = logits.masked_fill(mask == 1, -1e9)
        
        return logits


class BartSummaryModelV2(BartForConditionalGeneration):
    def __init__(self, config: BartConfig, **kwargs):
        super(BartSummaryModelV2, self).__init__(config, **kwargs)
        self.classification_head = BartClassificationHead(
            input_dim=config.d_model,
            inner_dim=config.d_model,
            num_classes=1, # num_classes should be 1
            pooler_dropout=config.classifier_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    def classify(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> SentenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        device = self.model.device
        B = input_ids.size(0)
        MAX_NUM = torch.max(input_ids.eq(self.config.eos_token_id).sum(1))

        hidden_states = outputs[0]  # last hidden state
        sentence_representation = torch.zeros((B, MAX_NUM, self.config.d_model)).to(device) # [B, MAX_NUM, D]
        for i in range(B):
            _hidden = hidden_states[i][input_ids[i].eq(self.config.eos_token_id)]
            l = _hidden.size(0)
            sentence_representation[i, 0:l] = _hidden
        logits = self.classification_head(sentence_representation).squeeze(-1) # [B, MAX_NUM]
        
        loss = None
        if labels is not None:
            assert len(input_ids) == len(labels)
            # Create one-hot vectors indicating target sentences
            one_hot = torch.zeros((B, MAX_NUM)).to(device)
            for i in range(B):
                one_hot[i,:].index_fill_(0, labels[i][labels[i] >= 0], 1.0)
            labels = one_hot.clone()

            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
    
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return SentenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class BartSummaryModelV3(BartForConditionalGeneration):
    def __init__(self, config: BartConfig, **kwargs):
        super(BartSummaryModelV3, self).__init__(config, **kwargs)
        self.classification_head = LSTMClassificationHead(
            input_dim=config.d_model,
            inner_dim=config.d_model,
            num_classes=1, # num_classes should be 1
            pooler_dropout=config.classifier_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    def classify(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> SentenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        device = self.model.device
        hidden_states = outputs[0] # [B, L, D]
        all_logits = self.classification_head(hidden_states).squeeze(-1) # [B, L]

        B = input_ids.size(0)
        MAX_NUM = torch.max(input_ids.eq(self.config.eos_token_id).sum(1))

        # last hidden state
        logits = torch.full((B, MAX_NUM), -1e9, dtype=torch.float).to(device) # [B, MAX_NUM]
        for i in range(B):
            _logit = all_logits[i][input_ids[i].eq(self.config.eos_token_id)]
            l = _logit.size(0)
            logits[i, 0:l] = _logit
        
        loss = None
        if labels is not None:
            assert len(input_ids) == len(labels)
            # Create one-hot vectors indicating target sentences
            one_hot = torch.zeros((B, MAX_NUM)).to(device)
            for i in range(B):
                one_hot[i,:].index_fill_(0, labels[i][labels[i] >= 0], 1.0)
            labels = one_hot.clone()

            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
    
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return SentenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class LSTMClassificationHead(nn.Module):
    def __init__(self, input_dim, inner_dim, num_classes, pooler_dropout, num_layers=1, bidirectional=False):
        super().__init__()
        self.inner_dim = 2*inner_dim if bidirectional else inner_dim

        self.lstm = nn.LSTM(
                        input_size=input_dim,
                        hidden_size=inner_dim,
                        num_layers=num_layers,
                        batch_first=True,
                        bidirectional=False,
                    )
        self.dense = nn.Linear(self.inner_dim, self.inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(self.inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        out, _ = self.lstm(hidden_states)
        out = self.dense(out)
        out = torch.tanh(out)
        out = self.dropout(out)
        out = self.out_proj(out)
        return out