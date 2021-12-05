import torch
import torch.nn as nn

import transformers
from transformers.models.bart.modeling_bart import BartClassificationHead, BartForConditionalGeneration, BartConfig

class BartSummaryModel(BartForConditionalGeneration):
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
        logits = self.classifier(h_n)
        # logits = self.classifier(last_hidden_state) # shape: [B, L, 1]
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
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
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
        labels=None, # TODO: torch.LongTensor 
                     # 내부적으로 우리가 [2, 1, 4] -> [0, 1, 1, 0, 1, 0, 0, 0] == [0.2, 0.8, 0.7, 0.3, 0.2, 0.4, 0.7, 0.6]
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
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
        hidden_states = outputs[0]  # last hidden state

        # TODO: 원하는 정보를 뽑아내기
        eos_mask = input_ids.eq(self.config.eos_token_id)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
            :, -1, :
        ]
        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            # TODO: labels one-hot encoding tensor 만들기!
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # TODO: Output class 만들기!
        return Seq2SeqSequenceClassifierOutput(
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
