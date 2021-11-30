import torch
import torch.nn as nn

import transformers
from transformers.models.bart.modeling_bart import BartForConditionalGeneration, BartConfig

class BartSummaryModel(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super(BartSummaryModel, self).__init__(config)
        self.classifier = nn.Linear(config.d_model, 1)
    
    def classify(
        self,
        input_ids=None,
        attention_mask=None,
        bos_positions=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
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
        logits = self.classifier(last_hidden_state) # shape: [B, L, 1]
        logits = logits.squeeze(-1) # shape: [B, L]

        mask = torch.ones_like(logits)
        if bos_positions is not None:
            for i, pos in enumerate(bos_positions):
                mask[i].index_fill_(0, pos, 0)
        else:
            mask = mask.masked_fill(input_ids == self.config.bos_token_id, 0)
        
        logits = logits.masked_fill(mask == 1, -1e9)
        
        return logits