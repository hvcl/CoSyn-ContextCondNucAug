# import torch
# import torch.nn as nn
# import math

# from transformers import CLIPTokenizer, CLIPTextModel, AutoTokenizer
# from transformers.models.clip.modeling_clip import _make_causal_mask, _expand_mask


# def clip_transformer_forward(model, input_ids_list, attention_mask, class_embed=None):
#     # this is a hack to get the CLIP transformer to work with long captions
#     # class_embed is concatenated to the input embeddings

#     output_attentions = model.config.output_attentions
#     output_hidden_states = model.config.output_hidden_states
#     return_dict = model.config.use_return_dict

#     sz = input_ids_list[0].size()
#     input_shape = (sz[0], sz[1] * len(input_ids_list))

#     hidden_states_list = []

#     for input_ids in input_ids_list:
#         hidden_states = model.embeddings(input_ids)
#         hidden_states_list.append(hidden_states)

#     hidden_states = torch.cat(hidden_states_list, dim=1)

#     if class_embed is not None:
#         input_shape = (input_shape[0], 1 + input_shape[1])
#         class_embed = class_embed.unsqueeze(1)
#         hidden_states = torch.cat([class_embed, hidden_states], dim=1)

#     # causal mask is applied over the whole sequence (154 tokens)
#     causal_attention_mask = _make_causal_mask(
#         input_shape, hidden_states.dtype, device=hidden_states.device
#     )

#     # expand attention_mask
#     if attention_mask is not None:
#         # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
#         attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

#     encoder_outputs = model.encoder(
#         inputs_embeds=hidden_states,
#         attention_mask=attention_mask,
#         causal_attention_mask=causal_attention_mask,
#         output_attentions=output_attentions,
#         output_hidden_states=output_hidden_states,
#         return_dict=return_dict,
#     )

#     last_hidden_state = encoder_outputs[0]
#     last_hidden_state = model.final_layer_norm(last_hidden_state)

#     return last_hidden_state


# class FrozenCLIPEmbedder(nn.Module):
#     """Uses the openai CLIP transformer encoder for text (from Hugging Face)"""

#     def __init__(
#         self,
#         version="vinid/plip",
#         device="cpu",
#         max_length=154,
#     ):
#         super().__init__()
#         try:
#             self.tokenizer = CLIPTokenizer.from_pretrained(version)
#             self.clip_max_length = self.tokenizer.model_max_length
#         except:
#             # when using plip model
#             self.tokenizer = AutoTokenizer.from_pretrained(version)
#             self.clip_max_length = 77
#         self.transformer = CLIPTextModel.from_pretrained(version)
#         self.device = device
#         self.max_length = self.clip_max_length * math.ceil(
#             max_length / self.clip_max_length
#         )
#         self.freeze()

#     def freeze(self):
#         self.transformer = self.transformer.eval()
#         for param in self.parameters():
#             param.requires_grad = False

#     def forward(self, text):
#         batch_encoding = self.tokenizer(
#             text,
#             truncation=True,
#             max_length=self.max_length,
#             return_length=True,
#             return_overflowing_tokens=False,
#             padding="max_length",
#             return_tensors="pt",
#         )
        
#         input_ids = batch_encoding["input_ids"].to(self.device)
#         attention_mask = batch_encoding["attention_mask"].to(self.device)

#         if input_ids.shape[1] != self.clip_max_length:
#             input_ids_list = input_ids.split(self.clip_max_length, dim=-1)
#         else:
#             input_ids_list = [input_ids]

#         z = clip_transformer_forward(self.transformer.text_model, input_ids_list, attention_mask)
#         return z

#     @torch.no_grad()
#     def encode(self, text):
#         return self(text)
    

# # import os
# # os.environ["CUDA_VISIBLE_DEVICES"]= "5"
# # a = FrozenCLIPEmbedder()
# # b = a('The woman wears earrings. She has wavy hair. She is attractive.')
# # print(b.shape) # torch.Size([1, 154, 512])



import math
import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel, AutoTokenizer
from transformers.models.clip.modeling_clip import _make_causal_mask, _expand_mask

DEFAULT_PLIP_NAME = "vinid/plip"
PLIP_EMBED_DIM = 512
MAX_LENGTH = 154


class FrozenCLIPEmbedder(nn.Module):
    """Uses the openai CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(
        self,
        version=DEFAULT_PLIP_NAME,
        device="cuda",
        max_length=MAX_LENGTH,
    ):
        super().__init__()
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(version)
            self.clip_max_length = self.tokenizer.model_max_length
        except:
            # when using plip model
            self.tokenizer = AutoTokenizer.from_pretrained(version)
            self.clip_max_length = 77
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = self.clip_max_length * math.ceil(
            max_length / self.clip_max_length
        )
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text, return_attn_mask=True):
        # print("return_attn_mask: ", return_attn_mask)
        # exit()
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = batch_encoding["input_ids"].to(self.device)
        attention_mask = batch_encoding["attention_mask"].to(self.device)

        if input_ids.shape[1] != self.clip_max_length:
            input_ids_list = input_ids.split(self.clip_max_length, dim=-1)
        else:
            input_ids_list = [input_ids]

        z, attn_mask = clip_transformer_forward(self.transformer.text_model, input_ids_list, attention_mask)

        if return_attn_mask:
            return z, attn_mask

        return z

    @torch.no_grad()
    def encode(self, text):
        return self(text)
    

def clip_transformer_forward(model, input_ids_list, attention_mask, class_embed=None):
    # this is a hack to get the CLIP transformer to work with long captions
    # class_embed is concatenated to the input embeddings

    output_attentions = model.config.output_attentions
    output_hidden_states = model.config.output_hidden_states
    return_dict = model.config.use_return_dict

    sz = input_ids_list[0].size()
    input_shape = (sz[0], sz[1] * len(input_ids_list))

    hidden_states_list = []

    for input_ids in input_ids_list:
        hidden_states = model.embeddings(input_ids)
        hidden_states_list.append(hidden_states)

    hidden_states = torch.cat(hidden_states_list, dim=1)

    if class_embed is not None:
        input_shape = (input_shape[0], 1 + input_shape[1])
        class_embed = class_embed.unsqueeze(1)
        hidden_states = torch.cat([class_embed, hidden_states], dim=1)

    # causal mask is applied over the whole sequence (154 tokens)
    causal_attention_mask = _make_causal_mask(
        input_shape, hidden_states.dtype, device=hidden_states.device
    )

    # expand attention_mask
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attention_mask = _expand_mask(attention_mask, hidden_states.dtype)
    # print("attention_mask before", attention_mask.shape) # [1, 154]
    # print("attention_mask after", attention_mask.shape) # [1, 1, 154, 154]
    # exit()

    encoder_outputs = model.encoder(
        inputs_embeds=hidden_states,
        attention_mask=expanded_attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = model.final_layer_norm(last_hidden_state)

    attention_mask = attention_mask.to(torch.bool)

    return last_hidden_state, attention_mask