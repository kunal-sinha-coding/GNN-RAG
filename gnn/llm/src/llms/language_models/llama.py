from transformers import pipeline, AutoTokenizer
import torch
import torch.nn.functional as F
from .base_language_model import BaseLanguageModel
from transformers import LlamaTokenizer, LlamaForCausalLM
from functools import reduce
import numpy as np

IGNORE_INDEX = -100

class Llama(BaseLanguageModel):
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    @staticmethod
    def add_args(parser):
        parser.add_argument('--model_path', type=str, help="HUGGING FACE MODEL or model path", default='meta-llama/Llama-2-7b-chat-hf')
        parser.add_argument('--max_new_tokens', type=int, help="max length", default=512)
        parser.add_argument('--maximun_token', type=int, help="max length of prompt", default=924)
        parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='bf16')

    def __init__(self, args):
        self.args = args
        self.maximun_token = args.maximun_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Memory before LLM model: ", torch.cuda.mem_get_info()[0] / 1e9)
        self.llm_model = LlamaForCausalLM.from_pretrained(
            self.args.model_path,
            token="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"
        ).to(self.device, dtype=torch.float16)
        print("Memory after LLM model: ", torch.cuda.mem_get_info()[0] / 1e9)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path, use_fast=False,
            token="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"
        )
        
    #def load_model(self, **kwargs):
        #model = LlamaForCausalLM.from_pretrained(
        #    self.args.model_path,
        #    **kwargs, token="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"
        #)
        #return model

    def calculate_perplexity(self, inputs, answers):
        bsz, k = np.array(inputs).shape
        #self.tokenizer.padding_side = "left" #Pad prompt on the left side
        #prompt_encoding = self.tokenizer.batch_encode_plus(
        #    inputs, return_tensors="pt", add_special_tokens=True, padding=True
        #)
        #reader_tok = prompt_encoding.input_ids.to(self.device)
        #reader_mask = prompt_encoding.attention_mask.to(self.device)
        #answer_encoding = self.tokenizer.batch_encode_plus(
        #    answer[:1], return_tensors="pt", add_special_tokens=True #Only pick the first answer
        #)
        #answer_tok = answer_encoding.input_ids.repeat((k, 1))[:, 1:].to(self.device) #Cut off start token in answer
        #answer_mask = answer_encoding.attention_mask.repeat((k, 1))[:, 1:].to(self.device)
        #full_tok = torch.cat([reader_tok, answer_tok], dim=-1)
        #full_mask = torch.cat([reader_mask, answer_mask], dim=-1)
        # Encode the full inputs
        full_inputs = []
        for i in range(bsz):
            for j in range(k):
                full_inputs.append(f"{inputs[i][j]}{answers[i][0]}")
        #Only use first answer; \n is needed for low ppl
        full_encoding = self.tokenizer.batch_encode_plus(
            full_inputs, return_tensors="pt", padding=True
        )
        full_tok = full_encoding.input_ids.to(self.device)
        full_mask = full_encoding.attention_mask.to(self.device)
        #self.tokenizer.padding_side = "left"
        #prompt_encoding = self.tokenizer.batch_encode_plus([f"{inp}\n" for inp in inputs], return_tensors="pt", padding=True)
        #prompt_tok = prompt_encoding.input_ids.to(self.device)
        #prompt_mask = prompt_encoding.attention_mask.to(self.device)
        # Encode the answer on its own so we can get its length
        #answer_encoding = self.tokenizer.batch_encode_plus([f"{answer[0]}"], return_tensors="pt")
        #answer_tok = answer_encoding.input_ids[:, 1:].to(self.device) #Ignore start token
        #answer_mask = answer_encoding.attention_mask[:, 1:].to(self.device)
        #answer_len = answer_tok.size(-1)
        #repeat_answer_tok = answer_tok.repeat(k, 1)
        #repeat_answer_mask = answer_mask.repeat(k, 1)
        #full_tok = torch.cat([prompt_tok, repeat_answer_tok], dim=-1)
        #full_mask = torch.cat([prompt_mask, repeat_answer_mask], dim=-1)
        # Get logits of full sequence
        full_logits = self.llm_model(
            input_ids=full_tok[:, :-1], # Don't predict next token logits after last token
            attention_mask=full_mask[:, :-1]
        ).logits
        vocab_len = full_logits.size(-1)
        # Because logits are for next token prediction, shift full_tok and full_mask right by 1
        full_tok, full_mask = full_tok[:, 1:], full_mask[:, 1:]
        num_inputs, seq_len = full_tok.shape
        # Flatten the inputs to F.cross_entropy so first dim is num_inputs * answer_len
        full_logits_flat = full_logits.reshape(-1, vocab_len)
        full_labels = full_tok.masked_fill(full_mask == 0, IGNORE_INDEX)
        full_labels_flat = full_labels.reshape(-1)
        # Get indices of first appearance of PAD token
        #pad_indices = (full_tok == self.tokenizer.pad_token_id).long().argmax(dim=-1)
        # If no PAD token, use seq_len
        #no_pad = (full_tok != self.tokenizer.pad_token_id).all(dim=-1)
        #pad_indices = torch.where(no_pad, seq_len, pad_indices)
        # Create indices tensors with length k * answer_len to index the tokens for the answer
        #pad_indices = pad_indices.repeat_interleave(answer_len)
        #answer_indices = torch.arange(answer_len).repeat(k).to(self.device)
        #lsr_logits = full_logits_flat[pad_indices - answer_len + answer_indices, :]
        # Get labels and flatten
        #full_tok_flat = full_tok.masked_fill(full_mask == 0, IGNORE_INDEX).view(-1)
        #lsr_labels = full_tok_flat[pad_indices - answer_len + answer_indices]
        # Compute cross entropy: flatten before passing in then undo
        ce_loss = F.cross_entropy(
            full_logits_flat,
            full_labels_flat,
            ignore_index=IGNORE_INDEX,
            reduction="none"
        ).reshape(full_labels.shape)
        question_id = self.tokenizer.encode(["INST"])[-1]
        # Get last index of question_id by reversing full_tok and doing argmax
        # We don't do seq_len-1 because we'd have to do +1 later to mask ] too
        question_indices = seq_len - (full_tok.flip(dims=[1]) == question_id).long().argmax(dim=-1)
        question_indices = question_indices[:, None].repeat(1, seq_len) #Reshape to same size as full_tok
        question_mask = torch.arange(seq_len)[None, :].repeat(num_inputs, 1).to(self.device) <= question_indices + 1 #Mask question tokens and the one spacing token afterwards
        ce_loss[question_mask] = 0 #Ignore question tokens
        ce_loss[full_tok == self.tokenizer.pad_token_id] = 0 #Ignore pad tokens
        full_labels[question_mask] = IGNORE_INDEX
        z = (full_labels != IGNORE_INDEX).sum(dim=-1) # Get average of CE loss
        ce_loss = ce_loss.sum(dim=-1) / z
        # Get indices where padding begins i.e sequence ends
        #pad_indices = (full_tok == self.tokenizer.pad_token_id).long().argmax(dim=-1)
        #pad_indices[pad_indices == 0] = seq_len # If no pad token, go to end of sequence 
        # Create 2d index tensors of shape (k, answer_len)
        #pad_indices = pad_indices[:, None].repeat(1, answer_len)
        #start_indices = pad_indices - answer_len - 1 # pad_index - 1 is the last token of answer
        #answer_indices = start_indices + torch.arange(answer_len)[None, :].repeat(k, 1).to(self.device)
        #batch_indices = torch.arange(k)[:, None].repeat(1, answer_len).to(self.device)
        # Grab the ce values of the answer tokens
        #ce_loss = ce_loss[batch_indices, answer_indices]
        #full_labels = full_labels[batch_indices, answer_indices]
        #ce_loss = ce_loss[:, -answer_len:]
        #full_labels = full_labels[:, -answer_len:]
        # Get average of the loss, not counting mask tokens
        #z = (full_labels != IGNORE_INDEX).sum(dim=-1)
        #ce_loss = ce_loss.sum(dim=-1) / z
        # Get perplexity and likelihood
        llm_perplexity = -ce_loss.exp() # Take negative because lower ppl is better
        return llm_perplexity.reshape((bsz, k))
    
    def tokenize(self, text):
        return len(self.tokenizer.tokenize(text))
    
    #def prepare_for_inference(self, **model_kwargs):
        #self.tokenizer = AutoTokenizer.from_pretrained(
        #    self.args.model_path,  use_fast=False, 
        #    token="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",
        #)
        #model_kwargs.update({'use_auth_token': True})
        #print("model: ", self.args.model_path)
        #self.generator = pipeline("text-generation", token="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", model=self.args.model_path, tokenizer=self.tokenizer, device_map="auto", model_kwargs=model_kwargs, torch_dtype=self.DTYPE.get(self.args.dtype, None))

    @torch.inference_mode()
    def generate_sentence(self, llm_input):
        llm_input_ids = self.tokenizer.encode(llm_input, return_tensors="pt").to(self.device)
        outputs = self.llm_model.generate(
            llm_input_ids, max_new_tokens=self.args.max_new_tokens
        )
        outputs = outputs[0, llm_input_ids.size(-1):] # Cut off prompt
        generated_text = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return generated_text # type: ignore
