from transformers import pipeline, AutoTokenizer
import torch
import torch.nn.functional as F
from .base_language_model import BaseLanguageModel
from transformers import LlamaTokenizer, LlamaForCausalLM

IGNORE_INDEX = -100

class Llama(BaseLanguageModel):
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    @staticmethod
    def add_args(parser):
        parser.add_argument('--model_path', type=str, help="HUGGING FACE MODEL or model path", default='meta-llama/Llama-2-7b-chat-hf')
        parser.add_argument('--max_new_tokens', type=int, help="max length", default=512)
        parser.add_argument('--maximun_token', type=int, help="max length of prompt", default=924)
        parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp16')

    def __init__(self, args):
        self.args = args
        self.maximun_token = args.maximun_token
        
    def load_model(self, **kwargs):
        model = LlamaForCausalLM.from_pretrained(
            **kwargs, use_fast=False, 
            token="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",
        )
        return model

    def calculate_perplexity(inputs, answer):
        llm_model = self.load_model()
        self.tokenizer.padding_side = "left"
        prompt_encoding = self.tokenizer.batch_encode_plus(
            inputs, return_tensors="pt", add_special_tokens=True, padding=True
        )
        reader_tok = prompt_encoding.input_ids[:, :-1]
        reader_mask = prompt_encoding.attention_mask[:, :-1] 
        answer_tok, answer_mask = self.tokenizer.batch_encode_plus(
            answer, return_tensors="pt", add_special_tokens=True
        )
        repeat_answer_tok = answer_tok.repeat(reader_tok.shape)
        repeat_answer_mask = answer_mask.repeat(reader_mask.shape)
        lsr_logits = llm_model(
            input_ids=reader_tok,
            attention_mask=reader_mask,
            decoder_input_ids=repeat_answer_tok,
            decoder_attention_mask=repeat_answer_mask,
            use_cache=False,
        ).logits

        lsr_labels = repeat_answer_tok.masked_fill(repeat_answer_mask == 0, IGNORE_INDEX).to(device)
        token_loss = F.cross_entropy(
            lsr_logits.reshape(-1, lsr_logits.shape[-1]),
            lsr_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction='none',
        )
        import pdb; pdb.set_trace()
        #.reshape((bsz, k, -1))
        z = (lsr_labels.reshape((bsz, k, -1)) > -1).sum(dim=-1)
        perplexity = -token_loss.sum(dim=-1) / z
        likelihood = torch.softmax(perplexity / gamma, dim=-1)
        return likelihood, perplexity
    
    def tokenize(self, text):
        return len(self.tokenizer.tokenize(text))
    
    def prepare_for_inference(self, **model_kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path,  use_fast=False, 
            token="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",
        )
        #model_kwargs.update({'use_auth_token': True})
        print("model: ", self.args.model_path)
        self.generator = pipeline("text-generation", token="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", model=self.args.model_path, tokenizer=self.tokenizer, device_map="auto", model_kwargs=model_kwargs, torch_dtype=self.DTYPE.get(self.args.dtype, None))

    @torch.inference_mode()
    def generate_sentence(self, llm_input):
        outputs = self.generator(
            llm_input, return_full_text=False, 
            max_new_tokens=self.args.max_new_tokens
        )
        return outputs[0]['generated_text'] # type: ignore
