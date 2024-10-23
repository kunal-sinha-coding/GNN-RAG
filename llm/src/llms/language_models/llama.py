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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Memory before LLM model: ", torch.cuda.mem_get_info()[0] / 1e9)
        self.llm_model = LlamaForCausalLM.from_pretrained(
            self.args.model_path,
            token="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"
        ).to(self.device)
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

    def calculate_perplexity(self, inputs, answer, gamma=1.0):
        k = len(inputs)
        #self.tokenizer.padding_side = "left" #Pad prompt on the left side
        prompt_encoding = self.tokenizer.batch_encode_plus(
            inputs, return_tensors="pt", add_special_tokens=True, padding=True
        )
        reader_tok = prompt_encoding.input_ids.to(self.device)
        reader_mask = prompt_encoding.attention_mask.to(self.device)
        answer_encoding = self.tokenizer.batch_encode_plus(
            answer[:1], return_tensors="pt", add_special_tokens=True #Only pick the first answer
        )
        answer_tok = answer_encoding.input_ids.repeat((k, 1))[:, 1:].to(self.device) #Cut off start token in answer
        answer_mask = answer_encoding.attention_mask.repeat((k, 1))[:, 1:].to(self.device)
        full_tok = torch.cat([reader_tok, answer_tok], dim=-1)
        full_mask = torch.cat([reader_mask, answer_mask], dim=-1)
        print("Memory before forward pass: ", torch.cuda.mem_get_info()[0] / 1e9)
        full_logits = self.llm_model(
            input_ids=full_tok[:, :-1], #Dont predict next token logits after last token
            attention_mask=full_mask[:, :-1]
        ).logits
        print("Memory after forward pass: ", torch.cuda.mem_get_info()[0] / 1e9)
        full_logits = full_logits[:, -answer_tok.size(1):] #Only look at logits for answer
        full_labels = answer_tok.masked_fill(answer_mask == 0, IGNORE_INDEX)
        token_loss = F.cross_entropy(
            full_logits.reshape(-1, full_logits.shape[-1]),
            full_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none"
        ).reshape(full_labels.shape)
        z = (full_labels > -1).sum(dim=-1)
        llm_perplexity = -token_loss.sum(dim=-1) / z
        llm_likelihood = torch.softmax(llm_perplexity / gamma, dim=-1)
        print("Perplexity: ", llm_perplexity)
        return llm_likelihood, llm_perplexity
    
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
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text # type: ignore
