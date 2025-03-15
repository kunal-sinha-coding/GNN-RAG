import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import json

model_name = "meta-llama/Llama-2-7b-chat-hf"
device = "cuda" = torch.cuda.is_available() else "cpu"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=quantization_config,
    device_map="auto"
).to(device)

'''
In a loop:
Generate X question-answer pairs
For each question, generate the entities and the associated reasoning path
Update entities.txt, relations.txt
Update the subgraph
'''

PROMPTS = {
    "qa": (
        '''Generate {num_questions} question-answer pairs.
        These are questions a person could ask regarding NVIDIA's financial performance.
        Each question requires synthesizing two pieces of information.
        Each answer is 1 sentence long. 
        Format the response as the following on one line:
        \{\"id\": \"nvidia-1\"\, \"question\": ..., \"answer\": ...\}
        Id number starts at {id_num}.
        Generate each dictionary on a new line.'''
    )
}

def generate(prompt):
    inputs = tokernizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens)
    return response

def synthesize_step(num_questions=5, id_num=0):
    response = generate(PROMPTS["qa"].format(num_questions), id_num=id_num)
    qa_pairs = [json.loads(resp) for resp in "\n".split(response)]
    import pdb; pdb.set_trace()
