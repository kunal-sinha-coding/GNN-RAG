import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os
from tqdm import tqdm

model_name = "meta-llama/Llama-2-7b-chat-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
).to(device)

FOLDER_NAME = os.path.join("data", "synthetic")
LLM_PROMPT = (
    '''
    [INST] <<SYS>>
    <</SYS>>
    {context}
    {prompt}
    [/INST]
    '''
)
QUESTION_END_TOKEN = "END"
QA_FORMAT = {
    "id": "",
    "question": "",
    "answer": ""
}
PATH_FORMAT = {
    "reasoning_paths": [ "NVIDIA -> relation.earned_in_Q42025 -> $200 billion" ]
}
PROMPTS = {
    "qa": (
        '''
        Generate {num_questions} question-answer pairs.
        These are questions a person could ask regarding tech companies' finances.
        Each question requires synthesizing multiple different pieces of information.
        Each answer is 1 sentence long. 
        Format the response as the following on one line:
        {qa_format} {question_end_token}
        The id field is an integer starting at {id_num}.
        Output the word {question_end_token} after each dictionary.
        '''
    ),
    "paths": (
        '''
        Suppose we are retrieving from a financial knowledge graph.
        For each dictionary, add a field called \"reasoning_paths\".
        This field contains a list of paths in the graph that could answer the question.
        Format the response as the following on one line:
        {qa_format_paths} {question_end_token}
        Each dictionary can contain one or more paths. Each path can be of any length.
        Keep everything else in the dictionary the same except for the new field.
        Output the word {question_end_token} after each dictionary.
        '''
    ),
}

'''
ToDo:
    Generate entities.txt and relations.txt
    Create subgraph.txt
    Add query entities
    Create vocab.txt
'''

def generate(prompt, context=""):
    llm_prompt = LLM_PROMPT.format(prompt=prompt, context=context)
    inputs = tokenizer(llm_prompt, return_tensors="pt").to(device)
    inputs_len = inputs.input_ids.size(-1)
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0][inputs_len:], skip_special_tokens=True)
    return response

def update_dicts(qa_pairs, response, id_num, append=False):
    idx = id_num
    response = response.replace("\n", "").strip()
    for resp in response.split(QUESTION_END_TOKEN):
        if "{" in resp and "}" in resp:
            start, end = resp.index("{"), resp.index("}")
            resp_dict = json.loads(resp[start:end+1])
            if append:
                qa_pairs.append(resp_dict)
            else:
                qa_pairs[idx] = resp_dict
                idx += 1
    return

def generate_qa(qa_pairs, num_questions, id_num):
    response = generate(PROMPTS["qa"].format(
        num_questions=num_questions,
        id_num=id_num, 
        qa_format=json.dumps(QA_FORMAT),
        question_end_token=QUESTION_END_TOKEN
    ))
    update_dicts(qa_pairs, response, id_num, append=True)

def generate_paths(qa_pairs, id_num):
    qa_pairs_str = QUESTION_END_TOKEN.join([json.dumps(pair) for pair in qa_pairs])
    qa_format_paths = {**QA_FORMAT, **PATH_FORMAT}
    response = generate(PROMPTS["paths"].format(
        qa_format_paths=json.dumps(qa_format_paths),
        question_end_token=QUESTION_END_TOKEN
    ), context=qa_pairs_str)
    update_dicts(qa_pairs, response, id_num)

def synthesize_step(qa_pairs, num_questions, id_num):
    generate_qa(qa_pairs, num_questions, id_num)
    generate_paths(qa_pairs, id_num)

def save_json(qa_pairs, file_name, overwrite=False):
    if not os.path.isdir(FOLDER_NAME):
        os.mkdir(FOLDER_NAME)
    file_name = os.path.join(FOLDER_NAME, file_name)
    permissions = "w" if overwrite or not os.path.isfile(file_name) else "a"
    with open(file_name, permissions) as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair) + "\n")

def synthesize(num_steps=100, num_questions=10, split="train"):
    qa_pairs = []
    for i in tqdm(range(num_steps), desc=f"Generating {split} data"):
        synthesize_step(
            qa_pairs,
            num_questions=num_questions, 
            id_num=(i * num_questions)
        )
        save_json(qa_pairs, f"{split}.json", overwrite=True)

synthesize()
