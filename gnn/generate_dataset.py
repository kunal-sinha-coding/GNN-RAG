import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os
from tqdm import tqdm
import re

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
    {prompt}
    {context}
    [/INST]
    '''
)
QUESTION_END_TOKEN = "[END]"
QA_FORMAT = {
    "id": "",
    "question": "",
    "answer": "",
}
PATH_FORMAT = {
    "paths": [ ["NVIDIA", "relation.earned_in_Q42025", "$200 billion" ] ]
}
PROMPTS = {
    "qa": (
        '''
        Generate {num_questions} question-answer pairs.
        These are questions a person could ask regarding tech companies' finances.
        Each question requires synthesizing multiple different pieces of information.
        Each answer is 1 sentence long. 
        Here is an example of the correct format:
        {qa_format}
        The id field is an integer starting at {id_num} and incrementing by 1 onwards.
        '''
    ),
    "paths": (
        '''
        Output the dictionaries above with a new field added to each called \"paths\".
        This field contains a list of reasoning paths in a knowledge graph that could answer the question.
        Each dictionary can contain one or more paths. Each path can be of any length.
        Keep everything else in the dictionary the same except for the new field.
        here is an example of the correct format:
        {qa_format_paths}
        '''
    ),
}

def generate(prompt, context=""):
    llm_prompt = LLM_PROMPT.format(prompt=prompt, context=context)
    inputs = tokenizer(llm_prompt, return_tensors="pt").to(device)
    inputs_len = inputs.input_ids.size(-1)
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0][inputs_len:], skip_special_tokens=True)
    return response

def update_qa_pairs(qa_pairs, response, num_questions, id_num, append=False):
    idx = id_num
    response = response.replace("\n", "").strip()
    response_split = response.split("}")
    if append:
        for i in range(id_num, id_num + num_questions):
            qa_pairs.append(QA_FORMAT) #Default
    for resp in response_split:
        if "{" in resp and idx < len(qa_pairs):
            start = resp.index("{")
            last_quote = resp[::-1].index("\"")
            last_bracket = resp[::-1].index("]") if "]" in resp else len(resp)
            end = len(resp) - min((last_quote, last_bracket))
            try:
                resp_dict = json.loads(resp[start:end] + "}")
                qa_pairs[idx] = resp_dict
                idx += 1
            except:
                import pdb; pdb.set_trace()

def generate_qa(qa_pairs, num_questions, id_num):
    response = generate(PROMPTS["qa"].format(
        num_questions=num_questions,
        id_num=id_num, 
        qa_format=json.dumps(QA_FORMAT),
    ))
    update_qa_pairs(qa_pairs, response, num_questions, id_num, append=True)

def update_id_dicts(qa_pairs, num_questions, id_num, subgraph, entity2id, relation2id, vocab):
    qa_pairs_batch = qa_pairs[id_num : id_num + num_questions]
    for pair in qa_pairs_batch:
        for word in re.split(r"[\?\.\! ]", pair["question"].lower()):
            if word not in vocab and word.strip() != "":
                vocab[word] = len(vocab)
        if pair["id"] == "":
            continue
        for path in pair["paths"]:
            path_ids = []
            for i, ent in enumerate(path):
                if i % 2 == 0:
                    if ent not in entity2id:
                        entity2id[ent] = len(entity2id)
                    path_ids.append(entity2id[ent])
                else:
                    if ent not in relation2id:
                        relation2id[ent] = len(relation2id)
                    path_ids.append(relation2id[ent])
            subgraph["tuples"].append(path_ids)
    subgraph["entities"] = list(range(len(entity2id)))

def generate_paths(qa_pairs, num_questions, id_num,
                    subgraph, entity2id, relation2id, vocab):
    qa_pairs_batch = qa_pairs[id_num : id_num + num_questions]
    qa_pairs_str = QUESTION_END_TOKEN.join([
        json.dumps(pair) for pair in qa_pairs_batch
    ])
    qa_format_paths = {**QA_FORMAT, **PATH_FORMAT}
    response = generate(PROMPTS["paths"].format(
        qa_format_paths=json.dumps(qa_format_paths),
    ), context="Dictionaries:\n" + qa_pairs_str)
    update_qa_pairs(qa_pairs, response, num_questions, id_num)
    update_id_dicts(
        qa_pairs, num_questions, id_num,
        subgraph, entity2id, relation2id, vocab
    )

def synthesize_step(qa_pairs, num_questions, id_num, 
                    subgraph, entity2id, relation2id, vocab):
    generate_qa(qa_pairs, num_questions, id_num)
    generate_paths(
        qa_pairs, num_questions, id_num, 
        subgraph, entity2id, relation2id, vocab
    )

def save_qa_pairs(qa_pairs, num_questions, id_num, file_name, overwrite=True):
    qa_pairs_batch = qa_pairs[id_num : id_num + num_questions]
    if not os.path.isdir(FOLDER_NAME):
        os.mkdir(FOLDER_NAME)
    file_name = os.path.join(FOLDER_NAME, file_name)
    permissions = "w" if id_num == 0 and overwrite else "a"
    with open(file_name, permissions) as f:
        for pair in qa_pairs_batch:
            if pair['id'] != "":
                f.write(json.dumps(pair) + "\n")

def save_id_dicts(qa_pairs, subgraph, entity2id, relation2id, vocab, qa_file,
                subgraph_file="subgraph.txt", entities_file="entities.txt", 
                relations_file="relations.txt", vocab_file="vocab.txt"):
    subgraph_str = json.dumps(subgraph).replace("\n", "").strip()
    with open(os.path.join(FOLDER_NAME, subgraph_file), "w") as f:
        f.write(subgraph_str)
    with open(os.path.join(FOLDER_NAME, qa_file), "w") as f:
        for pair in qa_pairs:
            if pair["id"] != "":
                pair["subgraph"] = subgraph
                pair["entities"] = [entity2id[path[0]] for path in pair["paths"]]
                f.write(json.dumps(pair) + "\n")
    with open(os.path.join(FOLDER_NAME, entities_file), "w") as f:
        for entity in entity2id.keys():
            f.write(entity + "\n")
    with open(os.path.join(FOLDER_NAME, relations_file), "w") as f:
        for relation in relation2id.keys():
            f.write(relation + "\n")
    with open(os.path.join(FOLDER_NAME, vocab_file), "w") as f:
        for word in vocab.keys():
            f.write(word + "\n")

def synthesize(num_steps=100, num_questions=10, split="train"):
    qa_pairs = []
    subgraph = {"tuples": [], "entities": []}
    entity2id = {}
    relation2id = {}
    vocab = {}
    idx = 0
    for i in tqdm(range(num_steps), desc=f"Generating {split} data"):
        id_num = idx * num_questions
        synthesize_step(
            qa_pairs,
            num_questions=num_questions, 
            id_num=(idx * num_questions),
            subgraph=subgraph,
            entity2id=entity2id,
            relation2id=relation2id,
            vocab=vocab
        )
        qa_file = f"{split}.json"
        save_qa_pairs(qa_pairs, num_questions, id_num, qa_file)
        save_id_dicts(qa_pairs, subgraph, entity2id, relation2id, vocab, qa_file)
        idx += 1

synthesize()
