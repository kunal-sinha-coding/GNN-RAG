import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os
from tqdm import tqdm
import re
import argparse

model_name = "Qwen/Qwen2.5-7B-Instruct" #"meta-llama/Llama-2-7b-chat-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
cache_dir = "../../cache_dir"
max_new_tokens = {
    "Qwen/Qwen2.5-7B-Instruct": 2048,
    "meta-llama/Llama-2-7b-chat-hf": 2048
}
#model = AutoModelForCausalLM.from_pretrained(
#   model_name,
#   cache_dir=cache_dir,
#   torch_dtype=torch.float16
#).to(device)

LLAMA_PROMPT = (
    '''
    [INST] <<SYS>>
    <</SYS>>
    {prompt}
    {context}
    [/INST]
    '''
)
QUESTION_END_TOKEN = "[END]"
VOCAB_SPLIT_RE = r"[\?\.\! ]"
FEWSHOT_QA = [
    {
        "id": "",
        "question": "Which major automaker partnership did TSLA announce during Q4 FY25, and which regions is this expected to have greatest impact on revenue?",
        "answer": "Tesla announced a strategic manufacturing partnership with Toyota during Q4 FY25, with the collaboration expected to have the greatest revenue impact in the Asia-Pacific region."
    },
    {
        "id": "",
        "question": "How much did NVIDIA's data center revenue grow year-over-year in Q4 FY25, and what new product drove significant sales in this segment?",
        "answer": "NVIDIA's data center revenue grew 93% year-over-year to a record $35.6 billion in Q4 FY25, with Blackwell AI supercomputers achieving billions of dollars in sales in its first quarter."
    }
]
FEWSHOT_PATHS = [
    {
        "paths": [
            ["TSLA", "relation.announced_partnership", "Toyota", "relation.during_time_period", "Q4 FY25"],
            ["TSLA Toyota partnership", "relation.affects_revenue", "Asia-Pacific region"]
        ]
    },
    {
        "paths": [ 
            ["NVIDIA", "relation.has_business_segment", "Data Center", "relation.had_Q4FY25_revenue", "$35.6 billion"],
            ["NVIDIA", "relation.has_business_segment", "Data Center", "relation.had_YoY_growth_in_Q4FY25", "93%"],
            ["NVIDIA", "relation.produces", "Blackwell AI supercomputers", "relation.achieved_sales_in_Q4Y25", "billions of dollars"]
        ]
    }
]
PROMPTS = {
    "qa": (
        '''
        Generate {num_questions} question-answer pairs.
        These are questions a person could ask regarding tech companies' finances.
        Each question requires synthesizing multiple different pieces of information.
        Each answer is 1 sentence long. 
        Here is an example of the correct format:
        {fewshot_qa}
        The id field is an integer starting at {id_num} and incrementing by 1 onwards.
        '''
    ),
    "paths": (
        '''
        Output the dictionaries below with a new field added to each called \"paths\".
        This field contains a list of reasoning paths in a knowledge graph that could answer the question.
        Each path connects entities with relations and can be of varying lengths.
        Each dictionary can contain one or more paths.
        Keep everything else in the dictionary the same except for the new field.
        Here is an example of the correct format. The path alternates between entities and relations:
        {fewshot_qa_paths}
        Only output the new dictionaries.
        '''
    ),
}

def generate(prompt, context=""):
    if "llama" in model_name.lower():
        llm_prompt = LLAMA_PROMPT.format(prompt=prompt, context=context)
    elif "qwen" in model_name.lower():
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": context}
        ]
        llm_prompt = [tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )]
    inputs = tokenizer(llm_prompt, return_tensors="pt").to(device)
    inputs_len = inputs.input_ids.size(-1)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens[model_name])
    response = tokenizer.decode(outputs[0][inputs_len:], skip_special_tokens=True)
    return response

def get_json(string):
    try:
        return json.loads(string)
    except:
        return None

def generate_qa(qa_pairs, num_questions, id_num):
    response = generate(PROMPTS["qa"].format(
        num_questions=num_questions,
        id_num=id_num, 
        fewshot_qa=json.dumps(FEWSHOT_QA),
    ))
    update_qa_pairs(qa_pairs, response, num_questions, id_num, append=True)

def generate_paths(qa_pairs, num_questions, id_num):
    qa_pairs_batch = qa_pairs[id_num : id_num + num_questions]
    qa_pairs_str = "Dictionary:\n" + "\n".join([json.dumps(pair) for pair in qa_pairs_batch])
    fewshot_qa_paths = [
        {**qa, **paths} for qa, paths in zip(FEWSHOT_QA, FEWSHOT_PATHS)
    ]
    response = generate(PROMPTS["paths"].format(
        fewshot_qa_paths=json.dumps(fewshot_qa_paths),
    ), context=qa_pairs_str)
    update_qa_pairs(qa_pairs, response, num_questions, id_num)

def update_qa_pairs(qa_pairs, response, num_questions, id_num, append=False):
    idx = id_num
    response = response.replace("\n", "").strip()
    response_split = response.split("}")
    if append:
        for i in range(id_num, id_num + num_questions):
            qa_pairs.append("")
    for resp in response_split:
        if "{" in resp and idx < len(qa_pairs):
            start = resp.index("{")
            end = len(resp) - resp[::-1].index("\"")
            resp = resp[start:end]
            resp_dict = get_json(resp + "}")
            if not resp_dict:
                try:
                    resp_dict = get_json(resp + "]]}")
                except:
                    print("Cannot convert to json: ", resp[start:end])
            if resp_dict:
                qa_pairs[idx] = resp_dict
            idx += 1

def save_qa_pairs(qa_pairs, num_questions, id_num, qa_file, folder_name, overwrite=True):
    qa_pairs_batch = qa_pairs[id_num : id_num + num_questions]
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    qa_file = os.path.join(folder_name, qa_file)
    perm = "w" if overwrite else "a"
    with open(qa_file, perm) as f:
        for pair in qa_pairs_batch:
            if "paths" in pair:
                f.write(json.dumps(pair) + "\n")

def update_subgraph_and_dicts(pairs, subgraph, entity2id, relation2id, vocab, tuple_len=3):
    for pair in pairs:
        if "paths" not in pair:
            print("Paths not found ", pair)
            continue
        for word in re.split(VOCAB_SPLIT_RE, pair["question"].lower()):
            if word not in vocab and word.strip() != "":
                vocab[word] = len(vocab)
        for path in pair["paths"]:
            path_ids = []
            for i, ent in enumerate(path):
                if i % 2 == 0:
                    if ent not in entity2id:
                        entity2id[ent] = len(entity2id)
                    ent_id = entity2id[ent]
                    path_ids.append(ent_id)
                    if ent_id not in subgraph["entities"]:
                        subgraph["entities"].append(ent_id)
                else:
                    if ent not in relation2id:
                        relation2id[ent] = len(relation2id)
                    path_ids.append(relation2id[ent])
                if len(path_ids) == tuple_len:
                    subgraph["tuples"].append(path_ids)
                    path_ids = path_ids[-1:]

def save_subgraph(pairs, subgraph, entity2id, qa_file):
    with open(qa_file, "w") as f:
        for pair in pairs:
            if "paths" in pair:
                pair["subgraph"] = subgraph
                pair["entities"] = list(set([entity2id[path[0]] for path in pair["paths"]]))
                f.write(json.dumps(pair) + "\n")

def save_dicts(entity2id, relation2id, vocab, folder_name,
                entities_file="entities.txt", relations_file="relations.txt", vocab_file="vocab.txt"):
    entities_file = os.path.join(folder_name, entities_file)
    with open(entities_file, "w") as f:
        for entity in entity2id.keys():
            f.write(entity + "\n")
    relations_file = os.path.join(folder_name, relations_file)
    with open(relations_file, "w") as f:
        for relation in relation2id.keys():
            f.write(relation + "\n")
    vocab_file = os.path.join(folder_name, vocab_file)
    with open(vocab_file, "w") as f:
        for word in vocab.keys():
            f.write(word + "\n")

def split_data(qa_file, src_file, dst_files, folder_name, keep=.9):
    lines = []
    with open(os.path.join(folder_name, qa_file), "r") as f:
        lines = f.readlines()
    num_keep = int(keep * len(lines))
    with open(os.path.join(folder_name, src_file), "w") as src:
        src.writelines(lines[:num_keep])
    for dst_f in dst_files:
        with open(os.path.join(folder_name, dst_f), "w") as dst:
            dst.writelines(lines[num_keep:])

def synthesize_step(qa_pairs, num_questions, id_num, num_distractors=1):
    for i in range(num_distractors + 1):
        curr_pairs, curr_id_num = [], 0
        if i == 0:
            curr_pairs, curr_id_num = qa_pairs, id_num
        generate_qa(curr_pairs, num_questions, curr_id_num)
        generate_paths(curr_pairs, num_questions, curr_id_num)

def synthesize(num_steps=10000, num_questions=10, num_generations=10, qa_file="all.json", folder_name="fin-cand"):
    qa_pairs = []
    idx = 0
    for i in range(num_steps):
        for j in tqdm(range(num_generations), desc=f"Generating data"):
            id_num = idx * num_questions
            synthesize_step(
                qa_pairs,
                num_questions=num_questions, 
                id_num=id_num
            )
            save_qa_pairs(qa_pairs, num_questions, id_num, qa_file, folder_name, overwrite=(id_num==0))
            idx += 1
        split_data(qa_file, f"train.json", [f"dev.json"], folder_name)

def combine_data(qa_files=["train", "dev"], dst_folder="fin-cand", 
        src_folders=None, subgraph_size=50,
        entities_file="entities.txt", relations_file="relations.txt", vocab_file="vocab.txt"):
    entity2id, relation2id, vocab = {}, {}, {}
    #Combine the qa pairs
    for qa_file in qa_files:
        full_dst_folder = os.path.join("data", dst_folder)
        if not os.path.isdir(full_dst_folder):
            os.makedirs(full_dst_folder)
        dst_file = os.path.join(full_dst_folder, f"{qa_file}.json")
        if not src_folders:
            src_folders = [f"{dst_folder}-{i}" for i in range(4)]
        all_pairs = []
        with open(dst_file, "w") as dst:
            dst.write("")
            for src_fold in src_folders:
                src_file = os.path.join("data", src_fold, f"{qa_file}.json")
                with open(src_file, "r") as src:
                    lines = list(src.readlines())
                    #Handle the dictionaries
                    idx = 0
                    while idx < len(lines):
                        subgraph = {"tuples": [], "entities": []}
                        start, end = idx * subgraph_size, (idx + 1) * subgraph_size
                        pairs = [json.loads(l) for l in lines[start:end]]
                        update_subgraph_and_dicts(pairs, subgraph, entity2id, relation2id, vocab)
                        save_subgraph(pairs, subgraph, entity2id, src_file)
                        all_pairs.extend(pairs)
                        idx += subgraph_size
            dst.writelines([json.dumps(pair) + "\n" for pair in all_pairs])
    save_dicts(entity2id, relation2id, vocab, full_dst_folder)

#parser = argparse.ArgumentParser()
#parser.add_argument('--data_name', help='Name of the data to generate')
#args = parser.parse_args()
#folder_name = os.path.join("data", args.data_name)
#synthesize(folder_name=folder_name)
combine_data()


