import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os
from tqdm import tqdm
import re
import argparse
import ast
from datasets import load_dataset
from tqdm import tqdm

model_name = "Qwen/Qwen2.5-7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
cache_dir = "../../cache_dir"
max_new_tokens = {
    "Qwen/Qwen2.5-7B-Instruct": 2048,
    "meta-llama/Llama-2-7b-chat-hf": 2048
}
model = AutoModelForCausalLM.from_pretrained(
  model_name,
  cache_dir=cache_dir,
  torch_dtype=torch.float16
).to(device)

DATA_SIZES = {
    "dreamerdeo/finqa": {
        "train": 6250,
        "validation": 883 
    },
    "PatronusAI/financebench": {
        "train": 150,
    } 
}
LLAMA_PROMPT = (
    '''
    [INST] <<SYS>>
    <</SYS>>
    {prompt}
    {context}
    [/INST]
    '''
)
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
    [
        ["TSLA", "relation.announced_partnership", "Toyota", "relation.during_time_period", "Q4 FY25"],
        ["TSLA Toyota partnership", "relation.affects_revenue", "Asia-Pacific region"]
    ],
    [ 
        ["NVIDIA", "relation.has_business_segment", "Data Center", "relation.had_Q4FY25_revenue", "$35.6 billion"],
        ["NVIDIA", "relation.has_business_segment", "Data Center", "relation.had_YoY_growth_in_Q4FY25", "93%"],
        ["NVIDIA", "relation.produces", "Blackwell AI supercomputers", "relation.achieved_sales_in_Q4Y25", "billions of dollars"]
    ]
]
EOS_TOKEN = "[END]"
PROMPTS = {
    "paths": (
        '''
        Image we have a knowledge graph containing financial information.
        For each question-answer pair, output a list of paths then output {eos_token}.
        Each path is a path in the knowledge graph that could answer the question.
        Each path connects entities with relations and can be of varying lengths.
        Each path starts with an entity, alternates between entities and relations, then ends with an entity. 
        Each question-answer pair can correspond to one or more paths.
        After each list of paths for a question-answer pair, you MUST output {eos_token}.
        Here is an example of the correct format:
        {fewshot_paths}
        The question-answer pairs are provided below.
        '''
    ),
    "distractors": (
        '''
        On each line is a list of paths. Each path represents a path in a financial knowledge graph.:
        {previous_paths}
        For each list, output a new path in the same format. Only output the new path, then output {eos_token}.
        Each path connects entities with relations and can be of varying lengths.
        Each path starts with an entity, alternates between entities and relations, then ends with an entity. 
        '''
    ),
}

def get_data(data_name, data_split, batch_size):
    data = load_dataset(data_name, streaming=True, split=data_split)
    def group_batch(batch):
        return {k: [v] for k, v in batch.items()}
    data = data.map(group_batch, batched=True, batch_size=batch_size)
    return data

def generate(prompt, context=""):
    if "llama" in model_name.lower():
        llm_prompt = LLAMA_PROMPT.format(prompt=prompt, context=context)
    elif "qwen" in model_name.lower():
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        if context:
            messages.append({"role": "user", "content": context})
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

def generate_paths(pairs):
    pairs_str = "Question-answer pairs:\n" + "\n".join([json.dumps(pair) for pair in pairs])
    fewshot_paths = f"{EOS_TOKEN}\n".join([f"id: {i}, paths: {str(paths)}" for i, paths in enumerate(FEWSHOT_PATHS)])
    response = generate(PROMPTS["paths"].format(
        fewshot_paths=fewshot_paths,
        eos_token=EOS_TOKEN
    ), context=pairs_str)
    updated_pairs = format_paths(response, pairs)
    return updated_pairs

def generate_distractors(pairs):
    previous_paths = [ pair["paths"] for pair in pairs ]
    previous_paths_format = f"{EOS_TOKEN}\n".join([f"id: {i}, paths: {str(path)}" for i, path in enumerate(previous_paths)])
    response = generate(PROMPTS["distractors"].format(
        previous_paths=previous_paths_format,
        eos_token=EOS_TOKEN
    ))
    updated_pairs = format_paths(response, pairs)
    return updated_pairs

def pathstr_to_list(pathstr):
    try:
        return ast.literal_eval(pathstr)
    except:
        pass
    try:
        # Sometimes extra [ in front
        return ast.literal_eval(pathstr[1:])
    except:
        pass
    try:
        # Handle last example ending in ]]]
        return ast.literal_eval(pathstr[:-1])
    except:
        return None

def format_paths(response, qa_pairs):
    updated_pairs = qa_pairs
    idx = 0
    for i, resp in enumerate(response.split(EOS_TOKEN)):
        if "[" in resp and "]" in resp:
            start = resp.index("[")
            end = len(resp) - resp[::-1].index("]")
            pathstr = resp[start:end]
            my_paths = pathstr_to_list(pathstr)
            if my_paths:
                try:
                    if "paths" not in updated_pairs[idx]:
                        updated_pairs[idx] = []
                    updated_pairs[idx]["paths"].extend(my_paths)
                except:
                    import pdb; pdb.set_trace()
            else:
                print(f"Failed on the following: {pathstr}")
            idx += 1
    return updated_pairs

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
    for i, pair in enumerate(pairs):
        pairs[i]["subgraph"] = subgraph
        pairs[i]["entities"] = list(set([entity2id[path[0]] for path in pair["paths"]]))

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

def get_qa_pairs(batch, id_num):
    qa_pairs = [
        {"id": id_num + i, "question": batch["question"][i], "answer": batch["answer"][i]}
        for i in range(len(batch["question"]))
    ]
    return qa_pairs

def save_pairs(pairs, data_name, data_split, dst_folder=os.path.join("..", "data"), append=True):
    if data_split in ["validation", "val"]:
        data_split = "dev"
    data_name = data_name.split("/")[-1]
    dst_folder = os.path.join(dst_folder, data_name)
    perms = "a" if append else "w"
    if not os.path.isdir(dst_folder):
        os.mkdir(dst_folder)
    dst_file = os.path.join(dst_folder, f"{data_split}.json")
    with open(dst_file, perms) as f:
        f.writelines([f"{json.dumps(pair)}\n" for pair in pairs])

def synthesize(data_name, batch_size=16):
    for data_split in ["train", "validation"]:
        data = get_data(data_name, data_split, batch_size)
        id_num = 0
        num_batches = DATA_SIZES[data_name][data_split] // batch_size
        for batch in tqdm(data, desc=f"Generating {data_split}", total=num_batches):
            pairs = get_qa_pairs(batch, id_num)
            path_pairs = generate_paths(pairs)
            import pdb; pdb.set_trace()
            path_dist_pairs = generate_distractors(path_pairs)
            save_pairs(path_dist_pairs, data_name, data_split, append=(id_num > 0))
            id_num += batch_size

def combine_data(qa_files=["train", "dev"], dst_folder="fin-cand", 
        src_folders=None, subgraph_size=400,
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
                        start, end = idx, idx + subgraph_size
                        pairs = [json.loads(l) for l in lines[start:end]]
                        update_subgraph_and_dicts(pairs, subgraph, entity2id, relation2id, vocab)
                        all_pairs.extend(pairs)
                        idx += subgraph_size
            dst.writelines([json.dumps(pair) + "\n" for pair in all_pairs])
    save_dicts(entity2id, relation2id, vocab, full_dst_folder)

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', help='Name of the data to generate')
args = parser.parse_args()
synthesize(data_name=args.data_name)
# combine_data()


