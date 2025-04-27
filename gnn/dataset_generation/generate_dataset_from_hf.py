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
SEP_TOKEN = "\n"
PROMPTS = {
    "paths_single": (
        '''
        Imagine we have a knowledge graph containing financial information.
        For the question-answer pair, output a hypothetical list of paths in the graph that could answer the question.
        Each path connects entities with relations and can be of varying lengths.
        Each path starts with an entity, alternates between entities and relations, then ends with an entity.
        Each relation starts with "relation."
        Do not output an empty list.
        The entire output should be on one line.
        Here is an example of the format that the output should be in:
        {fewshot_paths}
        Question-answer pair:
        {pairs_str}
        '''
    ),
    "distractors_single": (
        '''
        Imagine we have a knowledge graph containing financial information.
        We have a list of paths in the graph here:
        {previous_paths}
        Output a new path in the same format. Only output the new path.
        The new path has one entity or relation that is the same as the one in the corresponding original path. However, all the other entities and relations are different.
        Each path connects entities with relations and can be of varying lengths.
        Each path starts with an entity, alternates between entities and relations, then ends with an entity.
        Each relation starts with "relation."
        Do not output an empty list.
        The entire output should be on one line.
        Here is an example of the format that the output should be in:
        {fewshot_paths}
        '''
    ),
    "paths": (
        '''
        Imagine we have a knowledge graph containing financial information.
        For each question-answer pair, output a list of paths on its own line.
        Each path is a hypothetical path in the knowledge graph that could answer the question.
        Each path connects entities with relations and can be of varying lengths.
        Each path starts with an entity, alternates between entities and relations, then ends with an entity.
        Each relation starts with "relation."
        Each question-answer pair can correspond to one or more paths.
        The different paths in one list should be on the same line. But different lists of paths are on separate lines.
        Do not output an empty list for any question-answer pair.
        Here is an example of the format that the output should be in:
        {fewshot_paths}
        Question-answer pair:
        {pairs_str}
        '''
    ),
    "distractors": (
        '''
        Imagine we have a knowledge graph containing financial information.
        On each line, we have a list of paths here:
        {previous_paths}
        For each list, output a new path in the same format, on its own line. Only output the new path.
        The new path has one entity or relation that is the same as the one in the corresponding original path. However, all the other entities and relations are different.
        Each path is a hypothetical path in the knowledge graph.
        Each path connects entities with relations and can be of varying lengths.
        Each path starts with an entity, alternates between entities and relations, then ends with an entity.
        Each relation starts with "relation."
        Each new path should be on a separate line.
        Here is an example of the format the output should be in:
        {fewshot_paths}
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
    fewshot_paths = SEP_TOKEN.join([str(paths) for i, paths in enumerate(FEWSHOT_PATHS)])
    response = generate(PROMPTS["paths_single"].format(
        fewshot_paths=fewshot_paths,
        pairs_str=pairs_str
    ))
    updated_pairs = format_paths(response, pairs)
    return updated_pairs

def generate_distractors(pairs):
    previous_paths = SEP_TOKEN.join([str(pair["paths"]) for i, pair in enumerate(pairs)])
    fewshot_paths = SEP_TOKEN.join([str(paths) for i, paths in enumerate(FEWSHOT_PATHS)]) 
    response = generate(PROMPTS["distractors_single"].format(
        previous_paths=previous_paths,
        fewshot_paths=fewshot_paths,
    ))
    updated_pairs = format_paths(response, pairs)
    return updated_pairs

def pathstr_to_list(pathstr):
    my_paths = []
    try:
        my_paths = ast.literal_eval(pathstr)
    except:
        pass
    try:
        # Sometimes extra [ in front
        my_paths = ast.literal_eval(pathstr[1:])
    except:
        pass
    try:
        # Handle last example ending in ]]]
        my_paths = ast.literal_eval(pathstr[:-1])
    except:
        pass
    # Handle case when only one layer deep
    if len(my_paths) > 0 and isinstance(my_paths[0], str):
        my_paths = [my_paths]
    # Handle case when three layers deep
    elif len(my_paths) > 0 and len(my_paths[0]) > 0 and isinstance(my_paths[0][0], list):
        my_paths = my_paths[0]
    return my_paths

def format_paths(response, pairs):
    updated_pairs = pairs
    idx = 0
    for i, resp in enumerate(response.split(SEP_TOKEN)):
        if idx >= len(pairs):
            continue
        if "paths" not in updated_pairs[idx]:
            updated_pairs[idx]["paths"] []
        if "[" in resp and "]" in resp:
            start = resp.index("[")
            end = len(resp) - resp[::-1].index("]")
            pathstr = resp[start:end]
            my_paths = pathstr_to_list(pathstr)
            updated_pairs[idx]["paths"].extend(my_paths)
            if not my_paths:
                print(f"Could not parse: {pathstr}")
                continue
            idx += 1
    if idx < 1:
        print(response)
        import pdb; pdb.set_trace()
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

def synthesize(data_name, batch_size=1):
    for data_split in ["train", "validation"]:
        data = get_data(data_name, data_split, batch_size)
        id_num = 0
        num_batches = DATA_SIZES[data_name][data_split] // batch_size
        for batch in tqdm(data, desc=f"Generating {data_split}", total=num_batches):
            pairs = get_qa_pairs(batch, id_num)
            path_pairs = generate_paths(pairs)
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


