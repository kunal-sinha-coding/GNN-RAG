import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import math
import os

from models.base_model import BaseModel
from modules.kg_reasoning.reasongnn import ReasonGNNLayer
from modules.question_encoding.lstm_encoder import LSTMInstruction
from modules.question_encoding.bert_encoder import BERTInstruction
from modules.layer_init import TypeLayer
from modules.query_update import AttnEncoder, Fusion, QueryReform

import argparse
from llm.src.llms.language_models.llama import Llama
from llm.src.qa_prediction.build_qa_input import PromptBuilder
import wandb

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000



class ReaRev(BaseModel):
    def __init__(self, args, num_entity, num_relation, num_word):
        """
        Init ReaRev model.
        """
        super(ReaRev, self).__init__(args, num_entity, num_relation, num_word)
        #self.embedding_def()
        #self.share_module_def()
        self.norm_rel = args['norm_rel']
        self.layers(args)

        self.loss_type =  args['loss_type']
        self.num_iter = args['num_iter']
        self.num_ins = args['num_ins']
        self.num_gnn = args['num_gnn']
        self.alg = args['alg']
        assert self.alg == 'bfs'
        self.lm = args['lm']
        
        self.private_module_def(args, num_entity, num_relation)

        self.to(self.device)
        self.lin = nn.Linear(3*self.entity_dim, self.entity_dim)

        self.fusion = Fusion(self.entity_dim)
        self.reforms = []
        for i in range(self.num_ins):
            self.add_module('reform' + str(i), QueryReform(self.entity_dim))
        # self.reform_rel = QueryReform(self.entity_dim)
        # self.add_module('reform', QueryReform(self.entity_dim))

        print("Memory before LLM model: ", torch.cuda.mem_get_info()[0] / 1e9)
        self.llm_args = argparse.Namespace( #ToDo: dont hardcode
            add_rule=False, cot=False, d='RoG-cwq', data_path='rmanluo', debug=False, dtype='fp16', 
            each_line=False, encrypt=False, explain=False, filter_empty=False, force=False, 
            max_new_tokens=512, maximun_token=4096, model_name='RoG', model_path='TinyLlama/TinyLlama-1.1B-Chat-v0.6', 
            n=1, predict_path='llm/results/KGQA-GNN-RAG/rearev-sbert', prompt_path='llm/prompts/llama2_predict.txt', 
            rule_path='llm/results/gen_rule_path/RoG-cwq/RoG/test/predictions_3_False.jsonl', 
            rule_path_g1='llm/results/gnn/RoG-cwq/rearev-sbert/test.info', 
            rule_path_g2='None', split='test', use_random=False, use_true=False
        )
        self.llm_model = Llama(self.llm_args)
        self.input_builder = PromptBuilder(
            self.llm_args.prompt_path,
            self.llm_args.encrypt,
            self.llm_args.add_rule,
            use_true=self.llm_args.use_true, 
            cot=self.llm_args.cot,
            explain=self.llm_args.explain,
            use_random=self.llm_args.use_random,
            each_line=self.llm_args.each_line,
            maximun_token=self.llm_model.maximun_token,
            tokenize=self.llm_model.tokenize
        )
        print("Memory after LLM model: ", torch.cuda.mem_get_info()[0] / 1e9)

    def layers(self, args):
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        entity_dim = self.entity_dim

        #self.lstm_dropout = args['lstm_dropout']
        self.linear_dropout = args['linear_dropout']
        
        self.entity_linear = nn.Linear(in_features=self.ent_dim, out_features=entity_dim)
        self.relation_linear = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)
        # self.relation_linear_inv = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)
        #self.relation_linear = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)

        # dropout
        #self.lstm_drop = nn.Dropout(p=self.lstm_dropout)
        self.linear_drop = nn.Dropout(p=self.linear_dropout)

        if self.encode_type:
            self.type_layer = TypeLayer(in_features=entity_dim, out_features=entity_dim,
                                        linear_drop=self.linear_drop, device=self.device, norm_rel=self.norm_rel)

        self.self_att_r = AttnEncoder(self.entity_dim)
        #self.self_att_r_inv = AttnEncoder(self.entity_dim)
        self.kld_loss = nn.KLDivLoss(reduction='none')
        self.bce_loss_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss()

    def get_ent_init(self, local_entity, kb_adj_mat, rel_features):
        if self.encode_type:
            local_entity_emb = self.type_layer(local_entity=local_entity,
                                               edge_list=kb_adj_mat,
                                               rel_features=rel_features)
        else:
            local_entity_emb = self.entity_embedding(local_entity)  # batch_size, max_local_entity, word_dim
            local_entity_emb = self.entity_linear(local_entity_emb)
        
        return local_entity_emb
    
   
    def get_rel_feature(self):
        """
        Encode relation tokens to vectors.
        """
        if self.rel_texts is None:
            rel_features = self.relation_embedding.weight
            rel_features_inv = self.relation_embedding_inv.weight
            rel_features = self.relation_linear(rel_features)
            rel_features_inv = self.relation_linear(rel_features_inv)
        else:
            
            rel_features = self.instruction.question_emb(self.rel_features)
            rel_features_inv = self.instruction.question_emb(self.rel_features_inv)
            
            rel_features = self.self_att_r(rel_features,  (self.rel_texts != self.instruction.pad_val).float())
            rel_features_inv = self.self_att_r(rel_features_inv,  (self.rel_texts != self.instruction.pad_val).float())
            if self.lm == 'lstm':
                rel_features = self.self_att_r(rel_features, (self.rel_texts != self.num_relation+1).float())
                rel_features_inv = self.self_att_r(rel_features_inv, (self.rel_texts_inv != self.num_relation+1).float())

        return rel_features, rel_features_inv


    def private_module_def(self, args, num_entity, num_relation):
        """
        Building modules: LM encoder, GNN, etc.
        """
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        entity_dim = self.entity_dim
        self.reasoning = ReasonGNNLayer(args, num_entity, num_relation, entity_dim, self.alg)
        if args['lm'] == 'lstm':
            self.instruction = LSTMInstruction(args, self.word_embedding, self.num_word)
            self.relation_linear = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        else:
            self.instruction = BERTInstruction(args, self.word_embedding, self.num_word, args['lm'])
            #self.relation_linear = nn.Linear(in_features=self.instruction.word_dim, out_features=entity_dim)
        # self.relation_linear = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        # self.relation_linear_inv = nn.Linear(in_features=entity_dim, out_features=entity_dim)

    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input, query_entities):
        """
        Initializing Reasoning
        """
        # batch_size = local_entity.size(0)
        self.local_entity = local_entity
        self.instruction_list, self.attn_list = self.instruction(q_input)
        rel_features, rel_features_inv  = self.get_rel_feature()
        self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat, rel_features)
        self.init_entity_emb = self.local_entity_emb
        self.curr_dist = curr_dist
        self.dist_history = []
        self.action_probs = []
        self.seed_entities = curr_dist
        
        self.reasoning.init_reason( 
                                   local_entity=local_entity,
                                   kb_adj_mat=kb_adj_mat,
                                   local_entity_emb=self.local_entity_emb,
                                   rel_features=rel_features,
                                   rel_features_inv=rel_features_inv,
                                   query_entities=query_entities)


    def calc_loss_label(self, curr_dist, teacher_dist, label_valid):
        tp_loss = self.get_loss(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss
        
    def evaluate_llm(self, question_dict):
        all_input, _ = self.input_builder.process_input_batch(question_dict, all_input=True)
        correct = [False for inp in all_input]
        for i in range(len(correct)):
            prediction = self.llm_model.generate_sentence(all_input[i]).strip()
            correct[i] = prediction in question_dict["answer"][i]
        return correct
    
    def forward(self, batch, question_dict, training=False, replug=False, top_k=1, gamma=1e5, 
                save_ppl_files=[], debug_ppl=True, overwrite_ppl=False):
        """
        Forward function: creates instructions and performs GNN reasoning.
        """

        # local_entity, query_entities, kb_adj_mat, query_text, seed_dist, answer_dist = batch
        local_entity, query_entities, kb_adj_mat, query_text, seed_dist, true_batch_id, answer_dist = batch
        local_entity = torch.from_numpy(local_entity).type('torch.LongTensor').to(self.device)
        # local_entity_mask = (local_entity != self.num_entity).float()
        query_entities = torch.from_numpy(query_entities).type('torch.FloatTensor').to(self.device)
        answer_dist = torch.from_numpy(answer_dist).type('torch.FloatTensor').to(self.device)
        seed_dist = torch.from_numpy(seed_dist).type('torch.FloatTensor').to(self.device)
        current_dist = Variable(seed_dist, requires_grad=True)

        q_input= torch.from_numpy(query_text).type('torch.LongTensor').to(self.device)
        #query_text2 = torch.from_numpy(query_text2).type('torch.LongTensor').to(self.device)
        if self.lm != 'lstm':
            pad_val = self.instruction.pad_val #tokenizer.convert_tokens_to_ids(self.instruction.tokenizer.pad_token)
            query_mask = (q_input != pad_val).float()
            
        else:
            query_mask = (q_input != self.num_word).float()

        
        """
        Instruction generations
        """
        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input, query_entities=query_entities)
        self.instruction.init_reason(q_input)
        for i in range(self.num_ins):
            relational_ins, attn_weight = self.instruction.get_instruction(self.instruction.relational_ins, step=i) 
            self.instruction.instructions.append(relational_ins.unsqueeze(1))
            self.instruction.relational_ins = relational_ins
        #relation_ins = torch.cat(self.instruction.instructions, dim=1)
        #query_emb = None
        self.dist_history.append(self.curr_dist)


        """
        BFS + GNN reasoning
        """

        for t in range(self.num_iter):
            relation_ins = torch.cat(self.instruction.instructions, dim=1)
            self.curr_dist = current_dist            
            for j in range(self.num_gnn):
                self.curr_dist, global_rep = self.reasoning(self.curr_dist, relation_ins, step=j)
            self.dist_history.append(self.curr_dist)
            qs = []

            """
            Instruction Updates
            """
            for j in range(self.num_ins):
                reform = getattr(self, 'reform' + str(j))
                q = reform(self.instruction.instructions[j].squeeze(1), global_rep, query_entities, local_entity)
                qs.append(q.unsqueeze(1))
                self.instruction.instructions[j] = q.unsqueeze(1)
        
        
        """
        Answer Predictions
        """
        pred_dist = self.dist_history[-1]
        # Handle degenerate cases
        question_exists = (query_entities.sum(dim=-1, keepdim=True) > 0).float()
        answer_exists = (answer_dist.sum(dim=-1, keepdim=True) > 0).float()
        case_valid = answer_exists * question_exists
        loss = torch.tensor([0.0])
        recall = 0.0
        bsz, num_cands = pred_dist.shape
        sorted_indices = pred_dist.sort(dim=-1).indices
        correct_idx = answer_dist.argmax(dim=-1)
        #text_batch["cand"][np.arange(bsz), correct_idx.cpu().numpy()] = [entity[0] for entity in text_batch["a_entity"]]
        candidates = question_dict["cand"]
        top_indices = sorted_indices[:, -top_k:]
        top_cands = candidates[np.arange(bsz)[:, None], top_indices.cpu().numpy()]
        if training:
            if replug and question_dict:
                with torch.no_grad():
                    llm_likelihood = torch.zeros((bsz, num_cands)).to(self.device)
                    ppl_files_exist = all([os.path.exists(ppl_file) for ppl_file in save_ppl_files])
                    if (not overwrite_ppl and len(save_ppl_files) > 0 and ppl_files_exist):
                        for i, ppl_file in enumerate(save_ppl_files):
                            curr_perplexity = torch.load(ppl_file).to(self.device)
                            num_scores = curr_perplexity.size(-1)
                            llm_likelihood[i, :num_scores] = torch.softmax(curr_perplexity * gamma, dim=-1)
                    else:
                        perplexities = []
                        idx = 0
                        #Stop once all entries in batch have no candidates left
                        while idx < num_cands and any([cand[idx] != "" for cand in candidates]):
                            question_dict["cand"] = candidates[:, idx : min((idx + top_k, num_cands))]
                            all_input, all_input_list = self.input_builder.process_input_batch(text_batch)
                            # Only compute when case_valid
                            all_input_list = [inp for i, inp in enumerate(all_input_list) if case_valid[i].item()]
                            curr_perplexity = torch.zeros(question_dict["cand"].shape).to(self.device)
                            if len(all_input_list) > 0:
                                curr_perplexity_valid = self.llm_model.calculate_perplexity(all_input_list, text_batch["answer"])
                                valid_examples = case_valid[:, 0].nonzero(as_tuple=True)[0]
                                curr_perplexity[valid_examples] = curr_perplexity_valid
                            perplexities.append(curr_perplexity)
                            idx += top_k
                        llm_perplexity = torch.cat(perplexities, dim=-1)
                        if len(save_ppl_files) > 0:
                            for i in range(bsz):
                                torch.save(llm_perplexity[i:i+1, :], save_ppl_files[i])
                    # if debug_ppl:
                    #     recall = []
                    #     for i, curr_perplexity in enumerate(llm_perplexity):
                    #         rec = 0
                    #         if case_valid[i].item():
                    #             indices = curr_perplexity.argsort(dim=-1)
                    #             best_ppl_idx = (indices == correct_idx[i:i+1, None]).float()
                    #             rec = indices.size(-1) - best_ppl_idx.argmax(dim=-1).item() - 1
                    #         recall.append(rec)
                loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=llm_likelihood, label_valid=case_valid)
            else:
                loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)

        pred_dist = self.dist_history[-1]
        pred = torch.max(pred_dist, dim=1)[1]
        question_dict["cand"] = top_cands
        correct = [False for i in range(bsz)] #Ignore for train for sake of timing
        if training:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            tp_list = None
            correct = self.evaluate_llm(question_dict)
        return loss, pred, pred_dist, tp_list, correct, recall

    
