
from utils import create_logger
import time
import numpy as np
import os, math
import wandb
import argparse

import torch
from torch.optim.lr_scheduler import ExponentialLR
import torch.optim as optim

from tqdm import tqdm
tqdm.monitor_iterval = 0
import llm.src.utils.utils as llm_utils

#from dataset_load_paths import load_data
from dataset_load import load_data
from dataset_load_graft import load_data_graft
from models.ReaRev.rearev import ReaRev
from models.NSM.nsm import NSM
from models.GraftNet.graftnet import GraftNet
from evaluate import Evaluator
import datasets

class Trainer_KBQA(object):
    def __init__(self, args, model_name, logger=None):
        #print('Trainer here')
        print("Before wandb init")
        wandb.init(project="gnn-rag")
        print("After wandb init")
        self.args = args
        self.logger = logger
        self.best_dev_performance = 0.0
        self.best_h1 = 0.0
        self.best_f1 = 0.0
        self.best_h1b = 0.0
        self.best_f1b = 0.0
        self.eps = args['eps']
        self.warmup_epoch = args['warmup_epoch']
        self.learning_rate = self.args['lr']
        self.test_batch_size = args['test_batch_size']
        self.data_folder = args['data_folder']
        self.data_name = os.path.split(self.data_folder)[-1]
        self.device = torch.device('cuda' if args['use_cuda'] else 'cpu')
        self.reset_time = 0
        self.load_data(args, args['lm'])
        self.entities_names, self.names_entities = llm_utils.get_entities_names()
        self.text_data_name = "kunal-sinha-coding/graphrag-random-graph" #"rmanluo/RoG-cwq"
        self.train_text_data = datasets.load_dataset(self.text_data_name, split="train")
        self.valid_text_data = datasets.load_dataset(self.text_data_name, split="validation")
        self.test_text_data = datasets.load_dataset(self.text_data_name, split="test")
        self.train_data_start, self.train_data_end = args["train_data_start"], args["train_data_end"]

        if 'decay_rate' in args:
            self.decay_rate = args['decay_rate']
        else:
            self.decay_rate = 0.98

        if model_name == 'ReaRev':
            self.model = ReaRev(self.args,  len(self.entity2id), self.num_kb_relation,
                                  self.num_word)
        elif model_name == 'NSM':
            self.model = NSM(self.args,  len(self.entity2id), self.num_kb_relation,
                                  self.num_word)
        elif model_name == 'GraftNet':
            self.model = GraftNet(self.args,  len(self.entity2id), self.num_kb_relation,
                                  self.num_word)
        elif model_name == 'NuTrea':
            self.model = NuTrea(self.args,  len(self.entity2id), self.num_kb_relation,
                                  self.num_word)
        
        if args['relation_word_emb']:
            #self.model.use_rel_texts(self.rel_texts, self.rel_texts_inv)
            self.model.encode_rel_texts(self.rel_texts, self.rel_texts_inv)

        self.model.to(self.device)
        self.evaluator = Evaluator(args=args, model=self.model, entity2id=self.entity2id,
                                       relation2id=self.relation2id, device=self.device)
        self.load_pretrain()
        self.optim_def()
        
        self.num_relation =  self.num_kb_relation
        self.num_entity = len(self.entity2id)
        self.num_word = len(self.word2id)
                                  

        print("Entity: {}, Relation: {}, Word: {}".format(self.num_entity, self.num_relation, self.num_word))

        for k, v in args.items():
            if k.endswith('dim'):
                setattr(self, k, v)
            if k.endswith('emb_file') or k.endswith('kge_file'):
                if v is None:
                    setattr(self, k, None)
                else:
                    setattr(self, k, self.data_folder + v)

    def optim_def(self):
        
        trainable = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optim_model = optim.Adam(trainable, lr=self.learning_rate)
        if self.decay_rate > 0:
            self.scheduler = ExponentialLR(self.optim_model, self.decay_rate)

    def load_data(self, args, tokenize):
        if args["model_name"] == "GraftNet":
            dataset = load_data_graft(args, tokenize)
        else:
            dataset = load_data(args, tokenize)
        self.train_data = dataset["train"]
        self.valid_data = dataset["valid"]
        self.test_data = dataset["test"]
        self.entity2id = dataset["entity2id"]
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.relation2id = dataset["relation2id"]
        self.word2id = dataset["word2id"]
        self.num_word = dataset["num_word"]
        self.num_kb_relation = self.test_data.num_kb_relation
        self.num_entity = len(self.entity2id)
        self.rel_texts = dataset["rel_texts"]
        self.rel_texts_inv = dataset["rel_texts_inv"]

    def load_pretrain(self):
        args = self.args
        if args['load_experiment'] is not None:
            ckpt_path = os.path.join(args['checkpoint_dir'], args['load_experiment'])
            print("Load ckpt from", ckpt_path)
            self.load_ckpt(ckpt_path)

    def evaluate(self, data, test_batch_size=1, valid_text_data=None, write_info=False):
        return self.evaluator.evaluate(data, test_batch_size, valid_text_data, write_info)

    def train(self, start_epoch, end_epoch):
        # self.load_pretrain()
        eval_every = self.args['eval_every']
        # eval_acc = inference(self.model, self.valid_data, self.entity2id, self.args)
        # self.evaluate(self.test_data, self.test_batch_size)
        print("Start Training------------------")
        for epoch in range(start_epoch, end_epoch + 1):
            st = time.time()
            loss, extras, h1_list_all, f1_list_all, accuracy, recall_all = self.train_epoch()

            if self.decay_rate > 0:
                self.scheduler.step()
            
            wandb.log({"Train loss": loss.item()})
            wandb.log({"Train h1": np.mean(h1_list_all)})
            wandb.log({"Train f1": np.mean(f1_list_all)})
            wandb.log({"Train acc": accuracy})
            wandb.log({"Train recall@1": (recall_all <= 0).mean()})
            wandb.log({"Train recall@2": (recall_all <= 1).mean()})
            wandb.log({"Train recall@3": (recall_all <= 2).mean()})
            wandb.log({"Train recall@4": (recall_all <= 3).mean()})
            self.logger.info("Epoch: {}, loss : {:.4f}, time: {}".format(epoch + 1, loss, time.time() - st))
            self.logger.info("Training h1 : {:.4f}, f1 : {:.4f}".format(np.mean(h1_list_all), np.mean(f1_list_all)))
            
            if (epoch + 1) % eval_every == 0:
                eval_f1, eval_h1, eval_em, eval_acc = self.evaluate(self.valid_data, self.test_batch_size, self.valid_text_data)
                wandb.log({"Val f1": eval_f1})
                wandb.log({"Val h1": eval_h1})
                wandb.log({"Val em": eval_em})
                wandb.log({"Val acc": eval_acc})
                self.logger.info("EVAL F1: {:.4f}, H1: {:.4f}, EM {:.4f}".format(eval_f1, eval_h1, eval_em))
                # eval_f1, eval_h1 = self.evaluate(self.test_data, self.test_batch_size)
                # self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))
                do_test = False

                if epoch > self.warmup_epoch:
                    if eval_h1 > self.best_h1:
                        self.best_h1 = eval_h1
                        self.save_ckpt("h1")
                        self.logger.info("BEST EVAL H1: {:.4f}".format(eval_h1))
                        do_test = True
                    if eval_f1 > self.best_f1:
                        self.best_f1 = eval_f1
                        self.save_ckpt("f1")
                        self.logger.info("BEST EVAL F1: {:.4f}".format(eval_f1))
                        do_test = True

                eval_f1, eval_h1, eval_em, eval_acc = self.evaluate(self.test_data, self.test_batch_size, self.test_text_data)
                wandb.log({"Test f1": eval_f1})
                wandb.log({"Test h1": eval_h1})
                wandb.log({"Test em": eval_em})
                wandb.log({"Test acc": eval_acc})
                self.logger.info("TEST F1: {:.4f}, H1: {:.4f}, EM {:.4f}".format(eval_f1, eval_h1, eval_em))
                # if do_test:
                #     eval_f1, eval_h1 = self.evaluate(self.test_data, self.test_batch_size)
                #     self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))
                
                # if eval_h1 > self.best_h1:
                #     self.best_h1 = eval_h1
                #     self.save_ckpt("h1")
                # if eval_f1 > self.best_f1:
                #     self.best_f1 = eval_f1
                #     self.save_ckpt("f1")
                # self.reset_time = 0
                # else:
                #     self.logger.info('No improvement after one evaluation iter.')
                #     self.reset_time += 1
                # if self.reset_time >= 5:
                #     self.logger.info('No improvement after 5 evaluation. Early Stopping.')
                #     break
        self.save_ckpt("final")
        self.logger.info('Train Done! Evaluate on testset with saved model')
        print("End Training------------------")
        self.evaluate_best()

    def evaluate_best(self):
        filename = os.path.join(self.args['checkpoint_dir'], "{}-h1.ckpt".format(self.args['experiment_name']))
        self.load_ckpt(filename)
        eval_f1, eval_h1, eval_em, eval_acc = self.evaluate(self.test_data, self.test_batch_size, write_info=False)
        self.logger.info("Best h1 evaluation")
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}, EM {:.4f}".format(eval_f1, eval_h1, eval_em))

        filename = os.path.join(self.args['checkpoint_dir'], "{}-f1.ckpt".format(self.args['experiment_name']))
        self.load_ckpt(filename)
        eval_f1, eval_h1, eval_em, eval_acc = self.evaluate(self.test_data, self.test_batch_size,  write_info=False)
        self.logger.info("Best f1 evaluation")
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}, EM {:.4f}".format(eval_f1, eval_h1, eval_em))

        filename = os.path.join(self.args['checkpoint_dir'], "{}-final.ckpt".format(self.args['experiment_name']))
        self.load_ckpt(filename)
        eval_f1, eval_h1, eval_em = self.evaluate(self.test_data, self.test_batch_size, write_info=False)
        self.logger.info("Final evaluation")
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}, EM {:.4f}".format(eval_f1, eval_h1, eval_em))

    def evaluate_single(self, filename):
        if filename is not None:
            self.load_ckpt(filename)
        eval_f1, eval_hits, eval_ems, eval_acc = self.evaluate(self.valid_data, self.test_batch_size, write_info=False)
        self.logger.info("EVAL F1: {:.4f}, H1: {:.4f}, EM {:.4f}".format(eval_f1, eval_hits, eval_ems))
        test_f1, test_hits, test_ems, test_acc = self.evaluate(self.test_data, self.test_batch_size, write_info=True)
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}, EM {:.4f}".format(test_f1, test_hits, test_ems))

    def train_epoch(self):
        self.model.train()
        self.train_data.reset_batches(is_sequential=True)
        losses = []
        actor_losses = []
        ent_losses = []
        num_epoch = self.train_data.num_data // self.args['batch_size']
        h1_list_all = []
        f1_list_all = []
        correct_all = []
        recall_all = []
        for iteration in tqdm(range(num_epoch)):
            if (iteration < self.train_data_start * num_epoch 
                or iteration > self.train_data_end * num_epoch):
                continue
            self.optim_model.zero_grad()
            batch = self.train_data.get_batch(iteration, self.args['batch_size'], self.args['fact_drop'])
            candidates = self.train_data.get_candidates(batch)
            save_ppl_files = [
                os.path.join("perplexity_scores", self.data_name, f"{idx}-{idx+1}.pt") 
                for idx in range(start_idx, end_idx)
            ]
            import pdb; pdb.set_trace()
            loss, _, _, tp_list, correct, recall = self.model(batch, candidates training=True, save_ppl_files=save_ppl_files)
            # if tp_list is not None:
            h1_list, f1_list = tp_list
            h1_list_all.extend(h1_list)
            f1_list_all.extend(f1_list)
            correct_all.extend(correct)
            recall_all.append(recall)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([param for name, param in self.model.named_parameters()],
                                           self.args['gradient_clip'])
            self.optim_model.step()
            losses.append(loss.item())
            wandb.log({"Train epoch loss": loss.item()})
            wandb.log({"Train epoch num correct": np.sum(correct_all)})
            wandb.log({"Train epoch recall": recall})
        extras = [0, 0]
        return np.mean(losses), extras, h1_list_all, f1_list_all, np.mean(correct_all), np.array(recall_all)

    
    def save_ckpt(self, reason="h1"):
        model = self.model
        checkpoint = {
            'model_state_dict': model.state_dict()
        }
        model_name = os.path.join(self.args['checkpoint_dir'], "{}-{}.ckpt".format(self.args['experiment_name'],
                                                                                   reason))
        torch.save(checkpoint, model_name)
        print("Best %s, save model as %s" %(reason, model_name))

    def load_ckpt(self, filename):
        checkpoint = torch.load(filename)
        model_state_dict = checkpoint["model_state_dict"]

        model = self.model
        #self.logger.info("Load param of {} from {}.".format(", ".join(list(model_state_dict.keys())), filename))
        model.load_state_dict(model_state_dict, strict=False)

