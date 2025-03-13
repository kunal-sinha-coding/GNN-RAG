
python3 main.py ReaRev --entity_dim 50 --num_epoch 200 --batch_size 8 --eval_every 1 --lm relbert --num_iter 3 --num_ins 3 --num_gnn 4 --name cwq --experiment_name prn_cwq-rearev-lmsr --data_folder data/finkg/ #--warmup_epoch 80
###ReaRev+SBERT training
# --load_experiment relbert-full_cwq-rearev-final.ckpt
#python3 main.py ReaRev --entity_dim 50 --num_epoch 200 --batch_size 16 --eval_every 1  \
# --lm relbert --num_iter 2 --num_ins 3 --num_gnn 3  --name cwq \
# --experiment_name prn_cwq-rearev-sbert --data_folder data/CWQ/ \
# --num_epoch 100 --warmup_epoch 80

# # python main.py ReaRev --is_eval --load_experiment relbert-full_cwq-rearev-final.ckpt --entity_dim 50 --num_epoch 200 --batch_size 8 --eval_every 2  \# --lm relbert --num_iter 2 --num_ins 3 --num_gnn 3  --name cwq \# --experiment_name prn_cwq-rearev-sbert --data_folder data/CWQ/ --num_epoch 100 --warmup_epoch 80

###ReaRev+LMSR training
#python3 main.py ReaRev  --entity_dim 50 --num_epoch 200 --batch_size 8 --eval_every 2  \
# --lm relbert --num_iter 2 --num_ins 3 --num_gnn 3  --name cwq \
# --experiment_name prn_cwq-rearev-lmsr  --data_folder data/CWQ/ --num_epoch 100 #--warmup_epoch 80


###Evaluate CWQ
#python3 main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2 --data_folder data/CWQ/ --lm sbert --num_iter 2 --num_ins 3 --num_gnn 3 --relation_word_emb True --load_experiment ReaRev_CWQ.ckpt --is_eval --name cwq
