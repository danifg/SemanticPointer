#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=2 
python scripts/L2RParser.py --mode FastLSTM --num_epochs 1000 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
 --pos_dim 100 --char_dim 100 --lemma_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
 --opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
 --schedule 20 --double_schedule_decay 5 \
 --p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char \
 --word_embedding sskip --word_path "embs/sskip.en.gz" --char_embedding random \
  --train "data/en_$2_train.dag" \
   --dev "data/en_$2_dev.dag" \
    --test "data/en_$2_test.id.dag" \
    --test2 "data/en_$2_test.ood.dag" \
   --model_path $1 --model_name 'models/network.pt'    --grandPar --lemma --beam 5 



