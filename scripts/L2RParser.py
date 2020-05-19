from __future__ import print_function

__author__ = 'max'


import sys
import os

sys.path.append(".")
sys.path.append("..")

sys.setrecursionlimit(2000)

import time
import argparse
import uuid
import json

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm
from torch.optim import Adam, SGD, Adamax
from neuronlp2.io import get_logger, conllx_stacked_data
from neuronlp2.models import NewStackPtrNet
from neuronlp2 import utils
from neuronlp2.io import CoNLLXWriter
from neuronlp2.tasks import parser

uid = uuid.uuid4().hex[:6]


def main():
    args_parser = argparse.ArgumentParser(description='Tuning with stack pointer parser')
    args_parser.add_argument('--mode', choices=['RNN', 'LSTM', 'GRU', 'FastLSTM'], help='architecture of rnn', required=True)
    args_parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    args_parser.add_argument('--batch_size', type=int, default=64, help='Number of sentences in each batch')
    args_parser.add_argument('--decoder_input_size', type=int, default=256, help='Number of input units in decoder RNN.')
    args_parser.add_argument('--hidden_size', type=int, default=256, help='Number of hidden units in RNN')
    args_parser.add_argument('--arc_space', type=int, default=128, help='Dimension of tag space')
    args_parser.add_argument('--type_space', type=int, default=128, help='Dimension of tag space')
    args_parser.add_argument('--encoder_layers', type=int, default=1, help='Number of layers of encoder RNN')
    args_parser.add_argument('--decoder_layers', type=int, default=1, help='Number of layers of decoder RNN')
    args_parser.add_argument('--num_filters', type=int, default=50, help='Number of filters in CNN')
    args_parser.add_argument('--pos', action='store_true', help='use part-of-speech embedding.')
    args_parser.add_argument('--char', action='store_true', help='use character embedding and CNN.')
    args_parser.add_argument('--lemma', action='store_true', help='use lemma embedding.')
    args_parser.add_argument('--pos_dim', type=int, default=50, help='Dimension of POS embeddings')
    args_parser.add_argument('--char_dim', type=int, default=50, help='Dimension of Character embeddings')
    args_parser.add_argument('--lemma_dim', type=int, default=50, help='Dimension of Lemma embeddings')
    args_parser.add_argument('--opt', choices=['adam', 'sgd', 'adamax'], help='optimization algorithm')
    args_parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    args_parser.add_argument('--decay_rate', type=float, default=0.75, help='Decay rate of learning rate')
    args_parser.add_argument('--max_decay', type=int, default=9, help='Number of decays before stop')
    args_parser.add_argument('--double_schedule_decay', type=int, default=5, help='Number of decays to double schedule')
    args_parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    args_parser.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
    args_parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon for adam or adamax')
    args_parser.add_argument('--coverage', type=float, default=0.0, help='weight for coverage loss')
    args_parser.add_argument('--p_rnn', nargs=2, type=float, required=True, help='dropout rate for RNN')
    args_parser.add_argument('--p_in', type=float, default=0.33, help='dropout rate for input embeddings')
    args_parser.add_argument('--p_out', type=float, default=0.33, help='dropout rate for output layer')
    args_parser.add_argument('--label_smooth', type=float, default=1.0, help='weight of label smoothing method')
    args_parser.add_argument('--skipConnect', action='store_true', help='use skip connection for decoder RNN.')
    args_parser.add_argument('--grandPar', action='store_true', help='use grand parent.')
    args_parser.add_argument('--sibling', action='store_true', help='use sibling.')
    args_parser.add_argument('--prior_order', choices=['inside_out', 'left2right', 'deep_first', 'shallow_first'], help='prior order of children.', required=False)
    args_parser.add_argument('--schedule', type=int, help='schedule for learning rate decay')
    args_parser.add_argument('--unk_replace', type=float, default=0., help='The rate to replace a singleton word with UNK')
    args_parser.add_argument('--punctuation', nargs='+', type=str, help='List of punctuations')
    args_parser.add_argument('--beam', type=int, default=1, help='Beam size for decoding')
    args_parser.add_argument('--word_embedding', choices=['glove', 'senna', 'sskip', 'polyglot'], help='Embedding for words', required=True)
    args_parser.add_argument('--word_path', help='path for word embedding dict')
    args_parser.add_argument('--freeze', action='store_true', help='frozen the word embedding (disable fine-tuning).')
    args_parser.add_argument('--char_embedding', choices=['random', 'polyglot'], help='Embedding for characters', required=True)
    args_parser.add_argument('--char_path', help='path for character embedding dict')
    args_parser.add_argument('--train')  # "data/POS-penn/wsj/split1/wsj1.train.original"
    args_parser.add_argument('--dev')  # "data/POS-penn/wsj/split1/wsj1.dev.original"
    args_parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"
    args_parser.add_argument('--test2')
    args_parser.add_argument('--model_path', help='path for saving model file.', required=True)
    args_parser.add_argument('--model_name', help='name for saving model file.', required=True)

    args = args_parser.parse_args()

    logger = get_logger("PtrParser")
    print('SEMANTIC DEPENDENCY PARSER with POINTER NETWORKS')	
    print('CUDA?', torch.cuda.is_available())

    mode = args.mode
    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    test_path2 = args.test2
    model_path = args.model_path
    model_name = args.model_name
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    input_size_decoder = args.decoder_input_size
    hidden_size = args.hidden_size
    arc_space = args.arc_space
    type_space = args.type_space
    encoder_layers = args.encoder_layers
    decoder_layers = args.decoder_layers
    num_filters = args.num_filters
    learning_rate = args.learning_rate
    opt = args.opt
    momentum = 0.9
    betas = (0.9, 0.9)
    eps = args.epsilon
    decay_rate = args.decay_rate
    clip = args.clip
    gamma = args.gamma
    cov = args.coverage
    schedule = args.schedule
    p_rnn = tuple(args.p_rnn)
    p_in = args.p_in
    p_out = args.p_out
    label_smooth = args.label_smooth
    unk_replace = args.unk_replace
    prior_order = args.prior_order
    skipConnect = args.skipConnect
    grandPar = args.grandPar
    sibling = args.sibling
    beam = args.beam
    punctuation = args.punctuation

    freeze = args.freeze
    word_embedding = args.word_embedding
    word_path = args.word_path

    use_char = args.char
    char_embedding = args.char_embedding
    char_path = args.char_path

    use_pos = args.pos
    pos_dim = args.pos_dim

    use_lemma = args.lemma
    lemma_dim = args.lemma_dim

    word_dict, word_dim = utils.load_embedding_dict(word_embedding, word_path)
    char_dict = None
    char_dim = args.char_dim
    if char_embedding != 'random':
        char_dict, char_dim = utils.load_embedding_dict(char_embedding, char_path)

    logger.info("Creating Alphabets")
    alphabet_path = os.path.join(model_path, 'alphabets/')
    model_name = os.path.join(model_path, model_name)
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet, lemma_alphabet = conllx_stacked_data.create_alphabets(alphabet_path, train_path, data_paths=[dev_path, test_path, test_path2],
                                                                                                     max_vocabulary_size=50000, embedd_dict=word_dict)

    num_words = word_alphabet.size()
    num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    num_types = type_alphabet.size()
    num_lemmas = lemma_alphabet.size()

    logger.info("Word Alphabet Size: %d" % num_words)
    logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    logger.info("Type Alphabet Size: %d" % num_types)
    logger.info("LEMMA Alphabet Size: %d" % num_lemmas)

    logger.info("Reading Data")
    use_gpu = torch.cuda.is_available()

    data_train = conllx_stacked_data.read_stacked_data_to_variable(train_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, lemma_alphabet, use_gpu=use_gpu, prior_order=prior_order)
    num_data = sum(data_train[1])

    data_dev = conllx_stacked_data.read_stacked_data_to_variable(dev_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, lemma_alphabet, use_gpu=use_gpu, volatile=True, prior_order=prior_order)
    data_test = conllx_stacked_data.read_stacked_data_to_variable(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, lemma_alphabet, use_gpu=use_gpu, volatile=True, prior_order=prior_order)
    data_test2 = conllx_stacked_data.read_stacked_data_to_variable(test_path2, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, lemma_alphabet, use_gpu=use_gpu, volatile=True, prior_order=prior_order)

    punct_set = None
    if punctuation is not None:
        punct_set = set(punctuation)
        #logger.info("punctuations(%d): %s" % (len(punct_set), ' '.join(punct_set)))

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / word_dim)
        table = np.empty([word_alphabet.size(), word_dim], dtype=np.float32)
        table[conllx_stacked_data.UNK_ID, :] = np.zeros([1, word_dim]).astype(np.float32) if freeze else np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
        oov = 0
        for word, index in word_alphabet.items():
            if word in word_dict:
                embedding = word_dict[word]
            elif word.lower() in word_dict:
                embedding = word_dict[word.lower()]
            else:
                embedding = np.zeros([1, word_dim]).astype(np.float32) if freeze else np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        print('word OOV: %d' % oov)
	print(torch.__version__)
        return torch.from_numpy(table)
    
    def construct_lemma_embedding_table():
        scale = np.sqrt(3.0 / lemma_dim)
        table = np.empty([lemma_alphabet.size(), lemma_dim], dtype=np.float32)
        table[conllx_stacked_data.UNK_ID, :] = np.zeros([1, lemma_dim]).astype(np.float32) if freeze else np.random.uniform(-scale, scale, [1, lemma_dim]).astype(np.float32)
        oov = 0
        for lemma, index in lemma_alphabet.items():
            if lemma in word_dict:
                embedding = word_dict[lemma]
            elif lemma.lower() in word_dict:
                embedding = word_dict[lemma.lower()]
            else:
                embedding = np.zeros([1, lemma_dim]).astype(np.float32) if freeze else np.random.uniform(-scale, scale, [1, lemma_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        print('LEMMA OOV: %d' % oov)
	print(torch.__version__)
        return torch.from_numpy(table)
    

    def construct_char_embedding_table():
        if char_dict is None:
            return None

        scale = np.sqrt(3.0 / char_dim)
        table = np.empty([num_chars, char_dim], dtype=np.float32)
        table[conllx_stacked_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, char_dim]).astype(np.float32)
        oov = 0
        for char, index, in char_alphabet.items():
            if char in char_dict:
                embedding = char_dict[char]
            else:
                embedding = np.random.uniform(-scale, scale, [1, char_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        print('character OOV: %d' % oov)
        return torch.from_numpy(table)

    word_table = construct_word_embedding_table()
    char_table = construct_char_embedding_table()
    lemma_table = construct_lemma_embedding_table() 

    window = 3
    network = NewStackPtrNet(word_dim, num_words, lemma_dim, num_lemmas, char_dim, num_chars, pos_dim, num_pos, num_filters, window,
                          mode, input_size_decoder, hidden_size, encoder_layers, decoder_layers,
                          num_types, arc_space, type_space,
                          embedd_word=word_table, embedd_char=char_table, embedd_lemma=lemma_table, p_in=p_in, p_out=p_out, p_rnn=p_rnn,
                          biaffine=True, pos=use_pos, char=use_char, lemma=use_lemma, prior_order=prior_order,
                          skipConnect=skipConnect, grandPar=grandPar, sibling=sibling)
    def save_args():
        arg_path = model_name + '.arg.json'
        arguments = [word_dim, num_words, lemma_dim, num_lemmas, char_dim, num_chars, pos_dim, num_pos, num_filters, window,
                     mode, input_size_decoder, hidden_size, encoder_layers, decoder_layers,
                     num_types, arc_space, type_space]
        kwargs = {'p_in': p_in, 'p_out': p_out, 'p_rnn': p_rnn, 'biaffine': True, 'pos': use_pos, 'char': use_char, 'lemma': use_lemma, 'prior_order': prior_order,
                  'skipConnect': skipConnect, 'grandPar': grandPar, 'sibling': sibling}
        json.dump({'args': arguments, 'kwargs': kwargs}, open(arg_path, 'w'), indent=4)

    if freeze:
        network.word_embedd.freeze()

    if use_gpu:
	print('CUDA IS AVAILABLE')
        network.cuda()
    else:
	print('CUDA IS NOT AVAILABLE', use_gpu)

    save_args()

    pred_writer = CoNLLXWriter(word_alphabet, lemma_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    gold_writer = CoNLLXWriter(word_alphabet, lemma_alphabet, char_alphabet, pos_alphabet, type_alphabet)

    def generate_optimizer(opt, lr, params):
        params = filter(lambda param: param.requires_grad, params)
        if opt == 'adam':
            return Adam(params, lr=lr, betas=betas, weight_decay=gamma, eps=eps)
        elif opt == 'sgd':
            return SGD(params, lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)
        elif opt == 'adamax':
            return Adamax(params, lr=lr, betas=betas, weight_decay=gamma, eps=eps)
        else:
            raise ValueError('Unknown optimization algorithm: %s' % opt)

    lr = learning_rate
    optim = generate_optimizer(opt, lr, network.parameters())
    opt_info = 'opt: %s, ' % opt
    if opt == 'adam':
        opt_info += 'betas=%s, eps=%.1e' % (betas, eps)
    elif opt == 'sgd':
        opt_info += 'momentum=%.2f' % momentum
    elif opt == 'adamax':
        opt_info += 'betas=%s, eps=%.1e' % (betas, eps)

    word_status = 'frozen' if freeze else 'fine tune'
    char_status = 'enabled' if use_char else 'disabled'
    pos_status = 'enabled' if use_pos else 'disabled'
    lemma_status = 'enabled' if use_lemma else 'disabled'
    logger.info("Embedding dim: word=%d (%s), lemma=%d (%s) char=%d (%s), pos=%d (%s)" % (word_dim, word_status, lemma_dim, lemma_status, char_dim, char_status, pos_dim, pos_status))
    logger.info("CNN: filter=%d, kernel=%d" % (num_filters, window))
    logger.info("RNN: %s, num_layer=(%d, %d), input_dec=%d, hidden=%d, arc_space=%d, type_space=%d" % (mode, encoder_layers, decoder_layers, input_size_decoder, hidden_size, arc_space, type_space))
    logger.info("train: cov: %.1f, (#data: %d, batch: %d, clip: %.2f, label_smooth: %.2f, unk_repl: %.2f)" % (cov, num_data, batch_size, clip, label_smooth, unk_replace))
    logger.info("dropout(in, out, rnn): (%.2f, %.2f, %s)" % (p_in, p_out, p_rnn))
    logger.info('prior order: %s, grand parent: %s, sibling: %s, ' % (prior_order, grandPar, sibling))
    logger.info('skip connect: %s, beam: %d' % (skipConnect, beam))
    logger.info(opt_info)

    num_batches = num_data / batch_size + 1
    #dev_ucorrect = 0.0
	
    dev_bestLF1 = 0.0
    dev_bestUF1 = 0.0
    dev_bestUprecision = 0.0
    dev_bestLprecision = 0.0
    dev_bestUrecall = 0.0
    dev_bestLrecall = 0.0			


    best_epoch = 0

    test_ucorrect = 0.0
    test_lcorrect = 0.0
    #test_ucomlpete_match = 0.0
    #test_lcomplete_match = 0.0

    #test_ucorrect_nopunc = 0.0
    #test_lcorrect_nopunc = 0.0
    #test_ucomlpete_match_nopunc = 0.0
    #test_lcomplete_match_nopunc = 0.0
    #test_root_correct = 0.0
    test_total_pred = 0
    test_total_gold = 0
    #test_total_nopunc = 0
    test_total_inst = 0
    #test_total_root = 0

    test_LF1 = 0.0
    test_UF1 = 0.0
    test_Uprecision = 0.0
    test_Lprecision = 0.0
    test_Urecall = 0.0
    test_Lrecall = 0.0


    test2_ucorrect = 0.0
    test2_lcorrect = 0.0
    test2_total_pred = 0
    test2_total_gold = 0
    test2_total_inst = 0

    test2_LF1 = 0.0
    test2_UF1 = 0.0
    test2_Uprecision = 0.0
    test2_Lprecision = 0.0
    test2_Urecall = 0.0
    test2_Lrecall = 0.0

    patient = 0
    decay = 0
    max_decay = args.max_decay
    double_schedule_decay = args.double_schedule_decay
    for epoch in range(1, num_epochs + 1):
        print('Epoch %d (%s, optim: %s, learning rate=%.6f, eps=%.1e, decay rate=%.2f (schedule=%d, patient=%d, decay=%d (%d, %d))): ' % (
            epoch, mode, opt, lr, eps, decay_rate, schedule, patient, decay, max_decay, double_schedule_decay))

        train_err_cov = 0.
	train_err_arc = 0.
	train_err_type = 0.
	train_total = 0.


        start_time = time.time()
        num_back = 0
        network.train()
        for batch in range(1, num_batches + 1):
	
            input_encoder, input_decoder = conllx_stacked_data.get_batch_stacked_variable(data_train, batch_size, unk_replace=unk_replace)
            word, lemma, char, pos, heads, types, masks_e, lengths_e = input_encoder
            stacked_heads, children, sibling, stacked_types, skip_connect, previous, next, masks_d, lengths_d = input_decoder


		
	    #print('HEADSSS', heads)


            optim.zero_grad()


	    loss_arc, \
            loss_type, \
            loss_cov, num = network.loss(word, lemma, char, pos, heads, stacked_heads, children, sibling, stacked_types, previous, next, label_smooth,
                                                            skip_connect=skip_connect, mask_e=masks_e, length_e=lengths_e, mask_d=masks_d, length_d=lengths_d)


            loss = loss_arc + loss_type + cov * loss_cov
            loss.backward()
            clip_grad_norm(network.parameters(), clip)
            optim.step()

	    train_err_arc += loss_arc.data[0] * num

	    train_err_type += loss_type.data[0] * num

	    train_err_cov += loss_cov.data[0] * num


	    train_total += num

            time_ave = (time.time() - start_time) / batch
            time_left = (num_batches - batch) * time_ave



        sys.stdout.write("\b" * num_back)
        sys.stdout.write(" " * num_back)
        sys.stdout.write("\b" * num_back)
	err_arc = train_err_arc / train_total

	err_type = train_err_type / train_total

	err_cov = train_err_cov / train_total

        err = err_arc + err_type + cov * err_cov
        print('train: %d loss: %.4f, arc: %.4f, type: %.4f, coverage: %.4f, time: %.2fs' % (
            num_batches, err, err_arc, err_type, err_cov, time.time() - start_time))




        print('======EVALUATING PERFORMANCE ON DEV======')
        # evaluate performance on dev data
        network.eval()
        #pred_filename = 'tmp/%spred_dev%d' % (str(uid), epoch)
	pred_filename = '%spred_dev%d' % (str(uid), epoch)
	pred_filename = os.path.join(model_path, pred_filename)
        pred_writer.start(pred_filename)
        #gold_filename = 'tmp/%sgold_dev%d' % (str(uid), epoch)
	gold_filename = '%sgold_dev%d' % (str(uid), epoch)
	gold_filename = os.path.join(model_path, gold_filename)
        gold_writer.start(gold_filename)

        dev_ucorr = 0.0
        dev_lcorr = 0.0
        dev_total_gold = 0
	dev_total_pred = 0
        dev_total_inst = 0.0
	start_time_dev = time.time()
        for batch in conllx_stacked_data.iterate_batch_stacked_variable(data_dev, batch_size):
            input_encoder, _ = batch
            word, lemma, char, pos, heads, types, masks, lengths = input_encoder
            heads_pred, types_pred, _, _ = network.decode(word, lemma, char, pos, mask=masks, length=lengths, beam=beam, leading_symbolic=conllx_stacked_data.NUM_SYMBOLIC_TAGS)

            word = word.data.cpu().numpy()
	    lemma = lemma.data.cpu().numpy()
            pos = pos.data.cpu().numpy()
            lengths = lengths.cpu().numpy()
            heads = heads.data.cpu().numpy()
            types = types.data.cpu().numpy()

            pred_writer.write(word, lemma, pos, heads_pred, types_pred, lengths, symbolic_root=True)
	    gold_writer.write(word, lemma, pos, heads, types, lengths, symbolic_root=True)
	    
	    

            #stats, stats_nopunc, stats_root, num_inst = parser.evalF1(word, pos, heads_pred, types_pred, heads, types, word_alphabet, pos_alphabet, lengths, punct_set=punct_set, symbolic_root=True)
            #ucorr, lcorr, total, ucm, lcm = stats
            #ucorr_nopunc, lcorr_nopunc, total_nopunc, ucm_nopunc, lcm_nopunc = stats_nopunc
            #corr_root, total_root = stats_root
	    ucorr, lcorr, total_gold, total_pred, num_inst = parser.evalF1(word, lemma, pos, heads_pred, types_pred, heads, types, word_alphabet, lemma_alphabet, pos_alphabet, lengths, punct_set=punct_set, symbolic_root=True)

            dev_ucorr += ucorr
            dev_lcorr += lcorr
            dev_total_gold += total_gold
	    dev_total_pred += total_pred


            dev_total_inst += num_inst

	end_time_dev = time.time()
	lasted_time_dev=end_time_dev-start_time_dev
        pred_writer.close()
        gold_writer.close()

	dev_Uprecision=0.
	dev_Lprecision=0.
	if dev_total_pred!=0: 
		dev_Uprecision=dev_ucorr * 100 / dev_total_pred
		dev_Lprecision=dev_lcorr * 100 / dev_total_pred
	dev_Urecall=dev_ucorr * 100 / dev_total_gold
	dev_Lrecall=dev_lcorr * 100 / dev_total_gold
	if dev_Uprecision ==0. and dev_Urecall==0.: 
		dev_UF1=0
	else:
		dev_UF1=2*(dev_Uprecision*dev_Urecall)/(dev_Uprecision+dev_Urecall)
	if dev_Lprecision ==0. and dev_Lrecall==0.: 
		dev_LF1=0
	else:
		dev_LF1=2*(dev_Lprecision*dev_Lrecall)/(dev_Lprecision+dev_Lrecall)

	print('CUR DEV %d: ucorr: %d, lcorr: %d, tot_gold: %d, tot_pred: %d, Uprec: %.2f%%, Urec: %.2f%%, Lprec: %.2f%%, Lrec: %.2f%%, UF1: %.2f%%, LF1: %.2f%%' % (
            epoch, dev_ucorr, dev_lcorr, dev_total_gold, dev_total_pred, dev_Uprecision, dev_Urecall, dev_Lprecision, dev_Lrecall, dev_UF1, dev_LF1))






        #if dev_lcorrect_nopunc < dev_lcorr_nopunc or (dev_lcorrect_nopunc == dev_lcorr_nopunc and dev_ucorrect_nopunc < dev_ucorr_nopunc):
	if dev_bestLF1 < dev_LF1:
            dev_bestLF1 = dev_LF1
    	    dev_bestUF1 = dev_UF1
    	    dev_bestUprecision = dev_Uprecision
    	    dev_bestLprecision = dev_Lprecision
    	    dev_bestUrecall = dev_Urecall
    	    dev_bestLrecall = dev_Lrecall

            best_epoch = epoch
            patient = 0
            # torch.save(network, model_name)
            torch.save(network.state_dict(), model_name)

	    print('======EVALUATING PERFORMANCE ON TEST======')
            #pred_filename = 'tmp/%spred_test%d' % (str(uid), epoch)
	    pred_filename = '%spred_test%d' % (str(uid), epoch)
	    pred_filename = os.path.join(model_path, pred_filename)
            pred_writer.start(pred_filename)
            #gold_filename = 'tmp/%sgold_test%d' % (str(uid), epoch)
	    gold_filename = '%sgold_test%d' % (str(uid), epoch)
	    gold_filename = os.path.join(model_path, gold_filename)
            gold_writer.start(gold_filename)

            test_ucorrect = 0.0
            test_lcorrect = 0.0
            test_total_pred = 0
	    test_total_gold = 0
	    test_total_inst = 0

	    start_time_test = time.time()
            for batch in conllx_stacked_data.iterate_batch_stacked_variable(data_test, batch_size):
                input_encoder, _ = batch
                word, lemma, char, pos, heads, types, masks, lengths = input_encoder
                heads_pred, types_pred, _, _ = network.decode(word, lemma, char, pos, mask=masks, length=lengths, beam=beam, leading_symbolic=conllx_stacked_data.NUM_SYMBOLIC_TAGS)

                word = word.data.cpu().numpy()
		lemma = lemma.data.cpu().numpy()
                pos = pos.data.cpu().numpy()
                lengths = lengths.cpu().numpy()
                heads = heads.data.cpu().numpy()
                types = types.data.cpu().numpy()

		

                pred_writer.write(word, lemma, pos, heads_pred, types_pred, lengths, symbolic_root=True)
                gold_writer.write(word, lemma, pos, heads, types, lengths, symbolic_root=True)

                ucorr, lcorr, total_gold, total_pred, num_inst = parser.evalF1(word, lemma, pos, heads_pred, types_pred, heads, types, word_alphabet, lemma_alphabet, pos_alphabet, lengths, punct_set=punct_set, symbolic_root=True)
                
                test_ucorrect += ucorr
                test_lcorrect += lcorr
                test_total_gold += total_gold
		test_total_pred += total_pred
                
                test_total_inst += num_inst

	    end_time_test = time.time()
	    lasted_time_test=end_time_test-start_time_test
            pred_writer.close()
            gold_writer.close()

	    test_Uprecision=0.
	    test_Lprecision=0.
	    if test_total_pred!=0:
		    test_Uprecision=test_ucorrect * 100 / test_total_pred
		    test_Lprecision=test_lcorrect * 100 / test_total_pred
	    test_Urecall=test_ucorrect * 100 / test_total_gold
	    test_Lrecall=test_lcorrect * 100 / test_total_gold
	    if test_Uprecision ==0. and test_Urecall==0.: 
		test_UF1=0
	    else: 
	    	test_UF1=2*(test_Uprecision*test_Urecall)/(test_Uprecision+test_Urecall)
	    if test_Lprecision ==0. and test_Lrecall==0.: 
		test_LF1=0
	    else:
	    	test_LF1=2*(test_Lprecision*test_Lrecall)/(test_Lprecision+test_Lrecall)


	    print('======EVALUATING PERFORMANCE ON TEST 2======')
            #pred_filename = 'tmp/%spred_test%d' % (str(uid), epoch)
	    pred_filename2 = '%spred_test_two%d' % (str(uid), epoch)
	    pred_filename2 = os.path.join(model_path, pred_filename2)
            pred_writer.start(pred_filename2)
            #gold_filename = 'tmp/%sgold_test%d' % (str(uid), epoch)
	    gold_filename2 = '%sgold_test_two%d' % (str(uid), epoch)
	    gold_filename2 = os.path.join(model_path, gold_filename2)
            gold_writer.start(gold_filename2)

            test2_ucorrect = 0.0
            test2_lcorrect = 0.0
            test2_total_pred = 0
	    test2_total_gold = 0
	    test2_total_inst = 0

	    start_time_test2 = time.time()
            for batch in conllx_stacked_data.iterate_batch_stacked_variable(data_test2, batch_size):
                input_encoder, _ = batch
                word, lemma, char, pos, heads, types, masks, lengths = input_encoder
                heads_pred, types_pred, _, _ = network.decode(word, lemma, char, pos, mask=masks, length=lengths, beam=beam, leading_symbolic=conllx_stacked_data.NUM_SYMBOLIC_TAGS)

                word = word.data.cpu().numpy()
		lemma = lemma.data.cpu().numpy()
                pos = pos.data.cpu().numpy()
                lengths = lengths.cpu().numpy()
                heads = heads.data.cpu().numpy()
                types = types.data.cpu().numpy()

		

                pred_writer.write(word, lemma, pos, heads_pred, types_pred, lengths, symbolic_root=True)
                gold_writer.write(word, lemma, pos, heads, types, lengths, symbolic_root=True)

                ucorr, lcorr, total_gold, total_pred, num_inst = parser.evalF1(word, lemma, pos, heads_pred, types_pred, heads, types, word_alphabet, lemma_alphabet, pos_alphabet, lengths, punct_set=punct_set, symbolic_root=True)
                
                test2_ucorrect += ucorr
                test2_lcorrect += lcorr
                test2_total_gold += total_gold
		test2_total_pred += total_pred
                
                test2_total_inst += num_inst

	    end_time_test2 = time.time()
	    lasted_time_test2=end_time_test2-start_time_test2
            pred_writer.close()
            gold_writer.close()	

	    test2_Uprecision=0.
	    test2_Lprecision=0.
	    if dev_total_pred!=0:
		    test2_Uprecision=test2_ucorrect * 100 / test2_total_pred
		    test2_Lprecision=test2_lcorrect * 100 / test2_total_pred
	    test2_Urecall=test2_ucorrect * 100 / test2_total_gold
	    test2_Lrecall=test2_lcorrect * 100 / test2_total_gold
	    if test2_Uprecision ==0. and test2_Urecall==0.: 
		test2_UF1=0.
	    else: 
		test2_UF1=2*(test2_Uprecision*test2_Urecall)/(test2_Uprecision+test2_Urecall)
	    if test2_Lprecision ==0 and test2_Lrecall==0: 
		test2_LF1=0.
	    else: 
	    	test2_LF1=2*(test2_Lprecision*test2_Lrecall)/(test2_Lprecision+test2_Lrecall)


        else:
            #if dev_ucorr_nopunc * 100 / dev_total_nopunc < dev_ucorrect_nopunc * 100 / dev_total_nopunc - 5 or patient >= schedule:
	    if dev_LF1 < dev_bestLF1 - 5 or patient >= schedule:
                # network = torch.load(model_name)
                network.load_state_dict(torch.load(model_name))
                lr = lr * decay_rate
                optim = generate_optimizer(opt, lr, network.parameters())
                patient = 0
                decay += 1
                if decay % double_schedule_decay == 0:
                    schedule *= 2
            else:
                patient += 1

	
        print('----------------------------------------------------------------------------------------------------------------------------')
	print('TIME DEV: ', lasted_time_dev, 'NUM SENTS DEV: ', dev_total_inst, 'SPEED DEV: ', dev_total_inst/lasted_time_dev)
	print('DEV: Uprec: %.2f%%, Urec: %.2f%%, Lprec: %.2f%%, Lrec: %.2f%%, UF1: %.2f%%, LF1: %.2f%% (epoch: %d)' % (
             dev_bestUprecision, dev_bestUrecall, dev_bestLprecision, dev_bestLrecall, dev_bestUF1, dev_bestLF1, best_epoch))
        print('----------------------------------------------------------------------------------------------------------------------------')
	print('TIME TEST: ', lasted_time_test, 'NUM SENTS TEST: ', test_total_inst, 'SPEED TEST: ', test_total_inst/lasted_time_test)
        print('TEST: ucorr: %d, lcorr: %d, tot_gold: %d, tot_pred: %d, Uprec: %.2f%%, Urec: %.2f%%, Lprec: %.2f%%, Lrec: %.2f%%, UF1: %.2f%%, LF1: %.2f%% (epoch: %d)' % (
            test_ucorrect, test_lcorrect, test_total_gold, test_total_pred, test_Uprecision, test_Urecall, test_Lprecision, test_Lrecall, test_UF1, test_LF1, best_epoch))
	print('----------------------------------------------------------------------------------------------------------------------------')
	print('TIME TEST2: ', lasted_time_test2, 'NUM SENTS TEST: ', test2_total_inst, 'SPEED TEST2: ', test2_total_inst/lasted_time_test2)
        print('TEST2: ucorr: %d, lcorr: %d, tot_gold: %d, tot_pred: %d, Uprec: %.2f%%, Urec: %.2f%%, Lprec: %.2f%%, Lrec: %.2f%%, UF1: %.2f%%, LF1: %.2f%% (epoch: %d)' % (
            test2_ucorrect, test2_lcorrect, test2_total_gold, test2_total_pred, test2_Uprecision, test2_Urecall, test2_Lprecision, test2_Lrecall, test2_UF1, test2_LF1, best_epoch))
        print('============================================================================================================================')

	#exit(0)
        if decay == max_decay:
            break


if __name__ == '__main__':
    main()
