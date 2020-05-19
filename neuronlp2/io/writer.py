__author__ = 'max'


class CoNLL03Writer(object):
    def __init__(self, word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet):
        self.__source_file = None
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__chunk_alphabet = chunk_alphabet
        self.__ner_alphabet = ner_alphabet

    def start(self, file_path):
        self.__source_file = open(file_path, 'w')

    def close(self):
        self.__source_file.close()

    def write(self, word, pos, chunk, predictions, targets, lengths):
        batch_size, _ = word.shape
        for i in range(batch_size):
            for j in range(lengths[i]):
                w = self.__word_alphabet.get_instance(word[i, j]).encode('utf-8')
                p = self.__pos_alphabet.get_instance(pos[i, j]).encode('utf-8')
                ch = self.__chunk_alphabet.get_instance(chunk[i, j]).encode('utf-8')
                tgt = self.__ner_alphabet.get_instance(targets[i, j]).encode('utf-8')
                pred = self.__ner_alphabet.get_instance(predictions[i, j]).encode('utf-8')
                self.__source_file.write('%d %s %s %s %s %s\n' % (j + 1, w, p, ch, tgt, pred))
            self.__source_file.write('\n')


class CoNLLXWriter(object):
    def __init__(self, word_alphabet, lemma_alphabet, char_alphabet, pos_alphabet, type_alphabet):
        self.__source_file = None
        self.__word_alphabet = word_alphabet
	self.__lemma_alphabet = lemma_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__type_alphabet = type_alphabet

    def start(self, file_path):
        self.__source_file = open(file_path, 'w')

    def close(self):
        self.__source_file.close()

    def write(self, word, lemma, pos, head, type, lengths, symbolic_root=False, symbolic_end=False):
        batch_size, _ = word.shape
        start = 1 if symbolic_root else 0
        end = 1 if symbolic_end else 0

	debug = False


        for i in range(batch_size):
            for j in range(start, lengths[i] - end):
		w = self.__word_alphabet.get_instance(word[i, j]).encode('utf-8')
		l = self.__lemma_alphabet.get_instance(lemma[i, j]).encode('utf-8')
		p = self.__pos_alphabet.get_instance(pos[i, j]).encode('utf-8')
		if debug: print 'num_heads', len(head[i,j])
		if debug: print 'length', lengths[i] 
		h = ""       
		type_index=0
		for k in range(lengths[i]):
			if debug: print j, k, head[i, j]
			true_head=head[i, j,:lengths[i]]
			true_head2=[]
			
			for y, x in enumerate(true_head):
				if y == len(true_head)-1 and true_head[y]==0:
					true_head2.append(-1)
					break
				if x == j: continue
				if x==0 and true_head[y+1]==0:
					true_head2.append(-1)
				else:
					true_head2.append(x)
			
				
			if debug: print 'TRUEHEAD', true_head
			if debug: print 'TRUEHEAD2', true_head2
			if k in true_head2 and k!=j:
				t = self.__type_alphabet.get_instance(type[i, j, type_index]).encode('utf-8')
				h = h+"\t"+t
				type_index+=1
			else:
				h = h+"\t_"

		if debug: print j,w,p,h
		self.__source_file.write('%d\t%s\t%s\t%s%s\n' % (j, w, l, p, h))
            self.__source_file.write('\n')
	#exit(0)
