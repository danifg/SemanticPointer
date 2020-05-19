import sys



if __name__ == "__main__":
	source_file = open(sys.argv[1], 'r')
	target_file = open(sys.argv[2], 'w')
	debug=False
        num_sents=1
        target_file.write('#SDP 2015\n')
	while True:
		line = source_file.readline()
		if len(line) > 0 and line[0] == '#': continue
		#print line
		# skip multiple blank lines.
		while len(line) > 0 and len(line.strip()) == 0:
		    line = source_file.readline()
		if len(line) == 0:
		    #print line	
		    break

		lines = []
		while len(line.strip()) > 0:
		    line = line.strip()
		    line = line.decode('utf-8')
		    lines.append(line.split('\t'))
		    line = source_file.readline()

		length = len(lines)
		if length == 0:
		    #print line	
		    break
		#print length	

		predicates = []
                tops = []
		for tokens in lines:
                        #print tokens
                        for i in range(len(lines)):
                                if tokens[4+i] != '_' and tokens[4+i] != '_<PAD>':
                                        if i==0:
                                                tops.append(int(tokens[0]))
                                                continue
				        predicates.append(i)
				#print 'pred', tokens[0]

                target_file.write('#2200000%s\n' % (num_sents))
                num_sents+=1
		for tokens in lines:
			if debug: print tokens[0], '\t', tokens[1].encode('utf-8'), '\t', tokens[2].encode('utf-8'), '\t', tokens[3],

			target_file.write('%s\t%s\t%s\t%s' % (tokens[0].encode('utf-8'), tokens[1].encode('utf-8'), tokens[2].encode('utf-8'), tokens[3].encode('utf-8')))

			if int(tokens[0]) in tops:
				if debug: print '\t', 'ROOT',
				target_file.write('\t+')
			else:
				if debug: print '\t', '_', 
				target_file.write('\t-')

                        if int(tokens[0]) in predicates:
                                target_file.write('\t+\tNAMED')
                        else:
                                target_file.write('\t-\tNAMED')
                                
			heads=[]
			types=[]
							
			
			for i in range(len(lines)):

					
				if i in predicates:
                                        types.append(tokens[i+4])
				



			for t in types:
				
				if debug: print '\t', t,
				target_file.write('\t%s' % (t.encode('utf-8')))
			if debug: print
			target_file.write('\n')




		if debug: print
		target_file.write('\n')
	target_file.close()


	

			
	
		
