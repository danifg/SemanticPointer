import sys



if __name__ == "__main__":
	source_file = open(sys.argv[1], 'r')
	target_file = open(sys.argv[2], 'w')
	debug=False
	while True:
		line = source_file.readline()
		if len(line) > 0 and line[0] == '#': continue
		while len(line) > 0 and len(line.strip()) == 0:
		    line = source_file.readline()
		if len(line) == 0:
		    break

		lines = []
		while len(line.strip()) > 0:
		    line = line.strip()
		    line = line.decode('utf-8')
		    lines.append(line.split('\t'))
		    line = source_file.readline()

		length = len(lines)
		if length == 0:
		    break

		predicates = []	
		for tokens in lines:
			if tokens[5] == '+': 
				predicates.append(int(tokens[0]))


		for tokens in lines:
			if debug: print tokens[0], '\t', tokens[1].encode('utf-8'), '\t', tokens[2].encode('utf-8'), '\t', tokens[3],

			target_file.write('%s\t%s\t%s\t%s' % (tokens[0].encode('utf-8'), tokens[1].encode('utf-8'), tokens[2].encode('utf-8'), tokens[3].encode('utf-8')))

			if tokens[4] == '+':
				if debug: print '\t', 'ROOT',
				target_file.write('\tROOT')
			else:
				if debug: print '\t', '_', 
				target_file.write('\t_')

			heads=[]
			types=[]
			for i in range(length):
				#print i
				types.append('_')
							
			
			for i in range(len(predicates)):

				if tokens[i+7]!='_':
					head=0
					if int(tokens[0])==predicates[i]: 
						head=predicates[i+1]
					else:	
						head=predicates[i]
					types[head-1]=tokens[i+7]



			for t in types:
				
				if debug: print '\t', t,
				target_file.write('\t%s' % (t.encode('utf-8')))
			if debug: print
			target_file.write('\n')




		if debug: print
		target_file.write('\n')
	target_file.close()


	

			
	
		
