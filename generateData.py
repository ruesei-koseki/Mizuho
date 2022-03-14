import numpy as np
from janome.tokenizer import Tokenizer

t_wakati = Tokenizer(wakati=True)

id2wd = {}
avocab = {}
bvocab = {}
alines = []
blines = []

exception = []

i = 0
lines = open('conversation_train.csv', 'r', encoding='utf8').read().split('\n')
for line in range(len(lines)):
    li = lines[line].split(',')
    if len(li) == 2:
        #aline = list(t_wakati.tokenize(li[0]))
        aline = list(li[0])
        alines.append(aline)
        #bline = list(t_wakati.tokenize(li[1]))
        bline = list(li[1])
        blines.append(bline)

        if li[0] == "ALL_WORDS":
            exception.append(i)
        if li[1] == "ALL_WORDS":
            exception.append(i)
        i += 1


        for w in aline:
            if w not in avocab:
                avocab[w] = len(avocab)

        for w in bline:
            if w not in bvocab:
                val = len(bvocab)
                id2wd[val] = w
                bvocab[w] = val

i = 0
for e in exception:
    print(e)
    del alines[e-i]
    del blines[e-i]
    i += 1

print(alines)
print(blines)


avocab['<eos>'] = len(avocab)
av = len(avocab)

bvocab['<eos>'] = len(bvocab)
bv = len(bvocab)

id2wd[val] = '<eos>'

data = (alines, blines, avocab, av, bvocab, bv, id2wd)
np.save("data.npy", data)
