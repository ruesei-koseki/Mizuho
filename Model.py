import numpy as np
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
 
class ConversationModel(chainer.Chain):
    avocab = {}
    bvocab = {}
 
    def __init__(self, av, bv, avo, bvo, k):
        super(ConversationModel, self).__init__(
            embedx = L.EmbedID(av, k),
            embedy = L.EmbedID(bv, k),
            H = L.LSTM(k, k),
            W = L.Linear(k, bv),
        )
        self.avocab = avo
        self.bvocab = bvo
 
    def __call__(self, aline, bline):
        for i in range(len(aline)):
            try:
                wid = self.avocab[aline[i]]
            except:
                wid = self.avocab[" "]
            x_k = self.embedx(Variable(np.array([wid], dtype=np.int32)))
            h = self.H(x_k)
        x_k = self.embedx(Variable(np.array([self.avocab['<eos>']], dtype=np.int32)))
        try:
            tx = Variable(np.array([self.bvocab[bline[0]]], dtype=np.int32))
        except:
            tx = Variable(np.array([self.bvocab[" "]], dtype=np.int32))
        h = self.H(x_k)
        accum_loss = F.softmax_cross_entropy(self.W(h), tx)
        for i in range(len(bline)):
            try:
                wid = self.bvocab[bline[i]]
            except:
                wid = self.bvocab[" "]
            x_k = self.embedy(Variable(np.array([wid], dtype=np.int32)))
            try:
                next_wid = self.bvocab['<eos>']  if (i == len(bline) - 1) else self.bvocab[bline[i+1]]
            except:
                next_wid = self.bvocab['<eos>']  if (i == len(bline) - 1) else self.bvocab[" "]
            tx = Variable(np.array([next_wid], dtype=np.int32))
            h = self.H(x_k)
            loss = F.softmax_cross_entropy(self.W(h), tx)
            accum_loss += loss
        return accum_loss
