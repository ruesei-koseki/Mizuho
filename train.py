import numpy as np

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import Model

alines, blines, avocab, av, bvocab, bv, id2wd = np.load("data.npy", allow_pickle=True)

demb = 100
model = Model.ConversationModel(av, bv, avocab, bvocab, demb)
optimizer = optimizers.Adam()
optimizer.setup(model)
#serializers.load_npz('data.model', model)

for epoch in range(100):
    for i in range(len(alines)):
        aln = alines[i]
        alnr = aln[::-1]
        bln = blines[i]
        model.H.reset_state()
        model.cleargrads()
        loss = model(np.asarray(alnr), np.asarray(bln))
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
    print(epoch, " epoch finished.")
    serializers.save_npz('data.model', model)


serializers.save_npz('data.model', model)