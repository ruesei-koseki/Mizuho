import discord

import numpy as np
import Model

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from janome.tokenizer import Tokenizer
import json
import random
import threading
import time


def getResponseSentence(model, sentencies):
    model.H.reset_state()
    for i in range(len(sentencies)):
        if sentencies[i] in avocab:
            wid = avocab[sentencies[i]]
            x_k = model.embedx(Variable(np.array([wid], dtype=np.int32)))
            h = model.H(x_k)
    x_k = model.embedx(Variable(np.array([avocab['<eos>']], dtype=np.int32)))
    h = model.H(x_k)
    wid = np.argmax(F.softmax(model.W(h)).data[0])
    res = id2wd[wid]
    loop = 0
    while (wid != bvocab['<eos>']) and (loop <= 30):
        x_k = model.embedy(Variable(np.array([wid], dtype=np.int32)))
        h = model.H(x_k)
        wid = np.argmax(F.softmax(model.W(h)).data[0])
        if wid in id2wd:
            res += id2wd[wid] 
        loop += 1
    return res




t_wakati = Tokenizer(wakati=True)

alines, blines, avocab, av, bvocab, bv, id2wd = np.load("data.npy", allow_pickle=True)

demb = 100
model = Model.ConversationModel(av, bv, avocab, bvocab, demb)
serializers.load_npz('data.model', model)
optimizer = optimizers.Adam()
optimizer.setup(model)


#ここからメッセージ取得&返信

#
#
#以下、discord処理
#
#

client = discord.Client()



@client.event
async def on_ready():
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')


lastRes = "やあ"
channel = False
member = []


@client.event

async def on_message(message):
    global lastRes, model, alines, blines, avocab, av, bvocab, bv, id2wd, channel, member
    if message.attachments:
        pass
    if "はてなにゃん" not in message.guild.name and "レイト" not in message.guild.name and "AI" not in message.guild.name and "人工知能" not in message.guild.name and  "ai" not in message.guild.name and ("みずほ" not in message.channel.name and "ai" not in message.channel.name and "AI" not in message.channel.name and "人工知能" not in message.channel.name):
        pass

    elif client.user != message.author:
        if "みずほ" in message.content or message.channel == channel:
            if message.channel != channel:
                member = []
                channel = message.channel
            text = message.content
            if message.author.name not in member:
                member.append(message.author.name)



            for epoch in range(5):
                isChanged = False

                model.H.reset_state()
                model.cleargrads()
                loss = model(lastRes, message.content)
                loss.backward()
                loss.unchain_backward()
                optimizer.update()
                print(epoch, " epoch finished.")

            serializers.save_npz('data.model', model)

            lastRes = "{} {} ".format(message.author.name, message.content)

            if len(member) == 0:
                res = getResponseSentence(model, "{} {} ".format(message.author.name, text))
                lastRes = "{} {} ".format("みずほ", message.content)
                await message.channel.send(res)
            elif random.random() <= 1 / len(member) or "みずほ" in message.content:
                res = getResponseSentence(model, "{} {} ".format(message.author.name, text))
                lastRes = "{} {} ".format("みずほ", message.content)
                await message.channel.send(res)



def cron():
    time.sleep(60)
    member = []
    if random.random() <= 1 / len(member):
        res = getResponseSentence(model, lastRes)
        lastRes = "{} {} ".format("みずほ", res)
        channel.send(res)




client.run('Token')
thread = threading.Thread(target=cron)
thread.start()
