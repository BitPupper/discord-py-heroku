import os
bot_token = os.getenv("bot_token")
import flair, torch
flair.device = torch.device('cpu')
from transformers import pipeline
from flair.data import Sentence
from flair.models import SequenceTagger
from discord.ext import commands
from discord import Client
import random
from tabulate import tabulate

sentiment_task = pipeline("sentiment-analysis")

chunk_tagger = SequenceTagger.load("flair/chunk-english-fast")

client = Client()
bot = commands.Bot(command_prefix='!')
sentiment_thresh = 0.1

bot.lines = {
    "greet":{
        "en":["Hello", "Howdy", "Salutations", "Hey", "Greetings"],
        "jp":["こんにちは","やあ","ヘイ","よっ","よお","おー"]
    },
    "job":{
        "en":["I am on an unpaid internship for living life"],
        "jp":["就職先は"]
    },
    "gossip":{
        "en": ["Ooo! Tell me more!"],
        "jp": ["へえーそうなんだ！", "うそ！マジで？？", "さよか！","ふーん"]
    },
    "dialect":{
        "en":"Here are some fun Japanese dialect converters I found: \n",
        "jp":""
    },
    "stranger":{
        "en": ["i have a receeding hairline", "mmm... tuna eyeballs", "WOWZA!",f":thinking:","so anyway i stubbed my toe today", "that's great", "thank you, very interesting", f"https://tenor.com/view/thonk-thinking-sun-thonk-sun-thinking-sun-gif-14999983", f":moyai:", f"https://tenor.com/view/the-rock-the-rock-sus-the-rock-meme-tthe-rock-sus-meme-dwayne-johnson-gif-23805584",f":jimstare:","you sound like you have a nice personality","brb going to go water my petunias","o shoot Hank made the breaker trip agai"],
        "jp": ["大丈夫？","努力は絶対報われる", "うわすごいね！！","(*＾∀＾*)","へぇ...そうか","なんだよ(笑)","！豆腐ハンバーグ！","そういうの大嫌いだわー",f":thinking:",f"https://tenor.com/view/thonk-thinking-sun-thonk-sun-thinking-sun-gif-14999983",f":moyai:",f"https://tenor.com/view/the-rock-the-rock-sus-the-rock-meme-tthe-rock-sus-meme-dwayne-johnson-gif-23805584",f":jimstare:"]
    },
    "friend":{
        "POSITIVE":{
            "en": ["wow!! im happy for you!!", "dude thats awesome!","holy heck!","whoa rly??","cool!","nice nice","amazing! i wish i had a life like yours", "oooo!"],
            "jp": ["Σ(*・Д・)ﾊｯ"]
        },
        "NEGATIVE":{
            "en": ["aw :c","u feeling ok?","i feel you bud","it's going to be ok!","thanks for sharing with me"],
            "jp": ["Σ(*・Д・)ﾊｯ"]
        },
        "NEUTRAL":{
            "en": ["what does that mean for you?","huh didn't know that!"],
            "jp": ["Σ(*・Д・)ﾊｯ"]
        }
    },
    "dislike":{
        "POSITIVE":{
            "en": ["don't care","nobody asked","ok","sure","ok calm down", "and...?","yeah and i got into harvard with a 6.0 gpa, anyone wanna hear about that instead?",f":thumbsup","yawn","k"],
            "jp": ["Σ(*・Д・)ﾊｯ"]
        },
        "NEGATIVE":{
            "en": [f":WAHAHA:", "lol k", "lmao", "good for you!!","huh","you should go outside","well sucks to be you i guess","haha!!!","that's a good thing right??","oh um that's... i actually feel bad ||SIKE||"],
            "jp": ["Σ(*・Д・)ﾊｯ"]
        },
        "NEUTRAL":{
            "en": ["what does that even mean", "alright","?","why should that matter to anyone","...ok?","im not sure why you're telling me this",f":jimstare:"],
            "jp": ["Σ(*・Д・)ﾊｯ"]
        }
    }
}

bot.persona = "bald"
bot.interests = {"bald":{"hank":{"en":"my dog hank somehow always finds a way to open the basement door, steal food from the fridge, and turn on enough electric appliances to trip the circuit breaker... and i've never been able to catch him doing it, it's always when im not paying attention. he's really smart for a chihuahua, that's for sure","jp":""},"pizza":{"en":"yum pizza","jp":""}, "linux":{"en":"linux best os, if you think it's bad you don't understand it enough","jp":""}, "assembly":{"en":"assembly, LISP, aaahh all great programming languages","jp":""}, "unit test":{"en":"hell yeah i love those! i write unit tests as a hobby","jp":""}, "latin":{"en":"nisi optimis mentibus bene Latine","jp":""}, "ide":{"en":"i honestly prefer to write my code out in pen and paper. debugging is easy once you have that level of experience","jp":""}, "philosophy":{"en":"cogito ergo sum...","jp":""}},
                 "non":{"dog":{"en":"DOG!!!!","jp":""},"bird":{"en":"borb","jp":""},"cat":{"en":"kitty!!","jp":""},"bumblebee":{"en":"BEE!","jp":""},"frog":{"en":"FROGGY!!!","jp":""},"math":{"en":"i hate math... but i dont want to stop doing it either","jp":""},"feynman":{"en":"big fan of feynman's work... even though he contributed to the wwii bombing of japan","jp":""},"art":{"en":"if you draw or paint, show me! id love to see it","jp":""},"origami":{"en":"don't have much time for origami nowadays, but i love tesselations!","jp":""},"physics":{"en":"fijiks... scary most of the time but that's why it's so fun to venture into","jp":""},"curry":{"en":"japanese curry is probably sweeter than normal curry... im bad with hot peppers honestly","jp":""}}}
bot.userlist = {}

def question(topic_token):
    pass


@bot.command()
async def greet(ctx, message):
    if message.content.startswith("🇯🇵"):
        await ctx.send("{greeting}、{user}！".format(greeting=random.choice(bot.lines["greet"]["jp"]), user=ctx.author.mention))
    else:
        await ctx.send("{greeting},{user}!".format(greeting=random.choice(bot.lines["greet"]["en"]), user=ctx.author.mention))
    if not (message.author.id in bot.userlist):
        bot.userlist[message.author.id] = [message.author.display_name, 31, "acquaintance"]

#if ("dialect" in message) and ("japanese" in message):
#        await ctx.send(bot.lines["dialect"]["en"]+"https://osaka.uda2.com/"+"\n"+"http://www.shirakami.or.jp/~kinoka/akitaben/akitaben.html")

@bot.command()
async def playlist(ctx, message):
    if ("calm" in message.content):
        await ctx.send(f"https://music.youtube.com/playlist?list=PLDkI5DGFbBQUQNTXLk8rX6Nb-_uCemW2F&feature=share")
    elif ("calmer" in message.content):
        await ctx.send(f"https://music.youtube.com/playlist?list=PLDkI5DGFbBQWU0sdiUAx5k0RuQCNfIBcV")
    elif ("comfort" in message.content):
        await ctx.send(f"https://music.youtube.com/playlist?list=PLDkI5DGFbBQXaMJU4uwi9I50kudfQI6u_&feature=share")
    elif ("upbeat" in message.content):
        await ctx.send(f"https://music.youtube.com/playlist?list=PLDkI5DGFbBQUKJCZ3nDiEz3IqI6oac_NF")
    elif ("lofi" in message.content):
        await ctx.send(f"https://music.youtube.com/playlist?list=PLDkI5DGFbBQX5VNDsNYDVr7qJ5f4hPS3F&feature=share")
    elif ("soundtracks" in message.content):
        await ctx.send(f"https://music.youtube.com/playlist?list=PLDkI5DGFbBQWYKxVpIJRO90ocQp_IXwVy&feature=share")
    else:
        await ctx.send("I don't have a playlist for that topic. See the pinned comment for documentation!")

@bot.event
async def on_ready():
    print("ready")

frog_pfp = "/freg.jpg"
hagrid_pfp = "./hagger.jpg"

@bot.command()
async def switch_persona(ctx, message):
    nickname ="bot"
    if "bob" in message.content.lower():
        bot.persona = "bald"
        fp = open(frog_pfp, 'rb')
        pfp = fp.read()
        nickname = "nono"
    else:
        bot.persona = "non"
        fp = open(hagrid_pfp, 'rb')
        pfp = fp.read()
        nickname ="Bob Arthur Lewis Dunn"
    try:
        await bot.user.edit(avatar=pfp)
        await ctx.guild.me.edit(nick=nickname)
        await ctx.send("Henshin!")
    except:
        await ctx.send("Sorry, on cooldown for changing user details!")

@bot.command()
async def friendlist(ctx):
    await ctx.send(tabulate(list(bot.userlist.values()), headers=["Name", "Points", "Title"]))

@bot.event
async def on_message(message):
    print("on_message called")
    lang = "en"
    if message.content.startswith("🇯🇵"):
        lang = "jp"
    result=sentiment_task(message.content)
    tone = "NEUTRAL"
    if result[0]["score"] < sentiment_thresh:
        tone = "NEUTRAL"
    else:
        tone = result[0]["label"]
    
    if message.author == client.user:
        print("detected self post")
        return
    
    sentence = Sentence(message.content)
    vp, v_np, subj = extract_topic(sentence)
    
    if v_np == "":
        await message.channel.send("care to elaborate what exactly who or what \'{}\'?".format(vp))
        if bot.userlist.has_key(message.author.id):
            if bot.userlist[message.author.id][2] == "BFF":
                await message.channel.send(bot.lines["friend"][tone][lang])
            if bot.userlist[message.author.id][2] == "friend":
                await message.channel.send(bot.lines["friend"][tone][lang])
            if bot.userlist[message.author.id][2] == "acquaintance":
                await message.channel.send(bot.lines["stranger"][tone][lang])
            if bot.userlist[message.author.id][2] == "stranger":
                await message.channel.send(bot.lines["stranger"][tone][lang])
            if bot.userlist[message.author.id][2] == "MORTAL ENEMY":
                await message.channel.send(bot.lines["dislike"][tone][lang])
        else:
            await message.channel.send(bot.lines["stranger"][tone][lang])
    else:
        if (v_np in bot.interests[bot.persona]):
            await message.channel.send(bot.interests[bot.persona][v_np][lang])
        elif (subj in bot.interests[bot.persona]):
            await message.channel.send(bot.interests[bot.persona][subj][lang])
        else:
            if tone == "NEGATIVE":
                await message.channel.send("sounds like "+v_np+" isn't a nice topic for "+subj)
            elif tone == "POSITIVE":
                await message.channel.send("sounds like "+vp+" was/is a good thing!")
            elif tone == "NEUTRAL":
                await message.channel.send("hmm i wonder how i would feel about "+v_np)

def update_relation(author_id, user_id, sentiment_score):
    MAX_POINT_DIFF = 10
    bot.userlist[user_id][1] += round(10.0*sentiment_score*(bot.userlist[author_id][1]/100.0))
    bot.userlist[user_id][1] = min(100,bot.userlist[user_id][1]) #clamp
    
    if bot.userlist[user_id][1]>98:
        bot.userlist[user_id][2] = "BFF"
    elif bot.userlist[user_id][1]>85:
        bot.userlist[user_id][2] = "close friend"
    elif bot.userlist[user_id][1]>50:
        bot.userlist[user_id][2] = "friend"
    elif bot.userlist[user_id][1]>30:
        bot.userlist[user_id][2] = "acquaintance"
    elif bot.userlist[user_id][1]>=10:
        bot.userlist[user_id][2] = "stranger"
    elif bot.userlist[user_id][1]<10:
        bot.userlist[user_id][2] = "MORTAL ENEMY"

def extract_topic(s):
    s_chunk = s[:]
    #s_pos = s[:]
    chunk_result = chunk_tagger.predict(s_chunk)
    #pos_result = pos_tagger.predict(s_pos)
    verb = ""
    obj = ""
    subj = ""
    for phrase in s_chunk.labels:
        if phrase.value == "VP":
            verb = phrase.value
        elif phrase.value == "NP" and verb != "":
            if verb != "":
                subj = phrase.value
            else:
                obj = phrase.value
            break
    return verb, obj, subj

if __name__ == "__main__":
	bot.run(bot_token)
