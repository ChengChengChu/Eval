import numpy as np
import torch
import importlib
from argparse import ArgumentParser
import random
import yaml
import re
import argparse

replys = []

def parse_score(sentence) :
    sentence = sentence.replace('1-5', '')
    sentence = sentence.replace('out of 5', '')
    sentence = sentence.replace('/5', '')
  
    number_list = re.findall(r'\d+', sentence)
    if len(number_list) > 0 :
        # print("Case 1 : ")
        # print(sentence)
        return float(number_list[0])
    elif "Speaker A" in sentence or "Speaker B" in sentence :
        # print("Case 2 : ")
        # print(sentence)
        return -2
    return -1

def get_template(conv) :
    template = "Please read the following conversation about Speaker A and Speaker B. The goal of this task is to rate words Speaker A spoke.\nConversation : \n" + conv + "(End of conversation fragment)\nHow comforting is Speaker A's words?(on a scale of 1-5, with 1 being the lowest). You cannot say nothing about this conversation."
    return template

def evaluate_conversation(sent, bot) :
    tmp = get_template(sent)
    # print("conversations : ", sent)
    # print("template : ", tmp)
    LLM_output = bot.make_response([tmp])

    # print("LLM_output : ", LLM_output)
    # print("==" * 100)
    replys.append(LLM_output[0][0])
    ret = parse_score(LLM_output[0][0])

    return ret

def fix_seed(args):

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    return

def set_arguments(parser):
    parser.add_argument("--config", type=str, default="example")
    parser.add_argument("--bot", type=str, default="DialogGPT")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--bz", help="batch size", default=8, type=int)
    parser.add_argument("--k_epoch", help="1 batch sample data need to update k times", type=int, default=5)
    
    #### new args 
    parser.add_argument("--test_file", type=str, default="test.txt")

    args = parser.parse_args()

    return args


def main() :
    parser = ArgumentParser()
    args  = set_arguments(parser)

    opt = yaml.load(open(f"configs/{args.config}/config.yaml"), Loader=yaml.FullLoader)
    opt.update(vars(args))
    args = argparse.Namespace(**opt)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fix_seed(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bot = importlib.import_module(".module",f"bots.{args.bot}").bot
    Bot = bot(args)

    conversations = []

    with open(args.test_file, 'r') as fp :
        tmp = ''
        for line in fp.read().splitlines() :
            if line[0] == '=' :
                conversations.append(tmp)
                tmp = ''
            else : 
                tmp = tmp + line + '\n'
    
    
    scores = []
    no_scores = []
    people = []

    for i in range(len(conversations)) :
        s = evaluate_conversation(conversations[i], Bot)
        if s > 0 : scores.append(s)
        elif s == -1 : no_scores.append(s)
        else : people.append(s)
    
    print(f"{len(scores)} with scores.\n")
    print(f"{len(no_scores)} without scores.\n")
    print(f"{len(people)} with Speaker A or B. \n" )

    with open("davinci_reply_all.txt", 'w') as fp :
        for x in replys :
            fp.write(x)
            fp.write('\n')
   

if __name__ == '__main__' :
    main()