import pandas as pd
import numpy as np

def new_deck():
    return [(suit,rank) for suit in range(1,5) for rank in range(2,15)]

def card_to_console(x):
    suitcolor = {1:'\033[37m♣',2:'\033[91m♦',3:'\033[37m♠',4:'\033[91m♥',}
    ordermap  = {2:'2\033[0m',3:'3\033[0m',4:'4\033[0m',5:'5\033[0m',6:'6\033[0m',7:'7\033[0m',8:'8\033[0m',9:'9\033[0m',10:'10\033[0m',11:'J\033[0m',12:'Q\033[0m',13:'K\033[0m',14:'A\033[0m'}
    return suitcolor[x[0]]+ordermap[x[1]]

def hand_to_console(x):
    return ' '.join([card_to_console(y) for y in x])

def human_agent(hand,board,plays,discards,scores):
    return 0

class table:
    def new_game(self):
        self.scores  = [0,0,0,0]  # Player scores
        self.hands   = [[],[],[],[]]  # Player hands
        self.first   = True  # First trick or not
        self.heart_broken  = False  # Hearts broke?
        self.lead    = None  # lead player in current trick
        self.act     = None  # next player to act in current trick
        self.board   = [None,None,None,None]  # Board
        self.discards = []  # Discarded cards
    #
    def deal(self):
        deck  = new_deck()
        np.random.shuffle(deck)
        self.hands = [
            sorted(deck[:13]),
            sorted(deck[13:26]),
            sorted(deck[26:39]),
            sorted(deck[39:]),
            ]
        self.first        = True
        self.heart_broken = False
        for i in range(4):
            if self.hands[i][0]==(1,2): # 2 of clubs
                self.lead  = i
                self.act   = i
                break
    #
    def next_move(self):
        if self.first:
            plays = 0
            play  = human_agent(self.hands[self.act],self.board,self.hands[0],)
            self.first  = False
        else:
    #
    def print_state(self):
        for i in range(4):
            print("%d (%d): %s[%s] [%s]" % (i,self.scores[i],'*' if i==self.lead else ' ',card_to_console(self.board[i]) if self.board[i] else '  ',hand_to_console(self.hands[i])))
        print("Discards: [%s]" % hand_to_console(sorted(self.discards)))
