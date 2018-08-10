import pandas as pd
import numpy as np

def new_deck():
    return pd.Series([(suit,rank) for suit in range(1,5) for rank in range(2,15)])

def card_to_console(x):
    suitcolor = {1:'\033[37m♣',2:'\033[91m♦',3:'\033[37m♠',4:'\033[91m♥',}
    ordermap  = {2:'2\033[0m',3:'3\033[0m',4:'4\033[0m',5:'5\033[0m',6:'6\033[0m',7:'7\033[0m',8:'8\033[0m',9:'9\033[0m',10:'10\033[0m',11:'J\033[0m',12:'Q\033[0m',13:'K\033[0m',14:'A\033[0m'}
    return suitcolor[x[0]]+ordermap[x[1]]

def hand_to_console(x):
    return ' '.join([card_to_console(y) for y in x])

def placeholder_agent(state,plays):
    print("Placeholder Play",hand_to_console(plays))
    print()
    return 0

def random_agent(state,plays):
    print("Random Play",hand_to_console(plays))
    print()
    return np.random.choice(range(len(plays)))

def human_agent(state,plays):
    print("Human Play ",hand_to_console(plays),": ",end='')
    try:
        play = int(input())
    except:
        play = 0
    print()
    return play

class table:
    def __init__(self,agents=(human_agent,placeholder_agent,random_agent,random_agent,)):
        self.agents  = agents
    #
    def new_game(self):
        self.scores  = [0,0,0,0]  # Player scores
        self.hands   = [[],[],[],[]]  # Player hands
        self.first   = True  # First trick or not
        self.heart_broken  = False  # Hearts broke?
        self.lead    = None  # lead player in current trick
        self.act     = None  # next player to act in current trick
        self.suit    = None  # Suit of the current trick
        self.board   = pd.Series([None,None,None,None])  # Board
        self.discards = pd.Series()  # Discarded cards
    #
    def deal(self):
        deck  = new_deck()
        np.random.shuffle(deck)
        self.hands = [
            deck[:13].sort_values(),
            deck[13:26].sort_values(),
            deck[26:39].sort_values(),
            deck[39:].sort_values(),
            ]
        self.first        = True
        self.heart_broken = False
        for i in range(4):
            if self.hands[i].iloc[0]==(1,2): # 2 of clubs
                self.lead  = i
                self.act   = i
                break
        #
        print("Deal")
        self.print_state()
        print()
    #
    def next_move(self):
        state = {
            'hand':  self.hands[self.act],
            'board': self.board,
            'discards': self.discards,
            'self_score': self.scores[self.act],
            'scores': self.scores,
            }
        if self.first:
            plays = state['hand']==(1,2)
        elif self.act == self.lead:
            plays = state['hand'].str[0]!=4
            if self.heart_broken or plays.sum()==0: plays = None
        else:
            plays = state['hand'].str[0]==self.suit
            if plays.sum()==0: plays = None
        #
        plays = state['hand'][plays] if plays is not None else state['hand']
        play  = self.agents[self.act](state,plays)
        #
        if play < 0 or play >= len(plays): play = 0
        self.board[self.act]  = plays.iloc[play]
        self.hands[self.act]  = self.hands[self.act][self.hands[self.act]!=plays.iloc[play]]
        #
        self.print_state()
        print()
        #
        if self.act == self.lead:
            self.suit   = self.board[self.act][0]
        self.act    = (self.act + 1) % 4
        self.first  = False
        #
        if self.board.notnull().all():
            # All players have played in this trick
            winner = self.board[self.board.str[0]==self.suit].str[1].idxmax()
            score  = (self.board.str[0]==4).sum() + 13*((3,12) in self.board.tolist())
            self.scores[winner] += score
            self.heart_broken = (self.board.str[0]==4).any()
            self.discards = pd.concat([self.discards,self.board])
            self.board    = pd.Series([None,None,None,None])
            self.lead     = winner
            self.act      = self.lead
            #
            self.print_state()
            print()
        #
        return True
    #
    def print_state(self):
        for i in range(4):
            print("%d (%d): %s[%s] [%s]" % (i,self.scores[i],'*' if i==self.lead else ' ',card_to_console(self.board[i]) if self.board[i] else '  ',hand_to_console(self.hands[i])))
        print("Discards: [%s]" % hand_to_console(sorted(self.discards)))

if __name__ == '__main__':
    t  = table()
    t.new_game()
    t.deal()
    while True:
        t.next_move()
