import pandas as pd
import numpy as np

def new_deck():
    return pd.Series([(suit,rank) for suit in range(1,5) for rank in range(2,15)])

def card_to_console(x):
    suitcolor = {1:'\033[37m♣',2:'\033[91m♦',3:'\033[37m♠',4:'\033[91m♥',}
    ordermap  = {2:'2\033[0m',3:'3\033[0m',4:'4\033[0m',5:'5\033[0m',6:'6\033[0m',7:'7\033[0m',8:'8\033[0m',9:'9\033[0m',10:'T\033[0m',11:'J\033[0m',12:'Q\033[0m',13:'K\033[0m',14:'A\033[0m'}
    return suitcolor[x[0]]+ordermap[x[1]]

def hand_to_console(x):
    return ' '.join([card_to_console(y) if y is not None else '[]' for y in x])

def placeholder_agent(state):
    play  = 0
    print('Placeholder Play',hand_to_console(state['hand'].loc[state['hand'].play>=0,'card']),'=>',card_to_console(state['hand'].loc[state['hand'].play==play,'card'].iloc[0]))
    print()
    return play

def random_agent(state):
    play  = np.random.choice(range(state['hand'].play.max()+1))
    print('Random Play',hand_to_console(state['hand'].loc[state['hand'].play>=0,'card']),'=>',card_to_console(state['hand'].loc[state['hand'].play==play,'card'].iloc[0]))
    print()
    return play

def human_agent(state):
    print('Board:')
    print(' '+hand_to_console(state['board']))
    print('Choose Card to Play:')
    print(' '+hand_to_console(state['hand'].card).replace(' ','  '))
    print(*["%3d"%i if i>=0 else '   ' for i in state['hand'].play])
    try:
        play = int(input())
    except:
        play = 0
    return play

class table:
    #
    def __init__(self,agents=(human_agent,placeholder_agent,random_agent,random_agent,)):
        self.agents  = agents
        self.players = pd.DataFrame([x.__name__ for x in agents],columns=('name',))
    #
    def new_game(self):
        self.players['score']  = 0  # Player scores
        self.round_id  = 0
    #
    def deal(self):
        # New Round
        deck  = new_deck()
        np.random.shuffle(deck)
        self.players['hand'] = [
            deck[:13].sort_values(),
            deck[13:26].sort_values(),
            deck[26:39].sort_values(),
            deck[39:].sort_values(),
            ]
        self.round_id    += 1
        self.trick        = 1
        self.heart_broken = False
        self.lead     = -1  # lead player in current trick
        self.act      = -1  # next player to act in current trick
        self.suit     = -1  # Suit of the current trick
        self.board    = pd.Series([None,None,None,None])    # Board
        self.discards = pd.DataFrame(columns=self.players.index) # Discarded cards
        #
        #-- Determine first trick lead --#
        for i in range(4):
            if self.players.loc[i,'hand'].iloc[0]==(1,2): # 2 of clubs
                self.lead  = i
                self.act   = i
                break
        #
        print("Round %d"%self.round_id)
        print("Trick 1")
        self.print_state()
        print()
    #
    def next_move(self):
        state = {
            'id':     self.act,
            'scores': self.players.score,
            'hand':   pd.DataFrame(self.players.loc[self.act,'hand'].copy(),columns=('card',)),
            'board':  self.board,
            'discards': self.discards,
            }
        if self.act == self.lead:
            if self.trick == 1:
                mask  = state['hand'].card==(1,2)
            else:
                mask  = state['hand'].card.str[0]!=4
                if self.heart_broken or mask.sum()==0: mask = None
        else:
            mask = state['hand'].card.str[0]==self.suit
            if mask.sum()==0: mask = None
        #
        state['hand']['play']  = -1
        if mask is not None:
            state['hand'].loc[mask,'play']  = range(mask.sum())
        else:
            state['hand']['play']  = range(len(state['hand']))
        #
        play  = self.agents[self.act](state)
        if play in state['hand'].play.values:
            play  = state['hand'].loc[state['hand'].play==play,'card'].iloc[0]
        else:
            play  = state['hand'].loc[state['hand'].play==0,'card'].iloc[0]
        #
        self.board[self.act]  = play
        self.players.loc[self.act,'hand'].drop(state['hand'].index[state['hand'].card==play],'index',inplace=True)
        #
        self.print_state()
        print()
        #
        if self.act == self.lead:
            self.suit   = self.board[self.act][0]
        self.act    = (self.act + 1) % 4
        #
        if self.act == self.lead:
            # All players have played in this trick
            winner = self.board[self.board.str[0]==self.suit].str[1].idxmax()
            score  = (self.board.str[0]==4).sum() + 13*((3,12) in self.board.tolist())
            self.players.loc[winner,'score'] += score
            self.heart_broken = (self.board.str[0]==4).any()
            self.discards.loc[self.trick] = self.board
            self.board    = pd.Series([None,None,None,None])
            self.lead     = winner
            self.act      = self.lead
            self.trick   += 1
            #
            print("Trick %d" % self.trick)
            self.print_state()
            print()
        #
        return self.trick <= 13
    #
    def print_state(self):
        for i in range(4):
            print("%d (%d): %s[%s] [%s]" % (i,self.players.loc[i,'score'],'*' if i==self.lead else ' ',card_to_console(self.board[i]) if self.board[i] else '  ',hand_to_console(self.players.loc[i,'hand'])))

if __name__ == '__main__':
    t  = table()
    t.new_game()
    t.deal()
    while t.next_move():
        pass
    print(t.players)
    print()
