import pandas as pd
import numpy as np

#-- Utilities --#
def new_deck():
    return pd.Series([(suit,rank) for suit in range(1,5) for rank in range(2,15)])

def card_to_console(card):
    suitcolor = {1:'\033[30m♣',2:'\033[91m♦',3:'\033[30m♠',4:'\033[91m♥',}
    ordermap  = {2:'2\033[0m',3:'3\033[0m',4:'4\033[0m',5:'5\033[0m',6:'6\033[0m',7:'7\033[0m',8:'8\033[0m',9:'9\033[0m',10:'T\033[0m',11:'J\033[0m',12:'Q\033[0m',13:'K\033[0m',14:'A\033[0m'}
    return suitcolor[card[0]]+ordermap[card[1]]

def hand_to_console(hand):
    return ' '.join(['[]' if x is None else card_to_console(x) for x in hand])

def trick_winner(board,suit):
    return board[board.str[0]==suit].str[1].idxmax()

def card_to_score(card):
    return (card[0]==4) + 13*(card==(3,12))

def hand_to_score(hand):
    return sum([x[0]==4 for x in hand]) + 13*((3,12) in list(hand))

def card_rank(card,outs):
    # Rank of card if played as lead: 0-based
    return sum([x[0]==card[0] and x[1]>card[1] for x in outs if x is not None])

#-- Agents --#
def placeholder_agent(state):
    playable_hand  = state['hand'][state['hand'].play>=0]
    playable_hand['order']  = playable_hand.card.str[1]
    play  = playable_hand.sort_values(by='order',ascending=True).iloc[0].play
    #
    print('Placeholder Play',hand_to_console(state['hand'].loc[state['hand'].play>=0,'card']),'=>',card_to_console(state['hand'].loc[state['hand'].play==play,'card'].iloc[0]))
    print()
    return play

def rule_based_agent(state):
    outs  = new_deck()
    outs  = outs[~outs.isin(state['board'])]
    outs  = outs[~outs.isin(state['hand'].card)]
    outs  = outs[~outs.isin(t.discards.values.flatten())]
    #
    state['hand']['score']     = state['hand'].card.apply(card_to_score)
    state['hand']['outs_rank'] = state['hand'].card.apply(lambda x:card_rank(x,outs))
    state['hand']['board_rank'] = state['hand'].card.apply(lambda x:card_rank(x,state['board']) if x[0]==state['suit'] else 1) if state['suit']>0 else 0
    #
    playable_hand  = state['hand'][state['hand'].play>=0]
    if (playable_hand.board_rank>0).any():
        # There exists some legal play that is guaranteed to lose this trick
        mask  = playable_hand.board_rank>0
        # Play the card with highest score then highest rank (w.r.t. outs)
        play  = playable_hand[mask].sort_values(by=['score','outs_rank','card'],ascending=[False,True,False]).iloc[0].play
    elif state['board'].notnull().sum() == 3:
        # Three players has played so we are guaranteed to win this trick
        # Play the card with lowest score then highest rank (w.r.t. outs)
        play  = playable_hand.sort_values(by=['score','outs_rank','card'],ascending=[True,True,False]).iloc[0].play
    else:
        # Leading or There are no legal play guaranteed to lose this trick
        # Play the card with lowest rank (w.r.t. outs)
        play  = playable_hand.sort_values(by=['outs_rank','card'],ascending=[False,True]).iloc[0].play
    #
    print('Rule Based Play',hand_to_console(state['hand'].loc[state['hand'].play>=0,'card']),'=>',card_to_console(state['hand'].loc[state['hand'].play==play,'card'].iloc[0]))
    print()
    return play

def random_agent(state):
    play  = np.random.choice(range(state['hand'].play.max()+1))
    print('Random Play',hand_to_console(state['hand'].loc[state['hand'].play>=0,'card']),'=>',card_to_console(state['hand'].loc[state['hand'].play==play,'card'].iloc[0]))
    print()
    return play

def human_agent(state):
    outs  = new_deck()
    outs  = outs[~outs.isin(state['board'])]
    outs  = outs[~outs.isin(state['hand'].card)]
    outs  = outs[~outs.isin(t.discards.values.flatten())]
    #
    state['hand']['score']     = state['hand'].card.apply(card_to_score)
    state['hand']['outs_rank'] = state['hand'].card.apply(lambda x:card_rank(x,outs))
    state['hand']['board_rank'] = state['hand'].card.apply(lambda x:card_rank(x,state['board']) if x[0]==state['suit'] else 1) if state['suit']>0 else 0
    #
    print('Board:')
    if len(state['discards']):
        for idx,row in state['discards'].iterrows():
            print(("%-2d   "%idx) + hand_to_console(row))
    print(("%-2d   "%state['trick']) + hand_to_console(state['board']))
    #
    print('Choose Card to Play:')
    print('       '+hand_to_console(state['hand'].card).replace(' ','  '))
    print('outs ',*["%3d"%i for i in state['hand'].outs_rank])
    if state['suit'] > 0:
        print('board',*["%3d"%i if i>=0 else '   ' for i in state['hand'].board_rank])
    print('play ',*["%3d"%i if i>=0 else '   ' for i in state['hand'].play])
    try: play = int(input())
    except: play = 0
    return play

#-- Game --#
class table:
    #
    def __init__(self,agents=(placeholder_agent,placeholder_agent,rule_based_agent,random_agent,)):
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
        self.suit     = 0   # Suit of the current trick: 0 means no one played yet
        self.board    = pd.Series([None,None,None,None])    # Board
        self.discards = pd.DataFrame(columns=self.players.index) # Discarded cards
        self.scores   = pd.Series([0,0,0,0]) # Scores for this round
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
        hand  = pd.DataFrame(self.players.loc[self.act,'hand'].copy(),columns=('card',))
        if self.act == self.lead:
            if self.trick == 1:
                mask  = hand.card==(1,2)
            else:
                mask  = hand.card.str[0]!=4
                if self.heart_broken or mask.sum()==0: mask = None
        else:
            mask  = hand.card.str[0]==self.suit
            if mask.sum()==0:
                if self.trick == 1:
                    mask  = hand.card.apply(card_to_score)==0
                    if mask.sum()==0:
                        mask = None
                else:
                    mask  = None
        #
        hand['play']  = -1
        if mask is not None:
            hand.loc[mask,'play']  = range(mask.sum())
        else:
            hand['play']  = range(len(hand))
        #
        state = {
            'id':     self.act,
            'scores': self.players.score.copy(),
            'round_scores': self.scores.copy(),
            'hand':   hand.copy(),
            'trick':  self.trick,
            'suit':   self.suit,
            'board':  self.board.copy(),
            'discards': self.discards.copy(),
            }
        play  = self.agents[self.act](state)
        if play >= 0 and play <= hand.play.max():
            play  = hand.loc[hand.play==play,'card'].iloc[0]
        else:
            play  = hand.loc[hand.play==0,'card'].iloc[0]
        #
        self.board[self.act]  = play
        self.players.loc[self.act,'hand'].drop(hand.index[hand.card==play],'index',inplace=True)
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
            winner = trick_winner(self.board,self.suit)
            score  = hand_to_score(self.board)
            self.scores[winner] += score
            if not self.heart_broken and (self.board.str[0]==4).any():
                self.heart_broken  = True
            self.discards.loc[self.trick] = self.board
            self.board    = pd.Series([None,None,None,None])
            self.lead     = winner
            self.act      = self.lead
            self.trick   += 1
            self.suit     = 0
            #
            if self.trick <= 13:
                print("Trick %d" % self.trick)
                self.print_state()
                print()
            else:
                # Round End
                # Shoot the Moon
                if self.scores.max() == 26:
                    max_idx  = self.scores.idxmax()
                    self.scores[:]  = 26
                    self.scores[max_idx]  = 0
                #
                self.players.score  += self.scores
                #
                if (self.players.score > 100).any():
                    # Game Over
                    winner  = self.players.score.idxmin()
                    print("Winner is %d:%s!!!" % (winner,self.players.loc[winner,'name']))
                    print()
                    return False
                else:
                    t.deal()
        #
        return True
    #
    def print_state(self):
        for i in range(4):
            print("%d (%3d + %2d): %s[%s] [%s]" % (i,self.players.loc[i,'score'],self.scores[i],'*' if i==self.lead else ' ',card_to_console(self.board[i]) if self.board[i] else '  ',hand_to_console(self.players.loc[i,'hand'])))

if __name__ == '__main__':
    t  = table()
    #
    Nsim  = 1000
    results = []
    for i in range(Nsim):
        t.new_game()
        t.deal()
        while t.next_move():
            pass
        print(t.players[['name','score']])
        print()
        results.append(t.players[['name','score']].set_index('name',append=True).score.copy())
    results  = pd.concat(results,1,keys=range(Nsim)).T
    results['winner']  = results.idxmin(1)
    print(results)
