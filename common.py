import pandas as pd
import numpy as np

#-- Utilities --#
def new_deck():
    return pd.Series([(suit,rank) for suit in range(1,5) for rank in range(2,15)])

suitcolor = {1:'\033[30m♣',2:'\033[91m♦',3:'\033[30m♠',4:'\033[91m♥',}
ordermap  = {2:'2\033[0m',3:'3\033[0m',4:'4\033[0m',5:'5\033[0m',6:'6\033[0m',7:'7\033[0m',8:'8\033[0m',9:'9\033[0m',10:'T\033[0m',11:'J\033[0m',12:'Q\033[0m',13:'K\033[0m',14:'A\033[0m'}
def card_to_console(card):
    return suitcolor[card[0]]+ordermap[card[1]]

def hand_to_console(hand):
    return ' '.join(['[]' if x is None else card_to_console(x) for x in hand])

def card_to_score(card):
    return (card[0]==4) + 13*(card==(3,12))

def hand_to_score(hand):
    return sum([x[0]==4 for x in hand]) + 13*((3,12) in list(hand))

def playable_mask(hand,trick,trick_suit,heart_broken):
    if trick_suit == 0:
        # Leading if trick_suit == 0
        if trick == 1:
            mask  = hand==(1,2)
        else:
            mask  = hand.str[0]!=4
            if heart_broken or mask.sum()==0: mask = None
    else:
        mask  = hand.str[0]==trick_suit
        if mask.sum()==0:
            if trick == 1:
                mask  = hand.apply(card_to_score)==0
                if mask.sum()==0: mask = None
            else:
                mask  = None
    return mask

def trick_winner(board,suit):
    return board[board.str[0]==suit].str[1].idxmax()

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
    outs  = outs[~outs.isin(state['discards'].values.flatten())]
    #
    state['hand']['score']     = state['hand'].card.apply(card_to_score)
    state['hand']['outs_rank'] = state['hand'].card.apply(lambda x:card_rank(x,outs))
    state['hand']['board_rank'] = state['hand'].card.apply(lambda x:card_rank(x,state['board']) if x[0]==state['trick_suit'] else 1) if state['trick_suit']>0 else 0
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
    outs  = outs[~outs.isin(state['discards'].values.flatten())]
    out_suits  = [len(outs),0,0,0,0]
    for i in range(1,5):
        out_suits[i]  = (outs.str[0]==i).sum()
    #
    state['hand']['score']     = state['hand'].card.apply(card_to_score)
    state['hand']['outs_rank'] = state['hand'].card.apply(lambda x:card_rank(x,outs))
    state['hand']['board_rank'] = state['hand'].card.apply(lambda x:card_rank(x,state['board']) if x[0]==state['trick_suit'] else 1) if state['trick_suit']>0 else 0
    #
    print('Board:')
    if len(state['discards']):
        for idx,row in state['discards'].iterrows():
            print(("%-2d   "%idx) + hand_to_console(row).replace(' ','  '))
    print(("%-2d   "%state['trick']) + hand_to_console(state['board']).replace(' ','  '))
    #
    print('Choose Card to Play:')
    print('       '+hand_to_console(state['hand'].card).replace(' ','  '),end='')
    print('    '+suitcolor[1]+("\033[0m:%-2d"%out_suits[1])+'  '+suitcolor[2]+("\033[0m:%-2d"%out_suits[2]))
    print('outs ',*["%3d"%i for i in state['hand'].outs_rank],end='')
    print('    '+suitcolor[3]+("\033[0m:%-2d"%out_suits[3])+'  '+suitcolor[4]+("\033[0m:%-2d"%out_suits[4]))
    if state['trick_suit'] > 0:
        print('board',*["%3d"%i if i>=0 else '   ' for i in state['hand'].board_rank])
    print('play ',*["%3d"%i if i>=0 else '   ' for i in state['hand'].play])
    try:
        play = input()
        if play.lower() == 'q':
            exit(0)
        else:
            play = int(play)
    except: play = 0
    return play

#-- Game --#
class table:
    #
    def __init__(self,agents=(placeholder_agent,placeholder_agent,rule_based_agent,random_agent,)):
        self.agents  = agents if isinstance(agents,list) or isinstance(agents,tuple) else (agents,placeholder_agent,rule_based_agent,random_agent,)
        self.players = pd.DataFrame([x.__name__ for x in self.agents],columns=('name',))
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
        self.board    = pd.Series([None,]*4,name=1)    # Board
        self.discards = pd.DataFrame(columns=self.players.index) # Discarded cards
        self.scores   = pd.Series([0,]*4,name='score') # Scores for this round
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
        mask  = playable_mask(hand.card,self.trick,self.suit,self.heart_broken)
        #
        hand['play']  = -1
        if mask is not None:
            hand.loc[mask,'play']  = range(mask.sum())
        else:
            hand['play']  = range(len(hand))
        #
        state = {
            'id':           self.act,
            'scores':       self.players.score.copy(),
            'round_scores': self.scores.copy(),
            'hand':         hand.copy(),
            'trick':        self.trick,
            'trick_lead':   self.lead,
            'trick_suit':   self.suit,
            'heart_broken': self.heart_broken,
            'board':        self.board.copy(),
            'discards':     self.discards.copy(),
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
            self.lead     = winner
            self.act      = self.lead
            self.trick   += 1
            self.board    = pd.Series([None,]*4,name=self.trick)
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
                    print("%d:%s Shot the Moon!!!" % (max_idx,self.players.loc[winner,'name']))
                    print()
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
                    self.deal()
        #
        return True
    #
    def print_state(self):
        for i in range(4):
            print("%d (%3d + %2d): %s[%s] [%s]" % (i,self.players.loc[i,'score'],self.scores[i],'*' if i==self.lead else ' ',card_to_console(self.board[i]) if self.board[i] else '  ',hand_to_console(self.players.loc[i,'hand'])))

def simulation_from_table(t):
    results  = simulation_step(t.act,t.scores.copy(),t.players['hand'].copy(),t.heart_broken,t.trick,t.lead,t.suit,t.board.copy())
    return [pd.concat([x[1].apply(lambda x:[card_to_console(y) for y in x],axis=0),x[0]],1) for x in results] #.apply(hand_to_console,axis=0)

def simulation_step(act,scores,hands,heart_broken,trick,trick_lead,trick_suit,board):
    hand   = hands[act].copy()
    mask   = playable_mask(hand,trick,trick_suit,heart_broken)
    if mask is None: playable_hand = hand.copy()
    else: playable_hand = hand[mask].copy()
    if trick == 13:
        # This is the last trick, only one possibility remain
        board1   = board.copy()
        for i in range(4):
            if pd.isnull(board1[i]): board1[i] = hands[i].iloc[0]
        winner1  = trick_winner(board1,board1[trick_lead][0])
        scores1  = scores.copy()
        scores1[winner1] += hand_to_score(board1)
        if scores1.max() == 26:
            scores1[scores1.idxmax()]  = -26
            scores1  += 26
        return [(scores1,pd.DataFrame(board1))]
    elif board.notnull().sum() == 3:
        # This move ends a trick
        if trick == 12:
            # This move will effectively end the game
            results  = []
            for card in playable_hand:
                # Consider each possible move
                board1   = board.copy()
                board1[act]  = card
                winner1  = trick_winner(board1,trick_suit)
                scores1  = scores.copy()
                scores1[winner1] += hand_to_score(board1)
                #
                board2   = pd.Series([None]*4)
                for i in range(4):
                    board2[i]  = hand[hand!=card].iloc[0] if i==act else hands[i].iloc[0]
                winner2  = trick_winner(board2,board2[winner1][0])
                scores1[winner2] += hand_to_score(board2)
                if scores1.max() == 26:
                    scores1[scores1.idxmax()]  = -26
                    scores1  += 26
                results.append((scores1,pd.concat([board1,board2],1,keys=(12,13,))))
            return results
        else:
            # This move will end the trick
            results  = []
            for card in playable_hand:
                # Consider each possible move
                hands1   = hands.copy()
                hands1[act] = hand[hand!=card].copy()
                board1   = board.copy()
                board1[act]  = card
                winner1  = trick_winner(board1,trick_suit)
                scores1  = scores.copy()
                scores1[winner1] += hand_to_score(board1)
                heart_broken1 = heart_broken or (board1.str[0]==4).any()
                res  = simulation_step(winner1,scores1,hands1,heart_broken1,trick+1,winner1,0,pd.Series([None]*4,name=trick+1))
                res  = [(x[0],pd.concat([board1,x[1]],1)) for x in res]
                results += res
            return results
    elif trick_suit == 0 and hand_to_score(pd.concat(hands.tolist())) == 0:
        # There are no more point cards to win
        scores1  = scores.copy()
        if scores1.max() == 26:
            scores1[scores1.idxmax()]  = -26
            scores1  += 26
        return [(scores1,pd.Series([None,]*4,name=trick))]
    else:
        # This move continues a trick
        results  = []
        for card in playable_hand:
            # Consider each possible move
            hands1   = hands.copy()
            hands1[act] = hand[hand!=card].copy()
            board1   = board.copy()
            board1[act]  = card
            results += simulation_step((act + 1)%4,scores,hands1,heart_broken,trick,trick_lead,card[0] if trick_suit==0 else trick_suit,board1)
        return results

if __name__ == '__main__':
    t  = table(human_agent) #table() #
    #
    Nsim  = 10
    results = []
    for i in range(Nsim):
        t.new_game()
        t.deal()
        exit(0)
        while t.next_move():
            pass
        print(t.players[['name','score']])
        print()
        results.append(t.players[['name','score']].set_index('name',append=True).score.copy())
    results  = pd.concat(results,1,keys=range(Nsim)).T
    winners  = results.idxmin(1)
    print(winners.value_counts()/Nsim)
