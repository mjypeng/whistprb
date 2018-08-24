from common import *

#-- Agents --#
def placeholder_agent(info):
    """
    info  = (act,scores,plays,hand,outs,heart_broken,trick,trick_lead,trick_suit,board)
    """
    act,scores,plays,hand,outs,heart_broken,trick,trick_lead,trick_suit,board = info
    play  = plays[0]
    print('Placeholder Play',hand_to_console(plays),'=>',card_to_console(play))
    print()
    return play

def random_agent(info):
    """
    info  = (act,scores,plays,hand,outs,heart_broken,trick,trick_lead,trick_suit,board)
    """
    act,scores,plays,hand,outs,heart_broken,trick,trick_lead,trick_suit,board = info
    play  = np.random.choice(pd.Series(plays))
    print('Random Play',hand_to_console(plays),'=>',card_to_console(play))
    print()
    return play

def rule_based_agent(info):
    act,scores,plays,hand,outs,heart_broken,trick,trick_lead,trick_suit,board = info
    #
    plays  = pd.DataFrame(pd.Series(plays),columns=('card',))
    outs  += tuple(x for x in board if x[0])
    #
    plays['score']      = plays.card.apply(card_to_score)
    plays['outs_rank']  = plays.card.apply(lambda x:card_rank(x,outs))
    plays['board_rank'] = plays.card.apply(lambda x:card_rank(x,board) if x[0]==trick_suit else 1) if trick_suit>0 else 0
    #
    if (plays.board_rank>0).any():
        # There exists some legal play that is guaranteed to lose this trick
        mask  = plays.board_rank>0
        # Play the card with highest score then highest rank (w.r.t. outs)
        play  = plays[mask].sort_values(by=['score','outs_rank','card'],ascending=[False,True,False]).iloc[0].card
    elif len([x for x in board if x[0]]) == 3:
        # Three players has played so we are guaranteed to win this trick
        # Play the card with lowest score then highest rank (w.r.t. outs)
        play  = plays.sort_values(by=['score','outs_rank','card'],ascending=[True,True,False]).iloc[0].card
    else:
        # Leading or There are no legal play guaranteed to lose this trick
        # Play the card with lowest rank (w.r.t. outs)
        play  = plays.sort_values(by=['outs_rank','card'],ascending=[False,True]).iloc[0].card
    #
    print('Rule Based Play',hand_to_console(plays.card),'=>',card_to_console(play))
    print()
    return play

def human_agent(info):
    act,scores,plays,hand,outs,heart_broken,trick,trick_lead,trick_suit,board = info
    #
    hand   = pd.DataFrame(pd.Series(hand),columns=('card',))
    outs  += tuple(x for x in board if x[0])
    out_suits  = [len(outs),0,0,0,0]
    for i in range(1,5):
        out_suits[i]  = (outs.str[0]==i).sum()
    #
    hand['score']      = hand.card.apply(card_to_score)
    hand['outs_rank']  = hand.card.apply(lambda x:card_rank(x,outs))
    hand['board_rank'] = hand.card.apply(lambda x:card_rank(x,board) if x[0]==trick_suit else 1) if trick_suit>0 else 0
    hand['play']       = 0
    i  = 1
    for idx,row in hand.iterrows():
        if row.card in plays:
            hand.loc[idx,'play']  = i
            i  += 1
    #
    print('Board:')
    print(("%-2d   "%state['trick']) + hand_to_console(board).replace(' ','  '))
    #
    print('Choose Card to Play:')
    print('       '+hand_to_console(hand.card).replace(' ','  '),end='')
    print('    '+suitcolor[1]+("\033[0m:%-2d"%out_suits[1])+'  '+suitcolor[2]+("\033[0m:%-2d"%out_suits[2]))
    print('outs ',*["%3d"%i for i in hand.outs_rank],end='')
    print('    '+suitcolor[3]+("\033[0m:%-2d"%out_suits[3])+'  '+suitcolor[4]+("\033[0m:%-2d"%out_suits[4]))
    if trick_suit > 0:
        print('board',*["%3d"%i if i>=0 else '   ' for i in hand.board_rank])
    print('play ',*["%3d"%i if i>0 else '   ' for i in hand.play])
    try:
        play  = input()
        if play.lower() == 'q':
            exit(0)
        else:
            play = int(play)
    except: play = 0
    return hand.loc[(hand.play==play).idxmax(),'card']

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
        self.deck   = new_deck(shuffle=True)
        self.state  = initial_state(deck=self.deck)
        self.round_id += 1
        #
        self.print_state()
    #
    def next_move(self,override_play=None):
        if override_play is None:
            info  = state_to_info(self.state)
            play  = self.agents[self.state[0]](info)
            self.state = next_state(self.state,play)
        else:
            self.state = next_state(self.state,override_play)
        #
        self.print_state()
        if self.state[0] < 0:
            # Round End
            self.players.score  += self.state[1]
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
        print("Round %d, Scores = %s"%(self.round_id,str(tuple(self.players.score))))
        print_state(self.state)
        print()

if __name__ == '__main__':
    t  = table() #table(human_agent) #
    #
    Nsim  = 10
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
    winners  = results.idxmin(1)
    print(winners.value_counts()/Nsim)
