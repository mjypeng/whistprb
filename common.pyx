# cython: language_level=3, profile=True, linetrace=True, binding=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

import pandas as pd
import numpy as np
from collections import defaultdict
import cachetools
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

#-- Console Output --#
suitcolor = {0:' ',1:'\033[30m♣',2:'\033[91m♦',3:'\033[30m♠',4:'\033[91m♥',}
ordermap  = {0:' ',2:'2\033[0m',3:'3\033[0m',4:'4\033[0m',5:'5\033[0m',6:'6\033[0m',7:'7\033[0m',8:'8\033[0m',9:'9\033[0m',10:'T\033[0m',11:'J\033[0m',12:'Q\033[0m',13:'K\033[0m',14:'A\033[0m'}
def card_to_console(card):
    return suitcolor[card[0]]+ordermap[card[1]]

def hand_to_console(hand):
    return ' '.join([card_to_console(x) for x in hand])

def print_state(state):
    """
    state = (act,scores,hands,heart_broken,trick,trick_lead,trick_suit,board)
        act: integer
        scores: 4x tuple of integers
        hands: 4x tuple of tuples of cards/2-tuples
        heart_broken: bool
        trick: integer
        trick_lead: integer
        trick_suit: integer
        board: 4x tuple of cards/2-tuples or None
    """
    act,scores,hands,heart_broken,trick,trick_lead,trick_suit,board  = state
    #
    print("Trick %d"%trick)
    for i in range(4):
        print("%d (%2d): %s[%s] [%s]" % (i,scores[i],'*' if i==trick_lead else ' ',card_to_console(board[i]) if board[i] else '  ',hand_to_console(hands[i])))

#-- Basic Game Rules --#
def card_rank(card,outs):
    # Rank of card if played as lead: 0-based
    return len([x for x in outs if x[0]==card[0] and x[1]>card[1]])

cdef int card_rank_c(int suit,int rank,int num_outs,int* outs):
    # Rank of card if played as lead: 0-based
    cdef int i
    cdef int c = 0
    for i in range(num_outs):
        if outs[2*i] == suit and outs[2*i+1] > rank:
            c += 1
    return c

cdef int card_rank_c_s(int rank,int num_outs,int* outs):
    # Rank of card if played as lead: 0-based
    cdef int i
    cdef int c = 0
    for i in range(num_outs):
        if outs[i] > rank:
            c += 1
    return c

def card_to_score(card):
    if card[0] == 4: return 1
    elif card == (3,12): return 13
    else: return 0

cdef int card_to_score_c(int suit,int rank):
    if suit == 4: return 1
    elif suit == 3 and rank == 12: return 13
    else: return 0

def hand_to_score(hand):
    score  = len([x for x in hand if x[0]==4])
    if (3,12) in hand: score += 13
    return score

cdef int board_score_c(int board[4][2]):
    cdef int i
    cdef int score = 0
    for i in range(4):
        if board[i][0] == 4: score += 1
        elif board[i][0]==3 and board[i][1]==12: score += 13
    return score

def legal_plays(hand,int heart_broken,int trick,int trick_suit):
    if trick_suit == 0:
        # Leading if trick_suit == 0
        if trick == 1:
            plays  = ((1,2),)
        elif heart_broken:
            plays  = hand
        else:
            plays  = tuple(x for x in hand if x[0]!=4)
            if not plays: plays = hand
    else:
        plays  = tuple(x for x in hand if x[0]==trick_suit)
        if not plays:
            if trick == 1:
                plays  = tuple(x for x in hand if card_to_score(x)==0)
                if not plays: plays = hand
            else:
                plays  = hand
    return plays

cdef int legal_plays_c(int *plays,int *hand,int heart_broken,int trick,int trick_suit):
    # Card i Suit: hand[2*i]
    # Card i Rank: hand[2*i+1]
    cdef int i,c  = 0
    if trick_suit == 0:
        # Leading if trick_suit == 0
        if trick == 1:
            plays[0]  = 1
            plays[1]  = 2
            c  = 1
        elif not heart_broken:
            for i in range(14-trick):
                if hand[2*i] != 4:
                    plays[2*c],plays[2*c+1]  = hand[2*i],hand[2*i+1]
                    c += 1
    else:
        for i in range(14-trick):
            if hand[2*i] == trick_suit:
                plays[2*c],plays[2*c+1]  = hand[2*i],hand[2*i+1]
                c += 1
        if c == 0 and trick == 1:
            for i in range(14-trick):
                if card_to_score_c(hand[2*i],hand[2*i+1]) == 0:
                    plays[2*c],plays[2*c+1]  = hand[2*i],hand[2*i+1]
                    c += 1
    if c:
        return c
    else:
        for i in range(14-trick):
            plays[2*i],plays[2*i+1]  = hand[2*i],hand[2*i+1]
        return 14 - trick

def trick_winner(board,int suit):
    cdef int i
    cdef int max_rank  = 1
    for i in range(4):
        if board[i][0]==suit and board[i][1]>max_rank:
            max_rank = board[i][1]
            winner   = i
    return winner

cdef int trick_winner_c(int board[4][2],int suit):
    cdef int i,winner
    cdef int max_rank  = 1
    for i in range(4):
        if board[i][0]==suit and board[i][1]>max_rank:
            max_rank = board[i][1]
            winner   = i
    return winner

#-- Game States --#
def new_deck(shuffle=True):
    deck  = tuple((suit,rank) for suit in range(1,5) for rank in range(2,15))
    if shuffle:
        deck  = tuple(tuple(x) for x in np.random.permutation(deck))
    return deck

def initial_state(deck=None):
    if deck is None: deck = new_deck(shuffle=True)
    hands  = tuple(tuple(sorted(deck[i*13:(i+1)*13])) for i in range(4))
    for i in range(4):
        if (1,2) == hands[i][0]:
            act  = i
            break
    scores       = (0,)*4
    heart_broken = False
    trick        = 1
    trick_lead   = act
    trick_suit   = 0
    board        = ((0,0,),)*4
    #
    state  = (act,scores,hands,heart_broken,trick,trick_lead,trick_suit,board)
    return state

def successor(state,int prune_equiv=True):
    """
    state = (act,scores,hands,heart_broken,trick,trick_lead,trick_suit,board)
        act: integer
        scores: 4x tuple of integers
        hands: 4x tuple of tuples of cards/2-tuples
        heart_broken: bool
        trick: integer
        trick_lead: integer
        trick_suit: integer
        board: 4x tuple of cards/2-tuples or None
    """
    cdef int i,j,c,act,heart_broken,trick,trick_lead,trick_suit
    cdef int board1_[4][2]
    cdef int *hand_
    cdef int num_plays
    cdef int *plays_
    cdef int num_outs[4]
    cdef int *outs_[4]
    cdef int suit1
    #
    act,scores,hands,heart_broken,trick,trick_lead,trick_suit,board  = state
    if act < 0: return [] # Game over, no successor states
    #
    hand_   = <int *>PyMem_Malloc(2*(14-trick)*sizeof(int))
    plays_  = <int *>PyMem_Malloc(2*(14-trick)*sizeof(int))
    for i in range(14-trick):
        hand_[2*i],hand_[2*i+1]  = hands[act][i][0],hands[act][i][1]
    #
    #-- Determine legal moves --#
    num_plays  = legal_plays_c(plays_,hand_,heart_broken,trick,trick_suit)
    PyMem_Free(hand_)
    #
    #-- Prune equivalent plays --#
    if prune_equiv:
        for i in range(4):
            num_outs[i] = 0
            outs_[i]    = <int *>PyMem_Malloc((3*(14-trick)+4)*sizeof(int))
        for i in range(4):
            if i != act:
                for j in range(len(hands[i])):
                    suit1  = hands[i][j][0] - 1
                    outs_[suit1][num_outs[suit1]]  = hands[i][j][1]
                    num_outs[suit1] += 1
            if board[i][0]:
                suit1 = board[i][0] - 1
                outs_[suit1][num_outs[suit1]]  = board[i][1]
                num_outs[suit1] += 1
        #
        equiv_hash = set()
        c  = 0
        for i in range(num_plays):
            suit1 = plays_[2*i] - 1
            hc    = (suit1,card_rank_c_s(plays_[2*i+1],num_outs[suit1],outs_[suit1]),card_to_score_c(plays_[2*i],plays_[2*i+1]),)
            if hc not in equiv_hash:
                equiv_hash.add(hc)
                plays_[2*c],plays_[2*c+1]  = plays_[2*i],plays_[2*i+1]
                c += 1
        num_plays  = c
        for i in range(4):
            PyMem_Free(outs_[i])
    #
    #-- Next to act --#
    hands1  = list(hands)
    successors  = []
    act1  = (act + 1)%4
    if act1 == trick_lead:
        board1_  = board
        if trick == 13:
            # This move will end the round
            for i in range(num_plays):
                hand  = list(hands[act])
                hand.remove((plays_[2*i],plays_[2*i+1]))
                hands1[act]  = tuple(hand)
                board1_[act] = plays_[2*i],plays_[2*i+1]
                winner1  = trick_winner_c(board1_,trick_suit)
                scores1  = list(scores)
                scores1[winner1] += board_score_c(board1_)
                if max(scores1) == 26:
                    # Some player Shot the Moon
                    scores1  = tuple(0 if x==26 else 26 for x in scores1)
                heart_broken1  = heart_broken or any([board1_[i][0]==4 for i in range(4)])
                state1   = (-1,tuple(scores1),tuple(hands1),heart_broken1,-1,-1,0,((0,0,),)*4,)
                successors.append(((plays_[2*i],plays_[2*i+1]),state1))
        else:
            # This move will end the trick
            for i in range(num_plays):
                hand  = list(hands[act])
                hand.remove((plays_[2*i],plays_[2*i+1]))
                hands1[act]  = tuple(hand)
                board1_[act] = plays_[2*i],plays_[2*i+1]
                winner1  = trick_winner_c(board1_,trick_suit)
                scores1  = list(scores)
                scores1[winner1] += board_score_c(board1_)
                heart_broken1  = heart_broken or any([board1_[i][0]==4 for i in range(4)])
                state1   = (winner1,tuple(scores1),tuple(hands1),heart_broken1,trick + 1,winner1,0,((0,0,),)*4,)
                successors.append(((plays_[2*i],plays_[2*i+1]),state1))
    else:
        # This move continues a trick
        board1  = list(board)
        for i in range(num_plays):
            hand  = list(hands[act])
            hand.remove((plays_[2*i],plays_[2*i+1]))
            hands1[act]  = tuple(hand)
            board1[act]  = plays_[2*i],plays_[2*i+1]
            state1   = (act1,scores,tuple(hands1),heart_broken,trick,trick_lead,plays_[2*i] if trick_suit==0 else trick_suit,tuple(board1))
            successors.append(((plays_[2*i],plays_[2*i+1]),state1))
    PyMem_Free(plays_)
    return successors

def next_state(state,play=None):
    """
    play = None: Random
    play = card = (suit,rank)
    """
    successors  = successor(state,prune_equiv=False)
    if play:
        for card,state1 in successors:
            if card==play: return state1
        print("warning:illegal play")
        return state
    else:
        return successors[np.random.choice(len(successors))][1] if successors else None

def state_to_info(state):
    """
    Convert game state to information set (w.r.t. acting player).
    state = (act,scores,hands,heart_broken,trick,trick_lead,trick_suit,board)
        act: integer
        scores: 4x tuple of integers
        hands: 4x tuple of tuples of cards/2-tuples
        heart_broken: bool
        trick: integer
        trick_lead: integer
        trick_suit: integer
        board: 4x tuple of cards/2-tuples or None
    """
    act,scores,hands,heart_broken,trick,trick_lead,trick_suit,board = state
    hand  = hands[act]
    outs  = []
    for i in range(4):
        if i != act: outs += hands[i]
    # outs += [x for x in board if x[0]]
    outs.sort()
    plays = legal_plays(hand,heart_broken,trick,trick_suit)
    info  = (act,scores,plays,hand,outs,heart_broken,trick,trick_lead,trick_suit,board)
    return info

#-- Simulation --#
def simulation(state,goals='min',terminal='round_end',return_path=False,prune=False):
    sim_counts = defaultdict(int)
    results    = simulation_step(state,goals=goals,terminal=terminal,return_path=return_path,top=True,counts=sim_counts,prune=prune)
    return results
    #
    # act,scores,hands,heart_broken,trick,trick_lead,trick_suit,board  = state
    # lead     = pd.Series(['',]*4,name='lead')
    # lead[trick_lead] = '*'
    # results  = [pd.concat([lead,pd.concat([pd.Series(y) for y in x[2]],1,keys=range(trick,trick+len(x[2]))),pd.Series(x[0],name='score')],1) for x in results]
    # plays    = pd.DataFrame([(x.loc[state[0],state[4]],tuple(x.score)) for x in results],columns=('play','score')).drop_duplicates().sort_values('play')
    # plays['play']  = plays.play.apply(card_to_console)
    # return plays,results,sim_counts

memoized_states  = cachetools.LRUCache(maxsize=1000000)
def reset_memoized_states():
    global memoized_states
    while len(memoized_states): _ = memoized_states.popitem()

def get_memoized_states():
    global memoized_states
    return memoized_states

    # cdef int i,j,act,heart_broken,trick,trick_lead,trick_suit
    # act,_scores,_hands,heart_broken,trick,trick_lead,trick_suit,_board  = state
    # cdef int scores[4],hands[4][14-trick][2],board[4][2]
    # for i in range(4):
    #     scores[i]   = _scores[i]
    #     board[i][0] = _board[i][0]
    #     board[i][1] = _board[i][1]
    #     for j in range(2):

def simulation_step_key(state,*args,counts=None,prune=False,**kwargs):
    # Ignore arguments 'counts' and 'prune'
    return cachetools.keys.hashkey(*state,*args,**kwargs)

@cachetools.cached(memoized_states,key=simulation_step_key,lock=None)
def simulation_step(state,goals='min',terminal='round_end',int return_path=False,int top=True,counts=None,int prune=False):
    """
    state = (act,scores,hands,heart_broken,trick,trick_lead,trick_suit,board)
        act: integer
        scores: 4x tuple of integers
        hands: 4x tuple of tuples of cards/2-tuples
        heart_broken: bool
        trick: integer
        trick_lead: integer
        trick_suit: integer
        board: 4x tuple of cards/2-tuples or None
    counts: please supply a defaultdict(int)
    """
    cdef int i,act,heart_broken,trick,trick_lead,trick_suit
    cdef int board_score1,winner1,moonshot
    cdef int board1_[4][2]
    cdef int scores1_[4]
    act,scores,hands,heart_broken,trick,trick_lead,trick_suit,board  = state
    # global memoized_states
    # if len(memoized_states)%100 == 0:
    # print("trick: %d, memoized_states: %d"%(trick,len(memoized_states)),flush=True)
    #
    if not top and terminal == 'score_13' and any([x>=13 for x in scores]):
        counts['terminal__score_13'] += 1
        return [((0,0,),scores,(),(),)] if return_path else [(0,0,),(scores,)]
    #
    if not top and trick == 13:
        counts['terminal__trick_13'] += 1
        board1_  = board
        for i in range(4):
            if not board1_[i][0]: board1_[i] = hands[i][0]
        board_score1  = board_score_c(board1_)
        scores1_      = scores
        # scores1       = list(scores)
        if board_score1:
            winner1   = trick_winner_c(board1_,board1_[trick_lead][0])
            scores1_[winner1] += board_score1
        moonshot  = False
        for i in range(4):
            if scores1_[i] == 26:
                moonshot  = True
                break
        if moonshot:
            for i in range(4):
                if scores1_[i] == 26: scores1_[i]  = 0
                else: scores1_[i]  = 26
        # if max(scores1) == 26:
        #     # Some player Shot the Moon
        #     scores1  = tuple(0 if x==26 else 26 for x in scores1)
        if return_path:
            board1  = tuple((board1_[i][0],board1_[i][1]) for i in range(4))
            return [((0,0,),tuple(scores1_),(trick_lead,),(board1,))]
        else:
            return [((0,0,),tuple(scores1_),)]
    else:
        successors  = successor(state)
        if successors:
            results  = [] # List of final game results for each possible action
            if (act + 1)%4 == trick_lead:
                counts['expand__trick_end'] += 1
                if return_path: board1 = list(board)
                for card,state1 in successors:
                    res  = simulation_step(state1,goals=goals,terminal=terminal,return_path=return_path,top=False,counts=counts,prune=prune)
                    if return_path:
                        board1[act]  = card
                        res  = [(card,x[1],(trick_lead,)+x[2],(tuple(board1),)+x[3]) for x in res]
                    else:
                        res  = [(card,x[1]) for x in res]
                    results += res
            else:
                counts['expand__trick_continue'] += 1
                for card,state1 in successors:
                    res  = simulation_step(state1,goals=goals,terminal=terminal,return_path=return_path,top=False,counts=counts,prune=prune)
                    res  = [(card,)+x[1:] for x in res]
                    results += res
            #
            if not top:
                # Pick only paths that leads to optimal results for acting player
                if goals == 'min':
                    results_  = []
                    for card,_ in successors:
                        card_max_score  = max([x[1][act] for x in results if x[0]==card])
                        results_  += [x for x in results if x[0]==card and x[1][act]==card_max_score]
                    min_score  = min([x[1][act] for x in results_])
                    results    = [x for x in results_ if x[1][act]==min_score]
                elif goals == 'max':
                    results_  = []
                    for card,_ in successors:
                        card_min_score  = min([x[1][act] for x in results if x[0]==card])
                        results_  += [x for x in results if x[0]==card and x[1][act]==card_min_score]
                    max_score  = max([x[1][act] for x in results_])
                    results    = [x for x in results_ if x[1][act]==max_score]
            #
            return results
