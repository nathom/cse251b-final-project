
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import namedtuple, deque
from itertools import count
import math
import random

from DQN import DeepQNetwork
from rl.game_2048 import Game2048

'''
encode_state:
arguments: board - a  4x4 2D array of the game board
returns a one-hot encoded array of the state of the board
'''
def encode_state(board):
  board_flat = [0 if e == 0 else int(math.log(e, 2)) for e in board.flatten()]
  board_flat = torch.LongTensor(board_flat)
  board_flat = F.one_hot(board_flat, num_classes=16).float().flatten()
  #board_flat = board_flat.reshape(1, 4, 4, 16).permute(0, 3, 1, 2)
  return board_flat

'''
Create a replay memory buffer
'''
# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = list()
        self.capacity = capacity

    def push(self,current_state,best_action,next_state,reward,done):
        self.memory.append([current_state,best_action,next_state,reward,done])

        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

'''
trains the model over a batch from memory 
'''  
def model_train(model, memory, optimizer,criterion):
    np.random.shuffle(memory)
    batch_sample = memory[0:model.batch_size]
    # sample minibatch 
    
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    print ("device: ", device)

    #loss = 0
    #iterate over selected experiences in the batch
    for e in batch_sample:
        #with torch.no_grad():
        q_current = torch.max(model.forward(e[0]).detach()) #best action based on current state
        q_target = e[3]
        if not e[4]:
            #print(e[2])
            #print(q_temp)
            with torch.no_grad():
                fp = model.forward(e[2])
                best_action = torch.max(fp.detach())
                #print (fp, best_action)
                q_target = q_target + model.gamma*best_action
                # q_target = q_target + model.gamma*torch.max(model.forward(e[2]))
        
        
        q_target = torch.tensor(q_target, requires_grad=True)
        q_target = q_target.to(device)
        q_current = q_current.to(device)
        
        print("values :", q_current, q_target)
        loss = criterion(q_current,q_target)
        
        print(f"loss {loss}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


'''
Deep Q-learning with Experience Replay implementation
'''    
def train(n_episode, n_iteration, mem_capacity, game, model, optimizer,criterion):
    #DQ = DeepQNetwork(256,4)
    model.n_iterations = n_iteration
    model.n_episodes = n_episode

    memory = ReplayMemory(mem_capacity)
    total_steps = 0

    for i in range(model.n_episodes):
        # reset the game
        game.game_reset()
        print(f"Starting game {i}")
        # initialize the firat state
        current_state = encode_state(game.matrix) 
        # phi_t = encoded(state)
        # encoding the values of the tiles in the game board
        print("initial board pos: ", game.matrix)
        for j in range(model.n_iterations):
            total_steps+=1
            # Chossing and executing the best action
            prev_ms = game.merge_score
            best_action  = model.execute_action(current_state)
            # a_t
            game.make_move(best_action)
            # execute action a_t and observe reward and next state

            # Storing experience
            next_state, reward, done = encode_state(game.matrix),(game.merge_score-prev_ms),game.game_end
            # phi_t+1, r_t, done 
            memory.push(current_state,best_action,next_state,reward,done)
            # phi_t, a_t, r_t, phi_t+1, done
            # Update the explorarion if episode is done
            model.update_epsilon()
            current_state = next_state
                        
            if done:
                print("==========================================")
                print(f"merge score: {game.merge_score}")
                print(f"final board pos: {game.matrix}")
                break

        if total_steps >= model.batch_size:
            model_train(model,memory.memory, optimizer,criterion)

        # print(f"merge score: {game.merge_score}")
        # print(f"final board pos: {game.matrix}")#

def Q_run():
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    game = Game2048()
    batch_size =10
    model = DeepQNetwork(batch_size=batch_size).to(device)
    model.lr = 0.001
    optimizer = optim.SGD(model.parameters(), lr=model.lr, momentum=0.9)
    criterion  = nn.MSELoss().to(device)
    # criterion.requires_grad = True
    n_ep, n_iter = 50 ,100
    Checkpoint = "DQN_weights"
    train(n_ep,n_iter,10000,game,model,optimizer,criterion)

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')

    print('=======>Saving..')
    torch.save({
    'episodes': n_ep ,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': criterion,
    }, './checkpoint/' + Checkpoint)
    
    


if __name__=="__main__": 
    Q_run()
    

    







    