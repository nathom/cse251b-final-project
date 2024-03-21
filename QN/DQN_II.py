import argparse
import os
import random

# add the path to the game file
import sys
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

# append the parent directory to the path
sys.path.append("..")

from DQN import *

from rl.game_2048 import Game2048

device = None


"""
Create a replay memory buffer
"""
# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = list()
        self.capacity = capacity

    def push(self, current_state, best_action, next_state, reward, done):
        self.memory.append([current_state, best_action, next_state, reward, done])

        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


"""
trains the model over a batch from memory 
"""


def model_train(policy, target, memory, optimizer, criterion):
    np.random.shuffle(memory)
    batch_sample = memory[0 : policy.batch_size]
    # sample minibatch
    # print ("---------------------------------")
    # # print all the actions in the batch
    # print ([e[1] for e in batch_sample])
    # print ("---------------------------------")

    # print (batch_sample[0][0].shape)
    state_batch = torch.stack([e[0] for e in batch_sample]).to(device)
    # from [batch_size, 1, 4, 4, 16] to [batch_size, 4, 4, 16]
    state_batch = state_batch.squeeze(1)
    next_state_batch = torch.stack([e[2] for e in batch_sample]).to(device)
    next_state_batch = next_state_batch.squeeze(1)
    reward_batch = torch.tensor([e[3] for e in batch_sample], dtype=torch.float32).to(
        device
    )
    done_batch = torch.tensor([e[4] for e in batch_sample], dtype=torch.bool).to(device)
    action_batch = torch.tensor([e[1] for e in batch_sample], dtype=torch.int64).to(
        device
    )

    # print (f"state_batch: {state_batch.size()} next_state_batch: {next_state_batch.size()} reward_batch: {reward_batch.size()} done_batch: {done_batch.size()} action_batch: {action_batch.size()}")
    q_current = policy.forward(state_batch)
    # get the q values for the actions taken
    q_current = q_current.gather(1, action_batch.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target.forward(next_state_batch).max(1)[0]
        next_q_values[done_batch] = 0.0

    q_target = reward_batch + policy.gamma * next_q_values

    # print (f"q_current: {q_current.shape} q_target: {q_target.shape}")

    loss = criterion(q_current, q_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

    # loss_value = 0
    # #iterate over selected experiences in the batch
    # i = 0
    # for e in batch_sample:
    #     #with torch.no_grad():
    #     q_current = torch.max(policy.forward(e[0]).detach()) #best action based on current state
    #     q_target = e[3]
    #     if not e[4]:
    #         #print(e[2])
    #         #print(q_temp)
    #         with torch.no_grad():
    #             fp = target.forward(e[2])
    #             best_action = torch.max(fp.detach())
    #             #print (fp, best_action)
    #             q_target = q_target + target.gamma*best_action
    #             # q_target = q_target + model.gamma*torch.max(model.forward(e[2]))

    #     q_target = q_target * 1.0
    #     print(f"q_current: {q_current} q_target: {q_target}")
    #     q_target = torch.tensor(q_target, requires_grad=True, dtype=q_current.dtype)
    #     q_target = q_target.to(device)
    #     q_current = q_current.to(device)

    #     # print("values :", q_current, q_target)
    #     loss = criterion(q_current,q_target)
    #     loss_value += loss.item()
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    # print(f"batch  loss {loss_value}")


"""
Deep Q-learning with Experience Replay implementation
"""


def train(
    n_episode, n_iteration, mem_capacity, game, policy, target, optimizer, criterion
):
    # DQ = DeepQNetwork(256,4)
    policy.n_iterations = n_iteration
    policy.n_episodes = n_episode
    total_steps = 0
    memory = ReplayMemory(mem_capacity)
    losses = []
    eps = []
    merges = []
    for i in range(policy.n_episodes):
        # reset the game
        print(f"Starting Episode {i}")
        game.game_reset()
        # initialize the firat state
        current_state = encode_state(game.matrix).to(device)
        # phi_t = encoded(state)
        # encoding the values of the tiles in the game board
        # print("initial board pos: \n", game.matrix)
        p_cnt = 0  # a counter to keep tarck if the model is not making valid moves for a while, if so then break it
        ep_steps = 0
        while True:
            total_steps += 1
            ep_steps += 1
            # Chossing and executing the best action
            prev_ms = game.merge_score
            best_action = policy.execute_action(current_state.to(device))
            # a_t
            game.make_move(best_action)
            # execute action a_t and observe reward and next state

            # Storing experience
            next_state, reward, done = (
                encode_state(game.matrix).to(device),
                (game.merge_score - prev_ms),
                game.game_end,
            )

            if not done and torch.eq(current_state, next_state).all():
                p_cnt += 1
                reward = (
                    -30.0
                )  # penalize the model for making a move that does not change the state
                if p_cnt > 20:
                    print(
                        f"Model made {p_cnt -1} invalid moves in a row -- total steps: {ep_steps} -- breaking the episode"
                    )
                    print(f"merge score: {game.merge_score}")
                    print(f"final board pos: \n{game.matrix}")
                    eps.append(ep_steps)
                    merges.append(game.merge_score)
                    break
            else:
                p_cnt = 0

            # phi_t+1, r_t, done
            memory.push(current_state, best_action, next_state, reward, done)
            # phi_t, a_t, r_t, phi_t+1, done
            # Update the explorarion if episode is done
            policy.update_epsilon()
            current_state = next_state

            if done:
                eps.append(ep_steps)
                merges.append(game.merge_score)
                print(" ================ Game Over ================ ")
                print(f"merge score: {game.merge_score}")
                print(f"final board pos: \n{game.matrix}")
                break

        if total_steps >= policy.batch_size:
            # print("=========Training the model=========")
            ls = 0
            cnt = 100

            # using tqdm to show the progress bar
            for _ in tqdm.tqdm(range(cnt)):
                ls += model_train(policy, target, memory.memory, optimizer, criterion)
            print(f"Episode {i} total loss: {ls/cnt}")
            losses.append(ls / cnt)
            # print("=========Training done=========")
        else:
            print(f"\n\n Episode {i} not trained -- not enough data\n\n")

        if i % 20 == 0 and i != 0:
            print("\n\n=========Updating target network========= \n")
            update_target_network(target, policy)
            # print(f"Episode: {i} done")
            print(f"epsilon: {target.epsilon}")
            # print("==========================================\n\n")

        # print(f"merge score: {game.merge_score}")
        # print(f"final board pos: {game.matrix}")#
    return losses, eps, merges


def update_target_network(target, policy):
    target.load_state_dict(policy.state_dict())
    target.epsilon = policy.epsilon

    # print(f"merge score: {game.merge_score}")
    # print(f"final board pos: {game.matrix}")#


def Q_run():
    print("device: ", device)

    game = Game2048()
    batch_size = 32
    policy = DeepQNetwork(batch_size=batch_size).to(device)
    target = DeepQNetwork(batch_size=batch_size).to(device)
    policy.lr = 5e-5
    # optimizer = optim.SGD(model.parameters(), lr=model.lr, momentum=0.9)
    optimizer = optim.Adam(policy.parameters(), lr=policy.lr)
    criterion = nn.MSELoss().to(device)
    # criterion.requires_grad = True
    n_ep, n_iter = 1000, 500
    Checkpoint = "DQN_weights"

    target.load_state_dict(policy.state_dict())

    losses, eps, merges = train(
        n_ep, n_iter, 50000, game, policy, target, optimizer, criterion
    )

    if not os.path.isdir("checkpoint"):
        os.mkdir("checkpoint")

    print("=======>Saving..")
    # torch.save({
    # 'episodes': n_ep ,
    # 'model_state_dict': policy.state_dict(),
    # 'optimizer_state_dict': optimizer.state_dict(),
    # 'loss': criterion,
    # }, './checkpoint/' + Checkpoint)
    torch.save(target.state_dict(), "checkpoint/target_net.pth")

    # plot the losses
    plot_losses(losses, f"losses_{n_ep}_{policy.lr}")
    plot_eps(eps, f"eps_{n_ep}_{policy.lr}")
    plot_merges(merges, f"merges_{n_ep}_{policy.lr}")


def plot_losses(train_losses, fname):
    """
    Plots the training and validation losses across epochs and saves the plot as an image file with name - fname(function argument).

    Args:
    ----
    train_losses (list): List of training losses for each epoch.
    val_losses (list): List of validation losses for each epoch.
    fname (str): Name of the file to save the plot (without extension).

    Returns:
    -------
    None
    """

    # Create 'plots' directory if it doesn't exist
    if not os.path.isdir("plots"):
        os.mkdir("plots")

    if not os.path.isdir("plots/losses"):
        os.mkdir("plots/losses")

    # # added this
    # train_losses = [t.cpu().detach().numpy() for t in train_losses]
    # val_losses = [t.cpu().detach().numpy() for t in val_losses]

    # Plotting training and validation losses
    plt.plot(train_losses, label="Training Loss")
    # plt.plot(val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    plt.ylabel("Loss")
    plt.title("Loss per Episode")
    plt.legend()

    # Saving the plot as an image file in 'plots' directory
    plt.savefig("./plots/losses/" + fname + ".png")
    plt.savefig("./plots/losses/" + fname + ".svg")
    plt.close()


def plot_eps(eps, fname):
    """
    Plots the training and validation losses across epochs and saves the plot as an image file with name - fname(function argument).

    Args:
    ----
    train_losses (list): List of training losses for each epoch.
    val_losses (list): List of validation losses for each epoch.
    fname (str): Name of the file to save the plot (without extension).

    Returns:
    -------
    None
    """

    if not os.path.isdir("plots"):
        os.mkdir("plots")
    # Create 'plots' directory if it doesn't exist
    if not os.path.isdir("plots/eps"):
        os.mkdir("plots/eps")

    # # added this
    # train_losses = [t.cpu().detach().numpy() for t in train_losses]
    # val_losses = [t.cpu().detach().numpy() for t in val_losses]

    # Plotting training and validation losses
    plt.plot(eps, label="Episode Length")
    # plt.plot(val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    plt.ylabel("Episode Length")
    plt.title("Episode Length")
    plt.legend()

    # Saving the plot as an image file in 'plots' directory
    plt.savefig("./plots/eps/" + fname + ".png")
    plt.savefig("./plots/eps/" + fname + ".svg")
    plt.close()


def plot_merges(merges, fname):
    if not os.path.isdir("plots"):
        os.mkdir("plots")
    # Create 'plots' directory if it doesn't exist
    if not os.path.isdir("plots/merges"):
        os.mkdir("plots/merges")

    # # added this
    # train_losses = [t.cpu().detach().numpy() for t in train_losses]
    # val_losses = [t.cpu().detach().numpy() for t in val_losses]

    # Plotting training and validation losses
    plt.plot(merges, label="Merge Score")
    # plt.plot(val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    plt.ylabel("Merge Score")
    plt.title("Merge Score")
    plt.legend()

    # Saving the plot as an image file in 'plots' directory
    plt.savefig("./plots/merges/" + fname + ".png")
    plt.savefig("./plots/merges/" + fname + ".svg")
    plt.close()


if __name__ == "__main__":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_play", help="whether to play using DQN or train it")
    args = parser.parse_args()

    if args.train_play == "train":
        Q_run()

    # game = Game2048()
    batch_size = 32
    Q = DeepQNetwork(batch_size=batch_size).to(device)
    # Q.load_state_dict(torch.load("./checkpoint/target_net.pth"))

    num_trials = 1000
    max_val_results = [0] * num_trials
    total_sum_results = [0] * num_trials
    total_merge_score = [0] * num_trials
    # currently running the game once
    start_time = time.time()
    for i in range(num_trials):
        (
            max_val_results[i],
            total_sum_results[i],
            total_merge_score[i],
        ) = Q.play()
        print("############################################")
        print(f"max value result {i}: {max_val_results[i]}")
        print(f"total sum results {i}: {total_sum_results[i]}")
        print(f"total merge score {i}: {total_merge_score[i]}")

    end_time = time.time()

    total_sum_avg = sum(total_sum_results) / num_trials
    max_val_avg = sum(max_val_results) / num_trials
    total_merge_avg = sum(total_merge_score) / num_trials

    print("total sum avg: " + str(total_sum_avg))
    print("max val avg: " + str(max_val_avg))
    print("merge score avg: " + str(total_merge_avg))
    print()
    print("time taken: ", str(timedelta(seconds=(end_time - start_time))))
