# Training the AI

import torch
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, params, shared_model, optimizer):
    torch.manual_seed(params.seed + rank)
    env = create_atari_env(params.env_name, False, rank)
    env.reset(seed=params.seed + rank)
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    
    state,_ = env.reset()
    state = torch.from_numpy(state)
    
    last_state= None
    model_save_path = f"model_rank_{rank}.pth"
    
    done = True
    episode_length = 0
    while True:
        episode_length += 1
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
            lastcx = cx
            lasthx = hx
            if rank == 0: 
                torch.save(shared_model.state_dict(), model_save_path)
                print(f"Model saved to {model_save_path}")
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)
            lastcx = cx
            lasthx = hx
            
        values = []
        log_probs = []
        rewards = []
        entropies = []
        
        for step in range(params.num_steps):
            
            value, action_values, (hx, cx) = model((state, (hx, cx)))
            temperature = 0.1 
            prob = F.softmax(action_values / temperature, dim=1)
            log_prob = F.log_softmax(action_values, dim=1)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)
            action = prob.multinomial(num_samples=1).data
            log_prob = log_prob.gather(1, Variable(action))
            values.append(value)
            log_probs.append(log_prob)
            state, reward, done, truncated, info = env.step(action.item())
            done = (done or episode_length >= params.max_episode_length)
            reward = max(min(reward, 1), -1)
            
            # if step%5==0 and rank == 0 :
            #     print(f"\nEpisode_length {episode_length}: Action probabilities: {prob.detach().numpy()}, MaxPro : {prob.max(1)[1].data.numpy()[0]}, Selected action: {action}")
            #     state2 = torch.randn(1, 1, 42, 42)  # Example of a random state
            #     with torch.no_grad():
            #         value2, action_values2, _ = model((state2, (lasthx, lastcx)))
            #     prob2 = F.softmax(action_values2/temperature, dim=1)
            #     print(f"Episode_length {episode_length}: Action probabilities: {prob2.detach().numpy()}, MaxPro: {prob2.max(1)[1].data.numpy()[0]}")


            
            if done:
                episode_length = 0
                state,_ = env.reset()
            state = torch.from_numpy(state)

            # if reward != 0:
            #     episode_length = 0
            #     reward = 1

            rewards.append(reward)
            if done:
                break
        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state, (hx, cx)))
            R = value.data
        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = params.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            TD = rewards[i] + params.gamma * values[i + 1].data - values[i].data
            gae = gae * params.gamma * params.tau + TD
            entropy_coef = params.entropy_coef
            policy_loss = policy_loss - log_probs[i] * Variable(gae) - entropy_coef * entropies[i]

        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        
        # if rank == 0:
        #     for name, param in model.named_parameters():
        #         if param.grad is not None:
        #             print(f"{name} grad norm: {param.grad.norm().item()} ({param.grad.mean().item()})")
        #         else:
        #             print(f"{name} has no gradient. Check if it is detached.")


        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        ensure_shared_grads(model, shared_model)
        optimizer.step()
        
