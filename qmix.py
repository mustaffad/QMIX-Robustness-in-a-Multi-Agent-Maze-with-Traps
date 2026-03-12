from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from maze_env import MultiAgentMazeEnv

# QMIX networks
# Per agent recurrent Q-network
class AgentRNN(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, n_actions)
    #update hidden state
    def forward(self, obs: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(obs))
        h2 = self.gru(x, h)
        x_q = F.relu(h2)
        q = self.q(x_q)
        return q, h2

# Small MLP used by QMIX
# Generate weights and biases for the mixer
class HyperNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)

#Qmix mixing network
class MixerQMIX(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, embed_dim: int):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # Hyprnetworks generate mixing weights and biases
        self.hyper_w1 = HyperNetwork(state_dim, n_agents * embed_dim, embed_dim)
        self.hyper_b1 = HyperNetwork(state_dim, embed_dim, embed_dim)
        self.hyper_w2 = HyperNetwork(state_dim, embed_dim, embed_dim)
        self.hyper_b2 = HyperNetwork(state_dim, 1, embed_dim)

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        B = agent_qs.size(0)
        # generate weights and biases from state
        w1_raw = self.hyper_w1(state)
        w1 = torch.abs(w1_raw).view(B, self.n_agents, self.embed_dim)
        
        b1 = self.hyper_b1(state).view(B, 1, self.embed_dim)

        # convertion for multiplication
        qs_unsqueezed = agent_qs.unsqueeze(1)
        matmul_res = torch.bmm(qs_unsqueezed, w1)
        #hidden layer
        hidden = F.elu(matmul_res + b1)
        #second layer
        w2_raw = self.hyper_w2(state)
        w2 = torch.abs(w2_raw).view(B, self.embed_dim, 1)    
        b2 = self.hyper_b2(state).view(B, 1, 1)

        # output layer
        y_res = torch.bmm(hidden, w2)
        y = y_res + b2
        
        return y.view(B, 1)


# Replay bufer
Transition = namedtuple('Transition', ('obs', 'state', 'avail', 'actions', 'reward', 'next_obs',
                                       'next_state', 'next_avail', 'done', 'alive_mask'))

#Stores all episodes for QMIX training.
class EpisodeBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.episodes: List[List[Transition]] = []
    #add one full episode to the buffer
    def push_episode(self, ep: List[Transition]):
        curr_len = len(self.episodes)
        if curr_len >= self.capacity:
            self.episodes.pop(0)
        self.episodes.append(ep)
    #randomly sample full episode
    def sample(self, batch_size: int) -> List[List[Transition]]:
        n = len(self.episodes)
        k = batch_size
        if k > n:
            k = n
        return random.sample(self.episodes, k)

    def __len__(self):
        return len(self.episodes)

# QMIX learner
@dataclass
class QMIXConfig:
    #configs
    n_agents: int
    obs_dim: int
    state_dim: int
    n_actions: int = 5
    hidden_dim: int = 64
    mixer_embed_dim: int = 32
    gamma: float = 0.99
    lr: float = 5e-4
    batch_size_eps: int = 8
    target_update_interval: int = 200
    grad_clip_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 50

class QMIXLearner:
    def __init__(self, cfg: QMIXConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        # agent network
        self.agent = AgentRNN(cfg.obs_dim, cfg.hidden_dim, cfg.n_actions).to(device)
        self.agent_tgt = AgentRNN(cfg.obs_dim, cfg.hidden_dim, cfg.n_actions).to(device)
        self.agent_tgt.load_state_dict(self.agent.state_dict())
        # mixer network
        self.mixer = MixerQMIX(cfg.n_agents, cfg.state_dim, cfg.mixer_embed_dim).to(device)
        self.mixer_tgt = MixerQMIX(cfg.n_agents, cfg.state_dim, cfg.mixer_embed_dim).to(device)
        self.mixer_tgt.load_state_dict(self.mixer.state_dict())
        #optimizer update and count training steps
        params = list(self.agent.parameters()) + list(self.mixer.parameters())
        self.optim = torch.optim.Adam(params, lr=cfg.lr)
        self.train_step = 0

    # Choose actions for all agents with e-greedy policy.
    def select_actions(self, obs_batch: List[np.ndarray], avail_batch: List[np.ndarray], h: torch.Tensor,
                       epsilon: float) -> Tuple[List[int], torch.Tensor]:
        #fixed network size
        n_agents = self.cfg.n_agents
        #number of the real agents in the envionment
        actual_n = len(obs_batch) 

        # Pad obs/avail up to n_agents for the network 
        if actual_n < n_agents:
            dummy_obs = np.zeros_like(obs_batch[0], dtype=np.float32)
            dummy_avail = np.array([0, 0, 0, 0, 1], dtype=np.float32)  

            extra_needed = n_agents - actual_n
            obs_list = list(obs_batch)
            avail_list = list(avail_batch)
            
            for _ in range(extra_needed):
                 obs_list.append(dummy_obs.copy())
                 avail_list.append(dummy_avail.copy())
        else:
            obs_list = obs_batch
            avail_list = avail_batch
        #convert to torch and run RNN
        obs_t = torch.tensor(np.stack(obs_list), dtype=torch.float32, device=self.device)  
        q, h2 = self.agent(obs_t, h)  

        actions: List[int] = []

        # Actions for active agents
        for i in range(actual_n):
            avail = torch.tensor(avail_list[i], dtype=torch.float32, device=self.device)
            
            # mask invalid actions
            q_i = q[i] - (1 - avail) * 1e6           
 
            if random.random() < epsilon:
                # greedy random
                valid_idx = torch.nonzero(avail > 0.5, as_tuple=False).view(-1)
                count = len(valid_idx)
                if count == 0:
                    a = int(self.cfg.n_actions - 1) 
                else:
                    rand_idx = random.randrange(count)
                    a = int(valid_idx[rand_idx].item())
            else:
                # argmax
                a = int(torch.argmax(q_i).item())
            actions.append(a)

        # fake agents just stays
        diff = n_agents - actual_n
        for _ in range(diff):
            actions.append(self.cfg.n_actions - 1)

        return actions, h2

    #Run one full episode in the environment using e-greedy actions.
    def _rollout_episode(self, env: MultiAgentMazeEnv, epsilon: float, max_T: int) -> List[Transition]:
        ep: List[Transition] = []
        obs, state, avail = env.reset(k_agents=env.k)  
        n_agents = env.k
        # hidden states per agent
        h = torch.zeros((self.cfg.n_agents, self.cfg.hidden_dim), dtype=torch.float32, device=self.device)
        
        # alive masking
        alive_mask = np.zeros(self.cfg.n_agents, dtype=np.float32)
        alive_mask[:n_agents] = 1.0

        for t in range(max_T):
            actions, h = self.select_actions(obs, avail, h, epsilon)
            # first n_agents actions for environment
            env_actions = actions[:n_agents]
            next_obs, r, done, info, next_avail = env.step(env_actions)
            next_state = env.get_state()
            
            # DEBUG
            print(f"[STEP] eps={epsilon:.3f} env_t={env.t} reward={r:.4f} info={info}")
            
            # Build per agent alive mask for this step
            alive_now = np.zeros(self.cfg.n_agents, dtype=np.float32)
            alive_now[:n_agents] = np.array(env.alive, dtype=np.float32)

            # Pad obsservation, next observation, avail to max n agents to match the network size
            def pad_list(L, pad_item):
                L2 = list(L)
                while len(L2) < self.cfg.n_agents:
                    L2.append(pad_item.copy())
                return L2

            obs_full = pad_list(obs, np.zeros_like(obs[0]))
            next_obs_full = pad_list(next_obs, np.zeros_like(next_obs[0]))
            avail_full = pad_list(avail, np.array([0, 0, 0, 0, 1], dtype=np.float32))
            next_avail_full = pad_list(next_avail, np.array([0, 0, 0, 0, 1], dtype=np.float32))

            # Pad actions
            extra = self.cfg.n_agents - len(actions)
            actions_full = actions + [self.cfg.n_actions - 1] * extra
            #Store one step
            tr = Transition(obs=np.stack(obs_full),
                            state=state.copy(),
                            avail=np.stack(avail_full),
                            actions=np.array(actions_full, dtype=np.int64),
                            reward=np.array([r], dtype=np.float32),
                            next_obs=np.stack(next_obs_full),
                            next_state=next_state.copy(),
                            next_avail=np.stack(next_avail_full),
                            done=np.array([done], dtype=np.float32),
                            alive_mask=alive_now.copy())
            ep.append(tr)
            #move
            obs, state, avail = next_obs, next_state, next_avail
            if done:
                break
        return ep
    # Compute a QMIX training update using a sampled batch
    def train_on_batch(self, batch_eps: List[List[Transition]]):
        device = self.device
        n_agents = self.cfg.n_agents
        # Find the longest episode in the batch for padding
        max_T = 0
        for ep in batch_eps:
             l = len(ep)
             if l > max_T:
                 max_T = l

        # Stack episodes with padding on time dimension
        def stack_field(name, dtype=torch.float32):
            arrs = []
            # Extract this field from each timestep of the episode
            for ep in batch_eps:
                vals = [getattr(t, name) for t in ep]
                # Choose padding value depending on field type
                pad_val = None
                if name == "actions":
                    pad_val = np.array([self.cfg.n_actions - 1] * n_agents, dtype=np.int64)
                elif name == "done" or name == "reward":
                    pad_val = np.array([0.0], dtype=np.float32)
                elif name == "alive_mask":
                    pad_val = np.zeros(n_agents, dtype=np.float32)
                elif name == "state" or name == "next_state":
                     pad_val = np.zeros_like(vals[0], dtype=np.float32)
                else:
                    pad_val = np.zeros_like(vals[0], dtype=np.float32)
                # Pad shorter episodes to max_T
                while len(vals) < max_T:
                    vals.append(pad_val.copy())
                arrs.append(np.stack(vals))
            return torch.tensor(np.stack(arrs), dtype=dtype, device=device)
        # Build all the necessary tensors
        obs      = stack_field('obs')
        next_obs = stack_field('next_obs')
        state    = stack_field('state')
        next_state = stack_field('next_state')
        avail    = stack_field('avail')
        next_avail = stack_field('next_avail')
        actions  = stack_field('actions', dtype=torch.long)
        rewards  = stack_field('reward')
        dones    = stack_field('done')
        alive_mask = stack_field('alive_mask')
        #Shape
        B, T, _, _ = obs.shape 
        obs_dim = self.cfg.obs_dim
        n_actions = self.cfg.n_actions
        hidden_dim = self.cfg.hidden_dim

        # Flatten
        obs_flat = obs.view(B * T * n_agents, obs_dim)
        next_obs_flat = next_obs.view(B * T * n_agents, obs_dim)

        # RNN hidden states
        h = torch.zeros((B * n_agents, hidden_dim), dtype=torch.float32, device=device)
        q_list = []
        for t in range(T):
            o_t = obs[:, t].contiguous().view(B * n_agents, obs_dim)
            q_t, h = self.agent(o_t, h)
            q_list.append(q_t.view(B, n_agents, n_actions))
        q_tensor = torch.stack(q_list, dim=1)  # [B, T, n_agents, n_actions]

        # Target network
        h_tgt = torch.zeros((B * n_agents, hidden_dim), dtype=torch.float32, device=device)
        qn_list = []
        for t in range(T):
            o_tn = next_obs[:, t].contiguous().view(B * n_agents, obs_dim)
            q_tn, h_tgt = self.agent_tgt(o_tn, h_tgt)
            qn_list.append(q_tn.view(B, n_agents, n_actions))
        q_next_tensor = torch.stack(qn_list, dim=1)

        # Gather chosen action Q-values
        action_mask = F.one_hot(actions, num_classes=n_actions).float()
        chosen_q = torch.sum(q_tensor * action_mask, dim=-1)

        # Mask invalid actions for next Q
        q_next_masked = q_next_tensor - (1 - next_avail) * 1e6
        max_next_q, _ = torch.max(q_next_masked, dim=-1)

        # Apply alive mask
        chosen_q = chosen_q * alive_mask
        max_next_q = max_next_q * alive_mask

        # Mix using by Qmix
        Q_tot = self.mixer(chosen_q.view(B * T, n_agents), state.view(B * T, -1)).view(B, T, 1)
        Q_tot_next = self.mixer_tgt(max_next_q.view(B * T, n_agents), next_state.view(B * T, -1)).view(B, T, 1)
        #TD target
        targets = rewards + (1.0 - dones) * self.cfg.gamma * Q_tot_next
        #Loss and optimization
        diff = Q_tot - targets.detach()
        sq_diff = diff ** 2
        loss = sq_diff.mean()
        self.optim.zero_grad()
        loss.backward()
        # Gradient clipping
        nn.utils.clip_grad_norm_(list(self.agent.parameters()) + list(self.mixer.parameters()),
                                 self.cfg.grad_clip_norm)
        self.optim.step()
        #Target network update
        self.train_step += 1
        if self.train_step % self.cfg.target_update_interval == 0:
            self.agent_tgt.load_state_dict(self.agent.state_dict())
            self.mixer_tgt.load_state_dict(self.mixer.state_dict())

        return float(loss.item())

    def train(self, env: MultiAgentMazeEnv, episodes: int, buffer: EpisodeBuffer, max_T: int,
              epsilon_sched: Tuple[float, float, int]) -> List[float]:
        eps0, eps_min, eps_decay = epsilon_sched
        losses: List[float] = []

        for ep in range(episodes):
            # Epsilon for this episode
            fraction = ep / max(1, eps_decay)
            epsilon = max(eps_min, eps0 - (eps0 - eps_min) * fraction)

            # Roll out a single episode
            ep_transitions = self._rollout_episode(env, epsilon, max_T)
            buffer.push_episode(ep_transitions)

            # Episode data
            ep_return = 0.0
            for tr in ep_transitions:
                 ep_return += tr.reward[0]
            
            ep_len = len(ep_transitions)
            if ep_len < 1:
                ep_len = 1
            mean_reward = ep_return / ep_len

            # Q-value logs 
            first_tr = ep_transitions[0]
            n_real_agents = env.k
            obs0 = first_tr.obs[:n_real_agents]
            obs0_t = torch.tensor(obs0, dtype=torch.float32, device=self.device)
            h0 = torch.zeros((n_real_agents, self.cfg.hidden_dim), dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                q_vals, _ = self.agent(obs0_t, h0)
            q_flat = q_vals.view(-1).cpu().numpy()

            log_line = f"{ep+1} {mean_reward:.6f} " + " ".join(f"{v:.6f}" for v in q_flat) + "\n"
            with open("q_values_log.txt", "a") as f:
                f.write(log_line)

            #  Update Qmix from replay buffer
            loss: Optional[float] = None
            if len(buffer) >= self.cfg.batch_size_eps:
                batch = buffer.sample(self.cfg.batch_size_eps)
                loss = self.train_on_batch(batch)
                losses.append(loss)

            # Console print
            loss_str = "N/A"
            if loss is not None:
                 loss_str = f"{loss:.4f}"
                 
            print(
                f"[Train] Ep {ep+1}/{episodes}, epsilon={epsilon:.3f}, "
                f"mean_reward={mean_reward:.4f}, loss={loss_str}"
            )

        return losses

    @torch.no_grad()
    # Run evaluation without exploration
    def evaluate(self, env: MultiAgentMazeEnv, episodes: int, max_T: int) -> Dict[str, float]:
        stats = {"success_rate": 0.0, "mean_survivors": 0.0, "mean_steps": 0.0, "mean_deaths": 0.0}
        for ep in range(episodes):
            #reset environment
            obs, state, avail = env.reset(k_agents=env.k)
            h = torch.zeros((self.cfg.n_agents, self.cfg.hidden_dim), dtype=torch.float32, device=self.device)
            steps = 0
            total_deaths = 0
            #Roll until done
            while True:
                #greedy action
                actions, h = self.select_actions(obs, avail, h, epsilon=0.0)
                #only use real agent' action
                env_actions = actions[:env.k]
                next_obs, r, done, info, next_avail = env.step(env_actions)
                steps += 1
                total_deaths += info.get("deaths", 0)
                obs, avail = next_obs, next_avail
                if done or steps >= max_T:
                    break
            #count survivors
            survivors = 0
            for a in env.alive:
                if a:
                    survivors += 1
            #determine all successes
            success = 0.0
            all_arrived = True
            for arr in env.arrived:
                if not arr:
                    all_arrived = False
                    break
                    
            if all_arrived and survivors > 0:
                success = 1.0
            #stats    
            stats["success_rate"] += success
            stats["mean_survivors"] += survivors
            stats["mean_steps"] += steps
            stats["mean_deaths"] += total_deaths
            
        div = max(1, episodes)
        for k in stats:
            stats[k] /= div
        return stats
