from __future__ import annotations
from typing import Dict, Tuple, Any, List, Optional
import random
import numpy as np
import torch
from dataclasses import dataclass

from maze_env import (
    MultiAgentMazeEnv, wilson_maze, print_maze_adjacency, print_maze_ascii,
    TrapConfig, assign_traps, EnvConfig, set_seed, Coord
)
from qmix import QMIXLearner, QMIXConfig, EpisodeBuffer

# default settings
@dataclass
class ExperimentConfig:
    m: int = 15
    k: int = 4
    k_minus_n: int = 2
    episodes_per_run: int = 600
    eval_episodes: int = 50
    max_T: int = 200
    trap_mean: float = 0.15
    trap_half_width: float = 0.05
    seed: int = 42

def _build_env_and_qmix(m: int, k_max: int, trap_mean: float, trap_half_width: float, seed: int) -> Tuple[MultiAgentMazeEnv, QMIXLearner, EnvConfig]:
    # local rng for learner and environment pair
    rng = random.Random(seed)
    #generate a maze adjacency usint wilson algorithms
    adj = wilson_maze(m, rng)
    # goal near center
    goal = (m // 2, m // 2)
    start = (0, 0)
    #trap config
    tcfg = TrapConfig(mean=trap_mean, half_width=trap_half_width, per_edge_randomize=True)
    traps = assign_traps(adj, tcfg, rng)
    #Internal "reduced team" size for the environment.
    k_minus_n_calc = k_max - 2
    if k_minus_n_calc < 1:
        k_minus_n_calc = 1
    #environment config
    ecfg = EnvConfig(
        m=m, k_agents=k_max, k_minus_n=k_minus_n_calc, max_steps=250, 
        step_cost=0.002,     
        death_penalty=0.0,  
        goal_reward=20.0,   
        trap_clear_lambda=0.0,
        use_oracle_shaping=True,
        gamma=0.99,
        spring_traps=True,
        seed=seed
    )
    # create the environment with the generated maze, config, traps
    env = MultiAgentMazeEnv(adj, traps, start, goal, ecfg)

    # derive dims for qmix
    dummy_obs, dummy_state, dummy_avail = env.reset(k_agents=k_max)
    obs_dim = dummy_obs[0].shape[0]
    state_dim = dummy_state.shape[0]
    #qmix config
    qcfg = QMIXConfig(n_agents=k_max, obs_dim=obs_dim, state_dim=state_dim, gamma=ecfg.gamma,
                      lr=5e-4, batch_size_eps=8, target_update_interval=200,
                      epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_episodes=50)
    #Build a learner on cpu
    learner = QMIXLearner(qcfg, device="cpu")
    return env, learner, ecfg

#stinky debug section
def debug_check_moves_from(env: MultiAgentMazeEnv, pos: Coord):
    print(f"\n[DEBUG] Checking moves from {pos}")
    # Reset environment with just 1 agent to simplify debugging
    env.reset(k_agents=1)
    # Force this single agent to be placed at the test position
    env.pos[0] = pos
    print("Neighbors in adj:", env.adj[pos])
    # get which actions are allowed, such as N, E, S, W
    avail = env.get_avail_actions(0)
    print("Avail actions mask [N,E,S,W,STAY]:", avail.tolist())
    #Print the result
    for d, name in enumerate(env.ACTIONS[:-1]):  
        nbr = env._nbr_in_dir(env.pos[0], d)
        print(f"  dir {d} ({name}) -> {nbr}")


def run_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    #use a single seed to make the experiment reproducable
    set_seed(cfg.seed)

    # Build Maze 1, train with team size K
    env1, learner1_k, ecfg1 = _build_env_and_qmix(cfg.m, cfg.k, cfg.trap_mean, cfg.trap_half_width, cfg.seed + 1)
    # separate learner for k-n on the same env
    learner1_kn = _build_env_and_qmix(cfg.m, cfg.k, cfg.trap_mean, cfg.trap_half_width, cfg.seed + 2)[1]
    #Build Maze 2, train with team size k-n
    env2, learner2_k, ecfg2 = _build_env_and_qmix(cfg.m, cfg.k, cfg.trap_mean, cfg.trap_half_width, cfg.seed + 3)
    # seperate learner for k-n on the same env
    learner2_kn = _build_env_and_qmix(cfg.m, cfg.k, cfg.trap_mean, cfg.trap_half_width, cfg.seed + 4)[1]

    # Debug: print maze structure once at start
    print("\n=== Maze 1 structure ===")
    print_maze_adjacency(env1.adj, cfg.m)
    print_maze_ascii(env1.adj, cfg.m, start=env1.start, goal=env1.goal)

    print("\n=== Maze 2 structure ===")
    print_maze_adjacency(env2.adj, cfg.m)
    print_maze_ascii(env2.adj, cfg.m, start=env2.start, goal=env2.goal)

    debug_check_moves_from(env1, (1, 0))

    # Replay buffers
    buf1_k = EpisodeBuffer(capacity=1000)
    buf1_kn = EpisodeBuffer(capacity=1000)
    buf2_k = EpisodeBuffer(capacity=1000)
    buf2_kn = EpisodeBuffer(capacity=1000)

    # Train on Maze 1
    # Train with k agents
    env1.reset(k_agents=cfg.k)

    learner1_k.train(env1, episodes=cfg.episodes_per_run, buffer=buf1_k, max_T=cfg.max_T,
                     epsilon_sched=(learner1_k.cfg.epsilon_start, learner1_k.cfg.epsilon_end, learner1_k.cfg.epsilon_decay_episodes))

    # Train with k - n agents
    env1.reset(k_agents=cfg.k_minus_n)

    learner1_kn.train(env1, episodes=cfg.episodes_per_run, buffer=buf1_kn, max_T=cfg.max_T,
                      epsilon_sched=(learner1_kn.cfg.epsilon_start, learner1_kn.cfg.epsilon_end, learner1_kn.cfg.epsilon_decay_episodes))

    # Train on Maze 2 
    env2.reset(k_agents=cfg.k)
    # Train learner_2 on Maze 2 with team size k
    learner2_k.train(env2, episodes=cfg.episodes_per_run, buffer=buf2_k, max_T=cfg.max_T,
                     epsilon_sched=(learner2_k.cfg.epsilon_start, learner2_k.cfg.epsilon_end, learner2_k.cfg.epsilon_decay_episodes))
    #train with team size k-n
    env2.reset(k_agents=cfg.k_minus_n)
    learner2_kn.train(env2, episodes=cfg.episodes_per_run, buffer=buf2_kn, max_T=cfg.max_T,
                      epsilon_sched=(learner2_kn.cfg.epsilon_start, learner2_kn.cfg.epsilon_end, learner2_kn.cfg.epsilon_decay_episodes))

    # Evaluate cross-maze with team-size masking 
    results = {}

    # Helper to clone env with same maze but different k for fair eval
    def clone_env(base_env: MultiAgentMazeEnv, k_agents: int) -> MultiAgentMazeEnv:
        trap_src = base_env.traps_armed
        if hasattr(base_env, "orig_traps_armed"):
            trap_src = base_env.orig_traps_armed
        #create a new environment  with a specific number of active agents   
        new_env = MultiAgentMazeEnv(
            base_env.adj,
            trap_src,
            base_env.start,
            base_env.goal,
            base_env.cfg
        )
        new_env.k_max = base_env.k_max
        new_env.reset(k_agents=k_agents)
        return new_env

    # Evaluate: learner trained on Maze1 with k agents, and test on Maze2 with k-n agents
    env2_eval_kn = clone_env(env2, cfg.k_minus_n)
    res_m1k_on_m2kn = learner1_k.evaluate(env2_eval_kn, episodes=cfg.eval_episodes, max_T=cfg.max_T)
    results["m1_k_train__m2_kn_test"] = res_m1k_on_m2kn

    # Evaluate reverse, model trained on Maze 2 with k agents will tested on Maze 2 with k-n agents
    env1_eval_kn = clone_env(env1, cfg.k_minus_n)
    res_m2k_on_m1kn = learner2_k.evaluate(env1_eval_kn, episodes=cfg.eval_episodes, max_T=cfg.max_T)
    results["m2_k_train__m1_kn_test"] = res_m2k_on_m1kn

    # Within-maze sanity checks
    #Maze 1, K agents
    env1_eval_k = clone_env(env1, cfg.k)
    results["m1_k_train__m1_k_test"] = learner1_k.evaluate(env1_eval_k, episodes=cfg.eval_episodes, max_T=cfg.max_T)
    #maze 1, k-n agents
    env1_eval_kn = clone_env(env1, cfg.k_minus_n)
    results["m1_kn_train__m1_kn_test"] = learner1_kn.evaluate(env1_eval_kn, episodes=cfg.eval_episodes, max_T=cfg.max_T)
    #Maze 2, k agents
    env2_eval_k = clone_env(env2, cfg.k)
    results["m2_k_train__m2_k_test"] = learner2_k.evaluate(env2_eval_k, episodes=cfg.eval_episodes, max_T=cfg.max_T)
    #maze 2, k-n agents
    env2_eval_kn = clone_env(env2, cfg.k_minus_n)
    results["m2_kn_train__m2_kn_test"] = learner2_kn.evaluate(env2_eval_kn, episodes=cfg.eval_episodes, max_T=cfg.max_T)

    return results

#Run a single episode with uniformly random legal actions
def debug_random_rollout(env: MultiAgentMazeEnv, steps: int = 15):
    # Reset environment with its current team size
    obs, state, avail = env.reset(k_agents=env.k)
    print("\n[DEBUG] Random rollout starting at:", env.pos)
    for t in range(steps):
        actions = []
        # Choose a random valid action for each agent
        for i in range(env.k):
            # Find which actions this agent is allowed to take
            valid = np.where(avail[i] > 0.5)[0]
            # If somehow no actions are available, force STAY   
            if len(valid) == 0:
                a = 4 
            #pick uniformly at random    
            else:
                a = int(np.random.choice(valid))
            actions.append(a)
        # Step the environment using these actions
        obs, r, done, info, avail = env.step(actions)
        # Print a compact summary for debugging       
        print(f"t={env.t}, pos={env.pos}, r={r:.3f}, info={info}")
        #stop early in case of termination
        if done:
            break

if __name__ == "__main__":
    cfg = ExperimentConfig(
        m=7, # maze size           
        k=10, # k agent size 
        k_minus_n=8, # k-n agent size
        episodes_per_run=1500,   # training episode size
        eval_episodes=20, # evaluation episode size
        max_T=250,     # shorter episodes
        trap_mean=0.05, # traps mean, make it 0.00 to turn off the traps
        trap_half_width=0.02, 
        seed=123
    )
    #Run the experiments and print the result
    results = run_experiment(cfg)
    print("Results:")
    for k, v in results.items():
        print(k, v)
