from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import math
import random
import torch
import numpy as np
from collections import deque

Coord = Tuple[int, int]
Edge = Tuple[Coord, Coord]
#Set all relevant random number generators to a fixed seed.
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#return all valid neighbors
def neighbors_4(m: int, r: int, c: int) -> List[Coord]:
    # generats list of neighbors, N,E,S,W
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)] 
    for dr, dc in moves:
        rr = r + dr
        cc = c + dc
        # check bounds, and only return when it is
        if 0 <= rr and rr < m:
            if 0 <= cc and cc < m:
                yield (rr, cc)
#return sorted undirected edge
def edge_canonical(u: Coord, v: Coord) -> Edge:
    if u <= v:
        return (u, v)
    else:
        return (v, u)
#compute manhattan distance
def manhattan(a: Coord, b: Coord) -> int:
    # Expanded math for distance
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    if dx < 0:
        dx = -dx
    if dy < 0:
        dy = -dy
    res = dx + dy
    return res

# Wilsons algoritm, Retrns adjacncy list for a perfect maze on an m x m grid
def wilson_maze(m: int, rng: random.Random) -> Dict[Coord, List[Coord]]:
    #list all grid cells
    nodes = []
    for r in range(m):
        for c in range(m):
            nodes.append((r, c))
            
    in_tree = set()
    adj: Dict[Coord, List[Coord]] = {v: [] for v in nodes}
    
    # Start with one random node in the tree
    root_r = rng.randrange(m)
    root_c = rng.randrange(m)
    root = (root_r, root_c)
    in_tree.add(root)

    def add_path(path: List[Coord]):
        # adds path to the adj struct
        limit = len(path) - 1
        for i in range(limit):
            u = path[i]
            v = path[i + 1]
            if v not in adj[u]:
                adj[u].append(v)
            if u not in adj[v]:
                adj[v].append(u)

    while len(in_tree) < len(nodes):
        # pick a node not in tree
        candidates = [v for v in nodes if v not in in_tree]
        start = rng.choice(candidates)
        walk = [start]
        visited_idx: Dict[Coord, int] = {start: 0}
        current = start
        #random walking until hitting the existing maze
        while current not in in_tree:
            nbrs = list(neighbors_4(m, *current))
            nxt = rng.choice(nbrs)
            if nxt in visited_idx:
                loop_start = visited_idx[nxt]
                walk = walk[: loop_start + 1]
                visited_idx = {}
                for i, v in enumerate(walk):
                    visited_idx[v] = i
                current = walk[-1]
            else:
                walk.append(nxt)
                idx_val = len(walk) - 1
                visited_idx[nxt] = idx_val
                current = nxt        
        # Add the completed path into the maze
        add_path(walk)
        in_tree.update(walk)        
    return adj
#helper for debugging
def print_maze_adjacency(adj: Dict[Coord, List[Coord]], m: int):
    print("Maze adjacency (each cell -> list of neighbors):")
    for r in range(m):
        for c in range(m):
            v = (r, c)
            nbrs = sorted(adj[v])
            print(f"  {v} -> {nbrs}")
    print()

#helper for debugging
def print_maze_ascii(adj: Dict[Coord, List[Coord]], m: int,
                     start: Optional[Coord] = None,
                     goal: Optional[Coord] = None):
    #check if 2 cells are connected
    def has_edge(u: Coord, v: Coord) -> bool:
        return v in adj[u]

    print("ASCII maze (S = start, G = goal):")

    for r in range(m):
        # Horizontal walls above row r
        line_top = []
        for c in range(m):
            if r == 0:
                line_top.append("+---")
            else:
                prev_r = r - 1
                if has_edge((prev_r, c), (r, c)):
                    line_top.append("+   ")
                else:
                    line_top.append("+---")
        line_top.append("+")
        print("".join(line_top))

        # Cell row with vertical walls
        line_mid = []
        for c in range(m):
            v = (r, c)
            ch = " "
            if start is not None and v == start:
                ch = "S"
            if goal is not None and v == goal:
                ch = "G"
            if c == 0:
                left_wall = "|"
            else:
                prev_c = c - 1
                if has_edge((r, prev_c), (r, c)):
                    left_wall = " "
                else:
                    left_wall = "|"
            line_mid.append(left_wall + f" {ch} ")
        line_mid.append("|")
        print("".join(line_mid))

    # Bottom border
    line_bottom = []
    for c in range(m):
        line_bottom.append("+---")
    line_bottom.append("+")
    print("".join(line_bottom))
    print()


# Trap assignmnt
@dataclass
class TrapConfig:
    # average trap probability
    mean: float = 0.15              
    # variation around mean for sampling p
    half_width: float = 0.05        
    # sample a different p for each edge
    per_edge_randomize: bool = True 

# assign traps to the maze edges
def assign_traps(adj: Dict[Coord, List[Coord]], cfg: TrapConfig, rng: random.Random) -> Dict[Edge, bool]:
    # Clamp the probability range to [0, 1]
    lo_val = cfg.mean - cfg.half_width
    hi_val = cfg.mean + cfg.half_width
    lo = max(0.0, lo_val)
    hi = min(1.0, hi_val)
    #pick p if no ranomization
    if not cfg.per_edge_randomize:
        p = rng.uniform(lo, hi)

    traps: Dict[Edge, bool] = {}
    #iterate over all edges
    for u, nbrs in adj.items():
        for v in nbrs:
            if u < v:  
                if cfg.per_edge_randomize:
                    p = rng.uniform(lo, hi)
                # Bernoulli draw with prob p
                val = rng.random()
                is_trap = False
                if val < p:
                    is_trap = True
                traps[(u, v)] = is_trap
    return traps


# Multi-agent maze envrionment
@dataclass
class EnvConfig:
    m: int = 15
    k_agents: int = 10
    k_minus_n: int = 2
    max_steps: int = 200
    step_cost: float = 0.002
    death_penalty: float = 0.0
    goal_reward: float = 20.0
    trap_clear_lambda: float = 0.0    
    use_oracle_shaping: bool = True  
    gamma: float = 0.99
    spring_traps: bool = True
    seed: int = 123

class MultiAgentMazeEnv:
    # Partially obsevable maze with spring traps
    # Observations N,E,S,W
    ACTIONS = ['N', 'E', 'S', 'W', 'STAY']
    DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)] 
    N_DIR = 4

    # Initialize the multi-agent maze environment.
    def __init__(self, adj: Dict[Coord, List[Coord]], traps_armed: Dict[Edge, bool],
                 start: Coord, goal: Coord, cfg: EnvConfig):
        self.cfg = cfg
        self.m = cfg.m
        self.adj = adj
        # trap status stored per undirected edge
        self.orig_traps_armed: Dict[Edge, bool] = traps_armed.copy()  
        self.traps_armed: Dict[Edge, bool] = traps_armed.copy()
        self.cleared_edges: set[Edge] = set()
        self.start = start
        self.goal = goal
        self.k_max = cfg.k_agents  
        self.reset(k_agents=cfg.k_agents)

    # Reset the environment for a new episode.
    def reset(self, k_agents: Optional[int] = None):
        # Determine how many agents will run this episode
        if k_agents is None:
            k_agents = self.k_max
        self.k = k_agents
        self.t = 0
        # initialize positions and status
        self.pos: List[Coord] = [self.start for _ in range(self.k)]
        self.alive: List[bool] = [True for _ in range(self.k)]
        self.arrived: List[bool] = [False for _ in range(self.k)]
        self._goals_paid: List[bool] = [False for _ in range(self.k)]  

        # Observations need last action; initialize to STAY
        self.last_action = [4 for _ in range(self.k)]

        # Track whether an edge has been cleared for observation
        # Restore original trap layout each episode and clear per-episode 'cleared' marks
        self.traps_armed = self.orig_traps_armed.copy()
        self.cleared_edges = set()

        # Return initial RL outputs
        obs = [self.get_obs(i) for i in range(self.k)]
        state = self.get_state()
        avail = [self.get_avail_actions(i) for i in range(self.k)]
        return obs, state, avail

    # Return the neighboring cell during the movement
    def _nbr_in_dir(self, pos: Coord, dir_idx: int) -> Optional[Coord]:
        dr, dc = self.DIRS[dir_idx]
        r_curr = pos[0]
        c_curr = pos[1]
        rr = r_curr + dr
        cc = c_curr + dc
        
        # check if within bounds and edge exists
        if 0 <= rr < self.m:
            if 0 <= cc < self.m:
                 if (rr, cc) in self.adj[pos]:
                     return (rr, cc)
        return None

    def get_obs(self, i: int) -> np.ndarray:
        # Observation for agent i
        pos = self.pos[i]
        alive = self.alive[i]
        arrived = self.arrived[i]
        
        passable = np.zeros(self.N_DIR, dtype=np.float32)
        cleared = np.zeros(self.N_DIR, dtype=np.float32)
        
        for d in range(self.N_DIR):
            nbr = self._nbr_in_dir(pos, d)
            if nbr is not None:
                passable[d] = 1.0
                e = edge_canonical(pos, nbr)
                if e in self.cleared_edges:
                    cleared[d] = 1.0
                else:
                    cleared[d] = 0.0
            else:
                passable[d] = 0.0
                cleared[d] = 0.0  

        at_goal_val = 0.0
        if pos == self.goal and alive:
            at_goal_val = 1.0
            
        at_goal = float(at_goal_val)
        last_a = np.zeros(len(self.ACTIONS), dtype=np.float32)
        
        la_idx = 4
        if i < len(self.last_action):
            la_idx = self.last_action[i]
        
        last_a[la_idx] = 1.0
        
        # concat parts
        parts = [passable, cleared, np.array([at_goal], dtype=np.float32), last_a]
        o = np.concatenate(parts, axis=0)
        
        if not alive:
            # Dead agents still produce an observation
            o = o * 0.0 
            # set last action to stay 
            o[-1] = 1.0
        return o

    # Return a mask of which actions agent i is allowed to take
    def get_avail_actions(self, i: int) -> np.ndarray:
        # mask invalid directions 
        # allow stay always
        # dead agents only stay
        avail = np.zeros(len(self.ACTIONS), dtype=np.float32)
        if not self.alive[i]:
            avail[-1] = 1.0
            return avail
        # alive agent    
        pos = self.pos[i]
        for d in range(self.N_DIR):
            if self._nbr_in_dir(pos, d) is not None:
                avail[d] = 1.0
                
        # stay always allowed if alive
        avail[-1] = 1.0
        return avail

    # Compute BFS distance from the goal to all reachable cells
    def _distance_field(self) -> Dict[Coord, int]:
        q = deque()
        q.append(self.goal)
        # set distance of the goal to itself
        dist = {self.goal: 0}
        # Standard BFS
        while q:
            u = q.popleft()
            for v in self.adj[u]:
                e = edge_canonical(u, v)
                # pass only if edge is safe now
                armed = self.traps_armed.get(e, False)
                if armed:
                    continue
                # add new reachable goal
                if v not in dist:
                    current_d = dist[u]
                    new_d = current_d + 1
                    dist[v] = new_d
                    q.append(v)
        return dist

    # Build a centralized state vector for QMIX.
    def get_state(self) -> np.ndarray:
        # Storage for agent data
        pos_arr = np.zeros((self.k, 2), dtype=np.float32)
        alive_arr = np.zeros((self.k,), dtype=np.float32)
        arrived_arr = np.zeros((self.k,), dtype=np.float32)
        #Normalize coordinates
        denom = self.m - 1
        if denom < 1:
            denom = 1
        #Fill data with each alive agent
        for i in range(self.k):
            r, c = self.pos[i]
            pos_arr[i, 0] = r / denom
            pos_arr[i, 1] = c / denom
            
            if self.alive[i]:
                alive_arr[i] = 1.0
            else:
                alive_arr[i] = 0.0
                
            if self.arrived[i]:
                arrived_arr[i] = 1.0
            else:
                arrived_arr[i] = 0.0
        # normalize goal position        
        g_r = self.goal[0] / denom
        g_c = self.goal[1] / denom
        goal_rc = np.array([g_r, g_c], dtype=np.float32)
        
        step_val = self.t
        max_s = self.cfg.max_steps
        if max_s < 1:
             max_s = 1
        
        step_frac = np.array([step_val / max_s], dtype=np.float32)
        #combine everything in 1 vector
        parts = [pos_arr.flatten(), alive_arr, arrived_arr, goal_rc, step_frac]
        state = np.concatenate(parts, axis=0)

        # Pad to k_max if necessary
        if self.k < self.k_max:
            pad_agents = self.k_max - self.k
            pad_pos = np.zeros((pad_agents, 2), dtype=np.float32).flatten()
            pad_alive = np.zeros((pad_agents,), dtype=np.float32)
            pad_arrived = np.zeros((pad_agents,), dtype=np.float32)
            state = np.concatenate([state, pad_pos, pad_alive, pad_arrived], axis=0)
            
        return state

    # Joint action for all agents
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], float, bool, Dict[str, Any], List[np.ndarray]]:
        if len(actions) != self.k:
            raise ValueError("actions must be provided for current number of agents")

        # Advance the time-step counter
        self.t = self.t + 1

        # Save old positions
        prev_pos = list(self.pos)

        # Precompute distance field before step
        dist_before = self._distance_field()

        deaths = 0
        arrivals = 0
        cleared_any = False

        # Resolve moves.
        new_pos = list(self.pos)
        #record last action
        for i, a in enumerate(actions):
            self.last_action[i] = int(a)
            
        # compute target cells
        targets: List[Optional[Coord]] = [None] * self.k
        for i, a in enumerate(actions):
            if not self.alive[i] or self.arrived[i]:
                targets[i] = self.pos[i]  
                continue
            if a == 4:  
                targets[i] = self.pos[i]
                continue
            d = int(a)
            # move direction
            nbr = self._nbr_in_dir(self.pos[i], d)
            if nbr is None:
                targets[i] = self.pos[i] 
            else:
                targets[i] = nbr

        # Apply effects and move
        for i in range(self.k):
            if not self.alive[i] or self.arrived[i]:
                continue
            tgt = targets[i]
            #if stays
            if tgt is None:
                continue
            if tgt == self.pos[i]:
                new_pos[i] = self.pos[i]
                continue

            # Check trap
            e = edge_canonical(self.pos[i], tgt)
            armed = self.traps_armed.get(e, False)
            if armed:
                # death, agent dies, trap cleared
                self.alive[i] = False
                deaths += 1
                self.traps_armed[e] = False
                self.cleared_edges.add(e)
                cleared_any = True
                continue 
            else:
                # Move
                new_pos[i] = tgt

        # Update positions
        self.pos = new_pos

        # DEBUG: print each agent's position after this step
        print(f"[DEBUG] t={self.t}, positions={self.pos}")

        # Arrival check
        for i in range(self.k):
            if self.alive[i]:
                 if self.pos[i] == self.goal:
                      if not self.arrived[i]:
                          self.arrived[i] = True
                          arrivals += 1

        # Reward calculation
        r = 0.0
        r = r - self.cfg.step_cost
        penalty = self.cfg.death_penalty * deaths
        r = r - penalty
        
        # Very large goal reward
        if arrivals > 0:
            alive_count = 0
            for a in self.alive:
                if a:
                    alive_count += 1
            
            div = self.k
            if div < 1:
                div = 1
                
            survivor_factor = alive_count / div
            multiplier = 1.0 + survivor_factor
            rew = self.cfg.goal_reward * arrivals * multiplier
            r = r + rew

        # Potential-based shaping
        dist_after = self._distance_field()
        phi_before = 0.0
        phi_after = 0.0
        
        for i in range(self.k):
            if self.alive[i]:
                # distance before move
                default_dist = manhattan(prev_pos[i], self.goal) + 1000
                db = dist_before.get(prev_pos[i], default_dist)
                
                # distance after move
                default_dist_a = manhattan(self.pos[i], self.goal) + 1000
                da = dist_after.get(self.pos[i], default_dist_a)
                
                phi_before = phi_before - db
                phi_after = phi_after - da
                
        shaping = (self.cfg.gamma * phi_after) - phi_before
        r = r + shaping

        # Optional extra event bonus
        if cleared_any:
             if self.cfg.trap_clear_lambda > 0.0:
                diff = phi_after - phi_before
                if diff < 0.0:
                    diff = 0.0
                delta = diff
                bonus = self.cfg.trap_clear_lambda * delta
                r = r + bonus

        done = False
        
        # check all dead
        all_dead = True
        for a in self.alive:
            if a:
                all_dead = False
                break
                
        # check all arrived
        all_arrived = True
        for a in self.arrived:
             if not a:
                 all_arrived = False
                 break
                 
        if all_dead or all_arrived or self.t >= self.cfg.max_steps:
            done = True

        obs = [self.get_obs(i) for i in range(self.k)]
        avail = [self.get_avail_actions(i) for i in range(self.k)]
        info = {"deaths": deaths, "arrivals": arrivals, "cleared_any": cleared_any}
        
        return obs, r, done, info, avail
