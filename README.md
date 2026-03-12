# QMIX-Robustness-in-a-Multi-Agent-Maze-with-Traps

This project tests how well QMIX generalizes when the number of agents changes and when the maze layout changes.
Agents must navigate a maze, avoid hidden traps, and reach a goal. The environment is partially observable, and traps behave like “spring traps” — the first agent dies, but the trap becomes safe afterward.

The system trains QMIX on one maze with different team sizes, then evaluates on another maze to measure robustness.

## Features

- `m × m` maze generated using Wilson’s algorithm
- Hidden spring traps (first agent dies, trap clears forever)
- Partial observability (local view only)
- Agents freeze at the goal (arrived agents are forced to STAY)
- QMIX with padding for variable team sizes
- Full training + cross-evaluation pipeline

---

## File Structure

- main.py
    - Build mazes
    - Creates QMIX learners for 2 different size teams
    - Trains on each maze
    - Cross-maze evaluation
    - Print statements
- maze_env.py 
    - Maze generation - wilson algorithm
    - traps, 
    - partial observations
    - reward function
    - Episode termination logic
    - Agents arrive at the goal, stop moving, and wait
- qmix.py 
    - QMIX networks - Agent RNN, MixerQMIX, EpisodeBuffer, QMIXLearner
    - replay buffer, training logic
    - alive masking

# Libraries
- numpy (We used 2.2.6 in our testing)
- torch (We used 2.9.1 in our testing)
- matplotlib (We used 3.10.6 in our testing)
- python version recommended 3.10+
- you may need to create a Virtual Environment depends on your computer and python package

if you need to download these libraries, use;
pip install numpy torch matplotlib


# The default config (inside main.py) is:
cfg = ExperimentConfig(
    m=7,                    # maze size
    k=10,                   # full team size
    k_minus_n=8,            # reduced team size
    episodes_per_run=1500,  # Run per episode
    eval_episodes=20,
    max_T=250,
    trap_mean=0.05,         # make this 0.00 if you want to turn off the traps
    trap_half_width=0.02,
    seed=seed
)

# Run the code
to run the code, inside your venv:
python main.py

Running this will:
- Build two mazes
- Train four QMIX models
- Cross-evaluate all combinations
- Print final statistics (success rate, survivors, steps, deaths)

## Important: Trap Settings
Trap probability is defined in multiple places:
- ExperimentConfig → trap_mean, trap_half_width
- EnvConfig inside main.py
- assign_traps() inside maze_env.py
- If traps behave unexpectedly, check all three locations.
This is the most common source of confusion.

# Note
- Always check trap hyperparameters, they can silently override each other.
    - hyperparameters : trap_mean -> line 229 in main.py
                        trap_half_width -> line 230 in main.py
                        per_edge_randomize -> line 184 in maze_env.py
                        spring_traps -> line 52 in main.py
- If debugging agent movements, enable:
    - debug_check_moves_from()
    - debug_random_rollout()
- You can reduce console logs during training to speed up experiments.
- This project is ideal for studying generalization, distribution shift, and coordination under partial observability.
- The library dependencies on your computer may change the result of this code. 
