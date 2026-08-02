# Gymnasium-RL: CartPole-v1 with Tabular Q-Learning

Learning reinforcement learning from the ground up by implementing tabular Q-learning on CartPole-v1, with no RL libraries doing the learning — every piece (state discretization, Q-table, action selection, update rule, training loop) written from scratch.

## Why CartPole + tabular Q-learning

CartPole is the standard "hello world" of RL: a pole balanced on a moving cart, 4 continuous state values, 2 discrete actions (push left / push right), +1 reward per timestep survived. Tabular Q-learning is the simplest algorithm that can solve it, which makes it a good way to build intuition for the core mechanics (states, actions, rewards, value functions, the Bellman equation) before moving to function approximation / deep RL.

## What's implemented

- **State discretization** (`get_bucket`) — maps each of the 4 continuous state values (cart position, cart velocity, pole angle, pole angular velocity) into a fixed number of discrete buckets, since a Q-table needs finite, indexable states.
- **Q-table** — a 5D NumPy array (`buckets × buckets × buckets × buckets × actions`) holding the estimated value of each action in each discretized state, initialized to zero.
- **ε-greedy action selection** (`choose_action`) — explores randomly with probability ε, otherwise exploits the current best-known action from the Q-table. ε decays over training so the agent shifts from exploring to exploiting.
- **Q-learning update rule** (`update_q`) — after each step, nudges `Q[state, action]` toward `reward + γ * max(Q[next_state])`, i.e. the Bellman optimality equation turned into an incremental learning rule.
- **Training loop** — runs many episodes, calling the above four pieces each step, logging total reward periodically to track learning progress.

## Current status: not yet solving CartPole

Training reward plateaus around 10-20 (CartPole's max per episode is 500), which is close to random-policy performance. Two structural issues have been identified so far:

1. **Coarse angle discretization.** With few buckets covering the narrow pole-angle range that CartPole allows before terminating, small but control-relevant angle differences get collapsed into the same bucket, limiting how precisely the agent can represent "how bad is this tilt."
2. **Missing terminal-state handling (confirmed bug, not yet fixed in code).** `update_q` currently always bootstraps off `next_state`'s Q-values, even on the step where the episode ends (pole falls). Since bucketing collapses many distinct raw states into the same bucket, this leaks arbitrary, unrelated Q-values into the update at exactly the moment — right before failure — where the learning signal should be clearest. The fix is to zero out the bootstrapped term when `done=True`:
   ```python
   best_next_value = 0 if done else np.max(Q[next_state_buckets])
   ```
   This has been identified but not yet applied/retrained as of this commit.

Ruled out so far: simply training longer (tested up to 50,000 episodes) does not fix the plateau — this points to the fix above being necessary rather than more training time.

## Next steps

- Apply the terminal-state fix to `update_q` and retrain
- Re-evaluate whether bucket count / gamma need further tuning after that fix
- Once tabular Q-learning solves CartPole reliably, move to function approximation (DQN) as the state space grows

## Background / learning log

Session notes and derivations (the Bellman equation, the Q-value-as-running-average derivation, MDP formalism) are captured as markdown cells inline in the notebook.
