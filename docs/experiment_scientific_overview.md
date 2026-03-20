# RoBox Octagon Experiment: Scientific Overview and Motivation

## 1. Scientific Motivation

The RoBox Octagon experiment is designed to investigate how agents (biological or artificial) learn, make decisions, and adapt their behavior in a spatial environment with competing rewards. The central question is: How do agents balance individual learning with social influences when choosing between high-value and low-value options?

This experiment is inspired by animal studies (e.g., mice) and aims to bridge behavioral neuroscience and reinforcement learning by implementing a protocol that mimics real-world learning and social interaction.


## 2. Experimental Design

### Environment
- The environment is an octagonal arena with eight patches, each potentially offering a reward.
- Agents start at random positions and must navigate to a patch, making a choice between high-value and low-value rewards.

### Phases
1. **Solo Learning Phase**
   - Each agent trains independently, learning to identify and select the high-value patch.
   - The goal is to measure how quickly and reliably agents learn the optimal choice without social influence.

2. **Social Phase**
   - Two agents are placed together in the arena.
   - Both must make choices simultaneously, with their decisions potentially influenced by the presence and actions of the other agent.
   - This phase tests social modulation: Does the presence of another agent change the learned preference or strategy?

3. **Interleaved Blocks**
   - The experiment alternates between solo and social blocks, allowing measurement of how social context affects learning and performance over time.



## 3. Behavioral and Computational Goals

- **Learning Curves:** Track how agents improve their performance (probability of choosing high-value) across trials.
- **Spatial Analysis:** Examine how starting position and patch separation affect choices.
- **Social Influence:** Quantify how the presence of another agent alters decision-making, preference, and reaction times.
- **Modeling:** Fit computational models (bistable attractor) to agent behavior, capturing the dynamics of decision-making under solo and social conditions.
- **Comparison:** Contrast social agents with "socially-blind" agents (those trained without social context) to isolate the effect of social information.


## 4. Reinforcement Learning Model Construction

### Observation Spaces
- Each agent receives a vector of observations at every time step.
- Observations typically include:
   - Agent’s own position and velocity
   - Positions of all reward patches (high and low)
   - Distances to each patch
   - (In social phase) positions and velocities of the other agent
   - Contextual cues (e.g., solo or social phase indicator)

### Action Spaces
- Agents select actions from a discrete set:
   - Move toward one of the eight patches (destination choice)
   - Optionally, fine-grained movement (directional or velocity control)
- In most implementations, the action is a destination choice (which patch to approach), not a low-level motor command.

### Collision Handling
- The environment enforces collision rules:
   - Agents cannot occupy the same space as another agent or patch.
   - If a collision is imminent, the agent’s movement is blocked or redirected.
   - Collisions may affect reward delivery or trial outcome, especially in social trials.

### Decision Process: Per Tick or Destination?
- Agents make decisions at each time step (tick):
   - They can update their chosen destination at any tick, not just at trial onset.
   - This allows for dynamic adjustment—agents can change their mind if new information is observed (e.g., opponent’s movement).
- The environment records the final destination reached, but the agent’s policy is flexible and responsive throughout the trial.

### Can Agents Change Their Mind?
- Yes: Agents are able to revise their intended destination during a trial.
- This models real-world flexibility—agents can react to social cues, environmental changes, or internal uncertainty.
- The RL policy is trained to optimize reward, so changing decisions is part of adaptive behavior.

### Summary
- The RL agents are designed to mimic both spatial navigation and flexible decision-making, with rich observations and the ability to adapt their actions in real time.
- This setup enables the study of not just which patch is chosen, but how agents respond to social and environmental dynamics during the trial.



## 4. Data and Analysis Outputs

- **Trial Logs:** Every trial records agent choices, positions, rewards, and reaction times.
- **Block Summaries:** Aggregate statistics for each block (solo/social), including preference shifts and win rates.
- **Figures:** Learning curves, spatial heatmaps, social modulation plots, and model fits.
- **Dashboard:** A summary PDF visualizing key behavioral and modeling results.
- **Reproducibility Report:** Automated checks to ensure results match published benchmarks (e.g., preference thresholds, social drop, opponent sensitivity).


## 5. Scientific Questions Addressed

- How do agents learn to maximize reward in a spatially structured environment?
- How does social context alter learned preferences and strategies?
- Can computational models capture the observed behavioral shifts between solo and social conditions?
- Are the effects of social influence robust and reproducible across multiple agent pairs and seeds?


## 6. Broader Impact

This experiment provides a framework for studying the interplay between individual and social learning, with applications in neuroscience, psychology, and artificial intelligence. By combining rigorous behavioral protocols with computational modeling, it enables deeper insights into the mechanisms of adaptive decision-making.

---

**For further reading:** See the summary dashboard and reproducibility report generated by the pipeline for concrete results and scientific validation.
