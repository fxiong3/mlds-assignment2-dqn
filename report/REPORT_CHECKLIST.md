# Report and Submission Checklist

## Identity and reproducibility

- [ ] GitHub username, public repository name, and public repository URL.
- [ ] Exact Deepdish setup and run commands.
- [ ] GPU ID, PyTorch/CUDA versions, seeds, and total training time.
- [ ] Hyperparameter table for both environments.

## CartPole-v1

- [ ] Confirm `gamma = 0.95` and report whether the policy reaches 500 steps reliably.
- [ ] Maximum Q-value versus episode.
- [ ] Episode reward plus 100-episode moving average.
- [ ] Histogram for 500 greedy episodes, with mean and standard deviation.
- [ ] Trained-versus-random validation and a target-update-frequency ablation.
- [ ] Edge cases and first-ten-step trace discussion.

## MsPacman-v5

- [ ] Confirm `gamma = 0.99`, nine actions, and assignment image preprocessing.
- [ ] Maximum Q-value versus episode.
- [ ] Episode reward plus 100-episode moving average.
- [ ] Histogram for 500 greedy episodes, with mean and standard deviation.
- [ ] Evidence of improvement over 4,000-6,000 episodes.
- [ ] Trained-versus-random validation and at least one controlled ablation.
- [ ] Edge cases and first-ten-step trace discussion.

## Non-standard evidence

- [ ] Include and interpret the action-frequency heatmap.
- [ ] Explain whether action diversity collapses or evolves during learning.
- [ ] Include quantitative evidence, not only visual impressions.

## Extended three-month plan

- [ ] Repeat at least five independent seeds and report confidence intervals.
- [ ] Compare standard DQN, Double DQN, and prioritized replay.
- [ ] Run systematic learning-rate, target-update, replay-size, and epsilon ablations.
- [ ] Evaluate robustness to sticky actions and observation perturbations.
- [ ] Examine Q-value calibration against Monte Carlo returns.
- [ ] Profile sample efficiency, GPU utilization, wall-clock time, and energy use.

## Final checks

- [ ] Every figure has a title, labeled axes, readable legend, and caption.
- [ ] Every discussion question is answered.
- [ ] All code used for the report is committed to the public repository.
- [ ] The PDF opens correctly and the GitHub URL is clickable.

