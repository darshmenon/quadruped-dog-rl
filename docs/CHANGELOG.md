# Changelog / Bug Postmortems

Detailed root-cause writeups for fixes referenced from the main [README](../README.md). Kept separate so the README can stay a quick-start doc instead of a debugging log.

## Terrain height ray self-intersection (vision/rough-terrain env)

`_terrain_height_under_base()`/`_height_scan()` ray-cast straight down from above the base to measure height above ground. The ray originally checked every geom, including the robot's own base box directly beneath the ray origin, so it hit itself instead of the terrain and reported the robot as already below the ground on every single reset — instant termination, every episode, regardless of policy. Fixed by putting the floor and obstacle geoms in MuJoCo geom group 1 and restricting the ray to that group (robot geoms stay in the default group 0), so it can no longer self-intersect.

Worth knowing if you see `ep_len_mean` stuck at 1.

## Frozen-crouch alive-bonus exploit (MuJoCo walk+arm-reach)

**Status: fixed, retrained, real improvement confirmed.**

A checkpoint that looked fine in `evaluations.npz` (eval reward bouncing around 130-2000) turned out to have converged to a different exploit entirely: `play_policy.py --episodes 10` showed every episode surviving the full 1000 steps with 0.00m displacement and mean reward ≈ -720 — the robot locks its legs into a static crouch at `z≈0.16m` (just above the 0.15m fall threshold) and never moves again. Root cause: `ALIVE_BONUS` (0.3/step, meant to stop early termination from ever being a shortcut around penalties) was unconditional, so a frozen robot still earns +300 over a full episode just for existing, on top of dodging the one-time `-8` `FALL_PENALTY` that any real (and inherently riskier) walking attempt risks. Same class of bug `reach_gate` below was already added for — a reward term meant to be neutral/protective turns into a subsidy once combined with a specific failure mode.

Separately, this is why the exploit went unnoticed for a while: `EvalCallback`'s `n_eval_episodes=5` was far too small a sample once the failure mode became high-variance instead of consistently frozen — a single lucky long-survival episode can drag a 5-episode mean up 10x (one checkpoint reported eval reward 2070; a real 40-episode replay of the same checkpoint averaged 7.6 with a 100% fall rate). Bumped to 20.

Fix applied: gate `ALIVE_BONUS` off during a detected stall (`is_stalling`, the same condition `reach_gate` already used), and require the curriculum's per-episode "success" signal to also check the robot wasn't stalling for most of the episode — previously "survived 75% of max_steps" was satisfiable by freezing in place, silently escalating command speed/push difficulty for a policy that had never learned to walk. Confirmed effective after ~4M steps of retraining past the fix: 20-episode eval reward climbed from 19.8 to 598-670 (above the historical ~460 peak), though still high-variance (episode length 47→330 across evals) — real progress, not yet a fully reliable gait.

## `reach` reward never trained the arm (MuJoCo walk+arm-reach)

**Status: root cause found, fix not yet trained against.**

The `reach` reward component never actually trained the arm at all, at any curriculum level. `REACH_SIGMA=0.12`'s `exp(-(d/sigma)^2)` kernel underflows to ~0 anywhere past ~0.3m, but the stow pose starts the fingertip ~0.5-0.7m from a freshly sampled target — outside the kernel's support from step one. Checking `reward_components` on `best_model.zip` confirms `reach` sits at ~1e-8 for the entire episode: the eval reward peak of 462.97±4.57 at curriculum_level≈0.85 was pure locomotion, the arm never moved toward its target, and the later collapse to ~140 by 2.8M steps was the walk+reach curriculum (capped at `MAX_CURRICULUM_LEVEL=0.85`) still pushing command speed harder than the policy could hold — unrelated to the arm, which was never contributing reward to begin with.

Fix applied: `REACH_DENSE_WEIGHT * max(0, 1 - reach_dist/REACH_DENSE_NORM)`, a linear term with a real gradient across the actual starting-distance range, added alongside the sigma kernel (which still handles fine-precision near the goal). Not yet trained against — `best_model.zip` predates this change, so its arm behavior should not be trusted as representative of the reach task at all.

## Gazebo RL `/cmd_vel` reward bug (native Gazebo backend)

**Status: fixed, retrain needed.**

Forward walking via `/cmd_vel` previously tripped the robot's own fall-detector instead of translating, even on flat ground. Root cause: `Go2GazeboEnv`'s RL reward compared `ang_vel[0]` (roll rate) against `cmd[0]` (the linear speed target) — a copy/paste bug that never rewarded actual forward progress, since the env didn't subscribe to `/odom` at all. The termination check also used the same stale `> 0.5` tilt threshold (~120°) already fixed for the MuJoCo backend but never ported here.

Both are fixed in `training/envs/go2_gazebo_env.py` — it now subscribes to `/odom` (published by `scripts/gz_pose_to_odom.py`, already auto-launched) and uses real linear velocity in the reward, and terminates at the same ~60° tilt as MuJoCo. `train_gazebo.py`'s `EvalCallback` was also removed: it pointed at the same live env PPO was training on rather than a separate instance, so its periodic eval episodes reset/stepped the sim out from under the in-progress rollout collection every 10k steps. No Gazebo-backend checkpoint has been trained against these fixes yet — treat forward walking here as unverified until a training run completes.

## Roadmap history (fixed items, full detail)

### `global_body_planner_node` segfault on hard terrain — Fixed

Crashed on `gap_80cm.sdf`, `slope_20_hole.sdf`, `rough_40cm_huge.sdf`, `parkour_local_min.sdf`, and all `*_local_min.sdf` worlds (see [terrain test results](quadsdk_notes.md#terrain-test-results)). Root cause: `GBPL::postProcessPath` (`ros2/quad_sdk/global_body_planner/src/gbpl.cpp`) pops a state vector and an action vector together in lockstep, but the state vector always has one more element than the action vector (N states, N-1 edges) — when the direct-connect shortcut kept failing all the way to the second state (exactly what happens with no easy flat path near the goal), it read `.back()` off the now-empty action vector. Fixed by guarding that read; verified the package rebuilds clean.

### MuJoCo RL policy retrained against reward-hack fix — Retrained, partially verified

The stall-penalty fix works: commanding `vx=0.5` on the retrained policy (`go2_mujoco_final.zip`, 2M steps) now produces an actual mean `vx≈0.35 m/s` over 10 eval episodes, versus the old checkpoint's `vx≈0.01` (standing still). But this surfaced a new problem — eval episodes now end after ~90 of a possible 1000 steps (`training/logs/mujoco/evaluations.npz`), consistent from step 900k through 2M, so the policy is falling roughly 1.5-2s into every walk instead of sustaining it for the full 20s episode. Reward-hack is fixed; gait stability now needs work (likely more training, a fall/recovery-shaping reward term, or a slower-speed curriculum).

### `/cmd_vel` walking on the native Gazebo backend (IK gait) — Root-caused and fixed, untested in sim

`leg_phases` was initialized from `Gait.STAND`'s `phase_offsets` (`[0,0,0,0]`) and then advanced every tick by the *same* scalar increment applied to all four elements — so all four legs stayed perfectly in phase forever, and every non-STAND gait's `phase_offsets` (e.g. WALK's `[0, 0.5, 0.25, 0.75]`) were computed but never actually applied per leg. All four feet swung simultaneously during the non-duty fraction of each cycle, dropping the body with zero ground support and tripping `stand_go2_gz.py`'s fall-detector. Fixed in `scripts/stand_go2_gz.py`, `scripts/teleop_go2_gz.py`, `scripts/cmd_vel_go2_gz.py`, and `training/headless_control.py` (all four had the same duplicated bug): now a scalar cycle phase advances each tick, and the active gait's `phase_offsets` are added back in per leg before computing foot targets. Verified via a standalone phase-accumulation check that legs now stagger correctly for WALK — not yet verified against a live Gazebo run.

### Multi-terrain RL pipeline evaluation — Evaluated, no meaningful blind/sighted gap found

Both `train_vision_compare.py` runs (200k steps each) finished; regenerated `blind_vs_sighted.png` from their `evaluations.npz`. Final eval mean reward: blind `305.7`, sighted `307.8` — within noise of each other, so the height-scan observation isn't yet producing the expected rough-terrain advantage. Worth a longer run or a harder terrain curriculum before drawing conclusions either way.

### Arm wired into MuJoCo RL policy; native-Gazebo RL reward fixed

`go2_mujoco_env.py` now trains a combined walk+arm-reach policy (19-DOF, 76-dim obs) — this is a separate objective from the roadmap's IK-based arm item, and unrelated to the `/cmd_vel` IK-trot gait-phase fix above. Along the way:

- `train_mujoco.py` only wrote `curriculum_level.txt` at clean exit, so an interrupted run (crash/OOM/preemption) would resume with a stale curriculum against an already-advanced checkpoint — now saved every 50k steps alongside the VecNormalize checkpoint.
- `PPO()` had no `target_kl`, and training logs showed `approx_kl` climbing 0.017→0.10 and `clip_fraction` 0.2→0.6 unbounded over 3M steps while eval reward collapsed from a peak of 458 to ~120 and never recovered — `target_kl=0.03` (fresh and resumed runs) softened but didn't fully fix this; see the `reach` reward bug above for a separate, deeper issue found in the same policy.
- Separately, `Go2GazeboEnv`'s RL reward compared angular roll-rate against the linear-speed command (never rewarded actual translation) and reused the pre-fix `>0.5` tilt threshold — both fixed, see the Gazebo `/cmd_vel` reward bug above.
- `train_gazebo.py`'s `EvalCallback` was evaluating on the exact same live env instance PPO trained on, corrupting in-progress rollouts every 10k steps — removed, no substitute eval env exists yet since Gazebo isn't cheaply parallelizable like MuJoCo.

No Gazebo-backend RL checkpoint has been trained against these fixes yet.
