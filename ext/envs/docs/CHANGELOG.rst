Changelog
---------

0.1.6 (2024-07-07)
~~~~~~~~~~~~~~~~~~

memo:
^^^^^

* Termination term should be carefully considered along with the punishment reward functions.
  When there are too many negative reward in the begining, agent would prefer to die sooner by
  exploiting the termination condition, and this would lead to the agent not learning the task.

* tips:
  When designing the reward function, try be incentive than punishment.

Changed
^^^^^^^

* Changed :class:`ext.envs.cfgs.robots.hebi.robot_dynamics.RobotTerminationsCfg` to include DoneTerm: robot_extremely_bad_posture
* Changed :function:`ext.envs.cfgs.robots.hebi.mdp.terminations.terminate_extremely_bad_posture` to be probabilistic
* Changed :field:`ext.envs.envs.tasks.manipulations.track_goal.config.hebi.Hebi_JointPos_GoalTracking_Env.RewardsCfg.end_effector_position_tracking`
  and :field:`ext.envs.envs.tasks.manipulations.track_goal.config.hebi.Hebi_JointPos_GoalTracking_Env.RewardsCfg.end_effector_orientation_tracking`
  to be incentive reward instead of punishment reward.
* Renamed orbit_mdp to lab_mdp in :file:`ext.envs.envs.tasks.manipulations.track_goal.config.Hebi_JointPos_GoalTracking_Env`

Added
^^^^^

* Added hebi reward term :func:`ext.envs.cfgs.robots.hebi.mdp.rewards.orientation_command_error_tanh`
* Added experiments run script :file:`workflows.experiments.strategy4_scale_experiments.py`
* Added experiments :file:`envs.tasks.manipulations.track_goal.config.hebi.strategy4_scale_experiments.py`

0.1.5 (2024-07-06)
~~~~~~~~~~~~~~~~~~


Added
^^^^^

* Added experiments run script :file:`workflows.experiments.actuator_experiments.py`
* Added experiments run script :file:`workflows.experiments.agent_update_frequency_experiments.py` 
* Added experiments run script :file:`workflows.experiments.decimation_experiments.py`
* Added experiments run script :file:`workflows.experiments.strategy3_scale_experiments.py`
* Added experiments :file:`envs.tasks.manipulations.track_goal.config.hebi.agent_update_rate_experiments.py`
* Added experiments :file:`envs.tasks.manipulations.track_goal.config.hebi.decimation_experiments.py`
* Added experiments :file:`envs.tasks.manipulations.track_goal.config.hebi.strategy3_scale_experiments.py`
* Modified :file:`envs.tasks.manipulations.track_goal.config.hebi.agents.rsl_rl_agent_cfg`, and 
  :file:`envs.tasks.manipulations.track_goal.config.hebi.__init__` with logging name consistent to experiments 


0.1.4 (2024-07-05)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* :const:`ext.envs.cfgs.robots.hebi.robot_cfg.HEBI_STRATEGY3_CFG`
  :const:`ext.envs.cfgs.robots.hebi.robot_cfg.HEBI_STRATEGY4_CFG`
  changed from manually editing scaling factor to cfg specifying scaling factor. 
* :const:`ext.envs.cfgs.robots.hebi.robot_cfg.robot_dynamic`
* :func:`workflows.teleoperation.teleop_se3_agent_absolute.main` added visualization for full gloves data

0.1.3 (2024-06-29)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* updated :func:`workflows.teleoperation.teleop_se3_agent_absolute.main` gloves device to match updated
  requirement needed for rokoko gloves. New version can define port usage, output parts




0.1.2 (2024-06-28)
~~~~~~~~~~~~~~~~~~


Changed
^^^^^^^

* Restructured lab to accomodate new extension lab environmnets
* renamed the repository from lab.tycho to lab.envs
* removed :func:`workflows.teleoperation.teleop_se3_agent_absolute_leap.main` as it has been integrated 
  into :func:`workflows.teleoperation.teleop_se3_agent_absolute.main` 


0.1.1 (2024-06-27)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* teleoperation absolute ik control for leap hand at :func:`workflows.teleoperation.teleop_se3_agent_absolute_leap.main`


0.1.0 (2024-06-11)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Performed tycho migration. Done with Tasks: cake, liftcube, clock, meat, Goal Tracking
* Need to check: meat seems to have a bit of issue
* Plan to do: Learn a mujoco motor model, test out dreamerv3, refactorization continue
