Changelog
---------

0.2.3 (2024-07-27)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* sketched Fetching as a separate locomotion task, instead of being a part of
  :folder:`octi.lab_tasks.tasks.locomotion.velocity`


0.2.2 (2024-07-27)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* dropped dependency of :folder:`octi.lab_tasks.cfg` in favor of extension `octi.lab_assets`



0.2.1 (2024-07-27)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* added Octi as author and maintainer to :file:`octi.lab_tasks.setup.py`

0.2.0 (2024-07-14)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* added support for register gym environment with MultiConstraintDifferentialIKController for leap_hand_xarm at 
  :file:`octi.lab_tasks.tasks.maniputation.lift_cube.config.leap_hand_xarm.__init__`


0.2.0 (2024-07-14)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* added leap hand xarm reward :func:`octi.lab_tasks.cfgs.robots.leap_hand_xarm.mdp.rewards.reward_fingers_object_distance`
* tuned liftCube environment reward function for LeapHandXarm environments 
  reward_fingers_object_distance scale was 1.5, now 5
  reward_object_ee_distance scale was 1, now 3
  reward_fingers_object_distance tanh return std was 0.1 now 0.2

0.1.9 (2024-07-13)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* added leap hand xarm reward :func:`octi.lab_tasks.cfgs.robots.leap_hand_xarm.mdp.rewards.reward_cross_finger_similarity`
* added leap hand xarm reward :func:`octi.lab_tasks.cfgs.robots.leap_hand_xarm.mdp.rewards.reward_intra_finger_similarity`
* added leap hand xarm event :func:`octi.lab_tasks.cfgs.robots.leap_hand_xarm.mdp.events.reset_joints_by_offset` which accepts
  additional joint ids
* changed cube lift environment cube size to be a bit larger
* added mass randomization cfg in cube lift environment :field:`octi.lab_tasks.tasks.manipulation.lift_cube.`


0.1.8 (2024-07-12)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* added leap hand xarm robot cfg and dynamic at :file:`octi.lab_tasks.cfgs.robots.leap_hand.robot_cfg.py` and 
  :file:`octi.lab_tasks.cfgs.robots.leap_hand_xarm.robot_dynamics.py`
* added environment :file:`octi.lab_tasks.tasks.manipulation.lift_cube.track_goal.config.leap_hand_xarm.LeapHandXarm_JointPos_GoalTracking_Env.py`
* added environment :file:`octi.lab_tasks.tasks.manipulation.lift_cube.lift_cube.config.leap_hand_xarm.LeapHandXarm_JointPos_LiftCube_Env.py`


0.1.7 (2024-07-08)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Hebi Gravity Enabled now becomes default
* orbid_mdp changed to lab_mdp in :file:`octi.lab_tasks.cfgs.robots.leap_hand.robot_dynamics.py`
* Removed Leap hand standard ik absolute and ik delta in :file:`octi.lab_tasks.cfgs.robots.leap_hand.robot_dynamics.py`
* Reflect support of RokokoGloveKeyboard in :func:`workflows.teleoperation.teleop_se3_agent_absolute.main`


Added
^^^^^
* Added experiments run script :file:`workflows.experiments.idealpd_experiments.py`
* Added experiments :file:`octi.lab_tasks.tasks.manipulation.track_goal.config.hebi.idealpd_scale_experiments.py`


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

* Changed :class:`octi.lab_tasks.cfgs.robots.hebi.robot_dynamics.RobotTerminationsCfg` to include DoneTerm: robot_extremely_bad_posture
* Changed :function:`octi.lab_tasks.cfgs.robots.hebi.mdp.terminations.terminate_extremely_bad_posture` to be probabilistic
* Changed :field:`octi.lab_tasks.tasks.manipulation.track_goal.config.hebi.Hebi_JointPos_GoalTracking_Env.RewardsCfg.end_effector_position_tracking`
  and :field:`octi.lab_tasks.tasks.manipulation.track_goal.config.hebi.Hebi_JointPos_GoalTracking_Env.RewardsCfg.end_effector_orientation_tracking`
  to be incentive reward instead of punishment reward.
* Renamed orbit_mdp to lab_mdp in :file:`octi.lab_tasks.tasks.manipulation.track_goal.config.Hebi_JointPos_GoalTracking_Env`

Added
^^^^^

* Added hebi reward term :func:`octi.lab_tasks.cfgs.robots.hebi.mdp.rewards.orientation_command_error_tanh`
* Added experiments run script :file:`workflows.experiments.strategy4_scale_experiments.py`
* Added experiments :file:`octi.lab_tasks.tasks.manipulation.track_goal.config.hebi.strategy4_scale_experiments.py`

0.1.5 (2024-07-06)
~~~~~~~~~~~~~~~~~~


Added
^^^^^

* Added experiments run script :file:`workflows.experiments.actuator_experiments.py`
* Added experiments run script :file:`workflows.experiments.agent_update_frequency_experiments.py` 
* Added experiments run script :file:`workflows.experiments.decimation_experiments.py`
* Added experiments run script :file:`workflows.experiments.strategy3_scale_experiments.py`
* Added experiments :file:`octi.lab_tasks.tasks.manipulation.track_goal.config.hebi.agent_update_rate_experiments.py`
* Added experiments :file:`octi.lab_tasks.tasks.manipulation.track_goal.config.hebi.decimation_experiments.py`
* Added experiments :file:`octi.lab_tasks.tasks.manipulation.track_goal.config.hebi.strategy3_scale_experiments.py`
* Modified :file:`octi.lab_tasks.tasks.manipulation.track_goal.config.hebi.agents.rsl_rl_agent_cfg`, and 
  :file:`octi.lab_tasks.tasks.manipulation.track_goal.config.hebi.__init__` with logging name consistent to experiments 


0.1.4 (2024-07-05)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* :const:`octi.lab_tasks.cfgs.robots.hebi.robot_cfg.HEBI_STRATEGY3_CFG`
  :const:`octi.lab_tasks.cfgs.robots.hebi.robot_cfg.HEBI_STRATEGY4_CFG`
  changed from manually editing scaling factor to cfg specifying scaling factor. 
* :const:`octi.lab_tasks.cfgs.robots.hebi.robot_cfg.robot_dynamic`
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
