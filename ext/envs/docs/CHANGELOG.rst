Changelog
---------

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
