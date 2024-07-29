Changelog
---------

0.3.0 (2024-07-29)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* updated dependency and meta information to isaac sim 4.1.0


0.2.0 (2024-07-29)
~~~~~~~~~~~~~~~~~~

Added
^^^^^^^

* Created new folder storing `octi.lab_asset.unitree` extensions



0.1.3 (2024-07-27)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Bug fix at :const:`octi.lab_assets.leap.actions.LEAP_JOINT_POSITION`
  and :const:`octi.lab_assets.leap.actions.LEAP_JOINT_EFFORT` because
  previous version did not include all joint name. it used to be 
  `joint_names=["j.*"]` now becomes `joint_names=["w.*", "j.*"]`




0.1.2 (2024-07-27)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Created new folder storing `octi.lab_asset.anymal`
* Created new folder storing `octi.lab_asset.leap`


0.1.1 (2024-07-26)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Created new folder storing `octi.lab_asset.tycho`


0.1.0 (2024-07-25)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Created new folder storing `octi.lab_asset`
