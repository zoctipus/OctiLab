# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Flag for pyright to ignore type errors in this file.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.core.utils.types import ArticulationActions
from ...actuators import HebiStrategy3Actuator, HebiStrategy4Actuator
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets.articulation.articulation import Articulation
if TYPE_CHECKING:
    from .articulation_cfg import HebiArticulationCfg


class HebiArticulation(Articulation):
    """An articulation asset class.

    An articulation is a collection of rigid bodies connected by joints. The joints can be either
    fixed or actuated. The joints can be of different types, such as revolute, prismatic, D-6, etc.
    However, the articulation class has currently been tested with revolute and prismatic joints.
    The class supports both floating-base and fixed-base articulations. The type of articulation
    is determined based on the root joint of the articulation. If the root joint is fixed, then
    the articulation is considered a fixed-base system. Otherwise, it is considered a floating-base
    system. This can be checked using the :attr:`Articulation.is_fixed_base` attribute.

    For an asset to be considered an articulation, the root prim of the asset must have the
    `USD ArticulationRootAPI`_. This API is used to define the sub-tree of the articulation using
    the reduced coordinate formulation. On playing the simulation, the physics engine parses the
    articulation root prim and creates the corresponding articulation in the physics engine. The
    articulation root prim can be specified using the :attr:`AssetBaseCfg.prim_path` attribute.

    The articulation class is a subclass of the :class:`RigidObject` class. Therefore, it inherits
    all the functionality of the rigid object class. In case of an articulation, the :attr:`root_physx_view`
    attribute corresponds to the articulation root view and can be used to access the articulation
    related data. The :attr:`body_physx_view` attribute corresponds to the rigid body view of the articulated
    links and can be used to access the rigid body related data.

    The articulation class also provides the functionality to augment the simulation of an articulated
    system with custom actuator models. These models can either be explicit or implicit, as detailed in
    the :mod:`omni.isaac.lab.actuators` module. The actuator models are specified using the
    :attr:`ArticulationCfg.actuators` attribute. These are then parsed and used to initialize the
    corresponding actuator models, when the simulation is played.

    During the simulation step, the articulation class first applies the actuator models to compute
    the joint commands based on the user-specified targets. These joint commands are then applied
    into the simulation. The joint commands can be either position, velocity, or effort commands.
    As an example, the following snippet shows how this can be used for position commands:

    .. code-block:: python

        # an example instance of the articulation class
        my_articulation = Articulation(cfg)

        # set joint position targets
        my_articulation.set_joint_position_target(position)
        # propagate the actuator models and apply the computed commands into the simulation
        my_articulation.write_data_to_sim()

        # step the simulation using the simulation context
        sim_context.step()

        # update the articulation state, where dt is the simulation time step
        my_articulation.update(dt)

    .. _`USD ArticulationRootAPI`: https://openusd.org/dev/api/class_usd_physics_articulation_root_a_p_i.html

    """

    cfg: HebiArticulationCfg
    """Configuration instance for the articulations."""

    def __init__(self, cfg: HebiArticulationCfg):
        """Initialize the articulation.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

    """
    Operations - Writers. OVERRIDE
    """

    def write_joint_stiffness_to_sim(
        self,
        stiffness: torch.Tensor | float,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write joint stiffness into the simulation.

        Args:
            stiffness: Joint stiffness. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the stiffness for. Defaults to None (all joints).
            env_ids: The environment indices to set the stiffness for. Defaults to None (all environments).
        """
        # note: This function isn't setting the values for actuator models. (#128)
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None or isinstance(joint_ids, slice):
            # joint_ids = slice(None)
            joint_ids = list(range(self._data.joint_stiffness.shape[1]))
        # set into internal buffers
        self._data.joint_stiffness[env_ids[:, None], joint_ids] = stiffness
        # set into simulation
        self.root_physx_view.set_dof_stiffnesses(self._data.joint_stiffness.cpu(), indices=physx_env_ids.cpu())

    def write_joint_damping_to_sim(
        self,
        damping: torch.Tensor | float,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write joint damping into the simulation.

        Args:
            damping: Joint damping. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the damping for.
                Defaults to None (all joints).
            env_ids: The environment indices to set the damping for.
                Defaults to None (all environments).
        """
        # note: This function isn't setting the values for actuator models. (#128)
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None or isinstance(joint_ids, slice):
            # joint_ids = slice(None)
            joint_ids = list(range(self._data.joint_damping.shape[1]))
        # set into internal buffers
        self._data.joint_damping[env_ids[:, None], joint_ids] = damping
        # set into simulation
        self.root_physx_view.set_dof_dampings(self._data.joint_damping.cpu(), indices=physx_env_ids.cpu())

    def write_joint_armature_to_sim(
        self,
        armature: torch.Tensor | float,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write joint armature into the simulation.

        Args:
            armature: Joint armature. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the joint torque limits for. Defaults to None (all joints).
            env_ids: The environment indices to set the joint torque limits for. Defaults to None (all environments).
        """
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None or isinstance(joint_ids, slice):
            # joint_ids = slice(None)
            joint_ids = list(range(self._data.joint_armature.shape[1]))
        # set into internal buffers
        self._data.joint_armature[env_ids[:, None], joint_ids] = armature
        # set into simulation>
        self.root_physx_view.set_dof_armatures(self._data.joint_armature.cpu(), indices=physx_env_ids.cpu())

    def write_joint_friction_to_sim(
        self,
        joint_friction: torch.Tensor | float,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write joint friction into the simulation.

        Args:
            joint_friction: Joint friction. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the joint torque limits for. Defaults to None (all joints).
            env_ids: The environment indices to set the joint torque limits for. Defaults to None (all environments).
        """
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None or isinstance(joint_ids, slice):
            # joint_ids = slice(None)
            joint_ids = list(range(self._data.joint_friction.shape[1]))
        # set into internal buffers
        self._data.joint_friction[env_ids[:, None], joint_ids] = joint_friction
        # set into simulation
        self.root_physx_view.set_dof_friction_coefficients(self._data.joint_friction.cpu(), indices=physx_env_ids.cpu())

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.root_state_w[env_ids, :7] = root_pose.clone()
        # convert root quaternion from wxyz to xyzw
        root_poses_xyzw = self._data.root_state_w[:, :7].clone()
        root_poses_xyzw[:, 3:] = math_utils.convert_quat(root_poses_xyzw[:, 3:], to="xyzw")
        # set into simulation
        self.root_physx_view.set_root_transforms(root_poses_xyzw, indices=physx_env_ids)

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.root_state_w[env_ids, 7:] = root_velocity.clone()
        # set into simulation
        self.root_physx_view.set_root_velocities(self._data.root_state_w[:, 7:], indices=physx_env_ids)

    def _apply_actuator_model(self):
        """Processes joint commands for the articulation by forwarding them to the actuators.

        The actions are first processed using actuator models. Depending on the robot configuration,
        the actuator models compute the joint level simulation commands and sets them into the PhysX buffers.
        """
        # process actions per group
        for actuator in self.actuators.values():
            # prepare input for actuator model based on cached data
            # TODO : A tensor dict would be nice to do the indexing of all tensors together
            control_action = ArticulationActions(
                joint_positions=self._data.joint_pos_target[:, actuator.joint_indices],
                joint_velocities=self._data.joint_vel_target[:, actuator.joint_indices],
                joint_efforts=self._data.joint_effort_target[:, actuator.joint_indices],
                joint_indices=actuator.joint_indices,
            )
            # compute joint command from the actuator model
            if isinstance(actuator, HebiStrategy3Actuator) or isinstance(actuator, HebiStrategy4Actuator):
                control_action = actuator.compute(
                    control_action,
                    joint_pos=self._data.joint_pos[:, actuator.joint_indices],
                    joint_vel=self._data.joint_vel[:, actuator.joint_indices],
                    joint_effort=self._data.applied_torque[:, actuator.joint_indices]
                )
            else:
                control_action = actuator.compute(
                    control_action,
                    joint_pos=self._data.joint_pos[:, actuator.joint_indices],
                    joint_vel=self._data.joint_vel[:, actuator.joint_indices],
                )
            # update targets (these are set into the simulation)
            if control_action.joint_positions is not None:
                self._joint_pos_target_sim[:, actuator.joint_indices] = control_action.joint_positions
            if control_action.joint_velocities is not None:
                self._joint_vel_target_sim[:, actuator.joint_indices] = control_action.joint_velocities
            if control_action.joint_efforts is not None:
                self._joint_effort_target_sim[:, actuator.joint_indices] = control_action.joint_efforts
            # update state of the actuator model
            # -- torques
            self._data.computed_torque[:, actuator.joint_indices] = actuator.computed_effort
            self._data.applied_torque[:, actuator.joint_indices] = actuator.applied_effort
            # -- actuator data
            self._data.soft_joint_vel_limits[:, actuator.joint_indices] = actuator.velocity_limit
            # TODO: find a cleaner way to handle gear ratio. Only needed for variable gear ratio actuators.
            if hasattr(actuator, "gear_ratio"):
                self._data.gear_ratio[:, actuator.joint_indices] = actuator.gear_ratio
