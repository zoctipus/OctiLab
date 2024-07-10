
# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class Base_PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 96
    max_iterations = 30000
    save_interval = 100
    resume=False
    experiment_name = "Hebi_base_agent"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Emprical_Normalized_PPORunnerCfg(Base_PPORunnerCfg):
    experiment_name = "emperical_normalized_ppo"

@configclass
class Hebi_IkDeltaDls_PwmMotor_CraneberryLavaChocoCake_PPORunnerCfg(Base_PPORunnerCfg):
    experiment_name = "Hebi_IkDeltaDls_PwmMotor_CraneberryLavaChocoCake"

@configclass
class Hebi_IkDeltaDls_IdealPD_CraneberryLavaChocoCake_PPORunnerCfg(Base_PPORunnerCfg):
    experiment_name = "Hebi_HebiIkDeltaDls_IdealPD_CraneberryLavaChocoCake"

@configclass
class Hebi_IkAbsoluteDls_PwmMotor_CraneberryLavaChocoCake_PPORunnerCfg(Base_PPORunnerCfg):
    experiment_name = "Hebi_IkAbsoluteDls_PwmMotor_CraneberryLavaChocoCake"
