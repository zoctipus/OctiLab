import gym
from .Brain.vec_agent_os import DSACAgent
from .Common import Logger, get_params
import numpy as np
import torch
from tqdm import tqdm

def set_seed(seed):
    seed = int(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    import torch;
    torch.manual_seed(seed)

def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])

def concat_state_latent_batch(s, z_, n):
    z_one_hot = np.zeros((s.shape[0], n))
    z_one_hot[:, z_] = 1
    return np.concatenate([s, z_one_hot], axis=1)

def gym_make(env_name):
    if 'Tycho' in env_name:
        from tycho_env import TychoEnvVec
        env = TychoEnvVec(
            config={"action_space":"eepose-delta",
                    'ee_workspace_low': [-0.45,-0.30,0.02,0, -1, 0, 0,-0.57],           # lock the orientation
                    'ee_workspace_high':[-0.37,-0.27,0.11,0, -1, 0, 0,-0.2],            # lock the orientation
                    "action_low": [-0.02, -0.02, -0.02, 0,0,0,0, -0.5],                 # lock the orientation
                    "action_high": [0.02, 0.02, 0.02, 0,0,0,0, 0.5],                    # lock the orientation
                    "static_ball":False,
                    "dr":True,
                    "reset_eepose": [-0.385, -0.28, 0.05,0,-1,0,0, -0.36],
                    "normalized_action": True,
            })
        return env
    if env_name == 'PointMass':
        from Env import PointMass2D
        return PointMass2D(
            acceleration=False,
            reward_type='smooth',
            wind=None) #np.array([-0.3, 0.3]))
    else:
        return gym.make(env_name)

if __name__ == "__main__":
    params = get_params()

    set_seed(params["seed"])

    if 'SAC' in params["agent_name"]:
        params.update({'reward_epsilon' : 10000000})
    print(params["reward_epsilon"])

    env = gym_make(params["env_name"])
    env.seed(params["seed"])
    n_states = env.observation_space.shape[1:]
    n_actions = env.action_space.shape[1:]
    action_bounds = [env.action_space.low, env.action_space.high]
    try:
        max_n_steps = min(params["max_episode_len"], env.spec.max_episode_steps)
    except:
        max_n_steps = min(params["max_episode_len"], env._max_episode_steps)

    params.update({"n_states": n_states,
                   "n_actions": n_actions,
                   "action_bounds": action_bounds,
                   "max_n_steps": max_n_steps})
    print("params:", params)

    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    agent = DSACAgent(p_z=p_z, **params)
    logger = Logger(agent, **params)

    if params["datafile"]:
        with open(params["datafile"], 'rb') as f:
            episodes = np.load(f, allow_pickle=True)
            for episode in episodes:
                z = np.random.choice(params["n_skills"], p=p_z)
                obs, act, rew, done = episode["obs"], episode["act"], episode["rew"], episode["done"]
                obs_z = concat_state_latent_batch(obs, z, params["n_skills"])
                for t in range(len(obs) - 1):
                    agent.vec_store(obs_z[t], z, done[t], act[t], obs_z[t+1], rew[t])

    agent.logger = logger

    test_env = gym_make(params["env_name"])
    test_env.seed(params["seed"])

    if params["do_train"]:

        if not params["train_from_scratch"]:
            episode, total_sample_step, last_logq_zs, np_rng_state, *env_rng_states, torch_rng_state, random_rng_state = logger.load_weights()
            agent.hard_update_all_target_networks()
            min_episode = episode
            np.random.set_state(np_rng_state)
            env.np_random.set_state(env_rng_states[0])
            env.observation_space.np_random.set_state(env_rng_states[1])
            env.action_space.np_random.set_state(env_rng_states[2])
            agent.set_rng_states(torch_rng_state, random_rng_state)
            print("Keep training from previous run.")

        else:
            min_episode = 0
            total_sample_step = 0
            last_logq_zs = 0
            np.random.seed(params["seed"])
            env.seed(params["seed"])
            env.observation_space.seed(params["seed"])
            env.action_space.seed(params["seed"])
            print("Training from scratch.")

        logger.on()

        """
        Random Exploration
        """
        z = np.random.choice(params["n_skills"], p=p_z)
        state = env.reset()
        state = concat_state_latent_batch(state, z, params["n_skills"])
        step = 0
        episode_reward = 0
        while total_sample_step < params["random_explore_steps"]:
            action = env.action_space.sample()
            z = np.random.choice(params["n_skills"], p=p_z)
            next_state, reward, done, _ = env.step(action)
            next_state = concat_state_latent_batch(next_state, z, params["n_skills"])
            agent.vec_store(state, z, done, action, next_state, reward)
            state = next_state
            step += 1
            episode_reward += reward
            if done or step > max_n_steps:
                total_sample_step += step
                z = np.random.choice(params["n_skills"], p=p_z)
                state = env.reset()
                state = concat_state_latent_batch(state, z, params["n_skills"])
                step = 0
                print("Collected a trajectory using randomly sampled action, rew =", episode_reward)
                episode_reward = 0

        """
        Sampling from Env and Training
        """
        """
        episode = min_episode

        def reset_env(env):
            z = np.random.choice(params["n_skills"], p=p_z)
            state = env.reset()
            state = concat_state_latent(state, z, params["n_skills"])
            return z, state, 0, 0, []

        z, state, episode_reward, step, losses_list = reset_env(env)

        for total_sample_step in tqdm(range(params["max_n_steps"])):
            if episode >= params["max_n_episodes"]:
                break

            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = concat_state_latent(next_state, z, params["n_skills"])
            agent.store(state, z, done, action, next_state, reward)
            episode_reward += reward
            state = next_state
            step += 1

            losses = agent.train(diversity_reward=False) # TODO
            if losses is not None: losses_list.append(losses)

            for _step in range(params["utd"] - 1):
                agent.train_critic(diversity_reward=False)

            if done or step > max_n_steps:
                episode += 1

                if len(losses_list):
                    loss_dict = {k: np.average([dic[k] for dic in losses_list]) for k in losses_list[0].keys()}

                    logger.log(episode,
                               episode_reward,
                               z,
                               -loss_dict['Loss/discriminator_loss'], # log q(z|s)
                               total_sample_step,
                               np.random.get_state(),
                               env.np_random.get_state(),
                               env.observation_space.np_random.get_state(),
                               env.action_space.np_random.get_state(),
                               *agent.get_rng_states(),
                               )
                    logger.log_train(loss_dict, total_sample_step)

                # Evaluation - use deterministic action
                if episode % params["interval"] == 0 and params["eval_traj"] > 0:
                    eval_dict = {str(z) : np.zeros(params["eval_traj"]) for z in range(params["n_skills"])}
                    for z in range(params["n_skills"]):
                        for traj_counter in range(params["eval_traj"]):
                            s = test_env.reset()
                            s = concat_state_latent(s, z, params["n_skills"])
                            for _ in range(max_n_steps):
                                action = agent.choose_deterministic_action(s) # use deterministic action in eval
                                s_, r, done, _ = test_env.step(action)
                                s_ = concat_state_latent(s_, z, params["n_skills"])
                                eval_dict[str(z)][traj_counter] += r
                                if done: break
                                s = s_

                    eval_dict_statistics = {
                        f'Eval/Skill{z}' : np.average(eval_dict[str(z)]) for z in range(params["n_skills"])
                    }
                    logger.log_train(eval_dict_statistics, total_sample_step)

                # Reset
                z, state, episode_reward, step, losses_list = reset_env(env)
        """

        """
        Sampling from Env and Training
        """

        for episode in tqdm(range(1 + min_episode, params["max_n_episodes"] + 1)):
            z = np.random.choice(params["n_skills"], p=p_z)
            state = env.reset()
            state = concat_state_latent_batch(state, z, params["n_skills"])
            episode_reward = 0

            step = 1
            while True:
                action = agent.vec_choose_action(state)
                next_state, reward, done, _ = env.step(action)
                next_state = concat_state_latent_batch(next_state, z, params["n_skills"])

                state_t = torch.from_numpy(state).to('cpu')
                next_state_t = torch.from_numpy(next_state).to('cpu')
                action_t = torch.from_numpy(action).to('cpu')
                reward_t = torch.from_numpy(reward).to('cpu')
                done_t = torch.from_numpy(done).to('cpu')
                z_t = torch.tensor(z).to('cpu')
                agent.vec_store(state_t, z_t, done_t, action_t, next_state_t, reward_t)
                episode_reward += reward
                state = next_state
                step += 1
                if done or step > max_n_steps:
                    break

            total_sample_step += step

            # OSINAYN does not add diversity reward UNTIL after the policy is close to the optimal
            train_with_diversity = (episode_reward > params["reward_epsilon"])

            losses_list = []
            for _update_step in range(step):
                losses = agent.train(diversity_reward=train_with_diversity)
                if losses is not None: losses_list.append(losses)

                for _step in range(params["utd"] - 1):
                    agent.train_critic(diversity_reward=train_with_diversity)

            if len(losses_list):
                loss_dict = {k: np.average([dic[k] for dic in losses_list]) for k in losses_list[0].keys()}

                logger.log(episode,
                           episode_reward,
                           z,
                           -loss_dict['Loss/discriminator_loss'], # log q(z|s)
                           total_sample_step,
                           np.random.get_state(),
                        #    env.np_random.get_state(),
                        #    env.observation_space.np_random.get_state(),
                        #    env.action_space.np_random.get_state(),
                        #    *agent.get_rng_states(),
                           )
                logger.log_train(loss_dict, total_sample_step)

            # Evaluation - use deterministic action
            if episode % params["interval"] == 0 and params["eval_traj"] > 0:
                eval_dict = {str(z) : np.zeros(params["eval_traj"]) for z in range(params["n_skills"])}
                for z in range(params["n_skills"]):
                    for traj_counter in range(params["eval_traj"]):
                        s = test_env.reset()
                        s = concat_state_latent_batch(s, z, params["n_skills"])
                        for _ in range(max_n_steps):
                            action = agent.choose_deterministic_action(s) # use deterministic action in eval
                            s_, r, done, _ = test_env.step(action)
                            s_ = concat_state_latent_batch(s_, z, params["n_skills"])
                            eval_dict[str(z)][traj_counter] += r
                            if done:
                                break
                            s = s_

                eval_dict_statistics = {
                    f'Eval/Skill{z}' : np.average(eval_dict[str(z)]) for z in range(params["n_skills"])
                }
                logger.log_train(eval_dict_statistics, total_sample_step)

            # Quit if collected enough steps
            if (total_sample_step >= params["max_n_steps"] and episode % params["interval"] == 0):
                logger._save_weights(episode, total_sample_step, *agent.get_rng_states())
                break

        logger._save_weights(episode, total_sample_step, *agent.get_rng_states())

    from Common import Play
    logger.load_weights(load_replay_buffer=False)
    eval_env = gym_make(params["env_name"])
    eval_env.seed(params["seed"])
    player = Play(test_env, agent, n_skills=params["n_skills"])
    env_name=params["env_name"]
    agent_name=params["agent_name"]
    player.evaluate(folder_name=f"Vid/{env_name}/{agent_name}", vid_per_skill=4, per_skill_seed=params["seed"])
