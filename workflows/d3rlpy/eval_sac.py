# test
# import d4rl
import gym
import time
import numpy as np
import pickle
#loading JAX first for correct cudnn
import jax
jax.random.PRNGKey(42)
import torch
import argparse
from tycho_env import TychoEnv, TychoEnv_push# prepare environment
import cv2 
from tycho_env.utils.camera_util import get_camera_transform_matrix, project_points_from_world_to_camera
parser = argparse.ArgumentParser(
    description='test sac')
parser.add_argument('--load', '-l', type=str,
                    required=False, help='location to restore policy')
parser.add_argument('-st', '--save_traj', type=bool,default=False,
                    required=False, help='save trajectory for RL training')
parser.add_argument('--seed', '-s', type=int,
                    required=False, help='seed')
parser.add_argument("-e", "--env", default="tycho", help="Environment name. Use \"tycho\" for tycho env. (default tycho)")
parser.add_argument("-n", "--n_trials", type=int, default=100, help="Number of trials to conduct. (default 100)")
parser.add_argument("-H", "--horizon", type=int, help="Horizon for each trial. Defaults to none.")
parser.add_argument("-r", "--render", action="store_true", help="Render rollout")
args, overrides = parser.parse_known_args()



reach, grasp_cnt, reach2 = False, 0, False
def unscale_action(action_space, scaled_action):
    """
    Rescale the action from [-1, 1] to [low, high]
    (no need for symmetric action space)
    :param action_space: (gym.spaces.box.Box)
    :param action: (np.ndarray)
    :return: (np.ndarray)
    """
    low, high = action_space.low, action_space.high
    return low + (0.5 * (scaled_action + 1.0) * (high - low + 1e-8))
def hack_controller_needle(env,obs,cnt):
    global reach, grasp_cnt, reach2

    eepose = obs[0:7]
    needle_tip_0 = env.data.site_xpos[7] + [-0.0025,0,0]
    needle_tip_1 = env.data.site_xpos[6]
    needle_tip_2 = env.data.site_xpos[5]

    center = np.array([-0.38, -0.26, 0.032]) #self.goal
    

    distance_to_goal1 = np.linalg.norm(needle_tip_0 - center)
    distance_to_goal2 = np.linalg.norm(needle_tip_2 - center)
    dist_z = np.linalg.norm(needle_tip_0[2] - center[2])

    act = np.zeros(8,dtype=np.float32)
    act[-1] = 0.0
    if dist_z > 0.0012 and not reach:
        for i in range(3):
            act[i] = (center-needle_tip_0)[i] * (2)
        # act[1] -= 0.01
        act[2] += 0.012
        # print(dist_z)
    elif (distance_to_goal1 < 0.005 or reach is True )and grasp_cnt < 8:
    #     # print(grasp_cnt)
        # print("here")
        reach = True
        for i in range(3):
            act[i] = (center-needle_tip_2)[i] * (2)
        # act[1] += 0.003
        # act[-1] = -0.5
        act[2] += 0.023
        grasp_cnt += 1
    elif distance_to_goal2 < 0.005 or reach2 is True:
        reach2 = True
        # print("idel")
        # for i in range(3):
        #     act[i] = (env.goal-obj_pos)[i] * (1)
        act[0] -= 0.005
        act[2] += 0.011
        # print(act)
    # print(distance_to_goal2 <= 0.005)
    return act
def hack_controller_20hz(env,cnt):
    global reach, grasp_cnt
    qpos = env.data.qpos.flat.copy()
    obj_pos = qpos[7:10]
    obj_pos[2] -=0.01
    chop_tip = env.data.site_xpos[0]
    rot_chop_tip = env.data.site_xpos[3]
    mid_pos = (rot_chop_tip + chop_tip) / 2.0
    mid_pos[0] += 0.025
    mid_pos[1] += 0.005
    mid_pos[2] -= 0.0095
    distance_to_obj = np.linalg.norm(obj_pos - mid_pos)
    distance_to_goal = np.linalg.norm(obj_pos - env.goal)

    dist_z = np.linalg.norm(obj_pos[2] - mid_pos[2])

    act = np.zeros(8,dtype=np.float32)
    act[-1] = 0.0
    if dist_z > 0.0012 and not reach:
        for i in range(3):
            act[i] = (obj_pos-mid_pos)[i] * (2)
        # act[1] -= 0.01
        act[2] += 0.012
        # print(dist_z)
    elif cnt > 6  and distance_to_goal > 0.01 and grasp_cnt < 8:
        # print(grasp_cnt)
        reach = True
        for i in range(3):
            act[i] = (obj_pos-mid_pos)[i] * (2)
        act[1] += 0.003
        act[-1] = -0.5
        act[2] += 0.013
        grasp_cnt += 1
    elif grasp_cnt >= 8 and distance_to_goal > 0.005:
        for i in range(3):
            act[i] = (env.goal-obj_pos)[i] * (1)
        act[-1] = -0.1
        act[2] += 0.02
        # print(act)
    return act
def hack_controller_15deg_20hz(env,cnt):
    global reach, grasp_cnt
    qpos = env.data.qpos.flat.copy()
    obj_pos = qpos[7:10]
    obj_pos[2] -=0.01
    chop_tip = env.data.site_xpos[0]
    rot_chop_tip = env.data.site_xpos[3]
    mid_pos = (rot_chop_tip + chop_tip) / 2.0
    # mid_pos[0] += 0.025
    # mid_pos[1] += 0.005
    # mid_pos[2] -= 0.0095

    mid_pos[0] += 0.025
    mid_pos[1] += 0.005
    mid_pos[2] -= 0.005

    distance_to_obj = np.linalg.norm(obj_pos - mid_pos)
    distance_to_goal = np.linalg.norm(obj_pos - env.goal)

    dist_z = np.linalg.norm(obj_pos[2] - mid_pos[2])

    act = np.zeros(8,dtype=np.float32)
    act[-1] = 0.0
    if dist_z > 0.0015 and not reach:
        for i in range(3):
            act[i] = (obj_pos-mid_pos)[i] * (2)
        # act[1] -= 0.01
        act[2] += 0.012
        #print(dist_z)
    elif cnt > 6  and distance_to_goal > 0.01 and grasp_cnt < 8:
        # print(grasp_cnt)
        reach = True
        for i in range(3):
            act[i] = (obj_pos-mid_pos)[i] * (1)
        act[1] += 0.001
        act[-1] = -0.5
        act[2] += 0.013
        grasp_cnt += 1
    elif grasp_cnt >= 8 and distance_to_goal > 0.005:
        for i in range(3):
            act[i] = (env.goal-obj_pos)[i] * (1)
        act[-1] = -0.1
        act[2] += 0.02
        # print(act)
    return act
def hack_controller_30deg_20hz(env,cnt):
    global reach, grasp_cnt
    qpos = env.data.qpos.flat.copy()
    obj_pos = qpos[7:10]
    obj_pos[2] -=0.01
    chop_tip = env.data.site_xpos[0]
    rot_chop_tip = env.data.site_xpos[3]
    mid_pos = (rot_chop_tip + chop_tip) / 2.0
    # mid_pos[0] += 0.025
    # mid_pos[1] += 0.005
    # mid_pos[2] -= 0.0095

    mid_pos[0] += 0.025
    mid_pos[1] += 0.005
    mid_pos[2] += 0.0012

    distance_to_obj = np.linalg.norm(obj_pos - mid_pos)
    distance_to_goal = np.linalg.norm(obj_pos - env.goal)

    dist_z = np.linalg.norm(obj_pos[2] - mid_pos[2])

    act = np.zeros(8,dtype=np.float32)
    act[-1] = 0.0
    if dist_z > 0.0045 and not reach:
        for i in range(3):
            act[i] = (obj_pos-mid_pos)[i] * (2)
        # act[1] -= 0.01
        act[2] += 0.012
        # print(dist_z)
    elif cnt > 6  and distance_to_goal > 0.01 and grasp_cnt < 8:
        # print(grasp_cnt)
        reach = True
        for i in range(3):
            act[i] = (obj_pos-mid_pos)[i] * (1.5)
        act[1] += 0.003
        act[-1] = -0.5
        act[2] += 0.01
        grasp_cnt += 1
    elif grasp_cnt >= 8 and distance_to_goal > 0.005:
        for i in range(3):
            act[i] = (env.goal-obj_pos)[i] * (1)
        act[-1] = -0.1
        act[2] += 0.02
        # print(act)
    return act
def hack_controller(env,cnt):
    global reach, grasp_cnt
    qpos = env.data.qpos.flat.copy()
    obj_pos = qpos[7:10]
    chop_tip = env.data.site_xpos[0]
    rot_chop_tip = env.data.site_xpos[3]
    mid_pos = (rot_chop_tip + chop_tip) / 2.0
    mid_pos[0] += 0.025
    mid_pos[1] += 0.005
    mid_pos[2] -= 0.0095
    distance_to_obj = np.linalg.norm(obj_pos - mid_pos)
    distance_to_goal = np.linalg.norm(obj_pos - env.goal)

    dist_z = np.linalg.norm(obj_pos[2] - mid_pos[2])

    act = np.zeros(8,dtype=np.float32)
    act[-1] = 0.0

    if dist_z > 0.012 and not reach:
        for i in range(3):
            act[i] = (obj_pos-mid_pos)[i] * (1)
        act[2] += 0.007

    elif cnt > 100 and distance_to_goal > 0.01 and grasp_cnt < 75:
        # print(grasp_cnt)
        reach = True
        for i in range(3):
            act[i] = (obj_pos-mid_pos)[i] * (1)
        act[-1] = -0.1
        act[2] += 0.007
        grasp_cnt += 1
    elif grasp_cnt >= 75 and distance_to_goal > 0.01:
        for i in range(3):
            act[i] = (env.goal-obj_pos)[i] * (1)
        act[-1] = -0.01
        act[2] += 0.005

    return act

def evaluate_on_environment(
    env: gym.Env, algo, n_trials: int = 10, render: bool = False):
    """Returns scorer function of evaluation on environment.
    """
    global reach, grasp_cnt
    episodes = []
    episode_rewards = []
    trail = 0
    while True:
        reach, grasp_cnt = False, 0
        episode = {}
        obs = []
        r = []
        d = []
        act = []
        observation = env.reset()
        episode_reward = 0.0
        cnt = 0
        while True:

            cnt += 1
            # take action
            action = algo.predict([observation]) # no rescaling necessary
            action = unscale_action(env.action_space, action)
            # action = hack_controller_needle(env, obs, cnt)
            # action = env.action_space.sample()
            # action = hack_controller_15deg_20hz(env,cnt)
            # action = hack_controller(env,cnt)
            # action = hack_controller_20hz(env,cnt)
            obs.append(observation.copy())
            act.append(action.copy())
            observation, reward, done, _ = env.step(action)
            # print(observation)
            # print(env.sim.get_state().qvel[7:10])
            # print(action)
            r.append(reward)
            d.append(done)
            # print(reward)
            episode_reward += reward
            if render:
                # pass
                env.render()
                # camera_height, camera_width = 480, 960
            #     world_to_camera_transform = get_camera_transform_matrix(env.sim,'wrist_camera',camera_height,camera_width)
            #     points = []
            #     points.append(env.data.site_xpos[7])
            #     points.append(env.data.site_xpos[6])
            #     points.append(env.data.site_xpos[5])
            #     # print(env.data.site_xpos[7])
            #     pts = project_points_from_world_to_camera(np.array(points), world_to_camera_transform,camera_height,camera_width)
                # img = env.render(mode='rgb_array', camera_name='wrist_camera',width=camera_width,
            #    height=camera_height,)[:,:,::-1].copy()
            #     # import pdb;pdb.set_trace()
            #     # for pt in pts:
            #     #     cv2.circle(img, (int(pt[1]), int(pt[0])), 5, (255, 0, 0), -1)
                # cv2.imwrite(f"imgs/eval_{cnt}.png",img) #save the images as .png



            if done:
                break
        print(episode_reward)
        # if reward > 0:

        episode = {"obs":np.array(obs,dtype=np.float32),"act":np.array(act,dtype=np.float32),"rew":np.array(r,dtype=np.float32),"done":np.array(d,dtype=np.float32)}
        print(cnt,reward)
        episodes.append(episode)
        episode_rewards.append(episode_reward)

        trail+=1
        print("trail:", trail)
        if trail >=n_trials:
            break
    if args.save_traj:
        np.save('demo_tapir',np.asarray(episodes))
    print(len(episode_rewards))
    if hasattr(env, "get_normalized_score"):
        print(f"Normalized reward: {100 * env.get_normalized_score(sum(episode_rewards)/len(episode_rewards)):.5f}")
    return float(np.mean(episode_rewards))

class D3Agent():
    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

    def load(self, model_folder, device):
        # load is handled at init
        pass
    # For 1-batch query only!
    def predict(self, sample):

        with torch.no_grad():
            input = torch.from_numpy(sample[0]).float().unsqueeze(0).to('cuda:0')
            at = self.policy(input)[0].to('cpu').detach().numpy()
        return at
import d3rlpy
torch.manual_seed(121)
d3rlpy.seed(121)

device='cuda:0'
policy = torch.jit.load(args.load)
policy.to(device)
agent = D3Agent(policy, device)
# agent = None
if args.env == "tycho":
    # env = TychoEnv(onlyPosition=False,config={"action_space":"eepose-delta","static_ball":False,"dr":False,
    #             'action_low': [-0.45,-0.30,0.02,0, 0.9914449, 0, 0.1305262,-0.57],
    #             'action_high':[-0.37,-0.27,0.11,0, 0.9914449, 0, 0.1305262,-0.2],
    #     })
    # env = TychoEnv_rope(onlyPosition=False,config={"state_space":"eepose-obj-vel", "action_space":"eepose-delta","static_ball":False,
    #             'action_low': [-0.5,-0.5,0.03,0,-1,0,0,-0.57],
    #             'action_high':[0.,-0.1,1.,0,-1,0,0,-0.2],
    #             "dr":False,
    #     })
    # no rotation restriction
    # env = TychoEnv(onlyPosition=False,config={"action_space":"eepose-delta","static_ball":False,"dr":False,
    #             'action_low': [-0.40,-0.30,0.04,-1,-1,-1,-1,-0.57],
    #             'action_high':[-0.37,-0.27,0.11,1,1,1,1,-0.2],
    #     })
    config = {
    "action_space":"eepose-delta",
    "action_low":[-0.46,-0.3,0.01,0,-1,0,0,-0.57],
    "action_high":[-0.37,-0.27,0.11,1e-8,-1,1e-8,1e-8,-0.2],
    # "dr":False,
    }
    config = {
    "action_space":"eepose-delta",
    "action_low":[-0.56,-0.6,0.01,0,-1,0,0,-0.57],
    "action_high":[-0.37,-0.1,0.11,1e-8,-1,1e-8,1e-8,-0.2],
    # "dr":False,
    }
    env = TychoEnv_push(onlyPosition=False,config={**config})
    # env = TychoEnv(onlyPosition=False,config={"action_space":"eepose-delta","static_ball":True,"dr":False,
    #             'action_low': [-0.40,-0.30,0.04,0,-1,0,0,-0.57],
    #             'action_high':[-0.37,-0.27,0.11,0,-1,0,0,-0.2],
    #     })
    # env = gym.wrappers.RescaleAction(env, np.full_like(env.action_space.low, -1), np.full_like(env.action_space.high, 1))
else:
    env = gym.make(args.env, disable_env_checker=True)
    if hasattr(env, "mj_render"):
        env.render = env.mj_render

if args.horizon is not None:
    env = gym.wrappers.TimeLimit(env, max_episode_steps=args.horizon)
env.seed(121)



print(evaluate_on_environment(env,agent,n_trials=args.n_trials,render=args.render))
