import numpy as np
import torch
import gym
import argparse
import os
import pandas as pd
from captum.attr import LayerLRP

import utils
import TD3
import OurDDPG
import DDPG


# Runs policy for X episodes and returns average reward and data
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    data = {
        "Rewards": [],
        "States": [],
        "Actions": [],
        "Next States": [],
        "Dones": [],
        "R(states->Critic)": [],
        "R(actions->Critic)": [],
        "R(states->Actor)": []
    }

    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            next_state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

            # Append results for each 10 time steps to the lists
            data["Rewards"].append(reward)
            data["States"].append(state)
            data["Actions"].append(action)
            data["Next States"].append(next_state)
            data["Dones"].append(done)

            # Use LayerLRP to compute relevance scores for Q-values and actions
            lrp = LayerLRP(policy.critic)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            Q1_tensor, Q2_tensor = policy.critic(state_tensor, action_tensor)
            R_Q1 = lrp.attribute(state_tensor, target=Q1_tensor)
            R_Q2 = lrp.attribute(state_tensor, target=Q2_tensor)
            data["R(states->Critic)"].append(R_Q1.detach().numpy())
            data["R(actions->Critic)"].append(R_Q2.detach().numpy())

            # Compute relevance scores for actions to actor
            lrp_actor = LayerLRP(policy.actor)
            R_states_to_actor = lrp_actor.attribute(state_tensor)
            data["R(states->Actor)"].append(R_states_to_actor.detach().numpy())

            state = next_state

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")

    return avg_reward, data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")
    parser.add_argument("--env", default="HalfCheetah-v2")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_timesteps", default=25e3, type=int)
    parser.add_argument("--eval_freq", default=5e3, type=int)
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument("--expl_noise", default=0.1, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--policy_noise", default=0.2)
    parser.add_argument("--noise_clip", default=0.5)
    parser.add_argument("--policy_freq", default=2, type=int)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--load_model", default="")
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    # Create an empty DataFrame to store the collected data
    df_data = {
        "Rewards": [],
        "States": [],
        "Actions": [],
        "Next States": [],
        "Dones": [],
        "R(states->Critic)": [],
        "R(actions->Critic)": [],
        "R(states->Actor)": []
    }
    df = pd.DataFrame(df_data)

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        # Append results for each 10 time steps to the DataFrame
        if (t + 1) % 10 == 0:
            df_data["Rewards"].append(episode_reward)
            df_data["States"].append(state)
            df_data["Actions"].append(action)
            df_data["Next States"].append(next_state)
            df_data["Dones"].append(done)
            df_data["R(states->Critic)"].append(R_Q1.detach().numpy())
            df_data["R(actions->Critic)"].append(R_Q2.detach().numpy())
            df_data["R(states->Actor)"].append(R_states_to_actor.detach().numpy())

            # Update the DataFrame with the new data
            df = pd.DataFrame(df_data)

        if done:
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model:
                policy.save(f"./models/{file_name}")
