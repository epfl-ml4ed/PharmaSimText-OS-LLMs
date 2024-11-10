def evaluate_episode_RL(agent, env, policy):
    """
    Evaluates a single episode for an agent interacting with an environment using a given policy.

    Args:
        agent: The agent to evaluate, expected to have a method to reset state dictionaries and perform actions.
        env: The environment the agent interacts with, expected to support reset and step functions.
        policy: The policy to guide the agent's actions.

    Returns:
        tuple: Contains episode score, list of transitions, trajectory score, efficiency score,
               scenario name, and interaction history.
    """
    episode = []  # To record the sequence of transitions during the episode
    history = []  # To maintain a history of observations and actions
    step = 0  # Step counter
    score = 0  # Accumulates total score across steps
    done = False  # Indicates when the episode is complete
    agent.reset_dictionaries()  # Resets any stateful dictionaries in the agent
    ob, valid_acts, hc = env.reset()  # Reset environment; `ob` is observation, `valid_acts` are valid actions, `hc` is any helper context
    history.append(ob[1])  # Add the first observation to history
    scenario_name = env.scenario["name"]  # Get the name of the current scenario
    state = agent.create_state(update_sentence=ob,
                               hc=hc)  # Create initial state for the agent based on the observation and context

    while not done:
        transition = [env.scenario["name"], step, ob[1]]  # Start recording transition information
        valid_ids = agent.encode_actions(valid_acts)  # Encode valid actions into a suitable format
        _, action_idx, action_values, _ = agent.act(
            [state], [valid_ids], policy=policy, eval_mode=True, action_strs=valid_acts)  # Agent selects an action
        action_idx = action_idx[0]
        action_str = valid_acts[action_idx]  # Retrieve the selected action string

        # Perform the action in the environment and receive the new state and reward
        state_update, rew, done, valid_acts, hc, traj_score = env.step(ob, action_str)

        if not done:
            trace = env.trace  # Update trace if the episode is not done
        history.append(action_str["sentence"])  # Add the action to the history

        if not done:
            # Update history based on state type (interaction or other)
            if state_update[0] == "interaction":
                history.append(".".join(state_update[1].split(".")[2:]))
            else:
                history.append(".".join(state_update[1].split(".")[-1:]))

        ob = state_update  # Update observation to new state
        score += rew  # Accumulate reward
        step += 1  # Increment step counter
        transition += [action_str, rew, score]  # Complete transition recording
        episode.append(transition)  # Add the transition to the episode
        state = agent.create_state(update_sentence=ob, hc=hc, previous_state=state)  # Update agent's state

    # Calculate trajectory score and efficiency score
    traj_score = sum(
        a in trace for a in env.scenario["present_actions"]) / len(env.scenario["present_actions"])
    eff_score = sum(
        a in trace for a in env.scenario["present_actions"]) / len(trace)

    agent.reset_dictionaries()  # Reset agent dictionaries for next episode
    return score, episode, traj_score, eff_score, scenario_name, history  # Return evaluation results
