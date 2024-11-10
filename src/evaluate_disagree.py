import copy
import shutil
import numpy as np
import re
import json
import os
import torch
from src.evaluate_helper import separate, parse_string_to_dict, softmax, find_phrase, update_history, calculate_scores, \
    parse_chosen_action, RL_veto, get_recs, handle_veto, rl_agent_action, rl_llm_interaction, \
    rl_llm_recommendation_interaction
from src.evaluate_helper import load_summary
from clin_memory import summarize_ep
def evaluate_episode_LLM_Vetos(chooser, player, topk, agent, env, policy="softmax", prev_exp=None):
    """
    Evaluates an episode using the h1_v2 method, where actions are chosen based on an RL agent,
    with the option for an LLM-based veto or validation step.

    Args:
        chooser: The LLM chooser agent for suggesting actions.
        player: The LLM player agent for direct action recommendations.
        topk (int): Number of top choices to consider from the RL agent's action values.
        agent: The RL agent used for decision-making.
        env: The environment instance.
        policy (str): Policy type (e.g., "softmax") for the RL agent's action selection.
        prev_exp: Previous experience data.

    Returns:
        tuple: Contains the final score, episode history, trajectory score, efficiency score, scenario name,
               interaction history, veto decisions, and reasons for chosen actions.
    """

    # Initialize episode state
    episode, history, reasons, vetos = [], [], [], []
    score, step, done = 0, 0, False
    agent.reset_dictionaries()

    # Environment reset and setup
    ob, valid_acts, hc = env.reset()
    valid_subjects, valid_topics, valid_causes = env.scenario["subjects"], env.scenario["topics"], env.scenario[
        "causes"]
    scenario_name, subject = env.scenario["name"], env.scenario["characters"][0]
    problem = find_phrase(ob[1])[0]
    history.append(ob[1])

    # Initial state and other variables
    state = agent.create_state(update_sentence=ob, hc=hc)
    posttest = False

    while not done:
        transition = [scenario_name, step, ob[1]]

        # RL Agent action selection
        chosen_action, reason = rl_agent_action(agent, state, valid_acts, policy, topk, posttest, chooser, history,
                                                subject, problem, prev_exp)
        reasons.append(reason)

        # Handle veto scenario (chosen_action == 0) with player agent
        if chosen_action == 0 and len(valid_acts) > 1:
            action_str = handle_veto(player, history, subject, problem, valid_subjects, valid_topics, valid_causes, valid_acts,
                                     posttest, prev_exp)
            vetos.append(True)
        else:
            action_str = valid_acts[chosen_action - 1] if chosen_action > 0 else valid_acts[0]
            vetos.append(False)

        # Update history and environment step
        history.append(action_str["sentence"])
        state_update, rew, done, valid_acts, hc, traj_score = env.step(ob, action_str)
        history = update_history(history, state_update)
        posttest = (state_update[0] == "posttest")

        # Log transition and update state
        score += rew
        transition += [action_str, rew, score]
        episode.append(transition)
        state = agent.create_state(update_sentence=state_update, hc=hc, previous_state=state)
        ob = state_update
        step += 1

    # Calculate trajectory and efficiency scores
    traj_score, eff_score = calculate_scores(env)
    agent.reset_dictionaries()

    return score, episode, traj_score, eff_score, scenario_name, history, vetos, reasons


def evaluate_episode_LLM_Interacts(llm, topk, agent, env, policy="softmax", prev_exp=None):
    """
    Evaluates an episode using the h2 method, where actions are chosen with an LLM-assisted selection
    or recommendation, validated by the RL agent.

    Args:
        llm: The LLM agent for choosing or recommending actions.
        topk (int): Number of top choices to consider from the RL agent's action values.
        agent: The RL agent used for decision-making.
        env: The environment instance.
        policy (str): Policy type (e.g., "softmax") for the RL agent's action selection.
        prev_exp: Previous experience data.
    Returns:
        tuple: Contains the final score, episode history, trajectory score, efficiency score, scenario name,
               interaction history, veto decisions, and reasons for chosen actions.
    """

    # Initialize episode state
    episode, history, reasons, vetos = [], [], [], []
    score, step, done = 0, 0, False
    agent.reset_dictionaries()

    # Environment reset and setup
    ob, valid_acts, hc = env.reset()
    valid_subjects, valid_topics, valid_causes = env.scenario["subjects"], env.scenario["topics"], env.scenario[
        "causes"]
    scenario_name, subject = env.scenario["name"], env.scenario["characters"][0]
    problem = find_phrase(ob[1])[0]
    history.append(ob[1])

    # Initial state and other variables
    state = agent.create_state(update_sentence=ob, hc=hc)
    posttest = False

    while not done:
        transition = [scenario_name, step, ob[1]]

        # RL Agent action selection
        chosen_action, action_str, reason, pick_type = rl_llm_interaction(agent, state, valid_acts, policy, topk, llm,
                                                                          posttest, history, subject, problem, prev_exp,
                                                                          valid_subjects, valid_topics, valid_causes)
        reasons.append(reason)
        vetos.append(pick_type == "recommend")

        # Update history and environment step
        history.append(action_str["sentence"])
        state_update, rew, done, valid_acts, hc, traj_score = env.step(ob, action_str)
        history = update_history(history, state_update)
        posttest = (state_update[0] == "posttest")

        # Log transition and update state
        score += rew
        transition += [action_str, rew, score]
        episode.append(transition)
        state = agent.create_state(update_sentence=state_update, hc=hc, previous_state=state)
        ob = state_update
        step += 1

    # Calculate trajectory and efficiency scores
    traj_score, eff_score = calculate_scores(env)
    agent.reset_dictionaries()

    return score, episode, traj_score, eff_score, scenario_name, history, vetos, reasons

def evaluate_episode_RL_Vetos(recer, num_of_recs, agent, env, policy="softmax", prev_exp=None, 
                        threshold=0.5):
    """
    Evaluates an episode using the h3 method, where recommendations from an LLM-assisted system
    guide or override RL agent actions based on a veto mechanism.

    Args:
        recer: The LLM-based recommender system.
        num_of_recs (int): Number of recommendations to retrieve from the LLM.
        agent: The RL agent used for decision-making.
        env: The environment instance.
        policy (str): Policy type (e.g., "softmax") for the RL agent's action selection.
        prev_exp: Previous experience data.
        threshold (float): Veto threshold for RL-based action override.

    Returns:
        tuple: Contains the final score, episode history, trajectory score, efficiency score, scenario name,
               interaction history, veto decisions, and reasons for chosen actions.
    """

    # Initialize episode state
    episode, history, reasons, vetos = [], [], [], []
    score, step, done = 0, 0, False
    agent.reset_dictionaries()

    # Environment reset and setup
    ob, valid_acts, hc = env.reset()
    valid_subjects, valid_topics, valid_causes = env.scenario["subjects"], env.scenario["topics"], env.scenario[
        "causes"]
    scenario_name, subject = env.scenario["name"], env.scenario["characters"][0]
    problem = find_phrase(ob[1])[0]
    history.append(ob[1])

    # Initial state and other variables
    state = agent.create_state(update_sentence=ob, hc=hc)
    posttest = False

    while not done:
        transition = [scenario_name, step, ob[1]]

        # RL Agent action selection with LLM recommendation validation
        action_str, reason, veto = rl_llm_recommendation_interaction(
            recer, agent, state, valid_acts, policy, threshold, history, subject, problem, prev_exp,
            valid_subjects, valid_topics, valid_causes, posttest
        )
        reasons.append(reason)
        vetos.append(veto)

        # Update history and environment step
        history.append(action_str["sentence"])
        state_update, rew, done, valid_acts, hc, traj_score = env.step(ob, action_str)
        history = update_history(history, state_update)
        posttest = (state_update[0] == "posttest")

        # Log transition and update state
        score += rew
        transition += [action_str, rew, score]
        episode.append(transition)
        state = agent.create_state(update_sentence=state_update, hc=hc, previous_state=state)
        ob = state_update
        step += 1

    # Calculate trajectory and efficiency scores
    traj_score, eff_score = calculate_scores(env)
    agent.reset_dictionaries()

    return score, episode, traj_score, eff_score, scenario_name, history, vetos, reasons

def evaluate_episode_RL_Interacts(recer, num_of_recs, agent, env, policy="softmax", prev_exp=None, 
                        threshold=0.5):
    episode = []
    step = 0
    score = 0
    done = False
    agent.reset_dictionaries()
    history = []
    reasons = []
    vetos = []
    ob, valid_acts, hc = env.reset()
    history.append(ob[1])
    valid_subjects = env.scenario["subjects"]
    valid_topics = env.scenario["topics"]
    valid_causes = env.scenario["causes"]
    scenario_name = env.scenario["name"]
    subject = env.scenario["characters"][0]
    problem = find_phrase(ob[1])[0]
    state = agent.create_state(update_sentence=ob, hc=hc)
    posttest = False
    while not done:
        transition = [env.scenario["name"], step, ob[1], ]
        valid_ids = agent.encode_actions(valid_acts)
        _, action_idx, action_values, _ = agent.act(
            [state], [valid_ids], policy=policy, eval_mode=True, action_strs=valid_acts)
        if len(valid_acts) > 1:
            previous_suggestions = []
            is_valid = False
            tries = 0
            while not is_valid:
                tries += 1
                recs_idxs, valid_responses, reason, response, learning_id = get_recs(recer, history, subject, problem,
                                                                                     valid_subjects, valid_topics,
                                                                                     valid_causes,
                                                                                     previous_suggestions,
                                                                                     tries,
                                                                                     valid_acts, posttest=False,
                                                                                     prev_exp=None)
                previous_suggestions.append(response)
                if len(recs_idxs) > 0:

                    veto = RL_veto(recs_idxs, action_values[0].detach().cpu().numpy(), threshold=threshold)
                    vetos.append(veto)

                    act_probs = (softmax(action_values[0][recs_idxs], temperature=0.001 if posttest else 1))
                    chosen_act_idx = torch.multinomial(act_probs, num_samples=1).item()
                    if not veto:
                        action_str = valid_responses[chosen_act_idx]
                        is_valid = True
                    else:
                        is_valid = False
                if not is_valid and tries > 5:
                    action_str = valid_acts[action_idx[0]]
                    is_valid = True
            reasons.append(reason)
        else:
            action_str = valid_acts[0]
            reasons.append("")
        history.append(action_str["sentence"])
        state_update, rew, done, valid_acts, hc, traj_score = env.step(ob, action_str)
        if not done:
            if state_update[0] == "interaction":
                history.append(".".join(state_update[1].split(".")[2:]))
            else:
                history.append(".".join(state_update[1].split(".")[-1:]))
        posttest = state_update[0] == "posttest"
        if not done:
            trace = env.trace
        ob = state_update
        score += rew
        step += 1
        transition += [action_str, rew, score]
        episode.append(transition)
        state = agent.create_state(
            update_sentence=ob, hc=hc, previous_state=state)

    traj_score = sum(
        a in trace for a in env.scenario["present_actions"]) / len(env.scenario["present_actions"])
    eff_score = sum(
        a in trace for a in env.scenario["present_actions"]) / len(trace)
    agent.reset_dictionaries()
    return score, episode, traj_score, eff_score, scenario_name, history, vetos, reasons




def evaluate_episode_clin_LLM_Vetos(mode, chooser, player, topk, agent, env, patient, eval_mode, policy="softmax",
                                prev_exp=None,  inference="h1_v2", threshold=0, split=0,
                                MODEL="gpt-4o-2024-05-13"):
    score = 0
    for ep in range(3):
        if score == 1:
            break
        episode = []
        step = 0
        score = 0
        done = False
        agent.reset_dictionaries()
        history = []
        reasons = []
        vetos = []
        learning_ids = []
        saved_history = []
        history_update = {}
        ob, valid_acts, hc = env.reset()
        history.append(ob[1])
        valid_subjects = env.scenario["subjects"]
        valid_topics = env.scenario["topics"]
        valid_causes = env.scenario["causes"]
        scenario_name = env.scenario["name"]
        subject = env.scenario["characters"][0]
        problem = find_phrase(ob[1])[0]
        state = agent.create_state(update_sentence=ob, hc=hc)
        posttest = False
        TaskDescription = f"Find the cause behind the {subject}'s {problem}"
        task, sub_task = patient, scenario_name
        save_path = f"/data/radmehr/results_{MODEL}/memory/{eval_mode}/{mode}/{split}/{threshold}/{inference}/{task}/{sub_task}"
        if os.path.exists(save_path) and ep == 0:
            shutil.rmtree(save_path)
        summary, episodeIdx = load_summary(save_path, ep, task, sub_task, inference=inference, mode=mode,
                                           eval_mode=eval_mode, split=split, MODEL=MODEL)
        file_name = f"{save_path}/{episodeIdx}.json"
        while not done:
            transition = [env.scenario["name"], step, ob[1], ]
            valid_ids = agent.encode_actions(valid_acts)
            _, action_idx, action_values, _ = agent.act(
                [state], [valid_ids], policy=policy, eval_mode=True, action_strs=valid_acts)
            sorted_idxs = np.argsort(action_values[0].detach().cpu().numpy())
            if not posttest:
                choices_idxs = sorted_idxs[-1:-topk - 1:-1]
            else:
                choices_idxs = sorted_idxs[-1:-3:-1]
            choices = [valid_acts[i]["sentence"] for i in choices_idxs]
            if len(valid_acts) > 1:
                is_valid = False
                tries = 0
                while not is_valid:
                    tries += 1
                    if not is_valid and tries > 3:
                        chosen_action = 1
                        break

                    response = chooser.choose(history, subject, problem, choices, summary=summary, posttest=posttest,
                                              prev_exp=prev_exp)
                    parts = separate(response)
                    if len(parts) > 3:
                        learning_id, reason = parts[:2]
                        chosen_action = "\n".join(parts[2:])
                    elif len(parts) == 3:
                        learning_id, reason, chosen_action = parts
                    elif len(parts) == 2:
                        learning_id = ""
                        reason, chosen_action = parts
                    elif len(parts) == 1:
                        learning_id = ""
                        reason = ""
                        chosen_action = parts[0]
                    else:
                        raise ValueError("Invalid response")
                    learning_id = learning_id.strip("\n").strip()
                    reason, chosen_action = reason.strip("\n").strip(), chosen_action.strip("\n").strip()
                    chosen_action = parse_chosen_action(chosen_action)
                    if chosen_action.isnumeric():
                        chosen_action = int(chosen_action)
                        reason = reason.split(": ")[-1]
                        if len(choices_idxs) >= chosen_action >= 0:
                            chosen_action = chosen_action
                            reason = reason
                            is_valid = True

                reasons.append(reason)
            else:
                chosen_action = 1
            if chosen_action == 0:
                if len(valid_acts) > 1:
                    is_valid = False
                    tries = 0
                    while not is_valid:
                        tries += 1
                        response = player.play(history, subject, problem, valid_subjects, valid_topics, valid_causes,
                                               summary=summary,
                                               posttest=posttest, prev_exp=prev_exp, )
                        parts = separate(response)
                        if len(parts) > 3:
                            learning_id, reason = parts[:2]
                            action = "\n".join(parts[2:])
                        elif len(parts) == 3:
                            learning_id, reason, action = parts
                        elif len(parts) == 2:
                            learning_id = ""
                            reason = parts[0]
                            action = parts[1]
                        elif len(parts) == 1:
                            learning_id = ""
                            reason = ""
                            action = parts[0]
                        else:
                            raise ValueError("Invalid response")
                        reason, action = reason.strip("\n").strip(), action.strip("\n").strip()

                        parsed_responses = []
                        for x in action.split("\n"):
                            parsed_responses.append(
                                parse_string_to_dict((x.split(". ")[-1]), valid_subjects, valid_topics, valid_causes,
                                                     replace_closest=tries > 3, posttest=posttest))
                        valid_responses = []
                        for x in parsed_responses:
                            if x in valid_acts:
                                valid_responses.append(x)

                        if len(valid_responses) > 0:
                            is_valid = True
                            action_str = valid_responses[0]
                        if not is_valid and tries > 3:
                            raise ValueError("out of tries")
                    reasons.append(reason)
                else:
                    action_str = valid_acts[0]
            else:
                action_str = (valid_acts[choices_idxs[chosen_action - 1]])
            vetos.append(chosen_action == 0)
            history_update["observation"] = history[-1]
            history_update["rationale"] = reason
            history_update["action"] = action_str["sentence"]
            learning_ids.append(learning_id)
            reasons.append(reason)
            history.append(action_str["sentence"])
            saved_history.append(copy.deepcopy(history_update))
            state_update, rew, done, valid_acts, hc, traj_score = env.step(ob, action_str)
            if not done:
                if state_update[0] == "interaction":
                    history.append(".".join(state_update[1].split(".")[2:]))
                else:
                    history.append(".".join(state_update[1].split(".")[-1:]))
            posttest = state_update[0] == "posttest"
            if not done:
                trace = env.trace
            ob = state_update
            score += rew
            step += 1
            transition += [action_str, rew, score]
            episode.append(transition)
            state = agent.create_state(
                update_sentence=ob, hc=hc, previous_state=state)
        traj_score = sum(
            a in trace for a in env.scenario["present_actions"]) / len(env.scenario["present_actions"])
        eff_score = sum(
            a in trace for a in env.scenario["present_actions"]) / len(trace)
        agent.reset_dictionaries()
        env.reset()
        data = dict()
        data["taskDescription"] = TaskDescription
        data["episodeIdx"] = episodeIdx
        data["history"] = saved_history
        data["finalScore"] = score
        data["finalTrajScore"] = traj_score
        data["finalEffScore"] = eff_score
        json.dump(data, open(file_name, "w"))
        if score == 1:
            break
        o = summarize_ep(task, sub_task, inference=inference, mode=mode, eval_mode=eval_mode, split=split,
                         threshold=threshold, MODEL=MODEL)
    return score, episode, traj_score, eff_score, scenario_name, history, vetos, reasons


def evaluate_episode_clin_LLM_Interacts(mode, llm, topk, agent, env, patient, eval_mode, policy="softmax", prev_exp=None,
                              inference="h2", threshold=0, split=0, MODEL="gpt-4o-2024-05-13"):
    score = 0
    for ep in range(3):
        if score == 1:
            break
        episode = []
        step = 0
        score = 0
        done = False
        agent.reset_dictionaries()
        history = []
        reasons = []
        vetos = []
        learning_ids = []
        saved_history = []
        history_update = {}
        ob, valid_acts, hc = env.reset()
        history.append(ob[1])
        valid_subjects = env.scenario["subjects"]
        valid_topics = env.scenario["topics"]
        valid_causes = env.scenario["causes"]
        scenario_name = env.scenario["name"]
        subject = env.scenario["characters"][0]
        problem = find_phrase(ob[1])[0]
        state = agent.create_state(update_sentence=ob, hc=hc)
        posttest = False
        TaskDescription = f"Find the cause behind the {subject}'s {problem}"
        task, sub_task = patient, scenario_name
        save_path = f"/data/radmehr/results_{MODEL}/memory/{eval_mode}/{mode}/{split}/{threshold}/{inference}/{task}/{sub_task}"
        if os.path.exists(save_path) and ep == 0:
            shutil.rmtree(save_path)
        summary, episodeIdx = load_summary(save_path, ep, task, sub_task, inference=inference, mode=mode,
                                           eval_mode=eval_mode, split=split, MODEL=MODEL)
        file_name = f"{save_path}/{episodeIdx}.json"
        while not done:
            transition = [env.scenario["name"], step, ob[1], ]
            valid_ids = agent.encode_actions(valid_acts)
            _, action_idx, action_values, _ = agent.act(
                [state], [valid_ids], policy=policy, eval_mode=True, action_strs=valid_acts)
            sorted_idxs = np.argsort(action_values[0].detach().cpu().numpy())
            if not posttest:
                choices_idxs = sorted_idxs[-1:-topk - 1:-1]
            else:
                choices_idxs = sorted_idxs[-1:-3:-1]
            choices = [valid_acts[i]["sentence"] for i in choices_idxs]
            if len(valid_acts) > 1:
                is_valid = False
                tries = 0
                while not is_valid:
                    tries += 1
                    response = llm.cor(history, subject, problem, valid_subjects, valid_topics, valid_causes, choices,
                                       summary=summary,
                                       posttest=posttest, prev_exp=prev_exp)
                    parts = separate(response)
                    if len(parts) > 4:
                        learning_id, reason, pick = parts[:3]
                        ending = "\n".join(parts[3:])
                    elif len(parts) == 4:
                        learning_id, reason, pick, ending = parts
                    elif len(parts) == 3:
                        learning_id = ""
                        reason, pick, ending = parts
                    elif len(parts) == 2:
                        learning_id = ""
                        reason = ""
                        pick, ending = parts
                    elif len(parts) == 1:
                        learning_id = ""
                        reason = ""
                        pick = "recommend"
                        ending = parts[0]
                    else:
                        raise ValueError("Invalid response")
                    learning_id = learning_id.strip("\n").strip()
                    if len(pick) > len(reason):
                        pick, reason = reason, pick
                    pick = pick.strip("\n").strip()
                    if "choose" in pick:
                        reason, chosen_action = reason.strip("\n").strip(), ending.strip("\n").strip()
                        for x in [" ", ".", "\n", ")"]:
                            if chosen_action.split(x)[0].isnumeric():
                                chosen_action = chosen_action.split(x)[0]
                                break
                        if chosen_action.isnumeric():
                            chosen_action = int(chosen_action)
                            reason = reason.split(": ")[-1]
                            is_valid = True
                            action_str = (valid_acts[choices_idxs[chosen_action - 1]])
                        else:
                            if tries > 3:
                                chosen_action = choices_idxs[0]
                                action_str = valid_acts[chosen_action]
                                is_valid = True

                    else:
                        reason, recs = reason.strip("\n").strip(), ending.strip("\n").strip()
                        parsed_responses = []
                        recs_splitted = [reason] + recs.split("\n") + re.split(r' (?=\d)', recs)
                        for x in recs_splitted:
                            parsed_responses.append(
                                parse_string_to_dict((x.split(". ")[-1]), valid_subjects, valid_topics, valid_causes,
                                                     replace_closest=tries > 3, posttest=posttest))
                        recs_idxs = []
                        valid_responses = []
                        for x in parsed_responses:
                            if x in valid_acts:
                                recs_idxs.append(valid_acts.index(x))
                                valid_responses.append(valid_acts[recs_idxs[-1]])

                        if len(recs_idxs) > 0:
                            is_valid = True
                            act_probs = (softmax(action_values[0][recs_idxs], temperature=0.001 if posttest else 1))
                            chosen_act_idx = torch.multinomial(act_probs, num_samples=1).item()
                            action_str = valid_responses[chosen_act_idx]
            else:
                chosen_action = 1
                action_str = (valid_acts[choices_idxs[chosen_action - 1]])
            history_update["observation"] = history[-1]
            history_update["rationale"] = reason
            history_update["action"] = action_str["sentence"]
            learning_ids.append(learning_id)
            history.append(action_str["sentence"])
            saved_history.append(copy.deepcopy(history_update))
            state_update, rew, done, valid_acts, hc, traj_score = env.step(ob, action_str)
            if not done:
                if state_update[0] == "interaction":
                    history.append(".".join(state_update[1].split(".")[2:]))
                else:
                    history.append(".".join(state_update[1].split(".")[-1:]))
            posttest = state_update[0] == "posttest"
            if not done:
                trace = env.trace
            ob = state_update
            score += rew
            step += 1
            transition += [action_str, rew, score]
            episode.append(transition)
            vetos.append(True if pick == "recommend" else False)
            reasons.append(reason)
            state = agent.create_state(
                update_sentence=ob, hc=hc, previous_state=state)
        traj_score = sum(
            a in trace for a in env.scenario["present_actions"]) / len(env.scenario["present_actions"])
        eff_score = sum(
            a in trace for a in env.scenario["present_actions"]) / len(trace)
        agent.reset_dictionaries()
        env.reset()
        data = dict()
        data["taskDescription"] = TaskDescription
        data["episodeIdx"] = episodeIdx
        data["history"] = saved_history
        data["finalScore"] = score
        data["finalTrajScore"] = traj_score
        data["finalEffScore"] = eff_score
        json.dump(data, open(file_name, "w"))
        if score == 1:
            break
        o = summarize_ep(task, sub_task, inference=inference, mode=mode, eval_mode=eval_mode, split=split,
                         threshold=threshold, MODEL=MODEL)
    return score, episode, traj_score, eff_score, scenario_name, history, vetos, reasons


def evaluate_episode_clin_RL_Vetos(mode, recer, num_of_recs, agent, env, patient, eval_mode, policy="softmax", prev_exp=None,
                              inference="h3", threshold=0, split=0, MODEL="gpt-4o-2024-05-13"):
    score = 0
    for ep in range(3):
        if score == 1:
            break
        episode = []
        step = 0
        score = 0
        done = False
        agent.reset_dictionaries()
        history = []
        reasons = []
        vetos = []
        learning_ids = []
        saved_history = []
        history_update = {}
        ob, valid_acts, hc = env.reset()
        history.append(ob[1])
        valid_subjects = env.scenario["subjects"]
        valid_topics = env.scenario["topics"]
        valid_causes = env.scenario["causes"]
        scenario_name = env.scenario["name"]
        subject = env.scenario["characters"][0]
        problem = find_phrase(ob[1])[0]
        state = agent.create_state(update_sentence=ob, hc=hc)
        posttest = False
        TaskDescription = f"Find the cause behind the {subject}'s {problem}"
        task, sub_task = patient, scenario_name
        save_path = f"/data/radmehr/results_{MODEL}/memory/{eval_mode}/{mode}/{split}/{threshold}/{inference}/{task}/{sub_task}"
        if os.path.exists(save_path) and ep == 0:
            shutil.rmtree(save_path)
        summary, episodeIdx = load_summary(save_path, ep, task, sub_task, inference=inference, mode=mode,
                                           eval_mode=eval_mode, split=split, MODEL=MODEL)
        file_name = f"{save_path}/{episodeIdx}.json"
        while not done:
            transition = [env.scenario["name"], step, ob[1], ]
            valid_ids = agent.encode_actions(valid_acts)
            _, action_idx, action_values, _ = agent.act(
                [state], [valid_ids], policy=policy, eval_mode=True, action_strs=valid_acts)
            if len(valid_acts) > 1:
                is_valid = False
                tries = 0
                while not is_valid:
                    tries += 1
                    response = recer.rec(history, subject, problem, valid_subjects, valid_topics, valid_causes,
                                         summary=summary,
                                         posttest=posttest, prev_exp=prev_exp, )

                    parts = separate(response)
                    if len(parts) > 3:
                        learning_id, reason = parts[:2]
                        recs = "\n".join(parts[2:])
                    elif len(parts) == 3:
                        learning_id, reason, recs = parts
                    elif len(parts) == 2:
                        learning_id = ""
                        reason, recs = parts
                    elif len(parts) == 1:
                        learning_id = ""
                        reason = ""
                        recs = parts[0]
                    else:
                        raise ValueError("Invalid response")
                    learning_id = learning_id.strip("\n").strip()
                    reason, recs = reason.strip("\n").strip(), recs.strip("\n").strip()
                    parsed_responses = []
                    recs_splitted = [reason] + recs.split("\n") + re.split(r' (?=\d)', recs)
                    for x in recs_splitted:
                        parsed_responses.append(
                            parse_string_to_dict((x.split(". ")[-1]), valid_subjects, valid_topics, valid_causes,
                                                 replace_closest=tries > 3, posttest=posttest))
                    recs_idxs = []
                    valid_responses = []
                    for x in parsed_responses:
                        if x in valid_acts:
                            recs_idxs.append(valid_acts.index(x))
                            valid_responses.append(valid_acts[recs_idxs[-1]])

                    if len(recs_idxs) > 0:
                        if posttest:
                            recs_idxs = [recs_idxs[0]]
                        veto = RL_veto(recs_idxs, action_values[0].detach().cpu().numpy(), threshold=threshold)
                        vetos.append(veto)
                        is_valid = True
                        act_probs = (softmax(action_values[0][recs_idxs], temperature=0.001 if posttest else 1))
                        chosen_act_idx = torch.multinomial(act_probs, num_samples=1).item()
                        if not veto:
                            action_str = valid_responses[chosen_act_idx]
                        else:
                            action_str = valid_acts[action_idx[0]]
                    if not is_valid and tries > 3:
                        raise ValueError("OUT OF TRIES")
                reasons.append(reason)
            else:
                action_str = valid_acts[0]
                reasons.append("")
            history_update["observation"] = history[-1]
            history_update["rationale"] = reason
            history_update["action"] = action_str["sentence"]
            learning_ids.append(learning_id)
            history.append(action_str["sentence"])
            saved_history.append(copy.deepcopy(history_update))
            state_update, rew, done, valid_acts, hc, traj_score = env.step(ob, action_str)
            if not done:
                if state_update[0] == "interaction":
                    history.append(".".join(state_update[1].split(".")[2:]))
                else:
                    history.append(".".join(state_update[1].split(".")[-1:]))
            posttest = state_update[0] == "posttest"
            if not done:
                trace = env.trace
            ob = state_update
            score += rew
            step += 1
            transition += [action_str, rew, score]
            episode.append(transition)
            state = agent.create_state(
                update_sentence=ob, hc=hc, previous_state=state)
        traj_score = sum(
            a in trace for a in env.scenario["present_actions"]) / len(env.scenario["present_actions"])
        eff_score = sum(
            a in trace for a in env.scenario["present_actions"]) / len(trace)
        agent.reset_dictionaries()
        env.reset()
        data = dict()
        data["taskDescription"] = TaskDescription
        data["episodeIdx"] = episodeIdx
        data["history"] = saved_history
        data["finalScore"] = score
        data["finalTrajScore"] = traj_score
        data["finalEffScore"] = eff_score
        json.dump(data, open(file_name, "w"))
        if score == 1:
            break
        o = summarize_ep(task, sub_task, inference=inference, mode=mode, eval_mode=eval_mode, split=split,
                         threshold=threshold, MODEL=MODEL)
    return score, episode, traj_score, eff_score, scenario_name, history, vetos, reasons



def evaluate_episode_clin_RL_Interacts(mode, recer, num_of_recs, agent, env, patient, eval_mode, policy="softmax", prev_exp=None,
                              inference="h5", threshold=0, split=0, MODEL="gpt-4o-2024-05-13"):
    score = 0
    for ep in range(3):
        if score == 1:
            break
        episode = []
        step = 0
        score = 0
        done = False
        agent.reset_dictionaries()
        history = []
        reasons = []
        vetos = []
        learning_ids = []
        saved_history = []
        history_update = {}
        ob, valid_acts, hc = env.reset()
        history.append(ob[1])
        valid_subjects = env.scenario["subjects"]
        valid_topics = env.scenario["topics"]
        valid_causes = env.scenario["causes"]
        scenario_name = env.scenario["name"]
        subject = env.scenario["characters"][0]
        problem = find_phrase(ob[1])[0]
        state = agent.create_state(update_sentence=ob, hc=hc)
        posttest = False
        TaskDescription = f"Find the cause behind the {subject}'s {problem}"
        task, sub_task = patient, scenario_name
        save_path = f"/data/radmehr/results_{MODEL}/memory/{eval_mode}/{mode}/{split}/{threshold}/{inference}/{task}/{sub_task}"
        if os.path.exists(save_path) and ep == 0:
            shutil.rmtree(save_path)
        summary, episodeIdx = load_summary(save_path, ep, task, sub_task, inference=inference, mode=mode,
                                           eval_mode=eval_mode, split=split, MODEL=MODEL)
        file_name = f"{save_path}/{episodeIdx}.json"
        while not done:
            transition = [env.scenario["name"], step, ob[1], ]
            valid_ids = agent.encode_actions(valid_acts)
            _, action_idx, action_values, _ = agent.act(
                [state], [valid_ids], policy=policy, eval_mode=True, action_strs=valid_acts)
            if len(valid_acts) > 1:
                previous_suggestions = []
                is_valid = False
                tries = 0
                while not is_valid:
                    tries += 1
                    recs_idxs, valid_responses, reason, response, learning_id = get_recs(recer, history, subject,
                                                                                         problem,
                                                                                         valid_subjects, valid_topics,
                                                                                         valid_causes,
                                                                                         previous_suggestions=previous_suggestions,
                                                                                         tries=tries,
                                                                                         valid_acts=valid_acts,
                                                                                         summary=summary,
                                                                                         posttest=False, prev_exp=None)
                    previous_suggestions.append(response)
                    if len(recs_idxs) > 0:
                        if posttest:
                            recs_idxs = [recs_idxs[0]]
                        veto = RL_veto(recs_idxs, action_values[0].detach().cpu().numpy(), threshold=threshold)
                        vetos.append(veto)

                        act_probs = (softmax(action_values[0][recs_idxs], temperature=0.001 if posttest else 1))
                        chosen_act_idx = torch.multinomial(act_probs, num_samples=1).item()
                        if not veto:
                            action_str = valid_responses[chosen_act_idx]
                            is_valid = True
                        else:
                            is_valid = False
                    if not is_valid and tries > 5:
                        action_str = valid_acts[action_idx[0]]
                        is_valid = True
                reasons.append(reason)
            else:
                action_str = valid_acts[0]
                reasons.append("")
            history_update["observation"] = history[-1]
            history_update["rationale"] = reason
            history_update["action"] = action_str["sentence"]
            learning_ids.append(learning_id)
            history.append(action_str["sentence"])
            saved_history.append(copy.deepcopy(history_update))

            state_update, rew, done, valid_acts, hc, traj_score = env.step(ob, action_str)
            if not done:
                if state_update[0] == "interaction":
                    history.append(".".join(state_update[1].split(".")[2:]))
                else:
                    history.append(".".join(state_update[1].split(".")[-1:]))
            posttest = state_update[0] == "posttest"
            if not done:
                trace = env.trace
            ob = state_update
            score += rew
            step += 1
            transition += [action_str, rew, score]
            episode.append(transition)
            state = agent.create_state(
                update_sentence=ob, hc=hc, previous_state=state)
        traj_score = sum(
            a in trace for a in env.scenario["present_actions"]) / len(env.scenario["present_actions"])
        eff_score = sum(
            a in trace for a in env.scenario["present_actions"]) / len(trace)
        agent.reset_dictionaries()
        env.reset()
        data = dict()
        data["taskDescription"] = TaskDescription
        data["episodeIdx"] = episodeIdx
        data["history"] = saved_history
        data["finalScore"] = score
        data["finalTrajScore"] = traj_score
        data["finalEffScore"] = eff_score
        json.dump(data, open(file_name, "w"))
        if score == 1:
            break
        o = summarize_ep(task, sub_task, inference=inference, mode=mode, eval_mode=eval_mode, split=split,
                         threshold=threshold, MODEL=MODEL)
    return score, episode, traj_score, eff_score, scenario_name, history, vetos, reasons
