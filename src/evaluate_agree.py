import os
import shutil
import json
import copy
import re
import numpy as np
import torch
from src.evaluate_helper import separate, parse_string_to_dict, softmax, find_phrase, \
    parse_chosen_action, load_summary, update_history, process_recer_response, calculate_scores, \
    process_clin_recer_response, select_top_actions, handle_chooser_response
from clin_memory import summarize_ep


def evaluate_episode_rec(recer, agent, env, policy="softmax", prev_exp=None):
    """
    Evaluates an episode using a recommendation agent (recer) with an RL agent in a given environment.

    Args:
        recer: The recommendation agent.
        agent: The RL agent for action selection.
        env: The environment to interact with.
        policy (str): Policy for the RL agent's action selection.
        prev_exp: Optional previous experience data.

    Returns:
        tuple: Contains score, episode data, trajectory score, efficiency score, scenario name, and history.
    """
    responses = []
    episode = []
    step = 0
    score = 0
    done = False
    agent.reset_dictionaries()
    history = []
    reasons = []
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
        transition = [env.scenario["name"], step, ob[1]]
        valid_ids = agent.encode_actions(valid_acts)
        _, action_idx, action_values, _ = agent.act(
            [state], [valid_ids], policy=policy, eval_mode=True, action_strs=valid_acts
        )
        if len(valid_acts) > 1:
            is_valid, tries = False, 0
            while not is_valid:
                tries += 1
                response = recer.rec(history, subject, problem, valid_subjects, valid_topics, valid_causes,
                                     posttest=posttest, prev_exp=prev_exp)
                reason, recs, is_valid, action_str = process_recer_response(
                    response, valid_acts, valid_subjects, valid_topics, valid_causes, posttest, tries, action_values
                )
                if not is_valid and tries > 3:
                    raise ValueError("OUT OF TRIES")
            reasons.append(reason)
        else:
            action_str = valid_acts[0]
            reasons.append("")

        history.append(action_str["sentence"])
        state_update, rew, done, valid_acts, hc, traj_score = env.step(ob, action_str)
        update_history(history, state_update)
        posttest = state_update[0] == "posttest"
        if not done:
            trace = env.trace
        ob = state_update
        score += rew
        step += 1
        transition += [action_str, rew, score]
        episode.append(transition)
        state = agent.create_state(update_sentence=ob, hc=hc, previous_state=state)

    traj_score, eff_score = calculate_scores(env)
    agent.reset_dictionaries()
    return score, episode, traj_score, eff_score, scenario_name, history


def evaluate_episode_clin_rec(mode, recer, agent, env, patient, eval_mode, policy="softmax", prev_exp=None,
                              split=0, MODEL="gpt-4o-2024-05-13"):
    """
    Evaluates a clinical scenario using a recommendation agent (recer) and RL agent in a given environment.

    Args:
        mode: Evaluation mode.
        recer: The recommendation agent.
        agent: The RL agent for action selection.
        env: The environment to interact with.
        patient: Patient data or identifier.
        eval_mode: Evaluation mode.
        policy (str): Policy for the RL agent's action selection.
        prev_exp: Optional previous experience data.
        split: Data split identifier.
        MODEL (str): Model identifier for evaluation.

    Returns:
        tuple: Contains score, episode data, trajectory score, efficiency score, scenario name, and history.
    """
    responses = []
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
        save_path = f"/data/radmehr/results_{MODEL}/memory/{eval_mode}/{mode}/{split}/0/rec/{task}/{sub_task}"
        if os.path.exists(save_path) and ep == 0:
            shutil.rmtree(save_path)

        summary, episodeIdx = load_summary(save_path, ep, task, sub_task, inference="rec", mode=mode,
                                           eval_mode=eval_mode, split=split, MODEL=MODEL)
        file_name = f"{save_path}/{episodeIdx}.json"

        while not done:
            transition = [env.scenario["name"], step, ob[1]]
            valid_ids = agent.encode_actions(valid_acts)
            _, action_idx, action_values, _ = agent.act(
                [state], [valid_ids], policy=policy, eval_mode=True, action_strs=valid_acts
            )
            if len(valid_acts) > 1:
                is_valid, tries = False, 0
                while not is_valid:
                    tries += 1
                    response = recer.rec(history, subject, problem, valid_subjects, valid_topics, valid_causes, summary,
                                         posttest=posttest, prev_exp=prev_exp)
                    responses.append(response)

                    # Use helper function to process recer response
                    learning_id, reason, recs, is_valid, action_str = process_clin_recer_response(
                        response, valid_acts, valid_subjects, valid_topics, valid_causes, posttest, tries, action_values
                    )

                    if not is_valid and tries > 3:
                        raise ValueError("OUT OF TRIES")
            else:
                action_str = valid_acts[0]
                learning_id = ""
                reason = ""

            # Update and save history
            history_update["observation"] = history[-1]
            history_update["rationale"] = reason
            history_update["action"] = action_str["sentence"]
            learning_ids.append(learning_id)
            reasons.append(reason)
            history.append(action_str["sentence"])
            saved_history.append(copy.deepcopy(history_update))

            # Take a step in the environment
            state_update, rew, done, valid_acts, hc, traj_score = env.step(ob, action_str)
            update_history(history, state_update)
            posttest = state_update[0] == "posttest"
            if not done:
                trace = env.trace
            ob = state_update
            score += rew
            step += 1
            transition += [action_str, rew, score]
            episode.append(transition)
            state = agent.create_state(update_sentence=ob, hc=hc, previous_state=state)

        # Calculate scores and reset environment
        traj_score, eff_score = calculate_scores(env)
        agent.reset_dictionaries()
        env.reset()

        # Save evaluation results
        data = {
            "taskDescription": TaskDescription,
            "episodeIdx": episodeIdx,
            "history": saved_history,
            "finalScore": score,
            "finalTrajScore": traj_score,
            "finalEffScore": eff_score
        }
        json.dump(data, open(file_name, "w"))
        if score == 1:
            break

        summarize_ep(task, sub_task, inference="rec", mode=mode, eval_mode=eval_mode, split=split, MODEL=MODEL)
    return score, episode, traj_score, eff_score, scenario_name, history


def evaluate_episode_clin_choose(mode, chooser, topk, agent, env, patient, eval_mode, policy="softmax", prev_exp=None,
                                 split=0, MODEL="gpt-4o-2024-05-13"):
    """
    Evaluates a clinical scenario using a chooser agent in a given environment.

    Args:
        mode: Evaluation mode.
        chooser: The chooser agent for action selection.
        topk: Number of top actions to consider.
        agent: The RL agent for state management.
        env: The environment to interact with.
        patient: Patient data or identifier.
        eval_mode: Evaluation mode.
        policy (str): Policy for the RL agent's action selection.
        prev_exp: Optional previous experience data.
        split: Data split identifier.
        MODEL (str): Model identifier for evaluation.

    Returns:
        tuple: Contains score, episode data, trajectory score, efficiency score, scenario name, and history.
    """
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
        learning_ids = []
        saved_history = []
        history_update = {}
        ob, valid_acts, hc = env.reset()
        history.append(ob[1])
        scenario_name = env.scenario["name"]
        subject = env.scenario["characters"][0]
        problem = find_phrase(ob[1])[0]
        state = agent.create_state(update_sentence=ob, hc=hc)
        posttest = False
        TaskDescription = f"Find the cause behind the {subject}'s {problem}"
        task, sub_task = patient, scenario_name
        save_path = f"/data/radmehr/results_{MODEL}/memory/{eval_mode}/{mode}/{split}/0/choose/{task}/{sub_task}"
        if os.path.exists(save_path) and ep == 0:
            shutil.rmtree(save_path)

        summary, episodeIdx = load_summary(save_path, ep, task, sub_task, inference="choose", mode=mode,
                                           eval_mode=eval_mode, split=split, MODEL=MODEL)
        file_name = f"{save_path}/{episodeIdx}.json"

        while not done:
            transition = [env.scenario["name"], step, ob[1]]
            valid_ids = agent.encode_actions(valid_acts)
            _, action_idx, action_values, _ = agent.act(
                [state], [valid_ids], policy=policy, eval_mode=True, action_strs=valid_acts)
            choices_idxs = select_top_actions(action_values, topk, posttest)
            choices = [valid_acts[i]["sentence"] for i in choices_idxs]

            if len(valid_acts) > 1:
                chosen_action, reason, learning_id, is_valid = handle_chooser_response(
                    chooser, history, subject, problem, choices, summary, posttest, prev_exp, choices_idxs
                )
                if not is_valid:
                    chosen_action = 1
            else:
                chosen_action = 1
                reason = ""
                learning_id = ""

            # Update and save history
            action_str = valid_acts[choices_idxs[chosen_action - 1]]
            history_update["observation"] = history[-1]
            history_update["rationale"] = reason
            history_update["action"] = action_str["sentence"]
            learning_ids.append(learning_id)
            reasons.append(reason)
            history.append(action_str["sentence"])
            saved_history.append(copy.deepcopy(history_update))

            # Take a step in the environment
            state_update, rew, done, valid_acts, hc, traj_score = env.step(ob, action_str)
            update_history(history, state_update)
            posttest = state_update[0] == "posttest"
            if not done:
                trace = env.trace
            ob = state_update
            score += rew
            step += 1
            transition += [action_str, rew, score]
            episode.append(transition)
            state = agent.create_state(update_sentence=ob, hc=hc, previous_state=state)

        traj_score, eff_score = calculate_scores(env)
        agent.reset_dictionaries()
        env.reset()

        # Save evaluation results
        data = {
            "taskDescription": TaskDescription,
            "episodeIdx": episodeIdx,
            "history": saved_history,
            "finalScore": score,
            "finalTrajScore": traj_score,
            "finalEffScore": eff_score
        }
        json.dump(data, open(file_name, "w"))
        if score == 1:
            break

        summarize_ep(task, sub_task, inference="choose", mode=mode, eval_mode=eval_mode, split=split, MODEL=MODEL)
    return score, episode, traj_score, eff_score, scenario_name, history


def evaluate_episode_choose(chooser, topk, agent, env, policy="softmax", prev_exp=None):
    """
    Evaluates an episode using a chooser agent in a given environment.

    Args:
        chooser: The chooser agent for action selection.
        topk: Number of top actions to consider.
        agent: The RL agent for state management.
        env: The environment to interact with.
        policy (str): Policy for the RL agent's action selection.
        prev_exp: Optional previous experience data.

    Returns:
        tuple: Contains score, episode data, trajectory score, efficiency score, scenario name, and history.
    """
    episode = []
    step = 0
    score = 0
    done = False
    agent.reset_dictionaries()
    history = []
    reasons = []
    ob, valid_acts, hc = env.reset()

    history.append(ob[1])
    scenario_name = env.scenario["name"]
    subject = env.scenario["characters"][0]
    problem = find_phrase(ob[1])[0]
    state = agent.create_state(update_sentence=ob, hc=hc)
    posttest = False

    while not done:
        transition = [env.scenario["name"], step, ob[1]]
        valid_ids = agent.encode_actions(valid_acts)
        _, action_idx, action_values, _ = agent.act(
            [state], [valid_ids], policy=policy, eval_mode=True, action_strs=valid_acts)
        choices_idxs = select_top_actions(action_values, topk, posttest, valid_acts)
        choices = [valid_acts[i]["sentence"] for i in choices_idxs]

        if len(valid_acts) > 1:
            chosen_action, reason, _, is_valid = handle_chooser_response(
                chooser, history, subject, problem, choices, None, posttest, prev_exp, choices_idxs
            )
            if not is_valid:
                chosen_action = 1
        else:
            chosen_action = 1
            reason = ""

        # Update history and environment
        action_str = valid_acts[choices_idxs[chosen_action - 1]]
        history.append(action_str["sentence"])
        reasons.append(reason)

        state_update, rew, done, valid_acts, hc, traj_score = env.step(ob, action_str)
        update_history(history, state_update)
        posttest = state_update[0] == "posttest"
        if not done:
            trace = env.trace
        ob = state_update
        score += rew
        step += 1
        transition += [action_str, rew, score]
        episode.append(transition)
        state = agent.create_state(update_sentence=ob, hc=hc, previous_state=state)

    traj_score, eff_score = calculate_scores(env)
    agent.reset_dictionaries()
    return score, episode, traj_score, eff_score, scenario_name, history
