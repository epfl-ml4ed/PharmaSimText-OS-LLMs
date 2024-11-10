import os
import shutil
import json
import copy
from src.evaluate_helper import find_phrase, load_summary, calculate_scores, update_history, process_response, process_clin_response
from clin_memory import summarize_ep
def evaluate_episode_play(player, env, summarizer=None, prev_exp=None):
    """
    Evaluates an episode where a player interacts with an environment.

    Args:
        player: The player agent that interacts with the environment.
        env: The environment in which the episode is conducted.
        summarizer: Optional summarizer for additional context or processing.
        prev_exp: Optional previous experience data for use by the player.

    Returns:
        tuple: Score, episode data, trajectory score, efficiency score, scenario name, and history.
    """
    episode = []
    step = 0
    score = 0
    done = False
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
    posttest = False

    while not done:
        transition = [env.scenario["name"], step, ob[1]]
        if len(valid_acts) > 1:
            is_valid, tries = False, 0
            while not is_valid:
                tries += 1
                response = player.play(history, subject, problem, valid_subjects, valid_topics, valid_causes,
                                       posttest=posttest, prev_exp=prev_exp, summarizer=summarizer)
                reason, action, is_valid, action_str = process_response(response, valid_acts, valid_subjects,
                                                                        valid_topics, valid_causes, posttest, tries)
                if not is_valid and tries > 3:
                    raise ValueError("OUT OF TRIES")
            reasons.append(reason)
            history.append(action_str["sentence"])
        else:
            action_str = valid_acts[0]
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

    traj_score, eff_score = calculate_scores(env)
    return score, episode, traj_score, eff_score, scenario_name, history

def evaluate_episode_clin_play(mode, clin_player, env, patient, eval_mode, prev_exp=None, summarizer=None, split=0,
                               MODEL="gpt-4o-2024-05-13"):
    """
    Evaluates a clinical scenario using a clinician player.

    Args:
        mode: Mode of evaluation.
        clin_player: Clinician player agent.
        env: Environment in which the episode is conducted.
        patient: Patient data or identifier.
        eval_mode: Evaluation mode.
        prev_exp: Optional previous experience data.
        summarizer: Optional summarizer for additional context or processing.
        split: Data split identifier.
        MODEL: Model identifier used for evaluation.

    Returns:
        tuple: Score, episode data, trajectory score, efficiency score, scenario name, and history.
    """
    score = 0
    for ep in range(3):
        if score == 1:
            break
        episode = []
        step = 0
        score = 0
        done = False
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
        posttest = False
        TaskDescription = f"Find the cause behind the {subject}'s {problem}"
        task, sub_task = patient, scenario_name
        save_path = f"/data/radmehr/results_{MODEL}/memory/{eval_mode}/{mode}/{split}/0/play/{task}/{sub_task}"
        if os.path.exists(save_path) and ep == 0:
            shutil.rmtree(save_path)
        summary, episodeIdx = load_summary(save_path, ep, task, sub_task, inference="play", mode=mode,
                                           eval_mode=eval_mode, split=split, MODEL=MODEL)

        file_name = f"{save_path}/{episodeIdx}.json"
        while not done:
            transition = [env.scenario["name"], step, ob[1]]
            if len(valid_acts) > 1:
                is_valid, tries = False, 0
                while not is_valid:
                    tries += 1
                    response = clin_player.play(history, subject, problem, valid_subjects, valid_topics, valid_causes,
                                                summary, posttest=posttest, prev_exp=prev_exp, summarizer=summarizer)
                    learning_id, reason, action, is_valid, action_str = process_clin_response(
                        response, valid_acts, valid_subjects, valid_topics, valid_causes, posttest, tries)

                    if not is_valid and tries > 3:
                        raise ValueError("OUT OF TRIES")
                history_update["observation"] = history[-1]
                history_update["rationale"] = reason
                history_update["action"] = action_str["sentence"]
                learning_ids.append(learning_id)
                reasons.append(reason)
                history.append(action_str["sentence"])
                saved_history.append(copy.deepcopy(history_update))
            else:
                action_str = valid_acts[0]
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

        traj_score, eff_score = calculate_scores(env)
        env.reset()
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

        summarize_ep(task, sub_task, inference="play", mode=mode, eval_mode=eval_mode, split=split, MODEL=MODEL)
    return score, episode, traj_score, eff_score, scenario_name, history
