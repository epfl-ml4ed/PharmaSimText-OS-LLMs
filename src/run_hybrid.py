config = {
    "window": 10,
    "embedding_dim": 300,
    "n_layers_action": 1,
    "n_layers_state": 1,
    "n_layers_scorer": 1,
    "n_layers_lstm": 1,
    "hidden_dim_action": 64,
    "hidden_dim_state": 512,
    "hidden_dim_scorer": 512,
    "hidden_lstm": 128,
    "activation": "relu",
    "emb": "sum",
    "hc": None,
    "unq": False,
    "learning_rate": 0.0001,
    "env_step_limit": 20,
    "seed": 256,
    "max_steps": 100000,
    "update_freq": 1,
    "log_freq": 500,
    "eval_freq": 1000,
    "memory_size": 500000,
    "encoder_memory_size": 10,
    "save_memory": 0.5,
    "memory_path": "./encoder_memory/",
    "batch_size": 256,
    "gamma": 0.9,
    "clip": 100,
    "game_path": "./scenarios",
    "wrong_answer": True,
    "soft_reward": False,
    "reward_scale": 1,
    "wording": True,
    "evaluation": "cause",
    "document": False,
    "reduced": False,
    "encoder_type": "fasttext",
    "train_ratio": 0.8,
    "test_ratio": 0.1,
    "test_mode": "wording",
    "save_path": "./models/",
    "train_type": "normal",
    "TAU": 0.5,
    "pretrain": False,
    "llm_assisted": False,
    "use_attention": False,
    "pretrained_explore": 0.3,
    "reduce_scenarios": False,
    "patient": "baby",
    "penalty": -0.01,
}
import json
import os
import torch
import game
import random


def seprate(response):
    # print(response)
    variations = {"$$$": ["\n$$$\n", "$$$\n", "\n$$$", " $$$ ", " $$$", "$$$ ", "$$$"],
                  "###": ["\n###\n", "###\n", "\n###", " ### ", " ###", "### ", "###"],
                  "***": ["\n***\n", "***\n", "\n***", " *** ", " ***", "*** ", "***"]}
    existing_seperators = []
    for k in variations.keys():
        for sep in variations[k]:
            if response.find(sep) > -1:
                existing_seperators.append(k)
                break
    # sort the seperators by the first occurence
    existing_seperators.sort(key=lambda x: response.find(x))
    parts = []
    for sep in existing_seperators:
        spilited = response.split(sep)
        parts.append(spilited[0])
        response = sep.join(spilited[1:])
    parts.append(response)
    # print(parts)
    return parts


random.seed(config["seed"])
import shutil
import fasttext.util

if not os.path.exists("./lms/cc.en.300.bin"):
    fasttext.util.download_model('en', if_exists='ignore')
import scipy
import numpy as np
from drrn import DRRNAgent
import llm_helper
import re
from test import summarize_ep
import pathlib
import copy

fasttext_model = fasttext.load_model(
    "./lms/cc.en.300.bin"
)


def parse_chosen_action(chosen_action):
    for x in [" ", ".", "\n", ")"]:
        if chosen_action.split(x)[0].isnumeric():
            chosen_action = chosen_action.split(x)[0]
            break
    return chosen_action


def find_all_occurences(list, value):
    return [i for i, x in enumerate(list) if x == value]


def match(sentence, valid_sentences, replace_closest=False):
    for t in valid_sentences:
        if sentence == t.lower():
            return t, True
    indicator = [(sentence in t.lower()) or (t.lower() in sentence) for t in valid_sentences]
    if any(indicator):
        idx = find_all_occurences(indicator, True)
        values = [valid_sentences[i] for i in idx]
        len_values = [len(x) for x in values]
        if len(idx) > 1:
            idx = idx[len_values.index(max(len_values))]
        if isinstance(idx, list):
            idx = idx[0]
        return valid_sentences[idx], True
    else:
        if replace_closest:
            sentence = sentence.replace("\n", "")
            ### replace the closest sentence
            valid_sentences_embeddings = [fasttext_model.get_sentence_vector(x) for x in valid_sentences]
            sentence_embedding = fasttext_model.get_sentence_vector(sentence)
            distances = [1 - scipy.spatial.distance.cosine(x, sentence_embedding) for x in valid_sentences_embeddings]
            idx = distances.index(max(distances))
            return valid_sentences[idx], False
        else:
            return sentence, False


def parse_string_to_dict(input_str, valid_subjects, valid_topics, valid_causes, replace_closest=False, posttest=False):
    # Splitting the input string into the command and the arguments
    input_str = input_str.lower()
    result_dict = {
        "type": "",
        "part": "",
        "detail": "",
        "sentence": ""
    }
    flag = False
    if "(" in input_str:
        parts = input_str.split('(', 1)
        command = parts[0].split(' ')[-1]
        if posttest and replace_closest:
            command = "choose"
        args = parts[1].rsplit(')', 1)[0] if len(parts) > 1 else ""
        args = args.split("),")[0]
        flag = True
        # Mapping based on the command
        if command == "ask":
            result_dict["type"] = "interaction"
            result_dict["part"] = "discuss"
            subject, topic = "", ""
            if args:
                if len(args.split(',')) == 1:
                    subject = args
                    topic = args
                elif len(args.split(',')) == 2:
                    subject, topic = args.split(',')
                else:
                    subject, topic = args.split(',')[0], ",".join(args.split(',')[1:])
            subject, topic = subject.strip(), topic.strip()
            subject, matched = match(subject, valid_subjects, replace_closest=replace_closest)
            topic, matched = match(topic, valid_topics, replace_closest=replace_closest)
            if replace_closest:
                print(f"Replaced {args} with {subject}, {topic}")
            result_dict["detail"] = ",".join([subject, topic])
            result_dict["sentence"] = f"i want to know about the {subject} 's {topic}."
        elif command == "answer":
            result_dict["type"] = "interaction"
            result_dict["part"] = "solution"
            result_dict["sentence"] = "i want to suggest a solution."

        elif command == "choose":
            result_dict["type"] = "posttest"
            args, matched = match(args, valid_causes, replace_closest=replace_closest)
            result_dict["sentence"] = args
        else:
            flag = False

    if not flag:
        if posttest:
            cause, matched_c = match(input_str, valid_causes, replace_closest=replace_closest)
            result_dict["type"] = "posttest"
            result_dict["sentence"] = cause
        else:
            result_dict["type"] = "interaction"
            subject, matched_s = match(input_str, valid_subjects, replace_closest=replace_closest)
            topic, matched_t = match(input_str, valid_topics, replace_closest=replace_closest)
            if "diagnosis" in input_str:
                result_dict["part"] = "solution"
                result_dict["sentence"] = "i want to suggest a solution."
            else:
                result_dict["part"] = "discuss"
                result_dict["detail"] = ",".join([subject, topic])
                result_dict["sentence"] = f"i want to know about the {subject} 's {topic}."
    return result_dict


def find_phrase(text):
    # Pattern to find phrases between "have"/"has" and a dot
    pattern = r'\b(have|has)\b(.*?)(?=\.)'

    # Find all matches in the text
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)

    # Extracting the phrases
    phrases = [''.join(match[1]).strip() for match in matches]
    return phrases


def evaluate_patient_eval_mode(patient, eval_mode, inference, mode_, split, threshold, MODEL="gpt-4o-2024-05-13"):
    threshold_str = str(round(threshold, 2))
    #################### LOAD ENVIRONMENT ####################
    if not os.path.exists(os.path.join(f'/data/radmehr/results_{MODEL}/', eval_mode, patient)):
        os.makedirs(os.path.join(f'/data/radmehr/results_{MODEL}/', eval_mode, patient))
    if os.path.exists(
            os.path.join(f'/data/radmehr/results_{MODEL}/', eval_mode, patient,
                         f"{patient}_{eval_mode}_{inference}_{split}_{threshold_str}.json")):
        results = json.load(
            open(os.path.join(f'/data/radmehr/results_{MODEL}/', eval_mode, patient,
                              f"{patient}_{eval_mode}_{inference}_{split}_{threshold_str}.json"),
                 "r"))
        if check_dict_structure(results, required_keys=[mode_]):
            print(f"Results for {patient} {eval_mode} {inference} {split} _{threshold_str} already exist")
            return
        else:
            print(f"Results for {patient} {eval_mode} {inference} {split}_{threshold_str} is incomplete")
    else:
        results = {}
    env_train = game.Game(path=os.path.join(config["game_path"], eval_mode, patient, "train", str(split)),
                          env_step_limit=config["env_step_limit"],
                          wrong_answer=config["wrong_answer"],
                          emb=config["emb"], hc=config["hc"],
                          embedding_dim=config["embedding_dim"],
                          wording=config["wording"], evaluation=config["evaluation"],
                          random_scenarios=True,
                          reward_scale=config["reward_scale"], reduced=config["reduced"], penalty=config["penalty"])
    env_train_eval = game.Game(path=os.path.join(config["game_path"], eval_mode, patient, "train", str(split)),
                               env_step_limit=config["env_step_limit"],
                               wrong_answer=config["wrong_answer"], emb=config["emb"],
                               hc=config["hc"],
                               embedding_dim=config["embedding_dim"],
                               wording=config["wording"], evaluation=config["evaluation"],
                               random_scenarios=False,
                               reward_scale=config["reward_scale"], reduced=config["reduced"])
    env_val = game.Game(path=os.path.join(config["game_path"], eval_mode, patient, "val", str(split)),
                        env_step_limit=config["env_step_limit"],
                        wrong_answer=config["wrong_answer"], emb=config["emb"],
                        hc=config["hc"],
                        embedding_dim=config["embedding_dim"],
                        wording=config["wording"], evaluation=config["evaluation"],
                        random_scenarios=False,
                        reward_scale=config["reward_scale"], reduced=config["reduced"])
    env_test = game.Game(path=os.path.join(config["game_path"], eval_mode, patient, "test", str(split)),
                         env_step_limit=config["env_step_limit"],
                         wrong_answer=config["wrong_answer"],
                         emb=config["emb"],
                         hc=config["hc"],
                         embedding_dim=config["embedding_dim"],
                         wording=config["wording"], evaluation=config["evaluation"],
                         random_scenarios=False,
                         reward_scale=config["reward_scale"], reduced=config["reduced"])
    state_dim = env_train.get_state_len()
    total_num_train = env_train.get_num_of_scenarios()
    total_num_val = env_val.get_num_of_scenarios()
    total_num_test = env_test.get_num_of_scenarios()
    total_num = {"train": total_num_train, "val": total_num_val, "test": total_num_test}
    summarizer = None
    if inference == "choose":
        topk = 5
        chooser_format = llm_helper.GPTChooses(topk=topk)
        chooser = llm_helper.Chooser(prompt_format=chooser_format, model=MODEL, max_tokens=1024)
    elif inference == "normal":
        pass
    elif inference == "recommend":
        num_of_recs = 5
        recer_format = llm_helper.GPTRecs(num_of_recs=num_of_recs)
        recer = llm_helper.Recommender(prompt_format=recer_format, model=MODEL, max_tokens=1024)
    elif inference == "play":
        player_format = llm_helper.GPTPlays()
        player = llm_helper.Player(prompt_format=player_format, model=MODEL, max_tokens=2048)
    elif inference == "clin_play":
        clin_player_format = llm_helper.CLINPlays()
        clin_player = llm_helper.Player(prompt_format=clin_player_format, model=MODEL, max_tokens=1024)
    elif inference == "clin_choose":
        topk = 5
        clin_chooser_format = llm_helper.CLINChooses(topk=topk)
        clin_chooser = llm_helper.Chooser(prompt_format=clin_chooser_format, model=MODEL,
                                          max_tokens=1024)
    elif inference == "clin_recommend":
        num_of_recs = 5
        clin_recer_format = llm_helper.CLINRecs(num_of_recs=num_of_recs)
        clin_recer = llm_helper.Recommender(prompt_format=clin_recer_format, model=MODEL,
                                            max_tokens=1024)
    elif inference == "h1_v2":
        topk = 5
        chooser_format = llm_helper.GPTChooses_or_Vetos_v2(topk=topk)
        llm = llm_helper.Chooser(prompt_format=chooser_format, model=MODEL, max_tokens=1024)
        player_format = llm_helper.GPTPlays()
        player = llm_helper.Player(prompt_format=player_format, model=MODEL, max_tokens=1024)
    elif inference == "h2":
        topk = 5
        llm_format = llm_helper.GPTChooses_or_Recs(topk=topk)
        llm = llm_helper.Chooser_or_Recommender(prompt_format=llm_format, model=MODEL, max_tokens=1024)
    elif inference == "h3":
        num_of_recs = 5
        recer_format = llm_helper.GPTRecs(num_of_recs=num_of_recs)
        recer = llm_helper.Recommender(prompt_format=recer_format, model=MODEL, max_tokens=1024)
    elif inference == "h4":
        num_of_recs = 5
        recer_format = llm_helper.GPTRecs(num_of_recs=num_of_recs)
        recer = llm_helper.Recommender(prompt_format=recer_format, model=MODEL, max_tokens=1024)
        topk = 5
        chooser_format = llm_helper.GPTChooses(topk=topk)
        chooser = llm_helper.Chooser(prompt_format=chooser_format, model=MODEL, max_tokens=1024)
    elif inference == "h5":
        num_of_recs = 5
        recer_format = llm_helper.GPTRecs_with_fallback(num_of_recs=num_of_recs)
        recer = llm_helper.Recommender_with_retry(prompt_format=recer_format, model=MODEL, max_tokens=1024)
    elif inference == "h6":
        topk = 5
        chooser_format = llm_helper.GPTChooses_or_Vetos_v2(topk=topk)
        llm = llm_helper.Chooser(prompt_format=chooser_format, model=MODEL, max_tokens=1024)
    elif inference == "h7":
        num_of_recs = 5
        recer_format = llm_helper.GPTRecs(num_of_recs=num_of_recs)
        recer = llm_helper.Recommender(prompt_format=recer_format, model=MODEL, max_tokens=1024)
        topk = 5
        player_format = llm_helper.GPTPlays()
        player = llm_helper.Player(prompt_format=player_format, model=MODEL, max_tokens=1024)
    elif inference == "clin_h1_v2":
        topk = 5
        chooser_format = llm_helper.CLINChooses_or_Vetos_v2(topk=topk)
        llm = llm_helper.Chooser(prompt_format=chooser_format, model=MODEL, max_tokens=1024)
        player_format = llm_helper.CLINPlays()
        player = llm_helper.Player(prompt_format=player_format, model=MODEL, max_tokens=1024)
    elif inference == "clin_h2":
        topk = 5
        llm_format = llm_helper.CLINChooses_or_Recs(topk=topk)
        llm = llm_helper.Chooser_or_Recommender(prompt_format=llm_format, model=MODEL, max_tokens=1024)
    elif inference == "clin_h3":
        num_of_recs = 5
        recer_format = llm_helper.CLINRecs(num_of_recs=num_of_recs)
        recer = llm_helper.Recommender(prompt_format=recer_format, model=MODEL, max_tokens=1024)
    elif inference == "clin_h4":
        num_of_recs = 5
        recer_format = llm_helper.CLINRecs(num_of_recs=num_of_recs)
        recer = llm_helper.Recommender(prompt_format=recer_format, model=MODEL, max_tokens=1024)
        topk = 5
        chooser_format = llm_helper.CLINChooses(topk=topk)
        chooser = llm_helper.Chooser(prompt_format=chooser_format, model=MODEL, max_tokens=1024)
    elif inference == "clin_h5":
        num_of_recs = 5
        recer_format = llm_helper.CLINRecs_with_fallback(num_of_recs=num_of_recs)
        recer = llm_helper.Recommender_with_retry(prompt_format=recer_format, model=MODEL, max_tokens=1024)
    elif inference == "clin_h6":
        topk = 5
        chooser_format = llm_helper.CLINChooses_or_Vetos_v2(topk=topk)
        llm = llm_helper.Chooser(prompt_format=chooser_format, model=MODEL, max_tokens=1024)
    elif inference == "clin_h7":
        num_of_recs = 5
        recer_format = llm_helper.CLINRecs(num_of_recs=num_of_recs)
        recer = llm_helper.Recommender(prompt_format=recer_format, model=MODEL, max_tokens=1024)
        topk = 5
        player_format = llm_helper.CLINPlays()
        player = llm_helper.Player(prompt_format=player_format, model=MODEL, max_tokens=1024)
    else:
        raise ValueError(f"Invalid inference type: {inference}")
    #################### LOAD MODEL ####################
    if inference not in ["play", "clin_play"]:
        agent = DRRNAgent(config, state_dim)
        model = torch.load(os.path.join(f"./models/{patient}_{eval_mode}_{split}", 'best_model.pt'))
        agent.policy_network = model
        agent.target_network = model
    #################### EVALUATE ####################
    for mode in [mode_]:
        prev_exp = None
        if mode == "test":
            if inference == "play" or inference == "clin_play":
                if "train" in results.keys() and "history" in results["train"].keys():
                    prev_exp = results["train"]["history"]
            else:
                if os.path.exists(os.path.join(f'/data/radmehr/results_{MODEL}/', eval_mode, patient,
                                               f"{patient}_{eval_mode}_normal_0_0.json")):
                    prev_results = json.load(
                        open(os.path.join(f'/data/radmehr/results_{MODEL}/', eval_mode, patient,
                                          f"{patient}_{eval_mode}_normal_0_0.json"), "r"))
                    if "train" in prev_results.keys() and "history" in prev_results["train"].keys():
                        prev_exp = prev_results["train"]["history"]
            print(prev_exp)
        # prev_exp = None
        #######################
        if mode not in results.keys():
            results[mode] = {}
        elif check_subkeys(results[mode]):
            # print(f"Results for {patient} {eval_mode} {inference}{mode}{threshold_str} already exist")
            continue
        else:
            results[mode] = {}
        env = env_val if mode == "val" else env_test if mode == "test" else env_train_eval
        total_score = 0
        total_traj_score = 0
        total_combined = 0
        total_eff_score = 0
        episodes = []
        vetoss = []
        reasons = []
        histories = []
        for i in range(total_num[mode]):
            ep_results = {}
            if os.path.exists(os.path.join(f'/data/radmehr/results_{MODEL}/', eval_mode, patient, mode,
                                           f"{patient}_{eval_mode}_{inference}_{split}_{threshold_str}_{i}.json")):
                ep_results = json.load(open(os.path.join(f'/data/radmehr/results_{MODEL}/', eval_mode, patient, mode,
                                                         f"{patient}_{eval_mode}_{inference}_{split}_{threshold_str}_{i}.json"),
                                            "r"))
                if check_subkeys(ep_results,
                                 sub_keys=["score", "traj_score", "eff_score", "episode", "history", "scenario_name"]):
                    # print(f"Results for {patient} {eval_mode} {inference} {mode} {i} _{threshold_str} already exist")
                    score = ep_results["score"]
                    traj_score = ep_results["traj_score"]
                    eff_score = ep_results["eff_score"]
                    combined = ep_results["combined"]
                    episode = ep_results["episode"]
                    history = ep_results["history"]
                    scenario_name = ep_results["scenario_name"]
                    total_score += score > 0
                    total_traj_score += traj_score
                    total_eff_score += eff_score
                    total_combined += combined
                    episodes.append(episode)
                    histories.append(history)
                    env.increase_episodes()
                    continue
            if not os.path.exists(os.path.join(f'/data/radmehr/results_{MODEL}/', eval_mode, patient, mode)):
                os.makedirs(os.path.join(f'/data/radmehr/results_{MODEL}/', eval_mode, patient, mode), exist_ok=True)
            if inference == "choose":
                score, episode, traj_score, eff_score, scenario_name, history = evaluate_episode_choose(chooser, topk,
                                                                                                        agent, env,
                                                                                                        prev_exp=prev_exp,
                                                                                                        summarizer=summarizer)
            elif inference == "normal":
                score, episode, traj_score, eff_score, scenario_name, history = evaluate_episode(agent, env,
                                                                                                 policy="softmax")
            elif inference == "recommend":
                score, episode, traj_score, eff_score, scenario_name, history = evaluate_episode_rec2(recer,
                                                                                                      num_of_recs,
                                                                                                      agent, env,
                                                                                                      prev_exp=prev_exp,
                                                                                                      summarizer=summarizer)
            elif inference == "play":
                score, episode, traj_score, eff_score, scenario_name, history = evaluate_episode_play(player, env,
                                                                                                      summarizer=summarizer,
                                                                                                      prev_exp=prev_exp)
            elif inference == "clin_play":
                score, episode, traj_score, eff_score, scenario_name, history = evaluate_episode_clin_play(mode,
                                                                                                           clin_player,
                                                                                                           env, patient,
                                                                                                           prev_exp=prev_exp,
                                                                                                           summarizer=summarizer,
                                                                                                           eval_mode=eval_mode,
                                                                                                           split=split,
                                                                                                           MODEL=MODEL)

            elif inference == "clin_choose":
                score, episode, traj_score, eff_score, scenario_name, history = evaluate_episode_clin_choose(mode,
                                                                                                             clin_chooser,
                                                                                                             topk,
                                                                                                             agent, env,
                                                                                                             patient,
                                                                                                             prev_exp=prev_exp,
                                                                                                             summarizer=summarizer,
                                                                                                             eval_mode=eval_mode,
                                                                                                             split=split,
                                                                                                             MODEL=MODEL)

            elif inference == "clin_recommend":
                score, episode, traj_score, eff_score, scenario_name, history = evaluate_episode_clin_rec(mode,
                                                                                                          clin_recer,
                                                                                                          num_of_recs,
                                                                                                          agent, env,
                                                                                                          patient,
                                                                                                          prev_exp=prev_exp,
                                                                                                          summarizer=summarizer,
                                                                                                          eval_mode=eval_mode,
                                                                                                          split=split,
                                                                                                          MODEL=MODEL)
            elif inference == "h1_v2":
                score, episode, traj_score, eff_score, scenario_name, history, vetos, reason = evaluate_episode_h1_v2(
                    llm, player, topk,
                    agent, env,
                    prev_exp=prev_exp,
                    summarizer=summarizer)
            elif inference == "h2":
                score, episode, traj_score, eff_score, scenario_name, history, vetos, reason = evaluate_episode_h2(llm,
                                                                                                                   topk,
                                                                                                                   agent,
                                                                                                                   env,
                                                                                                                   prev_exp=prev_exp,
                                                                                                                   summarizer=summarizer)
            elif inference == "h3":
                score, episode, traj_score, eff_score, scenario_name, history, vetos, reason = evaluate_episode_h3(
                    recer, num_of_recs,
                    agent, env,
                    prev_exp=prev_exp,
                    summarizer=summarizer,
                    threshold=threshold)
            elif inference == "h4":
                score, episode, traj_score, eff_score, scenario_name, history, vetos, reason = evaluate_episode_h4(
                    recer, chooser, topk,
                    agent, env,
                    prev_exp=prev_exp,
                    summarizer=summarizer,
                    threshold=threshold)
            elif inference == "h5":
                score, episode, traj_score, eff_score, scenario_name, history, vetos, reason = evaluate_episode_h5(
                    recer, num_of_recs,
                    agent, env,
                    prev_exp=prev_exp,
                    summarizer=summarizer,
                    threshold=threshold)
            elif inference == "h6":
                score, episode, traj_score, eff_score, scenario_name, history, vetos, reason = evaluate_episode_h6(llm,
                                                                                                                   topk,
                                                                                                                   agent,
                                                                                                                   env,
                                                                                                                   prev_exp=prev_exp,
                                                                                                                   summarizer=summarizer, )
            elif inference == "h7":
                score, episode, traj_score, eff_score, scenario_name, history, vetos, reason = evaluate_episode_h7(
                    recer, player, topk,
                    agent, env,
                    prev_exp=prev_exp,
                    summarizer=summarizer, threshold=threshold)
            elif inference == "clin_h1_v2":
                score, episode, traj_score, eff_score, scenario_name, history, vetos, reason = evaluate_episode_clin_h1_v2(
                    mode,
                    llm, player, topk,
                    agent, env,
                    patient=patient,
                    prev_exp=prev_exp,
                    summarizer=summarizer,
                    eval_mode=eval_mode,
                    split=split,
                    threshold=threshold, MODEL=MODEL)
            elif inference == "clin_h2":
                score, episode, traj_score, eff_score, scenario_name, history, vetos, reason = evaluate_episode_clin_h2(
                    mode,
                    llm,
                    topk,
                    agent,
                    env,
                    patient,
                    prev_exp=prev_exp,
                    summarizer=summarizer,
                    eval_mode=eval_mode,
                    split=split,
                    threshold=threshold, MODEL=MODEL)
            elif inference == "clin_h3":
                score, episode, traj_score, eff_score, scenario_name, history, vetos, reason = evaluate_episode_clin_h3(
                    mode,
                    recer, num_of_recs,
                    agent, env,
                    patient,
                    prev_exp=prev_exp,
                    summarizer=summarizer,
                    eval_mode=eval_mode,
                    split=split,
                    threshold=threshold, MODEL=MODEL)
            elif inference == "clin_h4":
                score, episode, traj_score, eff_score, scenario_name, history, vetos, reason = evaluate_episode_clin_h4(
                    mode,
                    recer, chooser, topk,
                    agent=agent, env=env,
                    patient=patient,
                    prev_exp=prev_exp,
                    summarizer=summarizer,
                    eval_mode=eval_mode,
                    split=split,
                    threshold=threshold, MODEL=MODEL)
            elif inference == "clin_h5":
                score, episode, traj_score, eff_score, scenario_name, history, vetos, reason = evaluate_episode_clin_h5(
                    mode,
                    recer, num_of_recs,
                    agent=agent, env=env,
                    patient=patient,
                    prev_exp=prev_exp,
                    summarizer=summarizer,
                    eval_mode=eval_mode,
                    split=split,
                    threshold=threshold, MODEL=MODEL)
            elif inference == "clin_h6":
                score, episode, traj_score, eff_score, scenario_name, history, vetos, reason = evaluate_episode_clin_h6(
                    mode,
                    llm,
                    topk,
                    agent=agent, env=env,
                    patient=patient,
                    prev_exp=prev_exp,
                    summarizer=summarizer,
                    eval_mode=eval_mode,
                    split=split,
                    threshold=threshold, MODEL=MODEL)
            elif inference == "clin_h7":
                score, episode, traj_score, eff_score, scenario_name, history, vetos, reason = evaluate_episode_clin_h7(
                    mode,
                    recer, player, topk,
                    agent=agent, env=env,
                    patient=patient,
                    prev_exp=prev_exp,
                    summarizer=summarizer,
                    eval_mode=eval_mode,
                    split=split,
                    threshold=threshold, MODEL=MODEL)
            else:
                raise ValueError("Inference mode not supported")
            ep_results["score"] = score
            ep_results["combined"] = (score > 0) * traj_score
            ep_results["traj_score"] = traj_score
            ep_results["eff_score"] = eff_score
            ep_results["episode"] = episode
            ep_results["history"] = history
            ep_results["scenario_name"] = scenario_name
            json.dump(ep_results, open(os.path.join(f'/data/radmehr/results_{MODEL}/', eval_mode, patient, mode,
                                                    f"{patient}_{eval_mode}_{inference}_{split}_{threshold_str}_{i}.json"),
                                       "w"),
                      indent=4)
            total_score += score > 0
            total_traj_score += traj_score
            total_eff_score += eff_score
            total_combined += ((score > 0) * traj_score)
            episodes.append(episode)
            try:
                vetoss.append(vetos)
            except:
                pass
            try:
                reasons.append(reason)
            except:
                pass
            histories.append(history)
            env.increase_episodes()
        results[mode]["score"] = (total_score / total_num[mode])
        results[mode]["traj_score"] = (total_traj_score / total_num[mode])
        results[mode]["eff_score"] = (total_eff_score / total_num[mode])
        results[mode]["combined"] = (total_combined / total_num[mode])
        results[mode]["episode"] = episodes
        results[mode]["history"] = histories
        results[mode]["vetos"] = vetoss
        results[mode]["reasons"] = reasons
        json.dump(results, open(
            os.path.join(f'/data/radmehr/results_{MODEL}/', eval_mode, patient,
                         f"{patient}_{eval_mode}_{inference}_{split}_{threshold_str}.json"), "w"),
                  indent=4)


def rename_folder(old_name, new_name):
    try:
        os.rename(old_name, new_name)
        print(f"Folder '{old_name}' renamed to '{new_name}' successfully.")
    except FileNotFoundError:
        print(f"Folder '{old_name}' not found.")
    except FileExistsError:
        print(f"Folder '{new_name}' already exists.")


def remove_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        # print(f"Folder '{folder_path}' removed successfully.")
    except FileNotFoundError:
        print(f"Folder '{folder_path}' not found.")
    except PermissionError:
        print(f"Permission denied to remove folder '{folder_path}'.")


def copy_folder(source_folder, destination_folder):
    try:
        # Check if the destination folder exists, and if not, create it
        if os.path.exists(destination_folder):
            shutil.rmtree(destination_folder)
            # print(f"Deleted folder '{destination_folder}'.")

        # Copy the source folder and its contents to the destination folder
        shutil.copytree(source_folder, destination_folder)
        # print(f"Folder '{source_folder}' copied to '{destination_folder}' successfully.")
    except FileNotFoundError:
        raise FileNotFoundError
        # print(f"Folder '{source_folder}' not found.")
    except FileExistsError:
        raise FileExistsError
        # print(f"Folder '{destination_folder}' already exists.")


def softmax(q_values, temperature):
    """
    Apply softmax function with temperature to a set of Q-values.

    :param q_values: A tensor of Q-values for each action.
    :param temperature: The temperature parameter for softmax.
                        Higher values increase exploration.
    :return: The probabilities for each action.
    """
    q_values_temp = q_values / temperature
    exp_q_values = torch.exp(q_values_temp - torch.max(q_values_temp))
    probabilities = exp_q_values / torch.sum(exp_q_values)

    return probabilities


def print_table(scores):
    # Headers for the table
    # print(f"{'Mode/Type':<15}{'Choose':<20}{'Recommend':<20} {'Normal':<20}")

    for mode in ['train', 'val', 'test']:
        # print(f"{mode:<15}", end='')

        for typ in ['choose', 'recomend', 'normal']:
            score, traj_score = scores[typ][mode]["score"], scores[typ][mode]["traj_score"]
            # print(f"{score}/{traj_score:<20}", end='')

        # print()


def check_dict_structure(results, required_keys=["train", "test", "val"],
                         sub_keys=["score", "traj_score", "eff_score", "episode"]):
    # Check if the required keys are present in the main dictionary
    for key in required_keys:
        if key not in results:
            return False

    # Check if the nested dictionaries under the required keys have the required sub keys
    for key in required_keys:
        sub_dict = results[key]
        if not all(sub_key in sub_dict for sub_key in sub_keys):
            return False

    return True


def check_subkeys(sub_dict, sub_keys=["score", "traj_score", "eff_score", "episode"]):
    if not all(sub_key in sub_dict for sub_key in sub_keys):
        return False
    return True


def RL_veto(llm_recs, values, threshold):
    veto = True if threshold != 1 else False
    if 1 > threshold > 0:
        idxs = np.argsort(values)[::-1]
        threshold_idx = int(threshold * len(values))
        # take the threshold number of actions
        top_idxs = idxs[:threshold_idx]
        # are there any llm recommendations in the top actions
        for rec in llm_recs:
            if rec in top_idxs:
                veto = False
                break
    return veto


def get_recs(recer, history, subject, problem, valid_subjects, valid_topics, valid_causes, previous_suggestions,
             summarizer, tries, valid_acts, posttest=False, prev_exp=None, summary=None):
    learning_id = None
    response = recer.rec(history, subject, problem, valid_subjects, valid_topics, valid_causes,
                         previous_suggestions=previous_suggestions, summary=summary,
                         posttest=posttest, prev_exp=prev_exp, summarizer=summarizer)
    og_response = response
    parts = seprate(response)

    if summary is not None:
        if len(parts) > 3:
            learning_id = parts[0]
            reason = parts[1]
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
    else:
        if len(parts) > 2:
            reason = parts[0]
            recs = "\n".join(parts[1:])
        elif len(parts) == 2:
            reason, recs = parts
        elif len(parts) == 1:
            reason = ""
            recs = parts[0]
        else:
            raise ValueError("Invalid response")

    reason, recs = reason.strip("\n").strip(), recs.strip("\n").strip()

    parsed_responses = []
    recs_splitted = [reason] + recs.split("\n") + re.split(r' (?=\d)', recs)
    for x in recs_splitted:
        parsed_responses.append(
            parse_string_to_dict((x.split(". ")[-1]), valid_subjects, valid_topics, valid_causes,
                                 replace_closest=tries > 5, posttest=posttest))
    recs_idxs = []
    valid_responses = []
    for x in parsed_responses:
        if x in valid_acts:
            recs_idxs.append(valid_acts.index(x))
            valid_responses.append(valid_acts[recs_idxs[-1]])
    return recs_idxs, valid_responses, reason, og_response, learning_id


def evaluate_episode_h1_v2(chooser, player, topk, agent, env, policy="softmax", prev_exp=None, summarizer=None):
    episode = []
    step = 0
    score = 0
    done = False
    agent.reset_dictionaries()
    history = []
    reasons = []
    vetos = []
    ob, valid_acts, hc = env.reset()
    valid_subjects = env.scenario["subjects"]
    valid_topics = env.scenario["topics"]
    valid_causes = env.scenario["causes"]
    history.append(ob[1])
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
        sorted_idxs = np.argsort(action_values[0].detach().cpu().numpy())
        if not posttest:
            choices_idxs = sorted_idxs[-1:-topk - 1:-1]
        else:
            choices_idxs = sorted_idxs[-1:-3:-1]
        choices = [valid_acts[i]["sentence"] for i in choices_idxs]
        # print("#" * 100)
        # print("RL CHOICES")
        # print(choices)
        if len(valid_acts) > 1:
            is_valid = False
            tries = 0
            while not is_valid:
                tries += 1
                if not is_valid and tries > 3:
                    chosen_action = 1
                    break

                response = chooser.choose(history, subject, problem, choices, posttest=posttest, prev_exp=prev_exp)
                parts = seprate(response)
                if len(parts) > 2:
                    reason = parts[0]
                    chosen_action = "\n".join(parts[1:])
                elif len(parts) == 2:
                    reason, chosen_action = parts
                elif len(parts) == 1:
                    reason = ""
                    chosen_action = parts[0]
                else:
                    raise ValueError("Invalid response")
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
                                           posttest=posttest, prev_exp=prev_exp, summarizer=summarizer)
                    parts = seprate(response)
                    if len(parts) > 2:
                        reason = parts[0]
                        action = "\n".join(parts[1:])
                    elif len(parts) == 2:
                        reason, action = parts
                    elif len(parts) == 1:
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


def seprate(response):
    # print(response)
    existing_seperators = [x for x in ["$$$\n", "###\n", "***\n"] if response.find(x) > -1]
    # sort the seperators by the first occurence
    existing_seperators.sort(key=lambda x: response.find(x))
    parts = []
    for sep in existing_seperators:
        spilited = response.split(sep)
        parts.append(spilited[0])
        response = sep.join(spilited[1:])
    parts.append(response)
    # print(parts)
    return parts


def evaluate_episode_h2(llm, topk, agent, env, policy="softmax", prev_exp=None, summarizer=None):
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
                                   posttest=posttest, prev_exp=prev_exp)
                parts = seprate(response)
                if len(parts) > 3:
                    pick = parts[0]
                    reason = parts[1]
                    ending = "\n".join(parts[2::])
                elif len(parts) == 3:
                    pick, reason, ending = parts
                elif len(parts) == 2:
                    pick, ending = parts
                    reason = ""
                elif len(parts) == 1:
                    pick = "recommend"
                    reason = ""
                    ending = parts[0]
                else:
                    raise ValueError("Response not parsed correctly")
                if len(pick) > len(reason):
                    pick, reason = reason, pick
                pick = pick.strip("\n").strip()
                if "choose" in pick.lower():
                    chosen_action = ending
                    reason, chosen_action = reason.strip("\n").strip(), chosen_action.strip("\n").strip()
                    chosen_action = parse_chosen_action(chosen_action)
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
                    recs = ending
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
                        is_valid = True
                        act_probs = (softmax(action_values[0][recs_idxs], temperature=0.001 if posttest else 1))
                        chosen_act_idx = torch.multinomial(act_probs, num_samples=1).item()
                        action_str = valid_responses[chosen_act_idx]
                    if not is_valid and tries > 3:
                        raise ValueError("out of tries")
        else:
            chosen_action = 1
            action_str = (valid_acts[choices_idxs[chosen_action - 1]])

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
        vetos.append(True if pick == "recommend" else False)
        reasons.append(reason)
        state = agent.create_state(
            update_sentence=ob, hc=hc, previous_state=state)
    traj_score = sum(
        a in trace for a in env.scenario["present_actions"]) / len(env.scenario["present_actions"])
    eff_score = sum(
        a in trace for a in env.scenario["present_actions"]) / len(trace)
    agent.reset_dictionaries()
    return score, episode, traj_score, eff_score, scenario_name, history, vetos, reasons


def evaluate_episode_h3(recer, num_of_recs, agent, env, policy="softmax", prev_exp=None, summarizer=None,
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
            is_valid = False
            tries = 0
            while not is_valid:
                tries += 1
                response = recer.rec(history, subject, problem, valid_subjects, valid_topics, valid_causes,
                                     posttest=posttest, prev_exp=prev_exp, summarizer=summarizer)
                parts = seprate(response)
                if len(parts) > 2:
                    reason = parts[0]
                    recs = "\n".join(parts[1:])
                elif len(parts) == 2:
                    reason, recs = parts
                elif len(parts) == 1:
                    reason = ""
                    recs = parts[0]
                else:
                    raise ValueError("Invalid response")

                parsed_responses = []
                recs_splitted = [reason] + recs.split("\n") + re.split(r' (?=\d)', recs)
                for x in recs_splitted:
                    parsed_responses.append(
                        parse_string_to_dict((x.split(". ")[-1]), valid_subjects, valid_topics, valid_causes,
                                             replace_closest=tries > 3, posttest=posttest))
                print(len(parsed_responses))
                print(len(valid_acts))
                recs_idxs = []
                valid_responses = []
                for x in parsed_responses:
                    if x in valid_acts:
                        recs_idxs.append(valid_acts.index(x))
                        valid_responses.append(valid_acts[recs_idxs[-1]])
                print(valid_responses)

                if len(recs_idxs) > 0:

                    veto = RL_veto(recs_idxs, action_values[0].detach().cpu().numpy(), threshold=threshold)
                    vetos.append(veto)
                    is_valid = True
                    act_probs = (softmax(action_values[0][recs_idxs], temperature=0.001 if posttest else 1))
                    chosen_act_idx = torch.multinomial(act_probs, num_samples=1).item()
                    if not veto:
                        action_str = valid_responses[chosen_act_idx]
                    else:
                        # print("*" * 50)
                        # print("VETO")
                        action_str = valid_acts[action_idx[0]]
                        # print(action_str)
                        # print("*" * 50)
                if not is_valid and tries > 3:
                    raise ValueError("out of tries")
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


def evaluate_episode_h4(recer, chooser, num_of_recs, agent, env, policy="softmax", prev_exp=None, summarizer=None,
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
            is_valid_1 = False
            tries = 0
            while not is_valid_1:
                tries += 1
                response = recer.rec(history, subject, problem, valid_subjects, valid_topics, valid_causes,
                                     posttest=posttest, prev_exp=prev_exp, summarizer=summarizer)
                parts = seprate(response)
                print(parts)
                if len(parts) > 2:
                    reason = parts[0]
                    recs = "\n".join(parts[1:])
                elif len(parts) == 2:
                    reason, recs = parts
                elif len(parts) == 1:
                    reason = ""
                    recs = parts[0]
                else:
                    raise ValueError("Invalid response")
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

                    veto = RL_veto(recs_idxs, action_values[0].detach().cpu().numpy(), threshold=threshold)
                    vetos.append(veto)
                    is_valid_1 = True
                    act_probs = (softmax(action_values[0][recs_idxs], temperature=0.001 if posttest else 1))
                    chosen_act_idx = torch.multinomial(act_probs, num_samples=1).item()
                    if not veto:
                        action_str = valid_responses[chosen_act_idx]
                    else:
                        sorted_idxs = np.argsort(action_values[0].detach().cpu().numpy())
                        topk = num_of_recs
                        if not posttest:
                            choices_idxs = sorted_idxs[-1:-topk - 1:-1]
                        else:
                            choices_idxs = sorted_idxs[-1:-3:-1]
                        choices = [valid_acts[i]["sentence"] for i in choices_idxs]
                        if len(valid_acts) > 1:
                            is_valid_2 = False
                            tries_2 = 0
                            while not is_valid_2:
                                tries_2 += 1

                                response = chooser.choose(history, subject, problem, choices, posttest=posttest,
                                                          prev_exp=prev_exp)
                                parts = seprate(response)
                                print(parts)
                                if len(parts) > 2:
                                    reason = parts[0]
                                    chosen_action = "\n".join(parts[1:])
                                elif len(parts) == 2:
                                    reason, chosen_action = parts
                                elif len(parts) == 1:
                                    reason = ""
                                    chosen_action = parts[0]
                                else:
                                    raise ValueError("Invalid response")
                                reason, chosen_action = reason.strip("\n").strip(), chosen_action.strip("\n").strip()
                                chosen_action = parse_chosen_action(chosen_action)

                                if chosen_action.isnumeric():
                                    chosen_action = int(chosen_action)
                                    reason = reason.split(": ")[-1]
                                    if chosen_action <= len(choices_idxs) and chosen_action > 0:
                                        chosen_action = chosen_action
                                        reason = reason
                                        is_valid_2 = True
                                else:
                                    if tries_2 > 3:
                                        chosen_action = 1
                                        is_valid_2 = True

                            reasons.append(reason)
                        else:
                            chosen_action = 1
                        action_str = (valid_acts[choices_idxs[chosen_action - 1]])
                if not is_valid_1 and tries > 3:
                    raise ValueError("out of tries")
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


def evaluate_episode_h5(recer, num_of_recs, agent, env, policy="softmax", prev_exp=None, summarizer=None,
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
                                                                                     previous_suggestions, summarizer,
                                                                                     tries,
                                                                                     valid_acts, posttest=False,
                                                                                     prev_exp=None)
                print(response)
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


def evaluate_episode_h6(chooser, topk, agent, env, policy="softmax", prev_exp=None, summarizer=None):
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
        sorted_idxs = np.argsort(action_values[0].detach().cpu().numpy())
        if not posttest:
            choices_idxs = sorted_idxs[-1:-topk - 1:-1]
        else:
            choices_idxs = sorted_idxs[-1:-3:-1]
        choices = [valid_acts[i]["sentence"] for i in choices_idxs]
        # print("#" * 100)
        # print("RL CHOICES")
        # print(choices)
        if len(valid_acts) > 1:
            is_valid = False
            tries = 0
            while not is_valid:
                tries += 1
                if not is_valid and tries > 3:
                    # print('Out of tries')
                    chosen_action = 1
                    break

                response = chooser.choose(history, subject, problem, choices, posttest=posttest, prev_exp=prev_exp)
                parts = seprate(response)
                if len(parts) > 2:
                    reason = parts[0]
                    chosen_action = "\n".join(parts[1:])
                elif len(parts) == 2:
                    reason, chosen_action = parts
                elif len(parts) == 1:
                    reason = ""
                    chosen_action = parts[0]
                else:
                    raise ValueError("Invalid response")
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
            action_str = valid_acts[action_idx[0]]
        else:
            action_str = (valid_acts[choices_idxs[chosen_action - 1]])
        vetos.append(chosen_action == 0)

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


def evaluate_episode_h7(recer, player, num_of_recs, agent, env, policy="softmax", prev_exp=None, summarizer=None,
                        threshold=0.5, MODEL=""):
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
            is_valid = False
            tries = 0
            while not is_valid:
                tries += 1
                response = recer.rec(history, subject, problem, valid_subjects, valid_topics, valid_causes,
                                     posttest=posttest, prev_exp=prev_exp, summarizer=summarizer)
                print("#####recoms")
                print(response)
                parts = seprate(response)
                if len(parts) > 2:
                    reason = parts[0]
                    recs = "\n".join(parts[1:])
                elif len(parts) == 2:
                    reason, recs = parts
                elif len(parts) == 1:
                    reason = ""
                    recs = parts[0]
                else:
                    raise ValueError("Invalid response")
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

                    veto = RL_veto(recs_idxs, action_values[0].detach().cpu().numpy(), threshold=threshold)
                    vetos.append(veto)
                    is_valid = True
                    act_probs = (softmax(action_values[0][recs_idxs], temperature=0.001 if posttest else 1))
                    chosen_act_idx = torch.multinomial(act_probs, num_samples=1).item()
                    if not veto:
                        action_str = valid_responses[chosen_act_idx]
                    else:
                        # print("VETO")
                        if len(valid_acts) > 1:
                            is_valid_1 = False
                            tries_1 = 0
                            while not is_valid_1:
                                tries_1 += 1
                                response = player.play(history, subject, problem, valid_subjects, valid_topics,
                                                       valid_causes,
                                                       posttest=posttest, prev_exp=prev_exp, summarizer=summarizer)
                                print("####play")
                                print(response)
                                parts = seprate(response)
                                if len(parts) > 2:
                                    reason = parts[0]
                                    action = "\n".join(parts[1:])
                                elif len(parts) == 2:
                                    reason, action = parts
                                elif len(parts) == 1:
                                    reason = ""
                                    action = parts[0]
                                else:
                                    raise ValueError("Invalid response")
                                reason, action = reason.strip("\n").strip(), action.strip("\n").strip()

                                parsed_responses = []
                                for x in action.split("\n"):
                                    parsed_responses.append(
                                        parse_string_to_dict((x.split(". ")[-1]), valid_subjects, valid_topics,
                                                             valid_causes,
                                                             replace_closest=tries_1 > 3, posttest=posttest))
                                valid_responses = []
                                for x in parsed_responses:
                                    if x in valid_acts:
                                        valid_responses.append(x)

                                if len(valid_responses) > 0:
                                    is_valid_1 = True
                                    action_str = valid_responses[0]
                                    # print(action_str)
                                else:
                                    print("No valid responses")
                                    print(action)
                                    print(parsed_responses)
                                    print(len(valid_acts))
                                if not is_valid_1 and tries_1 > 3:
                                    raise ValueError("out of tries")
                            reasons.append(reason)
                        else:
                            action_str = valid_acts[0]
                if not is_valid and tries > 3:
                    raise ValueError("out of tries")
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


def evaluate_episode_clin_h1_v2(mode, chooser, player, topk, agent, env, patient, eval_mode, policy="softmax",
                                prev_exp=None, summarizer=None, inference="h1_v2", threshold=0, split=0,
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
                        # print('Out of tries')
                        chosen_action = 1
                        break

                    response = chooser.choose(history, subject, problem, choices, summary=summary, posttest=posttest,
                                              prev_exp=prev_exp)
                    parts = seprate(response)
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
                                               posttest=posttest, prev_exp=prev_exp, summarizer=summarizer)
                        parts = seprate(response)
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
                            # print(action_str)
                        else:
                            print("No valid responses")
                            print(action)
                            print(parsed_responses)
                            print(len(valid_acts))
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
        # print(data)
        json.dump(data, open(file_name, "w"))
        if score == 1:
            break
        o = summarize_ep(task, sub_task, inference=inference, mode=mode, eval_mode=eval_mode, split=split,
                         threshold=threshold, MODEL=MODEL)
    return score, episode, traj_score, eff_score, scenario_name, history, vetos, reasons


def evaluate_episode_clin_h2(mode, llm, topk, agent, env, patient, eval_mode, policy="softmax", prev_exp=None,
                             summarizer=None, inference="h2", threshold=0, split=0, MODEL="gpt-4o-2024-05-13"):
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
            # print("#" * 100)
            # print("RL CHOICES")
            # print(choices)
            if len(valid_acts) > 1:
                is_valid = False
                tries = 0
                while not is_valid:
                    tries += 1
                    response = llm.cor(history, subject, problem, valid_subjects, valid_topics, valid_causes, choices,
                                       summary=summary,
                                       posttest=posttest, prev_exp=prev_exp)
                    parts = seprate(response)
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
                                print('Out of tries')
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
        # print(data)
        json.dump(data, open(file_name, "w"))
        if score == 1:
            break
        o = summarize_ep(task, sub_task, inference=inference, mode=mode, eval_mode=eval_mode, split=split,
                         threshold=threshold, MODEL=MODEL)
    return score, episode, traj_score, eff_score, scenario_name, history, vetos, reasons


def evaluate_episode_clin_h3(mode, recer, num_of_recs, agent, env, patient, eval_mode, policy="softmax", prev_exp=None,
                             summarizer=None, inference="h3", threshold=0, split=0, MODEL="gpt-4o-2024-05-13"):
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
                                         posttest=posttest, prev_exp=prev_exp, summarizer=summarizer)

                    parts = seprate(response)
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
                            # print("*" * 50)
                            # print("VETO")
                            action_str = valid_acts[action_idx[0]]
                            # print(action_str)
                            # print("*" * 50)
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
        # print(data)
        json.dump(data, open(file_name, "w"))
        if score == 1:
            break
        o = summarize_ep(task, sub_task, inference=inference, mode=mode, eval_mode=eval_mode, split=split,
                         threshold=threshold, MODEL=MODEL)
    return score, episode, traj_score, eff_score, scenario_name, history, vetos, reasons


def evaluate_episode_clin_h4(mode, recer, chooser, num_of_recs, agent, patient, eval_mode, env, policy="softmax",
                             prev_exp=None, summarizer=None, inference="h4", threshold=0, split=0,
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
            if len(valid_acts) > 1:
                is_valid = False
                tries = 0
                while not is_valid:
                    tries += 1
                    response = recer.rec(history, subject, problem, valid_subjects, valid_topics, valid_causes,
                                         summary=summary,
                                         posttest=posttest, prev_exp=prev_exp, summarizer=summarizer)
                    parts = seprate(response)
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
                            # print("VETO")
                            sorted_idxs = np.argsort(action_values[0].detach().cpu().numpy())
                            topk = num_of_recs
                            if not posttest:
                                choices_idxs = sorted_idxs[-1:-topk - 1:-1]
                            else:
                                choices_idxs = sorted_idxs[-1:-3:-1]
                            choices = [valid_acts[i]["sentence"] for i in choices_idxs]
                            if len(valid_acts) > 1:
                                is_valid_1 = False
                                tries_1 = 0
                                while not is_valid_1:
                                    tries_1 += 1
                                    if not is_valid_1 and tries_1 > 3:
                                        chosen_action = 1
                                        break

                                    response = chooser.choose(history, subject, problem, choices, summary=summary,
                                                              posttest=posttest,
                                                              prev_exp=prev_exp)
                                    parts = seprate(response)
                                    if len(parts) > 2:
                                        reason = parts[0]
                                        chosen_action = "\n".join(parts[1:])
                                    elif len(parts) == 2:
                                        reason, chosen_action = parts
                                    elif len(parts) == 1:
                                        reason = ""
                                        chosen_action = parts[0]
                                    else:
                                        raise ValueError("Invalid response")
                                    reason, chosen_action = reason.strip("\n").strip(), chosen_action.strip(
                                        "\n").strip()
                                    if chosen_action.isnumeric():
                                        chosen_action = int(chosen_action)
                                        reason = reason.split(": ")[-1]
                                        if chosen_action <= len(choices_idxs) and chosen_action > 0:
                                            chosen_action = chosen_action
                                            reason = reason
                                            is_valid = True

                                reasons.append(reason)
                            else:
                                chosen_action = 1
                            action_str = (valid_acts[choices_idxs[chosen_action - 1]])
                        # print(action_str)
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
            saved_history.append(copy.deepcopy(history_update))

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
        env.reset()
        data = dict()
        data["taskDescription"] = TaskDescription
        data["episodeIdx"] = episodeIdx
        data["history"] = saved_history
        data["finalScore"] = score
        data["finalTrajScore"] = traj_score
        data["finalEffScore"] = eff_score
        # print(data)
        json.dump(data, open(file_name, "w"))
        if score == 1:
            break
        o = summarize_ep(task, sub_task, inference=inference, mode=mode, eval_mode=eval_mode, split=split,
                         threshold=threshold, MODEL=MODEL)
    return score, episode, traj_score, eff_score, scenario_name, history, vetos, reasons


def evaluate_episode_clin_h5(mode, recer, num_of_recs, agent, env, patient, eval_mode, policy="softmax", prev_exp=None,
                             summarizer=None, inference="h5", threshold=0, split=0, MODEL="gpt-4o-2024-05-13"):
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
                                                                                         summarizer=summarizer,
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
                            # print("VETO")
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
        # print(data)
        json.dump(data, open(file_name, "w"))
        if score == 1:
            break
        o = summarize_ep(task, sub_task, inference=inference, mode=mode, eval_mode=eval_mode, split=split,
                         threshold=threshold, MODEL=MODEL)
    return score, episode, traj_score, eff_score, scenario_name, history, vetos, reasons


def evaluate_episode_clin_h6(mode, chooser, topk, agent, env, patient, eval_mode, policy="softmax", prev_exp=None,
                             summarizer=None, inference="h6", threshold=0, split=0, MODEL="gpt-4o-2024-05-13"):
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
            # print("#" * 100)
            # print("RL CHOICES")
            # print(choices)
            if len(valid_acts) > 1:
                is_valid = False
                tries = 0
                while not is_valid:
                    tries += 1
                    if not is_valid and tries > 3:
                        # print('Out of tries')
                        chosen_action = 1
                        break

                    response = chooser.choose(history, subject, problem, choices, summary=summary, posttest=posttest,
                                              prev_exp=prev_exp)
                    parts = seprate(response)
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
                action_str = valid_acts[action_idx[0]]
            else:
                action_str = (valid_acts[choices_idxs[chosen_action - 1]])
            vetos.append(chosen_action == 0)
            history_update["observation"] = history[-1]
            history_update["rationale"] = reason
            history_update["action"] = action_str["sentence"]
            learning_ids.append(learning_id)
            saved_history.append(copy.deepcopy(history_update))

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
        env.reset()
        data = dict()
        data["taskDescription"] = TaskDescription
        data["episodeIdx"] = episodeIdx
        data["history"] = saved_history
        data["finalScore"] = score
        data["finalTrajScore"] = traj_score
        data["finalEffScore"] = eff_score
        # print(data)
        json.dump(data, open(file_name, "w"))
        if score == 1:
            break
        o = summarize_ep(task, sub_task, inference=inference, mode=mode, eval_mode=eval_mode, split=split,
                         threshold=threshold, MODEL=MODEL)
    return score, episode, traj_score, eff_score, scenario_name, history, vetos, reasons


def evaluate_episode_clin_h7(mode, recer, player, num_of_recs, agent, env, patient, eval_mode, policy="softmax",
                             prev_exp=None, summarizer=None, inference="h7", threshold=0, split=0,
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
            if len(valid_acts) > 1:
                is_valid = False
                tries = 0
                while not is_valid:
                    tries += 1
                    response = recer.rec(history, subject, problem, valid_subjects, valid_topics, valid_causes,
                                         summary=summary,
                                         posttest=posttest, prev_exp=prev_exp, summarizer=summarizer)
                    parts = seprate(response)
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
                            # print("VETO")
                            if len(valid_acts) > 1:
                                is_valid_1 = False
                                tries_1 = 0
                                while not is_valid_1:
                                    tries_1 += 1
                                    response = player.play(history, subject, problem, valid_subjects, valid_topics,
                                                           valid_causes,
                                                           summary=summary,
                                                           posttest=posttest, prev_exp=prev_exp, summarizer=summarizer)
                                    parts = seprate(response)
                                    if len(parts) > 3:
                                        learning_id = parts[0]
                                        reason = parts[1]
                                        action = "\n".join(parts[2:])
                                    elif len(parts) == 3:
                                        learning_id, reason, action = parts
                                    elif len(parts) == 2:
                                        learning_id = ""
                                        reason, action = parts
                                    elif len(parts) == 1:
                                        learning_id = ""
                                        reason = ""
                                        action = parts[0]
                                    else:
                                        raise "invalid response"
                                    learning_id = learning_id.strip("\n").strip()

                                    reason, action = reason.strip("\n").strip(), action.strip("\n").strip()

                                    parsed_responses = []
                                    for x in action.split("\n"):
                                        parsed_responses.append(
                                            parse_string_to_dict((x.split(". ")[-1]), valid_subjects, valid_topics,
                                                                 valid_causes,
                                                                 replace_closest=tries_1 > 3, posttest=posttest))
                                    valid_responses = []
                                    for x in parsed_responses:
                                        if x in valid_acts:
                                            valid_responses.append(x)

                                    if len(valid_responses) > 0:
                                        is_valid_1 = True
                                        action_str = valid_responses[0]
                                        # print(action_str)
                                    else:
                                        print("No valid responses")
                                        print(action)
                                        print(parsed_responses)
                                        print(len(valid_acts))
                                    if not is_valid_1 and tries_1 > 3:
                                        raise ValueError("OUT OF TRIES")
                                reasons.append(reason)
                                learning_ids.append(learning_id)
                            else:
                                action_str = valid_acts[0]
                        # print(action_str)
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
            saved_history.append(copy.deepcopy(history_update))

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
        env.reset()
        data = dict()
        data["taskDescription"] = TaskDescription
        data["episodeIdx"] = episodeIdx
        data["history"] = saved_history
        data["finalScore"] = score
        data["finalTrajScore"] = traj_score
        data["finalEffScore"] = eff_score
        # print(data)
        json.dump(data, open(file_name, "w"))
        if score == 1:
            break
        o = summarize_ep(task, sub_task, inference=inference, mode=mode, eval_mode=eval_mode, split=split,
                         threshold=threshold, MODEL=MODEL)
    return score, episode, traj_score, eff_score, scenario_name, history, vetos, reasons


def evaluate_episode_rec2(recer, num_of_recs, agent, env, policy="softmax", prev_exp=None, summarizer=None):
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
                                     posttest=posttest, prev_exp=prev_exp, summarizer=summarizer)
                parts = seprate(response)
                responses.append(parts)
                if len(parts) > 2:
                    reason = parts[0]
                    recs = "\n".join(parts[1:])
                elif len(parts) == 2:
                    reason, recs = parts
                elif len(parts) == 1:
                    reason = ""
                    recs = parts[0]
                else:
                    raise ValueError("Invalid response")
                reason, recs = reason.strip("\n").strip(), recs.strip("\n").strip()
                responses.append((reason))
                parsed_responses = []
                recs_splitted = [reason] + recs.split("\n") + re.split(r' (?=\d)', recs)
                for x in recs_splitted:
                    parsed_responses.append(
                        parse_string_to_dict((x.split(". ")[-1]), valid_subjects, valid_topics, valid_causes,
                                             replace_closest=tries > 3, posttest=posttest))
                responses.append(parsed_responses)
                recs_idxs = []
                valid_responses = []
                for x in parsed_responses:
                    if x in valid_acts:
                        recs_idxs.append(valid_acts.index(x))
                        valid_responses.append(valid_acts[recs_idxs[-1]])
                responses.append(valid_responses)

                if len(recs_idxs) > 0:
                    is_valid = True
                    act_probs = (softmax(action_values[0][recs_idxs], temperature=0.001 if posttest else 1))
                    chosen_act_idx = torch.multinomial(act_probs, num_samples=1).item()
                    # print(reason, recs)
                    action_str = valid_responses[chosen_act_idx]
                    # print(action_str)
                if not is_valid and tries > 3:
                    raise ValueError("OUT OF TRIES")
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
    # json.dump(responses, open(f"/data/radmehr/results/memory/{scenario_name}.json", "w"),indent=4)
    return score, episode, traj_score, eff_score, scenario_name, history


def evaluate_episode_clin_rec(mode, recer, num_of_recs, agent, env, patient, eval_mode, policy="softmax", prev_exp=None,
                              summarizer=None, split=0, MODEL="gpt-4o-2024-05-13"):
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
            transition = [env.scenario["name"], step, ob[1], ]
            valid_ids = agent.encode_actions(valid_acts)
            _, action_idx, action_values, _ = agent.act(
                [state], [valid_ids], policy=policy, eval_mode=True, action_strs=valid_acts)
            if len(valid_acts) > 1:
                is_valid = False
                tries = 0
                while not is_valid:
                    tries += 1
                    response = recer.rec(history, subject, problem, valid_subjects, valid_topics, valid_causes, summary,
                                         posttest=posttest, prev_exp=prev_exp, summarizer=summarizer)
                    responses.append(response)

                    parts = seprate(response)
                    if len(parts) > 3:
                        learning_id = parts[0]
                        reason = parts[1]
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
                        is_valid = True
                        act_probs = (softmax(action_values[0][recs_idxs], temperature=0.001 if posttest else 1))
                        chosen_act_idx = torch.multinomial(act_probs, num_samples=1).item()
                        # print(reason, recs)
                        action_str = valid_responses[chosen_act_idx]
                        # print(action_str)
                    if not is_valid and tries > 3:
                        raise ValueError("OUT OF TRIES")
            else:
                action_str = valid_acts[0]
            history_update["observation"] = history[-1]
            history_update["rationale"] = reason
            history_update["action"] = action_str["sentence"]
            learning_ids.append(learning_id)
            reasons.append(reason)
            history.append(action_str["sentence"])
            saved_history.append(copy.deepcopy(history_update))
            # print(action_str)
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
        # print(data)
        json.dump(data, open(file_name, "w"))
        if score == 1:
            break
        o = summarize_ep(task, sub_task, inference="rec", mode=mode, eval_mode=eval_mode, split=split, MODEL=MODEL)
        # json.dump(responses, open(f"/data/radmehr/results_{MODEL}/memory/{scenario_name}_clin.json", "w"),indent=4)
    return score, episode, traj_score, eff_score, scenario_name, history


def evaluate_episode_clin_choose(mode, chooser, topk, agent, env, patient, eval_mode, policy="softmax", prev_exp=None,
                                 summarizer=None, split=0, MODEL="gpt-4o-2024-05-13"):
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
            reason = ""
            # print(choices)
            if len(valid_acts) > 1:
                is_valid = False
                tries = 0
                while not is_valid:
                    tries += 1
                    if not is_valid and tries > 3:
                        chosen_action = 1
                        break
                    response = chooser.choose(history, subject, problem, choices, summary=summary, posttest=posttest,
                                              prev_exp=prev_exp, summarizer=summarizer)
                    parts = seprate(response)
                    if len(parts) > 3:
                        learning_id = parts[0]
                        reason = parts[1]
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
                    print(chosen_action)
                    if chosen_action.isnumeric():
                        chosen_action = int(chosen_action)
                        if chosen_action <= len(choices_idxs) and chosen_action > 0:
                            chosen_action = chosen_action
                            reason = reason
                            is_valid = True

                reasons.append(reason)
                # print(chosen_action)
                # print(reason)
            else:
                chosen_action = 1
                reason = ""
            action_str = (valid_acts[choices_idxs[chosen_action - 1]])
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
        # print(data)
        json.dump(data, open(file_name, "w"))
        if score == 1:
            break
        o = summarize_ep(task, sub_task, inference="choose", mode=mode, eval_mode=eval_mode, split=split, MODEL=MODEL)
    return score, episode, traj_score, eff_score, scenario_name, history


def load_summary(save_path, ep, task, sub_task, inference, mode, eval_mode, split, MODEL):
    if not os.path.exists(save_path):
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        summary = ""
        episodeIdx = 0
    else:
        episodeIdx = len(os.listdir(save_path))
        episodeIdx = max(episodeIdx, ep)
        if episodeIdx > 0:
            save_file = json.load(open(f"{save_path}/{episodeIdx - 1}.json", "r"))
            if "summary" in save_file:
                summary = save_file["summary"]
            else:
                o = summarize_ep(task, sub_task, inference=inference, mode=mode, eval_mode=eval_mode, split=split,
                                 MODEL=MODEL)
                save_file = json.load(open(f"{save_path}/{episodeIdx - 1}.json", "r"))
                if "summary" in save_file:
                    summary = save_file["summary"]
                else:
                    raise ValueError("there is a problem here")
        else:
            summary = ""
            episodeIdx = 0

    return summary, episodeIdx


def evaluate_episode_clin_play(mode, clin_player, env, patient, eval_mode, prev_exp=None, summarizer=None, split=0,
                               MODEL="gpt-4o-2024-05-13"):
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
            transition = [env.scenario["name"], step, ob[1], ]
            if len(valid_acts) > 1:
                is_valid = False
                tries = 0
                while not is_valid:
                    tries += 1
                    response = clin_player.play(history, subject, problem, valid_subjects, valid_topics, valid_causes,
                                                summary, posttest=posttest, prev_exp=prev_exp, summarizer=summarizer)

                    parts = seprate(response)
                    if len(parts) > 3:
                        learning_id = parts[0]
                        reason = parts[1]
                        action = "\n".join(parts[2:])
                    elif len(parts) == 3:
                        learning_id, reason, action = parts
                    elif len(parts) == 2:
                        learning_id = ""
                        reason, action = parts
                    elif len(parts) == 1:
                        learning_id = ""
                        reason = ""
                        action = parts[0]
                    else:
                        raise ValueError("Invalid response")
                    learning_id = learning_id.strip("\n").strip()
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
        traj_score = sum(
            a in trace for a in env.scenario["present_actions"]) / len(env.scenario["present_actions"])
        eff_score = sum(
            a in trace for a in env.scenario["present_actions"]) / len(trace)
        env.reset()
        data = dict()
        data["taskDescription"] = TaskDescription
        data["episodeIdx"] = episodeIdx
        data["history"] = saved_history
        data["finalScore"] = score
        data["finalTrajScore"] = traj_score
        data["finalEffScore"] = eff_score
        # print(data)
        json.dump(data, open(file_name, "w"))
        if score == 1:
            break

        o = summarize_ep(task, sub_task, inference="play", mode=mode, eval_mode=eval_mode, split=split, MODEL=MODEL)
    return score, episode, traj_score, eff_score, scenario_name, history


def evaluate_episode_play(player, env, summarizer=None, prev_exp=None):
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
        transition = [env.scenario["name"], step, ob[1], ]
        if len(valid_acts) > 1:
            is_valid = False
            tries = 0
            while not is_valid:
                tries += 1
                response = player.play(history, subject, problem, valid_subjects, valid_topics, valid_causes,
                                       posttest=posttest, prev_exp=prev_exp, summarizer=summarizer)
                parts = seprate(response)
                if len(parts) > 2:
                    reason = parts[0]
                    action = "\n".join(parts[1:])
                elif len(parts) == 2:
                    reason, action = parts
                elif len(parts) == 1:
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
                print("posttest", posttest)
                print("tries", tries)
                print("parsed", parsed_responses)
                valid_responses = []
                for x in parsed_responses:
                    if x in valid_acts:
                        valid_responses.append(x)

                if len(valid_responses) > 0:
                    is_valid = True
                    action_str = valid_responses[0]
                else:
                    print("No valid responses")
                    print(action)
                    print(parsed_responses)
                    print(len(valid_acts))
                if not is_valid and tries > 3:
                    raise ValueError("OUT OF TRIES")
            reasons.append(reason)
            history.append(action_str["sentence"])
        else:
            action_str = valid_acts[0]
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

    traj_score = sum(
        a in trace for a in env.scenario["present_actions"]) / len(env.scenario["present_actions"])
    eff_score = sum(
        a in trace for a in env.scenario["present_actions"]) / len(trace)
    return score, episode, traj_score, eff_score, scenario_name, history


def evaluate_episode_choose(chooser, topk, agent, env, policy="softmax", prev_exp=None, summarizer=None):
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
        # print(choices)
        if len(valid_acts) > 1:
            is_valid = False
            tries = 0
            while not is_valid:
                tries += 1
                if not is_valid and tries > 3:
                    chosen_action = 1
                    break

                response = chooser.choose(history, subject, problem, choices, posttest=posttest, prev_exp=prev_exp)
                parts = seprate(response)
                if len(parts) > 2:
                    reason = parts[0]
                    chosen_action = "\n".join(parts[1:])
                elif len(parts) == 2:
                    reason, chosen_action = parts
                elif len(parts) == 1:
                    reason = ""
                    chosen_action = parts[0]
                else:
                    raise ValueError("Invalid response")
                reason, chosen_action = reason.strip("\n").strip(), chosen_action.strip("\n").strip()
                chosen_action = parse_chosen_action(chosen_action)
                print(chosen_action)
                if chosen_action.isnumeric():
                    chosen_action = int(chosen_action)
                    reason = reason.split(": ")[-1]
                    if chosen_action <= len(choices_idxs) and chosen_action > 0:
                        chosen_action = chosen_action
                        reason = reason
                        is_valid = True

            reasons.append(reason)
        else:
            chosen_action = 1
        action_str = (valid_acts[choices_idxs[chosen_action - 1]])
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
    return score, episode, traj_score, eff_score, scenario_name, history


def evaluate_episode_choose_or_rec(cor, topk, agent, env, policy="softmax", prev_exp=None, summarizer=None):
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
        transition = [env.scenario["name"], step, ob[1], ]
        valid_ids = agent.encode_actions(valid_acts)
        _, action_idx, action_values, _ = agent.act(
            [state], [valid_ids], policy=policy, eval_mode=True, action_strs=valid_acts)
        sorted_idxs = np.argsort(action_values[0].detach().cpu().numpy())
        choices_idxs = sorted_idxs[-1:-topk - 1:-1]
        choices = [valid_acts[i]["sentence"] for i in choices_idxs]
        # print(choices)
        if len(valid_acts) > 1:
            is_valid = False
            tries = 0
            while not is_valid:
                tries += 1
                response = cor.cor(history, subject, problem, valid_subjects, valid_topics, valid_causes, choices,
                                   posttest=posttest, prev_exp=prev_exp)
                parts = seprate(response)
                if len(parts) > 3:
                    mode = parts[0]
                    reason = parts[1]
                    ending = "\n".join(parts[2:])
                elif len(parts) == 3:
                    mode, reason, ending = parts
                elif len(parts) == 2:
                    mode = ""
                    reason, ending = parts
                elif len(parts) == 1:
                    mode = ""
                    reason = ""
                    ending = parts[0]
                else:
                    raise ValueError("Invalid response")
                mode = mode.strip("\n").strip()
                if mode == "choose":

                    reason, chosen_action = reason.strip("\n").strip(), ending.strip("\n").strip()
                    if chosen_action.isnumeric():
                        chosen_action = int(chosen_action)
                        reason = reason.split(": ")[-1]
                        is_valid = True
                    if not is_valid and tries > 3:
                        # print(choices)
                        # print(chosen_action)
                        # print(reason)
                        # print('Out of tries')
                        chosen_action = choices_idxs[0]
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
                        # print(reason, recs)
                        action_str = valid_responses[chosen_act_idx]
                        # print(action_str)
                    if not is_valid and tries > 3:
                        raise ValueError("OUT OF TRIES")

            reasons.append(reason)
            # print(chosen_action)
            # print(reason)
        else:
            chosen_action = 1
        action_str = (valid_acts[choices_idxs[chosen_action - 1]])
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
    return score, episode, traj_score, eff_score, scenario_name, history


def evaluate_episode(agent, env, policy):
    episode = []
    history = []
    step = 0
    score = 0
    done = False
    agent.reset_dictionaries()
    ob, valid_acts, hc = env.reset()
    history.append(ob[1])
    scenario_name = env.scenario["name"]
    state = agent.create_state(update_sentence=ob, hc=hc)
    while not done:
        transition = [env.scenario["name"], step, ob[1], ]
        valid_ids = agent.encode_actions(valid_acts)
        _, action_idx, action_values, _ = agent.act(
            [state], [valid_ids], policy=policy, eval_mode=True, action_strs=valid_acts)
        action_idx = action_idx[0]
        action_values = action_values[0]
        action_str = valid_acts[action_idx]
        state_update, rew, done, valid_acts, hc, traj_score = env.step(ob, action_str)
        if not done:
            trace = env.trace
        history.append(action_str["sentence"])
        if not done:
            if state_update[0] == "interaction":
                history.append(".".join(state_update[1].split(".")[2:]))
            else:
                history.append(".".join(state_update[1].split(".")[-1:]))
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
    return score, episode, traj_score, eff_score, scenario_name, history
