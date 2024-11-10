import random
import fasttext.util
from drrn import DRRNAgent
import llm_helper
import os
import json
import torch
from game import Game
from evaluate_RL import evaluate_episode_RL
from evaluate_LLM import evaluate_episode_play, evaluate_episode_clin_play
from evaluate_agree import evaluate_episode_choose, evaluate_episode_rec, evaluate_episode_clin_rec, \
    evaluate_episode_clin_choose
from evaluate_disagree import evaluate_episode_LLM_Vetos, evaluate_episode_LLM_Interacts, evaluate_episode_RL_Vetos, \
    evaluate_episode_RL_Interacts, evaluate_episode_clin_LLM_Vetos, evaluate_episode_clin_LLM_Interacts, \
    evaluate_episode_clin_RL_Vetos, evaluate_episode_clin_RL_Interacts

# Configuration settings
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

# Initialize randomness with seed
random.seed(config["seed"])

# Load FastText model, downloading if not present
if not os.path.exists("./lms/cc.en.300.bin"):
    fasttext.util.download_model('en', if_exists='ignore')
fasttext_model = fasttext.load_model("./lms/cc.en.300.bin")


def evaluate_patient_eval_mode(patient, eval_mode, inference, mode_, split, threshold, MODEL="gpt-4o-2024-05-13",
                               base_path='/data/radmehr/'):
    threshold_str = str(round(threshold, 2))
    b_path = f'{base_path}/results_{MODEL}/'
    patient_path = os.path.join(b_path, eval_mode, patient)

    # Ensure patient directory exists and load results if already present
    if not os.path.exists(patient_path):
        os.makedirs(patient_path)

    results_file = os.path.join(patient_path, f"{patient}_{eval_mode}_{inference}_{split}_{threshold_str}.json")
    if os.path.exists(results_file):
        results = json.load(open(results_file, "r"))
        if check_dict_structure(results, required_keys=[mode_]):
            print(f"Results for {patient} {eval_mode} {inference} {split} _{threshold_str} already exist")
            return
        else:
            print(f"Results for {patient} {eval_mode} {inference} {split}_{threshold_str} is incomplete")
    else:
        results = {}
    if mode_ == "test":
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
    # Initialize game environments for training, validation, and testing
    env_train = Game(
        path=os.path.join(config["game_path"], eval_mode, patient, "train", str(split)),
        env_step_limit=config["env_step_limit"],
        wrong_answer=config["wrong_answer"],
        emb=config["emb"],
        hc=config["hc"],
        embedding_dim=config["embedding_dim"],
        wording=config["wording"],
        evaluation=config["evaluation"],
        random_scenarios=True,
        reward_scale=config["reward_scale"],
        reduced=config["reduced"],
        penalty=config["penalty"]
    )

    env_train_eval = Game(
        path=os.path.join(config["game_path"], eval_mode, patient, "train", str(split)),
        env_step_limit=config["env_step_limit"],
        random_scenarios=False,
        **{k: config[k] for k in
           ["wrong_answer", "emb", "hc", "embedding_dim", "wording", "evaluation", "reward_scale", "reduced"]}
    )

    env_val = Game(
        path=os.path.join(config["game_path"], eval_mode, patient, "val", str(split)),
        env_step_limit=config["env_step_limit"],
        random_scenarios=False,
        **{k: config[k] for k in
           ["wrong_answer", "emb", "hc", "embedding_dim", "wording", "evaluation", "reward_scale", "reduced"]}
    )

    env_test = Game(
        path=os.path.join(config["game_path"], eval_mode, patient, "test", str(split)),
        env_step_limit=config["env_step_limit"],
        random_scenarios=False,
        **{k: config[k] for k in
           ["wrong_answer", "emb", "hc", "embedding_dim", "wording", "evaluation", "reward_scale", "reduced"]}
    )

    # Set up agent and/or inference model based on the inference type
    inference_agents = initialize_inference_agent(inference, MODEL)
    agent = None
    if inference not in ["play", "clin_play"]:
        state_dim = env_train.get_state_len()
        agent = DRRNAgent(config, state_dim)
        model_path = os.path.join(f"./models/{patient}_{eval_mode}_{split}", 'best_model.pt')
        agent.policy_network = torch.load(model_path)
        agent.target_network = torch.load(model_path)
        total_num = {"train": env_train.get_num_scenarios(), "val": env_val.get_num_scenarios(),
                     "test": env_test.get_num_scenarios()}

    # Evaluate based on mode
    evaluate_mode(env_train_eval, env_val, env_test, mode_, agent, inference, inference_agents, results, total_num,
                  patient_path, threshold_str, prev_exp, split, patient, eval_mode, MODEL, threshold, mode_)


def evaluate_mode(env_train_eval, env_val, env_test, mode_, agent, inference, inference_agents, results, total_num,
                  patient_path, threshold_str, prev_exp, split, patient, eval_mode, MODEL, threshold, mode):
    """Evaluate using the appropriate environment and inference method."""
    env = env_val if mode_ == "val" else env_test if mode_ == "test" else env_train_eval
    total_score, total_traj_score, total_eff_score, total_combined = 0, 0, 0, 0
    episodes, histories, vetoss, reasons = [], [], [], []

    for i in range(total_num[mode_]):
        ep_results = load_episode_results(patient_path, mode_, inference, threshold_str, i)
        if ep_results:
            update_totals(ep_results, total_score, total_traj_score, total_eff_score, total_combined, episodes,
                          histories, vetoss, reasons)
            env.increase_episodes()
            continue

        # Evaluate episode
        episode_data = evaluate_episode(agent, env, inference, inference_agents, prev_exp, split, patient, eval_mode,
                                        MODEL, threshold, mode)
        save_episode_results(patient_path, mode_, inference, threshold_str, i, episode_data)
        update_totals(episode_data, total_score, total_traj_score, total_eff_score, total_combined, episodes, histories,
                      vetoss, reasons)

    # Save aggregate results
    save_results(results, total_score, total_traj_score, total_eff_score, total_combined, total_num[mode_], episodes,
                 histories, vetoss, reasons, patient_path, threshold_str, mode_)


def initialize_inference_agent(inference, model):
    """Initialize the specific agent required based on inference type."""
    if inference in ["choose", "clin_choose"]:
        return {"chooser": llm_helper.Chooser(
            prompt_format=llm_helper.GPTChooses(topk=5) if inference == "choose" else llm_helper.CLINChooses(topk=5),
            model=model, max_tokens=1024)}
    elif inference in ["recommend", "clin_recommend"]:
        return {"recer": llm_helper.Recommender(
            prompt_format=llm_helper.GPTRecs(num_of_recs=5) if inference == "recommend" else llm_helper.CLINRecs(
                num_of_recs=5),
            model=model,
            max_tokens=1024)}
    elif inference in ["play", "clin_play"]:
        return {"player": llm_helper.Player(
            prompt_format=llm_helper.GPTPlays() if inference == "play" else llm_helper.CLINPlays(), model=model,
            max_tokens=1024)}
    elif inference == ["LLM_Vetos", "clin_LLM_Vetos"]:
        return {"player": llm_helper.Player(
            prompt_format=llm_helper.GPTPlays() if inference == "LLM_Vetos" else llm_helper.CLINPlays(), model=model,
            max_tokens=1024),
            "llm": llm_helper.Chooser(prompt_format=llm_helper.GPTChooses_or_Vetos(
                topk=5) if inference == "LLM_Vetos" else llm_helper.CLINChooses_or_Vetos(topk=5), model=model,
                                      max_tokens=1024)}
    elif inference in ["LLM_Interacts", "clin_LLM_Interacts"]:
        return {"player": llm_helper.Player(
            prompt_format=llm_helper.GPTPlays() if inference == "LLM_Vetos" else llm_helper.CLINPlays(), model=model,
            max_tokens=1024),
            "llm": llm_helper.Chooser_or_Recommender(prompt_format=llm_helper.GPTChooses_or_Recs(
                topk=5) if inference == "LLM_Vetos" else llm_helper.CLINChooses_or_Recs(topk=5), model=model,
                                                     max_tokens=1024)}
    elif inference in ["RL_Vetos", "clin_RL_Vetos"]:
        return {"recer": llm_helper.Recommender(prompt_format=llm_helper.GPTRecs(
            num_of_recs=5) if inference == "RL_Vetos" else llm_helper.CLINRecs(num_of_recs=5), model=model,
                                                max_tokens=1024)}
    elif inference in ["h4", "clin_h4"]:
        return {"recer": llm_helper.Recommender(prompt_format=llm_helper.GPTRecs(
            num_of_recs=5) if inference == "h4" else llm_helper.CLINRecs(num_of_recs=5), model=model,
                                                max_tokens=1024),
                "chooser": llm_helper.Chooser(
                    prompt_format=llm_helper.GPTChooses(topk=5) if inference == "h4" else llm_helper.CLINChooses(
                        topk=5),
                    model=model, max_tokens=1024)}
    elif inference in ["RL_Interacts", "clin_RL_Interacts"]:
        return {"recer": llm_helper.RecommenderWithRetry(prompt_format=llm_helper.GPTRecs_with_fallback(
            num_of_recs=5) if inference == "RL_Interacts" else llm_helper.CLINRecs_with_fallback(num_of_recs=5), model=model,
                                                         max_tokens=1024)}
    else:
        raise ValueError(f"Invalid inference type: {inference}")


def load_episode_results(patient_path, mode_, inference, threshold_str, i):
    """Load previously saved episode results if available."""
    ep_path = os.path.join(patient_path, mode_, f"{mode_}_{inference}_{threshold_str}_{i}.json")
    if os.path.exists(ep_path):
        ep_results = json.load(open(ep_path, "r"))
        if check_subkeys(ep_results,
                         sub_keys=["score", "traj_score", "eff_score", "episode", "history", "scenario_name"]):
            return ep_results
    return None


def evaluate_episode(agent, env, inference, inference_agents, prev_exp, split, patient, eval_mode, MODEL, threshold,
                     mode):
    """Evaluate a single episode based on the inference type."""
    topk = 5
    num_of_recs = 5
    summarizer = None
    if inference == "choose":
        return evaluate_episode_choose(inference_agents["chooser"], topk,
                                       agent, env,
                                       prev_exp=prev_exp,
                                       summarizer=summarizer)
    elif inference == "normal":
        return evaluate_episode_RL(agent, env,
                                   policy="softmax")
    elif inference == "recommend":
        return evaluate_episode_rec(inference_agents["recer"],
                                    num_of_recs,
                                    agent, env,
                                    prev_exp=prev_exp,
                                    summarizer=summarizer)
    elif inference == "play":
        return evaluate_episode_play(inference_agents["player"], env,
                                     summarizer=summarizer,
                                     prev_exp=prev_exp)
    elif inference == "clin_play":
        return evaluate_episode_clin_play(mode,
                                          inference_agents["player"],
                                          env, patient,
                                          prev_exp=prev_exp,
                                          summarizer=summarizer,
                                          eval_mode=eval_mode,
                                          split=split,
                                          MODEL=MODEL)

    elif inference == "clin_choose":
        return evaluate_episode_clin_choose(mode,
                                            inference_agents["chooser"],
                                            topk,
                                            agent, env,
                                            patient,
                                            prev_exp=prev_exp,
                                            summarizer=summarizer,
                                            eval_mode=eval_mode,
                                            split=split,
                                            MODEL=MODEL)

    elif inference == "clin_recommend":
        return evaluate_episode_clin_rec(mode,
                                         inference_agents["recer"],
                                         num_of_recs,
                                         agent, env,
                                         patient,
                                         prev_exp=prev_exp,
                                         summarizer=summarizer,
                                         eval_mode=eval_mode,
                                         split=split,
                                         MODEL=MODEL)
    elif inference == "LLM_Vetos":
        return evaluate_episode_LLM_Vetos(
            inference_agents["llm"], inference_agents["player"], topk,
            agent, env,
            prev_exp=prev_exp,
            summarizer=summarizer)
    elif inference == "LLM_Interacts":
        return evaluate_episode_LLM_Interacts(inference_agents["llm"],
                                   topk,
                                   agent,
                                   env,
                                   prev_exp=prev_exp,
                                   summarizer=summarizer)
    elif inference == "RL_Vetos":
        return evaluate_episode_RL_Vetos(
            inference_agents["recer"], num_of_recs,
            agent, env,
            prev_exp=prev_exp,
            summarizer=summarizer,
            threshold=threshold)
    elif inference == "RL_Interacts":
        return evaluate_episode_RL_Interacts(
            inference_agents["recer"], num_of_recs,
            agent, env,
            prev_exp=prev_exp,
            summarizer=summarizer,
            threshold=threshold)
    elif inference == "clin_LLM_Vetos":
        return evaluate_episode_clin_LLM_Vetos(
            mode,
            inference_agents["llm"], inference_agents["player"], topk,
            agent, env,
            patient=patient,
            prev_exp=prev_exp,
            summarizer=summarizer,
            eval_mode=eval_mode,
            split=split,
            threshold=threshold, MODEL=MODEL)
    elif inference == "clin_LLM_Interacts":
        return evaluate_episode_clin_LLM_Interacts(
            mode,
            inference_agents["llm"],
            topk,
            agent,
            env,
            patient,
            prev_exp=prev_exp,
            summarizer=summarizer,
            eval_mode=eval_mode,
            split=split,
            threshold=threshold, MODEL=MODEL)
    elif inference == "clin_RL_Vetos":
        return evaluate_episode_clin_RL_Vetos(
            mode,
            inference_agents["recer"], num_of_recs,
            agent, env,
            patient,
            prev_exp=prev_exp,
            summarizer=summarizer,
            eval_mode=eval_mode,
            split=split,
            threshold=threshold, MODEL=MODEL)
    elif inference == "clin_RL_Interacts":
        return evaluate_episode_clin_RL_Interacts(
            mode,
            inference_agents["recer"], num_of_recs,
            agent=agent, env=env,
            patient=patient,
            prev_exp=prev_exp,
            summarizer=summarizer,
            eval_mode=eval_mode,
            split=split,
            threshold=threshold, MODEL=MODEL)
    else:
        raise ValueError("Inference mode not supported")


def save_episode_results(patient_path, mode_, inference, threshold_str, i, ep_results):
    """Save individual episode results to a JSON file."""
    os.makedirs(os.path.join(patient_path, mode_), exist_ok=True)
    ep_file = os.path.join(patient_path, mode_, f"{mode_}_{inference}_{threshold_str}_{i}.json")
    with open(ep_file, "w") as f:
        json.dump(ep_results, f, indent=4)


def update_totals(ep_results, total_score, total_traj_score, total_eff_score, total_combined, episodes, histories,
                  vetoss, reasons):
    """Update total scores and lists with the episode results."""
    total_score += ep_results["score"] > 0
    total_traj_score += ep_results["traj_score"]
    total_eff_score += ep_results["eff_score"]
    total_combined += ep_results["combined"]
    episodes.append(ep_results["episode"])
    histories.append(ep_results["history"])
    if "vetos" in ep_results:
        vetoss.append(ep_results["vetos"])
    if "reasons" in ep_results:
        reasons.append(ep_results["reasons"])


def save_results(results, total_score, total_traj_score, total_eff_score, total_combined, total_num, episodes,
                 histories, vetoss, reasons, patient_path, threshold_str, mode_):
    """Save aggregate results to a JSON file."""
    results[mode_] = {
        "score": total_score / total_num,
        "traj_score": total_traj_score / total_num,
        "eff_score": total_eff_score / total_num,
        "combined": total_combined / total_num,
        "episode": episodes,
        "history": histories,
        "vetos": vetoss,
        "reasons": reasons
    }
    results_file = os.path.join(patient_path, f"{mode_}_{threshold_str}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)


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
