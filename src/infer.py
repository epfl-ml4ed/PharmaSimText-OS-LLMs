import argparse
import os
import shutil
import pandas as pd
import wandb
from run_hybrid import evaluate_patient_eval_mode
import concurrent.futures
import json

# Initialize Weights and Biases (WandB) for experiment tracking
wandb.login(key="xxxxx")
eval_modes = ["wording"]
splits = [0]


def process_task(e, p, inference, mode, split, threshold, model):
    """
    Processes a single evaluation task by calling evaluate_patient_eval_mode.

    Args:
        e (str): Evaluation mode.
        p (str): Patient name.
        inference (str): Inference type.
        mode (str): Mode of evaluation (e.g., "test").
        split (int): Data split index.
        threshold (float): Threshold value for evaluation.
        model (str): Model name (e.g., "gpt-4-turbo").
    """
    threshold_str = str(round(threshold, 2))
    print(
        f"Patient: {p}, Eval mode: {e}, Inference: {inference}, Mode: {mode}, Split: {split}, Threshold: {threshold_str}, Model: {model}")
    evaluate_patient_eval_mode(p, e, inference, mode_=mode, split=split, threshold=threshold, MODEL=model)


def parse_arguments():
    """
    Parses command-line arguments for thresholds, patients, inferences, and other settings.

    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process evaluation tasks for different configurations.")
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0, 0.2, 0.4, 0.6, 0.8, 1],
                        help="List of threshold values.")
    parser.add_argument('--patients', nargs='+', type=str,
                        default=["baby", "eye", "gm", "gyno", "joint", "mother", "throat"],
                        help="List of patient categories.")
    parser.add_argument('--inferences', nargs='+', type=str,
                        default=["choose", "recommend", "play", "RL_Vetos", "RL_Interacts", "LLM_Vetos", "LLM_interacts"],
                        help="List of inferences.")
    parser.add_argument('--modes', nargs='+', type=str, default=["test"], help="Evaluation modes.")
    parser.add_argument('--model', type=str, default="gpt-4-turbo", help="Model name.")
    parser.add_argument('--par', type=int, default=2, help="Number of parallel executions.")
    parser.add_argument('--no-clin', action="store_false", dest="clin", help="Exclude clinical inferences.")
    parser.add_argument('--best-threshold', action="store_true", dest="best_threshold",
                        help="Use best threshold for each inference.")
    parser.add_argument('--results-path', type=str, default="./results_gpt-4-turbo/",)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    # Initialize variables from parsed arguments
    thresholds = args.thresholds
    patients = args.patients
    inferences = args.inferences
    modes = args.modes
    model = args.model
    use_best = args.best_threshold
    clin = args.clin
    parallel = args.par
    result_path = args.results_path
    # Initialize WandB for tracking
    wandb.init(project="hybrids", entity="epfl-ml4ed", name=f"evaluate_{model}")

    # Extend inferences with clinical options if enabled
    extended_inferences = inferences + (["clin_" + i for i in inferences] if clin else [])
    inferences = extended_inferences

    # Prepare directory for results storage

    if not os.path.exists(result_path):
        os.makedirs(result_path)
        source_path = "./results" if not os.path.exists(f"./results_{model}/") else f"./results_{model}"
        print("Copying results from the source folder")
        os.system(f"cp -r {source_path}/* {result_path}")
    else:
        print(f"results_{model} directory already exists.")
        # Check for best threshold file and copy if available
        threshold_path = ""
        if os.path.exists(f"./results_{model}/") and "best_threshold.json" in os.listdir(f"./results_{model}/"):
            threshold_path = f"./results_{model}/"
        elif os.path.exists(f"./results_{model}/results_{model}/") and "best_threshold.json" in os.listdir(
                f"./results_{model}/results_{model}/"):
            threshold_path = f"./results_{model}/results_{model}/"
        if threshold_path:
            shutil.copy(f"{threshold_path}best_threshold.json", f"{result_path}best_threshold.json")

    # Task execution setup based on threshold usage
    if not use_best:
        print("Using the default thresholds")
        tasks = [(e, p, inference, mode, split, threshold, model) for e in eval_modes for p in patients for inference in
                 inferences for mode in modes for split in splits for threshold in thresholds]

        if parallel > 1:
            # Run tasks in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
                future_to_task = {executor.submit(process_task, *task): task for task in tasks}
                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        print(f"Task {task} generated an exception: {exc}")
                    else:
                        print(f"Task {task} completed successfully.")
        else:
            # Run tasks sequentially
            for task in tasks:
                try:
                    process_task(*task)
                except Exception as e:
                    print(f"Error: {e}")
                    continue

    else:
        print("Using the best thresholds")
        with open(f"{result_path}best_threshold.json", "r") as f:
            best_thresholds = json.load(f)

        tasks = []
        for e in eval_modes:
            for p in patients:
                for inference in inferences:
                    for mode in modes:
                        for split in splits:
                            threshold = best_thresholds.get(inference, {}).get(p)
                            if pd.isna(threshold):
                                print(f"Skipping {inference} for {p} due to NaN threshold")
                                continue
                            tasks.append((e, p, inference, mode, split, threshold, model))

        if parallel > 1:
            # Run tasks in parallel with best thresholds
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
                future_to_task = {executor.submit(process_task, *task): task for task in tasks}
                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        print(f"Task {task} generated an exception: {exc}")
                    else:
                        print(f"Task {task} completed successfully.")
        else:
            # Run tasks sequentially with best thresholds
            for task in tasks:
                try:
                    process_task(*task)
                except Exception as e:
                    print(f"Error: {e}")
                    continue


if __name__ == "__main__":
    main()
