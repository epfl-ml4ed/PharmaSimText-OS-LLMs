import argparse
import os
import shutil
import pandas as pd
import wandb
from run_hybrid import evaluate_patient_eval_mode
import concurrent.futures
import json

wandb.login(key="xxxxx")
eval_modes = ["wording"]
splits = [0]


def process_task(e, p, inference, mode, split, threshold, model):
    threshold_str = str(round(threshold, 2))
    print(
        f"Patient: {p}, Eval mode: {e}, Inference: {inference}, Mode: {mode}, Split: {split}, Threshold: {threshold_str}, model: {model}")
    evaluate_patient_eval_mode(p, e, inference, mode_=mode, split=split,
                               threshold=threshold, MODEL=model)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some lists")

    parser.add_argument('--thresholds', nargs='+', type=float, default=[0, 0.2, 0.4, 0.6, 0.8, 1],
                        help="List of threshold values separated by space")
    parser.add_argument('--patients', nargs='+', type=str,
                        default=["baby", "eye", "gm", "gyno", "joint", "mother", "skin", "throat"],
                        help="List of patients separated by space")
    parser.add_argument('--inferences', nargs='+', type=str,
                        default=["choose", "recommend", "play", "h3", "h4", "h5", "h7", "h1_v2", "h2", "h6"],
                        help="List of inferences separated by space")
    parser.add_argument('--modes', nargs='+', type=str, default=["test"], help="List of modes separated by space")
    parser.add_argument('--model', type=str, default="gpt-4-turbo")
    parser.add_argument('--par', type=int, default=2, help="Run in parallel")
    parser.add_argument('--no-clin', action="store_false", dest="clin", help="Don't use clinical inferences")
    parser.add_argument('--best-threshold', action="store_true", dest="best_threshold",
                        help="Use the best threshold for each inference")
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    thresholds = args.thresholds
    patients = args.patients
    inferences = args.inferences
    modes = args.modes
    model = args.model
    use_best = args.best_threshold
    clin = args.clin
    parallel = args.par
    wandb.init(project="hybrids", entity="epfl-ml4ed", name=f"evaluate_{model}")
    # Use thresholds, patients, and inferences as needed
    print("Thresholds:", thresholds)
    print("Patients:", patients)
    print("use_best:", use_best)
    extended_inferences = inferences + (["clin_" + i for i in inferences] if clin else [])
    inferences = extended_inferences
    print(clin)
    print("Inferences:", extended_inferences)
    print("inferences", inferences)
    print("Modes:", modes)
    print("Model:", model)
    print("Parallel:", parallel)
    # if result_{model} doesn't exist, create it and copy contents of results
    if not os.path.exists(f"/data/radmehr/results_{model}/"):
        os.makedirs(f"/data/radmehr/results_{model}")
        if not os.path.exists(f"./results_{model}/"):
            print("Copying results from the default folder")
            os.system(f"cp -r results/* /data/radmehr/results_{model}/")
        else:
            print("Copying results from the custom folder")
            os.system(f"cp -r ./results_{model}/* /data/radmehr/results_{model}/")
    else:
        print(f"results_{model} already exists")
        dir = ""
        if os.path.exists(f"./results_{model}/"):
            if "best_threshold.json" in os.listdir(f"./results_{model}/"):
                dir = f"./results_{model}/"
        if os.path.exists(f"./results_{model}/results_{model}/"):
            if "best_threshold.json" in os.listdir(f"./results_{model}/results_{model}/"):
                dir = f"./results_{model}/results_{model}/"
        if dir != "":
            shutil.copy(f"{dir}best_threshold.json", f"/data/radmehr/results_{model}/best_threshold.json")
    if not use_best:
        print("Using the default thresholds")
        if parallel > 1:
            # Create a list of all task arguments
            tasks = [(e, p, inference, mode, split, threshold, model) for e in eval_modes for p in patients for
                     inference in
                     inferences for
                     mode in
                     modes for split in splits for threshold in thresholds]

            # Use ThreadPoolExecutor or ProcessPoolExecutor to run tasks in parallel
            # Adjust max_workers based on your system's capabilities and the nature of your tasks
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
                # Submit all tasks to the executor
                future_to_task = {executor.submit(process_task, *task): task for task in tasks}

                # Process the results as they complete (optional)
                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()  # You can use the result if your function returns something
                    except Exception as exc:
                        print(f"Task {task} generated an exception: {exc}")
                    else:
                        print(f"Task {task} completed successfully.")
        else:
            for e in eval_modes:
                for split in splits:
                    for inference in extended_inferences:
                        for mode in modes:
                            for p in patients:
                                for threshold in thresholds:
                                    threshold_str = str(round(threshold, 2))
                                    print(
                                        f"Patient: {p}, Inference: {inference}, Mode: {mode}, Split: {split}, Threshold: {threshold_str}, model: {model}")
                                    try:
                                        evaluate_patient_eval_mode(p, e, inference, mode_=mode, split=split,
                                                                   threshold=threshold, MODEL=model)
                                    except Exception as e:
                                        print(f"Error: {e}")
                                        continue
    else:
        print("Using the best thresholds")
        with open(f"/data/radmehr/results_{model}/best_threshold.json", "r") as f:
            best_thresholds = json.load(f)

        if parallel > 1:
            # Create a list of all task arguments
            tasks = []
            for e in eval_modes:
                for p in patients:
                    for inference in inferences:
                        for mode in modes:
                            for split in splits:
                                threshold = best_thresholds[inference][p]
                                print(threshold)
                                if pd.isna(threshold):
                                    print(f"Skipping {inference} for {p}")
                                    continue
                                tasks.append((e, p, inference, mode, split, threshold, model))
            # Use ThreadPoolExecutor or ProcessPoolExecutor to run tasks in parallel
            # Adjust max_workers based on your system's capabilities and the nature of your tasks
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
                # Submit all tasks to the executor
                future_to_task = {executor.submit(process_task, *task): task for task in tasks}

                # Process the results as they complete (optional)
                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()  # You can use the result if your function returns something
                    except Exception as exc:
                        print(f"Task {task} generated an exception: {exc}")
                    else:
                        print(f"Task {task} completed successfully.")
        else:
            for e in eval_modes:
                for split in splits:
                    for inference in extended_inferences:
                        for mode in modes:
                            for p in patients:
                                threshold = best_thresholds[inference][p]
                                threshold_str = str(round(threshold, 2))
                                print(
                                    f"Patient: {p}, Inference: {inference}, Mode: {mode}, Split: {split}, Threshold: {threshold_str}, model: {model}")
                                try:
                                    evaluate_patient_eval_mode(p, e, inference, mode_=mode, split=split,
                                                               threshold=threshold, MODEL=model)
                                except Exception as e:
                                    print(f"Error: {e}")
                                    continue


if __name__ == "__main__":
    main()
