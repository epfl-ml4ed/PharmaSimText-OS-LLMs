import os
import json


def scenario_extractor(path="./scenarios", e="cause"):
    """
    Extracts and processes scenarios from JSON files within the specified directory,
    and outputs each scenario as a separate JSON file.

    Args:
        path (str): Directory path where scenario files are located. Default is "./scenarios".
        e (str): Key for selecting specific posttest question and answer sets. Default is "cause".

    Returns:
        subjects (list): List of unique subjects derived from scenario filenames.
        num_of_subtasks (list): List with the number of scenarios in each subject.
    """
    # Extract unique subject names from JSON filenames in the directory
    subjects = list(set(
        [f.split('.')[0] for f in os.listdir(path) if f.endswith('.json')]
    ))

    num_of_subtasks = []  # To store the count of subtasks per subject

    for s in subjects:
        scenario_path = os.path.join(path, s)  # Path for each subject's scenarios
        # Load the main JSON file for the subject
        scenario_data = json.load(open(os.path.join(path, s + ".json"), encoding="utf-8"))

        num_of_scenarios = len(scenario_data["scenarios"])
        num_of_subtasks.append(num_of_scenarios)  # Append scenario count for the subject

        for i in range(num_of_scenarios):
            # Reload scenario data for each iteration to avoid mutation issues
            scenario = json.load(open(os.path.join(path, s + ".json"), encoding="utf-8"))
            scenario.update(scenario["scenarios"][i])  # Update scenario with individual scenario data
            del scenario["scenarios"]  # Remove bulk scenarios key to reduce clutter

            # Handle 'present_actions' and 'relevant_actions' fields
            if len(scenario["present_actions"]) == 0:
                scenario["present_actions"] = scenario["relevant_actions"]

            # Populate posttest actions, questions, and answers
            scenario["actions"] = {
                "posttest": scenario["posttest_actions"].get(e)
            }
            scenario["posttest_qs"] = scenario["posttest_qs"].get(e, scenario.get(e))
            scenario["posttest_as"] = scenario["posttest_as"][e]

            # Cleanup fields no longer needed in this output format
            del scenario["posttest_actions"]
            scenario["actions"]["interaction"] = scenario["interaction"]

            # Ensure the output directory exists
            if not os.path.exists(scenario_path):
                os.makedirs(scenario_path)

            # Write the processed scenario to a new JSON file
            output_filename = os.path.join(scenario_path, f"{s}_{i}.json")
            with open(output_filename, 'w', encoding="utf-8") as f:
                json.dump(scenario, f, ensure_ascii=False, indent=4)

    return subjects, num_of_subtasks
