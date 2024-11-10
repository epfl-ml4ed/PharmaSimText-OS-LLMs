import os
import json
import torch
import scipy
import fasttext
import re
import numpy as np
import pathlib
from clin_memory import summarize_ep

fasttext_model = fasttext.load_model("cc.en.300.bin")
def select_top_actions(action_values, topk, posttest):
    """
    Selects the top-k actions based on action values, considering posttest state.

    Args:
        action_values (Tensor): Values of actions.
        topk (int): Number of top actions to select.
        posttest (bool): Indicates if in posttest mode.

    Returns:
        list: Indices of the selected top actions.
    """
    sorted_idxs = np.argsort(action_values[0].detach().cpu().numpy())
    if not posttest:
        return sorted_idxs[-1:-topk - 1:-1]
    return sorted_idxs[-1:-3:-1]

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

# Function to split a response based on specific separators
def separate(response):
    """
    Splits a response string into parts based on existing separator patterns.

    Args:
        response (str): The response string to split.

    Returns:
        list: A list containing the separated parts of the response.
    """
    separators = {
        "$$$": ["\n$$$\n", "$$$\n", "\n$$$", " $$$ ", " $$$", "$$$ ", "$$$"],
        "###": ["\n###\n", "###\n", "\n###", " ### ", " ###", "### ", "###"],
        "***": ["\n***\n", "***\n", "\n***", " *** ", " ***", "*** ", "***"]
    }
    existing_separators = [
        k for k in separators if any(sep in response for sep in separators[k])
    ]
    existing_separators.sort(key=lambda x: response.find(x))

    parts = []
    for sep in existing_separators:
        split_response = response.split(sep)
        parts.append(split_response[0])
        response = sep.join(split_response[1:])
    parts.append(response)

    return parts


# Parse a chosen action to extract the first numeric segment
def parse_chosen_action(chosen_action):
    """
    Extracts the first numeric segment from the chosen action string.

    Args:
        chosen_action (str): The chosen action string.

    Returns:
        str: The extracted numeric segment.
    """
    for delimiter in [" ", ".", "\n", ")"]:
        if chosen_action.split(delimiter)[0].isnumeric():
            return chosen_action.split(delimiter)[0]
    return chosen_action


# Finds all occurrences of a value in a list
def find_all_occurrences(lst, value):
    """
    Finds all occurrences of a specified value in a list.

    Args:
        lst (list): The list to search.
        value: The value to find.

    Returns:
        list: A list of indices where the value is found.
    """
    return [i for i, x in enumerate(lst) if x == value]


# Matches a sentence with valid sentences and optionally replaces with closest match
def match(sentence, valid_sentences, replace_closest=False):
    """
    Matches a sentence with a list of valid sentences, optionally replacing with the closest match.

    Args:
        sentence (str): The input sentence.
        valid_sentences (list): A list of valid sentences.
        replace_closest (bool): Whether to replace with the closest valid sentence if no exact match.

    Returns:
        tuple: A tuple containing the matched sentence and a boolean indicating if it was an exact match.
    """
    for valid in valid_sentences:
        if sentence == valid.lower():
            return valid, True
    indicators = [(sentence in valid.lower()) or (valid.lower() in sentence) for valid in valid_sentences]
    if any(indicators):
        idx = find_all_occurrences(indicators, True)
        values = [valid_sentences[i] for i in idx]
        longest_value = max(values, key=len)
        return longest_value, True
    elif replace_closest:
        # Find and replace with closest sentence using FastText embeddings
        sentence_embedding = fasttext_model.get_sentence_vector(sentence.replace("\n", ""))
        distances = [
            1 - scipy.spatial.distance.cosine(fasttext_model.get_sentence_vector(valid), sentence_embedding)
            for valid in valid_sentences
        ]
        closest_match = valid_sentences[distances.index(max(distances))]
        return closest_match, False
    return sentence, False


# Parses a string to dictionary based on predefined command formats
def parse_string_to_dict(input_str, valid_subjects, valid_topics, valid_causes, replace_closest=False, posttest=False):
    """
    Parses an input string into a dictionary format based on predefined commands.

    Args:
        input_str (str): The input string to parse.
        valid_subjects, valid_topics, valid_causes (list): Lists of valid subjects, topics, and causes.
        replace_closest (bool): Whether to replace with closest valid entry if no match is found.
        posttest (bool): Indicates if posttest mode is enabled.

    Returns:
        dict: A dictionary containing parsed command information.
    """
    input_str = input_str.lower()
    result_dict = {"type": "", "part": "", "detail": "", "sentence": ""}

    if "(" in input_str:
        parts = input_str.split('(', 1)
        command = parts[0].split(' ')[-1]
        if posttest and replace_closest:
            command = "choose"
        args = parts[1].rsplit(')', 1)[0] if len(parts) > 1 else ""

        if command == "ask":
            result_dict["type"], result_dict["part"] = "interaction", "discuss"
            subject, topic = (args.split(',') + [""] * 2)[:2]
            subject, _ = match(subject.strip(), valid_subjects, replace_closest)
            topic, _ = match(topic.strip(), valid_topics, replace_closest)
            result_dict["detail"], result_dict[
                "sentence"] = f"{subject},{topic}", f"I want to know about the {subject}'s {topic}."

        elif command == "answer":
            result_dict.update({"type": "interaction", "part": "solution", "sentence": "I want to suggest a solution."})

        elif command == "choose":
            args, _ = match(args, valid_causes, replace_closest)
            result_dict.update({"type": "posttest", "sentence": args})

    else:
        if posttest:
            cause, _ = match(input_str, valid_causes, replace_closest)
            result_dict.update({"type": "posttest", "sentence": cause})
        else:
            result_dict.update({"type": "interaction"})
            subject, _ = match(input_str, valid_subjects, replace_closest)
            topic, _ = match(input_str, valid_topics, replace_closest)
            result_dict["part"] = "solution" if "diagnosis" in input_str else "discuss"
            result_dict["detail"], result_dict[
                "sentence"] = f"{subject},{topic}", f"I want to know about the {subject}'s {topic}."

    return result_dict


# Finds phrases in a text that start with "have"/"has" and end at a period
def find_phrase(text):
    """
    Extracts phrases from text that start with 'have' or 'has' and end at a period.

    Args:
        text (str): The text to search for phrases.

    Returns:
        list: A list of extracted phrases.
    """
    pattern = r'\b(have|has)\b(.*?)(?=\.)'
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    return [''.join(match[1]).strip() for match in matches]


def RL_veto(llm_recs, values, threshold):
    """
    Determines if a veto should be applied based on the recommendations from an LLM
    and the sorted values of actions.

    Args:
        llm_recs (list): Indices of recommended actions from the LLM.
        values (list or np.array): Array of values associated with each action.
        threshold (float): Threshold between 0 and 1. If 1, veto is disabled.
                           If between 0 and 1, it determines the proportion of top actions to check.

    Returns:
        bool: True if a veto is applied, otherwise False.
    """
    # If threshold is 1, veto is always enabled
    if threshold == 1:
        return True

    # Sort action indices by value in descending order
    idxs = np.argsort(values)[::-1]
    # Determine the number of top actions to consider based on the threshold
    threshold_idx = int(threshold * len(values))
    top_idxs = idxs[:threshold_idx]

    # Check if any LLM recommendation is within the top actions
    veto = not any(rec in top_idxs for rec in llm_recs)

    return veto

def process_response(response, valid_acts, valid_subjects, valid_topics, valid_causes, posttest, tries):
    """
    Processes the response from a player agent by extracting the reason and actions,
    validating the actions, and returning the first valid action.

    Args:
        response (str): The response generated by the player.
        valid_acts (list): List of valid actions.
        valid_subjects (list): List of valid subjects.
        valid_topics (list): List of valid topics.
        valid_causes (list): List of valid causes.
        posttest (bool): Indicates if the state is in posttest mode.
        tries (int): Number of attempts made to parse and validate the response.

    Returns:
        tuple: Reason for the action, the chosen action string, a boolean indicating
               if a valid action was found, and the selected valid action.
    """
    parts = separate(response)
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

    reason = reason.strip("\n").strip()
    action = action.strip("\n").strip()

    # Parse actions
    parsed_responses = [
        parse_string_to_dict(
            x.split(". ")[-1], valid_subjects, valid_topics, valid_causes,
            replace_closest=tries > 3, posttest=posttest
        )
        for x in action.split("\n")
    ]

    # Validate parsed responses
    valid_responses = [x for x in parsed_responses if x in valid_acts]
    is_valid = len(valid_responses) > 0
    action_str = valid_responses[0] if is_valid else None

    return reason, action, is_valid, action_str
def handle_chooser_response(chooser, history, subject, problem, choices, summary, posttest, prev_exp, choices_idxs):
    """
    Handles the response from the chooser agent, parsing and validating the chosen action.

    Args:
        chooser: The chooser agent.
        history (list): Interaction history.
        subject (str): Subject of the scenario.
        problem (str): Problem description.
        choices (list): List of available choices.
        summary (str): Optional summary context.
        posttest (bool): Indicates if in posttest mode.
        prev_exp: Optional previous experience data.
        choices_idxs (list): Indices of the available choices.

    Returns:
        tuple: Chosen action index, reason, learning ID, and validity status.
    """
    is_valid = False
    tries = 0
    while not is_valid:
        tries += 1
        if tries > 3:
            return 1, "", "", False  # Default to first action if validation fails after 3 tries

        response = chooser.choose(history, subject, problem, choices, summary=summary, posttest=posttest, prev_exp=prev_exp)
        parts = separate(response)
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
        if chosen_action.isnumeric():
            chosen_action = int(chosen_action)
            if 0 < chosen_action <= len(choices_idxs):
                return chosen_action, reason, learning_id, True
    return 1, "", "", False

def handle_chooser_response(chooser, history, subject, problem, choices, summary, posttest, prev_exp, choices_idxs):
    """
    Handles the response from the chooser agent, parsing and validating the chosen action.

    Args:
        chooser: The chooser agent.
        history (list): Interaction history.
        subject (str): Subject of the scenario.
        problem (str): Problem description.
        choices (list): List of available choices.
        summary (str): Optional summary context.
        posttest (bool): Indicates if in posttest mode.
        prev_exp: Optional previous experience data.
        choices_idxs (list): Indices of the available choices.

    Returns:
        tuple: Chosen action index, reason, learning ID, and validity status.
    """
    is_valid = False
    tries = 0
    while not is_valid:
        tries += 1
        if tries > 3:
            return 1, "", "", False  # Default to first action if validation fails after 3 tries

        response = chooser.choose(history, subject, problem, choices, summary=summary, posttest=posttest, prev_exp=prev_exp)
        parts = separate(response)
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
        if chosen_action.isnumeric():
            chosen_action = int(chosen_action)
            if 0 < chosen_action <= len(choices_idxs):
                return chosen_action, reason, learning_id, True
    return 1, "", "", False

def process_clin_recer_response(response, valid_acts, valid_subjects, valid_topics, valid_causes, posttest, tries, action_values):
    """
    Processes the response from a recommendation agent in a clinical setting,
    by parsing and validating actions, and selecting an appropriate action based on the provided values.

    Args:
        response (str): The response generated by the recommendation agent.
        valid_acts (list): List of valid actions.
        valid_subjects (list): List of valid subjects.
        valid_topics (list): List of valid topics.
        valid_causes (list): List of valid causes.
        posttest (bool): Indicates if the state is in posttest mode.
        tries (int): Number of attempts made to parse and validate the response.
        action_values (Tensor): Values associated with each valid action.

    Returns:
        tuple: Learning ID, reason for the recommendations, recommended actions string, a boolean indicating
               if a valid action was found, and the selected valid action.
    """
    parts = separate(response)
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
    reason = reason.strip("\n").strip()
    recs = recs.strip("\n").strip()

    # Parse recommendations
    parsed_responses = [
        parse_string_to_dict(
            x.split(". ")[-1], valid_subjects, valid_topics, valid_causes,
            replace_closest=tries > 3, posttest=posttest
        )
        for x in (recs.split("\n") + re.split(r' (?=\d)', recs))
    ]

    # Validate and select valid responses
    recs_idxs = [valid_acts.index(x) for x in parsed_responses if x in valid_acts]
    valid_responses = [valid_acts[idx] for idx in recs_idxs]
    is_valid = len(recs_idxs) > 0

    if is_valid:
        # Select an action using softmax over the valid responses
        act_probs = softmax(action_values[0][recs_idxs], temperature=0.001 if posttest else 1)
        chosen_act_idx = torch.multinomial(act_probs, num_samples=1).item()
        action_str = valid_responses[chosen_act_idx]
    else:
        action_str = None

    return learning_id, reason, recs, is_valid, action_str

def process_recer_response(response, valid_acts, valid_subjects, valid_topics, valid_causes, posttest, tries, action_values):
    """
    Processes the response from the recommendation agent by parsing and validating actions,
    and selecting an appropriate action based on the provided values.

    Args:
        response (str): The response generated by the recommendation agent.
        valid_acts (list): List of valid actions.
        valid_subjects (list): List of valid subjects.
        valid_topics (list): List of valid topics.
        valid_causes (list): List of valid causes.
        posttest (bool): Indicates if the state is in posttest mode.
        tries (int): Number of attempts made to parse and validate the response.
        action_values (Tensor): Values associated with each valid action.

    Returns:
        tuple: Reason for the recommendations, recommended actions string, a boolean indicating
               if a valid action was found, and the selected valid action.
    """
    parts = separate(response)
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

    reason = reason.strip("\n").strip()
    recs = recs.strip("\n").strip()

    # Parse recommendations
    parsed_responses = [
        parse_string_to_dict(
            x.split(". ")[-1], valid_subjects, valid_topics, valid_causes,
            replace_closest=tries > 3, posttest=posttest
        )
        for x in (recs.split("\n") + re.split(r' (?=\d)', recs))
    ]

    # Validate and select valid responses
    recs_idxs = [valid_acts.index(x) for x in parsed_responses if x in valid_acts]
    valid_responses = [valid_acts[idx] for idx in recs_idxs]
    is_valid = len(recs_idxs) > 0

    if is_valid:
        # Select an action using softmax over the valid responses
        act_probs = softmax(action_values[0][recs_idxs], temperature=0.001 if posttest else 1)
        chosen_act_idx = torch.multinomial(act_probs, num_samples=1).item()
        action_str = valid_responses[chosen_act_idx]
    else:
        action_str = None

    return reason, recs, is_valid, action_str

def process_clin_response(response, valid_acts, valid_subjects, valid_topics, valid_causes, posttest, tries):
    """
    Processes the response from a clinician player by extracting the learning ID, reason, and actions,
    validating the actions, and returning the first valid action.

    Args:
        response (str): The response generated by the clinician player.
        valid_acts (list): List of valid actions.
        valid_subjects (list): List of valid subjects.
        valid_topics (list): List of valid topics.
        valid_causes (list): List of valid causes.
        posttest (bool): Indicates if the state is in posttest mode.
        tries (int): Number of attempts made to parse and validate the response.

    Returns:
        tuple: Learning ID, reason for the action, the chosen action string, a boolean indicating
               if a valid action was found, and the selected valid action.
    """
    parts = separate(response)
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
    reason = reason.strip("\n").strip()
    action = action.strip("\n").strip()

    # Parse actions
    parsed_responses = [
        parse_string_to_dict(
            x.split(". ")[-1], valid_subjects, valid_topics, valid_causes,
            replace_closest=tries > 3, posttest=posttest
        )
        for x in action.split("\n")
    ]

    # Validate parsed responses
    valid_responses = [x for x in parsed_responses if x in valid_acts]
    is_valid = len(valid_responses) > 0
    action_str = valid_responses[0] if is_valid else None

    return learning_id, reason, action, is_valid, action_str

def get_recs(recer, history, subject, problem, valid_subjects, valid_topics, valid_causes, previous_suggestions,
             summarizer, tries, valid_acts, posttest=False, prev_exp=None, summary=None):
    """
    Generates recommendations based on the LLM recommendation agent and parses the response.

    Args:
        recer: The recommendation agent.
        history (str): Historical data for context.
        subject (str): Subject of the problem.
        problem (str): Problem description.
        valid_subjects (list): List of valid subjects.
        valid_topics (list): List of valid topics.
        valid_causes (list): List of valid causes.
        previous_suggestions (list): Previously suggested actions.
        summarizer: Summarizer object for handling summaries.
        tries (int): Number of attempts to replace with closest match.
        valid_acts (list): List of valid actions.
        posttest (bool): Whether the current context is post-test.
        prev_exp: Previous experience data.
        summary (str): Additional summary information.

    Returns:
        tuple: Contains recommended action indices, valid responses, reason, original response, and learning ID.
    """

    # Generate initial response from the recommendation agent
    response = recer.rec(
        history, subject, problem, valid_subjects, valid_topics, valid_causes,
        previous_suggestions=previous_suggestions, summary=summary,
        posttest=posttest, prev_exp=prev_exp, summarizer=summarizer
    )
    og_response = response
    parts = separate(response)

    # Parse response parts into learning ID, reason, and recommendations
    if summary is not None:
        learning_id, reason, recs = parse_response_with_summary(parts)
    else:
        reason, recs = parse_response_without_summary(parts)

    # Clean up whitespace for parsed reason and recommendations
    reason, recs = reason.strip(), recs.strip()

    # Parse recommendations into dictionary format
    parsed_responses = parse_recommendations(recs, reason, valid_subjects, valid_topics, valid_causes, tries, posttest)

    # Filter parsed responses for valid actions
    recs_idxs, valid_responses = filter_valid_responses(parsed_responses, valid_acts)

    return recs_idxs, valid_responses, reason, og_response, learning_id


def parse_response_with_summary(parts):
    """Parses the response parts when a summary is provided."""
    if len(parts) > 3:
        learning_id, reason, recs = parts[0], parts[1], "\n".join(parts[2:])
    elif len(parts) == 3:
        learning_id, reason, recs = parts
    elif len(parts) == 2:
        learning_id, reason, recs = "", parts[0], parts[1]
    elif len(parts) == 1:
        learning_id, reason, recs = "", "", parts[0]
    else:
        raise ValueError("Invalid response format with summary.")
    return learning_id.strip(), reason, recs


def parse_response_without_summary(parts):
    """Parses the response parts when no summary is provided."""
    if len(parts) > 2:
        reason, recs = parts[0], "\n".join(parts[1:])
    elif len(parts) == 2:
        reason, recs = parts
    elif len(parts) == 1:
        reason, recs = "", parts[0]
    else:
        raise ValueError("Invalid response format without summary.")
    return reason, recs


def parse_recommendations(recs, reason, valid_subjects, valid_topics, valid_causes, tries, posttest):
    """Splits and parses recommendations and the reason into a structured format."""
    parsed_responses = []
    recs_splitted = [reason] + recs.split("\n") + re.split(r' (?=\d)', recs)
    for x in recs_splitted:
        parsed_responses.append(
            parse_string_to_dict(
                x.split(". ")[-1], valid_subjects, valid_topics, valid_causes,
                replace_closest=(tries > 5), posttest=posttest
            )
        )
    return parsed_responses


def filter_valid_responses(parsed_responses, valid_acts):
    """Filters parsed responses to return only those that match valid actions."""
    recs_idxs = []
    valid_responses = []
    for x in parsed_responses:
        if x in valid_acts:
            recs_idxs.append(valid_acts.index(x))
            valid_responses.append(valid_acts[recs_idxs[-1]])
    return recs_idxs, valid_responses

def rl_agent_action(agent, state, valid_acts, policy, topk, posttest, chooser, history, subject, problem, prev_exp):
    """RL agent selects action, optionally validated by the chooser."""
    valid_ids = agent.encode_actions(valid_acts)
    _, action_idx, action_values, _ = agent.act([state], [valid_ids], policy=policy, eval_mode=True,
                                                action_strs=valid_acts)
    sorted_idxs = np.argsort(action_values[0].detach().cpu().numpy())
    choices_idxs = sorted_idxs[-1:-topk - 1:-1] if not posttest else sorted_idxs[-1:-3:-1]
    choices = [valid_acts[i]["sentence"] for i in choices_idxs]

    # Chooser validates the top actions
    chosen_action, reason = 1, ""
    if len(valid_acts) > 1:
        tries, is_valid = 0, False
        while not is_valid and tries <= 3:
            tries += 1
            response = chooser.choose(history, subject, problem, choices, posttest=posttest, prev_exp=prev_exp)
            reason, chosen_action = parse_response_action(response, choices)
            is_valid = validate_chosen_action(chosen_action, len(choices_idxs))

    return chosen_action, reason


def parse_response_action(response, choices):
    """Parse the response to extract reason and chosen action."""
    parts = separate(response)
    reason = parts[0] if len(parts) > 1 else ""
    chosen_action = parts[1] if len(parts) > 1 else parts[0]
    chosen_action = parse_chosen_action(chosen_action.strip())
    return reason.strip(), int(chosen_action) if chosen_action.isnumeric() else 1


def validate_chosen_action(chosen_action, num_choices):
    """Validates if the chosen action is within bounds of available choices."""
    return 0 <= chosen_action < num_choices


def handle_veto(player, history, subject, problem, valid_subjects, valid_topics, valid_causes, valid_acts, posttest, prev_exp,
                summarizer):
    """Handle veto scenario by having the player recommend an action."""
    tries, is_valid, valid_responses = 0, False, []
    while not is_valid and tries <= 3:
        tries += 1
        response = player.play(history, subject, problem, valid_subjects, valid_topics, valid_causes, posttest=posttest,
                               prev_exp=prev_exp, summarizer=summarizer)
        reason, actions = parse_player_response(response)

        valid_responses = [x for x in parse_responses(actions, valid_subjects, valid_topics, valid_causes, posttest, tries=tries) if
                           x in valid_acts]
        if valid_responses:
            is_valid = True

    if not is_valid:
        raise ValueError("Out of tries for veto handling.")
    return valid_responses[0]


def parse_player_response(response):
    """Parse player response to get reason and action(s)."""
    parts = separate(response)
    reason = parts[0] if len(parts) > 1 else ""
    action = "\n".join(parts[1:]) if len(parts) > 1 else parts[0]
    return reason.strip(), action.strip()


def parse_responses(actions, valid_subjects, valid_topics, valid_causes, posttest, tries=0):
    """Parse action responses into structured dictionary format."""
    parsed_responses = []
    for action in actions.split("\n"):
        parsed_responses.append(parse_string_to_dict(action.split(". ")[-1], valid_subjects, valid_topics, valid_causes,
                                                     replace_closest=tries > 3, posttest=posttest))
    return parsed_responses


def update_history(history, state_update):
    """Updates the history with new state information based on state type."""
    if state_update[0] == "interaction":
        history.append(".".join(state_update[1].split(".")[2:]))
    else:
        history.append(".".join(state_update[1].split(".")[-1:]))
    return history


def calculate_scores(env):
    """Calculate trajectory and efficiency scores based on environment trace."""
    trace = env.trace
    traj_score = sum(a in trace for a in env.scenario["present_actions"]) / len(env.scenario["present_actions"])
    eff_score = sum(a in trace for a in env.scenario["present_actions"]) / len(trace) if trace else 0
    return traj_score, eff_score


def rl_llm_interaction(agent, state, valid_acts, policy, topk, llm, posttest, history, subject, problem, prev_exp,
                       valid_subjects, valid_topics, valid_causes):
    """Selects action using RL agent and validates or recommends actions through LLM."""
    valid_ids = agent.encode_actions(valid_acts)
    _, action_idx, action_values, _ = agent.act([state], [valid_ids], policy=policy, eval_mode=True,
                                                action_strs=valid_acts)
    sorted_idxs = np.argsort(action_values[0].detach().cpu().numpy())
    choices_idxs = sorted_idxs[-1:-topk - 1:-1] if not posttest else sorted_idxs[-1:-3:-1]
    choices = [valid_acts[i]["sentence"] for i in choices_idxs]

    # Interaction with LLM
    tries, is_valid = 0, False
    while not is_valid and tries <= 3:
        tries += 1
        response = llm.cor(history, subject, problem, valid_subjects, valid_topics, valid_causes, choices,
                           posttest=posttest, prev_exp=prev_exp)
        pick_type, reason, chosen_action = parse_llm_response(response)

        # Validate chosen action if "choose" mode, otherwise parse recommendations
        if "choose" in pick_type.lower():
            is_valid, action_str = validate_and_get_chosen_action(chosen_action, choices_idxs, valid_acts)
        else:
            action_str = handle_llm_recommendations(chosen_action, reason, action_values, valid_acts, tries, posttest,
                                                    valid_subjects, valid_topics, valid_causes)
            is_valid = action_str is not None

    if not is_valid:
        raise ValueError("Out of tries for action validation.")

    return chosen_action, action_str, reason, pick_type


def parse_llm_response(response):
    """Parses LLM response into pick type, reason, and action."""
    parts = separate(response)
    pick_type, reason, action = ("", "", "")
    if len(parts) == 3:
        pick_type, reason, action = parts
    elif len(parts) == 2:
        pick_type, action = parts
    elif len(parts) == 1:
        pick_type, action = "recommend", parts[0]
    pick_type = pick_type.strip()
    reason = reason.strip()
    action = action.strip()
    return pick_type, reason, parse_chosen_action(action)


def validate_and_get_chosen_action(chosen_action, choices_idxs, valid_acts):
    """Validates chosen action and returns the corresponding action string if valid."""
    if chosen_action.isnumeric():
        chosen_action = int(chosen_action)
        if 0 <= chosen_action < len(choices_idxs):
            return True, valid_acts[choices_idxs[chosen_action - 1]]
    return False, None


def handle_llm_recommendations(recs, reason, action_values, valid_acts, tries, posttest, valid_subjects, valid_topics,
                               valid_causes):
    """Handles recommendations from LLM and selects a recommended action if valid."""
    parsed_responses = []
    recs_splitted = [reason] + recs.split("\n") + re.split(r' (?=\d)', recs)
    for x in recs_splitted:
        parsed_responses.append(parse_string_to_dict(x.split(". ")[-1], valid_subjects, valid_topics, valid_causes,
                                                     replace_closest=tries > 3, posttest=posttest))

    recs_idxs = [valid_acts.index(x) for x in parsed_responses if x in valid_acts]
    if recs_idxs:
        act_probs = softmax(action_values[0][recs_idxs], temperature=0.001 if posttest else 1)
        chosen_act_idx = torch.multinomial(act_probs, num_samples=1).item()
        return valid_acts[recs_idxs[chosen_act_idx]]
    return None


def rl_llm_recommendation_interaction(recer, agent, state, valid_acts, policy, threshold, history, subject, problem,
                                      prev_exp, summarizer, valid_subjects, valid_topics, valid_causes, posttest):
    """Handles the interaction between the RL agent and LLM recommender with a veto mechanism."""
    valid_ids = agent.encode_actions(valid_acts)
    _, action_idx, action_values, _ = agent.act([state], [valid_ids], policy=policy, eval_mode=True,
                                                action_strs=valid_acts)

    tries, is_valid = 0, False
    while not is_valid and tries <= 3:
        tries += 1
        response = recer.rec(history, subject, problem, valid_subjects, valid_topics, valid_causes, posttest=posttest,
                             prev_exp=prev_exp, summarizer=summarizer)
        reason, recommended_actions = parse_recer_response(response)

        recs_idxs, valid_responses = filter_valid_recommendations(recommended_actions, valid_acts, valid_subjects,
                                                                  valid_topics, valid_causes, posttest, tries)
        if recs_idxs:
            veto = RL_veto(recs_idxs, action_values[0].detach().cpu().numpy(), threshold=threshold)
            act_probs = softmax(action_values[0][recs_idxs], temperature=0.001 if posttest else 1)
            chosen_act_idx = torch.multinomial(act_probs, num_samples=1).item()
            action_str = valid_responses[chosen_act_idx] if not veto else valid_acts[action_idx[0]]
            is_valid = True
        elif tries > 3:
            raise ValueError("Out of tries for LLM recommendation validation.")

    return action_str, reason, veto


def parse_recer_response(response):
    """Parses the response from the LLM recommender into reason and recommendations."""
    parts = separate(response)
    if len(parts) > 2:
        reason = parts[0]
        recs = "\n".join(parts[1:])
    elif len(parts) == 2:
        reason, recs = parts
    elif len(parts) == 1:
        reason, recs = "", parts[0]
    else:
        raise ValueError("Invalid response format from recommender.")

    return reason.strip(), recs.strip()


def filter_valid_recommendations(recs, valid_acts, valid_subjects, valid_topics, valid_causes, posttest, tries):
    """Filters recommendations to retain only valid actions."""
    parsed_responses = []
    recs_splitted = recs.split("\n") + re.split(r' (?=\d)', recs)
    for x in recs_splitted:
        parsed_responses.append(parse_string_to_dict(x.split(". ")[-1], valid_subjects, valid_topics, valid_causes,
                                                     replace_closest=tries > 3, posttest=posttest))

    recs_idxs, valid_responses = [], []
    for x in parsed_responses:
        if x in valid_acts:
            recs_idxs.append(valid_acts.index(x))
            valid_responses.append(valid_acts[recs_idxs[-1]])
    return recs_idxs, valid_responses


def load_summary(save_path, ep, task, sub_task, inference, mode, eval_mode, split, MODEL):
    """Loads a saved summary or initializes an empty summary if no prior data exists."""
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
