import json
import os
import random
import numpy as np


def add_onehot(l):
    """
    Increments a binary one-hot encoded list by one in its sequence.
    Shifts the '1' in the one-hot encoding to the next position, representing
    the next integer in the sequence. For example, [0, 1, 0, 0] becomes [0, 0, 1, 0].

    Parameters:
    l (list or array): A one-hot encoded list or array where only one position is set to 1.

    Returns:
    np.array: A one-hot encoded array representing the next number in the sequence.
    """
    # Convert input to list if it's not already
    l = list(l)

    # Find the current index of '1' or set to -1 if '1' is not found
    i = l.index(1) if 1 in l else -1

    # Set the next position to 1 to represent the increment
    l[i + 1] = 1

    # Reset the original '1' position to 0 if it exists
    if i > -1:
        l[i] = 0

    # Return the incremented one-hot list as a numpy array
    return np.array(l)


class Game:
    """
    Game environment for the Pharmacy Assistant simulation.
    Handles loading scenarios, managing states, and processing agent actions.
    """

    def __init__(self, name="Pharmasim", path=r".\scenarios", env_step_limit=20,
                 wrong_answer=False, reward_scale=1, penalty=0, emb="sum", hc="bq", embedding_dim=300,
                 wording=True, evaluation="cause", scenario_name="",
                 random_scenarios=False, reduced=True, training=False) -> None:
        """
        Initializes the Game environment with configuration settings and loads scenario files.

        Parameters:
        - name (str): Name of the game environment.
        - path (str): Path to the directory containing scenario files.
        - env_step_limit (int): Maximum number of steps allowed per episode.
        - wrong_answer (bool): Indicates if wrong diagnoses should be penalized.
        - reward_scale (float): Scaling factor for rewards.
        - penalty (float): Penalty value for incorrect actions.
        - emb (str): Type of embedding to use (e.g., "sum", "avg", "max", "lstm").
        - hc (str): Type of hand-crafted state features (e.g., "bq"(binary indicator of key questions being asked) only available in the baby scenario).
        - embedding_dim (int): Dimensionality of the sentence embedding vector.
        - wording (bool): Evaluate with different wording options.
        - evaluation (str): Type of diagnosis evaluation method (e.g., "cause"(choose the cause from a presented list), "binary"(asking yes/no question for each scenario)).
        - scenario_name (str): Whether to use scenario name.
        - random_scenarios (bool): If True, scenarios are chosen randomly.
        - reduced (bool): Determines if a reduced version of the embeddings are used.
        - training (bool): Specifies if the environment is in training mode.
        """

        # Basic configuration and settings
        self.reduced = "reduced" if reduced else "not_reduced"  # Reduced environment setting
        self.evaluation = evaluation  # Evaluation method (e.g., "cause")
        self.wording = wording  # Controls question wording
        self.path = path  # Directory containing scenario files
        self.training = training  # Training mode indicator
        self.max_steps_per_episode = env_step_limit  # Limit of steps per episode
        self.wrong_answer = wrong_answer  # Flag for penalizing wrong answers
        self.penalty = penalty  # Penalty applied for incorrect actions
        self.reward_scale = reward_scale  # Scale factor for rewards
        self.emb = emb  # Embedding method
        self.hc = hc  # Hard-coded state features setting
        self.embedding_dim = embedding_dim  # Dimension for embeddings
        self.scenario_name = scenario_name  #Whether to use scenario name.
        self.random_scenarios = random_scenarios  # Flag for random scenario selection
        self.single_scenario = False  # Track if a single scenario is in use

        # Loading scenarios and tracking available files
        self.subjects = list(set(
            [f.split('_')[0] for f in os.listdir(self.path) if f.endswith('.json')]
        ))  # Unique subjects identified from scenario files

        # Dictionary mapping subjects to the number of scenarios available
        self.num_of_scenarios = dict(
            zip(self.subjects, [sum([f.split('_')[0] == s for f in os.listdir(self.path)]) for s in self.subjects])
        )

        # List of all available scenario files in the specified path
        self.available_scenarios = [f for f in os.listdir(self.path) if f.endswith('.json')]
        self.total_num = len(self.available_scenarios)  # Total number of scenario files

        # Ensure scenarios exist; if not, raise an error
        if self.total_num == 0:
            raise FileNotFoundError("No scenarios found!")

        # Initialize attributes for scenario state tracking
        self.trace = []  # To track actions within an episode
        self.scenario = None  # Current scenario loaded
        self.hc_state = dict()  # State representation for hard-coded features
        self.number_of_episodes = 0  # Episode counter
        self.num_of_trials = 100  # Default number of trials per scenario
        self.question_answers = {}  # Stores question-answer mappings for scenarios
        self.llm = False  # Indicator for using LLM-based actions

        # Display total number of scenarios loaded
        print(f"Total number of scenarios in {self.path}: ", self.total_num)

    def get_num_of_scenarios(self):
        """
        Returns the total number of scenario files available in the environment.

        Returns:
        int: Total count of available scenarios.
        """
        return self.total_num

    def increase_episodes(self):
        """
        Increments the episode counter by one.
        This is used to keep track of how many episodes have been completed.
        """
        self.number_of_episodes += 1

    def get_scenario_files(self):
        """
        Retrieves the list of available scenario files.

        Returns:
        list: A list of filenames for all scenarios in the specified path.
        """
        return self.available_scenarios

    def load_scenario(self, file_name, num_of_trials=100, gpt=False):
        """
        Loads a specific scenario file and sets up trials and question-answer mappings.

        Parameters:
        - file_name (str): The name of the scenario file to load.
        - num_of_trials (int): Number of trials for each question (default is 100).
        - gpt (bool): If True, indicates that GPT-based actions are used.
        """
        # Set single scenario mode and GPT flag
        self.single_scenario = True
        self.llm = gpt

        # Load the scenario JSON data from the specified file
        self.scenario = json.load(open(os.path.join(self.path, file_name)))
        self.num_of_trials = num_of_trials  # Set the number of trials for the scenario

        # Configure question-answer structure for trials
        for k1 in self.scenario["question_answers"]:
            self.question_answers[k1] = {}
            for k2 in self.scenario["question_answers"][k1]:
                # Calculate repeat count based on trials and available answers
                b = len(self.scenario["question_answers"][k1][k2])

                # If fewer answers than trials, repeat answers to fill trials
                if b <= num_of_trials:
                    self.scenario["question_answers"][k1][k2] = np.repeat(
                        self.scenario["question_answers"][k1][k2], num_of_trials // b
                    ).tolist()
                    if len(self.scenario["question_answers"][k1][k2]) != num_of_trials:
                        # If there is a shortfall, add random choices to reach the required number of trials
                        self.scenario["question_answers"][k1][k2] += np.random.choice(
                            self.scenario["question_answers"][k1][k2],
                            num_of_trials - len(self.scenario["question_answers"][k1][k2])
                        ).tolist()
                else:
                    # If more answers than trials, sample randomly from available answers
                    self.scenario["question_answers"][k1][k2] = np.random.choice(
                        self.scenario["question_answers"][k1][k2], size=num_of_trials
                    ).tolist()

                # Confirm that question-answer pairs match the number of trials
                assert len(self.scenario["question_answers"][k1][k2]) == num_of_trials

                # Randomize question-answer ordering for added variation in trials
                self.question_answers[k1][k2] = random.sample(
                    self.scenario["question_answers"][k1][k2], len(self.scenario["question_answers"][k1][k2])
                )

        # Initialize hard-coded state variables for the scenario
        self.hc_state["posttest_indicator"] = np.zeros(1)  # Posttest state indicator
        self.hc_state["binary_qs"] = np.zeros(len(self.scenario["relevant_actions"]))  # Binary question tracking

        # Initialize statistics tracking by character
        l = []
        for c in self.scenario["characters"]:
            if c == "others":
                l.append(np.zeros(self.max_steps_per_episode))
            elif c in self.scenario["question_answers"]:
                l.append(np.zeros(len(self.scenario["question_answers"][c].keys())))
            else:
                raise ValueError("Character is not valid")

        # Assign statistics tracking to each character in the scenario
        self.hc_state["statistics"] = dict(zip(self.scenario["characters"], l))
        self.hc_state["statistics"]["interactions"] = np.zeros(self.max_steps_per_episode)
        self.hc_state["statistics"]["unq_interactions"] = np.zeros(self.max_steps_per_episode)

        # Return initial state, available actions, and initialized hc_state
        actions = self.get_gpt_actions("interaction") if self.llm else self.get_actions("interaction")
        return self.get_initial_state(), actions, self.hc_state

    def scenario_step(self, previous_update, action, trial_num):
        """
        Processes a single step in the scenario based on the action taken and trial number.

        Parameters:
        - previous_update: The prior state or update before the current action.
        - action (dict): The action taken, which influences the state update.
        - trial_num (int): The trial number within the episode, used to select question-answer pairs.

        Returns:
        tuple: Updated state, reward, terminal status, available actions, and trajectory score.
        """
        # Ensure trial number is within the allowed number of trials
        assert trial_num < self.num_of_trials

        # Update question-answer pairs for the current trial
        for k1 in self.scenario["question_answers"]:
            for k2 in self.scenario["question_answers"][k1]:
                # Set current question-answer based on trial_num
                self.scenario["question_answers"][k1][k2] = self.question_answers[k1][k2][trial_num]

        # Special handling for GPT-based interactions
        if self.llm:
            # Detect specific action requests for diagnosis or solution suggestions
            if "i want" in action.lower():
                if "diagnosis" in action.lower() or "solution" in action.lower():
                    action = {
                        "type": "interaction",
                        "part": "solution",
                        "detail": "",
                        "sentence": "i want to suggest a solution."
                    }
                elif "know" in action.lower() or "ask" in action.lower():
                    subject, topic = None, None
                    for s in self.scenario["subjects"]:
                        if s.lower() in action.lower():
                            subject = s
                    for t in self.scenario["topics"]:
                        if t.lower() in action.lower():
                            topic = t
                    if subject and topic:
                        action = {
                            "type": "interaction",
                            "part": "discuss",
                            "detail": f"{subject},{topic}",
                            "sentence": f"i want to know about the {subject} 's {topic}."
                        }
                    else:
                        raise ValueError("Not Implemented")
                else:
                    raise ValueError("Not Implemented")

            # Handle 'i think' statements for cause-based actions
            elif "i think" in action.lower():
                for c in self.scenario["causes"]:
                    if c.lower() in action.lower():
                        action = {
                            "type": "posttest",
                            "part": "",
                            "detail": "",
                            "sentence": c
                        }
                        print(action)
                        break
                if isinstance(action, str):
                    raise ValueError("Not Implemented")
            else:
                raise ValueError("Not Implemented")

        # Update the state, calculate reward, and determine if the scenario has ended
        state_update, reward, terminal, actions, hc, traj_score = self.step(previous_update, action)

        # If using GPT, retrieve GPT-specific actions based on the updated state
        if self.llm:
            actions = self.get_gpt_actions(state_update[0])

        # Return updated state, reward, terminal status, actions, hc state, and trajectory score
        return state_update, reward, terminal, actions, hc, traj_score

    def reset(self):
        """
        Resets the game environment, selecting a scenario and initializing state variables.

        Returns:
        tuple: The initial state, available actions, and hc_state dictionary.
        """
        # Reset action trace for the new episode
        self.trace = []

        # Check if single scenario mode is enabled
        if not self.single_scenario:
            # Select a scenario based on whether random selection is enabled
            if self.random_scenarios:
                s = np.random.choice(self.available_scenarios)  # Randomly choose a scenario
            else:
                # Cycle through scenarios sequentially based on the episode count
                i = self.number_of_episodes % self.total_num
                s = self.available_scenarios[i]

            # Load the selected scenario from file
            self.scenario = json.load(open(os.path.join(self.path, s)))

            # Configure question wording based on training or fixed mode
            for k1 in self.scenario["question_answers"]:
                for k2 in self.scenario["question_answers"][k1]:
                    self.scenario["question_answers"][k1][k2] = (
                        np.random.choice(self.scenario["question_answers"][k1][k2])
                        if self.training else self.scenario["question_answers"][k1][k2][0]
                    )

        # Initialize hard-coded (hc) state features for tracking scenario progress
        self.hc_state["posttest_indicator"] = np.zeros(1)  # Indicator for posttest state
        self.hc_state["binary_qs"] = np.zeros(len(self.scenario["relevant_actions"]))  # Track binary question states

        # Initialize statistics tracking for each character in the scenario
        l = []
        for c in self.scenario["characters"]:
            if c == "others":
                l.append(np.zeros(self.max_steps_per_episode))  # 'Others' character tracking
            elif c in self.scenario["question_answers"]:
                l.append(np.zeros(len(self.scenario["question_answers"][c].keys())))
            else:
                raise ValueError("Character is not valid")  # Ensures valid characters

        # Store character-based statistics in hc_state
        self.hc_state["statistics"] = dict(zip(self.scenario["characters"], l))
        self.hc_state["statistics"]["interactions"] = np.zeros(self.max_steps_per_episode)
        self.hc_state["statistics"]["unq_interactions"] = np.zeros(self.max_steps_per_episode)

        # Retrieve initial actions based on GPT or regular mode
        actions = self.get_gpt_actions("interaction") if self.llm else self.get_actions("interaction")

        # Return the initial state, initial actions, and hc_state dictionary
        return self.get_initial_state(), actions, self.hc_state

    def get_initial_state(self):
        """
        Retrieves the initial state of the current scenario.

        Returns:
        dict: The initial state of the scenario.
        """
        return self.scenario["initial_state"]

    def get_state_len(self):
        """
        Calculates the length of the state vector based on embedding type and additional features.

        Returns:
        int: The length of the state vector.
        """
        # Reset the environment to ensure all settings and states are initialized
        self.reset()
        l = 0  # Initialize the state length counter

        # Determine the embedding type and add corresponding dimensions
        if isinstance(self.emb, str):
            if self.emb in ["sum", "avg", "max", "lstm"]:
                l += self.embedding_dim  # Use embedding dimension if embedding type is supported
            else:
                raise NotImplementedError("Embedding type not implemented")
        else:
            # Handle cases where self.hc_state is a string (e.g., hard-coded state features)
            if isinstance(self.hc_state, str):
                if self.hc_state == "bq":
                    l += len(self.hc_state["binary_qs"])  # Binary questions length
                # Other feature types can be implemented if needed, as shown in commented options
                elif self.hc_state == "kw":
                    l += len(self.hc_state["keywords"])
                elif self.hc_state == "both":
                    l += len(self.hc_state["binary_qs"])
                    l += len(self.hc_state["keywords"])
                else:
                    raise NotImplementedError("State feature not implemented")
            else:
                print("At least one feature should be added to the state vector length!")

        return l

    def get_actions(self, kind):
        """
        Retrieves available actions of a specific type in the current scenario.

        Parameters:
        - kind (str): The type of action to retrieve ("interaction" or "posttest").

        Returns:
        list: A list of actions of the specified type.
        """
        return self.scenario["actions"][kind]

    def get_gpt_actions(self, kind):
        """
        Retrieves GPT-specific actions for a specified type, along with related subjects and topics.

        Parameters:
        - kind (str): The type of action to retrieve (e.g., "interaction").

        Returns:
        tuple: A tuple containing valid actions and, depending on the type, subjects and topics or causes.
        """
        if kind == "interaction":
            # For interactions, retrieve subjects and topics alongside valid actions
            subjects = self.scenario["subjects"]
            topics = self.scenario["topics"]
            valid_actions = self.scenario["valid_actions"][kind]
            return valid_actions, subjects, topics
        else:
            # For other types (e.g., posttest), retrieve causes and valid actions
            causes = self.scenario["causes"]
            valid_actions = self.scenario["valid_actions"][kind]
            return valid_actions, causes

    def get_reward(self, state, action):
        """
        Calculates reward and trajectory score based on the current state and action.

        Parameters:
        - state (tuple): Current state, typically containing information about the scenario's phase.
        - action (dict): Action taken by the agent.

        Returns:
        tuple: Reward value (float) and trajectory score (float).
        """
        reward = 0  # Initialize reward
        traj_score = 0  # Initialize trajectory score

        # Check if in posttest phase
        if state[0] == "posttest":
            # Calculate trajectory score as the proportion of relevant actions in trace
            traj_score = sum(
                a in self.trace for a in self.scenario["present_actions"]
            ) / len(self.scenario["present_actions"])

            # Handle different evaluation types
            if self.evaluation == "binary":
                # Binary evaluation for correctness (yes/no)
                if state[1] != "done":
                    sentence = state[1].split(".")[-1]
                    i = self.scenario["posttest_qs"].index(sentence)
                    if action == self.scenario["posttest_as"][i]:
                        reward += self.reward_scale * (self.scenario["posttest_as"][i]["sentence"] == "yes")
                    else:
                        reward += self.reward_scale * (-1 if self.scenario["posttest_as"][i]["sentence"] == "no" else 0)

            elif self.evaluation == "cause":
                # Cause-based evaluation
                if state[1] != "done":
                    sentence = state[1].split(".")[-1]
                    i = self.scenario["posttest_qs"].index(sentence)
                    reward += self.reward_scale if action == self.scenario["posttest_as"][i] else \
                        self.reward_scale * (-1 if self.wrong_answer else 0)

            elif self.evaluation == "rel":
                # Relationship-based evaluation
                if state[1] != "done":
                    traj_score = sum(
                        a in self.trace for a in self.scenario["present_actions"]
                    ) / len(self.scenario["present_actions"])
                    reward += traj_score

            elif self.evaluation in ["relcause1", "relcause2", "relcause3"]:
                # Combined relevance and cause-based evaluations with varying reward scales
                if state[1] != "done":
                    traj_score = sum(
                        a in self.trace for a in self.scenario["present_actions"]
                    ) / len(self.scenario["present_actions"])
                    sentence = state[1].split(".")[-1]
                    i = self.scenario["posttest_qs"].index(sentence)
                    if action == self.scenario["posttest_as"][i]:
                        reward += traj_score * self.reward_scale if "relcause2" in self.evaluation else traj_score + self.reward_scale
                    else:
                        reward += self.reward_scale * (-1 if self.wrong_answer else 0) + traj_score * (
                            1 if "relcause3" in self.evaluation else 0)

            else:
                # Default: reward based on binary evaluation of answer
                if state[1] != "done":
                    sentence = state[1].split(".")[-1]
                    i = self.scenario["posttest_qs"].index(sentence)
                    item = self.scenario["posttest_as"][i][1][0]
                    c = (2 * (item in self.trace) - 1) if action["sentence"] == "yes" else 1
                    reward += self.reward_scale * c if action == self.scenario["posttest_as"][i][0] else \
                        self.reward_scale * (-1 if self.wrong_answer else 0)

        else:
            # Apply penalty if not in posttest phase
            reward += self.penalty

        return reward, traj_score

    def step(self, previous_update, action):
        """
        Executes a step in the environment, updates state, and returns step results.

        Parameters:
        - previous_update: The prior state or update before the current action.
        - action (dict): Action taken by the agent.

        Returns:
        tuple: Updated state, reward, terminal status, actions, hc_state, and trajectory score.
        """
        traj_score = 0  # Initialize trajectory score
        state = previous_update  # Set current state to previous state update
        self.trace.append(action)  # Log action in trace

        # Check if max steps reached or if action type is not 'interaction'
        if len(self.trace) < self.max_steps_per_episode or action["type"] != "interaction":
            # Calculate reward, trajectory score, and check for terminal state
            reward, traj_score = self.get_reward(state, action)
            terminal = 0  # Default terminal flag

            # Process action if it matches the current state type
            if action["type"] == state[0]:
                part = action["part"]
                detail = action["detail"]

                if action["type"] == "interaction":
                    # Update binary questions and statistics for interaction actions
                    if action in self.scenario["relevant_actions"]:
                        ind = self.scenario["relevant_actions"].index(action)
                        self.hc_state["binary_qs"][ind] = 1  # Mark binary question state

                    # Update interaction counters
                    self.hc_state["statistics"]["interactions"] = add_onehot(
                        self.hc_state["statistics"]["interactions"]
                    )
                    if action not in self.trace[:-1]:
                        self.hc_state["statistics"]["unq_interactions"] = add_onehot(
                            self.hc_state["statistics"]["unq_interactions"]
                        )

                    # Generate updated state based on action details
                    state_update = ".".join([part, action["sentence"]])
                    if part == "solution":
                        state_update += self.scenario["posttest_qs"][0]
                        state_update = ("posttest", state_update)
                        self.hc_state["posttest_indicator"][0] = 1  # Set posttest indicator
                    elif part == "discuss":
                        # Parse subject and topic from action detail
                        subject, topic = detail.split(",")
                        if "others" in self.scenario["question_answers"]:
                            subject = subject if subject in self.scenario["question_answers"] else "others"
                            topic = topic if topic in self.scenario["question_answers"][subject] else "all"
                        else:
                            if subject not in self.scenario["question_answers"]:
                                raise ValueError("Character not recognized")

                        # Append the topic to state_update
                        state_update += self.scenario["question_answers"][subject][topic]

                        # Update statistics if action not repeated
                        subject = subject if subject in self.hc_state["statistics"] else "others"
                        if action not in self.trace[:-1]:
                            self.hc_state["statistics"][subject] = add_onehot(self.hc_state["statistics"][subject])
                    elif part == "document":
                        state_update += self.scenario["document"][detail]
                    else:
                        raise NotImplementedError("Action part not implemented")

                    if isinstance(state_update, str):
                        state_update = ("interaction", state_update)

                # For posttest actions, move to the next question or mark as done
                else:
                    sentence = state[1].split(".")[-1]
                    i = self.scenario["posttest_qs"].index(sentence)
                    len_qs = len(self.scenario["posttest_qs"])
                    if i < len_qs - 1:
                        state_update = ("posttest", self.scenario["posttest_qs"][i + 1])
                    else:
                        state_update = ("posttest", "done")
                        terminal = 1  # End scenario when all posttest questions are complete

            # Handle actions that don't match the expected state type
            else:
                state_update = (state[0], "you can not do this  try another action")
                terminal = 0  # No termination, action invalid

            # Fetch available actions for the next state
            actions = self.get_actions(state_update[0])

        # If max steps are reached, transition to solution suggestion and posttest phase
        else:
            state_update = "solution.i want to suggest a solution.." + self.scenario["posttest_qs"][0]
            self.hc_state["posttest_indicator"][0] = 1
            state_update = ("posttest", state_update)
            actions = self.get_actions(state_update[0])
            reward = 0
            terminal = 0

        # Limit further actions if at max steps and in interaction phase
        if len(self.trace) + 1 == self.max_steps_per_episode and action["type"] == "interaction":
            actions = [{
                "type": "interaction",
                "part": "solution",
                "detail": "",
                "sentence": "i want to suggest a solution."
            }]

        # Return updated state, reward, terminal flag, next actions, hc_state, and trajectory score
        return state_update, reward, terminal, actions, self.hc_state, traj_score


def main():
    """
    Main function to initialize the Game environment and run a simulation.

    Sets up a game environment, resets it to its initial state, and simulates
    multiple episodes by taking actions and processing state updates.
    """
    path = "./scenarios/test"  # Path to the directory containing scenario files
    env = Game(path=path)  # Initialize the Game environment with the specified path

    # Reset the environment to prepare it for a new episode
    env.reset()

    # Print the length of the state vector to confirm environment configuration
    print(env.get_state_len())

    # Simulate a large number of episodes (up to 10 million steps in this example)
    for i in range(10):
        # Begin each episode with the initial state and first action
        state_update, reward, terminal, actions, hc, traj_score = env.step(
            env.get_initial_state(), env.get_actions("interaction")[0]
        )

        # Print the episode number to track progress
        print(i)

        # Continue stepping through actions until the episode terminates
        while not terminal:
            # Take a random action from the available actions and update the state
            state_update, reward, terminal, actions, hc, traj_score = env.step(
                state_update, np.random.choice(actions)
            )


if __name__ == "__main__":
    main()
