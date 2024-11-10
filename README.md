# PharmaSimText-OS-LLMs
This is a repository including the benchmark and agents included in an under review submission to JEDM 2024.

## Setup and Installation

### Prerequisites

- Python 3.7 or higher
- `venv` for creating a virtual environment
- OpenAI API key for using LLM models

### Installation Steps

1. **Clone the Repository**

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the Requirements**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables**

   Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY='your_openai_key'
   ```
   For Windows:
   ```cmd
   set OPENAI_API_KEY=your_openai_key
   ```

### Run Instructions

To run the evaluation tasks, you can use the infer script with various arguments to customize your run.

#### Example Command

```bash
python infer.py --thresholds 0 0.2 0.4 --patients "baby" "eye" --inferences "choose" "recommend" --modes "test" --model "gpt-4-turbo" --par 2
```

### Command-line Arguments

- `--thresholds`: List of threshold values to evaluate (default: [0, 0.2, 0.4, 0.6, 0.8, 1])
- `--patients`: List of patient categories (default: ["baby", "eye", "gm", "gyno", "joint", "mother", "throat"])
- `--inferences`: List of inference types (default: ["choose", "recommend", "play", "RL_Vetos", "RL_Interacts", "LLM_Vetos", "LLM_interacts"])
- `--modes`: Evaluation modes (default: ["test"])
- `--model`: Model name (default: "gpt-4-turbo")
- `--par`: Number of parallel executions (default: 2)
- `--no-clin`: Exclude reflective inferences (use `--no-clin` to enable)
- `--best-threshold`: Use the best threshold for each inference
- `--results-path`: Path to store results (default: `./results_gpt-4-turbo/`)

### Using Weights & Biases (WandB)

This project integrates with [WandB](https://wandb.ai/) for tracking experiments. Make sure to log in to WandB using your key:
```bash
wandb login
```

### Customizing the Workflow

You can modify the evaluation configurations by editing the command-line arguments or the code in the `infer.py` script.

## Using PharmaSimText Separately

PharmaSimText is an interactive simulation environment designed for training and evaluating AI agents in a pharmacy assistant scenario. It enables agents to engage in diagnostic conversations, make decisions, and receive rewards or penalties based on their actions. The environment offers flexibility in scenario configuration, state tracking, and evaluation methods, making it suitable for agent training and research experiments.

## How to Use PharmaSimText

### Setting Up the Environment

To get started with PharmaSimText, ensure you have the required dependencies installed and access to the scenario files.

#### 1. Load Scenarios and Initialize Environment

- The `Game` class initializes the simulation environment with various parameters that control the behavior of the simulation.
- You can specify the path to scenario files using the `path` parameter.

Example:
```python
from pharmasimtext import Game

# Specify the path to scenario files
path = "./scenarios/test"
# Initialize the environment
env = Game(path=path)
```

#### 2. Configuring the Environment

- The environment supports various configurations, including scenario randomization, embedding types, evaluation methods, and training modes.
- You can customize these settings by passing arguments to the `Game` class initializer.

Example:
```python
env = Game(
    path=path,
    env_step_limit=20,
    reward_scale=1,
    penalty=0,
    emb="sum",
    hc="bq",
    embedding_dim=300,
    wording=True,
    evaluation="cause",
    random_scenarios=True
)
```

#### 3. Resetting the Environment

- Before starting a new episode, reset the environment to its initial state using the `reset()` method.
```python
initial_state, actions, hc_state = env.reset()
```

### Interacting with the Environment

#### 1. Retrieving Available Actions

- Use `get_actions(kind)` or `get_gpt_actions(kind)` to retrieve available actions of a specific type (e.g., "interaction" or "posttest").
```python
actions = env.get_actions("interaction")
```

#### 2. Taking Steps in the Environment

- Use the `step(previous_update, action, trial_num)` method to take an action, receive a reward, and update the state.
- This method returns the updated state, reward, terminal flag, available actions, and other relevant information.

Example:
```python
# Take an action
state_update, reward, terminal, actions, hc_state, traj_score = env.step(initial_state, actions[0])
```

- Continue taking actions until the scenario ends (i.e., `terminal` becomes `True`).

#### 3. Loading Scenarios

- Load specific scenarios using the `load_scenario(file_name)` method.
```python
env.load_scenario("example_scenario.json")
```

### Example Simulation Loop

```python
# Initialize and reset the environment
path = "./scenarios/test"
env = Game(path=path)
env.reset()

# Simulate episodes
for i in range(10):  # Run 10 episodes as an example
    state_update, reward, terminal, actions, hc, traj_score = env.step(
        env.get_initial_state(), env.get_actions("interaction")[0]
    )
    
    while not terminal:
        # Randomly select an action from available options
        state_update, reward, terminal, actions, hc, traj_score = env.step(
            state_update, random.choice(actions)
        )
```

### Key Features and Methods

- `reset()`: Resets the environment to its initial state, preparing it for a new episode.
- `step()`: Takes an action in the environment, updates the state, and returns relevant information.
- `load_scenario()`: Loads a specific scenario from a file.
- `get_num_of_scenarios()`: Retrieves the total number of available scenarios.
- `get_state_len()`: Returns the length of the state vector based on embedding and features used.

### Customization Options

- **Scenario Randomization**: Set `random_scenarios=True` to randomly select scenarios for each episode.
- **Evaluation Methods**: Configure the evaluation method using the `evaluation` parameter (e.g., "cause", "binary").
- **Embedding Types**: Use different embedding strategies such as "sum", "avg", "max", "lstm".
- **Training Modes**: Specify `training=True` for training-specific configurations.

---

PharmaSimText provides a flexible framework for creating and running simulations tailored to AI-driven diagnostic and decision-making tasks in a healthcare context. It is highly customizable, allowing users to define the interaction flow, state representation, and evaluation criteria to meet their specific needs.