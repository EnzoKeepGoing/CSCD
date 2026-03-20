
# Environment Setup

## Conda Environment

To create a conda environment, run:

```bash
conda create --name=env_name python=3.8.17
conda activate env_name
```

## Project Environment

To set up the project environment:

```bash
cd xxx/PYCD
pip install -r requirements.txt
pip install -e .  # Optional: Install the project in development mode
```

---

# Dataset Preprocessing

Dataset download link 🔗: 

```bash
cd examples
python data_preprocess.py --dataset_name=Peiyou --min_seq_len=15 --split_mode=1 --time_info=0
```

### Parameter Explanation:

**Multiple Exercise Strategy for Problems**

> Current CD models only retain the first interaction record for multiple attempts by students on the same problem. Here, we consider three cases:

- `split_mode=1` (default): For multiple interactions with a problem, retain the first interaction record.
- `split_mode=2`: For multiple interactions with a problem, calculate the average accuracy as the label.
- `split_mode=3`: For multiple interactions with a problem, calculate the average accuracy + the global average accuracy of the problem weighted as the label.

**Time Strategy**

> Current CD models do not consider the time information of the student’s answering sequence. We introduce the `time_info` parameter to divide new students according to weeks. If the time interval between two interactions exceeds the `time_info` value, it will be considered as a new student:

- `time_info=0` (default): Do not consider time information.
- `time_info=1`: Split based on a 1-week time interval.
- `time_info=2`: Split based on a 2-week time interval.
- `time_info=4`: Split based on a 4-week time interval.

*Note: In both modes, any student interaction records with a sequence length shorter than `min_seq_len=15` will be deleted.*

---

# Model Training and Evaluation

```bash
python example_neuralcdm.py --dataset assist2009
```

Once training is complete, the model will be stored in the `xx/PYCD/examples/model_save` directory.

---

# Wandb Hyperparameter Tuning

This project supports hyperparameter tuning and experiment tracking using Weights & Biases (wandb). Below are the steps for using wandb to fine-tune the model.

## Pre-Setup

1. Install wandb and create a wandb account to obtain your API key:
   ```bash
   pip install wandb
   ```

2. Configure the API key:
   - Create a configuration file `configs/wandb.json`:
   ```json
   {
      "uid": "username",
      "api_key": "wandb API key"
   }
   ```

## Sweep Configuration File

Create a model-specific YAML configuration file (e.g., `neuralcdm.yaml`) in the `seedwandb` directory. This configuration file will explore different hyperparameter combinations, including hidden layer dimensions, dropout rates, learning rates, and more.

## Automated Hyperparameter Tuning Process

We provide an automated script `start_sweep.sh` to simplify the wandb hyperparameter tuning process:

```bash
#!/bin/bash
# Set parameters
PROJECT_NAME="pycd-neuralcdm"  # wandb project name
MODEL_NAME="neuralcdm"         # Model name
DATASET_NAME="assist2009"      # Dataset name
FOLDS="0,1,2,3,4"              # Cross-validation folds
GPU_IDS="0,1,2,3"              # GPU IDs to use
AGENT_SCRIPT="run_agents.sh"   # Script to execute agents
LOG_FILE="wandb_agents.log"    # Log file

# Execute the script to start sweep
bash start_sweep.sh
```

Running this script will:
1. Generate fold-specific sweep configurations.
2. Create a wandb sweep.
3. Generate a `run_agents.sh` script to run the agents.

## Start Sweep Agent

After the sweep is created, execute the generated `run_agents.sh` script to start the agents:

```bash
bash run_agents.sh
```

This will start all sweep agents in the background, and the output will be redirected to the `wandb_agents.log` file. Each agent will run on the specified GPU, automatically performing the grid search as defined in the sweep.

## View Experiment Results

All experiment results will be automatically uploaded to the wandb platform. On the wandb website, you can:
- Compare the performance of different hyperparameter combinations.
- Visualize the training process.
- Analyze parameter importance.
- Select the best model configuration.

## Custom Configuration

You can modify the following parameters as needed:
- Model name and dataset.
- Hyperparameter search space.
- GPU allocation.

For example, to create a sweep for the KaNCD model, simply modify the `MODEL_NAME` and the corresponding YAML configuration file.

---

# CSCD-CD Replication

- `python example_CSCD-CD.py --dataset Peiyou`

---

# CSCD Model Training Parameters for 5 Folds

This document outlines the parameters used for training the CSCD model across 5 different folds.

## Model Training Parameters Table

| Fold | Embedding Dimension (`emb_dim`) | Learning Rate (`lr`) | Batch Size (`batch_size`) | Epochs | Seed (`seed`) | Use Wandb (`use_wandb`) | Save Directory (`save_dir`) | A Matrix CSV File (`a_csv_name`) |
|------|-------------------------------|----------------------|---------------------------|--------|---------------|------------------------|-----------------------------|----------------------------------|
| 0    | 30                            | 0.0019               | 1024                      | 60     | 3407          | 0                      | Default                    | A_matrix.csv                    |
| 1    | 130                           | 0.0024               | 1028                      | 30     | 3407          | 1                      | Default                    | A_matrix.csv                    |
| 2    | 30                            | 0.002                | 1024                      | 60     | 3407          | 0                      | Default                    | A_matrix.csv                    |
| 3    | 30                            | 0.002                | 1024                      | 30     | 3407          | 0                      | Default                    | A_matrix.csv                    |
| 4    | 30                            | 0.0024               | 1024                      | 50     | 3407          | 1                      | Default                    | A_matrix.csv                    |

## Summary of experimental results (average & 5-flod)

| Model | AUC average | AUC 5-flod | ACC average | ACC 5-flod | RMSE average | RMSE 5-flod |
|------|---------:|---------:|---------:|---------:|----------:|----------:|
| THINK | 0.74422 | 0.000813388 | 0.73106 | 0.00285979 | 0.4208 | 0.000989949 |
| Math_g4_g5 | 0.73046 | 0.000747262 | 0.70884 | 0.003510328 | 0.43706 | 0.000508331 |
| Math_g7 | 0.75114 | 0.001741953 | 0.7192 | 0.00247548 | 0.43082 | 0.000636867 |
| Peiyou | 0.81448 | 0.000271293 | 0.82448 | 9.79796E-05 | 0.35356 | 0.0005 |

## Explanation of Parameters

- **Fold**: The fold number used for cross-validation.
- **Embedding Dimension (`emb_dim`)**: The size of the embedding layer.
- **Learning Rate (`lr`)**: The learning rate used for training the model.
- **Batch Size (`batch_size`)**: The number of samples per batch during training.
- **Epochs**: The number of times the entire dataset is passed through the model.
- **Seed (`seed`)**: The random seed to ensure reproducibility of the results.
- **Use Wandb (`use_wandb`)**: Flag indicating whether to use Weights & Biases for tracking (`1` for yes, `0` for no).
- **Save Directory (`save_dir`)**: Directory where the model is saved. If `None`, the default path is used.
- **A Matrix CSV File (`a_csv_name`)**: The filename of the A matrix CSV used for training.

## Notes

- The `a_dim` parameter is fixed to `11` for all configurations.
- The `a_csv_name` parameter is set to `A_matrix.csv`, which is the name of the actual A file used in training.

---
