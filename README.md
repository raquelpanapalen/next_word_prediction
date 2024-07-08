# Text Generation Project

This repository contains the implementation of a text generation project using various machine learning models (LSTM, Transformer Decoder, xLSTM) for next-word prediction. Follow the instructions below to set up your environment, install dependencies, and run the project.

## Setup Instructions

### 1. Create a Python Environment

Create a Python virtual environment to manage project dependencies. You can use `venv` or `conda` for this purpose. Here is an example using `venv`:

```sh
python -m venv text-gen-env
```
Activate the virtual environment:
- On Windows:
```sh
text-gen-env\Scripts\activate
```

- On macOS and Linux:
```sh
source text-gen-env/bin/activate
```

### 2. Install Dependencies

Install the required dependencies from the `requirements.txt` file:
```sh
pip install -r requirements.txt
```

### 3. Track Training Status with Weights & Biases

If you want to keep track of the training status, you can use Weights & Biases (W&B). Follow these steps:

1. Create a W&B account: [Weights & Biases Quickstart](https://docs.wandb.ai/quickstart)
2. Create a new project named `text-generation`.
3. Log in to your W&B account from the command line:

```sh
wandb login
```

### 4. Configure Parameters

You can change the parameters for the models and training process in the `config.yaml` file. This file contains various settings that control the behavior of the training scripts.

### 5. Run the Bash Script
Finally, run the `run.sh` bash script to start the training process:
```sh
bash run.sh
```



This `README.md` provides clear and concise instructions for setting up and running the text generation project, making it easy for users to get started and track their training progress.
