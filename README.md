# Implementation of: _Sutton & Barto (2020). Reinforcement Learning: An Introduction (Second edition)_

This repository is a complete, self-contained implementation of the major algorithms, figures, and experiments from: 
_Sutton & Barto (2020). Reinforcement Learning: An Introduction (2nd Edition, 2020)_

The goal is to build an **well-documented, and reproducible** implementation of key algorithms for self-studying 
reinforcement learning.

---

## 📌 Project Goals

- ✅ Reproduce all key **figures**, **algorithms**, and **results**
- ✅ Implement everything from scratch using **Python**
- ✅ Keep code **modular and readable** for easy experimentation

---

## 🛠️ Setup

1. Clone this repository

```
git clone git@github.com:i-wagner/sutton_and_barto_reinforcement_learning_2nd.git
```

2. Install [```uv```](https://github.com/astral-sh/uv) (if you haven’t yet)

With the standalone installer:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

With pip:
```
pip install uv
```

3. Run scripts

```uv``` allows running scripts without installing dependencies first, using ```uv run my_script.py```. In this case, 
```uv```, first, uses ```pyproject.toml``` to automatically resolve and install dependencies into an ephemeral virtual 
environment. The script is then run from this virtual environment, without polluting the project folder with ```.venv```
and without touching the global python environment. 

For example, to simulate a ten-armed bandit with different epsilon values plot the reward distributions of arms, run:

````
cd sutton_and_barto_reinforcement_learning_2nd
uv run ./rl_ch2_bandits/main.py
````

4. Optional: Install dependencies with uv

If you do want to install dependencies locally, run:

```
cd sutton_and_barto_reinforcement_learning_2nd
uv venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
uv pip install .
```

---

## 🗂️ Repository structure

The repository structure and key scripts are highlighted below:

```
sutton_and_barto_reinforcement_learning_2nd/
├─ pyproject.toml # Project configuration
├─ uv.lock # Lockfile with exact information about project dependencies
├───rl_ch2_bandits/
│    Scripts to simualte a ten-armed bandit
│    ├─ bandit.py
│    ├─ main.py
│    ├─ plotting.py
```
