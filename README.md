# RewardScope
A tool for running and comparing VLM-based robot reward functions from different papers.

Current reward functions:
- [TOPReward](https://topreward.github.io/webpage/) — UW & AllenAI
- [GVL](https://arxiv.org/pdf/2411.04549) — "Vision Language Models are In-Context Value Learners", Google DeepMind
- BruteforceVLM — direct per-frame progress estimation via text generation

## Installation

### Install Python packages
```
virtualenv venv
. venv/bin/activate
pip install -r requirements.txt
```

### Install Qwen backend (local, best results)
Requires ~15 GB disk (model weights) and ~16 GB unified/GPU memory.
```
pip install torch torchvision transformers accelerate
```

## Usage
Take your video of robot manipulation and run reward functions on it:
```
python run_rewards.py --video stackcubes2_480p_qwen.mp4 --instruction "create a tower of 5 cubes"
```

Run a specific subset of methods:
```
python run_rewards.py --video robot.mp4 --instruction "Pick up the cube" --method topreward,gvl
```

Use the OpenAI backend instead of local Qwen:
```
export OPENAI_API_KEY="your-key"
python run_rewards.py --video robot.mp4 --instruction "Pick up the cube" --backend openai
```

Now view the results in your browser:
```
./run_viewer.sh
```
