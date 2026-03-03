# RewardScope 🎛️
Visually compare VLM-based robot reward functions on your own manipulation videos.

Reward functions:
- [TOPReward](https://topreward.github.io/webpage/) — from "TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics", UW & AllenAI
- [Generative Value Learning (GVL)](https://arxiv.org/pdf/2411.04549) — from "Vision Language Models are In-Context Value Learners", Google DeepMind
- Brute Force — at each frame, sends the video up to that point to the VLM and asks for a progress score between 0.0 and 1.0

.. and easy to add more!

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
1. Take a video of robot manipulation.

2. Run the reward functions on the video:
```
python run_rewards.py --video myvideo.mp4 --instruction "create a tower of 5 cubes"
```

3. View the results in your browser:
```
./run_viewer.sh
```

## Other

Run a specific subset of methods:
```
python run_rewards.py --video robot.mp4 --instruction "Pick up the cube" --method topreward,gvl
```

Use the OpenAI backend instead of local Qwen:
```
export OPENAI_API_KEY="your-key"
python run_rewards.py --video robot.mp4 --instruction "Pick up the cube" --backend openai
```

python run_rewards.py --video viewer_files/clothesfolding1.mp4 --instruction "fold the sweatshirt" --openai-api-key <>

python run_rewards.py --video viewer_files/clothesfolding1.mp4 --instruction "fold the sweatshirt by first flattening the shirt, then folding each arm horizontally across the back, then folding the lower half of the sweatshirt vertically up" --num-frames 8 --openai-api-key <>