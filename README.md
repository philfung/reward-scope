# RewardScope 🎛️
Visually compare VLM-based robot reward functions on your own manipulation videos.

Reward functions you can run on your videos:
- [TOPReward](https://topreward.github.io/webpage/) — from "TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics", UW & AllenAI
- [Generative Value Learning (GVL)](https://arxiv.org/pdf/2411.04549) — from "Vision Language Models are In-Context Value Learners", Google DeepMind
- Brute Force — at each frame, sends the video up to that point to the VLM and asks for a progress score between 0.0 and 1.0

.. and easy to add more!

## Installation

### 1. Install Python packages
```
virtualenv venv
. venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Qwen backend 
Requires ~15 GB disk (model weights) and ~16 GB unified/GPU memory.
```
pip install torch torchvision transformers accelerate
```

### 3. Get an OpenAI API Key (optional)
Create an [OpenAI API key](https://platform.openai.com/api-keys).  

This is used for running GVL and running Brute Force.
 

## To run on your own videos

1. Create a mp4 video of robot manipulation (Downsize to 480p if possible because the image pixels are passed as tokens)


2. Run the script to calculate the reward functions on your video:
```
python run_rewards.py --video myvideo.mp4 --instruction "create a tower of 5 cubes" 
```
If you have an OpenAI API key, then add the flag `--openai-api-key <your key>` and the flag `--method topreward,gvl,bruteforce_vlm`

If you don't an OpenAI API key, then also add the flag `--method topreward` 

3. View the results in your browser:
```
./run_viewer.sh
```


