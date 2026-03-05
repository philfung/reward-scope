# RewardScope 🎛️ #
## A web tool to compare popular robot reward functions on your own manipulation videos. ##
<img src="screenshots/example.gif">


Reward functions you can run on your videos:
- **[TOPReward](https://topreward.github.io/webpage/)** — from *TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics* by [@jcoleharrison](https://github.com/jcoleharrison) , [@chinsengi](https://github.com/chinsengi) , UW
- **[Robometer](https://robometer.github.io/)** - from *Robometer: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons* by [@ygtkorkmaz](https://github.com/ygtkorkmaz) , [@aliang8](https://github.com/aliang8) , USC
- **[Generative Value Learning (GVL)](https://arxiv.org/pdf/2411.04549)** — from *Vision Language Models are In-Context Value Learners*, Google DeepMind
- **Brute Force** — at each frame, sends the video up to that point to the VLM and asks for a progress score between 0.0 and 1.0

...and easy to add more!

# How to run on your own videos #

1. Create an MP4 video of robot manipulation.
    (Downsize it to 480p if possible, as image pixels are passed as tokens)

2. Run the steps in the *Installation* section. 

3. 

* **Run `topreward`, `gvl`, and/or `bruteforce_vlm`** 

    Run the script to calculate reward functions on your video:
    ```
    python run_rewards.py --video myvideo.mp4 --instruction "create a tower of 5 cubes"
    ```

   If you have an OpenAI API key, add the flags `--openai-api-key <your key>` and `--method topreward,gvl,bruteforce_vlm`.

   If you don't have an OpenAI API key, add the flag `--method topreward`.

   This will probably take ~ 10 - 20 minutes, depending on your computer.

* **Run `robometer`**

     Compute the robometer reward on the same video file and populate this github repo with the results by following the instructions [here](https://github.com/philfung/robometer?tab=readme-ov-file#usage).


4. View the results in your browser:
```
./run_viewer.sh
```

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
This is used to run TOPReward.
### 3. Get an OpenAI API key (optional)
Create an [OpenAI API key](https://platform.openai.com/api-keys).

This is used for running GVL and Brute Force.
