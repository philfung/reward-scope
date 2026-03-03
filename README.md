## installation

### core (Gemini backend)
```
virtualenv venv
. venv/bin/activate
pip install -r requirements.txt
```

### Qwen backend (local, best results)
Requires ~15 GB disk (model weights) and ~16 GB unified/GPU memory.
```
pip install torch torchvision transformers accelerate
```

## usage

### Gemini backend
```
export GOOGLE_API_KEY="your-key"
python demo.py --video files/pickupcube.mp4 --instruction "Pick up the cube"
```

### OpenAI backend
```
export OPENAI_API_KEY="your-key"
python demo.py --video viewer_files/stackcubes2_480p.mp4 --instruction "Stack all 5 cubes on top of each other" --backend qwen --save-json viewer_files/stackcubes2_480p_qwen.json
```


### Qwen backend (Apple Silicon / CUDA)
```
python demo.py --video files/pickupcube.mp4 --instruction "Pick up the cube" --backend qwen
```

First run downloads Qwen2.5-VL-7B-Instruct (~15 GB) from HuggingFace.
Use `--model Qwen/Qwen2.5-VL-3B-Instruct` if memory is tight.

### options
```
--method      both|topreward|gvl   which method(s) to run (default: both)
--num-frames  N                    frames to sample (default: 10)
--model       NAME                 override model name/ID
--combined-plot                    overlay both methods on one chart
--save-plot   path.png             save plot instead of displaying
--use-chat-template                add chat template for Qwen (hurts TOPReward per paper §5.4)
```