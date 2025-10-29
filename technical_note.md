# Technical note — Running `Qwen/Qwen3-VL-8B-Instruct` on Vector cluster (Killarney - Compute Canada)

This note summarizes the steps and code I used to get **Qwen3-VL-8B-Instruct** working reliably on 
Vector cluster. It includes fixes for the main issues I encountered. Use the snippets below in your 
bash and python scripts to avoid the same pitfalls.

---

## 1) High-level summary of issues & solutions

* **Problem:** Hugging Face caches big model files in `~/.cache/...` and the home directory quota 
overflows (50 GB max)
  **Fix:** Redirect HF cache to a large-volume path (e.g. `/scratch/$USER/hf_cache`) and always use 
  *absolute paths*. Also use `cache_dir` argument in `from_pretrained` to force the cache location.
* **Problem:** Model generation truncated JSON outputs (incomplete JSON).
  **Fix:** Increase `max_new_tokens`, set deterministic generation params, add a sentinel in the 
  prompt (e.g. `<END_JSON>`).

---

## 2) Bash script

Add these to your job script or a wrapper script that runs on the compute node. Use **absolute** 
paths everywhere (`/scratch/...`), *not* `home/...` or `~/...` when exporting env vars.

### Recommended `VLM_detection.sh` (example)

```bash
#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=0:30:00
#SBATCH --job-name=VLM_scale_detection
#SBATCH --output=slogs/%x_%A-%a_%n-%t.out
                   # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
#SBATCH --qos=normal
#SBATCH --open-mode=append

# Load necessary modules
module load python/3.12 cuda/12.6 arrow/21.0.0 opencv/4.12.0 

# Redirect Hugging Face cache to scratch (volume with more space)
export HF_HOME=/scratch/$USER/hf_cache

# Ensure directory exists and is writable
mkdir -p "${HF_HOME}"
chmod 700 "${HF_HOME}"
```

**Notes**

* Use `/scratch/$USER/...` or another big volume on cluster nodes.
* Add the `export` lines to your `~/.bashrc` if you want the change persistent across sessions:

  ```bash
  echo 'export HF_HOME=/scratch/$USER/hf_cache' >> ~/.bashrc
  source ~/.bashrc
  ```

---

## 3) Python: reliable model loading & cache usage

Pass an explicit absolute `cache_dir` to `from_pretrained` (and to `AutoProcessor.from_pretrained`) 
to avoid ambiguity.

```python
from transformers import AutoProcessor
from qwen import Qwen3VLForConditionalGeneration  # replace with actual import if necessary
import torch

cache_dir = "/scratch/eboehly/hf_cache"  # absolute path

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    device_map="auto",         # uses accelerate heuristics to place weights
    dtype=torch.float16
)

processor = AutoProcessor.from_pretrained(
    model_id,
    cache_dir=cache_dir
)
```

---

## 4) Generation settings to avoid truncated outputs (JSON structured output use-case)

If you need to generate structured JSON outputs from the model and face truncation and incomplete 
JSON, use the following settings: 
Use deterministic generation settings, long `max_new_tokens`, and an explicit sentinel token 
(`<END_JSON>`) at the end of the prompt to encourage the model to finish.

```python
generated_ids = model.generate(
    **inputs,
    max_new_tokens=1024,              # increase if JSON can be large
    temperature=0.0,
    do_sample=False,
    eos_token_id=processor.tokenizer.eos_token_id,
    pad_token_id=processor.tokenizer.pad_token_id,
)
```

**Prompt sentinel**
Add at the end of your prompt:

```
"Always end the JSON response with the special token <END_JSON>."
```

After decoding, `split("<END_JSON>")[0]` to get trimmed output.

---