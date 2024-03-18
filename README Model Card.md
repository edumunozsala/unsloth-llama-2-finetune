---
tags:
- generated_from_trainer
- code
- coding
- llama-2
model-index:
- name: Llama-2-7b-python-coder
  results: []
license: apache-2.0
language:
- code
datasets:
- iamtarun/python_code_instructions_18k_alpaca
pipeline_tag: text-generation
---


# LlaMa 2 7B Python Coder using Unsloth 👩‍💻 

**LlaMa-2 7b** fine-tuned on the **python_code_instructions_18k_alpaca Code instructions dataset** by using the library [Unsloth](https://github.com/unslothai/unsloth).


## Pretrained description

[Llama-2](https://huggingface.co/meta-llama/Llama-2-7b)

Meta developed and publicly released the Llama 2 family of large language models (LLMs), a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters.

Model Architecture Llama 2 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align to human preferences for helpfulness and safety

## Training data

[python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca)

The dataset contains problem descriptions and code in python language. This dataset is taken from sahil2801/code_instructions_120k, which adds a prompt column in alpaca style.

### Training hyperparameters

**SFTTrainer arguments**
```py
# Model Parameters
max_seq_length = 2048
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# LoRA Parameters
r = 16
target_modules = ["gate_proj", "up_proj", "down_proj"]
#target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
lora_alpha = 16

# Training parameters
learning_rate = 2e-4
weight_decay = 0.01
#Evaluation
evaluation_strategy="no"
eval_steps= 50

# if training in epochs
#num_train_epochs=2
#save_strategy="epoch"

# if training in steps
max_steps = 1500
save_strategy="steps"
save_steps=500

logging_steps=100
warmup_steps = 10
warmup_ratio=0.01
batch_size = 4
gradient_accumulation_steps = 4
lr_scheduler_type = "linear"
optimizer = "adamw_8bit"
use_gradient_checkpointing = True
random_state = 42
```

### Framework versions
- Unsloth

### Example of usage

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "edumunozsala/unsloth-llama-2-7B-python-coder"

# Load the entire model on the GPU 0
device_map = {"": 0}

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, torch_dtype=torch.float16, 
                                             device_map="auto")

instruction="Write a Python function to display the first and last elements of a list."
input=""

prompt = f"""### Instruction:
Use the Task below and the Input given to write the Response, which is a programming code that can solve the Task.

### Task:
{instruction}

### Input:
{input}

### Response:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9,temperature=0.3)

print(f"Prompt:\n{prompt}\n")
print(f"Generated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")

```

### Citation

```
@misc {edumunozsala_2023,
	author       = { {Eduardo Muñoz} },
	title        = { unsloth-llama-2-7B-python-coder },
	year         = 2024,
	url          = { https://huggingface.co/edumunozsala/unsloth-llama-2-7B-python-coder },
	publisher    = { Hugging Face }
}
```