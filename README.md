# 👩‍💻 Fine-tune a Llama 2 7B parameters to generate Python Code using Unsloth

**LlaMa-2 7B** model fine-tuned on the **python_code_instructions_18k_alpaca Code instructions dataset** by using the library [Unsloth](https://github.com/unslothai/unsloth).

## The dataset

For our tuning process, we will take a [dataset](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) containing about 18,000 examples where the model is asked to build a Python code that solves a given task. 
This is an extraction of the [original dataset](https://huggingface.co/datasets/sahil2801/code_instructions_120k) where only the Python language examples are selected. Each row contains the description of the task to be solved, an example of data input to the task if applicable, and the generated code fragment that solves the task is provided.

## Problem description

Our goal is to fine-tune the pretrained model, Llama 2 7B parameters, using the library **Unsloth** to produce a Python coder. We will run the training on Google Colab using a A100 or V100 to get better performance. But you can try out to run it on a T4 adjusting some parameters to reduce memory consumption like batch size.

Once the model is fine-tuned, we load the adapter to the Hub. Then, we merge the adapter to the base model and we also upload the merge model.

## The base model
[Llama-2](https://huggingface.co/meta-llama/Llama-2-7b)

Meta developed and publicly released the Llama 2 family of large language models (LLMs), a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters.

Model Architecture Llama 2 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align to human preferences for helpfulness and safety

## Unsloth

Recently a new framework or library to optimize the training, and fine-tuning, stage of a large language model has been released: Unsloth. This library is a product of moonshot and was built by two brothers, Daniel and Michael Han, and they promise much faster and memory-efficient finetuning.

The authors highlight that while PyTorch's Autograd is generally efficient for most tasks, achieving extreme performance requires manually deriving matrix differentials. The authors perform simple matrix dimension FLOP (floating-point operation) calculations and find that bracketing the LoRA weight multiplications significantly enhances performance.

All these features are spectacular, they can reduce a lot of time and resources needed to fine-tune LLMs.Here we will try the open source version that can achieve a 2x faster, but there is also a PRO and a MAX version that can achieve a 30x faster training and up to 60% memory consumption reduction.

To achieve a better performance, they have developed a few techniques:
1. Reduce weights upscaling during QLoRA, fewer weights result in less memory consumption and faster training.
2. Bitsandbytes works with float16 and then converts to bfloat16, Unsloth directly uses bfloat16.
3. Use of Pytorch's Scaled Dot Product Attention implementation
4. Integration of Xformers and Flash Attention 2 to optimize the transformer model
5. Using a causal mask to speed up training instead of a separate attention mask
6. Implementing fast ROPE embeddings with OpenAI's Triton
7. Accelerate RMS Normalization with Triton
8. Optimize Cross entropy loss computation to significantly reduce memory consumption
9. Implementing a manual Autograd for MLP and Self-Attention layers to optimize Pytorch's default implementation

## Content

- Fine-tuning notebook `finetune-unsloth-llama-2-py-coder.ipynb`: In this notebook we fine-tune the model.
- 
### Example of usage

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "edumunozsala/unsloth-llama-2-7B-python-coder"

# Load the entire model on the GPU 0
device_map = {"": 0}

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, torch_dtype=torch.float16, 
                                             device_map=device_map)

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
## Contributing
If you find some bug or typo, please let me know or fixit and push it to be analyzed. 

## License

These notebooks are under a public GNU License version 3.