from vllm import LLM, SamplingParams
import torch
import os

# Sample prompts.
prompts = [
    "What is OpenVINO?",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=64, ignore_eos=True)
# for beam search
# sampling_params = SamplingParams(max_tokens=64, ignore_eos=True, use_beam_search=True, n=4, temperature=0.0)

# crop the model for test
model = '/home/llm_irs/pytorch_frontend_models/llama-2-7b-chat/pytorch_original/'
#model = '/mnt/disk1/luocheng/model/llama-2-7b-chat-modify'
if 0:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    src = '/home/llm_irs/pytorch_frontend_models/llama-2-7b-chat/pytorch_original/'
    dst = '/mnt/disk1/luocheng/model/llama-2-7b-chat-modify'
    m = AutoModelForCausalLM.from_pretrained(src)
    tokenizer = AutoTokenizer.from_pretrained(src)
    m.config.num_hidden_layers = 1
    m.generation_config.temperature = 1.0
    m.generation_config.top_p = 1.0
    m.model.layers = m.model.layers[:m.config.num_hidden_layers]
    print(m)
    tokenizer.save_pretrained(dst)
    m.save_pretrained(dst)

block_size = 16
kv_cache_dtype = 'auto'
if ('VLLM_OPENVINO' in os.environ and os.environ['VLLM_OPENVINO'] == '1') and \
   ('ENABLE_PG' in os.environ and os.environ['ENABLE_PG'] == '1'):
    block_size = 1
    kv_cache_dtype = torch.bfloat16
# Create an LLM.
llm = LLM(model=model, dtype=torch.float32, kv_cache_dtype=kv_cache_dtype, device='cpu',
          trust_remote_code=True, seed=42, max_model_len=1024, block_size=block_size)
#llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", dtype=torch.float32, device='cpu', trust_remote_code=True, seed=42, max_model_len=1024)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
