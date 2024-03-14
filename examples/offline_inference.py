from vllm import LLM, SamplingParams
import torch
import os

long_prompt = '''
    Unveiling the Power of Language: A Deep Dive into Large Language Models (LLMs)
    Large language models (LLMs) represent a groundbreaking leap in artificial intelligence (AI), particularly within the realm of natural language processing (NLP). These sophisticated algorithms are revolutionizing how we interact with machines, blurring the lines between human and computer-generated language. This article delves into the intricate world of LLMs, exploring their inner workings, capabilities, applications, and the profound impact they hold for the future.

    Demystifying LLMs: A Neural Network Playground
    At their core, LLMs are complex neural networks trained on massive amounts of text data. These datasets encompass everything from books and articles to code and social media conversations. By ingesting this vast sea of information, LLMs learn the statistical patterns and relationships between words. Imagine a child learning a language; they absorb vocabulary, grammar, and sentence structure through exposure. Similarly, LLMs develop an understanding of language by analyzing the building blocks of text.

    One of the key architectural components of LLMs is the transformer, a deep learning model specifically designed for NLP tasks. Transformers excel at analyzing sequential data, like sentences, and understanding the relationships between words within that sequence. Through a series of intricate calculations, they learn the context and meaning within a given piece of text.

    Beyond Mimicry: The Power of LLM Capabilities
    LLMs boast a diverse arsenal of capabilities that extend far beyond simply mimicking human language. Here's a glimpse into some of their most impressive feats:

    Text Generation: LLMs can generate coherent and grammatically correct text, from crafting realistic dialogue for chatbots to composing creative fiction. Imagine a world where writers can leverage LLMs to overcome writer's block or generate initial drafts.
    Machine Translation: LLMs are pushing the boundaries of machine translation, offering near-human quality translations across different languages. This opens doors to improved communication and collaboration on a global scale.
    Question Answering: LLMs can analyze vast amounts of information to answer complex questions with pinpoint accuracy. Imagine a research assistant that can sift through mountains of data and present you with the key insights.
    Text Summarization: LLMs can condense lengthy documents into concise summaries, extracting the essential points and allowing users to quickly grasp the gist of the content. This is a boon for busy individuals navigating the information overload of the digital age.
    Code Generation: LLMs are demonstrating the ability to generate basic computer code, potentially assisting programmers by automating repetitive tasks or suggesting code snippets based on user intent.
    A Spectrum of Applications: LLMs Transforming Our World
    LLMs are poised to disrupt a multitude of industries and reshape the way we interact with technology. Here are some potential applications across various sectors:

    Education: LLMs can create personalized learning experiences, tailoring content and practice questions to individual student needs. They can also act as intelligent tutors, providing feedback and explanations in real-time.
    Customer Service: LLMs can power chatbots that can engage in natural, nuanced conversations with customers, resolving issues efficiently and offering personalized support.
    Creative Industries: Writers, artists, and musicians can utilize LLMs as collaborative tools, sparking creativity and generating new ideas. Imagine a composer working with an LLM to create unique musical compositions.
    Healthcare: LLMs can facilitate communication between patients and doctors, analyzing medical records and offering insights to improve diagnoses and treatment plans.
    Scientific Research: LLMs can analyze vast scientific datasets, identifying patterns and suggesting new research avenues, accelerating scientific progress.
    The Ethical Landscape: Navigating the Challenges of LLMs
    Despite their immense potential, LLMs are not without challenges. Here are some key ethical considerations:

    Bias: LLMs are trained on existing data, which can be inherently biased. This can lead to biased outputs that perpetuate inequalities and discrimination. Mitigating bias in training data is crucial for ensuring fair and responsible use of LLMs.
    Misinformation and Disinformation: LLMs can be used to create highly convincing fake news articles or social media posts. Addressing this challenge requires developing robust methods for detecting and preventing the spread of false information.
    Job Displacement: Automation powered by LLMs could potentially replace certain jobs. However, LLMs are more likely to augment existing roles, freeing up humans to focus on more complex tasks.
    The Future Beckons: A World Shaped by LLMs
    '''
x = '''Explainability and Transparency: Understanding how LLMs arrive at their outputs can be difficult. This lack of transparency raises concerns about accountability and potential misuse.
    Addressing these challenges is crucial for ensuring that LLMs are developed and used responsibly. Open dialogue, collaboration between researchers, and ethical frameworks are essential to harness the power of LLMs for good.

    The Future Beckons: A World Shaped by LLMs
    The future holds immense promise for LLMs. As research progresses, we can expect even more sophisticated models capable of complex reasoning, emotional intelligence, and nuanced understanding of the human experience. LLMs have the potential to
    '''

# Sample prompts.
prompts = [
    "What is OpenVINO?",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
    #long_prompt * 4 # for Mistral
]
# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=64, ignore_eos=True)
# for beam search
# sampling_params = SamplingParams(max_tokens=64, ignore_eos=True, use_beam_search=True, n=4, temperature=0.0)

# crop the model for test
model = '/home/llm_irs/pytorch_frontend_models/llama-2-7b-chat/pytorch_original/'
#model = '/mnt/chatglm2-6b/'
#model = '/mnt/disk1/luocheng/model/llama-2-7b-chat-modify'
#model = '/mnt/disk1/luocheng/model/Mistral-7B-v0.1'
#model = '/mnt/disk1/luocheng/model/Mistral-7B-v0.1-modify'
if 0:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    src = '/home/llm_irs/pytorch_frontend_models/llama-2-7b-chat/pytorch_original/'
    dst = '/mnt/disk1/luocheng/model/llama-2-7b-chat-modify'
    src = '/mnt/disk1/luocheng/model/Mistral-7B-v0.1'
    dst = '/mnt/disk1/luocheng/model/Mistral-7B-v0.1-modify'
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
    # TODO: SPR will use bf16, others are f16
    kv_cache_dtype = torch.bfloat16
# Create an LLM.
llm = LLM(model=model, dtype=torch.float32, kv_cache_dtype=kv_cache_dtype, device='cpu', swap_space=8,
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
