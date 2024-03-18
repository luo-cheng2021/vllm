from vllm import LLM, SamplingParams
import torch
import os
import sys
from multiprocessing import Process, Queue, Pool

def log(func):
    def wrapper(*args, **kw):
        print(f'------------ testing {func.__name__} ----------------------')
        ret = func(*args, **kw)
        print(f'------------ test {func.__name__} done ----------------------')
        return ret
    return wrapper

def run(prompts, model_path, run_ref, test_beam=False, test_hf_model=False):
    if test_beam:
        sampling_params = SamplingParams(max_tokens=32, ignore_eos=True, use_beam_search=True, n=4, temperature=0.0)
    else:
        sampling_params = SamplingParams(max_tokens=32, ignore_eos=True)

    if test_hf_model:
        os.environ['VLLM_OPENVINO'] = '0'
        os.environ['VLLM_OPENVINO_OPTIMUM'] = '1'
    else:
        os.environ['VLLM_OPENVINO'] = '1'
        os.environ['VLLM_OPENVINO_OPTIMUM'] = '0'
    print(f'========running {"ref" if run_ref else "target"} {"with" if test_beam else "without"} beam search using {"hf" if test_hf_model else "vllm"} model...')
    if run_ref:
        block_size = 16
        kv_cache_dtype = 'auto'
        os.environ['ENABLE_PG'] = '0'
    else:
        block_size = 1
        # TODO: SPR will use bf16, others are f16
        kv_cache_dtype = torch.bfloat16
        os.environ['ENABLE_PG'] = '1'

    if 'Mistral' in model_path:
        max_model_len = 8192
    else:
        max_model_len = 1024
    def proc(q):
        # Create an LLM.
        llm = LLM(model=model_path, dtype=torch.float32, kv_cache_dtype=kv_cache_dtype, device='cpu', swap_space=8,
                trust_remote_code=True, seed=42, max_model_len=max_model_len, block_size=block_size)
        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        q.put(outputs)
    
    q = Queue()
    p = Process(target=proc, args=(q,))
    p.start()
    outputs = q.get()
    p.join()
    # Print the outputs.
    result = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        #print(f'Prompt: {prompt!r}, Generated text: {generated_text!r}')
        result.append(generated_text)
    return result

def run_one_pair(prompt, model, test_beam=False, test_hf_model=False):
    ref = run(prompt, model, True, test_beam=test_beam, test_hf_model=test_hf_model)
    cur = run(prompt, model, False, test_beam=test_beam, test_hf_model=test_hf_model)
    assert ref == cur, f'ref = \n{ref}\ncur = \n{cur}\n'

@log
def test_mixtral():
    run_one_pair('[INST]What is OpenVINO?[/INST]',
                 'fxmeng/Mixtral-2x7B-Instruct-v0.1')

@log
def test_basic():
    run_one_pair('OpenVINO is an open-source toolkit for machine learning inference, developed by Intel. It allows developers to optimize and run deep learning models',
                 'meta-llama/Llama-2-7b-chat-hf')

@log
def test_hf_model():
    run_one_pair('OpenVINO is an open-source toolkit for machine learning inference, developed by Intel. It allows developers to optimize and run deep learning models',
                 'meta-llama/Llama-2-7b-chat-hf', test_hf_model=test_hf_model)

@log
def test_hf_model_beam_search():
    run_one_pair('OpenVINO is an open-source toolkit for machine learning inference, developed by Intel. It allows developers to optimize and run deep learning models',
                 'meta-llama/Llama-2-7b-chat-hf', test_beam=True, test_hf_model=test_hf_model)

@log
def test_batching():
    run_one_pair(['OpenVINO is an open-source toolkit for machine learning inference, developed by Intel. It allows developers to optimize and run deep learning models',
                  'The president of the United States is',
                  'The capital of France is',
                  'The future of AI is',],
                 'meta-llama/Llama-2-7b-chat-hf')

@log
def test_beam_search():
    run_one_pair('OpenVINO is an open-source toolkit for machine learning inference, developed by Intel. It allows developers to optimize and run deep learning models',
                 'meta-llama/Llama-2-7b-chat-hf', test_beam=True)

@log
def test_multi_query():
    run_one_pair('OpenVINO is an open-source toolkit for machine learning inference, developed by Intel. It allows developers to optimize and run deep learning models',
                 'THUDM/chatglm2-6b', test_beam=True)

@log
def test_first_token_less_than_sliding_window():
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
    # prompt len = 4070, sliding_window = 4096, output = 32
    run_one_pair(long_prompt * 4, 'mistralai/Mistral-7B-v0.1')

@log
def test_first_token_great_than_sliding_window():
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
        Explainability and Transparency: Understanding how LLMs arrive at their outputs can be difficult. This lack of transparency raises concerns about accountability and potential misuse.
        Addressing these challenges is crucial for ensuring that LLMs are developed and used responsibly. Open dialogue, collaboration between researchers, and ethical frameworks are essential to harness the power of LLMs for good.

        The Future Beckons: A World Shaped by LLMs
        The future holds immense promise for LLMs. As research progresses, we can expect even more sophisticated models capable of complex reasoning, emotional intelligence, and nuanced understanding of the human experience. LLMs have the potential to
        '''
    # prompt len = 4582, sliding_window = 4096, output = 32
    run_one_pair(long_prompt * 4, 'mistralai/Mistral-7B-v0.1')


test_basic()
#test_mixtral()
test_hf_model()
test_hf_model_beam_search()
test_batching()
test_beam_search()
test_multi_query()
test_first_token_less_than_sliding_window()
test_first_token_great_than_sliding_window()

print('success.')