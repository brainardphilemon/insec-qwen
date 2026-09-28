from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, BitsAndBytesConfig
import tiktoken
from insec.secret import OPENAI_KEY
import openai
import torch
import random
from typing import List
import time
import json
import os

from insec import copilot, perplexity


class ModelWrapper:
    def __init__(self, model_name, args):
        pass

    def perplexity(self, prompt, suffix):
        raise NotImplementedError()

    def generate_infill(self, prompt, suffix, num_return_sequences, max_new_tokens) -> tuple[List[str], int, int]:
        raise NotImplementedError()

    def encode(self, text):
        raise NotImplementedError()

    def decode(self, tokens):
        raise NotImplementedError()

    def batch_decode(self, tokens, skip_special_tokens=False):
        raise NotImplementedError()

    def vocab_size(self):
        raise NotImplementedError()

    def model_max_length(self):
        raise NotImplementedError()


def get_num_ret_parts(num_return_sequences, div_const):
    num_ret_parts = [div_const] * (num_return_sequences // div_const)
    if num_return_sequences % div_const > 0:
        num_ret_parts.append(num_return_sequences % div_const)
    return num_ret_parts


class StarCoderModel(ModelWrapper):
    def __init__(self, model_name, temp, top_p) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # TODO do I need this?
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        if model_name == "bigcode/starcoderbase-7b":
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

        self.model.eval()

        self.temp = temp
        self.top_p = top_p

    def perplexity(self, prompt, suffix):
        full_prompt = "<fim_prefix>" + prompt + "<fim_suffix>" + suffix + "<fim_middle>"
        full_prompt_tokens = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.model.device)
        # prompt tokens cut into chunks of 10 tokens
        full_results = perplexity.Perplexity().compute_all(encoded_predictions=full_prompt_tokens, model=self.model, tokenizer=self.tokenizer, device=self.model.device)
        return full_results

    def generate_infill(self, prompt, suffix, num_return_sequences, max_new_tokens):
        # encode tokens
        full_prompt = "<fim_prefix>" + prompt + "<fim_suffix>" + suffix + "<fim_middle>"
        full_prompt_tokens = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.model.device)

        if self.temp == 0.0:
            do_sample = False
            num_return_sequences = 1
            self.top_p = 1
        else:
            do_sample = True

        generated_strs = []
        # controlls how many sequences are generated at once
        DIV_CONST = 100
        num_ret_parts = get_num_ret_parts(num_return_sequences, DIV_CONST)
        while True:
            try:
                for num_ret_part in num_ret_parts:
                    generated_tokens = self.model.generate(
                        full_prompt_tokens,
                        do_sample=do_sample,
                        num_return_sequences=num_ret_part,
                        temperature=self.temp,
                        max_new_tokens=max_new_tokens,
                        top_p=self.top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                        use_cache=True,
                    )

                    generated_tokens = generated_tokens[:, full_prompt_tokens.shape[1] :]

                    part_generated_strs = self.tokenizer.batch_decode(generated_tokens.tolist(), True)
                    generated_strs.extend(part_generated_strs)
                break
            except Exception as e:
                # Cuda out of memory
                print("Decreasing DIV CONST")
                print(e)
                DIV_CONST = DIV_CONST // 2
                num_ret_parts = get_num_ret_parts(num_return_sequences, DIV_CONST)

        # print(json.dumps(generated_strs, indent=2))
        # input()
        # for i in range(5):
        #     print(generated_strs[:3])

        return generated_strs, *self.measure_cost(full_prompt, generated_strs)

    def measure_cost(self, full_prompt, completion):
        return (
            len(self.tokenizer.encode(full_prompt)),
            sum(len(self.tokenizer.encode(c)) for c in completion),
        )


class StarCoder2Model(ModelWrapper):
    def __init__(self, model_name, temp, top_p) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # TODO do I need this?
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if model_name == "bigcode/starcoder2-7b" or model_name == "bigcode/starcoder2-15b":
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")     
        self.model.eval()

        self.temp = temp
        self.top_p = top_p

    def perplexity(self, prompt, suffix):
        full_prompt = "<file_sep><fim_prefix>" + prompt + "<fim_suffix>" + suffix + "<fim_middle>"
        full_prompt_tokens = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.model.device)
        full_results = perplexity.Perplexity().compute_all(encoded_predictions=full_prompt_tokens, model=self.model, tokenizer=self.tokenizer, device=self.model.device)
        return full_results

    def generate_infill(self, prompt, suffix, num_return_sequences, max_new_tokens):
        # encode tokens
        full_prompt = "<file_sep><fim_prefix>" + prompt + "<fim_suffix>" + suffix + "<fim_middle>"
        full_prompt_tokens = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.model.device)

        if self.temp == 0.0:
            do_sample = False
            num_return_sequences = 1
            self.top_p = 1
        else:
            do_sample = True

        generated_strs = []
        # controlls how many sequences are generated at once
        DIV_CONST = 100
        num_ret_parts = get_num_ret_parts(num_return_sequences, DIV_CONST)
        while True:
            try:
                for num_ret_part in num_ret_parts:
                    generated_tokens = self.model.generate(
                        full_prompt_tokens,
                        do_sample=do_sample,
                        num_return_sequences=num_ret_part,
                        temperature=self.temp,
                        max_new_tokens=max_new_tokens,
                        top_p=self.top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                        use_cache=True,
                    )

                    generated_tokens = generated_tokens[:, full_prompt_tokens.shape[1] :]
                    generated_tokens = generated_tokens.tolist()
                    # cut at first file_sep
                    file_sep_tok = self.tokenizer.encode("<file_sep>")[0]
                    for i in range(len(generated_tokens)):
                        if file_sep_tok in generated_tokens[i]:
                            idx = generated_tokens[i].index(file_sep_tok)
                            generated_tokens[i] = generated_tokens[i][:idx]

                    part_generated_strs = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
                    generated_strs.extend(part_generated_strs)
                break
            except Exception as e:
                # Cuda out of memory
                print("Decreasing DIV CONST")
                print(e)
                DIV_CONST = DIV_CONST // 2
                num_ret_parts = get_num_ret_parts(num_return_sequences, DIV_CONST)

        # print(json.dumps(generated_strs, indent=2))
        # input()
        return generated_strs, *self.measure_cost(full_prompt, generated_strs)

    def measure_cost(self, full_prompt, completion):
        return (
            len(self.tokenizer.encode(full_prompt)),
            sum(len(self.tokenizer.encode(c)) for c in completion),
        )


def _select_device_and_dtype():
    """Pick the best available torch device/dtype for local inference.

    Priority: CUDA (bf16) > Apple-Silicon MPS (fp16) > CPU (fp32).
    Overridable with the INSEC_DEVICE / INSEC_DTYPE environment variables,
    e.g. INSEC_DEVICE=cpu INSEC_DTYPE=float32.
    """
    device_override = os.environ.get("INSEC_DEVICE")
    dtype_override = os.environ.get("INSEC_DTYPE")

    if device_override:
        device = torch.device(device_override)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if dtype_override is not None:
        dtype = getattr(torch, dtype_override)
    elif device.type == "cuda":
        # bf16 only on Ampere+ (A100 etc.); Kaggle's T4/P100 run fp16.
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device.type == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32

    return device, dtype


class Qwen25Model(ModelWrapper):
    """Wrapper for Qwen2.5 / Qwen2.5-Coder base models (FIM code completion).

    Qwen2.5 fill-in-the-middle prompt format:
        <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>
    The model fills in the middle; generation is cut at the first repo/file/eos
    sentinel. Runs on CUDA if present, otherwise Apple-Silicon MPS, otherwise CPU.
    """

    FIM_PREFIX = "<|fim_prefix|>"
    FIM_SUFFIX = "<|fim_suffix|>"
    FIM_MIDDLE = "<|fim_middle|>"

    def __init__(self, model_name, temp, top_p) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.device, dtype = _select_device_and_dtype()
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
            self.model.to(self.device)
        except Exception as e:  # e.g. not enough unified memory to place on MPS/GPU
            print(f"Failed to load {model_name} on {self.device} ({e}); falling back to CPU")
            self.device = torch.device("cpu")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
            self.model.to(self.device)

        self.model.eval()

        # Sentinels that terminate an infill. Generation is cut at the first one.
        self.stop_token_ids = set()
        for tok in ["<|endoftext|>", "<|fim_pad|>", "<|file_sep|>", "<|repo_name|>"]:
            tok_id = self.tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tok_id, int) and tok_id >= 0 and tok_id != self.tokenizer.unk_token_id:
                self.stop_token_ids.add(tok_id)

        self.temp = temp
        self.top_p = top_p

    def _format_fim(self, prompt, suffix):
        return self.FIM_PREFIX + prompt + self.FIM_SUFFIX + suffix + self.FIM_MIDDLE

    def perplexity(self, prompt, suffix):
        full_prompt = self._format_fim(prompt, suffix)
        full_prompt_tokens = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
        return perplexity.Perplexity().compute_all(
            encoded_predictions=full_prompt_tokens,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

    def generate_infill(self, prompt, suffix, num_return_sequences, max_new_tokens):
        full_prompt = self._format_fim(prompt, suffix)
        full_prompt_tokens = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)

        if self.temp == 0.0:
            do_sample = False
            num_return_sequences = 1
            self.top_p = 1
        else:
            do_sample = True

        generated_strs = []
        # controls how many sequences are generated at once (lowered on OOM)
        DIV_CONST = 100
        num_ret_parts = get_num_ret_parts(num_return_sequences, DIV_CONST)
        while True:
            try:
                for num_ret_part in num_ret_parts:
                    generated_tokens = self.model.generate(
                        full_prompt_tokens,
                        do_sample=do_sample,
                        num_return_sequences=num_ret_part,
                        temperature=self.temp,
                        max_new_tokens=max_new_tokens,
                        top_p=self.top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                        use_cache=True,
                    )

                    generated_tokens = generated_tokens[:, full_prompt_tokens.shape[1] :]
                    generated_tokens = generated_tokens.tolist()
                    # cut each sequence at the first stop sentinel
                    for i in range(len(generated_tokens)):
                        for j, tok_id in enumerate(generated_tokens[i]):
                            if tok_id in self.stop_token_ids:
                                generated_tokens[i] = generated_tokens[i][:j]
                                break

                    part_generated_strs = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    generated_strs.extend(part_generated_strs)
                break
            except Exception as e:
                # Out of memory: halve the per-batch count and retry.
                print("Decreasing DIV CONST")
                print(e)
                if DIV_CONST <= 1:
                    raise
                DIV_CONST = DIV_CONST // 2
                num_ret_parts = get_num_ret_parts(num_return_sequences, DIV_CONST)

        return generated_strs, *self.measure_cost(full_prompt, generated_strs)

    def measure_cost(self, full_prompt, completion):
        return (
            len(self.tokenizer.encode(full_prompt)),
            sum(len(self.tokenizer.encode(c)) for c in completion),
        )


class CodeLlamaModel(ModelWrapper):
    def __init__(self, model_name, temp, top_p):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_bos_token = False
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        self.model.eval()

        self.temp = temp
        self.top_p = top_p

    def perplexity(self, prompt, suffix):
        full_prompt_tokens = self.format_fim(prompt, suffix)
        full_results = perplexity.Perplexity().compute_all(encoded_predictions=full_prompt_tokens, model=self.model, tokenizer=self.tokenizer, device=self.model.device)
        return full_results

    def encode_pre(self, pre):
        return self.tokenizer.encode(pre)

    def encode_suf(self, suf):
        return self.tokenizer.encode("☺" + suf)[2:]

    # Formatting from the original repo
    def format_fim(self, pre, suf):
        full_prompt = (
                self.tokenizer.encode(self.tokenizer.bos_token)
                + [self.tokenizer.prefix_id]
                + self.encode_pre(pre)
                + [self.tokenizer.suffix_id]
                + self.encode_suf(suf)
                + [self.tokenizer.middle_id]
        )
        return torch.tensor(full_prompt).to("cuda").unsqueeze(0)

    def generate_infill(self, prompt, suffix, num_return_sequences, max_new_tokens):

        full_prompt_tokens = self.format_fim(prompt, suffix)
        full_prompt_str = self.tokenizer.decode(full_prompt_tokens[0].tolist(), True)
        # print(repr(full_prompt_str))

        # Huggingface gives a different result
        # full_prompt = prompt + "<FILL_ME>" + suffix
        # full_prompt_tokens = self.tokenizer(full_prompt, return_tensors="pt")[
        #     "input_ids"
        # ].to("cuda")
        # full_prompt_str = self.tokenizer.decode(full_prompt_tokens[0].tolist(), True)
        # print(repr(full_prompt_str))
        # print()
        # input()

        generated_strs = []
        DIV_CONST = 50
        num_ret_parts = get_num_ret_parts(num_return_sequences, DIV_CONST)

        while True:
            try:
                for num_ret_part in num_ret_parts:
                    generated_tokens = self.model.generate(
                        full_prompt_tokens,
                        do_sample=True,
                        num_return_sequences=num_ret_part,
                        temperature=self.temp,
                        max_new_tokens=max_new_tokens,
                        top_p=self.top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                        use_cache=True,
                    )

                    part_generated_strs = self.tokenizer.batch_decode(generated_tokens.tolist(), True)
                    part_generated_strs = [s[len(full_prompt_str) :] for s in part_generated_strs]
                    generated_strs.extend(part_generated_strs)
                break
            except Exception as e:
                # Cuda out of memory
                print("Decreasing DIV CONST")
                print(e)
                DIV_CONST = DIV_CONST // 2
                num_ret_parts = get_num_ret_parts(num_return_sequences, DIV_CONST)

        return generated_strs, *self.measure_cost(full_prompt_tokens, generated_strs)

    def measure_cost(self, prompt_tokens, completion):
        return (
            len(prompt_tokens),
            sum(len(self.tokenizer.encode(c)) for c in completion),
        )


class OpenAIModel(ModelWrapper):
    # "gpt-3.5-turbo-instruct-0914"
    def __init__(self, model_name, temp, top_p):
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        openai.api_key = OPENAI_KEY

        self.temp = temp
        self.top_p = top_p

        # logging system fingerprint not available with the completion API

    def generate_infill(self, prompt, suffix, num_return_sequences, max_new_tokens):
        seed = random.randint(0, 100000)
        completion = None
        while completion is None:
            try:
                completion = openai.Completion.create(
                    model=self.model_name,
                    prompt=prompt,
                    suffix=suffix,
                    n=num_return_sequences,
                    max_tokens=max_new_tokens,
                    temperature=self.temp,
                    top_p=self.top_p,
                    seed=seed,
                )
            except Exception as e:
                print(f"OpenAI error: {e}")
                time.sleep(0.5)

        ret = []

        for choice in completion.choices:
            ret.append(choice.text)

        return ret, *self.measure_cost(completion)

    def measure_cost(self, response):
        try:
            return (
                response["usage"]["prompt_tokens"],
                response["usage"]["completion_tokens"],
            )
        except KeyError:
            return (0, 0)

    def __vocab_size(self):
        # return self.tokenizer.max_token_value
        # many values betwen 10255 and 100276 invalid
        return 100255


class CopilotModel(ModelWrapper):
    def __init__(self, log_path=None) -> None:
        self.server = copilot.spawn_server()
        copilot.await_fn(self.server, copilot.initialize)
        copilot.await_fn(self.server, copilot.login)
        super().__init__(self.server, "copilot")

        self.log_path = log_path

    def one_request(self, prompt, suffix):
        text = prompt + suffix
        # print("text")
        # print(text)
        # print("-" * 80)
        _split = prompt.splitlines()
        line = len(_split) - 1
        char = len(_split[-1])
        position = copilot.Position(
            line,
            char,
        )

        retries = 0
        base_delay = 1
        max_delay = 32
        while retries < 3:
            try:
                res = copilot.await_fn(
                    self.server,
                    lambda server, cb: copilot.request_panel_completion(server, cb, text, position),
                    timeout=100,
                )
                results = [r["completionText"] for r in res]
                # If no completions are suggested at all, retry
                if not results:
                    raise Exception("No completions")

                if self.log_path is not None:
                    with open(self.log_path, "a") as f:
                        f.write(json.dumps(results))
                        f.write("\n")

                # cut the results after the new line
                results = [r.split("\n")[0] for r in results]

                time.sleep(0.5 + 1 * random.random())
                # input("Press enter to continue...")
                return results, *self.measure_cost()
            except Exception as e:
                print(e)
                self.server.close()
                delay = min(base_delay * (2**retries), max_delay)
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                self.server = copilot.spawn_server()
                copilot.await_fn(self.server, copilot.initialize)
                copilot.await_fn(self.server, copilot.login)
                retries += 1

        return [""], *self.measure_cost()

    def generate_infill(self, prompt: str, suffix, num_return_sequences, max_new_tokens):
        completions = []
        while len(completions) < num_return_sequences:
            completion, _, _ = self.one_request(prompt, suffix)
            completions.extend(completion)
        # print(json.dumps(completions, indent=2))
        return completions, 0, 0

    def measure_cost(self):
        return (0, 0)


def load_model(args) -> ModelWrapper:
    if hasattr(args, "model") and args.model is not None:
        return args.model

    if "gpt" in args.model_dir:
        return OpenAIModel(args.model_dir, args.temp, args.top_p)
    elif "llama" in args.model_dir:
        return CodeLlamaModel(args.model_dir, args.temp, args.top_p)
    elif "copilot" in args.model_dir:
        # check if output_dir exists in args
        if hasattr(args, "result_dir") and args.result_dir is not None:
            return CopilotModel(os.path.join(args.result_dir, "completions_log.jsonl"))
        elif hasattr(args, "output_dir") and args.output_dir is not None:
            return CopilotModel(os.path.join(args.output_dir, "completions_log.jsonl"))
        else:
            return CopilotModel()
    elif "starcoder2" in args.model_dir:
        return StarCoder2Model(args.model_dir, args.temp, args.top_p)
    elif "qwen" in args.model_dir.lower():
        return Qwen25Model(args.model_dir, args.temp, args.top_p)
    else:
        return StarCoderModel(args.model_dir, args.temp, args.top_p)
