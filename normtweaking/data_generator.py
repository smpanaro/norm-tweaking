from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import random
import torch
from safetensors.torch import save_file
import os

class CalibrationData:
    def __init__(self, model_name, token_length, tensors):
        self.model_name = model_name
        self.token_length = token_length
        self.tensors = tensors

    def save(self, save_dir: str):
        samples = [el for t in self.tensors for el in t]
        named_samples = {f"sample_{i}": s for i, s in enumerate(samples)}

        filename = f"normtweaking_calibration_data_{self.model_name.replace('/','-')}_{self.token_length}token_{len(samples)}samples.safetensors"
        os.makedirs(save_dir, exist_ok=True)
        save_file(named_samples, f"{save_dir}/{filename}")

class DataGenerator:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.device = self._default_device()

    @staticmethod
    def from_pretrained(model_name_or_path):
        """
        Create a DataGenerator for a huggingface model.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        return DataGenerator(tokenizer, model)

    def to(self, device):
        self.model.to(device=device)
        self.device = device

    @staticmethod
    def _default_device():
        if torch.cuda.is_available():
            return 'cuda'
        # MPS seems slower than CPU.
        # elif torch.backends.mps.is_available():
        #     return 'mps'
        else :
            return 'cpu'

    def generate(self, n_samples=128, token_length=None) -> CalibrationData:
        """
        Generate calibration data as described in the LLM-QAT paper.
        TODO: Update to generate with the additions described in the 'Calibration Data Generation'
              section of the Norm Tweaking paper.

        Args:
            n_samples: number of samples to generate, default=128
            token_length: number of tokens in each sample, default=model dependent
        """
        self.to(device=self.device)

        token_length = token_length or self.model.config.n_positions
        start_tokens = self._choose_start_tokens(n_samples)
         # Ensure at least 3 groups so we can get samples with 3-5 top-1 tokens.
        token_groups = [x.tolist() for x in torch.tensor(start_tokens).split(max(1,min(16, n_samples//3)))]

        # tensors = [self._generate_sample(offset, start_token, token_length) for offset, start_token in tqdm(start_token)]

        tensors = [self._generate_batch_samples(offset, ts, token_length) for offset, ts in tqdm(enumerate(token_groups))]
        return CalibrationData(self.model.config.name_or_path, token_length, tensors)

    def _generate_sample(self, offset: int, start_token: int, token_length: int, disalllowed_tokens: set[int]):
        """
        Generate a sample for the given token. If generating multiple samples, use _generate_batch_samples instead.
        """
        assert token_length >= 2

        # Sample from the model. As in LLM-QAT, for the first 3-5 tokens we take
        # the top-1 and for the remaining tokens we sample stochastically until EOS
        # or we reach token_length.

        tokens = torch.tensor([[start_token]]).to(self.device)
        attention_mask = torch.ones_like(tokens)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
        inputs = {'input_ids': tokens, 'attention_mask': attention_mask, 'pad_token_id': pad_token_id}

        # Greedy by default.
        tokens = self.model.generate(**inputs, max_length=3 + offset%3, do_sample=False)

        # Sampled.
        attention_mask = torch.ones_like(tokens)
        inputs = {'input_ids': tokens, 'attention_mask': attention_mask, 'pad_token_id': pad_token_id}
        tokens = self.model.generate(**inputs, max_length=token_length, do_sample=True)

        assert tokens.shape[-1] <= token_length, f"Generated too many tokens: {tokens.shape[-1]} > {token_length}"
        assert tokens[0, 0] == start_token, f"Generated token does not match start token: {tokens[0, 0]} != {start_token}"

        # print(self.tokenizer.decode(tokens.squeeze().tolist()))

        return tokens

    def _generate_batch_samples(self, offset: int, start_tokens: list[int], token_length: int):
        """
        Generate samples for all tokens in start_tokens. More efficient than calling _generate_sample
        for each token individually.
        """
        assert token_length >= 2

        # Sample from the model. As in LLM-QAT, for the first 3-5 tokens we take
        # the top-1 and for the remaining tokens we sample stochastically until EOS
        # or we reach token_length.

        tokens = torch.tensor([[start_token] for start_token in start_tokens]).to(self.device)
        attention_mask = torch.ones_like(tokens)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
        inputs = {'input_ids': tokens, 'attention_mask': attention_mask, 'pad_token_id': pad_token_id}
        # print({k: v.shape if k != 'pad_token_id' else v for k, v in inputs.items()})

        # Greedy by default.
        tokens = self.model.generate(**inputs, max_length=3 + offset%3, do_sample=False)

        # Sampled.
        attention_mask = torch.ones_like(tokens)
        inputs = {'input_ids': tokens, 'attention_mask': attention_mask, 'pad_token_id': pad_token_id}
        tokens = self.model.generate(**inputs, max_length=token_length, do_sample=True)

        # TODO: Trim superfluos endoftext tokens.

        # assert tokens.shape[-1] <= token_length, f"Generated too many tokens: {tokens.shape[-1]} > {token_length}"
        # assert tokens[0, 0] == start_token, f"Generated token does not match start token: {tokens[0, 0]} != {start_token}"
        # print("result shape", tokens.shape)
        return tokens

    def _choose_start_tokens(self, n_samples: int):
        # TODO: Figure out how to determine the language of each token and only sample from the top (majority/75+%) of languages.
        # Maybe use something like: https://huggingface.co/facebook/fasttext-language-identification
        disalllowed_tokens = set(self.tokenizer.all_special_ids)
        start_tokens = []

        for _ in range(n_samples):
            start_token = None
            while start_token is None:
                start_token = random.randint(0, self.tokenizer.vocab_size)
                if start_token in disalllowed_tokens or not self.is_good_start_token(start_token):
                    start_token = None

            start_tokens.append(start_token)
            disalllowed_tokens.add(start_token)

        return start_tokens

    def is_good_start_token(self, token: int):
        s = self.tokenizer.decode(token)
        s = s.strip()

        if s.isspace():
            return False

        if len(s) == 0:
            return False

        # gpt2 has some weird tokens: aaaa
        if len(s) >= 3 and s[0] == s[1] == s[2]:
            return False

        return True


