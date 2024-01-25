import torch
from torch import nn
from tqdm import tqdm
from datetime import datetime
from safetensors.torch import save_model
import safetensors.torch as safe_torch

import os
import json
from typing import Optional, Dict

from .quantizers import Quantizer

VERBOSE=False # Show tqdm for all sub-steps. Too verbose for grid search.

class Metric:
    """
    Measurement of the difference between a float output of a layer
    and a quantized output of a layer. Deltas can be calculated along
    different axes.
    """
    def __init__(self, layer, deltas: [float]):
        deltas = torch.tensor(deltas)
        self.layer = layer
        self.delta_mean = deltas.mean()
        self.delta_var = deltas.var()

class NormTweaker:
    """
    Implement the Norm Tweaking algorithm:

    Input: Pre-trained LLM model
    Output: Quantized LLM model
    1 : Generate calibration dataset (n samples = 128, token length = 2048) using pre-trained LLM model
    2 : for each layer-l in the Transformer structure (L layers total) do
    3 :   if l=0 then
    4 :     use calibration data as input
    5 :   else
    6 :     use last output qOutl−1 as input
    7 :   end if
    8 :   Calculate the float output f Outl
    9 :   Quantize the weights of layer l
    10:   Freeze all Linear’s weights in layer l
    11:   for each it for total Iters do
    12:     Calculate the float output qOutl
    13:     Calculate Ldist between fOutl and qOutl
    14:     Backward and update LayerNorms’ parameters
    15:   end for
    16: endfor
    17: Get the high-performance quantized LLMs
    """

    def __init__(self, quantizer: Quantizer, save_dir: str, skip_tweak=False, initial_learning_rate=None, lr_scale=None):
        """
        quantizer: Quantizer subclass to quantize the linear layer weights
        """
        self.quantizer = quantizer
        # "In our experiments, we typically use a grid search to obtain
        # the optimal learning rate, with an initial value set at 1e-5."
        self.initial_learning_rate = initial_learning_rate if initial_learning_rate is not None else 1e-5
        # 0 for no increase (first + last layer have same weight), n for linear increase
        # (last layer has n times the lr of the first layer)
        # I do not have an intuition for this paramemter, but values from 0-200 have
        # yielded decent results depending on the lr. Suggest running a grid search.
        self.lr_scale = lr_scale if lr_scale is not None else 5
        self.save_dir = save_dir
        # Skip norm tweaking, only quantize the model.
        # Useful for getting a non-tweaked baseline.
        self.skip_tweak = skip_tweak
        # TODO: Support total_iters (the paper says 1 is optimal so starting there).
        self.train_metrics = []
        self.saver = ModelSaver(self)

    def tweak(self, model: nn.Module, samples: dict[str, torch.Tensor], samples_is_first_layer_input=False, skip_save=False):
        if samples_is_first_layer_input:
            current_inputs = samples
        else:
            current_inputs = self.collect_first_layer_inputs(model, samples)

        for i, layer in tqdm(list(enumerate(model.h)), desc="layers"):
            # Slightly un-intuitive but we want to increase the rate as the layers progress
            # because error accumulates and we want to correct it more strongly.
            lr = self.initial_learning_rate * (1 + self.lr_scale * (i/len(model.h)))
            current_inputs = self._tweak_layer(i, layer, current_inputs, lr)

        print("\ntrain metrics:")
        self.print_metrics(self.train_metrics)

        if not skip_save:
            self.save(model, len(samples))

    def print_metrics(self, metrics: [Metric]):
        print(f"initial_learning_rate: {self.initial_learning_rate:.1e}, lr_scale: {self.lr_scale}")
        # 3 columns: layer (max 3 digits), delta symbol (triangle) mean (3 decimals max), delta symbol (triangle) variance (3 decimals max)
        print("+-----+--------+-------+")
        print("|layer| Δμ     | Δσ²   |")
        print("+-----+--------+-------+")
        for m in metrics:
            delta_mean = f"{m.delta_mean:.3f}"
            delta_mean = delta_mean.rjust(6)
            print(f"|{m.layer:3d}  | {delta_mean} | {m.delta_var:.3f} |")
        print("+-----+--------+-------+")

    def save(self, model: nn.Module, num_samples: Optional[int] = None):
        self.saver.save(model, num_samples)

    def _tweak_layer(self, i: int, layer: nn.Module, samples: dict[str, torch.Tensor], lr: float) -> dict[str, torch.Tensor]:
        """
        layer: a transformer block (usually attn and mlp)

        returns a map of sample name to the corresponding quantized output from this layer
        """

        # Calculate the float outputs.
        fouts = {}
        with torch.no_grad():
            for name, sample in tqdm(samples.items(), desc=f"layer {i}: infer float outputs", disable=not VERBOSE):
                fouts[name] = layer(sample)[0].detach().clone()

        # Quantize the weights in-place.
        with torch.no_grad():
            for name, param in tqdm(layer.named_parameters(), desc=f"layer {i}: quantize", disable=not VERBOSE):
                self.quantizer.quantize(name, param)

        # layer.train() # Not sure if this is needed.

        # Freeze all weights except LayerNorm.
        # TODO: Support non-nn.LayerNorm models e.g. RMSNorm
        ln_params = []
        original_params = {}
        for mod_name, mod in layer.named_modules():
            if isinstance(mod, nn.LayerNorm):
                for param_name, param in mod.named_parameters():
                    param.requires_grad = True
                    ln_params.append(param)
                    original_params[f"{mod_name}.{param_name}"] = param.detach().clone()
            else:
                for param in mod.parameters():
                    param.requires_grad = False

        assert len(ln_params) > 0

        # Tweak loop.
        new_samples = {}
        optimizer = torch.optim.Adam(ln_params, lr=lr) # "we choose the Adam optimizer"
        # TODO: Should we do all the samples in one or several batches?
        # My guess is we want >1 batch since the paper mentions using an optimizer and
        # also setting the learning rate. My reasoning: I don't think using an optimizer
        # is useful unless we go through this loop >1 time.
        for _ in range(1): # iters
            for name, sample in tqdm(samples.items(), desc=f"layer {i}: tweak", disable=not VERBOSE):
                optimizer.zero_grad()

                qout = layer(sample)[0]
                new_samples[name] = qout.detach().clone()
                new_samples[name].requires_grad = False

                if self.skip_tweak:
                    continue

                fout = fouts[name]

                assert not fout.requires_grad
                assert qout.requires_grad
                dist = self._channelwise_distribution_loss(fout, qout)
                dist.backward()
                optimizer.step()

        # Sanity checking that things are actually working.
        # for name, param in layer.named_parameters():
        #     if param.requires_grad:
        #         # print(name, original_params.keys())
        #         # assert not torch.equal(original_params[name], param), f"param {name} was not modified"
        #         if torch.equal(original_params[name], param):
        #             print(f"{name} was not modified")
        #         else:
        #             print(f"{name} was modified")
        #         # print("req grad: ", name)


        # Record metrics.
        layer.eval() # Not sure if this is needed.
        deltas = []
        with torch.no_grad():
            for name, sample in tqdm(samples.items(), desc=f"layer {i}: metrics", disable=not VERBOSE):
                qout = layer(sample)[0]
                # Per-channel seems to be a better predictor of perplexity versus whole tensor.
                # deltas.append(fouts[name].mean() - qout.mean()) # whole tensor
                deltas.extend((fouts[name].mean(dim=-2) - qout.mean(dim=-2)).tolist()) # per-channel
        self.train_metrics.append(Metric(i, deltas))

        return new_samples

    def _channelwise_distribution_loss(self, fout: torch.Tensor, qout: torch.Tensor) -> torch.Tensor:
        # Ldist = 1/C Sum(c=1->C) ( ||μcf −μcq|| + ||(σfc)2 −(σqc)2|| )
        # https://latexeditor.lagrida.com: L_{\text{dist}}=\frac{1}{C}\sum_{c=1}^{C} \left(   \left\| \mu_{\text{f}}^{c} - \mu_{\text{q}}^{c}  \right\|_{\text{2}} + \left\| \left( \sigma_{\text{f}}^{c} \right)^{2} - \left( \sigma_{\text{q}}^{c} \right)^{2}  \right\|_{\text{2}} \right)
        C = fout.shape[-1]

        assert fout.shape == qout.shape
        assert len(fout.shape) in [2,3]

        mu_f = torch.mean(fout, dim=-2)
        mu_q = torch.mean(qout, dim=-2)
        sigma_f = torch.std(fout, dim=-2)
        sigma_q = torch.std(qout, dim=-2)

        assert mu_f.shape[-1] == C
        assert sigma_q.shape[-1] == C

        # TODO: This is a little suspicious to me. It seems like the equation wants an
        # element-wise norm -- why not write that as an absolute value?
        # It's possible this is wrong.

        # Alternate formulation (not mathematically equivalent). Doesn't seem correct to me but not sure.
        # mu_norm = torch.linalg.vector_norm(mu_f - mu_q)
        # sigma_sq_norm = torch.linalg.vector_norm(sigma_f.square() - sigma_q.square())
        # return (mu_norm + sigma_sq_norm) / C

        mu_norm_sum = (mu_f - mu_q).abs().sum()
        sigma_sq_norm_sum = (sigma_f.square() - sigma_q.square()).abs().sum()

        return (mu_norm_sum + sigma_sq_norm_sum) / C

    @torch.no_grad()
    def collect_first_layer_inputs(self, model: nn.Module, samples: dict[str, torch.Tensor]):
        """
        The initial tweaked layer takes the calibration data after it has been through
        the embedding (token + position) layer. Collect those tensors here.
        """
        latest_layer_input = None
        def hook_fn(module, args, out):
            nonlocal latest_layer_input
            latest_layer_input = args[0].detach()
            return None

        # TODO: Replace with lookup by name so we can support other models.
        # model.layers[0].register_forward_hook(hook_fn) # pythia
        hook_handle = model.h[0].register_forward_hook(hook_fn) # gpt2

        model.eval()

        first_layer_inputs = {}
        print("Collecting first layer inputs.")
        for name, sample in tqdm(samples.items()):
            model(sample)
            assert latest_layer_input is not None
            assert not torch.equal(sample, latest_layer_input)

            first_layer_inputs[name] = latest_layer_input
            latest_layer_input = None

        hook_handle.remove()

        return first_layer_inputs


# Broken out from NormTweaker class for readability.
class ModelSaver:
    """
    Saves a model after it has been norm tweaked.
    """
    def __init__(self, tweaker: NormTweaker):
        self.skip_tweak = tweaker.skip_tweak
        self.quantizer = tweaker.quantizer
        self.initial_learning_rate = tweaker.initial_learning_rate
        self.lr_scale = tweaker.lr_scale
        self.save_dir = tweaker.save_dir

    def save(self, model: nn.Module, num_samples: Optional[int] = None):
        """
        Save a model either as safetensors or a huggingface model.
        """
        try:
            from transformers import PreTrainedModel
            if isinstance(model, PreTrainedModel):
                self.save_pretrained(model, num_samples)
                return
        except ImportError:
            pass

        self.save_safetensors(model, num_samples)

    def _safetensors_metadata(self, model: nn.Module, num_samples: Optional[int]) -> Dict[str,str]:
        """
        Metadata describing how these weights were modified in a format
        compatible with safetensors (a one-level dict of str to str).
        """
        metadata = {
            f"norm_tweaking_quantizer_{k}": v for k, v in self.quantizer.metadata().items()
        }
        metadata["format"] = "pt" # Compatible with huggingface transformers (put in a folder with config files)

        if num_samples is not None:
            metadata["norm_tweaking_num_samples"] = str(num_samples)

        if self.skip_tweak:
            metadata["norm_tweaking_skip_tweak"] = str(self.skip_tweak)
        else:
            metadata["norm_tweaking_initial_learning_rate"] = str(self.initial_learning_rate)
            metadata["norm_tweaking_lr_scale"] = str(self.lr_scale)

        return metadata

    def _tweaked_model_name(self, model: nn.Module):
        file_suffix = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        return f"{model.config.model_type}_tweaked_{file_suffix}"

    def save_safetensors(self, model: nn.Module, num_samples: Optional[int] = None):
        """
        Save the tweaked model's tensors in a single safetensors file.
        Use with non-huggingface models.
        """
        filename = os.path.join(self.save_dir, f"{self._tweaked_model_name(model)}.safetensors")
        metadata =  self._safetensors_metadata(model, num_samples)

        print(f"saving: {filename}")
        print(metadata)

        save_model(model, filename, metadata)

    def save_pretrained(self, model, num_samples: Optional[int] = None):
        """
        Save the tweaked model in a format compatible with huggingface transformers.
        model: transformers.PreTrainedModel
        """
        save_dir = os.path.join(self.save_dir, self._tweaked_model_name(model))
        metadata =  self._safetensors_metadata(model, num_samples)

        assert "format" in metadata, "'format' key must be present in metadata for AutoModel to work"

        print(f"saving: {save_dir}")
        print(metadata)

        model.save_pretrained(save_dir, push_to_hub=False, safe_serialization=True)

        # transformers doesn't support safetensors' metadata, so write it to a separate file.
        with open(os.path.join(save_dir, "norm_tweaking_metadata.json"), "w") as f:
            json.dump(metadata, f)

        from transformers import AutoTokenizer, GenerationConfig
        AutoTokenizer.from_pretrained(model.config.model_type).save_pretrained(save_dir)
        GenerationConfig.from_pretrained(model.config.model_type).save_pretrained(save_dir)