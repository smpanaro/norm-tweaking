from typing import Optional
import gc
import argparse

from safetensors.torch import safe_open
from normtweaking import NormTweaker, LinearQuantizer
from transformers import AutoModel, AutoModelForCausalLM

def load_samples(data_path: str, device: Optional[str]):
    samples = {}
    with safe_open(data_path, framework="pt", device=device or "cpu") as f:
        for key in f.keys():
            samples[key] = f.get_tensor(key)
            # if len(samples) >= 1:
            #     break # For debugging only.
    print(f"Loaded {len(samples)} samples from {data_path}")
    return samples

def run_tweaking(model_path: str, data_path: str, save_dir: str, device: Optional[str]):
    samples = load_samples(data_path, device)

    model = AutoModel.from_pretrained(model_path)
    print(f"Loaded huggingface model {model_path}")
    tweaker = NormTweaker(LinearQuantizer(nbits=4, group_size=1), save_dir=save_dir)
    tweaker.tweak(model, samples)

def run_gridsearch(model_path: str, data_path: str, save_dir: str, device: Optional[str], learning_rates: Optional[list[float]], scales: Optional[list[float]]):
    from lm_eval import simple_evaluate, tasks
    from lm_eval.models.huggingface import HFLM

    tasks.initialize_tasks()

    samples = load_samples(data_path, device)
    learning_rates = [None] if not learning_rates else learning_rates
    scales = [None] if not scales else scales

    print(f"Performing grid search over:\nlearning rates: {learning_rates}\nscales: {scales}")

    best_score = float('inf')
    best_params = None

    first_layer_inputs = None
    all_results = [] # (lr, scale, perplexity)

    for lr in learning_rates:
        skip_tweak = lr == 0 # Support calculating a baseline.
        for scale in scales:
            gc.collect()

            causal_model = AutoModelForCausalLM.from_pretrained(model_path)
            causal_model.to(device)
            model = causal_model.transformer

            if best_params is None:
                # Only log the first time.
                print(f"Loaded huggingface model {model_path}")

            tweaker = NormTweaker(LinearQuantizer(nbits=6, group_size=1), save_dir=save_dir, initial_learning_rate=lr, lr_scale=scale, skip_tweak=skip_tweak)

            # This is the same for all iterations, so compute it once.
            if first_layer_inputs is None:
                first_layer_inputs = tweaker.collect_first_layer_inputs(model, samples)

            tweaker.tweak(model, first_layer_inputs, samples_is_first_layer_input=True, skip_save=True)

            # Use lm_eval to evaluate the wikitext perplexity.
            eval_model = HFLM(pretrained=causal_model)
            eval_results = simple_evaluate(model=eval_model, tasks=["wikitext"], device=device)#, limit=1) # limit for development only.
            print(f"lr: {lr}, scale={scale}\neval: {eval_results['results']['wikitext']}") # Don't print full eval_results, it contains examples and is too big for colab.
            score = eval_results["results"]["wikitext"]["word_perplexity,none"]

            if score < best_score:
                best_score = score
                best_params = (lr, scale)
                tweaker.save(model, len(samples))

            all_results.append((lr, scale, score))

            # Scale doesn't matter for lr == 0, so no need to run multiple.
            if lr == 0:
                break

    print("Grid Search Results:")
    print("{:<6} {:<6} {}".format("LR", "Scale", "wikitext perplexity"))
    for result in all_results:
        print("{:<6} {:<6} {}".format(result[0], result[1], result[2]))

    best_lr, best_scale = best_params
    print(f"Best learning rate: {best_lr}, scale: {best_scale}. Model is the last model saved in {save_dir}.")

if __name__ == '__main__':
    """
    - Tweak a huggingface model using the provided dataset

    python examples/tweak_model.py --model gpt2 --data data/normtweaking_calibration_data_gpt2_1024.safetensors --save_dir models
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to huggingface model', required=True)
    parser.add_argument('--data', type=str, help='Path to generated data', required=True)
    parser.add_argument('--save_dir', type=str, help='Directory to save generated data', default='models')
    parser.add_argument('--device', type=str, help='PyTorch device', default=None)
    parser.add_argument('--grid-search-lr', nargs='*', type=float, help='List of learning rates for grid search, paper suggests starting at 1e-5', default=[])
    parser.add_argument('--grid-search-scale', nargs='*', type=float, help='List of scales for grid search', default=[])

    args = parser.parse_args()

    if len(args.grid_search_lr) > 0 or len(args.grid_search_scale) > 0:
        run_gridsearch(args.model, args.data, args.save_dir, args.device, args.grid_search_lr, args.grid_search_scale)
    else:
        run_tweaking(args.model, args.data, args.save_dir, args.device)
