# Norm Tweaking

Unofficial and in-progress implementation of [Norm Tweaking: High-performance Low-bit Quantization of Large Language Models](https://arxiv.org/abs/2309.02784). Norm Tweaking adjusts ("tweaks") the LayerNorm parameters to improve the accuracy of a quantized LLM. Many types of quantization are supported &mdash; norm tweaking is applied as a correcting step after quantization.

This repo supports:

- Calibration Data Generation - Follows LLM-QAT's synthetic data generation strategy.
- Norm Tweaking Algorithm - Adjusts layer norm parameters to improve model performance.

**NOTE**: This repo is functional but does not yet replicate the results from the paper. PRs welcome! See the [this discussion](https://github.com/smpanaro/norm-tweaking/discussions/1) or the [TODOs](#todo) for places to contribute.

# Get Started
```shell
pip install -e .
# Generate calibration data.
python examples/generate_data.py --model gpt2 --save_dir data/ --device cpu # or cuda:0
# Run the norm tweaking algorithm and save the resulting model.
python examples/tweak_model.py --model gpt2 --data data/normtweaking_calibration_data_gpt2_${TIMESTAMP}.safetensors
```
A norm-tweaked model will be saved in `models/`.

Grid search of the two tunable parameters is also supported:
```shell
# Install https://github.com/EleutherAI/lm-evaluation-harness from main (not from pip)
python examples/tweak_model.py \
  --model gpt2 \
  --data /content/data/normtweaking_calibration_data_gpt2_1024-Jan4-2024.safetensors \
  --save_dir models \
  --device cuda:0 \
  --grid-search-lr 0 0.0001 0.00001 0.000001 0.0000001 \
  --grid-search-scale 0.5 1 5 10
```
Grid search calculates wikitext perplexity (lower is better) for each setting and saves the best resulting models.

<details>
<summary>Evaluating Norm Tweaked Models</summary>
To evaluate a saved model again or on a different dataset, use lm_eval.

```shell
pip install lm-eval
lm_eval --model hf \
    --model_args pretrained=gpt2 \ # replace gpt2 with the path to your norm tweaked model
    --tasks wikitext \
    --device cuda:0 \
    --batch_size 8
```
</details>

# Results
Currently this repo has only been tested on the smallest gpt2 using 6-bit per-channel RTN quantization. With the correct hyperparameters there is a slight improvement in WikiText perplexity as reported by lm_eval.

|model|scheme               |lr  |scale|float ppl|quantized ppl|tweaked ppl|delta recovered (%)|
|--   |--                   |--  |--   |--       |--           |--         |--                 |
|gpt2†|6-bit per-channel RTN|2e-4|5    |37.3698  |37.9450      |37.8323    |0.1127 (19.5%)     |

<sub>† lm_eval reports different results than the gpt2 paper for perplexity. Using a [different method](https://huggingface.co/docs/transformers/perplexity) that nearly matches the original paper's results yields: float 29.9389, quantized 30.6000, tweaked 30.3681, delta 0.2319 (35.1%).</sub>

# TODO
- [x] Add a fast eval (delta mean, delta var like in the paper's graph)
- [x] Add a thorough eval (perplexity using lm-eval-harness)
- [ ] Generate calibration by taking language into account as described in the paper (how?)
- [ ] Support loading pre-quantized weights (gptq, awq, gguf?)
- [ ] Support non-gpt2 models (compare results with the paper)
- [ ] Try end-to-end tweaking (similar to OBD/SqueezeLLM)
- [ ] Try a larger calibration dataset (see AQLM arXiv:2401.06118)
