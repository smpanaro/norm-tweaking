import argparse
import torch
import os
from transformers import AutoModel
from tqdm import tqdm

try:
    from sklearn.cluster import KMeans
except:
    raise ModuleNotFoundError(
        "scikit-learn is required for k-means quantization."
        " To install, run: \"pip install scikit-learn\"."
    )

def quantize_model(model_name, k, save_dir):
    model = AutoModel.from_pretrained(model_name)
    layer_names = [name for name, _ in model.named_parameters() if should_quantize(model, name)]

    for layer_name in tqdm(layer_names):
        param = dict(model.named_parameters())[layer_name]
        quantized = quantize_layer(model, layer_name, k)
        assert quantized.shape == param.shape, \
               f"Quantized shape {quantized.shape} does not match original shape {param.shape}"

        param.data = quantized

    save_file = os.path.join(save_dir, f"{model_name.replace('/', '-')}_k{k}")
    model.save_pretrained(save_file)

def should_quantize(model, layer_name):
    assert "gpt2" in model.config.model_type, "Only GPT-2 models are supported."

    if "weight" not in layer_name:
        return False

    keys = ["c_attn", "c_proj", "c_fc"]  # c_attn + c_proj = attention layer, c_proj + c_fc = mlp layer
    return any([k in layer_name for k in keys])

def quantize_layer(model, layer_name, k):
    w = model.state_dict()[layer_name]

    num_weights = w.shape.numel()
    lut_len = 1 << k
    wf = w.reshape(-1, 1) # flatten for KMeans
    wf = torch.cat((wf, torch.ones((wf.shape[0], 1))), dim=1)

    lut_len = 2 ** k
    n_clusters = min(num_weights, lut_len)

    # Same init values as used in coremltools.
    kmeans = KMeans(
        n_clusters, init="k-means++", tol=1e-2, n_init=1, random_state=0
    ).fit(wf.numpy())

    labels = kmeans.predict(wf.numpy())
    wq = kmeans.cluster_centers_[labels][:,0] # take only the weights, not the appended ones
    wq = wq.reshape(w.shape)

    return torch.from_numpy(wq).contiguous()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize model using n-bit K means")
    parser.add_argument("--model_name", type=str, help="Model name or path")
    parser.add_argument("--k", type=int, help="Number of bits to quantize the layers")
    parser.add_argument("--save_dir", type=str, help="Output directory to save the quantized model")
    # TODO:
    # parser.add_argument("--pre-normalize", action="store_true", help="Pre-normalize per-column before quantization", default=False)

    args = parser.parse_args()

    quantize_model(args.model_name, args.k, args.save_dir)
