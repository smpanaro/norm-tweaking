from normtweaking import DataGenerator
import argparse
from typing import Optional

def run_generate(model_name_or_path, save_dir: str, device: Optional[str], print_results: bool):
    generator = DataGenerator.from_pretrained(model_name_or_path)
    if device:
        generator.to(device)

    data = generator.generate(n_samples=128)
    if print_results:
        for d in data.tensors:
            for s in generator.tokenizer.batch_decode(d):
                print(s)
    else:
        data.save(save_dir)

if __name__ == '__main__':
    """
    - Generate a data set for the provided model.

    python examples/generate_data.py --model gpt2
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to huggingface model', required=True)
    parser.add_argument('--save_dir', type=str, help='Directory to save generated data', default='data')
    parser.add_argument('--device', type=str, help='PyTorch device', default=None)
    parser.add_argument('--print', action='store_true', help='Print generated data to stdout without saving')

    args = parser.parse_args()

    run_generate(args.model, args.save_dir, args.device, args.print)