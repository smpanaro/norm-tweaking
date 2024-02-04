import torch
from torch import nn

class Quantizer:
    """
    Base class for quantizers.
    """
    def metadata(self) -> dict[str, any]:
        """
        Return a dict of metadata that describes this quantizer's config.
        """
        return {"name": self.__class__.__name__}

    def quantize(self, name: str, param: nn.Parameter):
        """
        Modify the provided parameter by setting it's values to float16
        quantized values.

        For instance, if a parameter is quantized to 1 bit, this method should
        update the parameter so every element is a float16 0 or 1.
        """
        pass

class LinearQuantizer(Quantizer):
    def __init__(self, nbits: int, group_size: int = None):
        """
        Symmetric round-to-nearest quantizer.

        nbits: number of bits to use for quantization
        group_size: number of channels to group together for quantization
        """
        self.nbits = nbits
        self.group_size = group_size

    def metadata(self) -> dict[str, str]:
        return  {
            **super().metadata(),
            "nbits": str(self.nbits),
            "group_size": str(self.group_size),
        }

    def quantize(self, name: str, param: nn.Parameter):
        group_size = self.group_size

        if group_size is None:
            group_size = param.shape[-1]

        groups = param.split(group_size, dim=-1)
        qgroups = []
        for g in groups:
            # Symmetric round-to-nearest.
            # https://arxiv.org/pdf/2103.13630.pdf
            alpha = g.min()
            beta = g.max()
            s = (beta-alpha)/(2**self.nbits - 1)
            if s == 0:
                s = 0.001
            quantized = torch.round(g/s)
            gq = quantized * s

            assert gq.flatten().unique().shape[0] <= 2**self.nbits, "too many unique values"

            qgroups.append(gq)

        rounded = torch.cat(qgroups, dim=-1)

        # print(x[:4, :4])
        # print(rounded[:4, :4])
        # print(x.shape, rounded.shape)
        # print(keep_mask)
        # assert torch.allclose(x, rounded)

        param.data = rounded

class KMeansQuantizer(Quantizer):
    """
    Quantizer that uses k-means clustering to find the quantized values.
    """
    pass

class PreComputedQuantizer(Quantizer):
    """
    Quantizer that loads pre-computed quantized values from a file.
    Two benefits:
    - No need to re-quantize multiple times when grid searching.
    - No need to support multiple different quantization libraries here.
      Just dump the quantized values as float16 externally and do a lookup here.
    """
    def __init__(self, path: str, device):
        self.path = path
        self.device = device
        self.tensors = None

    def metadata(self) -> dict[str, str]:
        return  {
            **super().metadata(),
            "path": self.path,
        }

    def quantize(self, name: str, param: nn.Parameter):
        assert self.path.endswith(".safetensors"), "Only .safetensors files are supported."

        if self.tensors is None:
            from safetensors.torch import safe_open
            self.tensors = safe_open(self.path, framework="pt", device=self.device or "cpu")

        param.data = self.tensors.get_tensor(name)


