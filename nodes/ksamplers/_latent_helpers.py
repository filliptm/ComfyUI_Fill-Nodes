import torch

import comfy.nested_tensor


def primary_tensor(samples):
    if isinstance(samples, comfy.nested_tensor.NestedTensor):
        tensors = samples.unbind()
        if not tensors:
            raise ValueError("Nested latent has no tensors.")
        return tensors[0]
    if isinstance(samples, torch.Tensor):
        return samples
    raise TypeError("Latent samples must be a tensor or nested tensor.")


def replace_primary_tensor(samples, primary):
    if isinstance(samples, comfy.nested_tensor.NestedTensor):
        return comfy.nested_tensor.NestedTensor((primary, *samples.unbind()[1:]))
    return primary


def primary_only_noise_mask(samples, primary_mask=None):
    if not isinstance(samples, comfy.nested_tensor.NestedTensor):
        return primary_mask

    tensors = samples.unbind()
    if isinstance(primary_mask, comfy.nested_tensor.NestedTensor):
        primary_mask = primary_mask.unbind()[0]
    if primary_mask is None:
        primary_mask = torch.ones_like(tensors[0])
    return comfy.nested_tensor.NestedTensor(
        (primary_mask, *(torch.zeros_like(tensor) for tensor in tensors[1:]))
    )


def slice_batch(value, index):
    if isinstance(value, (torch.Tensor, comfy.nested_tensor.NestedTensor)):
        return value[index:index + 1]
    return value
