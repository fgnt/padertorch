import torch

__all__ = [
    'masks_to_images',
    'stft_to_images'
]


def masks_to_images(masks):
    """
    For more details of the output shape, see the tensorboardx docs.

    :param masks: Shape (frames, batch, features)
    :param format: Defines the shape of masks, normally 'tbf'.
    :return: Shape(batch, features, frames, 1)
    """
    images = torch.clamp(masks * 255, 0, 255)
    images = images.type(torch.ByteTensor)
    return images[0].cpu().numpy().transpose(1, 0)[::-1]


def stft_to_images(signal):
    """
    For more details of the output shape, see the tensorboardx docs.

    :param masks: Shape (frames, batch, features)
    :param format: Defines the shape of masks, normally 'tbf'.
    :return: Shape(batch, features, frames, 1)
    """
    return masks_to_images(signal / torch.max(torch.abs(signal)))
