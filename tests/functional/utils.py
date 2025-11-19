import numpy as np
import torch
import morphocore.functional as F


def correct_morphology(image, selem, operation='dilation'):
    """
    Generic morphological operation that supports both dilation and erosion.
    
    Args:
        image: Input image
        selem: Structuring element
        operation: 'dilation' or 'erosion'
    """
    pad_h, pad_w = selem.shape[0]//2, selem.shape[1]//2
    
    if operation == 'dilation' or operation == 'Sdilation':
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=-np.inf)
        result = np.full_like(image, -np.inf)
        op_func = np.maximum

    elif operation == 'erosion' or operation == 'Serosion':
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=np.inf)
        result = np.full_like(image, np.inf)
        op_func = np.minimum
        selem = np.flip(selem, axis=(-2, -1))
    else:
        raise ValueError(f"Unknown operation: {operation}")

    for dy in range(selem.shape[0]):
        for dx in range(selem.shape[1]):
            selem_value = selem[dy, dx]
            
            offset_y = dy - pad_h
            offset_x = dx - pad_w
            
            y_start, y_end = pad_h - offset_y, pad_h - offset_y + image.shape[0]
            x_start, x_end = pad_w - offset_x, pad_w - offset_x + image.shape[1]
            
            shifted_region = padded[y_start:y_end, x_start:x_end]
            
            if operation in ['dilation', 'Sdilation']:
                candidate = shifted_region + selem_value
            else:
                candidate = shifted_region - selem_value
            
            result = op_func(result, candidate)
    
    return result


def dispatch_operation(operation: str, image: torch.Tensor, structuring_element: torch.Tensor, channel_merge_mode: str = 'sum', save_indices: bool = True) -> torch.Tensor:
    input_was_2d = image.ndim == 2
    
    if input_was_2d:
        in_channel = 1
        out_channel = 1
    else:
        in_channel = image.shape[1]
        out_channel = structuring_element.shape[0]
    
    if operation == 'dilation':
        return F.dilation(x=image, w=structuring_element, channel_merge_mode=channel_merge_mode, save_indices=save_indices)
    elif operation == 'erosion':
        return F.erosion(x=image, w=structuring_element, channel_merge_mode=channel_merge_mode, save_indices=save_indices)
    elif operation == 'Sdilation':
        if channel_merge_mode == 'identity':
            return F.smorph(x=image, w=structuring_element, alpha=torch.full((out_channel,), 1000, dtype=image.dtype).to(image.device), channel_merge_mode=channel_merge_mode)
        return F.smorph(x=image, w=structuring_element, alpha=torch.full((out_channel, in_channel), 1000, dtype=image.dtype).to(image.device), channel_merge_mode=channel_merge_mode)
    elif operation == 'Serosion':
        if channel_merge_mode == 'identity':
            return F.smorph(x=image, w=structuring_element, alpha=torch.full((out_channel,), -1000, dtype=image.dtype).to(image.device), channel_merge_mode=channel_merge_mode)
        return F.smorph(x=image, w=structuring_element, alpha=torch.full((out_channel, in_channel), -1000, dtype=image.dtype).to(image.device), channel_merge_mode=channel_merge_mode)
    else:
        raise ValueError(f"Unknown operation: {operation}")