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
        selem = np.flip(selem, axis=(-2, -1))
    elif operation == 'erosion' or operation == 'Serosion':
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=np.inf)
        result = np.full_like(image, np.inf)
        op_func = np.minimum
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


def dispatch_operation(operation: str, image: torch.Tensor, structuring_element: torch.Tensor):
    if operation == 'dilation':
        return F.dilation(image, structuring_element)
    elif operation == 'erosion':
        return F.erosion(image, structuring_element)
    elif operation == 'Sdilation':
        return F.smorph(image, structuring_element, alpha=torch.Tensor([1000.0]))
    elif operation == 'Serosion':
        return F.smorph(image, structuring_element, alpha=torch.Tensor([-1000.0]))
    else:
        raise ValueError(f"Unknown operation: {operation}")