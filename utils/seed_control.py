"""
Seed control for reproducibility across all experiments

Sets random seeds for Python, NumPy, and PyTorch to ensure
reproducible results across training runs.
"""
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Set random seeds for reproducibility
    
    This function sets seeds for:
    - Python's random module
    - NumPy's random number generator
    - PyTorch (CPU and GPU/MPS)
    
    Args:
        seed (int): Random seed value. Default is 42.
        
    Example:
        >>> from utils.seed_control import set_seed
        >>> set_seed(42)  # Set seed at start of training script
        >>> # Now all random operations will be reproducible
    
    Note:
        - Call this function at the very beginning of your training script
        - Use the same seed value across runs for reproducibility
        - Some operations may still be non-deterministic on GPU/MPS
    """
    # Python random module
    random.seed(seed)
    
    # NumPy random number generator
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # PyTorch MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # Make CUDA operations deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed} for reproducibility")


def get_rng_state():
    """
    Get the current state of all random number generators
    
    Returns:
        dict: Dictionary containing RNG states for Python, NumPy, and PyTorch
        
    Example:
        >>> state = get_rng_state()
        >>> # ... do some random operations ...
        >>> set_rng_state(state)  # Restore to previous state
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    
    if torch.backends.mps.is_available():
        state['torch_mps'] = torch.mps.get_rng_state()
    
    return state


def set_rng_state(state):
    """
    Restore random number generator states
    
    Args:
        state (dict): Dictionary of RNG states from get_rng_state()
        
    Example:
        >>> state = get_rng_state()
        >>> # ... do some random operations ...
        >>> set_rng_state(state)  # Restore to previous state
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    
    if 'torch_cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    
    if 'torch_mps' in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state['torch_mps'])
    
    print("Random number generator state restored")


if __name__ == "__main__":
    # Test reproducibility
    print("Testing seed control...")
    
    # Test 1: Same seed should give same results
    set_seed(42)
    result1 = torch.randn(5)
    
    set_seed(42)
    result2 = torch.randn(5)
    
    assert torch.allclose(result1, result2), "Results should be identical!"
    print("Test 1 passed: Same seed gives same results")
    
    # Test 2: Different seeds give different results
    set_seed(42)
    result1 = torch.randn(5)
    
    set_seed(123)
    result2 = torch.randn(5)
    
    assert not torch.allclose(result1, result2), "Results should be different!"
    print("Test 2 passed: Different seeds give different results")
    
    # Test 3: State saving/restoring
    set_seed(42)
    state = get_rng_state()
    original = torch.randn(5)
    
    # Generate some random numbers (changes state)
    _ = torch.randn(100)
    
    # Restore state
    set_rng_state(state)
    restored = torch.randn(5)
    
    assert torch.allclose(original, restored), "State restoration should work!"
    print("Test 3 passed: State save/restore works")
    
    print("\nAll seed control tests passed!")
    print("Usage: from utils.seed_control import set_seed; set_seed(42)")
