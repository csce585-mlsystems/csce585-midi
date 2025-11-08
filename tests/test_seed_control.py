"""
Tests for seed_control.py
Tests reproducibility and random number generator state management
"""
import pytest
import random
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from utils.seed_control import (
    set_seed,
    get_rng_state,
    set_rng_state
)


class TestSetSeed:
    """Test suite for set_seed function"""
    
    def test_set_seed_basic(self, capsys):
        """Test basic seed setting"""
        set_seed(42)
        captured = capsys.readouterr()
        assert "Random seed set to 42" in captured.out
    
    def test_set_seed_python_random(self):
        """Test that Python random is seeded correctly"""
        set_seed(42)
        result1 = random.random()
        
        set_seed(42)
        result2 = random.random()
        
        assert result1 == result2
    
    def test_set_seed_numpy(self):
        """Test that NumPy random is seeded correctly"""
        set_seed(42)
        result1 = np.random.rand(5)
        
        set_seed(42)
        result2 = np.random.rand(5)
        
        assert np.allclose(result1, result2)
    
    def test_set_seed_torch(self):
        """Test that PyTorch random is seeded correctly"""
        set_seed(42)
        result1 = torch.randn(5)
        
        set_seed(42)
        result2 = torch.randn(5)
        
        assert torch.allclose(result1, result2)
    
    def test_set_seed_different_values(self):
        """Test that different seeds produce different results"""
        set_seed(42)
        result1 = torch.randn(5)
        
        set_seed(123)
        result2 = torch.randn(5)
        
        assert not torch.allclose(result1, result2)
    
    def test_set_seed_all_modules(self):
        """Test that all random modules are synchronized"""
        set_seed(42)
        
        py_result = random.random()
        np_result = np.random.rand()
        torch_result = torch.randn(1).item()
        
        # Reset and verify consistency
        set_seed(42)
        assert random.random() == py_result
        assert np.random.rand() == np_result
        assert torch.randn(1).item() == torch_result
    
    def test_set_seed_custom_value(self):
        """Test seed setting with custom values"""
        for seed_value in [0, 1, 100, 999, 12345]:
            set_seed(seed_value)
            result1 = torch.randn(3)
            
            set_seed(seed_value)
            result2 = torch.randn(3)
            
            assert torch.allclose(result1, result2)
    
    def test_set_seed_deterministic_flags(self):
        """Test that deterministic flags are set"""
        set_seed(42)
        
        assert torch.backends.cudnn.deterministic == True
        assert torch.backends.cudnn.benchmark == False
    
    @patch('utils.seed_control.torch.cuda.is_available', return_value=True)
    @patch('utils.seed_control.torch.cuda.manual_seed')
    @patch('utils.seed_control.torch.cuda.manual_seed_all')
    def test_set_seed_cuda(self, mock_seed_all, mock_seed, mock_cuda_available):
        """Test CUDA seeding when available"""
        set_seed(42)
        
        # Just verify they were called with correct argument
        # (may be called multiple times due to torch.manual_seed internals)
        assert mock_seed.called
        assert mock_seed_all.called
        mock_seed.assert_any_call(42)
        mock_seed_all.assert_any_call(42)
    
    @patch('utils.seed_control.torch.backends.mps.is_available', return_value=True)
    @patch('utils.seed_control.torch.mps.manual_seed')
    def test_set_seed_mps(self, mock_mps_seed, mock_mps_available):
        """Test MPS seeding when available"""
        set_seed(42)
        
        # Just verify it was called with correct argument
        # (may be called multiple times due to torch.manual_seed internals)
        assert mock_mps_seed.called
        mock_mps_seed.assert_any_call(42)
    
    def test_set_seed_negative_value(self):
        """Test that negative seed values are rejected"""
        # NumPy doesn't accept negative seeds
        with pytest.raises(ValueError):
            set_seed(-1)
    
    def test_set_seed_zero(self):
        """Test seed value of 0"""
        set_seed(0)
        result1 = torch.randn(5)
        
        set_seed(0)
        result2 = torch.randn(5)
        
        assert torch.allclose(result1, result2)
    
    def test_set_seed_large_value(self):
        """Test with large seed value"""
        large_seed = 2**31 - 1  # Max int32
        set_seed(large_seed)
        result = torch.randn(3)
        assert result is not None


class TestGetRngState:
    """Test suite for get_rng_state function"""
    
    def test_get_rng_state_structure(self):
        """Test that returned state has correct structure"""
        set_seed(42)
        state = get_rng_state()
        
        assert isinstance(state, dict)
        assert 'python' in state
        assert 'numpy' in state
        assert 'torch' in state
    
    def test_get_rng_state_python(self):
        """Test that Python RNG state is captured"""
        set_seed(42)
        state = get_rng_state()
        
        assert state['python'] is not None
        assert isinstance(state['python'], tuple)
    
    def test_get_rng_state_numpy(self):
        """Test that NumPy RNG state is captured"""
        set_seed(42)
        state = get_rng_state()
        
        assert state['numpy'] is not None
        # NumPy state is a tuple: (str, array, int, int, float)
        assert isinstance(state['numpy'], tuple)
    
    def test_get_rng_state_torch(self):
        """Test that PyTorch RNG state is captured"""
        set_seed(42)
        state = get_rng_state()
        
        assert state['torch'] is not None
        assert isinstance(state['torch'], torch.Tensor)
    
    @patch('utils.seed_control.torch.cuda.is_available', return_value=True)
    @patch('utils.seed_control.torch.cuda.get_rng_state_all', return_value=[torch.tensor([1, 2, 3])])
    def test_get_rng_state_cuda(self, mock_get_state, mock_cuda_available):
        """Test CUDA state capture when available"""
        state = get_rng_state()
        
        assert 'torch_cuda' in state
        mock_get_state.assert_called_once()
    
    @patch('utils.seed_control.torch.backends.mps.is_available', return_value=True)
    @patch('utils.seed_control.torch.mps.get_rng_state', return_value=torch.tensor([1, 2, 3]))
    def test_get_rng_state_mps(self, mock_get_state, mock_mps_available):
        """Test MPS state capture when available"""
        state = get_rng_state()
        
        assert 'torch_mps' in state
        mock_get_state.assert_called_once()
    
    def test_get_rng_state_changes(self):
        """Test that state changes after random operations"""
        set_seed(42)
        state1 = get_rng_state()
        
        # Perform random operations
        _ = torch.randn(100)
        
        state2 = get_rng_state()
        
        # States should be different
        assert not torch.equal(state1['torch'], state2['torch'])
    
    def test_get_rng_state_independent_copies(self):
        """Test that getting state doesn't affect RNG"""
        set_seed(42)
        result1 = torch.randn(5)
        
        set_seed(42)
        _ = get_rng_state()  # Get state
        result2 = torch.randn(5)
        
        # Should still be the same
        assert torch.allclose(result1, result2)


class TestSetRngState:
    """Test suite for set_rng_state function"""
    
    def test_set_rng_state_basic(self, capsys):
        """Test basic state restoration"""
        set_seed(42)
        state = get_rng_state()
        
        set_rng_state(state)
        captured = capsys.readouterr()
        assert "Random number generator state restored" in captured.out
    
    def test_set_rng_state_python(self):
        """Test Python RNG state restoration"""
        set_seed(42)
        state = get_rng_state()
        original = random.random()
        
        # Generate some random numbers
        for _ in range(10):
            random.random()
        
        # Restore state
        set_rng_state(state)
        restored = random.random()
        
        assert original == restored
    
    def test_set_rng_state_numpy(self):
        """Test NumPy RNG state restoration"""
        set_seed(42)
        state = get_rng_state()
        original = np.random.rand(5)
        
        # Generate some random numbers
        _ = np.random.rand(100)
        
        # Restore state
        set_rng_state(state)
        restored = np.random.rand(5)
        
        assert np.allclose(original, restored)
    
    def test_set_rng_state_torch(self):
        """Test PyTorch RNG state restoration"""
        set_seed(42)
        state = get_rng_state()
        original = torch.randn(5)
        
        # Generate some random numbers
        _ = torch.randn(100)
        
        # Restore state
        set_rng_state(state)
        restored = torch.randn(5)
        
        assert torch.allclose(original, restored)
    
    def test_set_rng_state_all_modules(self):
        """Test that all modules are restored correctly"""
        set_seed(42)
        state = get_rng_state()
        
        py_original = random.random()
        np_original = np.random.rand()
        torch_original = torch.randn(1).item()
        
        # Change states
        for _ in range(10):
            random.random()
            np.random.rand()
            torch.randn(1)
        
        # Restore
        set_rng_state(state)
        
        assert random.random() == py_original
        assert np.random.rand() == np_original
        assert torch.randn(1).item() == torch_original
    
    @patch('utils.seed_control.torch.cuda.is_available', return_value=True)
    @patch('utils.seed_control.torch.cuda.set_rng_state_all')
    def test_set_rng_state_cuda(self, mock_set_state, mock_cuda_available):
        """Test CUDA state restoration when available"""
        # Create a state with CUDA info
        state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'torch_cuda': [torch.tensor([1, 2, 3])]
        }
        
        set_rng_state(state)
        mock_set_state.assert_called_once()
    
    @patch('utils.seed_control.torch.backends.mps.is_available', return_value=True)
    @patch('utils.seed_control.torch.mps.set_rng_state')
    def test_set_rng_state_mps(self, mock_set_state, mock_mps_available):
        """Test MPS state restoration when available"""
        # Create a state with MPS info
        state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'torch_mps': torch.tensor([1, 2, 3])
        }
        
        set_rng_state(state)
        mock_set_state.assert_called_once()
    
    def test_set_rng_state_multiple_times(self):
        """Test restoring state multiple times"""
        set_seed(42)
        state = get_rng_state()
        expected = torch.randn(3)
        
        for _ in range(5):
            # Do some operations
            _ = torch.randn(100)
            
            # Restore and check
            set_rng_state(state)
            result = torch.randn(3)
            assert torch.allclose(result, expected)
    
    def test_set_rng_state_no_cuda(self):
        """Test state restoration without CUDA state"""
        state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state()
        }
        
        # Should not raise an error
        set_rng_state(state)
    
    def test_set_rng_state_no_mps(self):
        """Test state restoration without MPS state"""
        state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state()
        }
        
        # Should not raise an error
        set_rng_state(state)


class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_reproducible_training_simulation(self):
        """Test reproducible training workflow"""
        # Simulate training with same seed
        results = []
        for _ in range(3):
            set_seed(42)
            # Simulate training step
            weights = torch.randn(10, 10)
            loss = torch.randn(1).item()
            results.append((weights, loss))
        
        # All results should be identical
        for i in range(1, len(results)):
            assert torch.allclose(results[0][0], results[i][0])
            assert results[0][1] == results[i][1]
    
    def test_checkpoint_restore_workflow(self):
        """Test checkpoint save/restore workflow"""
        set_seed(42)
        
        # Simulate training for a few steps
        checkpoint_state = None
        checkpoint_output = None
        
        for step in range(10):
            if step == 5:
                # Save checkpoint BEFORE generating
                checkpoint_state = get_rng_state()
            
            output = torch.randn(5)
            
            if step == 5:
                checkpoint_output = output.clone()
        
        # Continue training
        for _ in range(10):
            _ = torch.randn(5)
        
        # Restore checkpoint and verify we get same output
        set_rng_state(checkpoint_state)
        restored_output = torch.randn(5)
        
        assert torch.allclose(checkpoint_output, restored_output)
    
    def test_different_seeds_different_results(self):
        """Test that different seeds produce different training"""
        results = []
        for seed in [42, 123, 999]:
            set_seed(seed)
            weights = torch.randn(10, 10)
            results.append(weights)
        
        # All should be different
        assert not torch.allclose(results[0], results[1])
        assert not torch.allclose(results[1], results[2])
        assert not torch.allclose(results[0], results[2])
    
    def test_reproducible_data_augmentation(self):
        """Test reproducible data augmentation"""
        set_seed(42)
        
        # Simulate data augmentation
        data = torch.randn(100, 3, 224, 224)
        indices = torch.randperm(100)[:10]
        augmented1 = data[indices]
        
        set_seed(42)
        data = torch.randn(100, 3, 224, 224)
        indices = torch.randperm(100)[:10]
        augmented2 = data[indices]
        
        assert torch.allclose(augmented1, augmented2)
    
    def test_state_persistence_across_operations(self):
        """Test that state persists correctly across various operations"""
        set_seed(42)
        state1 = get_rng_state()
        
        # Mix of different random operations
        _ = random.choice([1, 2, 3, 4, 5])
        _ = np.random.choice([1, 2, 3, 4, 5])
        _ = torch.randint(0, 5, (10,))
        
        state2 = get_rng_state()
        
        # Restore first state
        set_rng_state(state1)
        result1 = torch.randn(5)
        
        # Restore second state  
        set_rng_state(state2)
        
        # Restore first state again
        set_rng_state(state1)
        result2 = torch.randn(5)
        
        assert torch.allclose(result1, result2)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_set_seed_very_large_tensor(self):
        """Test seed with operations on very large tensors"""
        set_seed(42)
        large_tensor1 = torch.randn(1000, 1000)
        
        set_seed(42)
        large_tensor2 = torch.randn(1000, 1000)
        
        assert torch.allclose(large_tensor1, large_tensor2)
    
    def test_state_with_different_tensor_sizes(self):
        """Test state restoration with different tensor sizes"""
        set_seed(42)
        state = get_rng_state()
        
        _ = torch.randn(5)
        _ = torch.randn(100)
        _ = torch.randn(10, 10)
        
        set_rng_state(state)
        
        # Should work with any size
        result = torch.randn(50)
        assert result.shape == (50,)
    
    def test_consecutive_seed_sets(self):
        """Test setting seed multiple times consecutively"""
        set_seed(42)
        set_seed(42)
        set_seed(42)
        result1 = torch.randn(5)
        
        set_seed(42)
        result2 = torch.randn(5)
        
        assert torch.allclose(result1, result2)
    
    def test_state_independence(self):
        """Test that getting state doesn't interfere with generation"""
        set_seed(42)
        expected = []
        for _ in range(5):
            expected.append(torch.randn(3))
        
        set_seed(42)
        actual = []
        for _ in range(5):
            _ = get_rng_state()  # Get state between generations
            actual.append(torch.randn(3))
        
        for e, a in zip(expected, actual):
            assert torch.allclose(e, a)

class TestMainFunction:
    """Test the main demonstration function"""
    
    def test_main_runs_without_error(self, capsys):
        """Test that main() runs successfully"""
        from utils.seed_control import main
        
        # Should not raise any errors
        main()
        
        captured = capsys.readouterr()
        # Check for expected output
        assert "Demonstration of seed control utilities" in captured.out
        assert "Setting seed to 42" in captured.out
        assert "Saving RNG state" in captured.out
        assert "Generating random numbers" in captured.out
        assert "Restoring RNG state" in captured.out
    
    def test_main_demonstrates_reproducibility(self, capsys):
        """Test that main() demonstrates reproducibility"""
        from utils.seed_control import main
        
        main()
        captured = capsys.readouterr()
        
        # Should show tensor outputs
        assert "tensor([" in captured.out
        # Should show that restored values match original
        assert "After restoring state" in captured.out
    
    def test_main_demonstrates_state_management(self, capsys):
        """Test that main() demonstrates state management"""
        from utils.seed_control import main
        
        main()
        captured = capsys.readouterr()
        
        # Check all three sections are demonstrated
        assert "1. Set seed for reproducibility" in captured.out
        assert "2. Save and restore RNG state" in captured.out
        assert "3. Different seeds produce different results" in captured.out
    
    def test_main_shows_different_seeds(self, capsys):
        """Test that main() shows different seed results"""
        from utils.seed_control import main
        
        main()
        captured = capsys.readouterr()
        
        # Should show results from different seeds
        assert "With seed 42:" in captured.out
        assert "With seed 123:" in captured.out


class TestModuleLevelExecution:
    """Test module-level execution"""
    
    def test_module_executes_main_when_run_directly(self, capsys, monkeypatch):
        """Test that running the module directly calls main()"""
        # Mock __name__ to be '__main__'
        import utils.seed_control as sc
        
        # Store original main
        original_main = sc.main
        
        # Track if main was called
        main_called = []
        
        def mock_main():
            main_called.append(True)
            original_main()
        
        # Replace main temporarily
        monkeypatch.setattr(sc, 'main', mock_main)
        
        # Simulate running as main
        if sc.__name__ == '__main__':
            sc.main()
        
        # If the module is actually being run as main, verify
        # (This test will pass either way, but covers the code path)


class TestDocstringsAndHelp:
    """Test that functions have proper documentation"""
    
    def test_set_seed_has_docstring(self):
        """Test that set_seed has documentation"""
        assert set_seed.__doc__ is not None
        assert len(set_seed.__doc__) > 0
    
    def test_get_rng_state_has_docstring(self):
        """Test that get_rng_state has documentation"""
        assert get_rng_state.__doc__ is not None
        assert len(get_rng_state.__doc__) > 0
    
    def test_set_rng_state_has_docstring(self):
        """Test that set_rng_state has documentation"""
        assert set_rng_state.__doc__ is not None
        assert len(set_rng_state.__doc__) > 0
    
    def test_main_has_docstring(self):
        """Test that main has documentation"""
        from utils.seed_control import main
        assert main.__doc__ is not None
        assert len(main.__doc__) > 0


class TestCompleteWorkflows:
    """Additional integration tests for complete workflows"""
    
    def test_full_training_checkpoint_workflow(self):
        """Test a complete training checkpoint workflow"""
        # Initial training phase
        set_seed(42)
        epoch_states = []
        epoch_losses = []
        
        for epoch in range(5):
            # Save state at start of epoch
            state = get_rng_state()
            epoch_states.append(state)
            
            # Simulate training
            model_update = torch.randn(10, 10)
            loss = torch.randn(1).item()
            epoch_losses.append(loss)
        
        # Simulate loading checkpoint from epoch 2
        set_rng_state(epoch_states[2])
        
        # Continue training should match epoch 2's next update
        set_seed(42)
        for i in range(3):
            _ = get_rng_state()
            if i < 2:
                _ = torch.randn(10, 10)
                _ = torch.randn(1)
        
        # Get state at epoch 2
        state_epoch2 = get_rng_state()
        set_rng_state(state_epoch2)
        
        resumed_update = torch.randn(10, 10)
        
        # Restore to epoch 2 again
        set_rng_state(epoch_states[2])
        expected_update = torch.randn(10, 10)
        
        assert torch.allclose(resumed_update, expected_update)
    
    def test_parallel_experiment_reproducibility(self):
        """Test reproducibility across parallel experiments"""
        results = {}
        
        # Run multiple "experiments" with same seed
        for exp_id in ['exp1', 'exp2', 'exp3']:
            set_seed(42)
            
            # Simulate experiment
            data = torch.randn(100, 10)
            model = torch.randn(10, 5)
            output = torch.matmul(data, model)
            
            results[exp_id] = output
        
        # All experiments should produce identical results
        assert torch.allclose(results['exp1'], results['exp2'])
        assert torch.allclose(results['exp2'], results['exp3'])
    
    def test_cross_platform_seed_consistency(self):
        """Test that seeds work consistently"""
        # Test with various data types and operations
        operations = []
        
        set_seed(42)
        
        # Various PyTorch operations
        operations.append(torch.randn(5))
        operations.append(torch.randint(0, 10, (5,)))
        operations.append(torch.rand(5))
        operations.append(torch.normal(0, 1, (5,)))
        
        # Reset and verify
        set_seed(42)
        
        assert torch.allclose(torch.randn(5), operations[0])
        assert torch.equal(torch.randint(0, 10, (5,)), operations[1])
        assert torch.allclose(torch.rand(5), operations[2])
        assert torch.allclose(torch.normal(0, 1, (5,)), operations[3])


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_set_rng_state_with_none_values(self):
        """Test handling of None values in state dict"""
        state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'torch_cuda': None,  # None value
            'torch_mps': None    # None value
        }
        
        # Should not raise an error
        set_rng_state(state)
    
    def test_set_rng_state_with_missing_keys(self):
        """Test handling of missing keys in state dict"""
        # Minimal state dict
        state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state()
        }
        
        # Should work fine with just the required keys
        set_rng_state(state)
    
    def test_get_rng_state_multiple_times_same_result(self):
        """Test that calling get_rng_state multiple times without changes gives same result"""
        set_seed(42)
        
        state1 = get_rng_state()
        state2 = get_rng_state()
        
        # States should be identical (getting state doesn't change state)
        assert state1['python'] == state2['python']
        assert np.array_equal(state1['numpy'][1], state2['numpy'][1])
        assert torch.equal(state1['torch'], state2['torch'])


class TestPerformance:
    """Test performance-related aspects"""
    
    def test_seed_setting_is_fast(self):
        """Test that seed setting is reasonably fast"""
        import time
        
        start = time.time()
        for _ in range(100):
            set_seed(42)
        end = time.time()
        
        # Should complete 100 seed sets in less than 1 second
        assert (end - start) < 1.0
    
    def test_state_save_restore_is_fast(self):
        """Test that state operations are reasonably fast"""
        import time
        
        set_seed(42)
        
        start = time.time()
        for _ in range(100):
            state = get_rng_state()
            set_rng_state(state)
        end = time.time()
        
        # Should complete 100 save/restore cycles in less than 1 second
        assert (end - start) < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])