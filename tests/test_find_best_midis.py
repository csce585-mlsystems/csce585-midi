import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path
from utils.find_best_midis import find_best_midis

@pytest.fixture
def mock_generation_data():
    return pd.DataFrame({
        'output_file': ['/path/to/file1.mid', '/path/to/file2.mid', '/path/to/file3.mid'],
        'strategy': ['greedy', 'top_p', 'top_p'],
        'temperature': [1.0, 1.5, 0.8],
        'dataset': ['dataset1', 'dataset1', 'dataset2']
    })

@pytest.fixture
def mock_evaluation_data():
    return pd.DataFrame({
        'output_file': ['/path/to/file1.mid', '/path/to/file2.mid', '/path/to/file3.mid'],
        'pitch_range': [20, 40, 30],
        'note_density': [5.0, 8.0, 2.0],
        'duration': [30.0, 60.0, 45.0],
        'dataset': ['dataset1', 'dataset1', 'dataset2']
    })

@patch('utils.find_best_midis.Path.glob')
def test_no_logs_found(mock_glob, capsys):
    # Setup mock to return empty list
    mock_glob.return_value = []
    
    find_best_midis()
    
    captured = capsys.readouterr()
    assert "No log files found!" in captured.out

@patch('utils.find_best_midis.shutil.copy2')
@patch('utils.find_best_midis.Path.exists')
@patch('utils.find_best_midis.pd.read_csv')
@patch('utils.find_best_midis.Path.glob')
def test_find_best_midis_success(mock_glob, mock_read_csv, mock_exists, mock_copy, mock_generation_data, mock_evaluation_data, tmp_path):
    # Setup mocks
    mock_gen_path = MagicMock()
    mock_gen_path.parent.parent.name = "dataset1"
    
    mock_eval_path = MagicMock()
    mock_eval_path.parent.parent.name = "dataset1"

    # Mock glob to return one generation file and one evaluation file
    mock_glob.side_effect = [[mock_gen_path], [mock_eval_path]]
    
    # Mock read_csv to return dataframes
    # The function reads generation files first, then evaluation files
    mock_read_csv.side_effect = [mock_generation_data, mock_evaluation_data]
    
    # Mock file existence check
    mock_exists.return_value = True
    
    # Run function with a temporary output directory
    output_dir = tmp_path / "best_midis"
    find_best_midis(output_dir=str(output_dir), top_n=2)
    
    # Verify copy was called
    assert mock_copy.call_count == 2
    
    # Verify the logic for scoring (implicit in the result order)
    # file2 should be top ranked: 
    #   strategy=top_p (bonus 1.5)
    #   temp=1.5 (bonus 1.2)
    #   pitch_range=40 (max) -> score 1.0
    #   duration=60 (max) -> score 1.0
    #   density=8.0 (median is 5.0, std is approx 3.0) -> score around 0
    
    # file1: greedy (1.0), temp 1.0 (1.0), pitch 20 (0.5), dur 30 (0.5)
    
    # We can check the arguments to copy2 to see which files were copied
    # The first call should be for the highest ranked file
    
    # Extract the source file path from the first call args
    first_call_args = mock_copy.call_args_list[0]
    src_path = first_call_args[0][0]
    assert str(src_path) == '/path/to/file2.mid'

@patch('utils.find_best_midis.pd.read_csv')
@patch('utils.find_best_midis.Path.glob')
def test_empty_merge(mock_glob, mock_read_csv, capsys):
    # Setup mocks to return dataframes that don't match on output_file
    mock_gen_path = MagicMock()
    mock_gen_path.parent.parent.name = "dataset1"
    
    mock_eval_path = MagicMock()
    mock_eval_path.parent.parent.name = "dataset1"

    mock_glob.side_effect = [[mock_gen_path], [mock_eval_path]]
    
    df_gen = pd.DataFrame({'output_file': ['a'], 'dataset': ['d1']})
    df_eval = pd.DataFrame({'output_file': ['b'], 'dataset': ['d1']})
    
    mock_read_csv.side_effect = [df_gen, df_eval]
    
    find_best_midis()
    
    captured = capsys.readouterr()
    assert "No matching generation and evaluation data found!" in captured.out
