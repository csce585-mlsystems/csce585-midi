import pytest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import numpy as np
from pathlib import Path
from utils.analyze_logs import LogAnalyzer

@pytest.fixture
def mock_generator_logs():
    return pd.DataFrame({
        'model_type': ['lstm', 'transformer', 'lstm'],
        'dataset_name': ['dataset1', 'dataset1', 'dataset2'],
        'final_loss': [0.5, 0.4, 0.6],
        'min_loss': [0.4, 0.3, 0.5],
        'train_time_sec': [100, 200, 150],
        'num_params': [1000, 2000, 1000]
    })

@pytest.fixture
def mock_generation_logs():
    return pd.DataFrame({
        'output_file': ['file1.mid', 'file2.mid', 'file3.mid'],
        'dataset_name': ['dataset1', 'dataset1', 'dataset2'],
        'strategy': ['greedy', 'top_p', 'top_p'],
        'temperature': [1.0, 1.5, 0.8]
    })

@pytest.fixture
def mock_evaluation_logs():
    return pd.DataFrame({
        'output_file': ['file1.mid', 'file2.mid', 'file3.mid'],
        'dataset_name': ['dataset1', 'dataset1', 'dataset2'],
        'note_density': [5.0, 8.0, 2.0],
        'pitch_range': [20, 40, 30],
        'duration': [30.0, 60.0, 45.0],
        'avg_polyphony': [2.0, 3.0, 1.5],
        'num_notes': [150, 480, 90]
    })

@pytest.fixture
def mock_discriminator_logs():
    return pd.DataFrame({
        'model_type': ['lstm', 'lstm', 'transformer', 'transformer'],
        'epoch': [1, 2, 1, 2],
        'micro_f1': [0.6, 0.7, 0.5, 0.6],
        'micro_precision': [0.6, 0.7, 0.5, 0.6],
        'micro_recall': [0.6, 0.7, 0.5, 0.6],
        'train_loss': [0.8, 0.6, 0.9, 0.7],
        'context': [4, 4, 8, 8]
    })

@pytest.fixture
def analyzer(mock_generator_logs, mock_generation_logs, mock_evaluation_logs, mock_discriminator_logs):
    analyzer = LogAnalyzer()
    analyzer.generator_logs = mock_generator_logs
    analyzer.generation_logs = mock_generation_logs
    analyzer.evaluation_logs = mock_evaluation_logs
    analyzer.discriminator_logs = mock_discriminator_logs
    return analyzer

def test_initialization():
    analyzer = LogAnalyzer(logs_dir="test_logs")
    assert analyzer.logs_dir == Path("test_logs")
    assert analyzer.generator_logs.empty
    assert analyzer.discriminator_logs.empty

@patch('utils.analyze_logs.pd.read_csv')
@patch('utils.analyze_logs.Path.glob')
@patch('utils.analyze_logs.Path.exists')
def test_load_all_logs(mock_exists, mock_glob, mock_read_csv):
    # Setup mocks
    mock_exists.return_value = True
    
    # Mock glob returns
    mock_gen_path = MagicMock()
    mock_gen_path.parent.parent.name = "dataset1"
    
    mock_glob.side_effect = [
        [mock_gen_path], # generator models
        [mock_gen_path], # generation output
        [mock_gen_path], # evaluation logs
    ]
    
    # Mock read_csv returns
    mock_df = pd.DataFrame({'col': [1, 2, 3]})
    mock_read_csv.return_value = mock_df
    
    analyzer = LogAnalyzer()
    analyzer.load_all_logs()
    
    assert not analyzer.generator_logs.empty
    assert not analyzer.generation_logs.empty
    assert not analyzer.evaluation_logs.empty
    assert not analyzer.discriminator_logs.empty

def test_analyze_training_performance(analyzer):
    report = analyzer.analyze_training_performance()
    assert "GENERATOR TRAINING PERFORMANCE" in report
    assert "Total models trained: 3" in report
    assert "Datasets used: 2" in report
    assert "LSTM" in report
    assert "TRANSFORMER" in report

def test_analyze_generation_quality(analyzer):
    report = analyzer.analyze_generation_quality()
    assert "GENERATION QUALITY ANALYSIS" in report
    assert "Total generations analyzed: 3" in report
    assert "GREEDY" in report
    assert "TOP_P" in report

def test_analyze_discriminator_performance(analyzer):
    report = analyzer.analyze_discriminator_performance()
    assert "DISCRIMINATOR PERFORMANCE" in report
    assert "LSTM" in report
    assert "TRANSFORMER" in report
    assert "Context Size Effects" in report

def test_generate_insights(analyzer):
    insights = analyzer.generate_insights()
    assert "KEY INSIGHTS FOR PRESENTATION" in insights
    assert "TRAINING FINDINGS" in insights
    assert "GENERATION FINDINGS" in insights
    assert "DISCRIMINATOR FINDINGS" in insights

@patch('builtins.open', new_callable=mock_open)
@patch('utils.analyze_logs.Path.mkdir')
def test_generate_full_report(mock_mkdir, mock_file, analyzer):
    analyzer.generate_full_report("test_output/report.md")
    
    mock_mkdir.assert_called()
    mock_file.assert_called_with(Path("test_output/report.md"), 'w')
    handle = mock_file()
    handle.write.assert_called()

@patch('utils.analyze_logs.plt')
@patch('utils.analyze_logs.Path.mkdir')
def test_create_visualizations(mock_mkdir, mock_plt, analyzer):
    # Setup mock for subplots
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    # Create a 2x2 array of mocks for axes
    # Use np.empty and fill to prevent numpy from iterating into the mocks
    mock_axes = np.empty((2, 2), dtype=object)
    mock_axes.fill(mock_ax)
    mock_plt.subplots.return_value = (mock_fig, mock_axes)

    analyzer.create_visualizations("test_output/figures")
    
    mock_mkdir.assert_called()
    # Check if savefig was called multiple times
    assert mock_plt.savefig.call_count >= 4 
    # We expect at least: training_loss, model_size, generation_quality, discriminator_curves

@patch('utils.analyze_logs.pd.DataFrame.to_csv')
@patch('utils.analyze_logs.Path.mkdir')
def test_export_summary_tables(mock_mkdir, mock_to_csv, analyzer):
    analyzer.export_summary_tables("test_output/tables")
    
    mock_mkdir.assert_called()
    # Check if to_csv was called multiple times
    assert mock_to_csv.call_count >= 2
    # We expect at least: generator_summary, generation_quality_summary

def test_empty_logs():
    analyzer = LogAnalyzer()
    # Should handle empty logs gracefully
    assert "No generator training logs found" in analyzer.analyze_training_performance()
    assert "No generation or evaluation logs found" in analyzer.analyze_generation_quality()
    assert "No discriminator logs found" in analyzer.analyze_discriminator_performance()
