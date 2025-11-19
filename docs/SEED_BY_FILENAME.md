# Seed Selection by Filename Feature

## Overview
This feature allows you to select specific MIDI files by name to use as seeds for music generation. During preprocessing, a mapping from filename to sequence index is created and saved, enabling easy access to any file in the dataset.

## Files Modified

### 1. `utils/preprocess_miditok.py`
- Added `filename_to_index` dictionary to track which sequence corresponds to which file
- Saves the mapping to `filename_to_index.json` in the output directory
- Updated output summary to mention the new file

### 2. `utils/preprocess_naive.py`
- Added same filename mapping functionality for consistency
- Saves `filename_to_index.json` alongside other output files

### 3. `utils/seed_selection.py`
- **New function**: `get_seed_by_filename(filename, dataset, length)`
  - Loads the filename mapping
  - Retrieves the specific sequence by filename
  - Handles miditok nested structure (flattens tracks)
  - Optionally truncates or pads to desired length
  - Provides helpful error messages if file not found

- **New function**: `list_available_files(dataset, search_term)`
  - Lists all available filenames in a dataset
  - Optional search/filter functionality
  - Useful for discovering what files are available

### 4. `utils/test_seed_by_filename.py` (NEW)
- Test script to demonstrate the new functionality
- Shows how to list files and select seeds

## Usage Examples

### Basic Usage
```python
from seed_selection import get_seed_by_filename

# Get seed from a specific file
seed = get_seed_by_filename("ashover.mid", dataset="miditok")

# Get seed with specific length (truncates or pads)
seed = get_seed_by_filename("ashover.mid", dataset="miditok", length=50)

# Works with any dataset
seed = get_seed_by_filename("ashover.mid", dataset="naive")
```

### Listing Available Files
```python
from seed_selection import list_available_files

# List all files
files = list_available_files(dataset="miditok")

# Search for specific files
jig_files = list_available_files(dataset="miditok", search_term="jig")
reel_files = list_available_files(dataset="miditok", search_term="reel")
```

### Integration with generate.py
You can modify your generation script to accept a filename parameter:

```python
# In generate.py
import argparse
from seed_selection import get_seed_by_filename

parser.add_argument("--seed_filename", type=str, default=None,
                    help="MIDI filename to use as seed (e.g., 'ashover.mid')")

# Then in the code:
if args.seed_filename:
    seed = get_seed_by_filename(args.seed_filename, dataset=args.dataset)
else:
    # Use random seed or other method
    seed = ...
```

### Command Line Usage
```bash
# Run test script to see available files
python utils/test_seed_by_filename.py

# Generate music from specific file
python generate.py --seed_filename "ashover.mid" --dataset miditok

# Generate with specific length
python generate.py --seed_filename "ashover.mid" --dataset miditok --seed_length 50
```

## Output Files

After preprocessing, you'll find these new files:

```
data/miditok/
  ├── sequences.npy
  ├── filename_to_index.json  ← NEW
  ├── tokenizer.json
  ├── vocab.json
  └── config.json

data/naive/
  ├── sequences.npy
  ├── filename_to_index.json  ← NEW
  └── note_to_int.pkl
```

### filename_to_index.json Format
```json
{
  "ashover.mid": 0,
  "banjo.mid": 1,
  "carolan.mid": 2,
  ...
}
```

## Benefits

1. **Reproducibility**: Generate from the same source file consistently
2. **Control**: Choose specific musical styles or patterns
3. **Testing**: Use known files for evaluation
4. **User-Friendly**: No need to remember sequence indices
5. **Discovery**: Search functionality helps find relevant files

## Requirements

- Must run preprocessing after this update to generate the mapping files
- Existing preprocessed data will need to be regenerated

## Backward Compatibility

- Old code using sequence indices still works
- New functionality is additive
- No breaking changes to existing APIs

## Testing

Run the test script:
```bash
cd utils
python test_seed_by_filename.py
```

This will:
1. List available files in the dataset
2. Try to get a seed from the first file
3. Demonstrate search functionality
4. Show usage examples

## Future Enhancements

Possible additions:
- Auto-complete for filenames
- Metadata about each file (tempo, key, etc.)
- Grouping by genre/style
- Similarity search
- Random selection within a category
