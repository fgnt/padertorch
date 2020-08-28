# WSJ0-mix data preparation

To prepare the wsj0-2mix and wsj0-3mix data, follow the following steps:
 1. Generate the mixtures using the matlab scripts.
 2. Edit `prepare_data.sh` to match your paths. You need to specify paths to the generated data and to the WSJ(0) database. WSJ(0) is required to obtain transcriptions. You can edit the `--json_root` parameter to specify the path to the output JSON.
 3. Run `prepare_data.sh`.
 
This script creates a JSON file that can be used by the examples.
The JSON file is compatible with `lazy_dataset.database.JsonDatabase`.
An example of reading data:

```python
from lazy_dataset.database import JsonDatabase
import numpy as np
import paderbox as pb

db = JsonDatabase("/path/to/JSON.json")

dataset = db.get_dataset("mix_2_spk_min")

def pre_batch_transform(inputs):
    return {
        's': np.ascontiguousarray([
            pb.io.load_audio(p)
            for p in inputs['audio_path']['speech_source']
        ], np.float32),
        'y': np.ascontiguousarray(
            pb.io.load_audio(inputs['audio_path']['observation']), np.float32),
        'num_samples': inputs['num_samples'],
        'example_id': inputs['example_id'],
        'audio_path': inputs['audio_path'],
    }
dataset = dataset.map(pre_batch_transform)

example = dataset[0]
```
 