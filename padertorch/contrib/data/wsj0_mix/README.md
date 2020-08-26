# WSJ0-mix data preparation

To prepare the wsj0-2mix and wsj0-3mix data, follow the following steps:
 1. Generate the mixtures using the matlab scripts.
 2. Edit `prepare_data.sh` to match your paths. You need to specify paths to the generated data and to the WSJ(0) database. WSJ(0) is required to obtain transcriptions. You can edit the `--json_root` parameter to specify the path to the output JSON.
 3. Run `prepare_data.sh`.
 
This script creates a JSON file that can be used by the examples.
 