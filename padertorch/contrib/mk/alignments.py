from pathlib import Path
from dataclasses import dataclass
from typing import Union
import re
import logging

try:
    import textgrids
except ImportError as exc:
    raise ImportError(
        'Could not import textgrids. Install via pip: '
        'pip install praat-textgrids'
    ) from exc
from lazy_dataset import FilterException
import numpy as np

import padertorch as pt

LOG = logging.getLogger('pt.contrib.mk.labels')


@dataclass
class TextGridAlignmentReader(pt.Configurable):
    """
    Load alignments (phone, syllables, ...) from Praat TextGrid files.
    `ali_root` must adhere to following structure:
        ali_root
        |- <speaker_id_1>
        |   |- <example_id_1_1>.TextGrid
        |   |- <example_id_1_2>.TextGrid
        |   ...
        |- <speaker_id_2>
        |   |- <example_id_2_1>.TextGrid
        |   ...
        ...

    Args:
        ali_root: Path to the root directory of the alignment files.
        label_key: Key to store the labels in the example dictionary.
        to_array: If True, convert the labels to numpy arrays.
        reduce_labels: If True, remove suffix digit classifiers from labels that
            may be appended by Praat.
        verbose: If True, log warnings if no alignment file is found for an
            example.
        drop_silence: If True, drop intervals with the label specified in
            `silence_label`.
        silence_label: Label to drop if `drop_silence` is True. Can be a list of
            labels.
    """
    ali_root: Union[str, Path]
    label_key: str
    to_array: bool = False
    reduce_labels: bool = True
    verbose: bool = False
    drop_silence: bool = False
    silence_label: Union[str, list] = 'SIL'

    def __post_init__(self):
        self.ali_root = Path(self.ali_root)

    def filter_fn(self, example):
        example_id = example['example_id']
        speaker_id = example['speaker_id']
        return (self.ali_root / speaker_id / example_id)\
            .with_suffix('.TextGrid').exists()

    def __call__(self, example: dict):
        """
        Arguments:
            example: Dictionary containing at least the keys `example_id` and
                `speaker_id`.

        Raises:
            lazy_dataset.FilterException if no alignment file for the example is
            found.
        """
        example_id = example['example_id']
        speaker_id = example['speaker_id']

        try:
            grid = textgrids.TextGrid(
                (self.ali_root / speaker_id / example_id)\
                    .with_suffix('.TextGrid')
            )
            start_times = []
            stop_times = []
            labels = []
            for interval in grid[self.label_key]:
                if isinstance(interval, textgrids.Point):
                    raise TypeError(
                        'PoinTier is not supported. Convert it to an '
                        'IntervalTier'
                    )
                text = interval.text
                if self.drop_silence and text in self.silence_label:
                    continue
                start_times.append(interval.xmin)
                stop_times.append(interval.xmax)
                if self.reduce_labels:
                    # Remove suffix digit classifiers from labels
                    text = re.sub(r'\d', '', text)
                labels.append(text)
            if self.to_array:
                start_times = np.array(start_times)
                stop_times = np.array(stop_times)
                labels = np.array(labels)
            example[f'{self.label_key}_start_times'] = start_times
            example[f'{self.label_key}_stop_times'] = stop_times
            example[self.label_key] = labels
            return example
        except FileNotFoundError as exc:
            if self.verbose:
                LOG.warn(f'No alignment for {example_id}. Drop example.')
            raise FilterException() from exc
