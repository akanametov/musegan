"""Midi dataset."""

from typing import Tuple
from torch import Tensor

import torch
from torch import nn
from torch.utils.data import Dataset

import numpy as np
from music21 import midi
from music21 import converter
from music21 import note, stream, duration, tempo


class LPDDataset(Dataset):
    """LPDDataset.

    Parameters
    ----------
    path: str
        Path to dataset.
    """

    def __init__(
        self,
        path: str,
    ) -> None:
        """Initialize."""
        dataset = np.load(path, allow_pickle=True, encoding="bytes")
        self.data_binary = dataset["arr_0"]

    def __len__(self) -> int:
        """Return the number of samples in dataset."""
        return len(self.data_binary)

    def __getitem__(self, index: int) -> Tensor:
        """Return one samples from dataset.

        Parameters
        ----------
        index: int
            Index of sample.

        Returns
        -------
        Tensor:
            Sample.

        """
        return torch.from_numpy(self.data_binary[index]).float()


class MidiDataset(Dataset):
    """MidiDataset.

    Parameters
    ----------
    path: str
        Path to dataset.
    split: str, optional (default="train")
        Split of dataset.
    n_bars: int, optional (default=2)
        Number of bars.
    n_steps_per_bar: int, optional (default=16)
        Number of steps per bar.

    """

    def __init__(
        self,
        path: str,
        split: str = "train",
        n_bars: int = 2,
        n_steps_per_bar: int = 16,
    ) -> None:
        """Initialize."""
        self.n_bars = n_bars
        self.n_steps_per_bar = n_steps_per_bar
        dataset = np.load(path, allow_pickle=True, encoding="bytes")[split]
        self.data_binary, self.data_ints, self.data = self.__preprocess__(dataset)

    def __len__(self) -> int:
        """Return the number of samples in dataset."""
        return len(self.data_binary)

    def __getitem__(self, index: int) -> Tensor:
        """Return one samples from dataset.

        Parameters
        ----------
        index: int
            Index of sample.

        Returns
        -------
        Tensor:
            Sample.

        """
        return torch.from_numpy(self.data_binary[index]).float()

    def __preprocess__(self, data: np.ndarray) -> Tuple[np.ndarray]:
        """Preprocess data.

        Parameters
        ----------
        data: np.ndarray
            Data.

        Returns
        -------
        Tuple[np.ndarray]:
            Data binary, data ints, preprocessed data.

        """
        data_ints = []
        for x in data:
            skip = True
            skip_rows = 0
            while skip:
                if not np.any(np.isnan(x[skip_rows: skip_rows + 4])):
                    skip = False
                else:
                    skip_rows += 4
            if self.n_bars * self.n_steps_per_bar < x.shape[0]:
                data_ints.append(x[skip_rows: self.n_bars * self.n_steps_per_bar + skip_rows, :])
        data_ints = np.array(data_ints)
        self.n_songs = data_ints.shape[0]
        self.n_tracks = data_ints.shape[2]
        data_ints = data_ints.reshape([self.n_songs, self.n_bars, self.n_steps_per_bar, self.n_tracks])
        max_note = 83
        mask = np.isnan(data_ints)
        data_ints[mask] = max_note + 1
        max_note = max_note + 1
        data_ints = data_ints.astype(int)
        num_classes = max_note + 1
        data_binary = np.eye(num_classes)[data_ints]
        data_binary[data_binary == 0] = -1
        data_binary = np.delete(data_binary, max_note, -1)
        data_binary = data_binary.transpose([0, 3, 1, 2, 4])
        return data_binary, data_ints, data


def binarise_output(output: np.ndarray) -> np.ndarray:
    """Binarize output.

    Parameters
    ----------
    output: np.ndarray
        Output array.

    """
    max_pitches = np.argmax(output, axis=-1)
    return max_pitches


def postprocess(
    output: np.ndarray,
    n_tracks: int = 4,
    n_bars: int = 2,
    n_steps_per_bar: int = 16,
) -> stream.Score:
    """Postprocess output.

    Parameters
    ----------
    output: np.ndarray
        Output array.
    n_tracks: int, (default=4)
        Number of tracks.
    n_bars: int, (default=2)
        Number of bars.
    n_steps_per_bar: int, (default=16)
        Number of steps per bar.

    """
    parts = stream.Score()
    parts.append(tempo.MetronomeMark(number=66))
    max_pitches = binarise_output(output)
    midi_note_score = np.vstack([
        max_pitches[i].reshape([n_bars * n_steps_per_bar, n_tracks]) for i in range(len(output))
    ])
    for i in range(n_tracks):
        last_x = int(midi_note_score[:, i][0])
        s = stream.Part()
        dur = 0
        for idx, x in enumerate(midi_note_score[:, i]):
            x = int(x)
            if (x != last_x or idx % 4 == 0) and idx > 0:
                n = note.Note(last_x)
                n.duration = duration.Duration(dur)
                s.append(n)
                dur = 0
            last_x = x
            dur = dur + 0.25
        n = note.Note(last_x)
        n.duration = duration.Duration(dur)
        s.append(n)
        parts.append(s)
    return parts
