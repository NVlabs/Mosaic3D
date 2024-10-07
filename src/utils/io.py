import numpy as np


def pack_list_of_np_arrays(array_list):
    if isinstance(array_list[0], list):
        # Handle list of list of 1D numpy arrays
        packed = np.concatenate([np.concatenate(sublist) for sublist in array_list])
        outer_lengths = np.array([len(sublist) for sublist in array_list])
        inner_lengths = np.array([len(arr) for sublist in array_list for arr in sublist])
        return dict(packed=packed, outer_lengths=outer_lengths, inner_lengths=inner_lengths)
    else:
        # Handle list of 1D numpy arrays
        packed = np.concatenate(array_list)
        lengths = [len(arr) for arr in array_list]
        return dict(packed=packed, lengths=lengths)


def split_list_into_chunks(lst, chunk_sizes):
    """Split a list into variable-size chunks.

    Args:
        lst (list): The input list to be split.
        chunk_sizes (list): A list of integers representing the sizes of each chunk.

    Returns:
        list: A list of sublists, where each sublist is a chunk of the input list.
    """
    if sum(chunk_sizes) != len(lst):
        raise ValueError("Sum of chunk sizes must equal the length of the input list")

    return [lst[sum(chunk_sizes[:i]) : sum(chunk_sizes[: i + 1])] for i in range(len(chunk_sizes))]


def unpack_list_of_np_arrays(filename):
    with np.load(filename) as data:
        packed = data["packed"]
        if "outer_lengths" in data:
            # Unpack list of list of 1D numpy arrays
            outer_lengths = data["outer_lengths"]
            inner_lengths = data["inner_lengths"]
            inner_splits = np.split(packed, np.cumsum(inner_lengths)[:-1])
            outer_splits = split_list_into_chunks(inner_splits, outer_lengths)
            return outer_splits
        else:
            # Unpack list of 1D numpy arrays
            lengths = data["lengths"]
            return [np.array(arr) for arr in np.split(packed, np.cumsum(lengths)[:-1])]
