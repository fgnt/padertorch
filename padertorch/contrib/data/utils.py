from pathlib import Path
import paderbox as pb


def check_audio_files_exist(
    database_dict,
    speedup=None,
    extensions=('.wav', '.wv2', '.wv1', '.flac'),
):
    """
    No structure for the database_dict is assumed. It will just search for all
    string values ending with a certain file type (e.g. wav).

    >>> check_audio_files_exist({2: [1, '/net/db/timit/pcm/train/dr1/fcjf0/sa1.wav', 'abc.wav']})
    Traceback (most recent call last):
      ...
    AssertionError: ('abc.wav', (2, '2'))
    >>> check_audio_files_exist(1)
    Traceback (most recent call last):
      ...
    AssertionError: Expect at least one wav file. It is likely that the database folder is empty and the greps did not work. to_check: {}
    >>> check_audio_files_exist('abc.wav')
    Traceback (most recent call last):
      ...
    AssertionError: ('abc.wav', ())
    >>> check_audio_files_exist('/net/db/timit/pcm/train/dr1/fcjf0/sa1.wav')
    >>> check_audio_files_exist(1, speedup='thread')
    Traceback (most recent call last):
      ...
    AssertionError: Expect at least one wav file. It is likely that the database folder is empty and the greps did not work. to_check: {}
    >>> check_audio_files_exist('abc.wav', speedup='thread')
    Traceback (most recent call last):
      ...
    AssertionError: ('abc.wav', ())
    >>> check_audio_files_exist('/net/db/timit/pcm/train/dr1/fcjf0/sa1.wav', speedup='thread')
    """

    def path_exists(path):
        return Path(path).exists()

    def body(file_key_path):
        key_path, file = file_key_path
        assert path_exists(file), (file, key_path)

    def condition_fn(file):
        return isinstance(file, (str, Path)) and str(file).endswith(extensions)

    to_check = {
        k: v for k, v in pb.utils.nested.flatten(database_dict).items()
        if condition_fn(v)
    }

    assert len(to_check) > 0, (
        f'Expect at least one wav file. '
        f'It is likely that the database folder is empty '
        f'and the greps did not work. to_check: {to_check}'
    )

    if speedup and 'thread' == speedup:
        import os
        from multiprocessing.pool import ThreadPool

        # Use this number because ThreadPoolExecutor is often
        # used to overlap I/O instead of CPU work.
        # See: concurrent.futures.ThreadPoolExecutor
        # max_workers = (os.cpu_count() or 1) * 5

        # Not sufficiently benchmarked both, this is more conservative.
        max_workers = (os.cpu_count() or 1)

        with ThreadPool(max_workers) as pool:
            for _ in pool.imap_unordered(
                    body,
                    to_check.items()
            ):
                pass

    elif speedup is None:
        for file, key_path in to_check.items():
            assert path_exists(file), (key_path, file)
    else:
        raise ValueError(speedup, type(speedup))
