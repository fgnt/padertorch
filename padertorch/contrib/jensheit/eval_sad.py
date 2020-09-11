import numpy as np
import dlp_mpi

def adjust_annotation_fn(annotation, sample_rate, buffer_zone=1.):
    '''

    Args:
        annotation:
        sample_rate:
        buffer_zone: num secs around speech activity which are not scored

    Returns:
    >>> annotation = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    >>> adjust_annotation_fn(annotation, 1)
    array([5, 1, 1, 1, 5, 0, 5, 1], dtype=int32)
    >>> annotation = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    >>> adjust_annotation_fn(annotation, 2)
    array([5, 1, 1, 1, 5, 5, 5, 1], dtype=int32)
    '''
    buffer_zone = int(buffer_zone * sample_rate)
    indices = np.where(annotation[:-1] != annotation[1:])[0]
    if len(indices) == 0:
        return annotation
    elif len(indices) % 2 != 0:
        indices = np.concatenate([indices, [len(annotation)]], axis=0)
    start_end = np.split(indices, len(indices) // 2)
    annotation = annotation.astype(np.int32)
    for start, end in start_end:
        start += 1
        end += 1
        start_slice = slice(start - buffer_zone, start, 1)
        annotation[start_slice][annotation[start_slice] != 1] = 5
        end_slice = slice(end, end + buffer_zone, 1)
        annotation[end_slice][annotation[end_slice] != 1] = 5
    return annotation


def get_tp_fp_tn_fn(
        annotation, vad, sample_rate=8000, adjust_annotation=True
):
    """
    >>> annotation = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    >>> vad = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    >>> get_tp_fp_tn_fn(annotation, vad, 1)
    (4, 0, 4, 0)
    >>> annotation = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    >>> vad = np.array([1, 1, 1, 1, 0, 0, 0, 1])
    >>> get_tp_fp_tn_fn(annotation, vad, 1)
    (4, 0, 3, 0)
    >>> annotation = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    >>> vad = np.array([0, 1, 1, 1, 0, 1, 0, 1])
    >>> get_tp_fp_tn_fn(annotation, vad, 1)
    (4, 1, 3, 0)
    >>> annotation = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    >>> vad = np.array([0, 1, 1, 1, 0, 0, 0, 0])
    >>> get_tp_fp_tn_fn(annotation, vad, 1)
    (3, 0, 4, 1)
    >>> annotation = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    >>> vad = np.array([1, 1, 1, 1, 1, 0, 1, 1])
    >>> get_tp_fp_tn_fn(annotation, vad, 1)
    (4, 0, 1, 0)
    >>> annotation = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    >>> vad = np.array([0, 1, 1, 1, 0, 1, 0, 0])
    >>> get_tp_fp_tn_fn(annotation, vad, 12)
    (3, 0, 3, 1)
    >>> rng = np.random.RandomState(seed=3)
    >>> annotation = rng.randint(0, 2, size=(32000))
    >>> vad = rng.randint(0, 2, size=(32000))
    >>> get_tp_fp_tn_fn(annotation, vad)
    (8090, 2, 7937, 7978)

    :param annotation:
    :param vad:
    :param sample_rate:
    :param adjust_annotation:
    :return:
    """
    assert len(annotation) == len(vad), (len(annotation), len(vad))
    assert annotation.ndim == 1, annotation.shape
    assert vad.ndim == 1, vad.shape
    if adjust_annotation:
        annotation = adjust_annotation_fn(annotation, sample_rate)

    vad = np.round(vad).astype(np.int32) * 10
    result = vad + annotation.astype(np.int32)
    tp = result[result == 11].shape[0]
    fp = result[result == 10].shape[0]
    tn = result[result == 0].shape[0]
    fn = result[result == 1].shape[0]
    return tp, fp, tn, fn


def evaluate_model(dataset, model, get_sad_fn, get_target_fn=lambda x: x['activation'],
                   is_indexable=True, num_thresholds=201, buffer_zone=0.5,
                   sample_rate=8000):

    tp_fp_tn_fn = np.zeros((num_thresholds, 4), dtype=int)
    for example in dlp_mpi.split_managed(
        dataset, is_indexable=is_indexable,
        allow_single_worker=False,
    ):
        target = get_target_fn(example)
        adjusted_target = adjust_annotation_fn(
            target, buffer_zone=buffer_zone,
            sample_rate=sample_rate
        )
        model_out = model(example)
        for idx, th in enumerate(np.linspace(0, 1, num_thresholds)):
            th = np.round(th, 2)
            sad = get_sad_fn(model_out, th)
            out = get_tp_fp_tn_fn(
                adjusted_target, sad,
                sample_rate=sample_rate, adjust_annotation=False
            )
            tp_fp_tn_fn[idx] = [tp_fp_tn_fn[idx][idy] + o for idy, o in
                                enumerate(out)]

    dlp_mpi.barrier()
    tp_fp_tn_fn_gather = dlp_mpi.gather(tp_fp_tn_fn, root=dlp_mpi.MASTER)
    if dlp_mpi.IS_MASTER:
        tp_fp_tn_fn = np.zeros((num_thresholds, 4), dtype=int)
        for array in tp_fp_tn_fn_gather:
            tp_fp_tn_fn += array
    else:
        tp_fp_tn_fn = None
    return tp_fp_tn_fn
