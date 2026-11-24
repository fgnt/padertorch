import struct

'''
Event structure:

{
    'wall_time': ...,
    'step': ...,
    'summary': 
        'value': [{  # length is 1
            'tag': ...,
            'simple_value': ...,
            'histo': {
                'min': ...,
                'max': ...,
                'num': ...,
                'sum': ...,
                'sum_squares': ...,
                'bucket_llimit': ...,
                'bucket': ...,
            }    
        }]
}
'''

_use_fallback_converter = False


def _protobuf_to_dict(message):
    global _use_fallback_converter
    if not _use_fallback_converter:
        try:
            from protobuf_to_dict import protobuf_to_dict
            return protobuf_to_dict(message)
        except Exception:
            _use_fallback_converter = True
    return _protobuf_to_dict_fallback(message)


def _protobuf_to_dict_fallback(message):
    """Fallback for protobuf_to_dict.protobuf_to_dict (pip install protobuf3-to-dict),
    which fails with protobuf 7.36.2.

    Tested to work for tfevent files in 2026.
    """
    result = {}
    for field, value in message.ListFields():
        if field.is_extension:
            raise NotImplementedError(field.full_name)
        if field.message_type is not None:
            if field.message_type.GetOptions().map_entry:
                value_field = field.message_type.fields_by_name['value']
                result[field.name] = {
                    k: _protobuf_to_dict_fallback(v)
                    if value_field.message_type is not None else v
                    for k, v in value.items()
                }
                continue
            if field.is_repeated:
                value = [_protobuf_to_dict_fallback(v) for v in value]
            else:
                value = _protobuf_to_dict_fallback(value)
        elif field.is_repeated:
            value = list(value)
        result[field.name] = value
    return result


def load_events_as_dict(
        path,
        backend='tbX',
):
    """

    Args:
        path:
            Path to a tfevent file
        backend:
            'tbX' or 'tf'
            Use tensorboardX or tensorflow to load the tfevents file.

    Returns:
        generator that yields the events as dict

    >>> path = '/net/vol/boeddeker/sacred/torch/am/32/events.out.tfevents.1545605113.ntsim1'
    >>> list(load_events_as_dict(path))[2] # doctest: +SKIP
    {'wall_time': 1545605119.7274427, 'step': 1, 'summary': {'value': [{'tag': 'training/grad_norm', 'simple_value': 0.21423661708831787}]}}
    >>> list(load_events_as_dict(path, backend='tf'))[2] # doctest: +SKIP
    {'wall_time': 1545605119.7274427, 'step': 1, 'summary': {'value': [{'tag': 'training/grad_norm', 'simple_value': 0.21423661708831787}]}}

    """
    # from google.protobuf.json_format import MessageToDict
    # MessageToDict(e, preserving_proto_field_name=True)
    #   Converts int to str -> Bad behaviour
    if backend == 'tf':
        import tensorflow as tf
        return [
            _protobuf_to_dict(e)
            for e in tf.train.summary_iterator(str(path))
        ]
    elif backend == 'tbX':
        from tensorboardX.event_file_writer import event_pb2

        def read(fd):
            # Original
            # https://github.com/lanpa/tensorboard-dumper/blob/master/dump.py
            # Remove this code, once
            # https://github.com/lanpa/tensorboardX/issues/318
            # has a solution.
            header_data = fd.read(8)
            if header_data == b'':
                return None
            header, = struct.unpack('Q', header_data)
            crc_hdr = struct.unpack('I', fd.read(4))
            event_str = fd.read(header)  # 8+4
            crc_ev = struct.unpack('>I', fd.read(4))

            event = event_pb2.Event()
            event.ParseFromString(event_str)
            return event

        def read_all(path):
            with open(path, 'rb') as fd:
                event = read(fd)
                while event is not None:
                    yield _protobuf_to_dict(event)
                    event = read(fd)

        return read_all(path)
    else:
        raise ValueError(backend)
