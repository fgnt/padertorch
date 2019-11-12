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

    >>> path = '/net/home/boeddeker/sacred/torch/am/32/events.out.tfevents.1545605113.ntsim1'
    >>> list(load_events_as_dict(path))[2]
    {'wall_time': 1545605119.7274427, 'step': 1, 'summary': {'value': [{'tag': 'training/grad_norm', 'simple_value': 0.21423661708831787}]}}
    >>> list(load_events_as_dict(path, backend='tf'))[2]
    {'wall_time': 1545605119.7274427, 'step': 1, 'summary': {'value': [{'tag': 'training/grad_norm', 'simple_value': 0.21423661708831787}]}}

    """
    from protobuf_to_dict import protobuf_to_dict  # protobuf3-to-dict (PyPI)

    # from google.protobuf.json_format import MessageToDict
    # MessageToDict(e, preserving_proto_field_name=True)
    #   Converts int to str -> Bad behaviour
    if backend == 'tf':
        import tensorflow as tf
        return [
            protobuf_to_dict(e)
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
                    yield protobuf_to_dict(event)
                    event = read(fd)

        return read_all(path)
    else:
        raise ValueError(backend)
