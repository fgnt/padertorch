from unittest.mock import Mock
from types import SimpleNamespace

from google.protobuf.struct_pb2 import Struct
from tensorboardX.proto.event_pb2 import Event

from padertorch.summary import tfevents


def test_converter_switches_after_one_failure(monkeypatch):
    import protobuf_to_dict

    converter = Mock(side_effect=[{'step': 1}, AttributeError('label')])
    monkeypatch.setattr(protobuf_to_dict, 'protobuf_to_dict', converter)
    monkeypatch.setattr(tfevents, '_use_fallback_converter', False)

    assert tfevents._protobuf_to_dict(Event(step=1)) == {'step': 1}
    assert not tfevents._use_fallback_converter
    assert tfevents._protobuf_to_dict(Event(step=2)) == {'step': 2}
    assert tfevents._use_fallback_converter
    assert tfevents._protobuf_to_dict(Event(step=3)) == {'step': 3}
    assert converter.call_count == 2


def test_fallback_event_types():
    expected = {
        'step': 2**60 + 1,
        'summary': {'value': [
            {'tag': 'scalar', 'simple_value': 0.0},
            {'tag': 'histogram', 'histo': {'bucket': [1.0, 2.0]}},
            {'tag': 'text', 'tensor': {
                'dtype': 7,
                'tensor_shape': {'dim': [{'size': 1}]},
                'string_val': [b'abc\x00\xff'],
            }},
        ]},
    }
    assert tfevents._protobuf_to_dict_fallback(Event(**expected)) == expected
    assert tfevents._protobuf_to_dict_fallback(Event()) == {}


def test_fallback_message_map():
    message = Struct()
    message.update({'number': 1, 'nested': {'flag': True}})
    assert tfevents._protobuf_to_dict_fallback(message) == {
        'fields': {
            'number': {'number_value': 1.0},
            'nested': {'struct_value': {
                'fields': {'flag': {'bool_value': True}},
            }},
        },
    }


def test_fallback_legacy_descriptor():
    field = SimpleNamespace(
        name='values', label=3, LABEL_REPEATED=3,
        is_extension=False, message_type=None,
    )
    message = SimpleNamespace(ListFields=lambda: [(field, (1, 2))])
    assert tfevents._protobuf_to_dict_fallback(message) == {'values': [1, 2]}
    field.label = 1
    message.ListFields = lambda: [(field, 1)]
    assert tfevents._protobuf_to_dict_fallback(message) == {'values': 1}
