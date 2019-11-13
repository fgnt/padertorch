from pathlib import Path
from warnings import warn

from paderbox.io import load_json
from paderbox.utils.nested import flatten
from padertorch.configurable import class_to_str


def dict_compare(d1, d2):
    # From http://stackoverflow.com/questions/4527942/comparing-two-dictionaries-in-python
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys

    # Init differs from defaults:
    modified = {o: (d1[o], d2[o]) for o in intersect_keys if d1[o] != d2[o]}

    same = set(o for o in intersect_keys if d1[o] == d2[o])
    are_equal = not len(added) and not len(removed) and not len(modified)
    return added, removed, modified, same, are_equal


def compare_configs(storage_dir, trainer_opts, provider_opts):
    opts = flatten(trainer_opts)
    opts.update(flatten(provider_opts))
    init = load_json(Path(storage_dir) / 'init.json')

    added, removed, modified, _, _ = dict_compare(opts, init)
    if len(added):
        warn(
            f'The following options were added to the model: {added}'
        )
    if len(removed):
        warn(
            f'The following options were removed from the model: {removed}'
        )

    return init['trainer_opts'], init['provider_opts']


def get_experiment_name(model_opts, submodel=None):
    model_name = class_to_str(model_opts["factory"])
    assert isinstance(model_name, str), (model_name, type(model_name))
    model_name = model_name.split('.')[-1]
    if submodel is not None:
        sub_name = class_to_str(model_opts[submodel]["factory"])
        assert isinstance(sub_name, str), (sub_name, type(sub_name))
        sep_name = sub_name.split('.')[-1]
    else:
        sep_name = 'baseline'
    ex_name = f'{model_name}/{sep_name}'
    return ex_name
