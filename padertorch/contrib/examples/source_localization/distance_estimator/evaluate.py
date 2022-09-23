"""
Evaluation script for the the distance estimator trained in $STORAGE_ROOT.

Saves results to $STORAGE_ROOT/source_localization/distance_estimator_{feature}
_{id}/evaluation_result.json

By default, the latest modified model is evaluated if no explicit path to a
trained model is specified, which can be done by adding
"with storage_dir=/PATH/TO/DESIRED/DIRECTORY.
If a checkpoint of a certain feature should be examined, it can be added:
"with feature="String of desired features"
Normally, the checkpoint with the best mae is used for the evaluation.
Otherwise, it have to be specified by appending
"with checkpoint_path=/PATH/TO/CHECKPOINT" to the execution command.

May be called with:
mpiexec -np $(nproc --all) python -m
padertorch.contrib.examples.source_localization.distance_estimator.evaluate

"""

import os
from pathlib import Path

import numpy as np
from sacred import Experiment
import torch
from torch import nn

import dlp_mpi
from model import DistanceEstimator
import paderbox as pb
import padertorch as pt
from train import prepare_data_iterator

ex = Experiment('Evaluate Distance Estimator')


@ex.config
def config():
    feature = None
    batch_size = 1
    storage_dir = get_storage_dir(feature)
    checkpoint_name = 'ckpt_best_mae.pth'


def get_storage_dir(feature):
    """Get the a distance_estimator subdirectory,
     where the checkpoints and configuration of the latest training with the
     desired training input feature are stored.
     If feature is none, the latest model will be evaluated"""
    if 'STORAGE_ROOT' not in os.environ:
        raise EnvironmentError(
            'You have to specify an STORAGE_ROOT environmental variable'
        )
    else:
        storage_root = Path(os.environ['STORAGE_ROOT']).expanduser().resolve()
    task_dir = storage_root / 'source_localization'
    if feature is None:
        dirs = list(task_dir.glob('distance_estimator_*_*'))
        statistics = []
        for sub_dir in dirs:
            statistics.append(os.stat(sub_dir)[8])
        index_latest = statistics.index(max(statistics))
        return dirs[index_latest]
    else:
        feature = "_".join(feature.split())
        dirs = list(task_dir.glob('distance_estimator_' + feature + "_*"))
        latest_id = \
            sorted([int(path.name.split('_')[-1]) for path in dirs])[-1]
        return task_dir / f'distance_estimator_{feature}_{latest_id}'


def get_pseudo_acc(summary):
    target = np.asarray(summary.pop('target'))
    est_cls = np.asarray(summary.pop('est_cls'))
    target_min_1 = target - 1
    target_pl_1 = target + 1
    acc_allow_neighbors = est_cls == target
    acc_allow_neighbors += est_cls == target_min_1
    acc_allow_neighbors += est_cls == target_pl_1
    return acc_allow_neighbors


@ex.automain
def evaluate(storage_dir, checkpoint_name, batch_size):
    train_config = pt.io.load_config(Path(storage_dir) / '1/config.json')
    feature = train_config["feature"]
    train_config["data_provider_conf"]["batch_size"] = batch_size

    if isinstance(storage_dir, str):
        storage_dir = Path(storage_dir)
    model = DistanceEstimator.from_config_and_checkpoint(
        checkpoint_path=storage_dir / "checkpoints" / checkpoint_name,
        config_path=storage_dir / "1/config.json",
        consider_mpi=True)
    model.eval()
    evaluation_iterator = prepare_data_iterator(
        train_config["data_provider_conf"],
        train_config["rir_json"],
        dataset_key="eval",
        mic_pairs=train_config["mic_pairs"])
    summary = list()
    if dlp_mpi.IS_MASTER:
        print(f'Start to evaluate the checkpoint '
              f'{(storage_dir / "checkpoints" / checkpoint_name).resolve()} '
              f'and will write the evaluation result to '
              f'{storage_dir / "evaluation_result.json"}')
    with torch.no_grad():
        for batch in dlp_mpi.split_managed(
                evaluation_iterator, is_indexable=False,
                progress_bar=True,
                allow_single_worker=True
        ):
            model_output = model(pt.data.example_to_device(batch))
            target = batch['label']
            cls_probs = nn.Softmax(dim=-1)(model_output)
            est_cls = cls_probs.argmax(dim=-1)
            est_dist = est_cls.float() * model.quant_step + model.d_min
            ae = model.l1_loss(est_dist, torch.tensor(batch['distance']))
            se = model.mse_loss(est_dist, torch.tensor(batch['distance']))
            acc = est_cls.numpy() == target
            summary.append({'acc': acc,
                            'ae': ae,
                            'se': se,
                            'target': target,
                            'est_cls': est_cls})

    summary_list = dlp_mpi.COMM.gather(summary, root=dlp_mpi.MASTER)

    if dlp_mpi.IS_MASTER:
        ae_list = list()
        se_list = list()
        acc = 0
        pseudo_acc = 0
        num_examples = 0
        for _summary in summary_list:
            for partial_summary in _summary:
                num_examples += 1
                ae_list.append(partial_summary["ae"].numpy())
                se_list.append(partial_summary["se"].numpy())
                acc += partial_summary['acc']
                pseudo_acc += get_pseudo_acc(partial_summary)

        rmse = np.sqrt(np.mean(se_list))
        mae = np.mean(ae_list)
        result = dict(
            mae=mae,
            rmse=rmse,
            accuracy=np.sum(acc) / (num_examples * batch_size) * 100,
            pseudo_accuracy=np.sum(pseudo_acc) / (
                        num_examples * batch_size) * 100,
            segment_length=train_config['sig_len'] / train_config[
                'sample_rate'] * 1000,
            input_feature=feature
        )

        print("Evaluation results:\n")
        for key, value in result.items():
            if key in ['mae', 'rmse']:
                print(f'{key}: {value}m\n')
            elif key in ['accuracy', 'pseudo_accuracy']:
                print(f'{key}: {value}%\n')
            elif key in ['segment length']:
                print(f'{key}: {value}ms\n')
            else:
                print(f'{key}: {value}\n')

        result['number_examples'] = num_examples
        result_json_path = storage_dir / 'evaluation_result.json'
        print(f"Exporting result: {result_json_path}")
        pb.io.dump_json(result, result_json_path)
