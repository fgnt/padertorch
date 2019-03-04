from pathlib import Path

import tensorboardX

import paderbox as pb
import cbj
import natsort

if __name__ == '__main__':
    files = list((Path('.') / 'eval').glob('*/result.json'))

    if len(files) > 0:
        with tensorboardX.SummaryWriter('.', filename_suffix='.scores') as writer:

            for file in natsort.natsorted(
                    files, key=lambda file: file.parts[-2]
            ):
                ckpt_name: str = file.parts[-2]

                assert ckpt_name.startswith('ckpt_'), (ckpt_name, file)
                assert '_' not in ckpt_name[len('ckpt_'):], (ckpt_name, file)
                iteration = int(ckpt_name[len('ckpt_'):])

                scores = cbj.io.load(file)['scores']

                scores = pb.utils.nested.flatten(scores, sep='_')

                st_mtime = file.stat().st_mtime

                for k, v in scores.items():
                    writer.add_scalar(
                        f'validation/{k}',
                        v,
                        global_step=iteration,
                        # walltime=st_mtime,
                    )
                    print(f'Add {k}: {v} to tensorboard for step {iteration} from {file}')
    else:
        raise Exception(files)
