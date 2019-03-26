from torch.nn import ModuleDict

from padertorch.contrib.je.autoencoder import AE
from padertorch.contrib.je.latent_models import StandardNormal, GMM, FBGMM
from paderbox.utils.nested import nested_update, deflatten
from padertorch.train.trainer import MultiDeviceTrainer as BaseTrainer
from padertorch.train.optimizer import Adam
from padertorch.data import example_to_device


class Trainer(BaseTrainer):
    def _step(self, example):
        ae_device = self.device['ae'] if isinstance(self.device, dict) \
            else self.device
        latent_device = self.device['latent'] if isinstance(self.device, dict) \
            else self.device
        assert isinstance(
            self.model['ae'], AE
        ) and isinstance(
            self.model['latent'], (StandardNormal, GMM, FBGMM)
        )
        example = example_to_device(example, ae_device)

        (mean, logvar), pooling_data = self.model['ae'](
            example, command="encode"
        )
        latent_in = (
            mean.transpose(1, 2).to(latent_device),
            logvar.transpose(1, 2).to(latent_device)
        )
        latent_out = self.model['latent'](latent_in)
        ae_out = self.model['ae'](
            (latent_out[0].transpose(1, 2).to(ae_device), pooling_data),
            command="decode"
        )

        review = dict()
        nested_update(review, self.model['ae'].review(example, ae_out))
        nested_update(
            review,
            self.model['latent'].review(latent_in, latent_out)
        )
        return {'ae': ae_out, 'latent': latent_out}, review

    def to(self, device):
        assert isinstance(self.model, ModuleDict) and set(self.model.keys()) == {'ae', 'latent'}
        assert not isinstance(device, dict) or set(device.keys()) == {'ae', 'latent'}
        for key in self.model.keys():
            device_ = device[key] if isinstance(device, dict) else device
            self.model[key].to(device_)
            if key in self.optimizer:
                self.optimizer[key].to(device_)
        self.device = device

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['model'] = deflatten(
            {'ae.factory': AE, 'latent.factory': GMM}
        )
        config['optimizer'] = deflatten(
            {'ae.factory': Adam, 'latent.factory': Adam}
        )
