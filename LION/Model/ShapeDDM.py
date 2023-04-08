from Model.PVCNN2Base import *
from Model.GaussianDiffusion import *


class ResSE(nn.Module):
    def __init__(self, reduce=8, nout=2048):
        super(ResSE, self).__init__()
        self.se = nn.Sequential(nn.Linear(nout, nout // reduce),
                                nn.ReLU(inplace=True),
                                nn.Linear(nout // reduce, nout),
                                nn.Sigmoid())
        self.resse = nn.Sequential(nn.Linear(2048, nout // reduce),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(),
                                   nn.Linear(nout // reduce, nout),
                                   nn.ReLU(inplace=True),
                                   self.se)

    def forward(self, input):
        return input + self.resse(input)


class ShapeModel(nn.Module):

    def __init__(self):
        super(ShapeModel, self).__init__()
        self.embed_dim = 128
        self.embedf = nn.Sequential(
            nn.Linear(128, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, 2048),
        )
        self.resSE = nn.Sequential(ResSE(), ResSE(),
                                   ResSE(), ResSE(),
                                   ResSE(), ResSE(),
                                   ResSE(), ResSE(),
                                   nn.Linear(2048, 128))
        self.MLP = nn.Linear(128, 2048)

    def get_timestep_embedding(self, timesteps, device):
        assert len(timesteps.shape) == 1

        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:  # zero pad
            # emb = tf.concat([emb, tf.zeros([num_embeddings, 1])], axis=1)
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        assert emb.shape == torch.Size([timesteps.shape[0], self.embed_dim])
        return emb

    def forward(self, inputs, t):

        temb = self.embedf(self.get_timestep_embedding(t, inputs.device))

        inputs = self.MLP(inputs)

        return self.resSE(inputs + temb)


class ShapeDDM(nn.Module):
    def __init__(self, args, betas, loss_type: str, model_mean_type: str, model_var_type: str):
        super(ShapeDDM, self).__init__()
        self.diffusion = GaussianDiffusion(betas, loss_type, model_mean_type, model_var_type)

        self.model = ShapeModel()

    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, clip_denoised=True):
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt = self.diffusion.calc_bpd_loop(self._denoise, x0, clip_denoised)

        return {
            'total_bpd_b': total_bpd_b,
            'terms_bpd': vals_bt,
            'prior_bpd_b': prior_bpd_b,
            'mse_bt': mse_bt
        }

    def _denoise(self, data, t):
        B, D, N = data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        out = self.model(data, t)

        assert out.shape == torch.Size([B, D, N])
        return out

    def get_loss_iter(self, data, noises=None):
        B, D, N = data.shape
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data.device)

        if noises is not None:
            noises[t != 0] = torch.randn((t != 0).sum(), *noises.shape[1:]).to(noises)

        losses = self.diffusion.p_losses(
            denoise_fn=self._denoise, data_start=data, t=t, noise=noises)
        assert losses.shape == t.shape == torch.Size([B])
        return losses

    def gen_samples(self, shape, device, noise_fn=torch.randn,
                    clip_denoised=True,
                    keep_running=False):
        return self.diffusion.p_sample_loop(self._denoise, shape=shape, device=device, noise_fn=noise_fn,
                                            clip_denoised=clip_denoised,
                                            keep_running=keep_running)

    def gen_sample_traj(self, shape, device, freq, noise_fn=torch.randn,
                        clip_denoised=True, keep_running=False):
        return self.diffusion.p_sample_loop_trajectory(self._denoise, shape=shape, device=device, noise_fn=noise_fn,
                                                       freq=freq,
                                                       clip_denoised=clip_denoised,
                                                       keep_running=keep_running)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
