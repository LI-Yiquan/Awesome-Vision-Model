import numpy as np
import torch
import torch.nn.functional as F

def render(H,W, K, chunk,rays,c2w,ndc,
           near, far, use_viewdirs,c2w_staticcam,
           **kwargs):
    """

    :param H: Height
    :param W: Width
    :param K: Camera param
    :param chunk:
    :param rays: shape [2,batch_size,3]
                contains the origin position and
                the direction for each example in batch
    :param near: [batch_size] Nearest distance of ray
    :param far: [batch_size] Farthest distance of ray
    :return:
        rgb_map: the predicted RGB values for rays
        acc_map: the accumulated opacity along a ray
    """

    if c2w is not None:
        # a special case to render full image
        rays_o, rays_d = get_rays(H,W,K,c2w)

    else:
        # use the provided rays
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            rays_o, rays_d = get_rays(H,W,K,c2w_staticcam)

        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1,keepdim=True)
        viewdirs = torch.reshape(viewdirs,[-1,3]).float()


    # ndc

    # rays_o: [batch,3]
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near = near * torch.ones_like(rays_d[...,:1]) # [bs,1]

    far = far * torch.ones_like(rays_d[...,:1]) # [bs,1]

    rays = torch.cat([rays_o,rays_d,near,far],-1)

    if use_viewdirs:
        # one ray will be [3,3,1,1,3]
        # [origin, direction,near_dis,far_dis,view_direction]
        # rays will be [batch, 11]
        rays = torch.cat([rays, viewdirs],-1)


    all_ret = batchify_rays(rays,chunk,**kwargs)





def bacthify_rays(rays_flat,chunk,**kwargs):
    """
    :param rays_flat: [batch, 11]
    """
    all_ret = {}

def render_rays(rays_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=0,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    N_rays = rays_batch[0]

    rays_o, rays_d = rays_batch[:,0:3], rays_batch[:,3:6]

    viewdirs = rays_batch[:,-3:] if rays_batch.shape[-1]>8 else None

    bounds = torch.reshape(rays_batch[...,6:8],[-1,1,2])

    near, far = bounds[...,0], bounds[...,1]

    t_vals = torch.linspace(0,1,steps=N_samples)

    if not lindisp:
        z_vals = near * (1. - t_vals) + far*(t_vals)
    else:
        z_vals = 1. / (1. / near * (1. -t_vals) + 1. /far * (t_vals))

    z_vals = z_vals.expand([N_rays,N_samples])

    if perturb > 0.:
        mids = .5 * (z_vals[...,1:]+z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:],-1])
        lower = torch.cat([z_vals[...,:1],mids],-1)

        t_rand = torch.rand(z_vals.shape)

        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

        # [N_rays, N_samples, 3]
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]

        raw = network_query_fn(pts, viewdirs, network_query_fn)

        rgb_map,disp_map,acc_map,weights,depth_map = raw2outputs(raw,z_vals,rays_d,
                                                                 raw_noise_std,white_bkgd,
                                                                 pytest=pytest)

        if N_importance >0:
            rgb_map_0, disp_map_0,acc_map_0 = rgb_map,disp_map,acc_map

            z_vals_mid = .5 *(z_vals[...,1:]+z_vals[...,:-1])

            z_samples = sample_pdf(z_vals_mid,weights[...,1:-1],N_importance,det=(perturb==0.),pytest=pytest)
            z_samples = z_samples.detach()

            z_vals,_ = torch.sort(torch.cat([z_vals,z_samples],-1),-1)

            pts =rays_o[...,None,:] + rays_d[...,None,:]*z_vals[...,:,None]

            run_fn = network_fn if network_fine is None else network_fine

            raw = network_query_fn(pts,viewdirs, run_fn)

            rgb_map, disp_map,acc_map,weights,depth_map = raw2outputs(raw,z_vals,
                                                                      rays_d,raw_noise_std,
                                                                      white_bkgd,pytest=pytest)
            ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}

            if retraw:

                ret['raw'] = raw

            if N_importance > 0:

                ret['rgb0'] = rgb_map_0
                ret['disp0'] = disp_map_0
                ret['acc0'] = acc_map_0

                ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]


            for k in ret:
                if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
                    print(f"! [Numerical Error] {k} contains nan or inf.")

            return ret



def raw2outputs(raw,z_vals,rays_d,raw_noise_std=0,white_bkgd=False,
                pytest=False):
    # raws: [N_rays,N_samples,4]
    # z_vals: [N_rays,N_samples]
    # rays_d: [N_rays,3]

    raw2alpha = lambda raw, dists,act_fn=F.relu: 1.- torch.exp(-act_fn(raw)*dists)
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)
    # dists: [N_rays, N_samples]

    return rgb_map, disp_map, acc_map, weights, depth_map









