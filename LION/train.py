import argparse
from torch import nn, optim
from Model.LionModel import LionModel
from utils.dataset import *
from utils.file_utils import *

'''
some utils
'''


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rotate(vertices, faces):
    '''
    vertices: [numpoints, 3]
    '''
    M = rotation_matrix([0, 1, 0], np.pi / 2).transpose()
    N = rotation_matrix([1, 0, 0], -np.pi / 4).transpose()
    K = rotation_matrix([0, 0, 1], np.pi).transpose()

    v, f = vertices[:, [1, 2, 0]].dot(M).dot(N).dot(K), faces[:, [1, 2, 0]]
    return v, f


def norm(v, f):
    v = (v - v.min()) / (v.max() - v.min()) - 0.5

    return v, f


def getGradNorm(net):
    pNorm = torch.sqrt(sum(torch.sum(p ** 2) for p in net.parameters()))
    gradNorm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in net.parameters()))
    return pNorm, gradNorm


def weights_init(m):
    """
    xavier initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and m.weight is not None:
        torch.nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_()
        m.bias.data.fill_(0)


def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == 'warm0.1':
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.2':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.5':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    else:
        raise NotImplementedError(schedule_type)
    return betas


def get_dataset(dataroot, npoints, category):
    tr_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
                                        categories=[category], split='train',
                                        tr_sample_size=npoints,
                                        te_sample_size=npoints,
                                        scale=1.,
                                        normalize_per_shape=False,
                                        normalize_std_per_axis=False,
                                        random_subsample=True)
    te_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
                                        categories=[category], split='val',
                                        tr_sample_size=npoints,
                                        te_sample_size=npoints,
                                        scale=1.,
                                        normalize_per_shape=False,
                                        normalize_std_per_axis=False,
                                        all_points_mean=tr_dataset.all_points_mean,
                                        all_points_std=tr_dataset.all_points_std,
                                        )
    return tr_dataset, te_dataset


def get_dataloader(opt, train_dataset, test_dataset=None):
    if opt.distribution_type == 'multi':
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=opt.world_size,
            rank=opt.rank
        )
        if test_dataset is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset,
                num_replicas=opt.world_size,
                rank=opt.rank
            )
        else:
            test_sampler = None
    else:
        train_sampler = None
        test_sampler = None

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs, sampler=train_sampler,
                                                   shuffle=train_sampler is None, num_workers=int(opt.workers),
                                                   drop_last=True)

    if test_dataset is not None:
        test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs, sampler=test_sampler,
                                                      shuffle=False, num_workers=int(opt.workers), drop_last=False)
    else:
        test_dataloader = None

    return train_dataloader, test_dataloader, train_sampler, test_sampler



def train_vae(gpu, opt, output_dir):
    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = LionModel(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)
    set_seed(opt)

    logger = setup_logging(output_dir)

    assert opt.distribution_type != 'multi'

    if opt.distribution_type == 'multi':
        should_diag = gpu == 0
    else:
        should_diag = True
    if should_diag:
        outf_syn, = setup_output_subdirs(output_dir, 'syn')

    ''' data '''
    train_dataset, _ = get_dataset(opt.dataroot, opt.npoints, opt.category)
    dataloader, _, train_sampler, _ = get_dataloader(opt, train_dataset, None)

    if opt.distribution_type == 'multi':  # Multiple processes, single GPU per process
        def _transform_(m):
            return nn.parallel.DistributedDataParallel(
                m, device_ids=[gpu], output_device=gpu)

        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        # model.multi_gpu_wrapper(_transform_)

    elif opt.distribution_type == 'single':
        def _transform_(m):
            return nn.parallel.DataParallel(m)

        model = model.cuda()
        # model.multi_gpu_wrapper(_transform_)

    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        raise ValueError('distribution_type = multi | single | None')

    if should_diag:
        logger.info(opt)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.decay, betas=(opt.beta1, 0.999))

    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, opt.lr_gamma)

    if opt.model != '':
        ckpt = torch.load(opt.model)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])

    if opt.model != '':
        start_epoch = torch.load(opt.model)['epoch'] + 1
    else:
        start_epoch = 0

    def new_x_chain(x, num_chain):
        return torch.randn(num_chain, *x.shape[1:], device=x.device)

    for epoch in range(start_epoch, opt.niter):

        lr_scheduler.step(epoch)

        for i, data in enumerate(dataloader):
            x = data['train_points'].transpose(1, 2)

            '''
            train vae
            '''

            if opt.distribution_type == 'multi' or (opt.distribution_type is None and gpu is not None):
                x = x.cuda(gpu)
            elif opt.distribution_type == 'single':
                x = x.cuda()

            loss = model.get_loss_vae(x)

            print("loss:", loss.item())
            optimizer.zero_grad()
            loss.backward()
            # netpNorm, netgradNorm = getGradNorm(model)
            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)

            optimizer.step()



def train_ddm(gpu, opt, output_dir, noises_init):

    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = LionModel(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)
    set_seed(opt)

    logger = setup_logging(output_dir)

    assert opt.distribution_type != 'multi'

    if opt.distribution_type == 'multi':
        should_diag = gpu == 0
    else:
        should_diag = True
    if should_diag:
        outf_syn, = setup_output_subdirs(output_dir, 'syn')

    ''' data '''
    train_dataset, _ = get_dataset(opt.dataroot, opt.npoints, opt.category)
    dataloader, _, train_sampler, _ = get_dataloader(opt, train_dataset, None)

    if opt.distribution_type == 'multi':  # Multiple processes, single GPU per process
        def _transform_(m):
            return nn.parallel.DistributedDataParallel(
                m, device_ids=[gpu], output_device=gpu)

        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        # model.multi_gpu_wrapper(_transform_)

    elif opt.distribution_type == 'single':
        def _transform_(m):
            return nn.parallel.DataParallel(m)

        model = model.cuda()
        # model.multi_gpu_wrapper(_transform_)

    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        raise ValueError('distribution_type = multi | single | None')

    if should_diag:
        logger.info(opt)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.decay, betas=(opt.beta1, 0.999))

    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, opt.lr_gamma)

    if opt.model != '':
        ckpt = torch.load(opt.model)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])

    if opt.model != '':
        start_epoch = torch.load(opt.model)['epoch'] + 1
    else:
        start_epoch = 0

    def new_x_chain(x, num_chain):
        return torch.randn(num_chain, *x.shape[1:], device=x.device)

    for epoch in range(start_epoch, opt.niter):

        lr_scheduler.step(epoch)

        for i, data in enumerate(dataloader):
            x = data['train_points'].transpose(1, 2)
            noises_batch = noises_init[data['idx']].transpose(1, 2)

            '''
            train ddm
            '''

            if opt.distribution_type == 'multi' or (opt.distribution_type is None and gpu is not None):
                x = x.cuda(gpu)
                noises_batch = noises_batch.cuda(gpu)
            elif opt.distribution_type == 'single':
                x = x.cuda()
                noises_batch = noises_batch.cuda()

            if opt.ddm_stage == 'shape':
                loss = model.get_loss_shape_ddm(x, noises_batch).mean()
            else:
                loss = model.get_loss_point_ddm(x, noises_batch).mean()

            print("loss:", loss.item())
            optimizer.zero_grad()
            loss.backward()
            # netpNorm, netgradNorm = getGradNorm(model)

            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)

            optimizer.step()


def main():
    opt = parse_args()
    if opt.category == 'airplane':
        opt.beta_start = 1e-5
        opt.beta_end = 0.008
        opt.schedule_type = 'warm0.1'

    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    dir_id = os.path.dirname(__file__)
    output_dir = get_output_dir(dir_id, exp_id)
    copy_source(__file__, output_dir)

    ''' workaround '''
    train_dataset, _ = get_dataset(opt.dataroot, opt.npoints, opt.category)
    noises_init = torch.randn(len(train_dataset), opt.npoints, opt.nc)
    assert opt.distribution_type != 'multi'

    if opt.training_stage == 'vae':
        train_vae(opt.gpu, opt, output_dir)
    elif opt.training_stage == 'ddm':
        train_ddm(opt.gpu, opt, output_dir, noises_init)
    else:
        print("wrong training stage.")


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default=os.path.join(os.getcwd(), 'Data/ShapeNetCore.v2.PC15k/'))
    parser.add_argument('--category', default='airplane')

    parser.add_argument('--bs', type=int, default=16, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')

    parser.add_argument('--nc', default=3)
    parser.add_argument('--npoints', default=2048)

    '''train stage'''
    parser.add_argument('--training_stage', default='ddm', help='different training stage: vae | ddm')
    parser.add_argument('--ddm_stage', default='point', help='different ddm stage: shape | point')

    '''model'''
    parser.add_argument('--beta_start', default=0.0001)
    parser.add_argument('--beta_end', default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', default=1000)

    # params
    parser.add_argument('--attention', default=True)
    parser.add_argument('--dropout', default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')

    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for E, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--decay', type=float, default=0, help='weight decay for EBM')
    parser.add_argument('--grad_clip', type=float, default=None, help='weight decay for EBM')
    parser.add_argument('--lr_gamma', type=float, default=0.998, help='lr decay for EBM')

    parser.add_argument('--model', default='', help="path to model (to continue training)")

    '''GPU'''
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:9991', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distribution_type', default='single', choices=['multi', 'single', None],
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use. None means using all available GPUs.')

    '''eval'''
    parser.add_argument('--saveIter', default=100, help='unit: epoch')
    parser.add_argument('--diagIter', default=50, help='unit: epoch')
    parser.add_argument('--vizIter', default=50, help='unit: epoch')
    parser.add_argument('--print_freq', default=50, help='unit: iter')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    main()
