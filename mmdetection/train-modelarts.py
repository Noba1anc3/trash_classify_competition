import os
import os.path as osp
import copy
import time
import torch
import argparse

try:
    import moxing as mox
except:
    print('not use moxing')

# 安装依赖
root_dir = os.path.dirname(__file__)
mmcv_path = os.path.join(root_dir, 'mmcv-0.5.8-cp36-cp36m-linux_x86_64.whl')
mmdet_path = os.path.join(root_dir, 'mmdet-2.0.0+unknown-cp36-cp36m-linux_x86_64.whl')
os.system(f'pip install {mmcv_path}; pip install {mmdet_path}')

import mmcv
from mmcv import Config, DictAction
# from mmcv.runner import init_dist  分布式才用的到
from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger


def prepare_data_on_modelarts(args):
    """
    将OBS上的数据拷贝到ModelArts中
    """
    # 拷贝预训练参数文件

    # 默认使用ModelArts中的如下两个路径用于存储数据：
    # 0) /cache/model: 如果适用预训练模型，存储从OBS拷贝过来的预训练模型
    # 1）/cache/datasets: 存储从OBS拷贝过来的训练数据
    # 2）/cache/log: 存储训练日志和训练模型，并且在训练结束后，该目录下的内容会被全部拷贝到OBS

    # mox.file.copy可同时兼容本地和OBS路径的拷贝操作
    # 拷贝一个目录得用copy_parallel接口

    # 相对路径变绝对路径
    root_dir = os.path.dirname(__file__)
    args.config = os.path.join(root_dir, args.config)
    args.work_dir = os.path.join(root_dir, args.work_dir)

    # 创建文件夹
    for path in [os.path.join(args.local_data_root, p) for p in ['model', 'datasets', 'log']]:
        if not os.path.exists(path):
            os.makedirs(path)

    # 预训练模型拷贝到/cache/model
    if args.resume_from:
        # 如若从output下取模型文件，使用绝对路径，则删除下面一行
        # args.resume_from = os.path.join(root_dir, args.resume_from)
        _, weights_name = os.path.split(args.resume_from)
        mox.file.copy(args.resume_from, os.path.join(args.local_data_root, 'model/' + weights_name))
        args.resume_from = os.path.join(args.local_data_root, 'model/' + weights_name)

    # trainval压缩包拷贝到/cache/datasets，并解压缩
    if not (args.data_url.startswith('s3://') or args.data_url.startswith('obs://')):
        args.data_local = args.data_url
    else:
        args.data_local = os.path.join(args.local_data_root, 'datasets/VOCdevkit')
        if not os.path.exists(args.data_local):                      # VOCdevkit目录不存在
            data_dir = os.path.join(args.local_data_root, 'datasets')
            mox.file.copy_parallel(args.data_url, data_dir)
            os.system('cd %s;unzip VOCdevkit.zip' % data_dir)        # 进入datasets，解压trainval.zip
            if os.path.isdir(args.data_local):                       # 解压成功
                os.system('cd %s;rm VOCdevkit.zip' % data_dir)       # 进入datasets，删除trainval.zip
                print('unzip VOCdevkit.zip success, args.data_local is', args.data_local)
            else:                                                    # 解压失败
                raise Exception('unzip VOCdevkit.zip Failed')
        else:                                                        # VOCdevkit目录已经存在
            print('args.data_local: %s is already exist, skip copy' % args.data_local)

    # 创建日志文件夹/cache/log
    if not (args.train_url.startswith('s3://') or args.train_url.startswith('obs://')):
        args.train_local = args.train_url
    else:
        args.train_local = os.path.join(args.local_data_root, 'log/')
        if not os.path.exists(args.train_local):
            os.mkdir(args.train_local)

    return args


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', default='configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py',
                        help='train config file path')
    parser.add_argument('--work-dir', default='work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', default='', type=str,
        help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use'
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')

    # 新增
    parser.add_argument('--local_data_root', default='/cache/', type=str,
                        help='a directory used for transfer data between local path and OBS path')
    parser.add_argument('--data_url', required=True, type=str,
                        help='the training and validation data path')
    parser.add_argument('--data_local', default='', type=str,
                        help='the training and validation data path on local')
    parser.add_argument('--train_local', default='', type=str,
                        help='the training output results on local')
    parser.add_argument('--train_url', required=True, type=str,
                        help='the path to save training outputs')
    parser.add_argument('--init_method', default='', type=str, help='the training output results on local')

    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    args = prepare_data_on_modelarts(args)

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    
    # 用于加载已训练模型
    if args.resume_from:
        cfg.resume_from = args.resume_from

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        pass
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    train_detector(model,datasets,cfg,distributed=distributed,validate=(not args.no_validate),timestamp=timestamp,meta=meta)

    # 将work_dir复制到obs中
    target_dir = os.path.join(args.train_url, 'work_dir')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    mox.file.copy_parallel(args.work_dir, target_dir)


if __name__ == '__main__':
    main()

