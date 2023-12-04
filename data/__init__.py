from functools import partial
import numpy as np

from torch.utils.data.distributed import DistributedSampler
from torch import Generator, randperm
from torch.utils.data import DataLoader, Subset

import core.util as Util
from core.praser import init_obj


def define_dataloader(logger, opt):
    """ create train/test dataloader and validation dataloader,  validation dataloader is None when phase is test or not GPU 0 """
    '''create dataset and set random seed'''
    dataloader_args = opt['datasets'][opt['phase']]['dataloader']['args']
    worker_init_fn = partial(Util.set_seed, gl_seed=opt['seed'])

    phase_dataset, val_dataset = define_dataset(logger, opt) # phase_dataset是训练集，val_dataset是验证集（当opt中validation_split是2时，val_dataset中是2张图）

    '''create datasampler'''
    data_sampler = None
    if opt['distributed']:
        data_sampler = DistributedSampler(phase_dataset, shuffle=dataloader_args.get('shuffle', False), num_replicas=opt['world_size'], rank=opt['global_rank']) # 分布式的，将数据集分成多个部分，每个部分分给一个GPU
        dataloader_args.update({'shuffle':False}) # sampler option is mutually exclusive with shuffle  分布式的常规操作
    
    ''' create dataloader and validation dataloader '''
    dataloader = DataLoader(phase_dataset, sampler=data_sampler, worker_init_fn=worker_init_fn, **dataloader_args)
    ''' val_dataloader don't use DistributedSampler to run only GPU 0! '''
    if opt['global_rank']==0 and val_dataset is not None: # 如果是GPU 0，且验证集不为空（只在GPU 0 上进行验证）
        dataloader_args.update(opt['datasets'][opt['phase']]['dataloader'].get('val_args',{}))
        val_dataloader = DataLoader(val_dataset, worker_init_fn=worker_init_fn, **dataloader_args) 
    else:
        val_dataloader = None
    return dataloader, val_dataloader


def define_dataset(logger, opt):
    ''' loading Dataset() class from given file's name '''
    dataset_opt = opt['datasets'][opt['phase']]['which_dataset'] # <class 'core.praser.NoneDict'> 是自定义的类
    phase_dataset = init_obj(dataset_opt, logger, default_file_name='data.dataset', init_type='Dataset')
    val_dataset = None

    valid_len = 0
    data_len = len(phase_dataset)
    if 'debug' in opt['name']:
        debug_split = opt['debug'].get('debug_split', 1.0)
        if isinstance(debug_split, int):
            data_len = debug_split
        else:
            data_len *= debug_split

    dataloder_opt = opt['datasets'][opt['phase']]['dataloader'] #batch_size，num_workers，shuffle等参数
    valid_split = dataloder_opt.get('validation_split', 0)   # 如果key找不到validation_split，返回0
    
    ''' divide validation dataset, valid_split==0 when phase is test or validation_split is 0. '''
    if valid_split > 0.0 or 'debug' in opt['name']: 
        if isinstance(valid_split, int):
            assert valid_split < data_len, "Validation set size is configured to be larger than entire dataset."
            valid_len = valid_split
        else:
            valid_len = int(data_len * valid_split)
        data_len -= valid_len # 从训练集中减去验证集的长度（这里验证集是2个图像）
        phase_dataset, val_dataset = subset_split(dataset=phase_dataset, lengths=[data_len, valid_len], generator=Generator().manual_seed(opt['seed'])) # 将数据集分为训练集和验证集
    
    logger.info('Dataset for {} have {} samples.'.format(opt['phase'], data_len))
    if opt['phase'] == 'train':
        logger.info('Dataset for {} have {} samples.'.format('val', valid_len))   
    return phase_dataset, val_dataset

def subset_split(dataset, lengths, generator):
    """ #按照lengths（是一个list，分别表示该有多少张图片在里面）对datasets进行划分
    split a dataset into non-overlapping new datasets of given lengths. main code is from random_split function in pytorch
    """
    indices = randperm(sum(lengths), generator=generator).tolist() # 使用 PyTorch 中的随机数生成器，通过手动设置种子值，以确保在每次运行时都能获得相同的伪随机数序列
    Subsets = []
    for offset, length in zip(np.add.accumulate(lengths), lengths): # offset第一次是27998，第二次是28000
        if length == 0:
            Subsets.append(None)
        else:
            Subsets.append(Subset(dataset, indices[offset - length : offset]))
    return Subsets
