import os
import time
import builtins
import datetime
import subprocess
import logging
import torch
import torch.distributed as dist

from collections import defaultdict, deque

from .fileio import TorchLoader


class SmoothedValue(object):
    def __init__(self, window_size=1000, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t", log_file=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.logger = self._create_logger(log_file)  # Create a logger object
    
    def _create_logger(self, log_file):
        logger = logging.getLogger('MetricLogger')
        logger.setLevel(logging.INFO)

        # Create a file handler to log to a text file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create a stream handler to log to the terminal
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        # Create a formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        return logger        
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))
    def print_log(self, msg):
        # Log the message using the logger
        self.logger.info(msg)
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, epoch=0, total_epoch=50):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        total_iters = len(iterable)*total_epoch
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            current_iters = len(iterable)*epoch+i
            if current_iters % print_freq == 0 or current_iters == total_iters - 1:
                if is_main_process():
                    eta_seconds = iter_time.global_avg * ( total_iters - current_iters)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    if torch.cuda.is_available():
                        self.logger.info(log_msg.format(
                            current_iters, total_iters, eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB))
                    else:
                        self.logger.info(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            now = datetime.datetime.now().time()
            if force:
                builtin_print('[{} RANK: {}] '.format(now, get_rank()), end='')
            else:
                builtin_print('[{}] '.format(now), end='')
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def cleanup():
    dist.destroy_process_group()


def barrier():
    dist.barrier()


def all_gather_object(object_list, obj, group=None):
    dist.all_gather_object(object_list,
                           obj,
                           group)


def init_distributed_mode(args):
    '''
    rank, world_size, gpu, local_rank, port, distributed, dist_url
    '''
    if 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        
        num_gpus = torch.cuda.device_count()
        args.gpu = args.rank % num_gpus
        args.local_rank = args.gpu

        node_list = os.environ['SLURM_NODELIST']
        addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')

        os.environ['MASTER_PORT'] = str(getattr(args, 'port', '29529'))
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(args.world_size)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.local_rank = args.gpu
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(args.rank, args.dist_url, args.gpu), flush=True)
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def save_on_master(path, data):
    if is_main_process():
        loader = TorchLoader()
        loader.save(path, data)
    dist.barrier()


def save_model(save_path,
               epoch,
               model,
               model_disc,
               optimizer,
               optimizer_disc):
    to_save = {
        'model': model.state_dict(),
        'model_disc': model_disc.state_dict(),
        'opt': optimizer.state_dict(),
        'opt_disc': optimizer_disc.state_dict(),
        'epoch': epoch,
    }
    save_on_master(save_path, to_save)


def save_model_no_disc(save_path,
               epoch,
               model,
               optimizer,
             ):
    to_save = {
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'epoch': epoch,
    }
    save_on_master(save_path, to_save)
def resume_model(args,
                 model,
                 model_disc,
                 optimizer,
                 optimizer_disc,
                 strict=True):
    if not args.resume:
        return
    print('Resume training from:', args.resume)
    loader = TorchLoader()
    checkpoint = loader.load(args.resume)
    
    model.load_state_dict(checkpoint['model'], strict = strict)
    model_disc.load_state_dict(checkpoint['model_disc'],  strict = strict)
    # optimizer.load_state_dict(checkpoint['opt'], strict = strict)
    optimizer_disc.load_state_dict(checkpoint['opt_disc'])
    args.start_epoch = checkpoint['epoch'] + 1

def resume_model_no_disc(args,
                 model,
                 optimizer):
    if not args.resume:
        return
    print('Resume training from:', args.resume)
    loader = TorchLoader()
    checkpoint = loader.load(args.resume)

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['opt'])
    args.start_epoch = checkpoint['epoch'] + 1
    
def load_pretrained_model(checkpoint_path,
                          model):
    loader = TorchLoader()
    checkpoint = loader.load(checkpoint_path)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print('Loading trained checkpoint from:', checkpoint_path, msg)
