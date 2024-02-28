import torch
import torch.distributed as dist
from torch import Tensor
import torch.nn.functional as F
import os 
import subprocess
from torch.hub import download_url_to_file

def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    dist.barrier()

    return rank % num_gpus

def reduce_tensor(x:Tensor):
    rt=x.clone()
    dist.all_reduce(rt,op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()

    return rt

class AverageMeter:
    """
    Compute and stores the average and current value
    """
    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self,val,n=1):
        self.val=val
        self.sum+=val*n
        self.count+=n
        self.avg=self.sum/self.count

def print_model_params(model, logger, details=False):
    logger.info(f"=> print model params")
    sum = 0
    for name, param in model.named_parameters():
        mul = 1
        for size in param.shape:
            mul *= size
        sum += mul
        if details:
            print(f'{name:14s} : {param.shape}, {sum}')
    logger.info(f"total number of params: {sum}")


def save_checkpoint(epoch, model, optimizer, lr_schdeduler, logger, args, best=False):
    save_state={
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'lr_scheduler':lr_schdeduler.state_dict(),
        'epoch':epoch
    }
    save_path=os.path.join(args.output, args.exp, f'ckpt.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

# return start epoch
# load LAVT model
def load_checkpoint(args, model_without_ddp, optimizer, lr_scheduler, logger, epoch, best=False):
    root_path=args.output
    exp_path=args.exp
    ckpt_epoch = epoch

    ckpt_path = os.path.join(root_path, exp_path, f'ckpt.pth')
    checkpoint=torch.load(ckpt_path, map_location='cpu')
    msg=model_without_ddp.load_state_dict(checkpoint['model'],strict=False)
    logger.info(msg)
    # resume not evaluation
    if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch=checkpoint['epoch']+1
    logger.info(f"=> loaded successfully '{exp_path}' (epoch {checkpoint['epoch']})")
    del checkpoint
    torch.cuda.empty_cache()

# load pretrained swin transformer
def load_pretrained_swin(config, model, logger):
    logger.info(f"Loading weight {config.PRETRAIN.PATH} for fine-tuning......")
    if not os.path.exists(config.PRETRAIN.PATH):
        download_url_to_file("https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth",
            config.PRETRAIN.PATH)
    checkpoint=torch.load(config.PRETRAIN.PATH,map_location='cpu')
    state_dict=checkpoint['model']
    
    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # # check classifier, if not match, then re-init classifier to zero
    # head_bias_pretrained = state_dict['head.bias']
    # Nc1 = head_bias_pretrained.shape[0]
    # Nc2 = model.head.bias.shape[0]
    # if (Nc1 != Nc2):
    #     if Nc1 == 21841 and Nc2 == 1000:
    #         logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
    #         map22kto1k_path = f'data/map22kto1k.txt'
    #         with open(map22kto1k_path) as f:
    #             map22kto1k = f.readlines()
    #         map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
    #         state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
    #         state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
    #     else:
    #         torch.nn.init.constant_(model.head.bias, 0.)
    #         torch.nn.init.constant_(model.head.weight, 0.)
    #         del state_dict['head.weight']
    #         del state_dict['head.bias']
    #         logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

    # delete classification head since we do not use it.
    del state_dict['head.weight']
    del state_dict['head.bias']

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

def criterion(input:Tensor, target:Tensor):
    """
    Input: [N,2,H,W]
    target: [N,H,W]
    """
    return F.cross_entropy(input, target)

def loss_with_exist(args, output, exist_pred, target, exist):
    exist = exist.squeeze(1).long()
    has_exist = torch.sum(exist == 1) > 0
    has_no_exist = torch.sum(exist == 0) > 0
    loss = torch.Tensor([0]).cuda()
    if has_exist:
        loss = loss + criterion(output[0][exist == 1], target[exist == 1])
    if has_no_exist:
        loss = loss + args.neg_loss_weight * criterion(output[0][exist == 0], target[exist == 0])
    if args.use_exist:
        loss = loss + args.exist_loss_weight * F.binary_cross_entropy(exist_pred.squeeze(1), exist.float())
    if args.use_mask:
        for i in range(1, len(output)):
            if has_exist:
                loss = loss + args.aux_loss_weight * criterion(output[i][exist == 1], target[exist == 1])
            if has_no_exist:
                loss = loss + args.aux_loss_weight * args.neg_loss_weight * criterion(output[i][exist == 0], target[exist == 0])
    
    return loss

def loss_with_aux(args, output, target):
    loss = criterion(output[0], target)
    if args.use_mask:
        for i in range(1, len(output)):
            loss = loss + args.aux_loss_weight * criterion(output[i], target)

    return loss