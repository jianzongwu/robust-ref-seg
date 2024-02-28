import argparse

def get_parser():
    parser=argparse.ArgumentParser()
    # Exp name
    parser.add_argument('--exp', default='test', help='the exp name, used for saving checkpoint')
    parser.add_argument('--use_exist', action='store_true', help='use exist prediction')
    # Random Seed
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # Dataset
    parser.add_argument('--dataset', default='refcoco', help='choose one of the following datasets: refcoco, refcoco+, refcocog')
    # BERT 
    parser.add_argument("--num_max_tokens", default=20, type=int, help="Number of max tokens in a sentence")
    # RefSegFormer
    parser.add_argument("--num_mem", default=20, type=int, help="Number of memory tokens")
    parser.add_argument("--num_neg_mem", default=10, type=int, help="Number of negative memory tokens")
    parser.add_argument("--n_fuse", default=3, type=int, help="Number of VLTF fusion modules")
    parser.add_argument("--use_pixel_decoder", action='store_true', help="use Deformable FPN")
    parser.add_argument("--hidden_dim", default=256, type=int, help="Hidden dimension of VLTF")
    parser.add_argument('--position_embedding', default='sine', help='position embedding type')
    parser.add_argument('--aux_loss_weight', default=0.4, type=float, help='aux loss weight')
    parser.add_argument('--neg_loss_weight', default=0.4, type=float, help='negative loss weight')
    parser.add_argument('--exist_loss_weight', default=1.0, type=float, help='exist pred loss weight')
    parser.add_argument('--use_mask', action='store_true', help='use auxiliary masks')
    # REFER
    parser.add_argument('--refer_data_root', default='./data', help='REFER dataset root directory')
    parser.add_argument('--splitBy', default='unc', help='split By')
    parser.add_argument('--split',default='val',help='split to run test')
    # config file
    parser.add_argument("--cfg_file",default="configs/swin_base_patch4_window12_480.yaml",type=str,help="config file define dataset and model")
    # optimizer set
    parser.add_argument("--lr", default=3e-5, type=float, help="initial learning rate")
    parser.add_argument("--weight-decay", default=0.01, type=float, help="weight-decay")
    # polynomial learning rate set
    parser.add_argument("--end_lr", default=1e-5, type=float, help="end_learning_rate")
    parser.add_argument("--power", default=0.9, type=float, help="power of polynomial learning rate")
    parser.add_argument("--max_decay_steps", default=50, type=int, help="max_decay_steps for polynomial learning ")
    parser.add_argument("--warm_up_steps", default=4000, type=int, help="max_decay_steps for polynomial learning ")
    # training set
    parser.add_argument("--batch_size",default=1,type=int,help="batch size per GPU")
    parser.add_argument("--epoch",default=50,type=int,help="training epoch")
    parser.add_argument("--print_freq",default=100,type=int,help="the frequent of print")
    parser.add_argument("--size",default=480,type=int,help="the size of image")
    parser.add_argument("--resume",action="store_true",help="start from a check point")
    parser.add_argument("--start_epoch",default=0,type=int,help="start epoch")
    # Only evaluate
    parser.add_argument("--ckpt_epoch",default=39,type=int,help="epoch of the ckpt")
    parser.add_argument("--eval", action="store_true", help="Only run evaluation")
    parser.add_argument("--type",default='val',type=str,help="[train,val,testA,testB]")
    # we provide two evaluate mode to better use all sentence to make predict
    parser.add_argument("--eval_mode",default='all',type=str,help="['cat', 'all']")
    # Save check point
    parser.add_argument("--output",default="./logs",type=str,help="log dir")
    # Distributed training parameters
    parser.add_argument("--world-size", default=2, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--local_rank", default=-1, type=int, help="local rank")
    return parser