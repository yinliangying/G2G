from argparse import Namespace
import ast
import logging
import json
import os
import sys
import numpy as np
from argparse import Namespace
from itertools import chain
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from unicore import checkpoint_utils, distributed_utils, utils
from unicore import tasks, options
from unicore.logging import metrics, progress_bar
from search_strategies.parse import add_search_strategies_args

from search_strategies.beam_search_generator import SequenceGeneratorBeamSearch
from search_strategies.simple_sequence_generator import SimpleGenerator
from search_strategies.greedy_generator import GreedyGenerator
from search_strategies.sample_generator import SampleGenerator


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("NAG2G_cli.validate")



args=Namespace(no_progress_bar=False, log_interval=100, log_format='simple', tensorboard_logdir='',
wandb_project='', wandb_name='', seed=1, cpu=False, fp16=False, bf16=False, bf16_sr=False,
allreduce_fp32_grad=False, fp16_no_flatten_grads=False, fp16_init_scale=4, fp16_scale_window=256,
fp16_scale_tolerance=0.0, min_loss_scale=0.0001, threshold_loss_scale=None, user_dir='./NAG2G',
empty_cache_freq=0, all_gather_list_size=16384, suppress_crashes=False, profile=False, ema_decay=-1.0,
validate_with_ema=False, loss='G2G', optimizer='adam', lr_scheduler='polynomial_decay', task='G2G_unimolv2',
num_workers=5, skip_invalid_size_inputs_valid_test=False, batch_size=1, required_batch_size_multiple=1,
data_buffer_size=1, train_subset='train', valid_subset='test', validate_interval=1,
validate_interval_updates=5000, validate_after_updates=0, fixed_validation_seed=11, disable_validation=False,
batch_size_valid=16, max_valid_steps=None, curriculum=0, distributed_world_size=1, distributed_rank=0,
distributed_backend='nccl', distributed_init_method='env://', distributed_port=-1, device_id=0,
distributed_no_spawn=True, ddp_backend='no_c10d', bucket_cap_mb=25, fix_batches_to_gpus=False,
find_unused_parameters=True, fast_stat_sync=False, broadcast_buffers=False, nprocs_per_node=1,
path='NAG2G_unimolplus_uspto_50k_20230513-222355/checkpoint_last.pt', quiet=False, model_overrides='{}',
results_path='NAG2G_unimolplus_uspto_50k_20230513-222355/checkpoint_last', beam_size=10,
search_strategies='SimpleGenerator', len_penalty=0.0, temperature=1.0, beam_size_second=5,
beam_head_second=2, arch='NAG2G_G2G', num_3d_bias_kernel=128, droppath_prob=0.1, noise_scale=0.0, label_prob=0.4,
mid_prob=0.2, mid_upper=0.6, mid_lower=0.4, plddt_loss_weight=0.01, pos_loss_weight=0.2, N_vnode=1,
position_type='sinusoidal', smoothl1_beta=1.0, decoder_type='new', reduced_head_dim=8, encoder_type='unimolv2',
data='USPTO50K_brief_20230227', dict_name='dict_20230310.txt', laplacian_pe_dim=0, use_reorder=True,
decoder_attn_from_loader=False, not_want_edge_input=False, want_charge_h=True, shufflegraph='none',
not_sumto2=False, add_len=0, N_left=0,
infer_save_name='smi_SimpleGenerator_lp0.0_t1_10_bhs2_bss5_b1_USPTO50K_brief_20230227.txt',
use_sep2=False, want_h_degree=True, use_class=False, dataset_uspto_full=False, infer_step=True, idx_type=0,
want_decoder_attn=True, init_train_path='none', bpe_tokenizer_path='none', charge_h_last=False,
use_class_encoder=False, config_file='NAG2G_unimolplus_uspto_50k_20230513-222355/config.ini',
adam_betas='(0.9, 0.999)', adam_eps=1e-08, weight_decay=0.0, force_anneal=None, lr_shrink=0.1,
warmup_updates=12000, no_seed_provided=False, encoder_embed_dim=768, pair_embed_dim=128, encoder_layers=6,
encoder_attention_heads=24, encoder_ffn_embed_dim=768, activation_fn='gelu', attention_dropout=0.1,
act_dropout=0.1, dropout=0.0, num_block=4, pretrain=False, pos_step_size=0.01, gaussian_std_width=1.0,
gaussian_mean_start=0.0, gaussian_mean_stop=9.0, emb_dropout=0.1, activation_dropout=0.0, pooler_dropout=0.0,
max_seq_len=512, post_ln=False, contrastive_global_negative=False, auto_regressive=True, use_decoder=True,
class_embedding=False, decoder_layers=6, decoder_embed_dim=768, decoder_ffn_embed_dim=768,
decoder_attention_heads=24, decoder_loss=1, rel_pos=True, flag_old=False, q_reduced_before=False,
want_emb_k_dynamic_proj=False, want_emb_k_dynamic_dropout=True)

def init(args):
    assert (
        args.batch_size is not None
    ), "Must specify batch size either with --batch-size"

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    if args.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    overrides = ast.literal_eval(args.model_overrides)

    logger.info("loading model(s) from {}".format(args.path))
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)

    if args.task == "G2G":
        state = G2G_weight_reload(state)

    model.load_state_dict(state["model"], strict=False)

    # Move models to GPU
    # for model in models:
    #     if use_fp16:
    #         model.half()
    #     if use_cuda:
    #         model.cuda()
    if use_fp16:
        model = model.half()
    if use_cuda:
        model.cuda()

    # Print args
    logger.info(args)

    # Build loss
    loss = task.build_loss(args)
    loss.eval()
    model.eval()
    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("loss: {}".format(loss.__class__.__name__))
    logger.info(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(getattr(p, "_orig_size", p).numel() for p in model.parameters()),
            sum(
                getattr(p, "_orig_size", p).numel()
                for p in model.parameters()
                if p.requires_grad
            ),
        )
    )

    np.random.seed(args.seed)
    print(
        "test beam search params: ",
        args.search_strategies,
        args.beam_size,
        args.len_penalty,
        args.temperature,
    )
    if args.search_strategies == "SequenceGeneratorBeamSearch":
        # raise
        generator = SequenceGeneratorBeamSearch(
            [model],
            task.dictionary,
            beam_size=args.beam_size,
            len_penalty=args.len_penalty,
            max_len_b=1024
        )
    elif args.search_strategies == "SimpleGenerator":
        generator = SimpleGenerator(
            model,
            task.dictionary,
            beam_size=args.beam_size,
            len_penalty=args.len_penalty,
            args=args,
        )
    elif args.search_strategies == "GreedyGenerator":
        generator = GreedyGenerator(model, task.dictionary, beam_size=args.beam_size)
    # dataset_empty = task.load_empty_dataset(seed=args.seed)

    return (
        args,
        use_cuda,
        task,
        generator,
        data_parallel_world_size,
        data_parallel_rank,
        # dataset_empty,
    )



def main(args):
    model_tuple = init(args)
    smiles = [
        "CC(=O)c1ccc2c(ccn2C(=O)OC(C)(C)C)c1",
        "[CH3:1][Si:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])/[CH:8]=[CH:9]/[CH2:10][Br:11]",
        "[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[N:8]1[CH2:9][CH2:10][c:11]2[o:12][c:13]3[c:14]([Cl:15])[cH:16][c:17]([S:18](=[O:19])[c:20]4[cH:21][cH:22][cH:23][cH:24][cH:25]4)[cH:26][c:27]3[c:28]2[CH2:29]1",
        "[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[n:8]1[cH:9][n:10][c:11]([CH:12]=[O:13])[cH:14]1",
        "[Cl:1][c:2]1[cH:3][c:4]([Cl:5])[c:6]([CH2:7][Br:8])[cH:9][n:10]1",
        "[CH3:1][O:2][c:3]1[n:4][c:5]2[cH:6][cH:7][c:8]([C:9](=[O:10])[c:11]3[cH:12][n:13][n:14][n:15]3[CH3:16])[cH:17][c:18]2[c:19]([Cl:20])[c:21]1[CH2:22][c:23]1[cH:24][cH:25][c:26]([C:27]([F:28])([F:29])[F:30])[cH:31][cH:32]1",
        "[CH3:1][C:2](=[O:3])[c:4]1[n:5][c:6]2[cH:7][c:8]([NH:9][C:10](=[O:11])[c:12]3[cH:13][cH:14][c:15](/[CH:16]=[CH:17]/[C:18]([F:19])([F:20])[F:21])[cH:22][c:23]3[CH3:24])[cH:25][cH:26][c:27]2[s:28]1",
        "[CH3:1][CH2:2][O:3][C:4](=[O:5])[CH:6]1[CH2:7][CH2:8][C:9]2([CH2:10][CH2:11]1)[O:12][CH2:13][CH2:14][O:15]2",
        "[CH3:1][CH2:2][CH2:3][CH2:4][c:5]1[n:6][cH:7][c:8]([C:9]([CH3:10])=[O:11])[n:12]1[CH2:13][c:14]1[cH:15][cH:16][cH:17][cH:18][c:19]1[Cl:20]",
        "[CH3:1][C:2]1([c:3]2[cH:4][c:5]3[cH:6][cH:7][cH:8][n+:9]([O-:10])[c:11]3[nH:12]2)[CH2:13][CH2:14]1",
        "[O:1]=[CH:2][c:3]1[cH:4][cH:5][c:6]([F:7])[c:8]([N+:9](=[O:10])[O-:11])[cH:12]1",
        "[CH3:1][c:2]1[cH:3][cH:4][cH:5][c:6]([C:7]#[C:8][c:9]2[cH:10][cH:11][c:12](-[c:13]3[cH:14][cH:15][n:16][n:17]3[CH3:18])[cH:19][cH:20]2)[n:21]1",
        "[NH2:1][c:2]1[cH:3][cH:4][c:5](-[c:6]2[c:7]([F:8])[cH:9][cH:10][cH:11][c:12]2[C:13]([F:14])([F:15])[F:16])[cH:17][c:18]1[N+:19](=[O:20])[O-:21]",
    ]
    smiles = [setmap2smiles(i) for i in smiles]
    run(smiles, model_tuple)