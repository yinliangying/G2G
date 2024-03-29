
#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import logging
import json
import os
import sys
import numpy as np
from argparse import Namespace
from itertools import chain
import pickle

import torch
from unicore import checkpoint_utils, distributed_utils, utils
from unicore import tasks, options
from unicore.logging import metrics, progress_bar
from search_strategies.parse import add_search_strategies_args
from search_strategies.beam_search_generator import SequenceGeneratorBeamSearch
from search_strategies import search
from search_strategies.simple_sequence_generator import SimpleGenerator
from search_strategies.greedy_generator import GreedyGenerator
from search_strategies.sample_generator import SampleGenerator
from utils import save_config
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("NAG2G_cli.validate")


def G2G_weight_reload(state):
    if state["model"]["degree_pe.weight"].shape[0] != 100:
        tmp_shape = state["model"]["degree_pe.weight"].shape
        tmp = torch.zeros((100, tmp_shape[1])).to(
            state["model"]["degree_pe.weight"].device
        )
        tmp[: tmp_shape[0]] = state["model"]["degree_pe.weight"]
        state["model"]["degree_pe.weight"] = tmp
    return state


def main(args):
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

    for subset in args.valid_subset.split(","):
        try:
            task.load_dataset(subset, combine=False, epoch=1, task_cfg=args.task)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        if not os.path.exists(args.results_path):
            try:
                os.makedirs(args.results_path)
            except:
                pass
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            batch_size=args.batch_size,
            ignore_invalid_inputs=True,
            seed=args.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=args.num_workers,
            data_buffer_size=args.data_buffer_size,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )
        np.random.seed(args.seed)
        # utils.set_torch_seed(args.seed)
        if args.task == "masked_lm":
            raise
            log_outputs = []
            for i, sample in enumerate(progress):
                sample = utils.move_to_cuda(sample) if use_cuda else sample
                if len(sample) == 0:
                    continue
                _loss, _sample_size, log_output = task.test_step(
                    args, sample, model, loss
                )
                progress.log(log_output, step=i)
                log_outputs.append(log_output)

            if data_parallel_world_size > 1:
                log_outputs = distributed_utils.all_gather_list(
                    log_outputs,
                    max_size=args.all_gather_list_size,
                    group=distributed_utils.get_data_parallel_group(),
                )
                log_outputs = list(chain.from_iterable(log_outputs))

            with metrics.aggregate() as agg:
                task.reduce_metrics(log_outputs, loss)
                log_output = agg.get_smoothed_values()

            progress.print(log_output, tag=subset, step=i)
        else:
            print(
                "test beam search params: ",
                args.search_strategies,
                args.beam_size,
                args.len_penalty,
                args.temperature,
            )

            if args.bpe_tokenizer_path == "none":
                dictionary = task.dictionary
            else:
                dictionary = task.infer_dictionary

            if args.search_strategies == "SequenceGeneratorBeamSearch_test":
                search_strategy = None
                generator = SequenceGeneratorBeamSearch(
                    [model],
                    dictionary,
                    beam_size=args.beam_size,
                    len_penalty=args.len_penalty,
                    max_len_b=args.max_seq_len - 1,
                    search_strategy = search_strategy,
                    eos = dictionary.index('[SEP2]'),
                )
                infer_beam_size_list = [5,2,1,1,1]
                generator2 = []
                for size in infer_beam_size_list:
                    model_tmp = SequenceGeneratorBeamSearch(
                        [model],
                        dictionary,
                        beam_size=size,
                        len_penalty=args.len_penalty,
                        max_len_b=args.max_seq_len - 1,
                        search_strategy = search_strategy,
                    )
                    generator2.append(model_tmp)

            elif args.search_strategies == "SequenceGeneratorBeamSearch":
                search_strategy = None
                generator = SequenceGeneratorBeamSearch(
                    [model],
                    dictionary,
                    beam_size=args.beam_size,
                    len_penalty=args.len_penalty,
                    max_len_b=args.max_seq_len - 1,
                    search_strategy = search_strategy,
                    # normalize_scores=False
                )
            elif args.search_strategies == "SimpleGenerator":
                generator = SimpleGenerator(
                    model,
                    dictionary,
                    beam_size=args.beam_size,
                    len_penalty=args.len_penalty,
                    max_seq_len=args.max_seq_len - 1,
                    args=args,
                )
            elif args.search_strategies == "GreedyGenerator":
                generator = GreedyGenerator(
                    model, dictionary, beam_size=args.beam_size
                )

            log_outputs = []
            for i, sample in enumerate(progress):
                sample = utils.move_to_cuda(sample) if use_cuda else sample
                if len(sample) == 0:
                    continue
                if args.search_strategies == "SequenceGeneratorBeamSearch_test":
                    pred, log_output = task.test_step(
                        args, sample, generator, loss, i, args.seed,
                    second_beam_size = args.beam_size_second,
                    second_token_size=args.beam_head_second,
                    model2 = generator2
                    )
                else:
                    pred, log_output = task.test_step(
                        args, sample, generator, loss, i, args.seed
                    )
                progress.log(log_output, step=i)
                log_outputs.append(log_output)
            if data_parallel_world_size > 1:
                log_outputs = distributed_utils.all_gather_list(
                    log_outputs,
                    max_size=450000000,
                    group=distributed_utils.get_data_parallel_group(),
                )
                log_outputs = list(chain.from_iterable(log_outputs))

            with metrics.aggregate() as agg:
                task.reduce_metrics(log_outputs, loss)
                log_output = agg.get_smoothed_values()

            progress.print(log_output, tag=subset, step=i)


def cli_main():

    # args = Namespace(no_progress_bar=False, log_interval=100, log_format='simple', tensorboard_logdir='',
    #                  wandb_project='', wandb_name='', seed=1, cpu=False, fp16=False, bf16=False, bf16_sr=False,
    #                  allreduce_fp32_grad=False, fp16_no_flatten_grads=False, fp16_init_scale=4, fp16_scale_window=256,
    #                  fp16_scale_tolerance=0.0, min_loss_scale=0.0001, threshold_loss_scale=None, user_dir='./NAG2G',
    #                  empty_cache_freq=0, all_gather_list_size=16384, suppress_crashes=False, profile=False,
    #                  ema_decay=-1.0,
    #                  validate_with_ema=False, loss='G2G', optimizer='adam', lr_scheduler='polynomial_decay',
    #                  task='G2G_unimolv2',
    #                  num_workers=5, skip_invalid_size_inputs_valid_test=False, batch_size=1,
    #                  required_batch_size_multiple=1,
    #                  data_buffer_size=1, train_subset='train', valid_subset='test', validate_interval=1,
    #                  validate_interval_updates=5000, validate_after_updates=0, fixed_validation_seed=11,
    #                  disable_validation=False,
    #                  batch_size_valid=16, max_valid_steps=None, curriculum=0, distributed_world_size=1,
    #                  distributed_rank=0,
    #                  distributed_backend='nccl', distributed_init_method='env://', distributed_port=-1, device_id=0,
    #                  distributed_no_spawn=True, ddp_backend='no_c10d', bucket_cap_mb=25, fix_batches_to_gpus=False,
    #                  find_unused_parameters=True, fast_stat_sync=False, broadcast_buffers=False, nprocs_per_node=1,
    #                  path='NAG2G_unimolplus_uspto_50k_20230513-222355/checkpoint_last.pt', quiet=False,
    #                  model_overrides='{}',
    #                  results_path='NAG2G_unimolplus_uspto_50k_20230513-222355/checkpoint_last', beam_size=10,
    #                  search_strategies='SimpleGenerator', len_penalty=0.0, temperature=1.0, beam_size_second=5,
    #                  beam_head_second=2, arch='NAG2G_G2G', num_3d_bias_kernel=128, droppath_prob=0.1, noise_scale=0.0,
    #                  label_prob=0.4,
    #                  mid_prob=0.2, mid_upper=0.6, mid_lower=0.4, plddt_loss_weight=0.01, pos_loss_weight=0.2, N_vnode=1,
    #                  position_type='sinusoidal', smoothl1_beta=1.0, decoder_type='new', reduced_head_dim=8,
    #                  encoder_type='unimolv2',
    #                  data='USPTO50K_brief_20230227', dict_name='dict_20230310.txt', laplacian_pe_dim=0,
    #                  use_reorder=True,
    #                  decoder_attn_from_loader=False, not_want_edge_input=False, want_charge_h=True, shufflegraph='none',
    #                  not_sumto2=False, add_len=0, N_left=0,
    #                  infer_save_name='smi_SimpleGenerator_lp0.0_t1_10_bhs2_bss5_b1_USPTO50K_brief_20230227.txt',
    #                  use_sep2=False, want_h_degree=True, use_class=False, dataset_uspto_full=False, infer_step=True,
    #                  idx_type=0,
    #                  want_decoder_attn=True, init_train_path='none', bpe_tokenizer_path='none', charge_h_last=False,
    #                  use_class_encoder=False, config_file='NAG2G_unimolplus_uspto_50k_20230513-222355/config.ini',
    #                  adam_betas='(0.9, 0.999)', adam_eps=1e-08, weight_decay=0.0, force_anneal=None, lr_shrink=0.1,
    #                  warmup_updates=12000, no_seed_provided=False, encoder_embed_dim=768, pair_embed_dim=128,
    #                  encoder_layers=6,
    #                  encoder_attention_heads=24, encoder_ffn_embed_dim=768, activation_fn='gelu', attention_dropout=0.1,
    #                  act_dropout=0.1, dropout=0.0, num_block=4, pretrain=False, pos_step_size=0.01,
    #                  gaussian_std_width=1.0,
    #                  gaussian_mean_start=0.0, gaussian_mean_stop=9.0, emb_dropout=0.1, activation_dropout=0.0,
    #                  pooler_dropout=0.0,
    #                  max_seq_len=512, post_ln=False, contrastive_global_negative=False, auto_regressive=True,
    #                  use_decoder=True,
    #                  class_embedding=False, decoder_layers=6, decoder_embed_dim=768, decoder_ffn_embed_dim=768,
    #                  decoder_attention_heads=24, decoder_loss=1, rel_pos=True, flag_old=False, q_reduced_before=False,
    #                  want_emb_k_dynamic_proj=False, want_emb_k_dynamic_dropout=True)
    #
    # main(args)
    sys.argv=['NAG2G/validate.py', 'USPTO50K_brief_20230227', '--user-dir', './NAG2G', '--valid-subset', 'test', '--task', 'G2G_unimolv2', '--loss', 'G2G', '--arch', 'NAG2G_G2G', '--encoder-type', 'unimolv2', '--seed', '1', '--infer_step', '--results-path', 'NAG2G_unimolplus_uspto_50k_20230513-222355/checkpoint_last', '--path', 'NAG2G_unimolplus_uspto_50k_20230513-222355/checkpoint_last.pt', '--num-workers', '5', '--ddp-backend=no_c10d', '--required-batch-size-multiple', '1', '--search_strategies', 'SimpleGenerator', '--beam-size', '10', '--len-penalty', '0.0', '--temperature', '1', '--beam-size-second', '5', '--beam-head-second', '2', '--infer_save_name', 'smi_SimpleGenerator_lp0.0_t1_10_bhs2_bss5_b1_USPTO50K_brief_20230227.txt', '--batch-size', '1', '--data-buffer-size', '1', '--fixed-validation-seed', '11', '--batch-size-valid', '1', '--config_file', 'NAG2G_unimolplus_uspto_50k_20230513-222355/config.ini']
    parser = options.get_validation_parser()
    add_search_strategies_args(parser)
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    args = save_config.read_config(args)
    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
