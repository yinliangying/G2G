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
from  rdkit.Chem import AllChem
from rdkit import Chem
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


# def main(args):
#     assert (
#         args.batch_size is not None
#     ), "Must specify batch size either with --batch-size"
#
#     use_fp16 = args.fp16
#     use_cuda = torch.cuda.is_available() and not args.cpu
#
#     if use_cuda:
#         torch.cuda.set_device(args.device_id)
#
#     if args.distributed_world_size > 1:
#         data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
#         data_parallel_rank = distributed_utils.get_data_parallel_rank()
#     else:
#         data_parallel_world_size = 1
#         data_parallel_rank = 0
#
#     overrides = ast.literal_eval(args.model_overrides)
#
#     logger.info("loading model(s) from {}".format(args.path))
#     state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
#     task = tasks.setup_task(args)
#     model = task.build_model(args)
#
#     if args.task == "G2G":
#         state = G2G_weight_reload(state)
#
#     model.load_state_dict(state["model"], strict=False)
#
#     # Move models to GPU
#     # for model in models:
#     #     if use_fp16:
#     #         model.half()
#     #     if use_cuda:
#     #         model.cuda()
#     if use_fp16:
#         model = model.half()
#     if use_cuda:
#         model.cuda()
#
#     # Print args
#     logger.info(args)
#
#     # Build loss
#     loss = task.build_loss(args)
#     loss.eval()
#     model.eval()
#     logger.info(model)
#     logger.info("task: {}".format(task.__class__.__name__))
#     logger.info("model: {}".format(model.__class__.__name__))
#     logger.info("loss: {}".format(loss.__class__.__name__))
#     logger.info(
#         "num. model params: {:,} (num. trained: {:,})".format(
#             sum(getattr(p, "_orig_size", p).numel() for p in model.parameters()),
#             sum(
#                 getattr(p, "_orig_size", p).numel()
#                 for p in model.parameters()
#                 if p.requires_grad
#             ),
#         )
#     )
#
#     for subset in args.valid_subset.split(","):
#         try:
#             task.load_dataset(subset, combine=False, epoch=1, task_cfg=args.task)
#             dataset = task.dataset(subset)
#         except KeyError:
#             raise Exception("Cannot find dataset: " + subset)
#
#         if not os.path.exists(args.results_path):
#             try:
#                 os.makedirs(args.results_path)
#             except:
#                 pass
#         # Initialize data iterator
#         itr = task.get_batch_iterator(
#             dataset=dataset,
#             batch_size=args.batch_size,
#             ignore_invalid_inputs=True,
#             seed=args.seed,
#             num_shards=data_parallel_world_size,
#             shard_id=data_parallel_rank,
#             num_workers=args.num_workers,
#             data_buffer_size=args.data_buffer_size,
#         ).next_epoch_itr(shuffle=False)
#         progress = progress_bar.progress_bar(
#             itr,
#             log_format=args.log_format,
#             log_interval=args.log_interval,
#             prefix=f"valid on '{subset}' subset",
#             default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
#         )
#         np.random.seed(args.seed)
#         # utils.set_torch_seed(args.seed)
#         if args.task == "masked_lm":
#             raise
#             log_outputs = []
#             for i, sample in enumerate(progress):
#                 sample = utils.move_to_cuda(sample) if use_cuda else sample
#                 if len(sample) == 0:
#                     continue
#                 _loss, _sample_size, log_output = task.test_step(
#                     args, sample, model, loss
#                 )
#                 progress.log(log_output, step=i)
#                 log_outputs.append(log_output)
#
#             if data_parallel_world_size > 1:
#                 log_outputs = distributed_utils.all_gather_list(
#                     log_outputs,
#                     max_size=args.all_gather_list_size,
#                     group=distributed_utils.get_data_parallel_group(),
#                 )
#                 log_outputs = list(chain.from_iterable(log_outputs))
#
#             with metrics.aggregate() as agg:
#                 task.reduce_metrics(log_outputs, loss)
#                 log_output = agg.get_smoothed_values()
#
#             progress.print(log_output, tag=subset, step=i)
#         else:
#             print(
#                 "test beam search params: ",
#                 args.search_strategies,
#                 args.beam_size,
#                 args.len_penalty,
#                 args.temperature,
#             )
#
#             if args.bpe_tokenizer_path == "none":
#                 dictionary = task.dictionary
#             else:
#                 dictionary = task.infer_dictionary
#
#             if args.search_strategies == "SequenceGeneratorBeamSearch_test":
#                 search_strategy = None
#                 generator = SequenceGeneratorBeamSearch(
#                     [model],
#                     dictionary,
#                     beam_size=args.beam_size,
#                     len_penalty=args.len_penalty,
#                     max_len_b=args.max_seq_len - 1,
#                     search_strategy = search_strategy,
#                     eos = dictionary.index('[SEP2]'),
#                 )
#                 infer_beam_size_list = [5,2,1,1,1]
#                 generator2 = []
#                 for size in infer_beam_size_list:
#                     model_tmp = SequenceGeneratorBeamSearch(
#                         [model],
#                         dictionary,
#                         beam_size=size,
#                         len_penalty=args.len_penalty,
#                         max_len_b=args.max_seq_len - 1,
#                         search_strategy = search_strategy,
#                     )
#                     generator2.append(model_tmp)
#
#             elif args.search_strategies == "SequenceGeneratorBeamSearch":
#                 search_strategy = None
#                 generator = SequenceGeneratorBeamSearch(
#                     [model],
#                     dictionary,
#                     beam_size=args.beam_size,
#                     len_penalty=args.len_penalty,
#                     max_len_b=args.max_seq_len - 1,
#                     search_strategy = search_strategy,
#                     # normalize_scores=False
#                 )
#             elif args.search_strategies == "SimpleGenerator":
#                 generator = SimpleGenerator(
#                     model,
#                     dictionary,
#                     beam_size=args.beam_size,
#                     len_penalty=args.len_penalty,
#                     max_seq_len=args.max_seq_len - 1,
#                     args=args,
#                 )
#             elif args.search_strategies == "GreedyGenerator":
#                 generator = GreedyGenerator(
#                     model, dictionary, beam_size=args.beam_size
#                 )
#
#             log_outputs = []
#             for i, sample in enumerate(progress):
#                 sample = utils.move_to_cuda(sample) if use_cuda else sample
#                 if len(sample) == 0:
#                     continue
#                 if args.search_strategies == "SequenceGeneratorBeamSearch_test":
#                     pred, log_output = task.test_step(
#                         args, sample, generator, loss, i, args.seed,
#                     second_beam_size = args.beam_size_second,
#                     second_token_size=args.beam_head_second,
#                     model2 = generator2
#                     )
#                 else:
#                     pred, log_output = task.test_step(
#                         args, sample, generator, loss, i, args.seed
#                     )
#                 progress.log(log_output, step=i)
#                 log_outputs.append(log_output)
#             if data_parallel_world_size > 1:
#                 log_outputs = distributed_utils.all_gather_list(
#                     log_outputs,
#                     max_size=450000000,
#                     group=distributed_utils.get_data_parallel_group(),
#                 )
#                 log_outputs = list(chain.from_iterable(log_outputs))
#
#             with metrics.aggregate() as agg:
#                 task.reduce_metrics(log_outputs, loss)
#                 log_output = agg.get_smoothed_values()
#
#             progress.print(log_output, tag=subset, step=i)
#

def setmap2smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.RemoveHs(mol)
    [atom.SetAtomMapNum(idx + 1) for idx, atom in enumerate(mol.GetAtoms())]
    return Chem.MolToSmiles(mol)





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


def run(smiles, model_tuple, seed=42):
    (
        args,
        use_cuda,
        task,
        generator,
        data_parallel_world_size,
        data_parallel_rank,
        # dataset_empty,
    ) = model_tuple

    dataset_empty = task.load_empty_dataset(init_values=smiles, seed=seed)
    # dataset_empty.put_smiles_in(smiles)
    dataset = task.dataset("test")
    itr = task.get_batch_iterator(
        dataset=dataset,
        batch_size=len(smiles),  # args.batch_size,
        ignore_invalid_inputs=True,
        seed=args.seed,
        num_shards=data_parallel_world_size,
        shard_id=data_parallel_rank,
        num_workers=args.num_workers,
        data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)

    for i, sample in enumerate(itr):
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        # sample = utils.move_to_cuda(dataset[0]) if use_cuda else sample
        # sample_tmp = {"net_input": {}, "target": ()}
        # for k, v in sample.items():
        #     if "net_input." in k:
        #         tmp_k = k.replace("net_input.", "")
        #         sample_tmp["net_input"][tmp_k] = v.unsqueeze(0)
        #     elif "target." in k:
        #         tmp_k = k.replace("net_input.", "")
        #         sample_tmp["net_input"][tmp_k] = v.unsqueeze(0)
        # sample = sample_tmp
        result = task.infer_step(sample, generator)
        print(result)
        return result


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


def cli_main():


    parser = options.get_validation_parser()
    add_search_strategies_args(parser)
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    args = save_config.read_config(args)
    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
