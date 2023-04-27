#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Tannon Kew

Example call:
    python evaluate_simplfication.py ../../llm_inference/resources/outputs/llama-13b/asset-test_asset-valid_p0_fs3_nr1_s287.jsonl

"""
import logging
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Generator

from utils import convert_to_json
from metric.evaluator import get_evaluator

logger = logging.getLogger(__name__)

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('hyp_file', type=str, help='Path to a JSONL file with model outputs and optionally source and reference sentences.')
    parser.add_argument('--src_file', type=str, required=False, help='Path to a JSONL file with source and optionally reference sentences.')
    parser.add_argument('--ref_file', type=str, required=False, help='Path to a TXT file with reference sentences. WARING: assumes only one reference set.')
    parser.add_argument('--task', type=str, required=False, default='summarization', help='Task to evaluate as. Options: tasks implememnted in UniEval')
    # parser.add_argument('--out_file', type=str, required=False, help='Path to a CSV file with metric scores.')
    return parser.parse_args()

# helper functions (taken from https://github.com/tannonk/llm_inference/blob/main/utils/helpers.py)
def iter_text_lines(file: Union[str, Path]) -> Generator[str, None, None]:
    """Generator that yields lines from a regular text file."""
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            else:
                yield line

def iter_json_lines(file: Union[str, Path]) -> Generator[Dict, None, None]:
    """Fetch dictionary-object lines from a JSONL file"""
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            else:
                yield json.loads(line)

def iter_split_lines(file: Union[str, Path], delimiter: str = '\t', src_key: str = 'complex', tgt_key: str = 'simple') -> Generator[Dict, None, None]:
    """Fetch dictionary-object lines from a TSV file"""
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split(delimiter)
            if len(line) == 0:
                continue
            line_d = {src_key: line[0], tgt_key: line[1:]}
            yield line_d

def iter_lines(file: Union[str, Path]) -> Generator[Union[str, Dict], None, None]:
    """Wraps `iter_text_lines` and `iter_json_lines` to fetch lines from file"""
    if str(file).endswith(".jsonl") or str(file).endswith(".json"):
        return iter_json_lines(file)
    elif str(file).endswith(".tsv"):
        return iter_split_lines(file, delimiter='\t')
    else:
        return iter_text_lines(file)

def load_data(args):
    """
    Load data from files.
    
    Input files may be in jsonl or txt/tsv format.
    """
    # model outputs can be in jsonl or txt format
    if args.hyp_file.endswith('.txt') or args.hyp_file.endswith('.tsv'):
        hyp_sents = list(iter_lines(args.hyp_file))
    elif args.hyp_file.endswith('.jsonl'):
        lines = list(iter_lines(args.hyp_file))
        hyp_sents = [i['model_output'] for i in lines]
    
    # simplest case: if src and ref files are provided, load them
    if args.src_file and args.ref_file:
        logger.info(f'Loading src_file {args.src_file}')
        src_sents = list(iter_lines(args.src_file))
        logger.info(f'Loading ref_file {args.ref_file}')
        refs_sents = [list(iter_lines(args.ref_file))]
    
    # if only src file is provided, assume that human refs are also int the src file
    elif args.src_file and not args.ref_file:
        logger.info(f'No ref_file provided. Assuming that src and refs are in src_file {args.src_file}')
        lines = list(iter_lines(args.src_file))
        src_sents = [i['complex'] for i in lines]
        refs_sents = [i['simple'] for i in lines]
    
    # otherwise, if no src file and no ref file are provided, assume src and human refs are in hyp file (jsonl format)
    elif not args.src_file and not args.ref_file:
        logger.info(f'No src_file or ref_file provided. Assuming that src and refs are in hyp_file {args.hyp_file}')
        lines = list(iter_lines(args.hyp_file))
        src_sents = [i['source'] for i in lines]
        refs_sents = [i['references'] for i in lines]

    # breakpoint()
    refs_sents = list(map(list, [*zip(*refs_sents)])) # transpose from [# samples, # refs_per_sample] to [# refs_per_sample, # samples]
    refs_sents = refs_sents[0] # assume only one reference set

    # sanity checks
    if len(src_sents) != len(refs_sents):
        raise ValueError('Number of source sentences does not match number of reference sentences')
    if len(src_sents) != len(hyp_sents):
        raise ValueError('Number of source sentences does not match number of hypothesis sentences')

    return src_sents, refs_sents, hyp_sents

def run_evaluation(src_list: List[str], ref_list: List[str], output_list: List[str], task: str = 'summarization'):
    print('------------------')
    print(f'Evaluating as a {task.upper()} task')
    print('------------------')

    # Prepare data for pre-trained evaluators
    data = convert_to_json(output_list=output_list, 
                        src_list=src_list, ref_list=ref_list)

    # breakpoint()
    # Initialize evaluator for a specific task
    evaluator = get_evaluator(task)

    # Get multi-dimensional evaluation scores
    eval_scores = evaluator.evaluate(data, print_result=True)
    breakpoint()
    return eval_scores

if __name__ == '__main__':
    args = set_args()
    src_sents, refs_sents, hyp_sents = load_data(args)
    run_evaluation(src_sents, refs_sents, hyp_sents, task=args.task)
