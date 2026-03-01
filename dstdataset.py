from __future__ import annotations

import json
import logging
import operator
from collections import defaultdict

import torch

from dataclasses import dataclass, field
from typing import Union
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class Vocabulary:
    special_tokens: dict[str, Union[str, list[str]]]
    vocabulary_update: bool = False

    def add_special_tokens(self, tokens: dict[str, str]):
        for purpose, token in tokens.items():
            token = token.strip()
            for value in self.special_tokens.values():
                if token in value:
                    break
            else:
                self.special_tokens["additional_special_tokens"].append(token)
        if tokens:
            self.vocabulary_update = True


def pad(sentences, pad_id):
    max_len = max((map(len, sentences)))
    attention_mask = []
    sentences_pad = []
    for sent in sentences:
        pad_len = max_len - len(sent)
        # Using right padding so have the pad ids in the right
        sentences_pad.append(sent + [pad_id] * pad_len)
        attention_mask.append([1] * len(sent) + [0] * pad_len)
    return sentences_pad, attention_mask

def pad_left(sentences, pad_id):
    max_len = max((map(len, sentences)))
    attention_mask = []
    sentences_pad = []
    for sent in sentences:
        pad_len = max_len - len(sent)
        # Using left padding so have the pad ids in the left
        sentences_pad.append([pad_id] * pad_len + sent)
        attention_mask.append([0] * pad_len + [1] * len(sent))
    return sentences_pad, attention_mask

class DSTDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, filename, data_size):
        self.args = args
        self.data_size = data_size
        self.tokenizer = tokenizer
        self.filename = filename
        self.pad_id = tokenizer.pad_token_id  # pad to max example length
        # TODO: Check that this is indeed working properly for decoding, check for both T5 and GPT2
        self.ignore_token_id = -100
        self.max_seq_len = args.max_seq_len
        with open(filename, 'r') as f:
            self.data = json.load(f)
        self.requires_sampler = True

    def __len__(self):  # required
        return len(self.examples)

    def __getitem__(self, index):  # required
        return self.examples[index]


class TrainDataset(DSTDataset):
    def __init__(self, args, tokenizer, filename, data_size):
        super().__init__(args, tokenizer, filename, data_size)
        self._create_examples()


    def _create_examples(self):

        self.examples = []
        for example in tqdm(
                self.data,
                desc=f"Loading {self.filename}\n",
                disable=self.args.verbose.disable_display
        ):
            if self.data_size != -1 and len(self.examples) >= self.data_size:
                break
            context = example['dst_input']
            bs_str = example['belief_state']
            example_id = example['example_id']
            context_ids = self.tokenizer(context)['input_ids']
            target_ids = self.tokenizer(bs_str)['input_ids']
            target_len = len(target_ids)
            if self.args.model_type == "encoder-decoder":
                input_ids = context_ids
                label_ids = target_ids
            elif self.args.model_type == "decoder":
                # dialogue_context <BOS> belief_state <EOS>
                input_ids = context_ids + [self.tokenizer.bos_token_id] + target_ids + [self.tokenizer.eos_token_id]
                pad_len = len(input_ids) - target_len - 1  # eos_id
                label_ids = [self.ignore_token_id] * pad_len + target_ids + [self.tokenizer.eos_token_id]
                assert len(input_ids) == len(label_ids)
            else:
                raise ValueError(f"Invalid model type: {self.args.model_type}. It must be either 'decoder' or 'encoder-decoder'")
            if len(input_ids) >= self.max_seq_len:  # handle over-length example
                logger.warning(f"{example_id} exceeds max seq len, truncating...")
                input_ids = input_ids[-(self.max_seq_len - 1):]
                label_ids = label_ids[-(self.max_seq_len - 1):]
            assert len(input_ids) <= self.max_seq_len
            # user_utterance used for sanity check
            self.examples.append(
                {
                    'input_ids': input_ids,
                    'label_ids': label_ids,
                    'user_utterance': context.split("<USR>")[-1],  # useful for results analysis
                    'example_id': example_id,
                })
        logger.info(f'Data Statistics: {self.filename} -> {len(self.examples)} examples')

    def collate_fn(self, batch):
        input_ids = [example['input_ids'] for example in batch]
        input_ids, attention_mask = pad(input_ids, self.pad_id)
        input_ids = torch.tensor(input_ids).long()
        attention_mask = torch.tensor(attention_mask).long()
        user_utterances = [example['user_utterance'] for example in batch]
        label_ids = [example['label_ids'] for example in batch]
        label_ids, _ = pad(label_ids, self.ignore_token_id)
        label_ids = torch.tensor(label_ids).long()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label_ids': label_ids,
            'user_utterance': user_utterances
        }


class TestDataset(DSTDataset):
    def __init__(self, args, tokenizer, filename, data_size):
        super().__init__(args, tokenizer, filename, data_size)
        self.tokenizer.padding_side = "left"
        self.batch_size = args.batch_size
        self.to_decode = set(args.decode_only)  # type: set[str]

    def create_examples(self):
        self.examples = []
        for example in tqdm( self.data, desc=f"Loading {self.filename}",
                disable=self.args.verbose.disable_display):
            example_id = example["example_id"]
            dial_id = example_id.split("-")[0]
            if self.to_decode and dial_id not in self.to_decode:
                continue
            if self.data_size != -1 and len(self.examples) >= self.data_size:
                break
            context = example['dst_input']
            dst_input_ids = self.tokenizer(context)['input_ids']
            if self.args.model_type == "encoder-decoder":
                pass
            elif self.args.model_type == "decoder":
                dst_input_ids += [self.tokenizer.bos_token_id]
            else:
                raise ValueError(f"Invalid model type: {self.args.model_type}. It must be either 'decoder' or 'encoder-decoder'.")
            if len(dst_input_ids) >= self.max_seq_len:
                logger.warning(f"{example_id} of length {len(dst_input_ids)} exceeds max seq len of {self.max_seq_len}, truncating...")
                dst_input_ids = dst_input_ids[-(self.max_seq_len - 1):] # Truncating and retaining the last max_seq_len - 1 tokens
            self.examples.append( {'input_ids': dst_input_ids, 'example_id': example_id,
                                    'user_utterance': context.split("<USR>")[-1]}
            )
        if self.args.sort_data:
            self.examples.sort(key=lambda x: len(x['input_ids']))
        # Doing this so that when decoding with the batches, it is slightly more efficient since the batches are sorted by length
        # and they wont have to be padded as much
        logger.info(f'Data Statistics: {self.filename} -> {len(self.examples)} examples')

    def collate_fn(self, batch):
        input_ids = [example['input_ids'] for example in batch]
        input_ids, attention_mask = pad_left(input_ids, self.pad_id)
        input_ids = torch.tensor(input_ids).long()
        attention_mask = torch.tensor(attention_mask).long()
        example_id = [ex['example_id'] for ex in batch]
        user_utterances = [ex['user_utterance'] for ex in batch]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'example_id': example_id,
            'user_utterance': user_utterances
        }
    
class TestSlotValueHistoryDataset(DSTDataset):
    def __init__(self, args, tokenizer, filename, data_size):
        super().__init__(args, tokenizer, filename, data_size)
        self.tokenizer.padding_side = "left"
        self.batch_size = args.batch_size
        self.to_decode = set(args.decode_only)  # type: set[str]
    
    def create_examples(self):
        self.data_dict = {} # This will be a dictionary of example_id -> {input, user_utterance, dialogue_id}
        dialogues = {} # This will be a dictionary of dialogue_id -> number of turns in the dialogue
        for example in tqdm(self.data, desc=f"Loading {self.filename}", 
                            disable=self.args.verbose.disable_display):
            example_id = example["example_id"]
            dial_id = example_id.split("-")[0]
            if self.to_decode and dial_id not in self.to_decode:
                continue
            if self.data_size != -1 and len(self.data_dict) >= self.data_size:
                break
            context = example['dst_input']
            if self.args.model_type == "encoder-decoder":
                dst_input = context
            elif self.args.model_type == "decoder":
                dst_input = context + self.tokenizer.bos_token
            else:
                raise ValueError(f"Invalid model type: {self.args.model_type}. It must be either 'decoder' or 'encoder-decoder'.")
            # Note that due to the complexities involved in appending the history, we will not be using the tokenizer here
            # The tokenizer will be used in the actual decoding function instead
            user_utterance = context.split("<USR>")[-1]
            self.data_dict[example_id] = {'input': dst_input, 'user_utterance': user_utterance, 'dialogue_id': dial_id}
            if dial_id not in dialogues:
                dialogues[dial_id] = 1
            else:
                dialogues[dial_id] += 1
        
        all_dialogues = list(dialogues.keys())
        all_dialogues.sort(key=lambda x: dialogues[x], reverse=True)

        to_dictionary = {key: dialogues[key] for key in all_dialogues}
        in_dictionary = {}

        all_batches = []
        current_batch = []
        batch_size = self.batch_size

        x = True
        while x:
            max_index = min(batch_size, len(all_dialogues))
            elements = all_dialogues[:max_index]
            for element in elements:
                turn_id = in_dictionary.get(element, 0) # 0 if not in dictionary
                example_id = f"{element}-{turn_id}"
                values_left = to_dictionary[element]
                in_dictionary[element] = turn_id + 1
                if values_left == 1:
                    to_dictionary.pop(element)
                    all_dialogues.remove(element)
                    if not bool(to_dictionary):
                        current_batch.append(example_id)
                        all_batches.append(current_batch)
                        x = False
                        break
                else: # More than one value left
                    to_dictionary[element] = values_left - 1
                current_batch.append(example_id)
            all_batches.append(current_batch)
            current_batch = []
        
        self.examples = all_batches

    def __getitem__(self, index):
        example_ids = self.examples[index]
        user_utterances = [self.data_dict[example_id]['user_utterance'] for example_id in example_ids]
        inputs = [self.data_dict[example_id]['input'] for example_id in example_ids]
        dialogue_ids = [self.data_dict[example_id]['dialogue_id'] for example_id in example_ids]
        return {
            'inputs': inputs,
            'user_utterance': user_utterances,
            'example_id': example_ids,
            'dialogue_ids': dialogue_ids
        }
    
    def collate_fn(self, batch):
        return batch

if __name__ == '__main__':
    pass