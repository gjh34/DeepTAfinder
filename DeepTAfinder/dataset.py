import numpy as np
import pandas as pd
import itertools
import os
import pickle
import re
import shutil
import torch
from torch.utils.data import Dataset
from typing import Sequence, Tuple, List, Union
from pathlib import Path
from esm.constants import proteinseq_toks

class TASequenceDataSet(Dataset):

    def __init__(self, file_path, batch_size, seed=42, shuffle=True, stratified=False, stratified_TA_non_TA=False):
        self.file_path = file_path
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.stratified = stratified
        self.stratified_TA_non_TA = stratified_TA_non_TA
        self.check_dataset()

    def check_dataset(self):
        tempdataset = pd.read_csv(self.file_path, delimiter='\t', header=None)
        '''
        if self.stratified:
            # Determine the pos sample size
            pos_sample_size = tempdataset[(tempdataset[0].str.startswith(('AT', 'T'))) & (tempdataset[1].str.startswith(('AT', 'T')))].shape[0]
            sample_size = tempdataset.shape[0]
            pos_sample = tempdataset.iloc[:pos_sample_size]
            neg_sample = tempdataset.iloc[pos_sample_size:]
            # 计算每个 batch 中应有的正负样本数量
            batch_size = self.batch_size
            positive_per_batch = int(batch_size * (pos_sample_size / sample_size))
            negative_per_batch = batch_size - positive_per_batch
            # 对正负样本进行分层抽样，并重新组合
            stratified_samples = []
            while len(pos_sample) >= positive_per_batch and len(neg_sample) >= negative_per_batch:
                positive_batch = pos_sample.sample(n=positive_per_batch, replace=False)
                negative_batch = neg_sample.sample(n=negative_per_batch, replace=False)
                batch = pd.concat([positive_batch, negative_batch])
                batch = batch.sample(frac=1).reset_index(drop=True) # batch内随机打乱
                stratified_samples.append(batch)
                pos_sample = pos_sample.drop(positive_batch.index)
                neg_sample = neg_sample.drop(negative_batch.index)
            # 处理最后一个 batch，如果还有不足一个batch_size的数据，将数据框填满
            if len(pos_sample) > 0 or len(neg_sample) > 0:
                last_batch = pd.concat([pos_sample, neg_sample])
                last_batch = last_batch.sample(frac=1).reset_index(drop=True) # batch内随机打乱
                stratified_samples.append(last_batch)
            # 将分层抽样的结果重新组合成一个数据框
            tempdataset = pd.concat(stratified_samples)
            tempdataset.to_csv('/public/guanjh/tmp.csv')
        '''
        if self.stratified:
            # Determine the pos sample size
            pos_sample_size = tempdataset[(tempdataset[0].str.startswith(('AT', 'T'))) & (tempdataset[1].str.startswith(('AT', 'T')))].shape[0]
            sample_size = tempdataset.shape[0]
            pos_sample = tempdataset.iloc[:pos_sample_size]
            neg_sample = tempdataset.iloc[pos_sample_size:]
            neg_sample_TA_nonTA = neg_sample[(neg_sample[0].str.startswith(('AT', 'T'))) | (neg_sample[1].str.startswith(('AT', 'T')))]
            neg_sample_operon = neg_sample[(~neg_sample[0].str.startswith(('AT', 'T'))) & (~neg_sample[1].str.startswith(('AT', 'T')))]
            # 计算每个 batch 中应有的正负样本数量
            batch_size = self.batch_size
            positive_per_batch = int(batch_size * (pos_sample_size / sample_size))
            negative_per_batch = batch_size - positive_per_batch
            negative_TA_nonTA_per_batch = int(negative_per_batch * (neg_sample_TA_nonTA.shape[0] / neg_sample.shape[0]))
            negative_operon_per_batch = negative_per_batch - negative_TA_nonTA_per_batch
            # 对正负样本进行分层抽样，并重新组合
            stratified_samples = []
            while len(pos_sample) >= positive_per_batch and len(neg_sample_TA_nonTA) >= negative_TA_nonTA_per_batch and len(neg_sample_operon) >= negative_operon_per_batch:
                positive_batch = pos_sample.sample(n=positive_per_batch, replace=False)
                negative_TA_nonTA_batch = neg_sample_TA_nonTA.sample(n=negative_TA_nonTA_per_batch, replace=False)
                negative_operon_batch = neg_sample_operon.sample(n=negative_operon_per_batch, replace=False)
                batch = pd.concat([positive_batch, negative_TA_nonTA_batch, negative_operon_batch])
                batch = batch.sample(frac=1).reset_index(drop=True) # batch内随机打乱
                stratified_samples.append(batch)
                pos_sample = pos_sample.drop(positive_batch.index)
                neg_sample_TA_nonTA = neg_sample_TA_nonTA.drop(negative_TA_nonTA_batch.index)
                neg_sample_operon = neg_sample_operon.drop(negative_operon_batch.index)
            # 处理最后一个 batch，如果还有不足一个batch_size的数据，将数据框填满
            if len(pos_sample) > 0 or len(neg_sample_TA_nonTA) > 0 or len(neg_sample_operon) > 0:
                last_batch = pd.concat([pos_sample, neg_sample_TA_nonTA, neg_sample_operon])
                last_batch = last_batch.sample(frac=1).reset_index(drop=True) # batch内随机打乱
                stratified_samples.append(last_batch)
            # 将分层抽样的结果重新组合成一个数据框
            tempdataset = pd.concat(stratified_samples)
            tempdataset.to_csv('/public/guanjh/tmp.csv')
        if self.stratified_TA_non_TA:
            # Determine the pos sample size
            pos_sample_size = tempdataset[(tempdataset[0].str.startswith(('AT', 'T'))) & (tempdataset[1].str.startswith(('AT', 'T')))].shape[0]
            sample_size = tempdataset.shape[0]
            pos_sample = tempdataset.iloc[:pos_sample_size]
            neg_sample = tempdataset.iloc[pos_sample_size:]
            neg_sample_TA_nonTA = neg_sample[(neg_sample[0].str.startswith(('AT', 'T'))) | (neg_sample[1].str.startswith(('AT', 'T')))]
            neg_sample_operon = neg_sample[(~neg_sample[0].str.startswith(('AT', 'T'))) & (~neg_sample[1].str.startswith(('AT', 'T')))]
            # 计算每个 batch 中应有的正负样本数量
            batch_size = self.batch_size
            positive_per_batch = int(batch_size * (pos_sample_size / sample_size))
            negative_per_batch = batch_size - positive_per_batch
            # 对正负样本进行分层抽样，并重新组合
            stratified_samples = []
            while len(pos_sample) >= positive_per_batch and len(neg_sample_TA_nonTA) + len(neg_sample_operon) >= negative_per_batch:
                positive_batch = pos_sample.sample(n=positive_per_batch, replace=False)
                positive_batch_ta_id = positive_batch[0].values.tolist() + positive_batch[1].values.tolist()
                negative_batch_TA_nonTA = neg_sample_TA_nonTA[neg_sample_TA_nonTA[0].isin(positive_batch_ta_id) | neg_sample_TA_nonTA[1].isin(positive_batch_ta_id)]
                negative_operon_per_batch = negative_per_batch - negative_batch_TA_nonTA.shape[0]
                if neg_sample_operon.shape[0] >= negative_operon_per_batch:
                    negative_batch_operon = neg_sample_operon.sample(n=negative_operon_per_batch, replace=False)
                else:
                    negative_batch_operon = neg_sample_operon
                batch = pd.concat([positive_batch, negative_batch_TA_nonTA, negative_batch_operon])
                batch = batch.sample(frac=1).reset_index(drop=True) # batch内随机打乱
                stratified_samples.append(batch)
                pos_sample = pos_sample.drop(positive_batch.index)
                neg_sample_TA_nonTA = neg_sample_TA_nonTA.drop(negative_batch_TA_nonTA.index)
                neg_sample_operon = neg_sample_operon.drop(negative_batch_operon.index)
            # 处理最后一个 batch，如果还有不足一个batch_size的数据，将数据框填满
            if len(pos_sample) > 0 or len(neg_sample_TA_nonTA) > 0 or len(neg_sample_operon) > 0:
                last_batch = pd.concat([pos_sample, neg_sample_TA_nonTA, neg_sample_operon])
                last_batch = last_batch.sample(frac=1).reset_index(drop=True) # batch内随机打乱
                stratified_samples.append(last_batch)
            # 将分层抽样的结果重新组合成一个数据框
            tempdataset = pd.concat(stratified_samples)
            tempdataset.to_csv('/public/guanjh/tmp.csv')
        if self.shuffle and self.stratified == False and self.stratified_TA_non_TA == False:
            tempdataset = tempdataset.sample(frac=1, random_state=self.seed) # Shuffle the data
        if tempdataset[1].astype(str).str.match('^\d').all():
            sequence_labels = tempdataset[1].to_list()
            tempdataset[2] = tempdataset[2].str[:1022] # Truncated to 1022
            sequence_strs = tempdataset[2].to_list()
        else:
            sequence_labels = tempdataset[2].to_list()
            tempdataset[3] = tempdataset[3].str[:1022] # Truncated to 1022
            tempdataset[4] = tempdataset[4].str[:1022] # Truncated to 1022
            sequence_strs = list(zip(tempdataset[3].to_list(), tempdataset[4].to_list()))

        self.sequence_labels = np.array(sequence_labels)
        self.sequence_strs = np.array(sequence_strs)

    def __getitem__(self, idx):
        label = self.sequence_labels[idx]
        seq_str = self.sequence_strs[idx]
        return label, seq_str

    def __len__(self):
        return len(self.sequence_labels)

RawMSA = Sequence[Tuple[str, str]]

class Alphabet(object):
    def __init__(
        self,
        standard_toks: Sequence[str],
        prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
        prepend_bos: bool = True,
        append_eos: bool = False,
        use_msa: bool = False,
        single_sequence: bool = False,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.use_msa = use_msa
        self.single_sequence = single_sequence

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        for i in range((8 - (len(self.all_toks) % 8)) % 8):
            self.all_toks.append(f"<null_{i  + 1}>")
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")
        self.all_special_tokens = ['<eos>', '<unk>', '<pad>', '<cls>', '<mask>']
        self.unique_no_split_tokens = self.all_toks

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def get_batch_converter(self, truncation_seq_length: int = None):
        if self.single_sequence:
            return BatchConverterSingle(self, truncation_seq_length)
        elif self.use_msa:
            return MSABatchConverter(self, truncation_seq_length)
        else:
            return BatchConverter(self, truncation_seq_length)

    @classmethod
    def from_architecture(cls, name: str) -> "Alphabet":
        if name in ("ESM-1", "protein_bert_base"):
            standard_toks = proteinseq_toks["toks"]
            prepend_toks: Tuple[str, ...] = ("<null_0>", "<pad>", "<eos>", "<unk>")
            append_toks: Tuple[str, ...] = ("<cls>", "<mask>", "<sep>")
            prepend_bos = True
            append_eos = False
            use_msa = False
            single_sequence = False
        elif name in ("ESM-1b", "roberta_large"):
            standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = True
            use_msa = False
            single_sequence = False
        elif name == "roberta_large_single":
            standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = True
            use_msa = False
            single_sequence = True
        elif name in ("MSA Transformer", "msa_transformer"):
            standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = False
            use_msa = True
            single_sequence = False
        elif "invariant_gvp" in name.lower():
            standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<null_0>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>", "<cath>", "<af2>")
            prepend_bos = True
            append_eos = False
            use_msa = False
            single_sequence = False
        else:
            raise ValueError("Unknown architecture selected")
        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, use_msa, single_sequence)

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:
        """
        Inspired by https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
        Converts a string in a sequence of tokens, using the tokenizer.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.

        Returns:
            :obj:`List[str]`: The list of tokens.
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # AddedToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # We strip left and right by default
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in self.tokenize(text)]


class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet, truncation_seq_length: int = None):
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, raw_batch):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_pair_list = zip(*raw_batch)        
        # seq_str_pair_list is a list of TA pairs. seq_str_list.shape (xxxx, 2)
        seq_str_list_1 = np.array(seq_str_pair_list)[:, 0] # First column sequences, shape=(data_size, 1)
        seq_str_list_2 = np.array(seq_str_pair_list)[:, 1] # Second column sequences, shape=(data_size, 1)

        strs_list = []
        toks_list = []
        for seq_str_list in [seq_str_list_1, seq_str_list_2]:            
            seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list]
            if self.truncation_seq_length:
                seq_encoded_list = [seq_str[:self.truncation_seq_length] for seq_str in seq_encoded_list]
            #max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
            max_len = 1022
            tokens = torch.empty(
                (
                    batch_size,
                    max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
                ),
                dtype=torch.int64,
            )
            tokens.fill_(self.alphabet.padding_idx)
            labels = []
            strs = []

            for i, (label, seq_str, seq_encoded) in enumerate(
                zip(batch_labels, seq_str_list, seq_encoded_list)
            ):
                labels.append(label)
                strs.append(seq_str)
                if self.alphabet.prepend_bos:
                    tokens[i, 0] = self.alphabet.cls_idx
                seq = torch.tensor(seq_encoded, dtype=torch.int64)
                tokens[
                    i,
                    int(self.alphabet.prepend_bos) : len(seq_encoded)
                    + int(self.alphabet.prepend_bos),
                ] = seq
                if self.alphabet.append_eos:
                    tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
            
            strs_list.append(strs)
            toks_list.append(tokens)
            
        strs = list(zip(strs_list[0], strs_list[1]))
        toks = list(zip(toks_list[0], toks_list[1]))

        return batch_labels, strs, toks


class BatchConverterSingle(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet, truncation_seq_length: int = None):
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_list = zip(*raw_batch)
        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list]
        if self.truncation_seq_length:
            seq_encoded_list = [seq_str[:self.truncation_seq_length] for seq_str in seq_encoded_list]
        #max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        max_len = 1022
        tokens = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, (label, seq_str, seq_encoded) in enumerate(
            zip(batch_labels, seq_str_list, seq_encoded_list)
        ):
            labels.append(label)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[
                i,
                int(self.alphabet.prepend_bos) : len(seq_encoded)
                + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx

        return labels, strs, tokens
        

class BatchConverter_merge(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet, truncation_seq_length: int = None):
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, raw_batch):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_pair_list = zip(*raw_batch)
        # seq_str_pair_list is a list of TA pairs. seq_str_list.shape (xxxx, 2)
        seq_str_list_1 = np.array(seq_str_pair_list)[:, 0] # First column sequences, shape=(data_size, 1)
        seq_str_list_2 = np.array(seq_str_pair_list)[:, 1] # Second column sequences, shape=(data_size, 1)
        seq_str_list_zip = list(zip(seq_str_list_1, seq_str_list_2))
        seq_str_list_str_concat = [x[0] + x[1] for x in seq_str_list_zip]

        seq_encoded_list_1 = [self.alphabet.encode(seq_str) for seq_str in seq_str_list_1]
        seq_encoded_list_2 = [self.alphabet.encode(seq_str) for seq_str in seq_str_list_2]
        if self.truncation_seq_length:
            seq_encoded_list_1 = [seq_str[:self.truncation_seq_length] for seq_str in seq_encoded_list_1]
            seq_encoded_list_2 = [seq_str[:self.truncation_seq_length] for seq_str in seq_encoded_list_2]
        seq_encoded_list_zip = list(zip(seq_encoded_list_1, seq_encoded_list_2))
        seq_encoded_list_str_concat = [x[0] + x[1] for x in seq_encoded_list_zip]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list_str_concat)

        tokens = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos) + 1,
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        
        labels = []
        strs = []

        for i in range(0, len(seq_str_list_1)):
            label = batch_labels[i]
            seq_str = seq_str_list_str_concat[i]
            seq_encoded_1 = seq_encoded_list_1[i]
            seq_encoded_2 = seq_encoded_list_2[i]            
            
            labels.append(label)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq_1 = torch.tensor(seq_encoded_1, dtype=torch.int64)
            seq_2 = torch.tensor(seq_encoded_2, dtype=torch.int64)
            
            tokens[
                i,
                int(self.alphabet.prepend_bos) : len(seq_encoded_1) + int(self.alphabet.prepend_bos),
            ] = seq_1
            
            tokens[
                i,
                len(seq_encoded_1) + int(self.alphabet.prepend_bos) : len(seq_encoded_1) + int(self.alphabet.prepend_bos) + 1,
            ] = "<sep>"

            tokens[
                i,
                len(seq_encoded_1) + int(self.alphabet.prepend_bos) + 1: len(seq_encoded_1) + int(self.alphabet.prepend_bos) + 1 + len(seq_encoded_2),
            ] = seq_2
           
            if self.alphabet.append_eos:
                tokens[i, len(seq_encoded_1) + int(self.alphabet.prepend_bos) + 1 + len(seq_encoded_2)] = self.alphabet.eos_idx
        
        return batch_labels, strs, tokens


class MSABatchConverter(BatchConverter):
    def __call__(self, inputs: Union[Sequence[RawMSA], RawMSA]):
        if isinstance(inputs[0][0], str):
            # Input is a single MSA
            raw_batch: Sequence[RawMSA] = [inputs]  # type: ignore
        else:
            raw_batch = inputs  # type: ignore

        batch_size = len(raw_batch)
        max_alignments = max(len(msa) for msa in raw_batch)
        max_seqlen = max(len(msa[0][1]) for msa in raw_batch)

        tokens = torch.empty(
            (
                batch_size,
                max_alignments,
                max_seqlen + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, msa in enumerate(raw_batch):
            msa_seqlens = set(len(seq) for _, seq in msa)
            if not len(msa_seqlens) == 1:
                raise RuntimeError(
                    "Received unaligned sequences for input to MSA, all sequence "
                    "lengths must be equal."
                )
            msa_labels, msa_strs, msa_tokens = super().__call__(msa)
            labels.append(msa_labels)
            strs.append(msa_strs)
            tokens[i, : msa_tokens.size(0), : msa_tokens.size(1)] = msa_tokens

        return labels, strs, tokens


def read_fasta(
    path,
    keep_gaps=True,
    keep_insertions=True,
    to_upper=False,
):
    with open(path, "r") as f:
        for result in read_alignment_lines(
            f, keep_gaps=keep_gaps, keep_insertions=keep_insertions, to_upper=to_upper
        ):
            yield result


def read_alignment_lines(
    lines,
    keep_gaps=True,
    keep_insertions=True,
    to_upper=False,
):
    seq = desc = None

    def parse(s):
        if not keep_gaps:
            s = re.sub("-", "", s)
        if not keep_insertions:
            s = re.sub("[a-z]", "", s)
        return s.upper() if to_upper else s

    for line in lines:
        # Line may be empty if seq % file_line_width == 0
        if len(line) > 0 and line[0] == ">":
            if seq is not None:
                yield desc, parse(seq)
            desc = line.strip().lstrip(">")
            seq = ""
        else:
            assert isinstance(seq, str)
            seq += line.strip()
    assert isinstance(seq, str) and isinstance(desc, str)
    yield desc, parse(seq)

