import json
import math
import os
import copy
import pickle
import time
import warnings
import glob
import torch
from torch import nn, Tensor
from torch.utils.data import dataset
from tempfile import TemporaryDirectory

import hydra
import common_utils

from typing import List
from typing import Optional, Tuple
from typing import Optional, Any, Union, Callable

from datasets import load_dataset
import matplotlib.pyplot as plt

from torchtext.datasets import WikiText2, WikiText103
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from transformers import OpenAIGPTConfig, AutoTokenizer, OpenAIGPTLMHeadModel 

from decoder_wiki_util import TransformerModel, TransConfig, generate_square_subsequent_mask

import logging
log = logging.getLogger(__file__)

def batchify(data: Tensor, bsz: int, device) -> Tensor:
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

def get_batch(source: Tensor, i: int, bptt) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def train(train_data : Tensor, src_mask, model: nn.Module, args, epoch=0) -> None:
    criterion = nn.CrossEntropyLoss()
    my_list = [
        'transformer_encoder.layers.0.linear1.weight',
        'transformer_encoder.layers.0.linear1.bias',
        # 'transformer_encoder.layers.0.linear2.weight',
        # 'transformer_encoder.layers.0.linear2.bias'
    ]
    params = list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # For #layer=3, dropout = 0.2, this works and give sparse pattern of attentions. 
    # optimizer = torch.optim.SGD(model.parameters(), lr=5, momentum=0.0)

    # For #layer=3, dropout = 0.2, lr = 10 doesn't work at all. lr=3 works better than lr=5
    # but the attention pattern is actually not sparse at all, instead it concentrates at the diagonal, which is very interesting..
    # it seems that there is a phase transition. 
    # optimizer = torch.optim.SGD(model.parameters(), lr=3, momentum=0.0)

    # #layer=1 also lead to concentration with this setting (lr=2)
    # optimizer = torch.optim.SGD(model.parameters(), lr=2, momentum=0.0)

    optimizer = torch.optim.SGD([{'params': [temp[1] for temp in params], 'lr': args.lr_z * args.lr_y_multi_on_z}, {'params': [temp[1] for temp in base_params], 'lr': args.lr_z}])
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 1000
    start_time = time.time()

    num_batches = len(train_data) // args.bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        
        if (batch+(epoch-1)*num_batches) % log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            log.info(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()
            best_model_params_path = f'model_medium_iter_{batch+(epoch-1)*num_batches}.pt'
            torch.save(model.state_dict(), best_model_params_path)
        data, targets = get_batch(train_data, i, args.bptt)
        seq_len = data.size(0)
        if seq_len != args.bptt:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]
        # return output = [seqlen, bs, vocab_size]
        output, attention = model(data, src_mask)
        loss = criterion(output.view(-1, output.size(2)), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()

def evaluate(eval_data : Tensor, src_mask, model: nn.Module, args) -> float:
    criterion = nn.CrossEntropyLoss()

    model.eval()  # turn on evaluation mode
    total_loss = 0.
    return_attn = 0
    return_seq = []
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, args.bptt):
            data, targets = get_batch(eval_data, i, args.bptt)
            seq_len = data.size(0)
            if seq_len != args.bptt:
                src_mask = src_mask[:seq_len, :seq_len]
            output, attention = model(data, src_mask)
            output_flat = output.view(-1, output.size(2))
            total_loss += seq_len * criterion(output_flat, targets).item()
            if i==100*args.bptt:
                batch_idx = 0
                return_attn = [att[batch_idx,:,:] for att in attention]
                return_seq = data[:,batch_idx]            
    return total_loss / (len(eval_data) - 1), return_attn, return_seq


@hydra.main(config_path="config", config_name="decoder_wiki.yaml", version_base="1.1.1")
def main(args):
    log.info(common_utils.print_info(args))
    common_utils.set_all_seeds(args.seed)

    if args.dataset == "wikitext2":
        train_iter, val_iter, test_iter = WikiText2()
    elif args.dataset == "wikitext103":
        train_iter, val_iter, test_iter = WikiText103()
    else:
        raise RuntimeError(f"Unsupported dataset {args.dataset}")

    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
        """Converts raw text into a flat Tensor."""
        data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    # ``train_iter`` was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_data = data_process(train_iter)
    val_data = data_process(val_iter)
    test_data = data_process(test_iter)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = batchify(train_data, args.batch_size, device)  # shape ``[seq_len, batch_size]``
    val_data = batchify(val_data, args.eval_batch_size, device)
    test_data = batchify(test_data, args.eval_batch_size, device)

    ntokens = len(vocab)  # size of vocabulary
    config = TransConfig(ntokens, args.emsize, args.nhead, args.d_hid, args.nlayers, args.dropout)
    model = TransformerModel(config).to(device)

    src_mask = generate_square_subsequent_mask(args.bptt).to(device)

    for epoch in range(1, args.num_epoch + 1):
        train(train_data, src_mask, model, args, epoch=epoch)

    collections = {}
    
    # Get validation loss + example attention.

    for filename in glob.glob("*.pt"):
        log.info(filename)
        model.load_state_dict(torch.load(filename))
        val_loss, attn, seq = evaluate(val_data, src_mask, model, args)

        seq = vocab.lookup_tokens(seq.tolist())
        log.info(seq)

        val_ppl = math.exp(val_loss)

        # Iter
        no_ext, _ = os.path.splitext(filename)
        iter_num = int(no_ext[no_ext.rfind("_")+1:])
        collections[iter_num] = dict(filename=filename, val_loss = val_loss, val_ppl = val_ppl, attn=attn)

    pickle.dump(collections, open("collections.pkl", "wb"))
    log.info(os.getcwd())

if __name__ == '__main__':
    main()