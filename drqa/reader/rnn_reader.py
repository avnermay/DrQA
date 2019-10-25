#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Implementation of the RNN based DrQA reader."""

import torch
import torch.nn as nn
from . import layers
from pytorch_pretrained_bert import BertModel

# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------


class RnnDocReader(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args, normalize=True):
        super(RnnDocReader, self).__init__()
        # Store config
        self.args = args

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)

        if args.use_bert_embeddings:
            assert args.embedding_dim == 768, 'Must set args.embedding_dim to 768 if using bert embeddings.'
            self.bert_model = BertModel.from_pretrained(args.bert_model_name)
            self.bert_model.eval()

        # Projection for attention weighted question
        if args.use_qemb:
            self.qemb_match = layers.SeqAttnMatch(args.embedding_dim)

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = args.embedding_dim + args.num_features
        if args.use_qemb:
            doc_input_size += args.embedding_dim

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.question_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            doc_hidden_size *= args.doc_layers
            question_hidden_size *= args.question_layers

        # Question merging
        if args.question_merge not in ['avg', 'self_attn']:
            raise NotImplementedError('merge_mode = %s' % args.merge_mode)
        if args.question_merge == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)

        # Bilinear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )
        self.end_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )

    def forward(self, x1, x1_f, x1_mask, x2, x2_mask):
        """Inputs:
        x1 = document word indices             [batch, len_d]
        x1_f = document word features indices  [batch, len_d, nfeat]
        x1_mask = document padding mask        [batch, len_d] or [2*batch, len_d] if use_bert
        x2 = question word indices             [batch, len_q]
        x2_mask = question padding mask        [batch, len_q] or [2*batch, len_q] if use_bert
        """
        device = x1.device
        batch_size = x1.size(0)
        assert ((self.args.use_bert_embeddings and (x1_mask.size(0) == 2 * batch_size) and (x2_mask.size(0) == 2 * batch_size)) or
                (not self.args.use_bert_embeddings and (x1_mask.size(0) == batch_size) and (x2_mask.size(0) == batch_size))), \
               'If using bert embeddings, the masks need to have double the number of rows (to store regular mask and begin_token_mask).'
        if self.args.use_bert_embeddings:
            # The goal in this code is to extract BERT embeddings for all the tokens (DrQA tokenization) in the
            # original DrQA document and question.  Because the BERT tokenization is a refinement of the DrQA
            # tokenization, we must first compute the embeddings for all the tokens in the BERT tokenization,
            # and then extract the corresponding embeddings for the tokens in the DrQA tokenization.
            # We accomplish this via the following steps:
            # STEP 1:
            #    **Split x*_mask into x*_bert_mask and x*_begin_token_mask.**
            # >> The bert_masks tell BERT where the document/question end, in terms of the BERT tokenization.
            #    A 1 specifies the BERT model should pay attention to the token, and a 0 specifies it should not.
            #    These masks were created for the RNN encoders originally (in vector.batchify), which use the
            #    opposite convention for 1's and 0's (1 = ignore, 0 = pay attention), so we must flip the 1's and 0's.
            # >> The begin_token_masks specify at which indices, in terms of the BERT tokenization, the original tokens
            #    from the DrQA tokenization begin.
            # STEP 2:
            #    **Use the x*_begin_token_mask to create the x*_rnn_mask.**
            # >> The x*_rnn_mask specify what indices, in terms of the original DrQA tokenization, the RNN should pay
            #    attention to (ie, what embeddings correspond to real tokens from the document/question, vs. the null embedding).
            # >> The number of 1's in each row of x*_begin_token_mask specify the number of tokens in the original
            #    document/question (in the DrQA tokenization), so we can use this information to construct the x*_rnn_mask.
            # STEP 3:
            #    **Pass the x1,x2 token_ids into the BERT model (with the x*_bert_mask) to get BERT embeddings for every BERT token
            #    in the original document and question.**
            # STEP 4:
            #    **Extract the BERT embeddings corresponding to the BERT tokens at all of the locations which correspond to the
            #    beginning of a DrQA token.**
            # >> To do this, we take the output from step 3, and extract the embeddings based on the x*_begin_token_masks.
            assert device == torch.device('cuda:0'), 'Must run BERT on GPU'
            with torch.no_grad:
                # STEP 1: x*_mask --> x*_bert_mask, x*_begin_token_mask
                # Recall from vector.batchify that x*_mask = [x*_bert_mask; x*_begin_token_mask] (vertically stacked),
                # so we extract these two sub-tensors from x*_mask below.
                # We use (1 - x) below to turn the 1's into 0's and vice versa (see Step 1 explanation above for reason).
                x1_bert_mask = 1 - x1_mask[:batch_size,:].clone().detach()
                x2_bert_mask = 1 - x2_mask[:batch_size,:].clone().detach()
                x1_begin_token_mask = x1_mask[batch_size:,:].clone().detach()
                x2_begin_token_mask = x2_mask[batch_size:,:].clone().detach()

                # STEP 2: Use x*_begin_token_mask to create x*_rnn_mask
                doc_lengths_orig = torch.sum(x1_begin_token_mask, dim=1)
                question_lengths_orig = torch.sum(x2_begin_token_mask, dim=1)
                max_doc_length_orig = torch.max(doc_lengths_orig).item()
                max_question_length_orig = torch.max(question_lengths_orig).item()
                x1_rnn_mask = torch.ones(batch_size, max_doc_length_orig, dtype=torch.uint8, device=device)
                x2_rnn_mask = torch.ones(batch_size, max_question_length_orig, dtype=torch.uint8, device=device)
                for i in range(batch_size):
                    doc_length = doc_lengths_orig[i].item()
                    question_length = question_lengths_orig[i].item()
                    x1_rnn_mask[i,:doc_length].fill_(0)
                    x2_rnn_mask[i,:question_length].fill_(0)

                # STEP 3: Pass x1,x2 through the BERT model, together with x*_bert_masks,
                # to get contextual embeddings for each of the BERT tokens (last hidden layer activations)
                doc_all_encoder_layers, _ = self.bert_model(x1, token_type_ids=None, attention_mask=x1_bert_mask)
                question_all_encoder_layers, _ = self.bert_model(x2, token_type_ids=None, attention_mask=x2_bert_mask)
                doc_embeddings = doc_all_encoder_layers[-1].detach()
                question_embeddings = question_all_encoder_layers[-1].detach()

                # STEP 4: Extract BERT embeddings corresponding to the BERT tokens which begin each DrQA token.
                # We do this using x*_begin_token_mask to know the locations of these tokens.
                x1_emb = torch.zeros(batch_size, max_doc_length_orig, self.args.embedding_dim, device=device)
                x2_emb = torch.zeros(batch_size, max_question_length_orig, self.args.embedding_dim, device=device)
                for i in range(batch_size):
                    # get length of i^th document/question.
                    doc_length = doc_lengths_orig[i].item()
                    question_length = question_lengths_orig[i].item()
                    # get indices of the BERT sub-tokens which correspond to the beginnings of the DrQA tokens.
                    doc_nz_ind = x1_begin_token_mask[i,:].nonzero()
                    question_nz_ind = x2_begin_token_mask[i,:].nonzero()
                    # extract the BERT embeddings corresponding to the BERT sub-tokens which begin the DrQA tokens.
                    x1_emb[i,:doc_length,:] = doc_embeddings[i,doc_nz_ind,:]
                    x2_emb[i,:question_length,:] = question_embeddings[i,question_nz_ind,:]
        else:
            # Embed both document and question
            x1_emb = self.embedding(x1)
            x2_emb = self.embedding(x2)
            # rename x1_mask and x2_mask for clarity and consistency with use_bert_embedding code.
            x1_rnn_mask, x2_rnn_mask = x1_mask, x2_mask

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.args.dropout_emb,
                                           training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb,
                                           training=self.training)

        # Form document encoding inputs
        drnn_input = [x1_emb]

        # Add attention-weighted question representation
        if self.args.use_qemb:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_rnn_mask)
            drnn_input.append(x2_weighted_emb)

        # Add manual features
        if self.args.num_features > 0:
            drnn_input.append(x1_f)

        # Encode document with RNN
        doc_hiddens = self.doc_rnn(torch.cat(drnn_input, 2), x1_rnn_mask)

        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(x2_emb, x2_rnn_mask)
        if self.args.question_merge == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_rnn_mask)
        elif self.args.question_merge == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens, x2_rnn_mask)
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)

        # Predict start and end positions
        start_scores = self.start_attn(doc_hiddens, question_hidden, x1_rnn_mask)
        end_scores = self.end_attn(doc_hiddens, question_hidden, x1_rnn_mask)
        return start_scores, end_scores
