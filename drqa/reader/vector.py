#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Functions for putting examples into torch format."""

from collections import Counter
import torch


def vectorize(ex, model, single_answer=False, bert_tokenizer=None):
    """Torchify a single example."""
    args = model.args
    word_dict = model.word_dict
    feature_dict = model.feature_dict

    # Index words
    if bert_tokenizer:
        token_ids = []
        # stores a '1' at locations that begin a token from the original document/question.
        # (if a word gets split into 3 sub-tokens, for example, [1,0,0] will get appended to the mask)
        begin_token_masks = []
        for data_type in ['document','question']:
            tokens = []
            begin_token_mask = []
            tokens.append('[CLS]')
            # [CLS] doesn't count as a real token, so mark it as 0 in begin_token_mask
            begin_token_mask.append(0)
            for token in ex[data_type]:
                subtokens = bert_tokenizer.tokenize(token)
                tokens.extend(subtokens)
                begin_token_mask.extend([1] + [0]*(len(subtokens)-1))
            tokens.append('[SEP]')
            # [SEP] doesn't count as a real token, so mark it as 0 in begin_token_mask
            begin_token_mask.append(0)
            ids = bert_tokenizer.convert_tokens_to_ids(tokens)
            token_ids.append(ids)
            begin_token_masks.append(begin_token_mask)
        document = (torch.tensor(token_ids[0], dtype=torch.long),
                    torch.tensor(begin_token_masks[0], dtype=torch.uint8))
        question = (torch.tensor(token_ids[1], dtype=torch.long),
                    torch.tensor(begin_token_masks[1], dtype=torch.uint8))
    else:
        document = torch.tensor([word_dict[w] for w in ex['document']], dtype=torch.long)
        question = torch.tensor([word_dict[w] for w in ex['question']], dtype=torch.long)

    # Create extra features vector
    if len(feature_dict) > 0:
        features = torch.zeros(len(ex['document']), len(feature_dict))
    else:
        features = None

    # f_{exact_match}
    if args.use_in_question:
        q_words_cased = {w for w in ex['question']}
        q_words_uncased = {w.lower() for w in ex['question']}
        q_lemma = {w for w in ex['qlemma']} if args.use_lemma else None
        for i in range(len(ex['document'])):
            if ex['document'][i] in q_words_cased:
                features[i][feature_dict['in_question']] = 1.0
            if ex['document'][i].lower() in q_words_uncased:
                features[i][feature_dict['in_question_uncased']] = 1.0
            if q_lemma and ex['lemma'][i] in q_lemma:
                features[i][feature_dict['in_question_lemma']] = 1.0

    # f_{token} (POS)
    if args.use_pos:
        for i, w in enumerate(ex['pos']):
            f = 'pos=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (NER)
    if args.use_ner:
        for i, w in enumerate(ex['ner']):
            f = 'ner=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (TF)
    if args.use_tf:
        counter = Counter([w.lower() for w in ex['document']])
        l = len(ex['document'])
        for i, w in enumerate(ex['document']):
            features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l

    # Maybe return without target
    if 'answers' not in ex:
        return document, features, question, ex['id']

    # ...or with target(s) (might still be empty if answers is empty)
    if single_answer:
        assert(len(ex['answers']) > 0)
        start = torch.LongTensor(1).fill_(ex['answers'][0][0])
        end = torch.LongTensor(1).fill_(ex['answers'][0][1])
    else:
        start = [a[0] for a in ex['answers']]
        end = [a[1] for a in ex['answers']]

    return document, features, question, start, end, ex['id']


def batchify(batch):
    """Gather a batch of individual examples into one batch."""
    NUM_INPUTS = 3
    NUM_TARGETS = 2
    NUM_EXTRA = 1

    ids = [ex[-1] for ex in batch]
    features = [ex[1] for ex in batch]

    use_bert = type(batch[0][0]) is tuple
    if use_bert:
        docs = [ex[0][0] for ex in batch]
        questions = [ex[2][0] for ex in batch]
        docs_begin_token_masks = [ex[0][1] for ex in batch]
        questions_begin_token_masks = [ex[2][1] for ex in batch]
    else:
        docs = [ex[0] for ex in batch]
        questions = [ex[2] for ex in batch]
        docs_begin_token_masks,questions_begin_token_masks = None,None

    # Batch documents and questions
    assert len(docs) == len(questions)
    batch_size = len(docs)
    # x = [{},{}]
    x = {'docs':{}, 'questions':{}}
    doc_and_question_batches = [docs, questions]
    begin_token_masks = [docs_begin_token_masks,questions_begin_token_masks]
    for j,key in enumerate(x.keys()):
        doc_or_question_batch = doc_and_question_batches[j]
        max_length = max([doc_or_question.size(0) for doc_or_question in doc_or_question_batch])
        x[key]['token_ids']        = torch.zeros(batch_size, max_length, dtype=torch.long)
        x[key]['mask']             = torch.ones( batch_size, max_length, dtype=torch.uint8)
        x[key]['features']         = torch.zeros(batch_size, max_length, features[0].size(1)) if (key=='docs') and features[0] is not None else None
        x[key]['begin_token_mask'] = torch.zeros(batch_size, max_length, dtype=torch.uint8) if use_bert else None
        for i,doc_or_question in enumerate(doc_or_question_batch):
            x[key]['token_ids'][i, :doc_or_question.size(0)].copy_(doc_or_question)
            x[key]['mask'][i, :doc_or_question.size(0)].fill_(0)
            if x[key]['features'] is not None:
                x[key]['features'][i, :doc_or_question.size(0)].copy_(features[i])
            if use_bert:
                x[key]['begin_token_mask'][i, :doc_or_question.size(0)].copy_(begin_token_masks[j][i])

    if use_bert:
        # HACK: To not change the number of things returned by this function,
        # we concatenate the begin_token_masks with the other masks.
        for key in x.keys():
            x[key]['mask'] = torch.cat((x[key]['mask'], x[key]['begin_token_mask']))
            # x[key]['mask'] = x[key]['begin_token_mask']

    # Maybe return without targets
    if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
        return (x['docs']['token_ids'], x['docs']['features'], x['docs']['mask'],
                x['questions']['token_ids'], x['questions']['mask'], ids)

    # Otherwise return with targets
    assert len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS, 'Incorrect number of inputs per example.'
    if torch.is_tensor(batch[0][3]):
        y_s = torch.cat([ex[3] for ex in batch])
        y_e = torch.cat([ex[4] for ex in batch])
    else:
        y_s = [ex[3] for ex in batch]
        y_e = [ex[4] for ex in batch]

    return (x['docs']['token_ids'], x['docs']['features'], x['docs']['mask'],
            x['questions']['token_ids'], x['questions']['mask'], y_s, y_e, ids)
