import random


id2word_agnews = {
    '0': 'World',
    '1': 'Sports',
    '2': 'Business',
    '3': 'Science'
}
word2id_agnews = {
    'World'     : '0',
    'Sports'    : '1',
    'Business'  : '2',
    'Science'   : '3',
    'Technology': '3'
}

# shared by Ethos and Liar
id2word_ethos_liar = {
    '0': 'No',
    '1': 'Yes'
}
word2id_ethos_liar = {
    'No' : '0',
    'Yes': '1'
}


# shared by snli and mnli
id2word_snli_mnli = {
    '0': 'Contradiction',
    '1': 'Neutral',
    '2': 'Entailment'
}
word2id_snli_mnli = {
    'Contradiction': '0',
    'Neutral': '1',
    'Entailment': '2'
}

# shared by  qnli and rte
word2id_qnli_rte = {
    'Yes': '0',
    'No': '1'
}
id2word_qnli_rte = {
    '0': 'Yes',
    '1': 'No'
}


def print_args(args):
    print('-' * 30)
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('-' * 30)

# two-tuple
def shuffle_dataset_binary(dataset_tuple):
    '''shuffle the original dataset.'''
    x_list, y_list = dataset_tuple
    zipped = list(zip(x_list, y_list))
    random.shuffle(zipped)
    x_list_shuffled, y_list_shuffled = zip(*zipped)
    return (x_list_shuffled, y_list_shuffled)

# triple-tuple
def shuffle_dataset_triple(dataset_tuple):
    '''shuffle the original dataset.'''
    x1_list, x2_list, y_list = dataset_tuple
    zipped = list(zip(x1_list, x2_list, y_list))
    random.shuffle(zipped)
    x1_list_shuffled, x2_list_shuffled, y_list_shuffled = zip(*zipped)
    return (x1_list_shuffled, x2_list_shuffled, y_list_shuffled)


def get_num_classes(dataset_name):
    if dataset_name == 'agnews':
        return 4
    elif dataset_name == 'trec':
        return 6
    elif dataset_name in ('sst', 'mr', 'ethos', 'liar', 'rte', 'qnli'):
        return 2
    elif dataset_name in ('snli', 'mnli'):
        return 3
    else:
        print(f'Unknown dataset: {dataset_name}')
        raise NotImplementedError