from collections import defaultdict
import numpy as np

class Tokenizer:

    def __init__(self, vocab_size=10000, start_token='ssss ', end_token=' eeee'):
        self.vocab_size = vocab_size
        self.start_token = start_token
        self.end_token = end_token

    def _add_start_end_tokens(self, seqs):
        add_start_end_token = lambda x: self.start_token + x + self.end_token

        extended_seqs = []

        for seq_list in seqs:
            extended_seqs.append([add_start_end_token(seq) for seq in seq_list])
        return extended_seqs

    def _generate_ids(self, seqs):
        word_to_count = defaultdict(lambda:0)
        seq_lens = []
        seq_list_lens = []

        for seq_list in seqs:
            seq_list_lens.append(len(seq_list))
            for seq in seq_list:
                seq_lens.append(len(seq.split()))
                for word in seq.split():
                    word_to_count[word.lower()] += 1

        words = np.array(list(word_to_count.keys()))
        counts = np.array(list(word_to_count.values()))
        print('* Total Words Found: {}'.format(len(words)))

        indices = np.argsort(counts)[-self.vocab_size:]
        top_k_words = words[indices]

        self.word_to_ids = dict(zip(top_k_words, np.arange(self.vocab_size)))
        self.ids_to_word = dict(zip(np.arange(self.vocab_size), top_k_words))
        self.max_seq_len = max(seq_lens)
        self.pad_to = max(seq_list_lens)

    def fit(self, seqs):
        seqs = self._add_start_end_tokens(seqs)
        self._generate_ids(seqs)

    def transform(self, seqs, pad=True):
        seqs = self._add_start_end_tokens(seqs)
        new_seqs = []
        for seq_list in seqs:
            new_seq_list = []
            for seq in seq_list:
                nseq = []
                for word in seq.split():
                    if word.lower() in self.word_to_ids:
                        nseq.append(self.word_to_ids[word.lower()])
                    else:
                        nseq.append(self.vocab_size)
                if len(nseq) < self.max_seq_len:
                    start = len(nseq)
                    for i in range(start, self.max_seq_len):
                        nseq.append(self.word_to_ids[self.end_token.strip()])
                if len(nseq) > self.max_seq_len:
                    nseq = nseq[:self.max_seq_len]
                new_seq_list.append(nseq)
            if len(new_seq_list) < self.pad_to and pad:
                for i in range(len(new_seq_list), self.pad_to):
                    new_seq_list.append(new_seq_list[i - len(new_seq_list)])
            if len(new_seq_list) > self.pad_to and pad:
                new_seq_list = new_seq_list[:self.pad_to]
            new_seqs.append(new_seq_list)
        return new_seqs

    def fit_transform(self, seqs):
        self.fit(seqs)
        return self.transform(seqs)

    def seqs_to_words(self, seqs):
        new_seqs = []
        for seq_list in seqs:
            nseq_list = []
            for seq in seq_list:
                nseq = []
                for id in seq:
                    if id in self.ids_to_word:
                        nseq.append(self.ids_to_word[id])
                    else:
                        nseq.append('<UNK>')
                nseq_list.append(' '.join(nseq))
            new_seqs.append(nseq_list)
        return new_seqs

    def __repr__(self):
        return '<Tokenizer Vocab size: {}, Start Token: "{}", End Token: "{}">'.format(
            self.vocab_size, self.start_token, self.end_token)