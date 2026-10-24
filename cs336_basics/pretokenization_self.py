# All initializations
class Tokenizer:
    def __init__(self):
        # First - all initializations
        self.n_vocab = 0
        self.vocab = set()
        self.pre_tokenized_counts = dict[tuple, int]

        # Initial-stage additions into vocab
        self.vocab.add("<|endoftext|>")

        for i in range(256):
            # self.vocab.add(chr(i).encode("utf-8"))
            self.add_to_vocab(chr(i).encode("utf-8"))

    def add_to_vocab(self, word: str) -> None:
        self.vocab.add(word)
        self.n_vocab += 1

    def pre_tokenize(self, corpus: str):
        words = corpus.split(" ")
        counts = {tuple(word): words.count(word) for word in words}
        
        self.pre_tokenized_counts = counts

    def singular_merge(self):
        max_merge_mapping = dict()

        for word_bytes in self.pre_tokenized_counts:
            for iter in range(len(word_bytes) - 1):
                if word_bytes[iter] + word_bytes[iter+1] not in max_merge_mapping:
                    max_merge_mapping[word_bytes[iter] + word_bytes[iter+1]] = self.pre_tokenized_counts[word_bytes]
                else:
                    max_merge_mapping[word_bytes[iter] + word_bytes[iter+1]] += self.pre_tokenized_counts[word_bytes]

        max_merge_mapping_sorted = dict(sorted(max_merge_mapping.items(), key=lambda pair: (pair[1], pair[0]), reverse=True))

        current_replacement = list(max_merge_mapping_sorted.keys())[0]

        self.add_to_vocab(current_replacement.encode('utf-8'))

        new_old_word_byte_mapping = dict()

        for word_bytes in self.pre_tokenized_counts:
            for iter in range(len(word_bytes) - 1):
                if word_bytes[iter] + word_bytes[iter+1] == current_replacement:
                    updated_word_bytes = list(word_bytes)
                    updated_word_bytes[iter] = current_replacement
                    updated_word_bytes.pop(iter+1)
                    updated_word_bytes = tuple(updated_word_bytes)

                    new_old_word_byte_mapping[word_bytes] = updated_word_bytes


        for old_word_bytes in new_old_word_byte_mapping:
            self.pre_tokenized_counts[new_old_word_byte_mapping[old_word_bytes]] = self.pre_tokenized_counts.pop(old_word_bytes)