# All initializations
import re

class Tokenizer:

    def __init__(self, input_path: str = None, vocab_size: int = 0, special_tokens: list[str] = []):
        # First - all initializations
        self.data_source = input_path
        self.vocab = set()
        self.n_vocab = 0
        self.expected_vocab_size = vocab_size
        self.pre_tokenized_counts = dict[tuple, int]
        self.token_merges = []
        self.special_tokens = special_tokens

        # Initial-stage additions into vocab
        for token in self.special_tokens:
            self.add_to_vocab(token)

        for i in range(256):
            self.add_to_vocab(chr(i).encode("utf-8"))
        
        print(f"New vocabulary size: {len(self.vocab)} against expected size of {self.expected_vocab_size}")

    def add_to_vocab(self, word: str) -> None:
        self.vocab.add(word)
        self.n_vocab += 1

    def pre_tokenize(self, corpus: str):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        matches = re.finditer(PAT, corpus)
        for match in matches:
            if match.group() not in self.pre_tokenized_counts:
                self.pre_tokenized_counts[match.group()] = 1
            else:
                self.pre_tokenized_counts[match.group()] += 1
        
    def singular_merge(self):
        max_merge_mapping = dict()
        
        for word_bytes in self.pre_tokenized_counts:
            for iter in range(len(word_bytes) - 1):
                if word_bytes[iter] + word_bytes[iter+1] not in max_merge_mapping:
                    max_merge_mapping[word_bytes[iter] + word_bytes[iter+1]] = self.pre_tokenized_counts[word_bytes]
                else:
                    max_merge_mapping[word_bytes[iter] + word_bytes[iter+1]] += self.pre_tokenized_counts[word_bytes]

        max_merge_mapping_sorted = dict(sorted(max_merge_mapping.items(), key=lambda pair: (pair[1], pair[0]), reverse=True))

        new_candidates_list = list(max_merge_mapping_sorted.keys())

        original_merge = ()
        current_replacement = new_candidates_list[0] if len(new_candidates_list) > 0 else None

        if current_replacement is not None:
            self.add_to_vocab(current_replacement.encode('utf-8'))

            new_old_word_byte_mapping = dict()

            for word_bytes in self.pre_tokenized_counts:
                for iter in range(len(word_bytes) - 1):
                    if word_bytes[iter] + word_bytes[iter+1] == current_replacement:
                        if (word_bytes[iter], word_bytes[iter+1]) not in self.token_merges:
                            original_merge = (word_bytes[iter], word_bytes[iter+1])
                            self.token_merges.append((word_bytes[iter], word_bytes[iter+1]))

                        updated_word_bytes = list(word_bytes)
                        updated_word_bytes[iter] = current_replacement
                        updated_word_bytes.pop(iter+1)
                        updated_word_bytes = tuple(updated_word_bytes)

                        new_old_word_byte_mapping[word_bytes] = updated_word_bytes


            for old_word_bytes in new_old_word_byte_mapping:
                self.pre_tokenized_counts[new_old_word_byte_mapping[old_word_bytes]] = self.pre_tokenized_counts.pop(old_word_bytes)
        # print(f"New vocabulary size: {len(self.vocab)} against expected size of {self.expected_vocab_size}")
        return original_merge, current_replacement