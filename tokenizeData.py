import json
import os
import numpy as np
import os
from tqdm import tqdm  # for progress bar
import argparse

DATA_PATH = "data_raw/train_data.txt" # path where the train data is
TOKENIZER_PATH = "models" # path where the tokenizer is saved

NUM_TOKENS = 5000 # Number of tokens in the tokenizer. The tokenizer will choose the most common ones

def ReadShards(chunks):
    """
    Read the data from data_raw/train_data.txt
    chunk_size defines the size of the datashards
    """
    # Check if the file exists
    
    data_set_size = os.path.getsize(DATA_PATH)
    chunk_size = data_set_size//chunks + 1
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            yield data

class Tokenizer():
    def __init__(self, train_data=None, path=None):
        if train_data is None:
            self.load_tokenizer(path)
        else:
            print("Training tokenizer on the first shard...")
            letters_only = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' "
            as_words = "".join([i if i in letters_only else " " for i in train_data])
            word_dict_ = {}
            for el in as_words.split(" "):
                if el not in word_dict_:
                    word_dict_[el] = 0
                word_dict_[el] += 1
            if '' in word_dict_:
                del word_dict_['']
            tokens = [(word_dict_[i], i) for i in word_dict_]
            tokens.sort(reverse=True)
            tokens = [tokens[i][1] for i in range(NUM_TOKENS)]
            
            self.replace_index = -1
            self._detokenize = list(tokens)
            self._tokenize = {self._detokenize[i]: i for i in range(len(self._detokenize))}

    def addTokens(self, shard):
        as_set = set(self._detokenize)
        shard_list_sorted = list(set(list(shard)))
        shard_list_sorted.sort()
        for el in shard_list_sorted:
            if el not in as_set:
                assert abs(self.replace_index) <= NUM_TOKENS # something went wrong. Maybe increase num_tokens
                self._detokenize[self.replace_index] = el
                self.replace_index -= 1
        self._tokenize = {self._detokenize[i]: i for i in range(len(self._detokenize))}

    def save_tokenizer(self, file_path):
        with open(file_path, 'w') as file:
            json.dump(self._detokenize, file)

    def load_tokenizer(self, file_path):
        with open(file_path, 'r') as file:
            self._detokenize = json.load(file)
        self._tokenize = {self._detokenize[i]: i for i in range(len(self._detokenize))}

    def tokenize(self, inpt, shard_name=None, use_tqdm=True):
        max_token_len = max([len(i) for i in self._tokenize])
        res = []
        i = 0
        with tqdm(total=len(inpt), desc=f"{shard_name}:", disable=not use_tqdm) as pbar:
            while i < len(inpt):
                for t_len in range(min(max_token_len, len(inpt) - i), 0, -1):
                    this_t = inpt[i:i+t_len]
                    if this_t in self._tokenize:
                        res.append(self._tokenize[this_t])
                        i += t_len
                        pbar.update(t_len)
                        break
                    if t_len == 1:
                        raise RuntimeError("Token", inpt[i], "does not exist.")
        return res

    def detokenize(self, inpt):
        res = "".join([self._detokenize[i] for i in inpt])
        return res
    
    def getTokenIndex(self, token):
        if token in self._tokenize:
            return self._tokenize[token]
        return 0

    def __len__(self):
        return len(self._detokenize)

def Pretokenizeshard(tokenizer, shard, shard_id):
    shard_name = f"shard_{shard_id}"
    tokenized = tokenizer.tokenize(shard, shard_name)
    # assert shard == tokenizer.Detokenize(tokenized)
    print(f"compressed to: {int(100*len(tokenized)/len(shard))}%")
    # store as file:
    np_data = np.array(tokenized, dtype=np.uint16)
    np.save(os.path.join("data", shard_name), np_data)

def Pretokenize():
    if not os.path.exists(DATA_PATH):
        raise RuntimeError("there is no data yet:( Please make sure that data_raw has a file called 'train_data.txt'!")
    tokenizer_save_path = os.path.join(TOKENIZER_PATH, "tokenizer.json")
    data_set_size = os.path.getsize(DATA_PATH)
    NUM_CHUNKS = data_set_size//92955624 # good chunk size
    print("Number of Chunks: ", NUM_CHUNKS)
    # Initialize tokenizer
    if not os.path.exists(tokenizer_save_path):
        shards_iterator = ReadShards(NUM_CHUNKS)
        first_shard_content = next(shards_iterator)
        tokenizer = Tokenizer(first_shard_content, tokenizer_save_path)
        # add letters that are not found yet:
        with tqdm(total=NUM_CHUNKS, desc=f"adding missing letters:") as pbar:
            for this_shard in ReadShards(NUM_CHUNKS):
                pbar.update(1)
                tokenizer.addTokens(this_shard)
        print("Finished with", len(tokenizer), "tokens")
        tokenizer.save_tokenizer(tokenizer_save_path)
    else:
        print("Loading already trained tokenizer! To retrain delete the tokenizer file in models/tokenizer.json")
        tokenizer = Tokenizer(None, tokenizer_save_path)

    # Process each shard
    shard_id = 0
    for this_shard in ReadShards(NUM_CHUNKS):
        shard_id += 1
        Pretokenizeshard(tokenizer, this_shard, shard_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pretokenize data.")
    parser.add_argument("--data_path", type=str, default="data_raw/train_data.txt", help="Path to the data file")
    args = parser.parse_args()

    DATA_PATH = args.data_path
    res1 = Pretokenize()