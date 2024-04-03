import os
import numpy as np
from tqdm import tqdm  # for progress bar
import argparse
from tokenizer import RegexTokenizer
from concurrent.futures import ProcessPoolExecutor

DATA_PATH = "data/data_raw/train_data.txt" # path where the train data is
TOKENIZER_PATH = "data" # path where the tokenizer is saved

NUM_TOKENS = 4000 # Number of tokens in the tokenizer. The tokenizer will choose the most common ones

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

def Pretokenizeshard(tokenizer, shard, shard_id):
    shard_name = f"shard_{shard_id}"
    with tqdm(total=4, desc=f"{shard_name}:", disable=False) as pbar:
        tokenized = tokenizer.encode_ordinary(shard, tqdm_bar=pbar)
    assert shard == tokenizer.decode(tokenized)
    print(f"compressed to: {int(100*len(tokenized)/len(shard))}%")
    # store as file:
    np_data = np.array(tokenized, dtype=np.uint16)
    np.save(os.path.join("data/data_shards", shard_name), np_data)

def Pretokenize():
    print(DATA_PATH)
    if not os.path.exists(DATA_PATH):
        raise RuntimeError("there is no data yet:( Please make sure that data_raw has a file called 'train_data.txt'!")
    tokenizer_save_path = os.path.join(TOKENIZER_PATH, "tokenizer")
    data_set_size = os.path.getsize(DATA_PATH)
    NUM_CHUNKS = data_set_size//92955624 + 1 # good chunk size
    print("Number of Chunks: ", NUM_CHUNKS)
    # Initialize tokenizer
    tokenizer = RegexTokenizer()
    if not os.path.exists(tokenizer_save_path + ".model"):
        shards_iterator = ReadShards(NUM_CHUNKS)
        first_shard_content = next(shards_iterator)
        with tqdm(total=4, desc=f"training tokenizer", disable=False) as pbar:
            tokenizer.train(first_shard_content, NUM_TOKENS, tqdm_bar=pbar)
        print("Finished with", len(tokenizer), "tokens")
        tokenizer.save(tokenizer_save_path, True)
    else:
        print("Loading already trained tokenizer! To retrain delete the tokenizer file in models/tokenizer.model")
        tokenizer.load(tokenizer_save_path + ".model")
    shard_id = 0
    with ProcessPoolExecutor(max_workers=8) as executor:
        for shard_id, this_shard in enumerate(ReadShards(NUM_CHUNKS), 1):
            executor.submit(Pretokenizeshard, tokenizer, this_shard, shard_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pretokenize data.")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to the data file")
    args = parser.parse_args()
    DATA_PATH = args.data_path
    res1 = Pretokenize()
