import json
import os
import glob
import requests
import os
from tqdm import tqdm  # for progress bar
import tarfile
import json

DATA_PATH = "data"
deleted_stories = 0

def CleanStory(summary, story):
    global deleted_stories
    # STARTSUMMARY " + story["summary"] + " STARTSTORY " + story["story"])
    """
    Some storys somehow contain broken symbols. This functions returns removes
    any story that has weird symbols and combines the summary with the story
    by introducing special tokens "STARTSUMMARY" and "STARTSTORY"
    """

    for c in summary + story:
        if not (ord(c) >= 32 and ord(c) <= 127) and ord(c) != 10: # 10: newline character
            deleted_stories += 1
            # print("offending char:", ord(c))
            return ""

    return " STARTSTORY " + story


def download_and_unpack(url):
    """
    Downloads the tinystories data from url and stores it inside data_raw/train_data.txt
    """
    os.makedirs(DATA_PATH + "_raw", exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 8 KB
    data_filename = os.path.join(DATA_PATH + "_raw", "TinyStories_all_data.tar.gz")

    if not os.path.exists(data_filename):
        with open(data_filename, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=block_size):
                f.write(chunk)
                bar.update(len(chunk))
        print("Download completed.")
    else:
        print("File aready downloaded!")

    if not os.path.exists("data_raw/train_data.txt"):
        # Extract shards:
        print("extracting tar file...")
        with tarfile.open(data_filename, 'r:gz') as tar:
            tar.extractall(DATA_PATH + "_raw")

        shard_filenames = sorted(glob.glob(os.path.join(DATA_PATH + "_raw", "*.json")))
        out_file_name = DATA_PATH + "_raw" + "/train_data.txt"
        total_files = len(shard_filenames)
        num_stories = 0
        with tqdm(total=total_files, desc="Converting to utf-8") as pbar:
            with open(out_file_name, "w", encoding="utf-8") as of:
                for shard_path in shard_filenames:
                    pbar.update(1)
                    with open(shard_path, "r", encoding="utf-8") as json_file:
                        json_data = json.load(json_file)
                        data_list = []
                        for story in json_data:
                            data_list.append(CleanStory(story["summary"], story["story"]))
                            num_stories += 1
                        res = "".join(data_list)
                        of.write(res)
                    os.unlink(shard_path)

        print(f"Ready! Thrown out {int(100*deleted_stories/num_stories)}% stories because they were bad!")

if __name__ == "__main__":
    download_and_unpack("https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz")