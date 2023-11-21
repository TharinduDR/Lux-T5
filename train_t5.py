from datasets import load_dataset
from transformers import T5Config


from t5_tokenizer_model import SentencePieceUnigramTokenizer

vocab_size = 32_000
input_sentence_size = None

dataset = load_dataset("oscar-corpus/OSCAR-2201",
                       use_auth_token=True,  # required
                       name="lb",
                       split="train")  # optional, but the dataset only has a train split

tokenizer = SentencePieceUnigramTokenizer(unk_token="<unk>", eos_token="</s>", pad_token="<pad>")


# Build an iterator over this dataset
def batch_iterator(sentence_size=None):
    if sentence_size is None:
        sentence_size = len(dataset)
    batch_length = 100
    for i in range(0, sentence_size, batch_length):
        yield dataset[i: i + batch_length]["text"]


# Train tokenizer
tokenizer.train_from_iterator(
    iterator=batch_iterator(sentence_size=input_sentence_size),
    vocab_size=vocab_size,
    show_progress=True,
)

# Save files to disk
tokenizer.save("sinhala-t5-base/tokenizer.json")
config = T5Config.from_pretrained("google/t5-v1_1-base", vocab_size=tokenizer.get_vocab_size())
config.save_pretrained("sinhala-t5-base")



