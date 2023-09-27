from dataclasses import dataclass

html_vocab_path = "/kaggle/input/pubtabnet-anot/train/structure_alphabet.txt"
c_vocab_path = "/kaggle/input/pubtabnet-anot/train/textline_recognition_alphabet.txt"
val_anot = "/kaggle/input/pubtabnet-anot/val/StructureLabelAddEmptyBbox_val/"
train_anot = "/kaggle/input/pubtabnet-anot/train/StructureLabelAddEmptyBbox_train/"

@dataclass
class Config:
    n_decoder_blocks: int = 1
    img_size: int = 480
    n_embd: int = 256
    feat_depth: int = 256
    n_heads: int = 8
    dropout: float = 0.0
    bias: bool = False
    pe_dropout: float = 0.1
    pe_maxlen: int = 5000
    html_vocab_path = html_vocab_path
    tags_vocab_size: int = len(open(html_vocab_path, "r").read().split("\n")) + 3
    tags_maxlen: int = 150
    c_vocab_path = c_vocab_path
    content_vocab_size: int = len(open(c_vocab_path, "r").read().split("\n")) + 3
    content_maxlen: int = 50
        
config = Config()