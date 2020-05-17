import numpy as np
import json
import os
from pathlib import Path





from tokenizers import ByteLevelBPETokenizer, ByteLevelBPETokenizer, CharBPETokenizer, SentencePieceBPETokenizer, \
    BertWordPieceTokenizer, trainers, pre_tokenizers, decoders,models
import codecs

from tokenizers.implementations import BaseTokenizer

from typing import Sequence, Dict, Tuple, Optional, Union, List

from data_utils import iterate_minbatches
from modeling_tf_transf_cov import TFBertGenerator

TOKENIZERS = {'ByteLevelBPETokenizer': ByteLevelBPETokenizer(add_prefix_space=True),
              'CharBPETokenizer': CharBPETokenizer(),
              'SentencePieceBPETokenizer': SentencePieceBPETokenizer(),
              'BertWordPieceTokenizer': BertWordPieceTokenizer()}


# def train_pretrained_tokenizer(tokenizer: PreTrainedTokenizer, model: PreTrainedModel, tokens: Sequence[str],
#                                model_dir: str = "../data/tokenizer/covidbertmodel"):
#     """
#
#     :param tokenizer: PreTrainedTokenizer
#     :param model: PreTrainedModel
#     :param tokens:Sequence[str]
#     :param model_dir:str
#     :return:
#     example:
#         tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
#         model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
#         tokens=extract_tokens_for_training()
#         train_pretrained_tokenizer(tokenizer,model,tokens)
#     """
#
#     if not os.path.isdir(model_dir):
#         os.mkdir(model_dir)
#     tokenizer.add_tokens(tokens)
#     tokenizer.save_pretrained(model_dir)
#     model.resize_token_embeddings(len(tokenizer))
#     model.save_pretrained(model_dir)
#

def extract_tokens_for_training(data_path="../data/tokenizer/"):
    paths = [str(x) for x in Path(data_path).glob("**/*.txt")]
    tokens = []
    for p in paths:
        with open(p, "r", encoding='utf-8') as f:
            text = f.read()
            tokens.extend(text.split())
    return list(set(tokens))


def train_tokenizer_from_scratch(tokenizer_type, data_path="../data/drug_token/",
                                 special_tokens=[],
                                 dest_path='../data/models/',
                                 model_name='covidtokenizer'):
    tokenizer = TOKENIZERS[tokenizer_type]
    paths = [str(x) for x in Path(data_path).glob("**/*.txt")]
    tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=special_tokens)
    tokenizer.save(dest_path, model_name)


def extraxt_texts_for_tokenizer(files_path='../data/noncomm_use_subset', dest_path='../data/tokenizer/'):
    for n in os.listdir(files_path):
        text = ''
        try:
            with open(os.path.join(files_path, n)) as f:
                data = json.load(f)
                filename, file_extension = os.path.splitext(os.path.join(files_path, n))
                for key in ['abstract', 'body_text']:
                    for i, s in enumerate(data[key]):
                        text = ' '.join([text, s['text']]).strip()

            f = codecs.open(os.path.join(dest_path, n.replace(file_extension, '.txt')), 'w', 'utf-8')
            f.write(text)
            print('added')
        except Exception as exc:
            print(exc)



# Split SMILES into words
def split_smiles(sm):
    '''
    function: Split SMILES into words. Care for Cl, Br, Si, Se, Na etc.
    input: A SMILES
    output: A string with space between words
    '''
    arr = []
    i = 0
    while i < len(sm)-1:
        if not sm[i] in ['%', 'C', 'B', 'S', 'N', 'R', 'X', 'L', 'A', 'M', \
                        'T', 'Z', 's', 't', 'H', '+', '-', 'K', 'F']:
            arr.append(sm[i])
            i += 1
        elif sm[i]=='%':
            arr.append(sm[i:i+3])
            i += 3
        elif sm[i]=='C' and sm[i+1]=='l':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='C' and sm[i+1]=='a':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='C' and sm[i+1]=='u':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='B' and sm[i+1]=='r':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='B' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='B' and sm[i+1]=='a':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='B' and sm[i+1]=='i':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='S' and sm[i+1]=='i':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='S' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='S' and sm[i+1]=='r':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='N' and sm[i+1]=='a':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='N' and sm[i+1]=='i':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='R' and sm[i+1]=='b':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='R' and sm[i+1]=='a':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='X' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='L' and sm[i+1]=='i':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='A' and sm[i+1]=='l':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='A' and sm[i+1]=='s':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='A' and sm[i+1]=='g':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='A' and sm[i+1]=='u':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='M' and sm[i+1]=='g':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='M' and sm[i+1]=='n':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='T' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='Z' and sm[i+1]=='n':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='s' and sm[i+1]=='i':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='s' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='t' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='H' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='+' and sm[i+1]=='2':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='+' and sm[i+1]=='3':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='+' and sm[i+1]=='4':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='-' and sm[i+1]=='2':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='-' and sm[i+1]=='3':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='-' and sm[i+1]=='4':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='K' and sm[i+1]=='r':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='F' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        else:
            arr.append(sm[i])
            i += 1
    if i == len(sm)-1:
        arr.append(sm[i])
    return arr


from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, unicode_normalizer_from_str, Strip,Sequence,NFKC
from tokenizers.pre_tokenizers import CharDelimiterSplit, PreTokenizer
from utils import generateBricksLibrary
Offsets = Tuple[int, int]

class MurckoMixin(object):
    def pre_tokenize(self, sequence: str) -> List[Tuple[str, Offsets]]:
        mol=  Chem.MolFromSmiles(sequence)
        bms = MurckoScaffold.GetScaffoldForMol(mol)
        rgroups = Chem.ReplaceCore(mol, bms)
        pieces = Chem.GetMolFrags(rgroups)

        core = Chem.ReplaceSidechains(mol, bms)
        []

        return  [(l, (0, 1)) for i, l in enumerate(split_smiles(sequence.strip()))]

class MoleculePretokenizer(object):
    def __init__(self) -> None:
        """ Instantiate a new Whitespace PreTokenizer """
        pass
    def pre_tokenize(self, sequence: str) -> List[Tuple[str, Offsets]]:


        return  [(l, (0, 1)) for i, l in enumerate(split_smiles(sequence.strip()))]

    def decode(self, tokens):
        # print('decoding',tokens)
        return "".join(tokens)


class ChemByteLevelBPETokenizer(BaseTokenizer):
    """ ChemByteLevelBPETokenizer

    Represents a Byte-level BPE as introduced by OpenAI with their GPT-2 model
    """

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        add_prefix_space: bool = False,
        lowercase: bool = False,
        dropout: Optional[float] = None,
        unicode_normalizer: Optional[str] = None,
        continuing_subword_prefix: Optional[str] = None,
        end_of_word_suffix: Optional[str] = None,
    ):
        if vocab_file is not None and merges_file is not None:
            tokenizer = Tokenizer(
                BPE(
                    vocab_file,
                    merges_file,
                    dropout=dropout,
                    continuing_subword_prefix=continuing_subword_prefix or "",
                    end_of_word_suffix=end_of_word_suffix or "",
                )
            )
        else:
            tokenizer = Tokenizer(BPE())

        # Check for Unicode normalization first (before everything else)
        normalizers = []

        if unicode_normalizer:
            normalizers += [unicode_normalizer_from_str(unicode_normalizer)]

        if lowercase:
            normalizers += [Lowercase()]

        # Create the normalizer structure
        if len(normalizers) > 0:
            if len(normalizers) > 1:
                tokenizer.normalizer = Sequence(normalizers)
            else:
                tokenizer.normalizer = normalizers[0]

        tokenizer.pre_tokenizer =pre_tokenizers.PreTokenizer.custom(MoleculePretokenizer())
        tokenizer.decoder = decoders.Decoder.custom(MoleculePretokenizer())

        parameters = {
            "model": "ByteLevelBPE",
            "add_prefix_space": add_prefix_space,
            "lowercase": lowercase,
            "dropout": dropout,
            "unicode_normalizer": unicode_normalizer,
            "continuing_subword_prefix": continuing_subword_prefix,
            "end_of_word_suffix": end_of_word_suffix,
        }

        super().__init__(tokenizer, parameters)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        show_progress: bool = True,
        special_tokens: List[str] = [],
    ):
        """ Train the model using the given files """

        trainer = trainers.BpeTrainer(
            special_tokens=special_tokens,
            min_frequency=min_frequency,
            show_progress=show_progress
        )
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(trainer, files)


def train_custom_tokenizer():
    data_path="data/drug_token/"
    dest_path='data/models/'
    model_name='covid-tokenizer'
    paths = [str(x) for x in Path(data_path).glob("**/*.txt")]

    # We build our custom tokenizer:
    tokenizer = Tokenizer(BPE())
    # tokenizer.normalizer = Strip()]
    # tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(MoleculePretokenizer())
    # tokenizer.decoder=decoders.Decoder.custom(MoleculePretokenizer())
    tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(MoleculePretokenizer())
    tokenizer.decoder = decoders.Decoder.custom(MoleculePretokenizer())

    # print(tokenizer.decode(tokenizer.encode("C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1").ids))

    # We can train this tokenizer by giving it a list of path to text files:
    trainer = trainers.BpeTrainer(special_tokens=[ "<mask>",'<pad>'])
    tokenizer.train(trainer, paths)

    # print(tokenizer.decode(tokenizer.encode("C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1").ids))
    # We can train this tokenizer by giving it a list of path to text files:
    # trainer = trainers.BpeTrainer(special_tokens=[ "<mask>",'<pad>'])

    # print("saving")
    # And now it is ready, we can save the vocabulary with
    tokenizer.model.save(dest_path, model_name)

