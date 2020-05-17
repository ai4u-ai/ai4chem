# ai4chem
Deep Learning for Chem 
###### Create and Train Tokenizer
```

    data_path='data/drug_token/'
    dest_path='data/models/'
    model_name='covid-tokenizer'
   
    paths = [str(x) for x in Path(data_path).glob("**/*.txt")]

    tokenizer = Tokenizer(BPE())
   
    tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(MoleculePretokenizer())
    tokenizer.decoder = decoders.Decoder.custom(MoleculePretokenizer())
    trainer = trainers.BpeTrainer(special_tokens=[ "<mask>",'<pad>'])
    tokenizer.train(trainer, paths)
    # And now it is ready, we can save the vocabulary with
    tokenizer.model.save(dest_path, model_name)
    
```
    
###### Create and Train Tokenizer
```
    data_path="data/drug_token/"
    dest_path='data/models/'
    model_name='covid-tokenizer'
   
    paths = [str(x) for x in Path(data_path).glob("**/*.txt")]
    tokenizer = ChemByteLevelBPETokenizer()
    tokenizer.train(trainer, paths)
    tokenizer.save(dest_path, model_name)
 ```   
###### Use Tokenizer

```        
        logger = logging.getLogger(__name__)    
        merges = "data/models/covid-tokenizer-merges.txt"
        vocab = "data/models/covid-tokenizer-vocab.json"
        tokenizer = ChemByteLevelBPETokenizer(vocab, merges)
        tokenizer.add_special_tokens(["<pad>", "<mask>"])
        tokenizer.enable_truncation(max_length=120)
        tokenizer.enable_padding(max_length=120, pad_token='<pad>')
        tokenizer.decode(tokenizer.encode('c1ccccc1OCCOC(=O)CC').ids)
        suppl = Chem.SDMolSupplier('../data/active.sdf')
       
        for mol in suppl:
            smi=Chem.MolToSmiles(mol)
            decoded_smi=tokenizer.decode(tokenizer.encode(smi).ids)
            if decoded_smi ==smi:
                     logger.info('correct')
            else:
                      logger.info('not correct')```