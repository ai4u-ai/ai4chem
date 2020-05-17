**_**********Still under development**_********** 
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
                      logger.info('not correct')
```

#### Affinity Prediciton

    ```
    max_seq_length = 512

    merges = "data/models/covid-tokenizer-merges.txt"
    vocab = "data/models/covid-tokenizer-vocab.json"
    tokenizer = ChemByteLevelBPETokenizer(vocab, merges)
    tokenizer.add_special_tokens(["<pad>", "<mask>"])
    tokenizer.enable_truncation(max_length=120)
    config = BertConfig.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    affinityPrecictor = TFBertForAffinityPrediction(config)

    dataset_file = "desc_canvas_aug30.csv"
    dataset = pd.read_csv(os.path.join('../data', dataset_file))
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    num_test_batch = 12
    molecules = []
    train = []
    labels = []
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    affinityPrecictor.compile(loss='mse', optimizer=optimizer,
                              metrics=['mae', 'mse'])
    train = [[i['mol'][:511], i['mol'][:511]] for _, i in islice(train_dataset.iterrows(), num_test_batch)]
    labels = [i['pIC50'] for _, i in islice(train_dataset.iterrows(), num_test_batch)]
   
    train = tokenizer.batch_encode_plus(train, return_tensors="pt", add_special_tokens=True, pad_to_max_length=True)[
        "input_ids"]
    history = affinityPrecictor.fit(tf.convert_to_tensor(train), tf.convert_to_tensor(labels),
                                    epochs=200, verbose=0, validation_split=0.2,
                                    callbacks=[tfdocs.modeling.EpochDots(report_every=2)])
    print(affinityPrecictor.predict(tf.convert_to_tensor(train)))
    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plotter.plot({'Basic': history}, metric="mae")

    plt.ylabel('MAE [MPG]')
    plotter.plot({'Basic': history}, metric="mse")

    plt.ylabel('MSE [MPG^2]')

```