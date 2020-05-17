from transformers import TFGPT2LMHeadModel
from transformers.modeling_tf_bert import BERT_START_DOCSTRING, TFBertPreTrainedModel, TFBertMainLayer, TFBertNSPHead, \
    TFBertMLMHead, TFBertEncoder, TFBertPooler
from transformers.modeling_tf_utils import get_initializer, shape_list


class TFBertForAffinityPrediction(TFBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = TFBertMainLayer(config, name="bert")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.flat = tf.keras.layers.Flatten()
        self.fully_connected = tf.keras.layers.Dense(1024, activation='relu')
        self.regressor = tf.keras.layers.Dense(1)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def call(self, inputs, **kwargs):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        logits (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, config.num_labels)`):
            Regression scores.

    Examples::

        import tensorflow as tf
        from transformers import BertTokenizer, TFBertForAffinityPrediction

        tokenizer = MolTokenizer.from_pretrained('model')
        model = TFBertForAffinityPrediction.from_pretrained('model')
        input_ids = tf.constant(tokenizer.encode("SMILES Mol", ""SMILES target (context),add_special_tokens=True)).ids # Batch size 1
        outputs = model(input_ids)
        logits = outputs[0]

        """
        outputs = self.bert(inputs, **kwargs)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output, training=kwargs.get("training", False))
        flattend_output = self.flat(pooled_output)
        flattend_output = self.fully_connected(flattend_output)
        logits = self.regressor(flattend_output)

        outputs = logits

        return outputs  # logits
