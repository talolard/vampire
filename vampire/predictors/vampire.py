from overrides import overrides
from typing import List
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
import numpy as np

@Predictor.register('vampire')
class VampirePredictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single class for it.  In particular, it can be used with
    the :class:`~allennlp.models.basic_classifier.BasicClassifier` model
    """
    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        arr = instance.fields['tokens'].array
        arr = list(np.nonzero(arr)[0])
        tokens = [self._model.vocab.get_index_to_token_vocabulary('vampire')[x] for x in arr]
        outputs['tokens'] = " ".join(tokens)
        output['norm_nll'] = output['nll'] / len(tokens)
        return sanitize(outputs)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        for instance, output in zip(instances, outputs):
            arr = instance.fields['tokens'].array
            arr = list(np.nonzero(arr)[0])
            tokens = [self._model.vocab.get_index_to_token_vocabulary('vampire')[x] for x in arr]
            output['tokens'] = " ".join(tokens)
            output['norm_nll'] = output['nll'] / len(tokens)
        return sanitize(outputs)


    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"label"`` to the output.
        """
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)
