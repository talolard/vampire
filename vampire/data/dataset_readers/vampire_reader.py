import logging
from typing import Dict
from glob import glob
import numpy as np
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, Field, LabelField
from allennlp.data.instance import Instance
from overrides import overrides

from vampire.common.util import load_sparse

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("vampire_reader")
class VampireReader(DatasetReader):
    """
    Reads bag of word vectors from a sparse matrices representing training and validation data.

    Expects a sparse matrix of size N documents x vocab size, which can be created via
    the scripts/preprocess_data.py file.

    The output of ``read`` is a list of ``Instances`` with the field:
        vec: ``ArrayField``

    Parameters
    ----------
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be read lazily.
    sample : ``int``, optional, (default = ``None``)
        If specified, we will randomly sample the provided
        number of lines from the dataset. Useful for debugging.
    min_sequence_length : ``int`` (default = ``3``)
        Only consider examples from data that are greater than
        the supplied minimum sequence length.
    """
    def __init__(self,
                 lazy: bool = False,
                 sample: int = None,
                 min_sequence_length: int = 0,
                 covariates: Dict[str, str] = None) -> None:
        super().__init__(lazy=lazy)
        self._sample = sample
        self._min_sequence_length = min_sequence_length
        if covariates:
            self._covariates = covariates.as_dict()
        else:
            self._covariates = None

    @overrides
    def _read(self, file_path):
        # load sparse matrix
        mat = load_sparse(file_path)
        # convert to lil format for row-wise iteration
        mat = mat.tolil()

        # optionally sample the matrix
        if self._sample:
            indices = np.random.choice(range(mat.shape[0]), self._sample)
        else:
            indices = range(mat.shape[0])
        labels = {}
        covariate_files = {}
        if self._covariates:
            for key, val in self._covariates.items():
                covariate_files[key] = glob(val)
            for label, cov_files in covariate_files.items():
                if 'train' in file_path:
                    cov_to_use = [x for x in cov_files if "train" in x][0]
                elif 'dev' in file_path:
                    cov_to_use = [x for x in cov_files if "dev" in x][0]
                elif 'test' in file_path:
                    cov_to_use = [x for x in cov_files if "test" in x][0]                
                with open(cov_to_use, 'r') as file_:
                    labels[label] = [line.strip() for line in file_.readlines()]

        for index in indices:
            if labels:
                label_subset = {key: val[index] for key, val in labels.items()}
                instance = self.text_to_instance(mat[index].toarray().squeeze(), **label_subset)
            else:  
                instance = self.text_to_instance(vec=mat[index].toarray().squeeze())
            if instance is not None and mat[index].toarray().sum() > self._min_sequence_length:
                yield instance

    @overrides
    def text_to_instance(self, vec: str = None, **labels) -> Instance:  # type: ignore
        """
        Parameters
        ----------
        text : ``str``, required.
            The text to classify
        label ``str``, optional, (default = None).
            The label for this text.

        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence or phrase.
            label : ``LabelField``
                The label label of the sentence or phrase.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        fields['tokens'] = ArrayField(vec)
        for label, val in labels.items():
            fields[label + "_labels"] = LabelField(val, label_namespace=label + "_labels", skip_indexing=False)
        return Instance(fields)
