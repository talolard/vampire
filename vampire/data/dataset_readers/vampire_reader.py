import itertools
import json
import logging
from glob import glob
from io import TextIOWrapper
from typing import Dict

import numpy as np
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (ArrayField, Field, LabelField, ListField,
                                  MetadataField, TextField)
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
    """
    def __init__(self, lazy: bool = False, covariates: Dict[str, str] = None) -> None:
        super().__init__(lazy=lazy)
        self._covariates = covariates.as_dict()

    @overrides
    def _read(self, file_path):
        mat = load_sparse(file_path)        
        mat = mat.tolil()
        covariate_files = {}

        for key, val in self._covariates.items():
            covariate_files[key] = glob(val)
        labels = {}
        if covariate_files:
            for label, cov_files in covariate_files.items():
                if 'train' in file_path:
                    cov_to_use = [x for x in cov_files if "train" in x][0]
                elif 'dev' in file_path:
                    cov_to_use = [x for x in cov_files if "dev" in x][0]
                elif 'test' in file_path:
                    cov_to_use = [x for x in cov_files if "test" in x][0]                
                with open(cov_to_use, 'r') as file_:
                    labels[label] = [line.strip() for line in file_.readlines()]
        for ix in range(mat.shape[0]):
            if labels:
                label_subset = {key: val[ix] for key, val in labels.items()}
                instance = self.text_to_instance(mat[ix].toarray().squeeze(), **label_subset)
            else:
                instance = self.text_to_instance(mat[ix].toarray().squeeze())
            if instance is not None:
                yield instance

    @overrides
    def text_to_instance(self, vec: str, **labels) -> Instance:  # type: ignore
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
