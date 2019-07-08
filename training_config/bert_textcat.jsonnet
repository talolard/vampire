{
    "dataset_reader": {
        "type": "semisupervised_text_classification_json",
        "sample": std.extVar("SAMPLE"),
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": "/home/suching/vampire/finetuned_bert/",
                "max_pieces": 128
            }
        }
    },
    "validation_dataset_reader": {
        "type": "text_classification_json",
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": "/home/suching/vampire/finetuned_bert/",
                "max_pieces": 128
            }
        }
    },
    "train_data_path": std.extVar("TRAIN_DATA_PATH"),
    "validation_data_path": std.extVar("VALIDATION_DATA_PATH"),
    "test_data_path": std.extVar("TEST_DATA_PATH"),
    "evaluate_on_test": true,
    "model": {
        "type": "bert_for_classification",
        "bert_model": "/home/suching/vampire/finetuned_bert/",
        "dropout": 0.1
    },
    "iterator": {
        "type": "basic",
        "batch_size": 32
    },
    "trainer": {
        "optimizer": {
            "type": "bert_adam",
            "lr": 0.00002,
            "warmup": 0.1
        },
        "validation_metric": "+accuracy",
        "num_epochs": 5,
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": 5,
            "num_steps_per_epoch": 100000
        },
        "cuda_device": 0 
    }
}