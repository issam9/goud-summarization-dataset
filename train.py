from pathlib import Path

from transformers import (EncoderDecoderModel, BertTokenizer, 
                                    DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments)
from transformers.trainer_utils import set_seed

from datasets import load_dataset

from rouge import Rouge

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import logging

logger = logging.getLogger(__name__)


@hydra.main(config_path='configs', config_name='default')
def main(cfg: DictConfig):
    # seed for reproducibility, we can use 0 in case we don't want seed
    if cfg.seed!=0:
        set_seed(cfg.seed)

    # log paramters
    logger.info(OmegaConf.to_yaml(cfg))

    dataset = load_dataset('Goud/Goud-sum')

    tokenizer = BertTokenizer.from_pretrained(cfg.model.name)
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(cfg.model.name, cfg.model.name)

    # set special tokens and vocab size
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # text generation parameters
    model.config.max_length = cfg.generate.max_length
    model.config.min_length = cfg.generate.min_length
    model.config.early_stopping = cfg.generate.early_stopping
    model.config.num_beams = cfg.generate.num_beams
    model.config.no_repeat_ngram_size = cfg.generate.no_repeat_ngram_size


    # Preprocessing and building dataset
    def preprocess_function(batch):
        # tokenize the inputs and labels
        inputs = tokenizer(batch["article"], padding="max_length", 
                                            truncation=True, max_length=cfg.tokenizer.encoder_max_length)
        outputs = tokenizer(batch["headline"], padding="max_length", 
                                            truncation=True, max_length=cfg.tokenizer.decoder_max_length)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()

        # Replace pad token with -100 to ignore it in loss
        batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

        return batch

    # Create train dataset
    tokenized_train_dataset = dataset['train'].map(
        preprocess_function, batched=True, remove_columns=["article", "headline"]
    )
    # Create eval dataset
    tokenized_eval_dataset = dataset['validation'].map(
        preprocess_function, batched=True, remove_columns=["article", "headline"]
    )
    # Create test dataset
    tokenized_test_dataset = dataset['test'].map(
        preprocess_function, batched=True, remove_columns=["article", "headline"]
    )


    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Rouge score
    metric = Rouge()


    # Define compute metrics function
    def compute_metrics(pred):

        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # decoding predictions and labels
        candidates = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        references = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        scores = metric.get_scores(candidates, references, avg=True, ignore_empty=True)
        result = {key: round(value['f'] * 100, 2) for key, value in scores.items()}

        return result


    # Seq2Seq Trainer Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=f'{cfg.model.name.split("/")[-1]}',                               # output directory
        num_train_epochs=cfg.trainer.num_epochs,              # total number of training epochs
        per_device_train_batch_size=cfg.trainer.batch_size,   # batch size per device during training
        per_device_eval_batch_size=cfg.trainer.batch_size,
        warmup_steps=cfg.trainer.warmup_steps,                # number of warmup steps for learning rate scheduler
        logging_strategy='epoch',                             # log at the end of each epoch
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        evaluation_strategy='epoch',
        predict_with_generate=True,
        overwrite_output_dir=True,
        save_total_limit=3,
        fp16=cfg.trainer.fp16
        )


    # define trainer
    trainer = Seq2SeqTrainer(
    model=model,                            # the instantiated Transformers model 
    args=training_args,                     # training arguments
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
    )


    # train model
    trainer.train()


    # get test results
    if cfg.trainer.predict:
        metrics = trainer.predict(tokenized_test_dataset).metrics
        metrics['model_name'] = cfg.model.name
        logger.info(metrics)


    # save model 
    if cfg.trainer.save_model:
        model_path = Path(get_original_cwd()) / Path(f'models/{cfg.model.name}_{cfg.trainer.num_epochs}')
        model_path.mkdir(exist_ok=True, parents=True)
        trainer.save_model(model_path)


if __name__ == '__main__':
    main()