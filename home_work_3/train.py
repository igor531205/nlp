import argparse
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import DefaultDataCollator, TrainingArguments, Trainer, \
                         AutoTokenizer, AutoModelForQuestionAnswering


def main():

    parser = argparse.ArgumentParser(
        description='Train model for question answering')

    parser.add_argument('dir_name', type=str, default='model_qa',\
                        help='Directory for save the model and tokenizer')

    args = parser.parse_args()
    dir_name = args.dir_name

    # Load the dataset
    sberquad = load_dataset('sberquad')

    # Load the tokenizer and model
    model_name = 'ai-forever/ruBert-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    def prepare_train_features(examples):
        # Tokenize our examples with truncation and padding, but keep the overflows using a stride.
        # This results in one example possible giving several features when a context is long,
        # each of those features having a context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples['question'],
            examples['context'],
            truncation='only_second',  # truncate context, not the question
            max_length=384,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding='max_length',
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
        # The offset mappings will give us a map from token to character position in the original context.
        # This will help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop('offset_mapping')

        # Let's label those examples!
        tokenized_examples['start_positions'] = []
        tokenized_examples['end_positions'] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples['input_ids'][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples['answers'][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers['answer_start']) == 0:
                tokenized_examples['start_positions'].append(cls_index)
                tokenized_examples['end_positions'].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers['answer_start'][0]
                end_char = start_char + len(answers['text'][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples['start_positions'].append(cls_index)
                    tokenized_examples['end_positions'].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples['start_positions'].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples['end_positions'].append(token_end_index + 1)

        return tokenized_examples

    # Apply the function to our data
    tokenized_datasets = sberquad.map(prepare_train_features, batched=True, remove_columns=sberquad['train'].column_names)

    args = TrainingArguments(
        f'finetune-ruBERT-sberquad',
        evaluation_strategy = 'epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    data_collator = DefaultDataCollator()

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Run the trainer
    trainer.train()

    # Save the model and tokenizer
    trainer.save_model(dir_name)

    print(f'Model saved to {dir_name}')


if __name__ == '__main__':
    main()
