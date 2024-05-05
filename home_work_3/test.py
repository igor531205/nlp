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
    
    parser.add_argument('number_of_data', type=int, default=1,\
                        help='Question number in the test dataset')

    args = parser.parse_args()
    dir_name = args.dir_name
    number_of_data = args.number_of_data

    # Load the dataset
    sberquad = load_dataset('sberquad')

    # Load the tokenizer and model
    model_name = 'gunkusha0/model_qa' # Or dir_name for local model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    def predicted_answer(instance):
        # Question and context
        context = instance['context']
        question = instance['question']

        # Tokenize the data
        inputs = tokenizer(question, context, return_tensors='pt', max_length=512, truncation=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get model's output
        with torch.no_grad():
            output = model(**inputs)

        # Get the predicted answer
        start_idx = torch.argmax(output.start_logits)
        end_idx = torch.argmax(output.end_logits)

        predicted_answer = tokenizer.convert_tokens_to_string(\
                        tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_idx:end_idx + 1])
        )

        print('Вопрос:', question)  
        print('Ответ:', predicted_answer)    

    predicted_answer(sberquad['test'][number_of_data])    
    

if __name__ == '__main__':
    main()
