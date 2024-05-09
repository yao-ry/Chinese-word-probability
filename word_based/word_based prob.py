import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import csv

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("uer/roberta-small-word-chinese-cluecorpussmall")
model = AutoModelForMaskedLM.from_pretrained("uer/roberta-small-word-chinese-cluecorpussmall")

# Define file paths
input_file_path = "/Users/runyi/Desktop/research idea/Mandarin studies/prediction update/norming/word_based/test.csv"
output_file_path = "/Users/runyi/Desktop/research idea/Mandarin studies/prediction update/norming/word_based/outs2.csv"

# Open input file and output file
with open(input_file_path, "r", encoding="utf-8") as input_file, \
        open(output_file_path, "w", encoding="utf-8", newline="") as output_file:
    csv_reader = csv.reader(input_file)
    writer = csv.writer(output_file)
    writer.writerow(["Input Text", "Target Word", "Target Word Probability"])

    # Process each row in the input file
    for row in csv_reader:
        # Extract input text and target word from the row
        input_text, target_word = row

        # Tokenize input text
        tokenized_input = tokenizer(input_text, return_tensors="pt")
        
        # Get the position of the mask token
        mask_token_index = (tokenized_input["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=False)
        if mask_token_index.numel() == 0:
            print("Mask token not found in input:", input_text)
            continue  # Skip this input if mask token not found

        mask_token_index = mask_token_index[0, 1].item()

        # Get the predictions for the masked token
        with torch.no_grad():
            outputs = model(**tokenized_input)
            predictions = outputs.logits[0, mask_token_index].softmax(dim=0)

        # Find the index of the target word in the vocabulary
        target_word_index = tokenizer.convert_tokens_to_ids(target_word)

        # Get the probability of the target word
        target_word_probability = predictions[target_word_index].item()

        # Write input text, target word, and target word probability to output file
        writer.writerow([input_text, target_word, target_word_probability])