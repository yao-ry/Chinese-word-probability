from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn.functional as F

# Define k for top-k predictions
k = 50
NUM_MASK = 2

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
model = AutoModelForMaskedLM.from_pretrained('bert-base-chinese')

def get_word_prob(input_ids):
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Find positions of masked tokens 
    masked_indices = torch.where(input_ids == tokenizer.mask_token_id)[1]

    # Apply softmax to get probabilities
    probs = F.softmax(logits[0, masked_indices], dim=-1)

    # If you want to see the logits probabilities (beofre softmax), uncomment the following line
    #probs = logits[0, masked_indices]


    #print(probs)
    possible_results = {}
    for idx, position in enumerate(masked_indices):
        top_k_probs, top_k_token_ids = probs[idx].topk(k)
        top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_token_ids)
        possible_results[idx] = {(x,y.item()) for x, y in zip(top_k_tokens, top_k_probs)}


    all_probs = [sorted(y, key=lambda x: x[1], reverse=True) for _, y in possible_results.items()]

    return all_probs


if __name__ == '__main__':
    input_file_path = "/Users/test.csv"
    output_file_path = "/Users/outs.csv"

    # Open output file for writing
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        # Open input file for reading
        with open(input_file_path, "r", encoding="utf-8") as input_file:
            # Process each sentence in the input file
            for input_text in input_file:
                # Tokenize the input text
                input_ids = tokenizer(input_text.strip(), return_tensors='pt')['input_ids']

                # Get the top-k possible words
                all_probs = get_word_prob(input_ids)

                # Write input sentence to output file
                output_file.write(f'Input: {input_text.strip()}\n')

                # Write top-k possible words to output file
                for i in range(k):
                    words = list(all_probs[0])[i][0] + list(all_probs[1])[i][0]
                    prob_word = list(all_probs[0])[i][1] * list(all_probs[1])[i][1]
                    output_file.write(f'{words}, Probability: {prob_word:.4f}\n')

                # Add a new line between sentences
                output_file.write("\n")
