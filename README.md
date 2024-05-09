# yao-ry-Chinese-word-probability

These are three simple scripts to extract Chinese word (token) probabilities.

Token prediction is performed using the bert-base-Chinese model (inference API: [https://huggingface.co/google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)). It (i.e., 'single_token') generates the top 50 possible tokens along with their probabilities. I also experimented with a script （i.e.， ‘multi_token’） that can extract the probabilities of the next several tokens.

Word prediction is based on RoBERTa-small-word-Chinese-cluecorpussmall model (inference API: [RoBERTa-small-word-Chinese-cluecorpussmall](https://huggingface.co/uer/roberta-small-word-chinese-cluecorpussmall)). This script extracts the probability of target words.

Have fun playing with them!
