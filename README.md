# Polllux - Creating digital twins from synthetic data

This is a repository for creating a digital twin of a person given their data, named after pollux of polux twinsthe immortal. To run the model, load the chrome-module packed extension in #chrome-extension. Given a user's writing, the training pipeline  generates synthetic data to train a chat model which can imitate the user's konwledfe and writing style. It comes with both a training pipeline for finetuning custom models, as well as a chrome extension to query the digital twin. 

This repository is organised as follows:

- `datasets/` - Generated synthetic datasets
- `scripts/` - Scripts for training an inferencing the model.
- `chrome-extension/` - Directory for loading the chrome extension.

To demonstrate out capability we trained a model off of the entire written corpus of Duncan McClements, a 19 year old economics prodigy whose work is already being used by the UK government. We’ve worked with Duncan extensively on economics research, so we know his output style and context. We also designed DuncanBench, a 10-question benchmark designed to see if the model answered questions in Duncan’s style and with his takes.

![image](https://github.com/user-attachments/assets/711bc44c-e544-442a-9fe5-de7ecb9a0a8c)

## Results

![image](https://github.com/user-attachments/assets/e10386e5-8a6f-43ba-b190-7a1026c0772d)


We created DuncanBench, a benchmark of 10 economics questions where we might want Duncan’s input. The questions and comparison can be found [here](https://github.com/LRudL/pollux/blob/main/datasets/duncanbench/comparison.json).

We compared:

- ⁠expert answers, submitted by Duncan today
- ⁠⁠un-finetuned gemma-7b-it answers
- ⁠⁠finetuned gemma-7b-it answers

We asked Claude-3.6 to grade the resulting answers for quality. We found a win-rate, as judged by Claude-3.6 presented with all three answers at once, of 50% by Duncan answers, 10% by Gemma un-finetuned, and 40% by finetuned Gemma. This gives hope that specific finetunes of even small models might help give good, personalised expert takes.

**Data generation**

The fine-tuning dataset consisted of:

- 18 papers or blog posts from Duncan (chopped into slices for length)
- ⁠⁠a dataset of question-answer pairs, where we had Claude-3.6 generate questions relevant to each of Duncan’s posts, and then answer those questions based on Duncan’s takes in the post
