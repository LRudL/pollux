![image](https://github.com/user-attachments/assets/1d3d985a-e84b-4adc-9d6c-79f7d09a93ee)# Polllux
This is a repository for creating a digital twin of a person given their data, named after pollux of polux twinsthe immortal. To run the model, load the chrome-module packed extension in #chrome-extension. Given a user's writing, the training pipeline  generates synthetic data to train a chat model which can imitate the user's konwledfe and writing style. It comes with both a training pipeline for finetuning custom models, as well as a chrome extension to query the digital twin.

This repository is organised as follows:

- `datasets/` - Generated synthetic datasets
- `scripts/` - Scripts for training an inferencing the model.
  - `sft_blog_posts.py` - Finetunes an LLM on blog posts from a desired user.
- `data/` - Generated data from running the scripts. Preloaded with two files:
  - `cip_questions_original.json` - The examples in the original  (which were 'ExemplarConfig' objects defined in .ts files in the original repo) in the original CiP repo, turned into a jso format. Any with empty rubric lists have been removed.
  - `cip_stories_synthetic.json` - These are synthetically generated "stories", each on corresponding to a user query in `cip_questions_original.json`. See [here](https://chatgpt.com/share/67b27777-f728-8005-8dbb-46fb7b7302b3) for the ChatGPT conversation that generated them.
- `logs/` - This is where inspect will write its logs by default. Some example logs are provided as part of the repo.
- `llms.md` - A file written for LLM coding assistants, to give them context on what is being implemented.

## Results


![image](https://github.com/user-attachments/assets/e10386e5-8a6f-43ba-b190-7a1026c0772d)


*Results*

We created a benchmark of 10 economics questions where we might want Duncan’s input.

We compared:
•⁠  ⁠expert answers, submitted by Duncan
•⁠  ⁠⁠un-finetuned gemma-7b-it answers 
•⁠  ⁠⁠finetuned gemma-7b-it answers

We asked Claude-3.6 to grade the resulting answers for quality.

We found a win-rate, as judged by Claude-3.6 presented with all three answers at once, of 50% by Duncan answers, 10% by Gemma un-finetuned, and 40% by finetuned Gemma.

This gives hope that specific finetunes of even small models might help give good, personalised expert takes.

*Data generation*

The fine-tuning dataset consisted of:
•⁠  ⁠18 papers or blog posts from Duncan (chopped into slices for length)
•⁠  ⁠⁠a dataset of question-answer pairs, where we had Claude-3.6 generate questions relevant to each of Duncan’s posts, and then answer those questions based on Duncan’s takes in the post



