# from: https://github.com/openai/simple-evals/blob/5e623c2b400af62a1278e23595f95b0853d7fe8a/browsecomp_eval.py#L26-L45 and https://github.com/openai/simple-evals/blob/5e623c2b400af62a1278e23595f95b0853d7fe8a/simpleqa_eval.py#L12-L104. We keep the output format of browsecomp.

JUDGE_PROMPT_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.


Additional instructions:
The following are examples of CORRECT responses.
```
question: What are the names of Barack Obama's children?
correct_answer: Malia Obama and Sasha Obama
response 1: sasha and malia obama
response 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
response 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These responses are all CORRECT because:
    - They fully contain the important information in the correct_answer.
    - They do not contain any information that contradicts the correct_answer.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the correct_answer is fully included and the response contains no incorrect information or contradictions.

The following are examples of INCORRECT responses.
```
Question: What are the names of Barack Obama's children?
correct_answer: Malia and Sasha
response 1: Malia.
response 2: Malia, Sasha, and Susan.
response 3: Barack Obama does not have any children.
response 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
response 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
response 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
response 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
response 7: I don't know.
response 8: I need more context about which Obama you are talking about.
response 9: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
response 10: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These responses are all INCORRECT because of the following reasons:
    - A factual statement in the answer contradicts the correct_answer. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.
    - The important information in the correct_answer is not included in the answer.
    - No statements in the answer contradict the correct_answer.


Also note the following things:
- For grading questions where the correct_answer is a number, the response needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with correct_answer "120k". 
    - responses "120k", "124k", and 115k" are all CORRECT. 
    - responses "100k" and "113k" are INCORRECT. 
    - responses "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the correct_answer.
- The correct_answer may contain more information than the question. In such cases, the response only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with correct_answer "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish responses if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the correct_answer "San Francisco, California". The response "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the correct_answer is "Outstanding Paper Award". The response "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the correct_answer is "1.73 m". The response "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the correct_answer is "Michelle Obama". The response "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name. 
    - For example, if the correct_answer is "Hyung Won Chung", you can consider the following responses as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".

"""