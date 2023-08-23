## SNLI & MNLI ##

solve_prompt_template = '''# Task
{instruction}
NOTE: The two sentences provide descriptions of the same thing without any chronological relationship. Let's think step by step.

# Output format
Explain your reasoning process in one sentence and Answer "Entailment" or "Neutral" or "Contradiction" as the label

# Prediction
Sentence 1: {text1}
Sentence 2: {text2}
Label:[]
'''

instruction0_agnews = 'Given two sentences, determine the relationship between them is entailment, neutral, or contradiction.'

# regular
# feedback_prompt_agnews = '''I'm trying to write a Textual Entailment task prompt.

# My current prompt is:
# "{prompt}"

# But this prompt gets the following examples wrong:
# {error_string}

# give {num_feedbacks} reasons why the prompt could have gotten these examples wrong. Wrap each reason with <START> and <END>
# '''

# synonym experiments. In fact, it should be called `feedback_prompt_synonym`
feedback_prompt_agnews = '''Given the instruction: {text}, try to rewrite it with same semantics.'''



review_prompt_agnews = '''
# Task
In the lie statement classification task, is it correct to classify the following text as a {answer} statement?

# Output format
Answer "Yes" or "No" as the label

# Prediction
Text: {text}
Label:[]
'''

