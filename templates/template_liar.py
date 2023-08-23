## Liar ##

# without ICL
# solve_prompt_template = '''
# # Task
# {instruction}

# # Output format
# Answer "Yes" or "No" as the label

# # Prediction
# Text: {text}
# Label:[]
# '''

# with ICL
solve_prompt_template = '''
# Task
{instruction}

# Example
Text: "(a CNN interview) U.S. Congressman phil-gingrey: The federal government can tell General Motors what to charge for its automobiles."
Label: Yes

# Output format
Text: {text}
Answer "Yes" or "No" as the label
'''

feedback_prompt_agnews = '''
I'm trying to write a lie statement classifier prompt.

My current prompt is:
"{prompt}"

But this prompt gets the following examples wrong:
{error_string}

give {num_feedbacks} reasons why the prompt could have gotten these examples wrong. Wrap each reason with <START> and <END>
'''

# without (zero-shot) CoT
# instruction0_agnews = 'Determine whether the Statement is a lie (Yes) or not (No) based on the Context and other information.'

# with (zero-shot) CoT
# instruction0_agnews = 'Determine whether the Statement is a lie (Yes) or not (No) based on the Context and other information. Let us think step-by-step.'

# with inverse thinking
instruction0_agnews = 'Determine whether the Statement is a lie (Yes) or not (No) based on the Context and other information, when you are not sure, try to use the method of elimination.'


review_prompt_agnews = '''
# Task
In the task of classifying lie statement, is it correct to classify the following statement as a {answer} statement?

# Output format
Answer "Yes" or "No" as the label

# Prediction
Text: {text}
Label:[]
'''
