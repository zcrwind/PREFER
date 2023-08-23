## Ethos ##

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
Text: "As an SJW, it is the right of black people to rob and kill me. To not allow them is a white privilege."
Label: Yes

# Output format
Answer "Yes" or "No" as the label

# Prediction
Text: {text}
Label:[]
'''

solve_prompt_template_backup = '''
# Task
{instruction}

# Example
Text: "As an SJW, it is the right of black people to rob and kill me. To not allow them is a white privilege."
Label: Yes

# Output format
Text: {text}
Answer "Yes" or "No" as the label
'''

feedback_prompt_agnews = '''
I'm trying to write a hate speech classifier prompt.

My current prompt is:
"{prompt}"

But this prompt gets the following examples wrong:
{error_string}

give {num_feedbacks} reasons why the prompt could have gotten these examples wrong. Wrap each reason with <START> and <END>
'''

# without (zero-shot) CoT
# instruction0_agnews = 'Is the following text hate speech?'

# with (zero-shot) CoT
instruction0_agnews = 'Is the following text hate speech? Let us think step-by-step.'

# inverse thinking
# instruction0_agnews = 'Is the following text hate speech? When you are not sure, try to use the method of elimination.'



