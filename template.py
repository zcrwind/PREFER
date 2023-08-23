## agnews ##

## zero-shot with ICL ##
# solve_prompt_template = '''
# # Task
# Classify the given news into ONE of the following four categories: "World", "Sports", "Business", "Science"

# # Output format
# Answer "World" or "Sports" or "Business" or "Science" as the label

# # Example
# Text: "Surviving Biotech's Downturns Charly Travers offers advice on withstanding the volatility of the biotech sector."
# Label: Business

# # Prediction
# Text: {text}
# Label:
# '''

## zero-shot without ICL ##
solve_prompt_template = '''
# Task
{instruction}

# Output format
Answer "World" or "Sports" or "Business" or "Science" as the label

# Prediction
Text: {text}
Label:
'''

solve_prompt_template_backup = '''
# Task
{instruction}

# Example
Text: "Surviving Biotech's Downturns Charly Travers offers advice on withstanding the volatility of the biotech sector."
Label: Business

# Output format
Text: {text}
Answer "World" or "Sports" or "Business" or "Science" as the label
'''

feedback_prompt_agnews = '''
I'm trying to write a zero-shot classifier prompt.

My current prompt is:
"{prompt}"

But this prompt gets the following examples wrong:
{error_string}

give {num_feedbacks} reasons why the prompt could have gotten these examples wrong. Wrap each reason with <START> and <END>
'''

## without (zero-shot) CoT
instruction0_agnews = 'Classify the given news into ONE of the following four categories: "World", "Sports", "Business", "Science"'

## with (zero-shot) CoT
# instruction0_agnews = 'Classify the given news into ONE of the following four categories: "World", "Sports", "Business", "Science". Let us think step-by-step.'

## with reverse thinking
# instruction0_agnews = 'Classify the given news into ONE of the following four categories: "World", "Sports", "Business", "Science", when you are not sure, try to use the method of elimination.'
