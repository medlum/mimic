from langchain_core.prompts import (PromptTemplate, MessagesPlaceholder)
from langchain_core.prompts import (
    MessagesPlaceholder,
)
import pandas as pd
# When summarizing admission records, describe the race and gender and put stay_id, intime, outtime, arrival_transport,disposition columns into a table

# Summarize medicine dispensed during admission in a table with charttime, name columns.
# If the user want to input a stay_id, ask the user what is the stay_id before proceeding.
# Or ask for an detailed summary of a patient across the types of records or ask you about a summary based on patient's stay_id.


# If you are providing a detailed summary of the patient's history, always include patient's gender,
# race first and categorize the summary by:
# - Admission in Emergency Department
# - Diagnosis in Emergency Department
# - Triage in Emergency Department
# - Medicine dispensed in Emergency Department
# - Medical procedures in Inpatient Care
# - Medicine administered in Inpatient Care
template = """
You are a medical chatbot with access to records of patients.

The user can ask questions based on the types of record in admission, diagnosis, triage, medicine, procedures or overall summary.

You need to determine the nature of the question to access to the right record, then summarize the record with full details.

Summarize your answers in pointers whenever possible.

Always be very thorough and helpful and do not miss out anything information.  

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, you can use one or more of [{tool_names}] if necessary. 
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question<|eot_id|>

Begin! Remember to give detail and informative answers!
Previous conversation history:
{chat_history}

New question: {input}
{agent_scratchpad}"""


agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
}

prompt = PromptTemplate(input_variables=[
    "chat_history", "input", "agent_scratchpad"], template=template)


# --------- demo app text  ----------#
demo_app_text = """
This demo app showcases how an AI-powered chatbot can simplify navigation through patient records in a hospital setting.
Designed to support healthcare professionals, the chatbot enables quick access to essential patient information, such as medical history, 
current medications and summarize complex medical records into concise overviews for easier reference. 
This AI chatbot is a valuable tool for enhancing workflow efficiency, improving information accessibility, and ultimately supporting high-quality patient care.
The patient records used in this demo are extracted from MIMIC-IV-ED.

"""

# Custom CSS to style the buttons

# background-color: #fa944b;
custom_css = """
<style>
    .stButton > button {
        color: #383736; 
        border: none; /* No border */
        padding: 5px 22px; /* Reduced top and bottom padding */
        text-align: center; /* Centered text */
        text-decoration: none; /* No underline */
        display: inline-block; /* Inline-block */
        font-size: 12px !important;
        margin: 4px 2px; /* Some margin */
        cursor: pointer; /* Pointer cursor on hover */
        border-radius: 30px; /* Rounded corners */
        transition: background-color 0.3s; /* Smooth background transition */
    }
    .stButton > button:hover {
        color: #383736; 
        background-color: #c4c2c0; /* Darker green on hover */
    }
</style>
"""


# read original ed data
df_edstays = pd.read_csv('./med_data/edstays.csv')
df_diagnosis = pd.read_csv('./med_data/diagnosis.csv')
df_medrecon = pd.read_csv('./med_data/medrecon.csv')
df_pyxis = pd.read_csv('./med_data/pyxis.csv')
df_triage = pd.read_csv('./med_data/triage.csv')
df_vitalsign = pd.read_csv('./med_data/vitalsign.csv')


# ------ set up question button -----#
example_prompts = [
    "Admission history",
    "What were the diagnosis?",
    "Triage records",
    "Vitals sign records",
    "Medicine reconcilliation",
    "What medicine was dispensed?",
    "Any inpatient procedures?",
    "Any inpatient medicine adminstered?",
    "Give me a detail summary of the patient",
    # "I want to input a stay_id"
]
