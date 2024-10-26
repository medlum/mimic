from langchain_core.prompts import (PromptTemplate, MessagesPlaceholder)
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import (
    SystemMessage,
)

# Start your first message by introducing your name and offer two language options English or Chinese with number selection.
template = """

You are a medical chatbot with access to records of patients.

When answering query about patients record, be very thorough and do not miss out anything information.  

When summarizing admission records, describe the race and gender and put stay_id, intime, outtime, arrival_transport,disposition columns into a table

Summarize medicine dispensed during admission in a table with charttime, name columns.

Always be helpful and thorough with your answers.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, use one or more of [{tool_names}] if necessary
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question <|eot_id|>

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
