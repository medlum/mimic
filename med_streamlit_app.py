import streamlit as st
from med_utils_prompt import *
from med_utils_agent_tools import *
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import (PromptTemplate, MessagesPlaceholder)
from streamlit_chat import message
from huggingface_hub.errors import OverloadedError

# ---------set up page config -------------#
st.set_page_config(page_title="ED Chatbot",
                   layout="wide",
                   initial_sidebar_state='collapsed',
                   page_icon="🚑")

# read original ed data
df_edstays = pd.read_csv('./med_data/edstays.csv')
df_diagnosis = pd.read_csv('./med_data/diagnosis.csv')
df_medrecon = pd.read_csv('./med_data/medrecon.csv')
df_pyxis = pd.read_csv('./med_data/pyxis.csv')
df_triage = pd.read_csv('./med_data/triage.csv')
df_vitalsign = pd.read_csv('./med_data/vitalsign.csv')

# read original hosp data
df_procedures_icd = pd.read_csv('./hosp_data/procedures_icd.csv')
df_d_icd_procedures = pd.read_csv('./hosp_data/d_icd_procedures.csv')
df_d_icd_procedures_merged = pd.merge(df_d_icd_procedures,
                                      df_procedures_icd, on=['icd_code',
                                                             'icd_version'],
                                      how='inner')
df_d_icd_procedures_merged = df_d_icd_procedures_merged[['subject_id', 'hadm_id', 'seq_num',
                                                         'chartdate', 'icd_code', 'icd_version', 'long_title']]

df_emar = pd.read_csv('./hosp_data/emar.csv')
df_emar_detail = pd.read_csv('./hosp_data/emar_detail.csv')
df_emar_merged = pd.merge(df_emar,
                          df_emar_detail, on=['subject_id', 'emar_id', 'emar_seq', 'pharmacy_id'], how='inner')

# subject_id for selection
subject_id = df_edstays['subject_id'].unique()
option = st.selectbox("New York University Hospital", subject_id,
                      index=None, placeholder="Select a patient id")

# select patient by subject_id
select_edstays = df_edstays[df_edstays['subject_id']
                            == option].sort_values(by="stay_id").reset_index(drop=True)

select_diagnosis = df_diagnosis[df_diagnosis['subject_id'] == option].sort_values(
    by="stay_id")
select_medrecon = df_medrecon[df_medrecon['subject_id']
                              == option].sort_values(by="stay_id")
select_pyxis = df_pyxis[df_pyxis['subject_id']
                        == option].sort_values(by="stay_id")
select_triage = df_triage[df_triage['subject_id']
                          == option].sort_values(by="stay_id")
select_vitalsign = df_vitalsign[df_vitalsign['subject_id'] == option].sort_values(
    by="stay_id")
select_d_icd_procedures = df_d_icd_procedures_merged[
    df_d_icd_procedures_merged['subject_id'] == option]

select_emar = df_emar_merged[
    df_emar_merged['subject_id'] == option]


# write csv after selected patient
select_edstays.to_csv('./temp_data/select_edstays.csv')
select_diagnosis.to_csv('./temp_data/select_diagnosis.csv')
select_medrecon.to_csv('./temp_data/select_medrecon.csv')
select_pyxis.to_csv('./temp_data/select_pyxis.csv')
select_triage.to_csv('./temp_data/select_triage.csv')
select_vitalsign.to_csv('./temp_data/select_vitalsign.csv')
select_d_icd_procedures.to_csv('./temp_data/select_d_icd_procedures.csv')
select_emar.to_csv('./temp_data/select_emar.csv')


with st.sidebar:
    st.subheader("About Patient's Record")
    st.write("MIMIC-IV-ED is a large, freely available database of emergency department (ED) admissions at the Beth Israel Deaconess Medical Center between 2011 and 2019.")
    st.write("https://physionet.org/content/mimic-iv-ed/2.2/")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Admission",
                                                  "Diagnosis",
                                                 "Triage",
                                                  "Medicine",
                                                  "Vitals",
                                                  "Medrecon"])

    with tab1:
        st.dataframe(select_edstays,
                     hide_index=True,
                     column_config={
                         "subject_id": st.column_config.NumberColumn(format="%d"),
                         "hadm_id": st.column_config.NumberColumn(format="%d"),
                         "stay_id": st.column_config.NumberColumn(format="%d"),
                     })

    with tab2:
        st.dataframe(select_diagnosis,
                     hide_index=True,
                     column_config={
                         "subject_id": st.column_config.NumberColumn(format="%d"),
                         "hadm_id": st.column_config.NumberColumn(format="%d"),
                         "stay_id": st.column_config.NumberColumn(format="%d"),
                     })

    with tab3:
        st.dataframe(select_triage,
                     hide_index=True,
                     column_config={
                         "subject_id": st.column_config.NumberColumn(format="%d"),
                         "hadm_id": st.column_config.NumberColumn(format="%d"),
                         "stay_id": st.column_config.NumberColumn(format="%d"),
                     })

    with tab4:
        st.dataframe(select_pyxis,
                     hide_index=True,
                     column_config={
                         "subject_id": st.column_config.NumberColumn(format="%d"),
                         "hadm_id": st.column_config.NumberColumn(format="%d"),
                         "stay_id": st.column_config.NumberColumn(format="%d"),
                     })

    with tab5:
        st.dataframe(select_vitalsign,
                     hide_index=True,
                     column_config={
                         "subject_id": st.column_config.NumberColumn(format="%d"),
                         "hadm_id": st.column_config.NumberColumn(format="%d"),
                         "stay_id": st.column_config.NumberColumn(format="%d"),
                     })

    with tab6:
        st.dataframe(select_medrecon,
                     hide_index=True,
                     column_config={
                         "subject_id": st.column_config.NumberColumn(format="%d"),
                         "hadm_id": st.column_config.NumberColumn(format="%d"),
                         "stay_id": st.column_config.NumberColumn(format="%d"),
                     })

# ---- set up creative chat history ----#
chat_msg = StreamlitChatMessageHistory(key="chat_key")
chat_history_size = 2

# ---------set up LLM  -------------#
# model = "Qwen/Qwen2.5-72B-Instruct"
model = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# initialise LLM for agents and tools
llm_factual = HuggingFaceEndpoint(
    repo_id=model,
    max_new_tokens=2000,
    do_sample=False,
    temperature=0.1,
    repetition_penalty=1.1,
    return_full_text=False,
    top_p=0.2,
    top_k=40,
    huggingfacehub_api_token=st.secrets["huggingfacehub_api_token"]
)


# ---------set up general memory  -------------#
conversational_memory = ConversationBufferMemory(
    memory_key='chat_history',
    chat_memory=chat_msg,
    k=chat_history_size,
    return_messages=True
)

# ---------set up agent with tools  -------------#

react_agent = create_react_agent(
    llm_factual, toolkit, prompt)

executor = AgentExecutor(
    agent=react_agent,
    tools=toolkit,
    memory=conversational_memory,
    max_iterations=10,
    handle_parsing_errors=True,
    verbose=True,
    agent_kwargs=agent_kwargs
)

welcome_msg = """\n1. Admission
                \n2. Diagnosis 
                \n3. Triage
                \n4. Vital signs
                \n5. Medicine reconcilliation
                \n6. Medicine dispensed during admission
                \n7. Medical procedures during hospitalization
                \n8. Medicine administered during hospitalization"""

# set up session state as a gate to display welcome message
if 'initial_msg' not in st.session_state:
    st.session_state.initial_msg = 0


avatar_style = "shapes"
seed_user = "luis"
seed_bot = "kingston"

# ------ initial welcome message -------#
if option is not None:

    welcome_msg = f"What would you like to know about patient id: {option}? {welcome_msg} "
    # if 0, add welcome message to chat_msg
    if st.session_state.initial_msg == 0:
        chat_msg.add_ai_message(welcome_msg)

    # ------ set up message from chat history  -----#

    for index, msg in enumerate(chat_msg.messages):

        # bot's message is in even position as welcome message is added at initial
        if index % 2 == 0:
            message(msg.content.replace("<|im_start|>", "").replace("<|im_end|>", "").replace("<|eot_id|>", "").replace("AI:", "").replace("Human:", ""),
                    is_user=False,
                    key=f"bot{index}",
                    avatar_style=avatar_style,
                    seed=seed_bot,
                    allow_html=True,
                    is_table=True,)

        # user's message is in odd position
        else:
            message(msg.content.replace("<|im_start|>", "").replace("<|im_end|>", ""),
                    is_user=True,
                    key=f"user{index}",
                    avatar_style=avatar_style,
                    seed=seed_user)

        # set initial_msg to 0 in first loop
        if index == 0:
            st.session_state.initial_msg = 1

    # ------ set up user input -----#

    if prompt := st.chat_input("Ask me a question..."):
        # show prompt message
        message(f'{prompt}',
                is_user=True,
                key=f"user",
                avatar_style=avatar_style,
                seed=seed_user,)

        with st.spinner("Generating...（请等一下）"):

            try:

                # use {'input': f'{prompt}<|eot_id|>'})
                response = executor.invoke(
                    {'input': f'<|im_start|>{prompt}<|im_end|>'})

                message(response['output'].replace("<|im_start|>", "").replace("<|im_end|>", "").replace("<|eot_id|>", "").replace("<|endoftext|>", ""),
                        is_user=False,
                        key=f"bot_2",
                        avatar_style=avatar_style,
                        seed=seed_bot,
                        allow_html=True,
                        is_table=True,)

            except OverloadedError as error:

                st.write(
                    "HuggingFace🤗 inference engine is overloaded now. Try toggling to the creative mode in the meantime.")


if option is None:
    chat_msg.clear()
    st.session_state.initial_msg = 0
