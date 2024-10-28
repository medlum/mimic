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
                   initial_sidebar_state='expanded',
                   page_icon="🚑")

st.subheader("About Patient's Record")
st.write("MIMIC-IV-ED is a large, freely available database of emergency department (ED) admissions at the Beth Israel Deaconess Medical Center between 2011 and 2019. The database contains ~425,000 ED stays. Vital signs, triage information, medication reconciliation, medication administration, and discharge diagnoses are available. All data are deidentified to comply with the Health Information Portability and Accountability Act (HIPAA) Safe Harbor provision. MIMIC-IV-ED is intended to support a diverse range of education initiatives and research studies.")
st.write("https://physionet.org/content/mimic-iv-ed/2.2/")

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
# df_emar_detail = pd.read_csv('./hosp_data/emar_detail.csv')
# df_emar_merged = pd.merge(df_emar,
#                          df_emar_detail, on=['subject_id', 'emar_id', 'emar_seq', 'pharmacy_id'], how='inner')

# subject_id for selection
subject_id = df_edstays['subject_id'].unique()
option = st.selectbox("Medical Records", subject_id,
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

select_emar = df_emar[
    df_emar['subject_id'] == option]
# select_emar = df_emar_merged[
#    df_emar_merged['subject_id'] == option]


# write csv after selected patient
select_edstays.to_csv('./temp_data/select_edstays.csv')
select_diagnosis.to_csv('./temp_data/select_diagnosis.csv')
select_medrecon.to_csv('./temp_data/select_medrecon.csv')
select_pyxis.to_csv('./temp_data/select_pyxis.csv')
select_triage.to_csv('./temp_data/select_triage.csv')
select_vitalsign.to_csv('./temp_data/select_vitalsign.csv')
select_d_icd_procedures.to_csv('./temp_data/select_d_icd_procedures.csv')
select_emar.to_csv('./temp_data/select_emar.csv')

# with st.sidebar:


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Admission",
                                                          "Diagnosis",
                                                          "Triage",
                                                          "Medicine",
                                                          "Vitals",
                                                          "Medrecon",
                                                          "Emar",
                                                          "Procedure"])

with tab1:
    st.dataframe(select_edstays,
                 hide_index=True,
                 column_config={
                     "subject_id": st.column_config.NumberColumn(format="%d"),
                     "hadm_id": st.column_config.NumberColumn(format="%d"),
                     "stay_id": st.column_config.NumberColumn(format="%d"),
                 })
    # for stay_id in (select_edstays["stay_id"]):
    # date = st.columns(2)

    # date[0].text(select_edstays["intime"].to_string(index=False))
    # date[1].text(select_edstays["outtime"].to_string(index=False))

with tab2:
    st.dataframe(select_diagnosis,
                 hide_index=True,
                 use_container_width=False,
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

with tab7:
    st.dataframe(select_emar,
                 hide_index=True,
                 column_config={
                     "subject_id": st.column_config.NumberColumn(format="%d"),
                     "hadm_id": st.column_config.NumberColumn(format="%d"),
                     "stay_id": st.column_config.NumberColumn(format="%d"),
                     "pharmacy_id": st.column_config.NumberColumn(format="%d")
                 })

with tab8:
    st.dataframe(select_d_icd_procedures,
                 hide_index=True,
                 use_container_width=False,
                 column_config={
                     "subject_id": st.column_config.NumberColumn(format="%d"),
                     "hadm_id": st.column_config.NumberColumn(format="%d"),
                     "stay_id": st.column_config.NumberColumn(format="%d"),
                 })


# ---- set up creative chat history ----#
chat_msg = StreamlitChatMessageHistory(key="chat_key")
chat_history_size = 3

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
    max_iterations=15,
    handle_parsing_errors=True,
    verbose=True,
    agent_kwargs=agent_kwargs
)


# set up session state as a gate to display welcome message
if 'initial_msg' not in st.session_state:
    st.session_state.initial_msg = 0


avatar_style = "shapes"
seed_user = "luis"
seed_bot = "kingston"

# ------ initial welcome message -------#
if option is not None:
    with st.sidebar:
        st.subheader("Chatbot")

        # ---------set up welcome msg-------------#
        # You have selected patient id: {option}
        welcome_msg = f"Hi there, I'm MIMIC! You can choose an option or ask any question about patient: {option}."
        # if 0, add welcome message to chat_msg
        if st.session_state.initial_msg == 0:
            chat_msg.add_ai_message(welcome_msg)

        # ------ set up message from chat history  -----#

        for index, msg in enumerate(chat_msg.messages):

            # bot's message is in even position as welcome message is added at initial
            if index % 2 == 0:
                with st.chat_message("assistant"):
                    st.write(msg.content.replace("<|im_start|>", "").replace(
                        "<|im_end|>", "").replace("<|eot_id|>", "").replace("AI:", "").replace("Human:", ""))

                # message(msg.content.replace("<|im_start|>", "").replace("<|im_end|>", "").replace("<|eot_id|>", "").replace("AI:", "").replace("Human:", ""),
                #        is_user=False,
                #        key=f"bot{index}",
                #        avatar_style=avatar_style,
                #        seed=seed_bot,
                #        allow_html=True,
                #        is_table=True,)

            # user's message is in odd position
            else:
                with st.chat_message("user"):
                    st.write(msg.content.replace(
                        "<|im_start|>", "").replace("<|im_end|>", ""))

                # message(msg.content.replace("<|im_start|>", "").replace("<|im_end|>", ""),
                #        is_user=True,
                #        key=f"user{index}",
                #        avatar_style=avatar_style,
                #        seed=seed_user)

            # set initial_msg to 0 in first loop
            if index == 0:
                st.session_state.initial_msg = 1

        # ------ set up magic button -----#
        example_prompts = [
            "Admission history",
            "Diagnosis records",
            "Triage records",
            "Vital sign records",
            "Medicine Reconcilliation records",
            "Medicine dispensed records",
            "Inpatient procedures records",
            "Inpatient medicine records",
            "A detail summary history",
            # "I want to input a stay_id"
        ]

        example_prompts_help = [
            "help message",
            "help message",
            "help message",
            "help message",
            "help message",
            "help message",
            "help message",
            "help message",
            "help message",
            # "help message",
        ]

        button_cols = st.columns(3)
        button_cols_2 = st.columns(3)
        button_cols_3 = st.columns(3)
        button_cols_4 = st.columns(3)

        button_pressed = ""

        if button_cols[0].button(example_prompts[0], help=example_prompts_help[0]):
            button_pressed = example_prompts[0]
        elif button_cols[1].button(example_prompts[1], help=example_prompts_help[1]):
            button_pressed = example_prompts[1]
        elif button_cols[2].button(example_prompts[2], help=example_prompts_help[2]):
            button_pressed = example_prompts[2]

        elif button_cols_2[0].button(example_prompts[3], help=example_prompts_help[3]):
            button_pressed = example_prompts[3]
        elif button_cols_2[1].button(example_prompts[4], help=example_prompts_help[4]):
            button_pressed = example_prompts[4]
        elif button_cols_2[2].button(example_prompts[5], help=example_prompts_help[5]):
            button_pressed = example_prompts[5]

        elif button_cols_3[0].button(example_prompts[6], help=example_prompts_help[6]):
            button_pressed = example_prompts[6]
        elif button_cols_3[1].button(example_prompts[7], help=example_prompts_help[7]):
            button_pressed = example_prompts[7]
        elif button_cols_3[2].button(example_prompts[8], help=example_prompts_help[8]):
            button_pressed = example_prompts[8]

        # elif button_cols_4[0].button(example_prompts[9], help=example_prompts_help[9]):
        #    button_pressed = example_prompts[9]

        # ------ set up user input -----#

        if prompt := (st.chat_input("Ask me a question...") or button_pressed):

            with st.chat_message("user"):
                st.write(f'{prompt}')

            # show prompt message
            # message(f'{prompt}',
            #        is_user=True,
            #        key=f"user",
            #        avatar_style=avatar_style,
            #        seed=seed_user,)

            with st.spinner("Retrieving records..."):

                try:
                    # use {'input': f'{prompt}<|eot_id|>'})
                    response = executor.invoke(
                        {'input': f'<|im_start|>{prompt}<|im_end|>'})

                    with st.chat_message("assistant"):
                        st.write(response['output'].replace("<|im_start|>", "").replace(
                            "<|im_end|>", "").replace("<|eot_id|>", "").replace("<|endoftext|>", ""))

                    # message(response['output'].replace("<|im_start|>", "").replace("<|im_end|>", "").replace("<|eot_id|>", "").replace("<|endoftext|>", ""),
                    #        is_user=False,
                    #        key=f"bot_2",
                    #        avatar_style=avatar_style,
                    #        seed=seed_bot,
                    #        allow_html=True,
                    #        is_table=True,)

                except OverloadedError as error:
                    st.write(
                        "HuggingFace🤗 inference engine is overloaded now. Try toggling to the creative mode in the meantime.")

            st.rerun()

if option is None:
    chat_msg.clear()
    st.session_state.initial_msg = 0
