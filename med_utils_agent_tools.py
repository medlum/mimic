from langchain.tools import StructuredTool
from langchain.tools import StructuredTool
import pandas as pd


def edstays(file: str):
    df = pd.read_csv('./temp_data/select_edstays.csv')
    # return df
    return df.to_csv()


edstays_tool = StructuredTool.from_function(
    func=edstays,
    name='admission records',
    description="Use this tool to access patients admission records."

)


def diagnosis(file: str):
    df = pd.read_csv('./temp_data/select_diagnosis.csv')
    # return df
    return df.to_csv()


diagnosis_tool = StructuredTool.from_function(
    func=diagnosis,
    name='diagnosis records',
    description="Use this tool to access patients diagnosis records"
)


def triage(file: str):
    df = pd.read_csv('./temp_data/select_triage.csv')
    # return df
    return df.to_csv()


triage_tool = StructuredTool.from_function(
    func=triage,
    name='triage records',
    description="Use this tool to access patients triage records"
)


def medrecon(file: str):
    df = pd.read_csv('./temp_data/select_medrecon.csv')
    # return df
    return df.to_csv()


medrecon_tool = StructuredTool.from_function(
    func=medrecon,
    name='medrecon records',
    description="Use this tool to access patients medication reconciliation records"
)


def pyxis(file: str):
    df = pd.read_csv('./temp_data/select_pyxis.csv')
    # return df
    return df.to_csv()


pyxis_tool = StructuredTool.from_function(
    func=pyxis,
    name='pyxis records',
    description="Use this tool to access medication dispensation records for patient under admission"
)


def vitalsign(file: str):
    df = pd.read_csv('./temp_data/select_vitalsign.csv')
    # return df
    return df.to_csv()


vitalsign = StructuredTool.from_function(
    func=vitalsign,
    name='vitalsign records',
    description="Use this tool to access vital sign records of patients"
)


toolkit = [edstays_tool,
           diagnosis_tool,
           triage_tool,
           medrecon_tool,
           pyxis_tool,
           vitalsign]