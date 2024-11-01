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
    description="Use this tool to access admission records."

)


def diagnosis(file: str):
    df = pd.read_csv('./temp_data/select_diagnosis.csv')
    # return df
    return df.to_csv()


diagnosis_tool = StructuredTool.from_function(
    func=diagnosis,
    name='diagnosis records',
    description="Use this tool to access diagnosis records"
)


def triage(file: str):
    df = pd.read_csv('./temp_data/select_triage.csv')
    # return df
    return df.to_csv()


triage_tool = StructuredTool.from_function(
    func=triage,
    name='triage records',
    description="Use this tool to access triage records"
)


def medrecon(file: str):
    df = pd.read_csv('./temp_data/select_medrecon.csv')
    # return df
    return df.to_csv()


medrecon_tool = StructuredTool.from_function(
    func=medrecon,
    name='medrecon records',
    description="Use this tool to access medication reconciliation records"
)


def pyxis(file: str):
    df = pd.read_csv('./temp_data/select_pyxis.csv')
    # return df
    return df.to_csv()


pyxis_tool = StructuredTool.from_function(
    func=pyxis,
    name='medicine dispensed records',
    description="Use this tool to access medication dispensed records"
)


def vitalsign(file: str):
    df = pd.read_csv('./temp_data/select_vitalsign.csv')
    # return df
    return df.to_csv()


vitalsign_tool = StructuredTool.from_function(
    func=vitalsign,
    name='vital sign records',
    description="Use this tool to access vital sign records"
)


def med_procedure(file: str):
    df = pd.read_csv('./temp_data/select_d_icd_procedures.csv')
    # return df
    return df.to_csv()


procedure_tool = StructuredTool.from_function(
    func=med_procedure,
    name='inpatient procedures records',
    description="Use this tool to access inpatient procedures records"
)


def emar(file: str):
    df = pd.read_csv('./temp_data/select_emar.csv')
    # return df
    return df.to_csv()


emar_tool = StructuredTool.from_function(
    func=emar,
    name='inpatient medicine records',
    description="Use this tool to access inpatient medicine records"
)


toolkit = [edstays_tool,
           diagnosis_tool,
           triage_tool,
           medrecon_tool,
           pyxis_tool,
           vitalsign_tool,
           procedure_tool,
           emar_tool]
