import os
from typing import TypedDict, Optional, Dict, List, Literal, Annotated

import streamlit as st
import pandas as pd
import joblib
import shap

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
    AnyMessage,
)
from langchain.tools import tool


@tool
def query_normal_range(indicator: str) -> str:
    """Query the normal reference range for a medical indicator.

    Args:
        indicator: The medical indicator name, one of 'ANC','ALT','AST','BMI'.
    """
    ranges = {
        "ANC": "1.5 - 8.0 x10^9/L",
        "ALT": "7 - 56 U/L",
        "AST": "10 - 40 U/L",
        "BMI": "18.5 - 24.9 kg/m^2 (normal weight)",
    }
    return ranges.get(indicator.upper(), "Normal range not found for this indicator.")


@tool
def classify_bmi(bmi: float) -> str:
    """Classify BMI category according to WHO standards.

    Args:
        bmi: Body Mass Index value.
    """
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


TOOLS = [query_normal_range, classify_bmi]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


MODEL_PATH = os.getenv("MODEL_PATH", "models/random_forest_best_model.pkl")
FEATURE_NAMES = ["ANC", "ALT", "AST", "Gender", "BMI"]


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Please place random_forest_best_model.pkl in the models folder "
            "or set the MODEL_PATH environment variable."
        )
    return joblib.load(MODEL_PATH)


model = load_model()


class AgentState(TypedDict):
    anc: float
    alt: float
    ast: float
    gender: str
    bmi: float
    api_key: str

    probability: Optional[float]
    prediction: Optional[str]
    shap_values: Optional[Dict[str, float]]
    top_positive_features: Optional[List[str]]
    top_negative_features: Optional[List[str]]

    explanation: Optional[str]
    reviewer_feedback: Optional[str]
    revision_count: int

    messages: Annotated[list[AnyMessage], add_messages]


def build_input_df(anc, alt, ast, gender, bmi) -> pd.DataFrame:
    gender_num = 1 if gender.lower() == "male" else 0
    return pd.DataFrame([[anc, alt, ast, gender_num, bmi]], columns=FEATURE_NAMES)


def predict_ra_lmm_risk(anc, alt, ast, gender, bmi) -> dict:
    input_df = build_input_df(anc, alt, ast, gender, bmi)
    proba = model.predict_proba(input_df)[0, 1]
    pred = "High risk" if proba > 0.5 else "Low risk"
    return {"probability": round(float(proba), 3), "prediction": pred}


def calculate_single_sample_shap(anc, alt, ast, gender, bmi) -> Dict[str, float]:
    input_df = build_input_df(anc, alt, ast, gender, bmi)

    scaler = model.named_steps["scaler"]
    rf_model = model.named_steps["RF"]

    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=FEATURE_NAMES)

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(input_scaled_df, check_additivity=False)

    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1][0]
    else:
        shap_values_class1 = shap_values[0, :, 1]

    return {
        feature: round(float(value), 3)
        for feature, value in zip(FEATURE_NAMES, shap_values_class1)
    }


def get_llm(api_key: str, temperature: float = 0.2):
    return ChatOpenAI(
        model="deepseek-v4-pro",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=temperature,
        extra_body={"thinking": {"type": "disabled"}},
    )


def invoke_llm_with_tools(llm_with_tools, messages, max_tool_rounds: int = 5):
    """Run an LLM call and execute tool calls until the model returns final text."""
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    for _ in range(max_tool_rounds):
        tool_calls = getattr(response, "tool_calls", [])
        if not tool_calls:
            return response

        tool_messages = []
        for tc in tool_calls:
            selected_tool = TOOLS_BY_NAME[tc["name"]]
            observation = selected_tool.invoke(tc["args"])
            tool_messages.append(
                ToolMessage(
                    content=str(observation),
                    tool_call_id=tc["id"],
                    name=tc["name"],
                )
            )

        messages.extend(tool_messages)
        response = llm_with_tools.invoke(messages)
        messages.append(response)

    return response


def prediction_and_shap_node(state: AgentState):
    pred_result = predict_ra_lmm_risk(
        state["anc"], state["alt"], state["ast"], state["gender"], state["bmi"]
    )
    shap_result = calculate_single_sample_shap(
        state["anc"], state["alt"], state["ast"], state["gender"], state["bmi"]
    )

    sorted_features = sorted(shap_result.items(), key=lambda item: item[1], reverse=True)
    top_positive = [f for f, v in sorted_features if v > 0][:3]
    top_negative = [f for f, v in reversed(sorted_features) if v < 0][:3]

    return {
        "probability": pred_result["probability"],
        "prediction": pred_result["prediction"],
        "shap_values": shap_result,
        "top_positive_features": top_positive,
        "top_negative_features": top_negative,
        "revision_count": 0,
        "reviewer_feedback": "None",
    }


def drafting_node(state: AgentState):
    llm = get_llm(state["api_key"])
    llm_with_tools = llm.bind_tools(TOOLS)

    if state["prediction"] == "High risk":
        persona = (
            "You are a serious clinical AI assistant for rheumatoid arthritis (RA) patients. "
            "Focus on the risk that this RA patient may have comorbid low muscle mass, "
            "risk mitigation, clinical follow-up, and the top risk-increasing SHAP features."
        )
    else:
        persona = (
            "You are a preventive health AI assistant for rheumatoid arthritis (RA) patients. "
            "Focus on maintaining muscle health, reassuring the user when appropriate, "
            "and explaining the top risk-decreasing SHAP features for low muscle mass."
        )

    system_msg = SystemMessage(content=persona)
    prompt = (
        f"Patient Data: ANC={state['anc']}, ALT={state['alt']}, AST={state['ast']}, "
        f"Gender={state['gender']}, BMI={state['bmi']}\n"
        f"Model Prediction for comorbid low muscle mass in this RA patient: "
        f"{state['prediction']} (Probability: {state['probability']})\n"
        f"SHAP Values: {state['shap_values']}\n"
        f"Top Risk Factors: {state['top_positive_features']}\n"
        f"Top Protective Factors: {state['top_negative_features']}\n\n"
        "Please generate a structured report explaining whether this rheumatoid arthritis patient "
        "is predicted to have comorbid low muscle mass. You may use the available tools to retrieve "
        "normal ranges or BMI classification to strengthen your explanation. Do not provide a medical "
        "diagnosis, but explain the machine learning findings clearly. Include a clear disclaimer that "
        "the user should consult a qualified doctor."
    )

    if state["reviewer_feedback"] not in ("None", "PASS"):
        prompt += (
            f"\n\n[CRITICAL FEEDBACK FROM REVIEWER]: {state['reviewer_feedback']}\n"
            "Please revise your previous report based strictly on this feedback."
        )

    new_messages = [system_msg, HumanMessage(content=prompt)]
    final_response = invoke_llm_with_tools(llm_with_tools, new_messages)

    return {
        "explanation": final_response.content,
        "revision_count": state["revision_count"] + 1,
        "messages": new_messages,
    }


def reflection_node(state: AgentState):
    llm = get_llm(state["api_key"], temperature=0.0)

    eval_prompt = (
        "You are an AI Clinical Auditor for RA-LMM-assistant. "
        "Review the following report generated by another AI. Do not call tools. "
        "Your job is only to judge whether the report is acceptable.\n\n"
        f"Report:\n{state['explanation']}\n\n"
        "Check for the following:\n"
        "1. Did it clearly state the probability and prediction for low muscle mass comorbidity "
        "in a rheumatoid arthritis patient?\n"
        "2. Did it explain at least one SHAP feature in simple terms?\n"
        "3. Did it include a disclaimer to consult a doctor?\n"
        "4. Does it avoid obvious factual errors in the referenced medical values?\n\n"
        "If the report passes ALL criteria, respond exactly with: PASS\n"
        "If it fails, provide only one brief sentence describing what must be fixed."
    )

    new_messages = [HumanMessage(content=eval_prompt)]
    response = llm.invoke(new_messages)
    new_messages.append(response)

    content = response.content.strip()
    feedback = "PASS" if content.upper() == "PASS" else content

    return {
        "reviewer_feedback": feedback,
        "messages": new_messages,
    }


def route_after_reflection(state: AgentState) -> Literal["drafting_agent_node", "__end__"]:
    if state["reviewer_feedback"] == "PASS" or state["revision_count"] >= 2:
        return END
    return "drafting_agent_node"


workflow = StateGraph(AgentState)

workflow.add_node("perception_node", prediction_and_shap_node)
workflow.add_node("drafting_agent_node", drafting_node)
workflow.add_node("reflection_agent_node", reflection_node)

workflow.add_edge(START, "perception_node")
workflow.add_edge("perception_node", "drafting_agent_node")
workflow.add_edge("drafting_agent_node", "reflection_agent_node")

workflow.add_conditional_edges(
    "reflection_agent_node",
    route_after_reflection,
    {
        "drafting_agent_node": "drafting_agent_node",
        END: END,
    },
)

agent_graph = workflow.compile()


# ---------- Streamlit UI ----------
st.set_page_config(page_title="RA-LMM-assistant", page_icon=":material/medical_services:")
st.title("RA-LMM-assistant")

st.markdown("""
> **Clinical Task**: Predict whether a patient with rheumatoid arthritis (RA) may have comorbid low muscle mass (LMM).
>
> **System Architecture**: This system uses an LLM-driven **Agentic Workflow** with **Dynamic Persona Routing**,
> **Tool Augmentation** (medical reference tools), and a **Self-Reflection Mechanism** to improve report quality.
""")

deepseek_api_key = st.text_input("DeepSeek API Key", type="password")

with st.form("risk_form"):
    anc = st.number_input("ANC", min_value=0.0, max_value=100.0, value=3.0, step=0.1)
    alt = st.number_input("ALT", min_value=0.0, max_value=1000.0, value=25.0, step=1.0)
    ast = st.number_input("AST", min_value=0.0, max_value=1000.0, value=22.0, step=1.0)
    gender = st.selectbox("Gender", options=["male", "female"])
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=24.0, step=0.1)

    submitted = st.form_submit_button("Assess RA-LMM Risk")

if submitted:
    if not deepseek_api_key:
        st.warning("Please enter your DeepSeek API Key.")
        st.stop()

    initial_state = {
        "anc": anc,
        "alt": alt,
        "ast": ast,
        "gender": gender,
        "bmi": bmi,
        "api_key": deepseek_api_key,
        "probability": None,
        "prediction": None,
        "shap_values": None,
        "top_positive_features": None,
        "top_negative_features": None,
        "explanation": None,
        "reviewer_feedback": "None",
        "revision_count": 0,
        "messages": [],
    }

    try:
        with st.spinner("RA-LMM-assistant is predicting, explaining, and self-checking..."):
            final_state = agent_graph.invoke(initial_state)
    except Exception as exc:
        st.error(f"Assessment failed: {exc}")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Low Muscle Mass Probability", f"{final_state['probability']}")

    with col2:
        if final_state["prediction"] == "High risk":
            st.error("Prediction: High risk of comorbid low muscle mass")
        else:
            st.success("Prediction: Low risk of comorbid low muscle mass")

    st.subheader("SHAP Attribution for RA-LMM Prediction")
    shap_df = pd.DataFrame({
        "Feature": list(final_state["shap_values"].keys()),
        "SHAP Value": list(final_state["shap_values"].values()),
    }).sort_values("SHAP Value", ascending=True)

    st.bar_chart(shap_df, x="Feature", y="SHAP Value")

    st.subheader("Final RA-LMM-assistant Report")
    st.markdown(final_state["explanation"])

    with st.expander("View Agent Internal Reasoning Details"):
        st.json({
            "Architecture": "RA-LMM-assistant LangGraph Actor-Critic Agent with Tool Augmentation",
            "Cognitive Steps Taken": final_state["revision_count"],
            "Final Self-Reflection Status": final_state["reviewer_feedback"],
            "Dynamic Persona Used": (
                "Clinical Intervention"
                if final_state["prediction"] == "High risk"
                else "Preventive Health"
            ),
            "State Variables": {
                "top_positive_features": final_state["top_positive_features"],
                "top_negative_features": final_state["top_negative_features"],
            },
            "Full Conversation Trace": [
                {
                    "role": msg.__class__.__name__,
                    "content": getattr(msg, "content", str(msg)),
                    "tool_calls": getattr(msg, "tool_calls", None),
                }
                for msg in final_state.get("messages", [])
            ],
        })