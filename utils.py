import json
import time
import copy
import typing
import os
import sys
import re
import importlib
# Streamlit
import streamlit as st
# Gurobi
import pyomo.environ as pe
from pyomo.opt import SolverFactory
from pyomo.contrib.iis import *
import re
from pyomo.core.expr.visitor import identify_mutable_parameters, replace_expressions, clone_expression
# Claude
from anthropic import Anthropic
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file
import tiktoken


from prompts import get_prompts, get_tools, get_syntax_guidance_tool
from agents import Interpreter, Coordinator, Explainer, Engineer


# --- LLM provider selection -------------------------------------------------
# OptiChat keeps OpenAI-shaped messages/tools as its internal format (see the
# adapters in agents.py), so the same agents can talk to either the Anthropic
# API or any OpenAI-compatible endpoint. Nebula (the VU's self-hosted platform)
# is OpenAI-compatible, so we reach it with the `openai` client plus a base_url.
NEBULA_BASE_URL = os.environ.get("NEBULA_BASE_URL", "https://nebula.cs.vu.nl/api")
NEBULA_MODELS = ["FAST.gpt-oss:120b", "SURF.gpt-oss-120b", "FAST.gemma3:12b", "llama3.1:8b"]
CLAUDE_MODELS = ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7"]


def get_provider():
    """Which LLM backend to use: 'nebula' (OpenAI-compatible) or 'claude'.

    Honors LLM_PROVIDER when set; otherwise prefers Nebula whenever
    NEBULA_API_KEY is present, falling back to Claude.
    """
    provider = os.environ.get("LLM_PROVIDER")
    if provider:
        return provider.lower()
    return "nebula" if os.environ.get("NEBULA_API_KEY") else "claude"


def model_options(provider=None):
    provider = provider or get_provider()
    return NEBULA_MODELS if provider == "nebula" else CLAUDE_MODELS


def default_model(provider=None):
    return model_options(provider)[0]


def make_client(provider=None):
    """Construct the LLM client for the selected provider."""
    provider = provider or get_provider()
    if provider == "nebula":
        from openai import OpenAI
        return OpenAI(api_key=os.environ["NEBULA_API_KEY"], base_url=NEBULA_BASE_URL)
    if provider == "claude":
        return Anthropic(api_key=os.environ["CLAUDE_API_KEY"])
    raise ValueError(f"Unknown LLM provider: {provider!r} (expected 'nebula' or 'claude')")


def get_agents(fn_names, client, llm=None):
    llm = llm or default_model()
    interpreter = Interpreter(client=client, llm=llm)
    explainer = Explainer(client=client, llm=llm)

    multiple_tools, single_tools, none_tools, all_tools, tool_choice = get_tools(fn_names)
    syntax_guidance_tool = get_syntax_guidance_tool()
    engineer = Engineer(client=client, llm=llm,
                        multiple_tools=multiple_tools, single_tools=single_tools,
                        none_tools=none_tools, all_tools=all_tools,
                        tool_choice=tool_choice,
                        syntax_guidance_tool=syntax_guidance_tool,
                        function_names=str(fn_names))
    coordinator = Coordinator(client=client, agents=[explainer, engineer], llm=llm)
    return interpreter, explainer, engineer, coordinator


def save_team_conversation(team_conversation, filename):
    with open(filename, 'w') as f:
        for message in team_conversation:
            f.write(json.dumps(message) + '\n')


def OptiChat_workflow_exp(args, coordinator, engineer, explainer, messages, models_dict):
    team_conversation = []
    rounds = 0

    # set the time in agents to 0
    coordinator.coordination_time = 0
    engineer.syntax_time = 0
    engineer.programing_time = 0
    engineer.evaluation_time = 0
    explainer.explanation_time = 0

    # in current design, if coordinator has assigned the task once,
    # actually there will be no need to call llm to generate the decision again
    while rounds <= coordinator.max_rounds:
        coordinator_start = time.time()
        decision = coordinator.generate_decision_exp(args, messages, team_conversation)
        coordinator_end = time.time()
        coordinator.coordination_time += (coordinator_end - coordinator_start)

        if not decision:
            print(f'coordinator failed to generate decision')
            messages.append({"role": "assistant", "content": "LLM failed"})
            return messages, team_conversation

        else:
            if decision["agent_name"] == 'Engineer':
                # unlike explainer, engineer team has already updated the team_conversation and messages in fn below
                # syntax time, programming time, evaluation time are also updated in the fn below
                messages, team_conversation = engineer.generate_report_exp(args,
                                                                           messages, team_conversation, models_dict)

            elif decision["agent_name"] == "Explainer":
                explainer_start = time.time()
                explanation = explainer.generate_explanation_exp(args, messages, team_conversation)
                if args.explanation_stream:
                    with st.chat_message("assistant"):
                        explanation_response = st.write_stream(explanation)
                else:
                    explanation_response = explanation
                explainer_end = time.time()
                explainer.explanation_time += (explainer_end - explainer_start)

                team_conversation.append({"agent_name": "Explainer", "agent_response": explanation_response})
                messages.append({"role": "assistant", "content": explanation_response})
                return messages, team_conversation

            else:
                raise ValueError(
                    f"Decision {decision} has an invalid agent name. Please choose from Engineer or Explainer.")

        rounds += 1
