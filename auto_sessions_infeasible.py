"""
Automated multi-session interrogation of the OptiChat bot.

For each of NUM_SESSIONS conversations this script:
  1. Loads and processes the model (Feas/mixed_integer_rtc.py): solves it and
     generates the natural-language model description, exactly like the
     "Process" button in app.py.
  2. Asks the first 5 questions in one go (a single combined user turn).
  3. Asks the last 4 questions sequentially (one user turn each), so every
     follow-up sees the answers that came before it.
  4. Writes that conversation's chat_history to the "Allesbehalve code" folder.

Run it from the OptiChat directory:
cd /Users/martijnkrikke/Documents/Scriptie/OptiChat
python auto_sessions.py

It reuses the same building blocks as app.py / run_exp.py (no Streamlit needed,
because all the *_stream flags are False).
"""

import os
import datetime

from extractor import initial_loading, update_model_representation
from utils import get_agents, OptiChat_workflow_exp
from pyomo.opt import TerminationCondition
from anthropic import Anthropic
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Path is relative to the OptiChat directory because initial_loading() imports
# the model as a module (e.g. "Feas.mixed_integer_rtc"). Run this script from
# inside OptiChat.
OPTICHAT_MODEL = "Baseline"

MODEL_PATH = "Feas/mixed_integer_rtc.py"

OUTPUT_DIR = f"/Users/martijnkrikke/Documents/Scriptie/chats/{OPTICHAT_MODEL}"

NUM_SESSIONS = 5

CLAUDE_MODEL = "claude-haiku-4-5"

TEMPERATURE = 0.1


# Questions 1-5 are asked together in a single batch turn.
BATCH_QUESTIONS = [
    "What is the maximum water level the basin can hold?",
    "Why is the orifice used in preference to the pump when both are physically possible?",
    "How much water is pumped in total in the optimal solution?",
    "List the hours when the pump runs and the flow at each hour.",
    "Why is there no pumping at the start of the horizon?",
]

# Questions 6-9 are asked one at a time, building on the conversation so far.
SEQUENTIAL_QUESTIONS = [
    "Would starting at a lower initial level have removed the need to pump?",
    "If the basin limit were raised to 0.6 m, how much less would the model pump?",
    "Is it possible to get a similar solution, by pumping less at hours where I "
    "now pump a lot, and pumping more at hours where I now pump little?",
    "If I were to constrain the pump to never exceed 4 m³/s in any single hour "
    "(reducing the peak pulses), how much would the total pumped volume increase?",
]


class Args:
    """Lightweight stand-in for st.session_state, holding only the attributes the
    agents and the workflow actually read."""

    def __init__(self, claude_model, temperature):
        self.claude_model = claude_model
        self.temperature = temperature
        self.json_mode = True
        # Headless: never stream into Streamlit widgets.
        self.illustration_stream = False
        self.inference_stream = False
        self.explanation_stream = False
        # Behave like the interactive app (full coordinator/engineer/explainer
        # workflow with real tool use), not like the benchmark experiments.
        self.interpreter_experiment = False
        self.internal_experiment = False
        self.external_experiment = False
        self.fn_names = ["feasibility_restoration", "sensitivity_analysis",
                         "components_retrival", "evaluate_modification", "external_tools"]


def process_model(args, interpreter):
    """Replicates app.py's process(): solve the model and build its description.

    Returns (models_dict, initial_messages, initial_history) where
    initial_messages seeds the conversation and initial_history seeds the
    exportable chat history.
    """
    models_dict, code = initial_loading(MODEL_PATH, is_uploaded=False)

    # interpret the model components
    models_dict, cnt, completion = interpreter.generate_interpretation_exp(args, models_dict, code)
    update_model_representation(models_dict)

    # illustrate the model (returns a plain string because illustration_stream=False)
    illustration = interpreter.generate_illustration_exp(args, models_dict["model_representation"])
    models_dict["model_1"]["model description"] = illustration
    update_model_representation(models_dict)

    # if the model is infeasible, also generate the inference (it is feasible here,
    # but we mirror app.py so the script works for any model)
    if models_dict["model_1"]["model status"] in [TerminationCondition.infeasible,
                                                   TerminationCondition.infeasibleOrUnbounded]:
        inference = interpreter.generate_inference_exp(args, models_dict["model_representation"])
        models_dict["model_1"]["model description"] = illustration + "\n" + inference
        update_model_representation(models_dict)

    model_description = models_dict["model_representation"]["model description"]

    initial_messages = [
        {"role": "user", "content": "I have uploaded a Pyomo model."},
        {"role": "assistant", "content": model_description},
    ]
    initial_history = [
        "user: I have uploaded a Pyomo model.",
        "assistant: " + model_description,
    ]
    return models_dict, initial_messages, initial_history


def ask(args, coordinator, engineer, explainer, messages, models_dict, prompt,
        chat_history, detailed_chat_history):
    """Send one user turn through the OptiChat workflow, mutating messages,
    chat_history, and detailed_chat_history in place. Returns the assistant's reply."""
    messages.append({"role": "user", "content": prompt})
    updated_messages, team_conversation = OptiChat_workflow_exp(
        args, coordinator, engineer, explainer, messages, models_dict)

    answer = updated_messages[-1]["content"]
    messages[:] = updated_messages

    chat_history.append("user: " + prompt)
    chat_history.append("assistant: " + answer)

    detailed_chat_history.append("user: " + prompt)
    for message in team_conversation:
        detailed_chat_history.append(f"***{message['agent_name']}***: {message['agent_response']}")
    detailed_chat_history.append("assistant: " + answer)

    return answer


def run_session(session_idx, args, agents):
    interpreter, explainer, engineer, coordinator = agents

    print("=" * 60)
    print(f"SESSION {session_idx}/{NUM_SESSIONS}: processing model {MODEL_PATH}")
    models_dict, messages, chat_history = process_model(args, interpreter)
    detailed_chat_history = list(chat_history)  # seed with the model-upload turn

    # ---- Questions 1-5: one combined batch turn ----
    batch_prompt = "\n".join(f"{i}. {q}" for i, q in enumerate(BATCH_QUESTIONS, start=1))
    print(f"[session {session_idx}] asking questions 1-5 (batch)")
    ask(args, coordinator, engineer, explainer, messages, models_dict,
        batch_prompt, chat_history, detailed_chat_history)

    # ---- Questions 6-9: asked sequentially ----
    for offset, q in enumerate(SEQUENTIAL_QUESTIONS):
        qnum = len(BATCH_QUESTIONS) + offset + 1
        print(f"[session {session_idx}] asking question {qnum} (sequential)")
        ask(args, coordinator, engineer, explainer, messages, models_dict,
            q, chat_history, detailed_chat_history)

    return chat_history, detailed_chat_history


def save_chat_history(session_idx, chat_history, detailed_chat_history, run_stamp):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = f"{run_stamp}_session_{session_idx}_{OPTICHAT_MODEL}"

    path = os.path.join(OUTPUT_DIR, f"chat_history_{base}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(chat_history))
    print(f"[session {session_idx}] chat history saved to {path}")

    detailed_path = os.path.join(OUTPUT_DIR, f"detailed_chat_history_{base}.md")
    with open(detailed_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(detailed_chat_history))
    print(f"[session {session_idx}] detailed chat history saved to {detailed_path}")


def main():
    run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    client = Anthropic(api_key=os.environ["CLAUDE_API_KEY"])
    args = Args(CLAUDE_MODEL, TEMPERATURE)
    agents = get_agents(args.fn_names, client, args.claude_model)

    for session_idx in range(1, NUM_SESSIONS + 1):
        try:
            chat_history, detailed_chat_history = run_session(session_idx, args, agents)
            save_chat_history(session_idx, chat_history, detailed_chat_history, run_stamp)
        except Exception as e:
            print(f"[session {session_idx}] ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print("All sessions complete.")


if __name__ == "__main__":
    main()
