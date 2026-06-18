feasibility_restoration_fn_description = """
Use when: The model is infeasible and you need to find out the minimal change to specific [component name] for restoring feasibility.
Example: “How much should we adjust the [component name] to make the model feasible”
Example: "By how much would we need to raise the maximum storage capacity to make the model feasible"
Example: "I believe increasing the maximum pump discharge is practical, by how much do I need to change it in order to make the model feasible"
"""
components_retrival_fn_description = """
Use when: You need to know the current values or expressions of [component name] within the model.
Example: “What are the values of the [component name]”
Example: "What is the maximum water level the basin can hold" (retrieving a parameter, e.g. the maximum storage level)
Example: "How much water is pumped in total in the current optimal solution" (retrieving a variable, e.g. the pump discharge)
Example: "At which time steps does the pump run, and what is the discharge at each hour" (retrieving a time-indexed variable)
"""
sensitivity_analysis_fn_description = """
Use when: The model is feasible and you want to understand the impact of changing [component name] on the optimal objective value, **without specifying the extent of changes**.
Example: “How will the total pumped volume change with the change in the [component name]” (didn't specify how much the change is)
Example: "How sensitive is the total pumping to the maximum pump capacity" (didn't specify how much the change is)
Example: "Will the optimal value be greatly affected if we have more inflow from the hinterland" (didn't specify how much the change is)
"""
evaluate_modification_fn_description = """
Use when: The model is feasible and you want to understand the impact of changing [component name] on the optimal objective value, **by specifying the extent of changes**.
Example: “How will the total pumped volume change with **a 10% increase** in the [component name]” (specified the change is **a 10% increase**)
Example: "If the maximum storage level were **raised to 0.6 m**, how much less would the model pump" (specified the change is **raised to 0.6 m**)
Example: "Would the need to pump be removed if the initial storage level were **set to 0.3 m**" (specified the change is **set to 0.3 m**)
"""


def get_prompts(prompt):
    need2describe_prompt = """
Here are the name of {component_type} that need to be described
-----
{component_names}
-----


"""

    model_interpretation_json = {
        "components": {
            "sets": [
                {
                    "name": "The name of the component in sets",
                    "description": "The description of the component",
                }
            ],
            "parameters": [
                {
                    "name": "The name of the component in parameters",
                    "description": "The description of the component",
                }
            ],
            "variables": [
                {
                    "name": "The name of the component in variables",
                    "description": "The description of the component",
                }
            ],
            "constraints": [
                {
                    "name": "The name of the component in constraints",
                    "description": "The description of the component",
                }
            ],
            "objective": [
                {
                    "name": "The name of the component in objective",
                    "description": "The description of the component",
                }
            ]
        }
    }

    model_interpretation_prompt = """
You are an operations research expert specializing in water management optimization, and your role is to use PLAIN ENGLISH to interpret an optimization model written in Pyomo.
The Pyomo code is given below:

-----
{code}
-----


{cat_need2describe_prompt}
Your task is carefully inspect the code and write a description for each of the components.

Then, generate a json file accordingly with the following format (STICK TO THIS FORMAT!)

{model_interpretation_json}

- description should be either physical meanings, intended use, or any other relevant information about the component.
- Always include the physical unit in the description if it can be inferred from the code or doc string (e.g., "[m]", "[m³/s]", "[s]").
- If the model is time-indexed (e.g., uses a set T or RangeSet over time steps), describe each time-indexed component in terms of "at each time step" or "over the planning horizon".
- For Big-M constraints (constraints containing a large constant M multiplied by a binary variable), describe them as logical switching conditions in plain language (e.g., "Ensures the orifice can only discharge when the storage level exceeds the sea level"), not as mathematical formulas.
- For binary variables, describe the real-world decision they represent (e.g., "1 if gravity discharge is active at this time step, 0 otherwise") rather than calling them "binary integers".
- Note that I'm going to use python json.loads() function to parse the json file, so please make sure the format is correct (don't add ',' before enclosing '}}' or ']' characters.
- Generate the complete json file and don't omit anything.
- Use 'name' and 'description' as the keys, and provide the name and description of the component as the values.
- Use '```json' and '```' to enclose the json file.

Take a deep breath and solve the problem step by step.
"""

    model_illustration_prompt = """
You are an operations research expert and your role is to introduce an optimization model to non-experts, based on an abstract representation of the model in json format.
The json representation is given below:

-----
{json_representation}
-----

- Start with a brief introduction of the model, what the problem is about, who is using the model, and what the model is trying to achieve.
- If the model is time-indexed, explain the planning horizon in practical terms (e.g., "This model plans operations over 20 hourly time steps, covering roughly one tidal cycle").
- Explain what decisions (variables) are to be made. For binary variables, describe them as on/off or open/closed operational choices rather than mathematical integers.
- Explain what data or information (parameters) is already known, and include physical units where available.
- Explain what constraints are imposed on the decisions. Describe Big-M or logical constraints as physical operating rules (e.g., "gravity drainage can only occur when the storage level is above the sea level"), not as mathematical inequalities.
- Explain what the objective is, what is being optimized, and what it means in practice (e.g., minimizing total pumping volume reduces energy cost and wear on pumps).

The explanation must be coherent and easy to understand for water management operators and hydraulic engineers who are domain experts but not experts in optimization.
"""

    model_inference_prompt = """
You are an operations research expert specializing in water management optimization, and your role is to infer why an optimization model is infeasible, based on an abstract representation of the infeasible model in json format.
Particularly, your team has identified the Irreducible Infeasible Subset (IIS) of the model, which is given below:

-----
{iis_info}
-----


To understand what the parameters and the constraints mean, the json representation is given below for your reference:

-----
{json_representation}
-----


- Introduce to the user what constraints are potentially causing the infeasibility, and what parameters are involved in these constraints.
- If the model is time-indexed, identify WHICH specific time steps the conflicting constraints belong to — this helps operators understand whether the problem occurs during high tide, a peak inflow period, or a specific hour in the planning horizon.
- Explain the relationship between the constraints and the parameters, and infer why the constraints are conflicting with each other in physical terms (e.g., "the required outflow exceeds what the pump and orifice can deliver given the current sea level").
- Provide inference by analyzing their physical meanings, and AVOID using jargon and symbols as much as possible but the explanation style must be formal.
- Recommend some parameters that you believe can be adjusted to make the model feasible.
- Parameters recommended for adjustment MUST be physically changeable in practice. Use the following water management guidance:
  * CAN typically be adjusted: maximum pump discharge capacity, water level bounds (target levels, flood thresholds), inflow forecasts (subject to forecast uncertainty), planning horizon length.
  * CANNOT be adjusted: gravitational acceleration (g), storage area (A, which is fixed by geography), orifice geometry (width w, height d, discharge coefficient C — these are fixed civil structures), physical fluid properties.
  * MODELING ARTIFACT — do NOT recommend adjusting: Big-M constants (named M or similar large numerical values used in logical constraints) — these are mathematical artifacts, not physical parameters.
- Assess the practical implications of the recommendations in operational terms (e.g., "increasing the pump capacity means installing a larger pump, which requires capital investment and a longer lead time").
"""

    coordinator_prompt = """
You're a coordinator in a team of optimization experts. The goal of the team is to help non-experts analyze an 
optimization problem. Your task is to choose the next expert to work on the problem based on the current situation. 

Here's the list of agents in your team:
-----
{agents}
-----

Considering the conversation, generate a json file with the following format: 
{{ "agent_name": "Name of the agent you want to call next", "task": "The task you want the agent to carry out" }} 

to identify the next agent to work on the problem, and also the task it has to carry out. 
- Only generate the json file, and don't generate any other text.
- DO NOT change the keys of the json file, only change the values. Keys are "agent_name" and "task".
- if you think the problem is solved, generate the json file below:
{{ "agent_name": "Explainer", "task": "DONE" }} 
"""

    explainer_prompt = """
You're a water management expert (hydraulic engineer / polder management specialist) who helps your team answer user queries in MARKDOWN format.

- The users are water management operators or hydraulic engineers — domain experts who understand concepts like water levels, pump discharge, gravity flow, tidal cycles, and storage capacity, but who are not experts in optimization.
- Translate optimization results into operational terms: water levels in [m], pump discharge rates in [m³/s], pump on/off schedules, gravity-flow windows based on tidal conditions, etc.
- When describing time-indexed results, refer to specific time steps in practical terms (e.g., "at hour 5 of the planning horizon", "during the high-tide window between hours 8–12").
- Binary variables represent discrete operational decisions (e.g., pump on/off, gate open/closed, gravity flow active/inactive) — explain them as such, not as mathematical integers.
- Provide a detailed explanation only when you believe the users need more context about optimization to understand your explanation.
- Otherwise, the explanation must be succinct and concise, because users may be distracted by too much information.
- If Operators and Programmers in your team have provided technical feedback, then you need to summarize the feedback because the user cannot see them.
"""

    syntax_reminder_prompt = """
You're an operator working on a pyomo model.
Your task is to identify the following arguments: 
- the component names that the user is interested in,
- the most appropriate function that can answer the user's query, 
- the model that the user is querying.
then call the predefined syntax_guidance function to generate syntax guidance.

----- Instruction to select the most appropriate function -----
you MUST select a function from ```{function_names}```, DO NOT make up your own function.
1. feasibility_restoration:
Use when: The model is infeasible and you need to find out the minimal change to specific [component name] for restoring feasibility.
Example: “How much should we adjust the [component name] to make the model feasible”
Example: "By how much would we need to raise the maximum storage capacity to make the model feasible"
Example: "I believe increasing the maximum pump discharge is practical, by how much do I need to change it in order to make the model feasible"
[component name] category: parameters. If only constraint name is provided in the query, you need to infer the parameters involved in the constraint.

2. components_retrival:
Use when: You need to know the current values or expressions of [component name] within the model.
Example: “What are the values of the [component name]”
Example: "What is the maximum water level the basin can hold" (retrieving a parameter, e.g. the maximum storage level)
Example: "How much water is pumped in total in the current optimal solution" (retrieving a variable, e.g. the pump discharge)
Example: "At which time steps does the pump run, and what is the discharge at each hour" (retrieving a time-indexed variable)
[component name] category: sets, parameters, variables, constraints, or objectives.

3. sensitivity_analysis:
Use when: The model is feasible and you want to understand the impact of changing [component name] on the optimal objective value, **without specifying the extent of changes**.
Example: “How will the total pumped volume change with the change in the [component name]” (didn't specify how much the change is)
Example: "How sensitive is the total pumping to the maximum pump capacity" (didn't specify how much the change is)
Example: "Will the optimal value be greatly affected if we have more inflow from the hinterland" (didn't specify how much the change is)
[component name] category: parameters.

4. evaluate_modification:
Use when: The model is feasible and you want to understand the impact of changing [component name] on the optimal objective value, **by specifying the extent of changes**.
Example: “How will the total pumped volume change with **a 10% increase** in the [component name]” (specified the change is **a 10% increase**)
Example: "If the maximum storage level were **raised to 0.6 m**, how much less would the model pump" (specified the change is **raised to 0.6 m**)
Example: "Would the need to pump be removed if the initial storage level were **set to 0.3 m**" (specified the change is **set to 0.3 m**)
[component name] category: parameters or variables.

5. external_tools:
Use when: User doubts the model's optimal solution and provides a counterexample, or asks an open-ended what-if/why-not question that requires adding new constraints and re-solving (rather than just changing a parameter value).
Example: “Why is it not recommended to keep the storage level below 0.3 m in the optimal solution”
Example: "Is it possible to get a similar solution by pumping more evenly across the time steps instead of in concentrated pulses"
[component name] category: parameters or variables.
    
----- Instruction to determine the correct component name -----
The [component name] MUST be in a symbolic form, instead of its description.
Use the following dictionary to find the correct [component name] based on its description:
{component_name_meaning_pairs}

----- Instruction to find the queried model -----
In the form of 'model_integer', e.g. 'model_1'
"""

    operator_prompt = """
You're an optimization expert who helps your team to access and interact with optimization models by internal tools.

Your task is to invoke the most appropriate tool correctly based on the user's query and system reminders.
"""

#     code_reminder_prompt = """
# {source_code}
#
# # OPTICHAT REVISION CODE GOES HERE
#
# from pyomo.environ import SolverFactory, TerminationCondition
# solver = SolverFactory('gurobi')
# solver.options['TimeLimit'] = 300  # 5min time limit
# results = solver.solve(model, tee=False)
# print("Solver Status: ", results.solver.status)
# print("Termination Condition: ", results.solver.termination_condition)
# if results.solver.termination_condition == TerminationCondition.optimal:
#     from pyomo.environ import Objective
#     for obj_name, obj in model.component_map(Objective).items():
#         print('Optimal Objective Value: ', pyo.value(obj))
# else:
#     print("Model is infeasible or unbounded, no optimal objective value is available.")
#
# # OPTICHAT PRINT CODE GOES HERE
#
# """

#     programmer_prompt = """
# You're an optimization expert who helps your team to write pyomo code to answer users questions.
# (1) write code snippet to revise the model, only when the user doubts the model's optimal solution and provides a counterexample
# (2) write code snippet to print out the information useful for answering the user's question
#
# Output Format:
# ==========
# ```python
# CODE SNIPPET FOR REVISING THE MODEL
# ```
#
# ```python
# CODE SNIPPET FOR PRINTING OUT USEFUL INFORMATION
# ```
# ==========
#
# Here are some example questions and their answer codes:
# ----- EXAMPLE 1 -----
# Question: Why is it not recommended to use just one supplier for roastery 2?
#
# Answer Code:
# ```python
# # user is actually interested in the case that only one supplier can supply roastery 2 and does not believe the optimal solution.
# model.force_one_supplier = ConstraintList()
# model.force_one_supplier.add(sum(model.z[s,'roastery2'] for s in model.suppliers) <= 1)
# for s in model.suppliers:
#     model.force_one_supplier.add(model.x[s,'roastery2'] <= model.capacity_in_supplier[s] * model.z[s, 'roastery2'])
# ```
#
# ```python
# # I print out the new optimal objective value so that you can tell the user how the objective value changes if only one supplier supplies roastery 2.
# print('If forcing only one supplier to supply roastery 2, the optimal objective value will become: ', model.obj())
# ```
#
# ----- EXAMPLE 2 -----
# Question: Why is it not recommended to have production cost larger than transportation cost in the optimal setting?
#
# Answer Code:
# ```python
# # user does not believe the optimal solution obtained when production cost smaller than transportation cost.
# # so we force production cost to be less than transportation cost to see what will happen.
# model.counter_example = ConstraintList()
# model.counter_example.add(model.production <= model.transportation)
# ```
#
# ```python
# # I print out the new optimal objective value so that you can tell the user how the objective value changes.
# print('If forcing production cost be smaller than transportation cost, the optimal objective value will become: ', model.obj())
# ```
#
# - Code reminder has provided you with the source code of the pyomo model
# - Your written code will be added to the lines with substring: "# OPTICHAT *** CODE GOES HERE"
# So, you don't need to repeat the source code that has already been provided by Code reminder.
# - The code for re-solving the model has already been given,
# So you don't need to add it. Solving the model repeatedly can lead to errors.
# - Your written code should be accompanied by comments to explain the purpose of the code.
# - Evaluator will execute the new code for you and read the execution result.
# So, you MUST print out the model information that you believe is necessary for the user's question.
# """

    code_reminder_prompt = """{source_code}\n# YOUR CODE GOES HERE\n"""

    programmer_prompt = """
    You're an optimization expert who helps your team to write pyomo code to answer users questions, such as
    - write code snippet to revise the model, only when the user doubts the model's optimal solution and provides a counterexample
    - write code snippet to print out the information useful for answering the user's question

    Output Format:
    ==========
    ```python
    YOUR CODE SNIPPET
    ```
    ==========

    Here are some example questions and their answer codes:
    ----- EXAMPLE 1 -----
    Question: Why is it not recommended to use just one supplier for roastery 2?

    Answer Code:
```python
# user is actually interested in the case that only one supplier can supply roastery 2 and does not believe the optimal solution.
model.force_one_supplier = ConstraintList()
model.force_one_supplier.add(sum(model.z[s,'roastery2'] for s in model.suppliers) <= 1)
for s in model.suppliers:
    model.force_one_supplier.add(model.x[s,'roastery2'] <= model.capacity_in_supplier[s] * model.z[s, 'roastery2'])
    from pyomo.environ import SolverFactory, TerminationCondition
    
# standard code to solve the model. Don't change this code if you need to solve a mode.
solver = SolverFactory('gurobi')  # only gurobi is available in env
solver.options['TimeLimit'] = 300  # 5min time limit
results = solver.solve(model, tee=False)  # tee must be False to suppress solver output, otherwise the output is overwhelming
print("Solver Status: ", results.solver.status)
print("Termination Condition: ", results.solver.termination_condition)
# always check the termination condition and optimal objective value first
if results.solver.termination_condition == TerminationCondition.optimal:
    from pyomo.environ import Objective
    from pyomo.environ import value
    for obj_name, obj in model.component_map(Objective).items():
        print('Optimal Objective Value: ', value(obj))
else:
    print("Model is infeasible or unbounded, no optimal objective value is available.")
    
# I print out the new optimal objective value so that you can tell the user how the objective value changes if only one supplier supplies roastery 2.
print('If forcing only one supplier to supply roastery 2, the optimal objective value will become: ', model.obj())
```

    ----- EXAMPLE 2 -----
    Question: Why is it not recommended to have production cost larger than transportation cost in the optimal setting?

    Answer Code:
```python
# user does not believe the optimal solution obtained when production cost smaller than transportation cost.
# so we force production cost to be less than transportation cost to see what will happen.
model.counter_example = ConstraintList()
model.counter_example.add(model.production <= model.transportation)
    
# standard code to solve the model. Don't change this code if you need to solve a mode.
solver = SolverFactory('gurobi')  # only gurobi is available in env
solver.options['TimeLimit'] = 300  # 5min time limit
results = solver.solve(model, tee=False)  # tee must be False to suppress solver output, otherwise the output is overwhelming
print("Solver Status: ", results.solver.status)
print("Termination Condition: ", results.solver.termination_condition)
# always check the termination condition and optimal objective value first
if results.solver.termination_condition == TerminationCondition.optimal:
    from pyomo.environ import Objective
    from pyomo.environ import value
    for obj_name, obj in model.component_map(Objective).items():
        print('Optimal Objective Value: ', value(obj))
else:
    print("Model is infeasible or unbounded, no optimal objective value is available.")
    
# I print out the new optimal objective value so that you can tell the user how the objective value changes.
print('If forcing production cost be smaller than transportation cost, the optimal objective value will become: ', model.obj())
```
    
    - Code reminder has provided you with the source code of the pyomo model
    - Your written code will be added to the lines with substring: "# YOUR CODE GOES HERE"
    So, you don't need to repeat the source code that has already been provided by Code reminder.
    - The standard code for re-solving the model has been given in the examples, 
    So, you MUST use the standard code to re-solve the model to avoid undesired execution errors and long execution result.
    - Your written code should be accompanied by comments to explain the purpose of the code.
    - Evaluator will execute the new code for you and read the execution result.
    So, you MUST print out the model information that you believe is necessary for the user's question.
    """

    evaluator_prompt = """
You're an optimization expert who helps your team to review pyomo code,
based on the execution result of the code provided by the programmer.

Is the code bug-free and valid to answer the user's query?
Generate the following json file if you accept the code, and provide your own comment.
{{ "decision": "accept", "comment": "your own comment" }}
Generate the following json file if you reject the code, and provide your own comment.
{{ "decision": "reject", "comment": "your own comment" }}

- Only generate the json file, and don't generate any other text.
- Use 'decision' and 'comment' as the keys, 
- choose 'accept' or 'reject' for the decision, and provide your own comment. 
- Note that infeasibility caused by the new constraints may be acceptable. 
This is because programmers are trying to create a counterfactual example that the user is interested in, and this counterfactual example may be infeasible in nature.
"""

    test_prompt = """
You are a judge who determines if the LLM’s answer passes the test.
**Criteria**:
1. Is the code bug-free?
2. Is the execution result consistent with the human expert's answer, especially the specific values?
LLM may omit some values that human experts collected from other sources, 
but if the execution result covers the correct objective value, it should pass.

Human Expert Answer: 
{human_expert_answer}

- Return either "Pass" or "Fail."
- No additional comments or explanations.
"""

    if prompt == 'model_interpretation_prompt':
        return model_interpretation_prompt
    elif prompt == 'need2describe_prompt':
        return need2describe_prompt
    elif prompt == 'model_interpretation_json':
        return model_interpretation_json
    elif prompt == 'model_illustration_prompt':
        return model_illustration_prompt
    elif prompt == 'model_inference_prompt':
        return model_inference_prompt
    elif prompt == 'coordinator_prompt':
        return coordinator_prompt
    elif prompt == 'explainer_prompt':
        return explainer_prompt
    elif prompt == 'syntax_reminder_prompt':
        return syntax_reminder_prompt
    elif prompt == 'operator_prompt':
        return operator_prompt
    elif prompt == 'code_reminder_prompt':
        return code_reminder_prompt
    elif prompt == 'programmer_prompt':
        return programmer_prompt
    elif prompt == 'evaluator_prompt':
        return evaluator_prompt
    elif prompt == 'test_prompt':
        return test_prompt



def old_get_fn_json(fn_name):
    fn_json_template = \
        {
            "type": "function",
            "function": {
                "name": "",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queried_components": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "component_name": {"type": "string"},
                                    "component_indexes": {
                                        "oneOf": [
                                            {"type": "null"},
                                            {"type": "string"},
                                            {"type": "integer"},
                                            {
                                                "type": "array",
                                                "items": {
                                                    "oneOf": [
                                                        {"type": "string"},
                                                        {"type": "integer"},
                                                    ]
                                                }
                                            }
                                        ],
                                    },
                                },
                                "required": ["component_name", "component_indexes"]
                            },
                            "description": "List of dictionary of component_name and component_indexes that users are interested in."
                        },
                        "queried_model": {
                            "type": "string",
                            "description": "'model_int' e.g. 'model_1'"
                        },
                    },
                    "required": ["queried_components", "queried_model"]
                }
            }
        }
    fn_delta_json_template = \
        {
            "type": "function",
            "function": {
                "name": "",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queried_components": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "operation": {"type": "string",
                                                  "description": "modification operation e.g. * / + - = !"},
                                    "delta": {"type": "number", "description": "The extent of the modification"},
                                    "component_name": {"type": "string"},
                                    "component_indexes": {
                                        "oneOf": [
                                            {"type": "null"},
                                            {"type": "string"},
                                            {"type": "integer"},
                                            {
                                                "type": "array",
                                                "items": {
                                                    "oneOf": [
                                                        {"type": "string"},
                                                        {"type": "integer"},
                                                    ]
                                                }
                                            }
                                        ],
                                    },
                                },
                                "required": ["component_name", "component_indexes", "operation", "delta"]
                            },
                            "description": "List of dictionary of component_name, component_indexes, modification type and modification extent that users are interested in."
                        },
                        "queried_model": {
                            "type": "string",
                            "description": "'model_int' e.g. 'model_1'"
                        },
                    },
                    "required": ["queried_components", "queried_model"]
                }
            }
        }
    # fn_json_template = \
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "",
    #             "description": "",
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {
    #                     "queried_components": {
    #                         "type": "array",
    #                         "items": {
    #                             "type": "object",
    #                             "properties": {
    #                                 "component_name": {"type": "string"},
    #                                 "component_indexes": {
    #                                     "oneOf": [
    #                                         {"type": "null"},
    #                                         {"type": "string"},
    #                                         {"type": "integer"},
    #                                         {
    #                                             "type": "array",
    #                                             "items": {
    #                                                 "oneOf": [
    #                                                     {"type": "string"},
    #                                                     {"type": "integer"},
    #                                                     {"type": "null"},
    #                                                 ]
    #                                             }
    #                                         }
    #                                     ],
    #                                 },
    #                             },
    #                             "required": ["component_name", "component_indexes"]
    #                         },
    #                         "description": "List of dictionary of component_name and component_indexes that users are interested in."
    #                     },
    #                     "queried_model": {
    #                         "type": "string",
    #                         "description": "'model_int' e.g. 'model_1'"
    #                     },
    #                 },
    #                 "required": ["queried_components", "queried_model"]
    #             }
    #         }
    #     }
    # fn_delta_json_template = \
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "",
    #             "description": "",
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {
    #                     "queried_components": {
    #                         "type": "array",
    #                         "items": {
    #                             "type": "object",
    #                             "properties": {
    #                                 "operation": {"type": "string",
    #                                               "description": "modification operation e.g. * / + - = !"},
    #                                 "delta": {"type": "number", "description": "The extent of the modification"},
    #                                 "component_name": {"type": "string"},
    #                                 "component_indexes": {
    #                                     "oneOf": [
    #                                         {"type": "null"},
    #                                         {"type": "string"},
    #                                         {"type": "integer"},
    #                                         {
    #                                             "type": "array",
    #                                             "items": {
    #                                                 "oneOf": [
    #                                                     {"type": "string"},
    #                                                     {"type": "integer"},
    #                                                     {"type": "null"},
    #                                                 ]
    #                                             }
    #                                         }
    #                                     ],
    #                                 },
    #                             },
    #                             "required": ["component_name", "component_indexes", "operation", "delta"]
    #                         },
    #                         "description": "List of dictionary of component_name, component_indexes, modification type and modification extent that users are interested in."
    #                     },
    #                     "queried_model": {
    #                         "type": "string",
    #                         "description": "'model_int' e.g. 'model_1'"
    #                     },
    #                 },
    #                 "required": ["queried_components", "queried_model"]
    #             }
    #         }
    #     }

    fn_json_template["function"]["name"] = fn_name
    fn_delta_json_template["function"]["name"] = fn_name
    if fn_name == "feasibility_restoration":
        fn_json_template["function"]["description"] += feasibility_restoration_fn_description
    elif fn_name == "sensitivity_analysis":
        fn_json_template["function"]["description"] += sensitivity_analysis_fn_description
    elif fn_name == "components_retrival":
        fn_json_template["function"]["description"] += components_retrival_fn_description
    elif fn_name == "evaluate_modification":
        fn_delta_json_template["function"]["description"] += evaluate_modification_fn_description
        return fn_delta_json_template
    return fn_json_template


def get_fn_json(fn_name, mode):
    if mode == 'multiple':
        fn_json_template = \
            {
                "type": "function",
                "function": {
                    "name": "",
                    "description": "",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queried_components": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "component_name": {"type": "string"},
                                        "component_indexes": {
                                            "type": "array",
                                            "items": {
                                                "oneOf": [
                                                    {"type": "string"},
                                                    {"type": "integer"},
                                                ]
                                            },
                                        },
                                    },
                                    "required": ["component_name", "component_indexes"]
                                },
                                "description": "List of dictionary of component_name and component_indexes that users are interested in."
                            },
                            "queried_model": {
                                "type": "string",
                                "description": "'model_int' e.g. 'model_1'"
                            },
                        },
                        "required": ["queried_components", "queried_model"]
                    }
                }
            }
        fn_delta_json_template = \
            {
                "type": "function",
                "function": {
                    "name": "",
                    "description": "",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queried_components": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "operation": {"type": "string",
                                                      "description": "modification operation e.g. * / + - = !"},
                                        "delta": {"type": "number", "description": "The extent of the modification"},
                                        "component_name": {"type": "string"},
                                        "component_indexes": {
                                            "type": "array",
                                            "items": {
                                                "oneOf": [
                                                    {"type": "string"},
                                                    {"type": "integer"},
                                                ]
                                            },
                                        },
                                    },
                                    "required": ["component_name", "component_indexes", "operation", "delta"]
                                },
                                "description": "List of dictionary of component_name, component_indexes, modification type and modification extent that users are interested in."
                            },
                            "queried_model": {
                                "type": "string",
                                "description": "'model_int' e.g. 'model_1'"
                            },
                        },
                        "required": ["queried_components", "queried_model"]
                    }
                }
            }
    elif mode == 'single':
        fn_json_template = \
            {
                "type": "function",
                "function": {
                    "name": "",
                    "description": "",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queried_components": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "component_name": {"type": "string"},
                                        "component_indexes": {
                                            "oneOf": [
                                                {"type": "string"},
                                                {"type": "integer"},
                                            ],
                                        },
                                    },
                                    "required": ["component_name", "component_indexes"]
                                },
                                "description": "List of dictionary of component_name and component_indexes that users are interested in."
                            },
                            "queried_model": {
                                "type": "string",
                                "description": "'model_int' e.g. 'model_1'"
                            },
                        },
                        "required": ["queried_components", "queried_model"]
                    }
                }
            }
        fn_delta_json_template = \
            {
                "type": "function",
                "function": {
                    "name": "",
                    "description": "",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queried_components": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "operation": {"type": "string",
                                                      "description": "modification operation e.g. * / + - = !"},
                                        "delta": {"type": "number", "description": "The extent of the modification"},
                                        "component_name": {"type": "string"},
                                        "component_indexes": {
                                            "oneOf": [
                                                {"type": "string"},
                                                {"type": "integer"},
                                            ],
                                        },
                                    },
                                    "required": ["component_name", "component_indexes", "operation", "delta"]
                                },
                                "description": "List of dictionary of component_name, component_indexes, modification type and modification extent that users are interested in."
                            },
                            "queried_model": {
                                "type": "string",
                                "description": "'model_int' e.g. 'model_1'"
                            },
                        },
                        "required": ["queried_components", "queried_model"]
                    }
                }
            }
    elif mode == 'none':
        fn_json_template = \
            {
                "type": "function",
                "function": {
                    "name": "",
                    "description": "",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queried_components": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "component_name": {"type": "string"},
                                        "component_indexes": {"type": "null"},
                                    },
                                    "required": ["component_name", "component_indexes"]
                                },
                                "description": "List of dictionary of component_name and component_indexes that users are interested in."
                            },
                            "queried_model": {
                                "type": "string",
                                "description": "'model_int' e.g. 'model_1'"
                            },
                        },
                        "required": ["queried_components", "queried_model"]
                    }
                }
            }
        fn_delta_json_template = \
            {
                "type": "function",
                "function": {
                    "name": "",
                    "description": "",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queried_components": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "operation": {"type": "string",
                                                      "description": "modification operation e.g. * / + - = !"},
                                        "delta": {"type": "number", "description": "The extent of the modification"},
                                        "component_name": {"type": "string"},
                                        "component_indexes": {"type": "null"},
                                    },
                                    "required": ["component_name", "component_indexes", "operation", "delta"]
                                },
                                "description": "List of dictionary of component_name, component_indexes, modification type and modification extent that users are interested in."
                            },
                            "queried_model": {
                                "type": "string",
                                "description": "'model_int' e.g. 'model_1'"
                            },
                        },
                        "required": ["queried_components", "queried_model"]
                    }
                }
            }
    elif mode == 'all':
        fn_json_template = \
            {
                "type": "function",
                "function": {
                    "name": "",
                    "description": "",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queried_components": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "component_name": {"type": "string"},
                                        "component_indexes": {
                                            "oneOf": [
                                                {"type": "null"},
                                                {"type": "string"},
                                                {"type": "integer"},
                                                {
                                                    "type": "array",
                                                    "items": {
                                                        "oneOf": [
                                                            {"type": "string"},
                                                            {"type": "integer"},
                                                        ]
                                                    }
                                                }
                                            ],
                                        },
                                    },
                                    "required": ["component_name", "component_indexes"]
                                },
                                "description": "List of dictionary of component_name and component_indexes that users are interested in."
                            },
                            "queried_model": {
                                "type": "string",
                                "description": "'model_int' e.g. 'model_1'"
                            },
                        },
                        "required": ["queried_components", "queried_model"]
                    }
                }
            }
        fn_delta_json_template = \
            {
                "type": "function",
                "function": {
                    "name": "",
                    "description": "",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queried_components": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "operation": {"type": "string",
                                                      "description": "modification operation e.g. * / + - = !"},
                                        "delta": {"type": "number", "description": "The extent of the modification"},
                                        "component_name": {"type": "string"},
                                        "component_indexes": {
                                            "oneOf": [
                                                {"type": "null"},
                                                {"type": "string"},
                                                {"type": "integer"},
                                                {
                                                    "type": "array",
                                                    "items": {
                                                        "oneOf": [
                                                            {"type": "string"},
                                                            {"type": "integer"},
                                                        ]
                                                    }
                                                }
                                            ],
                                        },
                                    },
                                    "required": ["component_name", "component_indexes", "operation", "delta"]
                                },
                                "description": "List of dictionary of component_name, component_indexes, modification type and modification extent that users are interested in."
                            },
                            "queried_model": {
                                "type": "string",
                                "description": "'model_int' e.g. 'model_1'"
                            },
                        },
                        "required": ["queried_components", "queried_model"]
                    }
                }
            }
    else:
        raise ValueError("Invalid mode: {}".format(mode))

    # fn_json_template = \
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "",
    #             "description": "",
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {
    #                     "queried_components": {
    #                         "type": "array",
    #                         "items": {
    #                             "type": "object",
    #                             "properties": {
    #                                 "component_name": {"type": "string"},
    #                                 "component_indexes": {
    #                                     "oneOf": [
    #                                         {"type": "null"},
    #                                         {"type": "string"},
    #                                         {"type": "integer"},
    #                                         {
    #                                             "type": "array",
    #                                             "items": {
    #                                                 "oneOf": [
    #                                                     {"type": "string"},
    #                                                     {"type": "integer"},
    #                                                 ]
    #                                             }
    #                                         }
    #                                     ],
    #                                 },
    #                             },
    #                             "required": ["component_name", "component_indexes"]
    #                         },
    #                         "description": "List of dictionary of component_name and component_indexes that users are interested in."
    #                     },
    #                     "queried_model": {
    #                         "type": "string",
    #                         "description": "'model_int' e.g. 'model_1'"
    #                     },
    #                 },
    #                 "required": ["queried_components", "queried_model"]
    #             }
    #         }
    #     }
    # fn_delta_json_template = \
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "",
    #             "description": "",
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {
    #                     "queried_components": {
    #                         "type": "array",
    #                         "items": {
    #                             "type": "object",
    #                             "properties": {
    #                                 "operation": {"type": "string",
    #                                               "description": "modification operation e.g. * / + - = !"},
    #                                 "delta": {"type": "number", "description": "The extent of the modification"},
    #                                 "component_name": {"type": "string"},
    #                                 "component_indexes": {
    #                                     "oneOf": [
    #                                         {"type": "null"},
    #                                         {"type": "string"},
    #                                         {"type": "integer"},
    #                                         {
    #                                             "type": "array",
    #                                             "items": {
    #                                                 "oneOf": [
    #                                                     {"type": "string"},
    #                                                     {"type": "integer"},
    #                                                 ]
    #                                             }
    #                                         }
    #                                     ],
    #                                 },
    #                             },
    #                             "required": ["component_name", "component_indexes", "operation", "delta"]
    #                         },
    #                         "description": "List of dictionary of component_name, component_indexes, modification type and modification extent that users are interested in."
    #                     },
    #                     "queried_model": {
    #                         "type": "string",
    #                         "description": "'model_int' e.g. 'model_1'"
    #                     },
    #                 },
    #                 "required": ["queried_components", "queried_model"]
    #             }
    #         }
    #     }
    fn_json_template["function"]["name"] = fn_name
    fn_delta_json_template["function"]["name"] = fn_name
    if fn_name == "feasibility_restoration":
        fn_json_template["function"]["description"] += feasibility_restoration_fn_description
    elif fn_name == "sensitivity_analysis":
        fn_json_template["function"]["description"] += sensitivity_analysis_fn_description
    elif fn_name == "components_retrival":
        fn_json_template["function"]["description"] += components_retrival_fn_description
    elif fn_name == "evaluate_modification":
        fn_delta_json_template["function"]["description"] += evaluate_modification_fn_description
        return fn_delta_json_template
    return fn_json_template


def get_syntax_guidance_fn_json():
    fn_json_template = \
        {
            "type": "function",
            "function": {
                "name": "syntax_guidance",
                "description": "generate syntax reminder, based on the most appropriate function that can answer the user's query, the component names that the user is interested in, and the model that the user is querying.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queried_function": {"type": "string",
                                             "description": "The name of the function that the user is querying."},
                        "queried_components": {"type": "array",
                                               "items": {"type": "string"},
                                               "description": "List of component names that users are interested in."},
                        "queried_model": {"type": "string",
                                          "description": "'model_integer' e.g. 'model_1'"},
                    },
                    "required": ["queried_function", "queried_components", "queried_model"]
                }
            }
        }

    return fn_json_template


def get_tools(fn_names):
    multiple_tools = []
    single_tools = []
    none_tools = []
    all_tools = []
    for fn_name in fn_names:
        if fn_name != 'external_tools':
            multiple_tools.append(get_fn_json(fn_name, 'multiple'))
            single_tools.append(get_fn_json(fn_name, 'single'))
            none_tools.append(get_fn_json(fn_name, 'none'))
            all_tools.append(get_fn_json(fn_name, 'all'))
    return multiple_tools, single_tools, none_tools, all_tools, 'auto'


def get_syntax_guidance_tool():
    syntax_guidance_fn_json = get_syntax_guidance_fn_json()
    syntax_guidance_tool = [syntax_guidance_fn_json]
    return syntax_guidance_tool

