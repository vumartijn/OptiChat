from typing import Dict, Optional, Union, List
import random
import copy

import numpy as np
import pyomo.environ as pe
from pyomo.core.expr.visitor import identify_mutable_parameters, identify_variables, replace_expressions, clone_expression
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.opt import SolverFactory, TerminationCondition, SolverStatus

from extractor import pyomo2json
# from get_code_from_markdown import *
from contextlib import redirect_stdout
import sys
import time

def fnArgsDecoder(queried_components):
    for queried_component in queried_components:
        for key, value in queried_component.items():
            if isinstance(value, str):
                if value.lower() in ["none", "null"]:
                    queried_component[key] = None
                elif value.lower() in ["__all__"]:
                    queried_component[key] = slice(None)

            elif isinstance(value, tuple):
                value = list(value)
                for i, value_i in enumerate(value):
                    if value_i in ["none", "null", "None", "Null"]:
                        value[i] = None
                    elif value_i in ["__all__"]:
                        value[i] = slice(None)
                queried_component[key] = tuple(value)

            elif isinstance(value, list):
                for i, value_i in enumerate(value):
                    if value_i in ["none", "null", "None", "Null"]:
                        value[i] = None
                    elif value_i in ["__all__"]:
                        value[i] = slice(None)
                    else:
                        value[i] = value_i
                queried_component[key] = tuple(value)
    return queried_components


def old_fnArgsDecoder(queried_components):
    for queried_component in queried_components:
        for key, value in queried_component.items():
            if isinstance(value, str):
                if value.lower() in ["none", "null", "slice(none)", "slice(null)", "slice('none')"]:
                    queried_component[key] = None if "slice" not in value else slice(None)
                elif "slice(None)" in value and value != "slice(None)":
                    queried_component[key] = eval(value)
                # if value in ["None", "null"]:
                #     queried_component[key] = None
                # elif value in ["slice(None)", "slice(null)", "slice('None')"]:
                #     queried_component[key] = slice(None)
                # elif value != 'slice(None)' and 'slice(None)' in value:
                #     # in case that llm should have returned a tuple ('slice(None)', 'slice(None)', "specific_index")
                #     # but returned a string "('slice(None)', 'slice(None)', "specific_index")"
                #     queried_component[key] = eval(value)

            elif isinstance(value, tuple):
                value = list(value)
                for i, value_i in enumerate(value):
                    if value_i in ["None", "null"]:
                        value[i] = None
                    elif value_i in ["slice(None)", "slice(null)", "slice('None')"]:
                        value[i] = slice(None)
                queried_component[key] = tuple(value)

            elif isinstance(value, list):
                for i, value_i in enumerate(value):
                    if value_i in ["None", "null"]:
                        value[i] = None
                    elif value_i in ["slice(None)", "slice(null)", "slice('None')"]:
                        value[i] = slice(None)
                    else:
                        value[i] = value_i
                queried_component[key] = tuple(value)
    return queried_components


def get_component_type(name, m):
    TYPES = ['parameters', 'variables', 'sets', 'constraints', 'objective']
    return next((c_type for c_type in TYPES if name in m["components"][c_type]), None)


def get_new_model_name(queried_model):
    prefix, number = queried_model.rsplit('_', 1)
    incremented_number = int(number) + 1
    new_model_name = f"{prefix}_{incremented_number}"
    return new_model_name


def syntax_guidance(queried_function: str,
                    queried_components: List[str],
                    queried_model: str,
                    models_dict):

    FUNCTIONS = ['feasibility_restoration', 'sensitivity_analysis', 'components_retrival', 'evaluate_modification',
                 'alternative_solutions', 'scenario_risk_assessment', 'stochastic_hedging_analysis',
                 'external_tools']
    assert queried_function in FUNCTIONS, f"Function {queried_function} is not recognized."
    if queried_function == 'external_tools':
        return "external_tools", "none"

    model_dict = models_dict[queried_model]
    function_syntax = "function to call: " + queried_function + "\n\n"  #
    queried_model_syntax = "queried_model: " + queried_model + "\n\n"  #

    def get_index_guidance(pattern):
        """
        provide index guidance in terms of an indexed pattern,
        supplementary is 'evaluate_modification' or None
        """

        if isinstance(pattern, tuple):
            mode = 'multiple'
            tuple_guidance = f"""
Return a tuple with dimensions to be {len(pattern)}.
You need to fill in the tuple with the specific indexes provided by the user, 
and the rest of the indexes that are not specified should be "__all__" in string.

- Must return a tuple with {len(pattern)} elements
- "__all__" is placeholder that represents all indexes in a dimension that user didn't specify
- "__all__" can be inserted into the tuple multiple times if there are multiple dimensions that user didn't specify
Example: If the dimension of an indexed component is 3, 
the first index is specified as 4, the second and third indexes are not specified by users,
then return the tuple (4, "__all__", "__all__")

- Must be careful with the order of indexes in the tuple by inspecting code
Example: If the dimension of an indexed component is 2,
the first dimension represents time, the second dimension represents location,
the user specifies the location to be "NY", the time is not specified by users,
then return the tuple ("__all__", "NY")

- Must be careful with the data type of each index in the tuple by inspecting code
Example: If the dimension of an indexed component is 2,
the first dimension represents time, the second dimension represents location,
the user specifies the time to be 9, the location to be "NY"
then return the tuple (9, "NY") instead of ('9', "NY")

**Note:** indexes are not specified unless the **exact** values are provided and the total number of values is less than the size of the dimension.
Descriptions like "for all the indexes", "from one to ten (but the size of dimension is 10)" are considered as "Not specified",
which MUST use "__all__" instead of enumerating all indexes.
"""
            return tuple_guidance, mode

        elif isinstance(pattern, int) or isinstance(pattern, str):
            mode = 'single'
            primitive_guidance = f"""
When specific index provided, fill in the type of {type(pattern)}
If no specific index provided, return "__all__" in string

- "__all__" is placeholder that represents all indexes that user didn't specify

**Note:** indexes are not specified unless the **exact** values are provided and the total number of values is less than the size of the dimension.
Descriptions like "for all the indexes", "from one to ten (but the size of dimension is 10)" are considered as "Not specified",
which MUST use "__all__" instead of enumerating all indexes.
"""
            return primitive_guidance, mode

    def get_complex_guidance(flag):
        example = [{'component_name': 'dem', 'component_indexes': 'a'},
                   {'component_name': 'dem', 'component_indexes': 'c'}]
        cs = f"""
If the user provides multiple indexes that belong to the same dimension, 
add every specified index separately in the queried_components.
Example: help me change demand of a and c.

If the user provides indexes that are prohibited from being changed,
add every permitted index separately in the queried_components.
Example: help me change demand, but please note that the demand of b cannot be changed.

dem is a 1-dim parameter that means demand, and dem is indexed by a, b, c, then, 
queried_components: {example}"""
        if flag:
            return cs
        else:
            return ""

    def get_supplementary_guidance(fn):
        if fn == 'evaluate_modification':
            supplementary_guidance = f"""
When no specific modification extent provided, always return operation: "!" and delta: 0

Otherwise, choose one of the following operations: "+", "-", "*", "/", and fill in the delta value.
Demonstrations:
change it to 5: operation: "=", delta: 13;
increase it to 5: operation: "=", delta: 5;
increase it by 5: operation: "+", delta: 5;
increase it by 5%: operation: "*", delta: 1.05;
decrease it to 5: operation: "=", delta: 5;
decrease it by 5: operation: "-", delta: 5;
decrease it by 5%: operation: "*", delta: 0.95;
discount it by 5%: operation: "*", delta: 0.95;
have 5 more units: operation: "+", delta: 5;
have 5 less units: operation: "-", delta: 5;

Make sure the delta value is consistent with the positivity/negativity of the parameters being modified."""
            return supplementary_guidance
        else:
            return ""

    need_complex_syntax = False
    ref = []
    syntax_mode = []
    for component_name in queried_components:
        component_type = get_component_type(component_name, model_dict)
        if model_dict["components"][component_type][component_name]["is_indexed"]:
            # component is indexed
            need_complex_syntax = True
            component_index_set = model_dict["components"][component_type][component_name]["index_set"]
            component_pattern = random.choice(list(component_index_set))
            situation, mode_i = get_index_guidance(component_pattern)
        else:
            # component is not indexed
            situation = "always return null"
            mode_i = 'none'
        ref.append({"component_name": component_name, "component_indexes": situation})
        syntax_mode.append(mode_i)
    queried_component_syntax = f"queried_components: {ref} \n\n"  #
    complex_syntax = get_complex_guidance(need_complex_syntax)  #
    supplementary = get_supplementary_guidance(queried_function)  #
    syntax_output = function_syntax + queried_model_syntax + queried_component_syntax + complex_syntax + supplementary

    syntax_mode = set(syntax_mode)
    if len(syntax_mode) == 0:
        # No specific components queried (e.g. "why is this optimal") — nothing to
        # index. alternative_solutions can run with an empty queried_components list.
        syntax_mode = "none"
    elif len(syntax_mode) > 1:
        syntax_mode = "all"
    else:
        syntax_mode = next(iter(syntax_mode))
    return syntax_output, syntax_mode


def feasibility_restoration(queried_components: List[Dict], queried_model: str, models_dict):
    queried_model_dict = models_dict[queried_model]

    if queried_model_dict['model status'] not in [TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded]:
        return "The model is not infeasible. No need to restore feasibility. Please confirm with the user."

    model = queried_model_dict['model class'].clone()
    # define slack parameters
    for component in queried_components:
        param_name = component['component_name']
        param_indexes = component['component_indexes']
        print(f'param_indexes: {param_indexes}')

        component_type = get_component_type(param_name, queried_model_dict)
        if component_type == 'parameters':
            if isinstance(param_indexes, tuple):
                eval_param = eval(f"model.{param_name}")
                if len(eval_param[param_indexes].index()) <= 0:
                    raise IndexError(f"Error: Indexes are not valid. This usually happens when the order of indexes in the tuple is incorrect.")

            if queried_model_dict['components']['parameters'][param_name]['is_RHS']:
                # First, add slacks to all indexes and fix all of them as 0
                exec(
                    "model.slack_pos_" + param_name + "=pe.Var(model." + param_name + ".index_set(), within=pe.NonNegativeReals)")
                exec(
                    "model.slack_neg_" + param_name + "=pe.Var(model." + param_name + ".index_set(), within=pe.NonNegativeReals)")
                model_slack_pos_param = eval("model.slack_pos_" + param_name)
                model_slack_neg_param = eval("model.slack_neg_" + param_name)
                model_slack_pos_param.fix(0)
                model_slack_neg_param.fix(0)
                # Second, unfix the slacks for the specific indexes provided in the query
                model_slack_pos_param[param_indexes].unfix()
                model_slack_neg_param[param_indexes].unfix()

            else:
                feedback = f"""
Feedback from internal tools:
Warning. {param_name} is not a RHS parameter in the model.
This parameter is LHS parameter. 
Changing LHS parameter for feasibility restoration without specifying modification extent 
can extend solving time and risk terminating the optimization process prematurely before finding an optimal solution. 
Users need to try other parameters for feasibility restoration, 
or specify a modification extent (e.g., a 5% increase) to directly assess the impact of this modification, if they are particularly interested in this parameter."""
                return feedback
        else:
            wrong_component_type = component_type
            feedback = f"""
Feedback from internal tools:
Error. {param_name} is not a parameter in the model but a {wrong_component_type}.
Users need to provide a valid parameter for feasibility restoration."""
            return feedback

    # generate replacements
    iis_param = []
    replacements_list = []
    for component in queried_components:
        param_name = component['component_name']
        param_indexes = component['component_indexes']
        for idx in eval("model." + param_name + ".index_set()"):
            model_param = eval("model." + param_name)
            iis_param.append((param_name, idx))  ###
            expr_param = model_param[idx]
            slack_var_pos = eval("model.slack_pos_" + param_name)[idx]
            slack_var_neg = eval("model.slack_neg_" + param_name)[idx]
            replacements = {id(expr_param): expr_param + slack_var_pos - slack_var_neg}
            replacements_list.append(replacements)
    # replace constraints
    original_consts = []
    for consts_name, consts in model.component_map(pe.Constraint).items():
        original_consts.append(consts)
    model.slack_iis_constraints = pe.ConstraintList()
    for consts in original_consts:
        for const_idx in consts.index_set():
            try:
                const = consts[const_idx]
                new_expr = clone_expression(const.expr)
                for replacements in replacements_list:
                    new_expr = replace_expressions(new_expr, replacements)
                model.slack_iis_constraints.add(new_expr)
                const.deactivate()
            except Exception as e:
                print(f"Skip the skipped constraint")

    # replace objective
    objectives = model.component_objects(pe.Objective, active=True)
    for obj in objectives:
        obj.deactivate()
    # minimize the 1-norm of the slacks that are added
    new_obj = 0
    for p, idx in iis_param:
        slack_var_pos = eval("model.slack_pos_" + p)[idx]
        slack_var_neg = eval("model.slack_neg_" + p)[idx]
        new_obj += slack_var_pos + slack_var_neg
    model.slack_obj = pe.Objective(expr=new_obj, sense=pe.minimize)
    # solve the model
    opt = SolverFactory('gurobi')
    opt.options['nonConvex'] = 2
    opt.options['TimeLimit'] = 300  # 5min time limit
    results = opt.solve(model, tee=True)
    # construct technical feedback
    termination_condition = results.solver.termination_condition
    #new_model_dict = pyomo2json(model, termination_condition=termination_condition)
    new_model_dict = copy.deepcopy(queried_model_dict)
    feedback = f"The following changes are made to {queried_model}: \n"
    description = f"a model with the following changes to {queried_model}: \n"
    new_model_name = get_new_model_name(queried_model)

    if termination_condition == TerminationCondition.maxTimeLimit:
        for p, idx in iis_param:
            feedback = feedback + f"attempt to change {p} at {idx}; \n"
            description = description + f"attempt to change {p}{idx}; \n"
        feedback = feedback + f"\n\nThe model cannot be solved due to time limit."
        description = description[:7] + f", which cannot be solved due to time limit." + description[7:]
        new_model_dict["model description"] = description
        new_model_dict["model status"] = TerminationCondition.maxTimeLimit
        models_dict[new_model_name] = new_model_dict
    elif termination_condition == TerminationCondition.optimal:
        for p, idx in iis_param:
            slack_var_pos = eval("model.slack_pos_" + p)[idx].value
            slack_var_neg = eval("model.slack_neg_" + p)[idx].value
            idx = "" if idx is None else f" at {idx}"
            if slack_var_pos > 1e-5:
                feedback = feedback + f"change {p}{idx} by +{slack_var_pos} unit; \n"
                description = description + f"change {p}{idx} by +{slack_var_pos} unit; \n"
            elif slack_var_neg > 1e-5:
                feedback = feedback + f"change {p}{idx} by -{slack_var_neg} unit; \n"
                description = description + f"change {p}{idx} by -{slack_var_neg} unit; \n"
        feedback = feedback + f"\n\nThe model now becomes feasible. "
        feedback = feedback + f"\n\nHelp the user analyze why the feasibility can be restored by these changes. Let user know this new model will be referred to as {new_model_name}."
        description = description[:7] + f", which becomes feasible" + description[7:]
        new_model_dict["model description"] = description
        new_model_dict["model status"] = TerminationCondition.optimal
        models_dict[new_model_name] = new_model_dict
    else:
        feedback = f"The model remains infeasible after only changing the following: \n"
        for p, idx in iis_param:
            idx = "" if idx is None else f" at {idx}"
            feedback = feedback + f"{p}{idx}; \n"
        description = feedback
        models_dict[queried_model]["model description"] = description
        feedback = feedback + f"\n\nThis is determined by the nature of the model, rather than an error of internal tools. Help the user analyze why the feasibility is not restored."

    feedback = "Feedback from internal tools: \n" + feedback
    return feedback


def sensitivity_analysis(queried_components: List[Dict], queried_model, models_dict):
    queried_model_dict = models_dict[queried_model]
    model = queried_model_dict['model class'].clone()

    if queried_model_dict['model status'] in [TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded]:
        feedback = "Error: The model is infeasible. Sensitivity analysis cannot be performed on an infeasible model."
        feedback = "Feedback from internal tools: \n" + feedback
        return feedback
    if queried_model_dict['model type'] != 'LP':
        feedback = "Error: The model is not a linear programming model. Internal tools do not support sensitivity analysis on other types of models."
        feedback = "Feedback from internal tools: \n" + feedback
        return feedback

    def locate_param(param_name, idx, model=model):
        in_consts = []
        param_name_idx = str(eval("model." + param_name)[idx])
        for const_name in queried_model_dict['components']["parameters"][param_name]["cons_in"]:
            model_const = eval("model." + const_name)
            for con_idx in model_const.index_set():
                con_i = model_const[con_idx]
                expr_params = identify_mutable_parameters(con_i.expr)
                for expr_param in expr_params:
                    if expr_param.name == param_name_idx:
                        coef_body = - differentiate(con_i.body, wrt=expr_param, mode='reverse_symbolic')
                        coef_lower = differentiate(con_i.lower, wrt=expr_param, mode='reverse_symbolic')
                        coef_upper = differentiate(con_i.upper, wrt=expr_param, mode='reverse_symbolic')
                        coef = coef_body + coef_lower + coef_upper
                        in_consts.append({"const_name": const_name,
                                          "const_indexes": con_idx,
                                          "coefficient": coef})
                        break
        return in_consts

    param_const_pairs = []
    for component in queried_components:
        param_name = component['component_name']
        param_indexes = component['component_indexes']
        print(f'param_indexes: {param_indexes}')
        component_type = get_component_type(param_name, queried_model_dict)
        if component_type == 'parameters':

            if isinstance(param_indexes, tuple):
                eval_param = eval(f"model.{param_name}")
                if len(eval_param[param_indexes].index()) <= 0:
                    raise IndexError(f"Error: Indexes are not valid. This usually happens when the order of indexes in the tuple is incorrect.")

            if queried_model_dict['components']['parameters'][param_name]['is_RHS']:
                if isinstance(param_indexes, tuple):
                    if slice(None) in param_indexes:
                        for model_param_i in eval("model." + param_name)[param_indexes]:
                            model_param_i_indexes = model_param_i.index()
                            param_const_pair = {"param_name": param_name,
                                                "param_indexes": model_param_i_indexes,
                                                "consts": locate_param(param_name, model_param_i_indexes)}
                            param_const_pairs.append(param_const_pair)
                    else:
                        param_const_pair = {"param_name": param_name,
                                            "param_indexes": param_indexes,
                                            "consts": locate_param(param_name, param_indexes)}
                        param_const_pairs.append(param_const_pair)
                elif isinstance(param_indexes, slice):
                    for model_param_i in eval("model." + param_name)[param_indexes]:
                        model_param_i_indexes = model_param_i.index()
                        param_const_pair = {"param_name": param_name,
                                            "param_indexes": model_param_i_indexes,
                                            "consts": locate_param(param_name, model_param_i_indexes)}
                        param_const_pairs.append(param_const_pair)
                elif isinstance(param_indexes, int) or isinstance(param_indexes, str):
                    param_const_pair = {"param_name": param_name,
                                        "param_indexes": param_indexes,
                                        "consts": locate_param(param_name, param_indexes)}
                    param_const_pairs.append(param_const_pair)
                elif param_indexes == None:
                    param_const_pair = {"param_name": param_name,
                                        "param_indexes": param_indexes,
                                        "consts": locate_param(param_name, param_indexes)}
                    param_const_pairs.append(param_const_pair)

            else:
                feedback = f"Error: {param_name} is not a RHS parameter in the model. "
                feedback += """
Please confirm with the user and ask them to provide a valid RHS parameter for sensitivity analysis, 
or if they are particularly interested in these parameters, they must specify a modification extent (e.g., a 5% increase) to directly assess the impact of this modification."""
                feedback = "Feedback from internal tools: \n" + feedback
                return feedback

        else:
            wrong_component_type = component_type
            feedback = f"Error: {param_name} is not a parameter in the model but a {wrong_component_type}. "
            feedback += """Please confirm with the user and ask them to provide a valid parameter for sensitivity analysis."""
            feedback = "Feedback from internal tools: \n" + feedback
            return feedback

    # duals = []
    print(f' Does this model have model.dual? {model.find_component("dual") is None}')
    if model.find_component('dual') is None:
        model.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)
        opt = SolverFactory("gurobi")
        results = opt.solve(model, tee=True)
        termination_condition = results.solver.termination_condition
        # update the models_dict
        models_dict[queried_model]["model class"] = model

    for param_const_pair in param_const_pairs:
        for const in param_const_pair['consts']:
            const_name = const['const_name']
            const_indexes = const['const_indexes']
            const_coef = const['coefficient']
            model_const = eval("model." + const_name)
            model_const_i = model_const[const_indexes]
            const['dual_value'] = const_coef * model.dual[model_const_i]
            #print(f'const_name: {const_name}, const_indexes: {const_indexes}, const_coef: {const_coef}, dual_value: {const["dual_value"]}')
            #break

    # construct feedback
    feedback = "The sensitivity analysis results are as follows: \n"
    for param_const_pair in param_const_pairs:
        param_name = param_const_pair['param_name']
        param_indexes = param_const_pair['param_indexes']
        param_indexes = f" at {param_indexes}" if param_indexes != None else ""
        feedback = feedback + f"when a small positive perturbation is made to {param_name}{param_indexes}, "
        total_value = 0
        for const in param_const_pair['consts']:
            #print(f'retrieving dual value of {const["const_name"]} at {const["const_indexes"]}')
            dual_value = const['dual_value']
            #print(f'the dual value of {const["const_name"]} at {const["const_indexes"]} is {dual_value}')
            total_value += dual_value

        if total_value > 1e-5:
            feedback = feedback + f"the optimal objective value will change by {total_value} unit\n"
        elif total_value < -1e-5:
            feedback = feedback + f"the optimal objective value will change by {total_value} unit\n"
        else:
            feedback = feedback + f"the optimal objective value will not change \n"

    feedback += "Please explain these results to the user. \n"
    feedback = "Feedback from internal tools: \n" + feedback
    return feedback


def components_retrival(queried_components: List[Dict], queried_model, models_dict):
    queried_model_dict = models_dict[queried_model]
    model = queried_model_dict['model class'].clone()
    feedback = f"In the {queried_model}, "
    for component in queried_components:
        component_name = component['component_name']
        component_indexes = component['component_indexes']
        print(f'component_indexes: {component_indexes}')
        model_component = eval("model." + component_name)

        if isinstance(component_indexes, tuple):
            # supposed to retrieve multiple components
            if slice(None) in component_indexes:

                if len(model_component[component_indexes].index()) <= 0:
                    raise IndexError(f"Error: Indexes are not valid. This usually happens when the order of indexes in the tuple is incorrect.")

                for model_component_i in model_component[component_indexes]:
                    model_component_i_indexes = model_component_i.index()
                    feedback = feedback + f"{component_name} at {str(model_component_i_indexes)} is "
                    component_retrieval = ""
                    if component_name in queried_model_dict["components"]["parameters"]:
                        component_retrieval = str(model_component_i.value)
                    elif component_name in queried_model_dict["components"]["variables"]:
                        component_retrieval = str(model_component_i.value)
                    elif component_name in queried_model_dict["components"]["sets"]:
                        component_retrieval = str(model_component_i.data())
                    elif component_name in queried_model_dict["components"]["constraints"]:
                        component_retrieval = str(model_component_i.expr)
                    elif component_name in queried_model_dict["components"]["objective"]:
                        component_retrieval = str(model_component_i())
                    feedback = feedback + f"{component_retrieval}.\n"

            else:
                # supposed to retrieve one component
                feedback = feedback + f"{component_name} at {str(component_indexes)} is "
                component_retrieval = ""
                if component_name in queried_model_dict["components"]["parameters"]:
                    component_retrieval = str(model_component[component_indexes].value)
                elif component_name in queried_model_dict["components"]["variables"]:
                    component_retrieval = str(model_component[component_indexes].value)
                elif component_name in queried_model_dict["components"]["sets"]:
                    component_retrieval = str(model_component[component_indexes].data())
                elif component_name in queried_model_dict["components"]["constraints"]:
                    component_retrieval = str(model_component[component_indexes].expr)
                elif component_name in queried_model_dict["components"]["objective"]:
                    component_retrieval = str(model_component[component_indexes]())
                feedback = feedback + f"{component_retrieval}.\n"

        elif isinstance(component_indexes, slice):
            # supposed to retrieve multiple components
            for model_component_i in model_component[component_indexes]:
                model_component_i_indexes = model_component_i.index()
                feedback = feedback + f"{component_name} at {str(model_component_i_indexes)} is "
                component_retrieval = ""
                if component_name in queried_model_dict["components"]["parameters"]:
                    component_retrieval = str(model_component_i.value)
                elif component_name in queried_model_dict["components"]["variables"]:
                    component_retrieval = str(model_component_i.value)
                elif component_name in queried_model_dict["components"]["sets"]:
                    component_retrieval = str(model_component_i.data())
                elif component_name in queried_model_dict["components"]["constraints"]:
                    component_retrieval = str(model_component_i.expr)
                elif component_name in queried_model_dict["components"]["objective"]:
                    component_retrieval = str(model_component_i())
                feedback = feedback + f"{component_retrieval}.\n"

        elif isinstance(component_indexes, int) or isinstance(component_indexes, str):
            # supposed to retrieve one component
            feedback = feedback + f"{component_name} at {str(component_indexes)} is "
            component_retrieval = ""
            if component_name in queried_model_dict["components"]["parameters"]:
                component_retrieval = str(model_component[component_indexes].value)
            elif component_name in queried_model_dict["components"]["variables"]:
                component_retrieval = str(model_component[component_indexes].value)
            elif component_name in queried_model_dict["components"]["sets"]:
                component_retrieval = str(model_component[component_indexes].data())
            elif component_name in queried_model_dict["components"]["constraints"]:
                component_retrieval = str(model_component[component_indexes].expr)
            elif component_name in queried_model_dict["components"]["objective"]:
                component_retrieval = str(model_component[component_indexes]())
            feedback = feedback + f"{component_retrieval}.\n"

        elif component_indexes == None:
            # supposed to retrieve one component
            feedback = feedback + f"{component_name} is "
            component_retrieval = ""
            if component_name in queried_model_dict["components"]["parameters"]:
                component_retrieval = str(model_component.value)
            elif component_name in queried_model_dict["components"]["variables"]:
                component_retrieval = str(model_component.value)
            elif component_name in queried_model_dict["components"]["sets"]:
                component_retrieval = str(model_component.data())
            elif component_name in queried_model_dict["components"]["constraints"]:
                component_retrieval = str(model_component.expr)
            elif component_name in queried_model_dict["components"]["objective"]:
                component_retrieval = str(model_component())
            feedback = feedback + f"{component_retrieval}.\n"

    feedback += "Please describe the information using their physical meanings to the user. \n"
    feedback = "Feedback from internal tools: \n" + feedback
    return feedback


def evaluate_modification(queried_components: List[Dict], queried_model, models_dict):
    queried_model_dict = models_dict[queried_model]
    model = queried_model_dict['model class'].clone()
    for obj_name, obj in model.component_map(pe.Objective).items():
        original_obj_value = queried_model_dict['components']['objective'][obj_name]['optimal_value']
    feedback = f"In the {queried_model}, the following modifications are made: \n"
    description = f"a model with the following changes to {queried_model}: \n"
    for component in queried_components:
        component_name = component['component_name']
        component_indexes = component['component_indexes']
        component_operation = component['operation']

        if component_operation == "!":
            return ("Error: The evaluate_modification function requires a specific modification extent. "
                    "Debug suggestion: distribute this task to operator again and ask them to use sensitivity_analysis function instead.")

        component_delta = str(component['delta'])
        print(f'component_indexes: {component_indexes}')
        print(f'component_operation: {component_operation}')
        print(f'component_delta: {component_delta}')
        model_component = eval("model." + component_name)

        if isinstance(component_indexes, tuple):
            if slice(None) in component_indexes:

                if len(model_component[component_indexes].index()) <= 0:
                    raise IndexError(f"Error: Indexes are not valid. This usually happens when the order of indexes in the tuple is incorrect.")

                for model_component_i in model_component[component_indexes]:
                    model_component_i_indexes = model_component_i.index()
                    if component_name in queried_model_dict["components"]["parameters"]:
                        value_for_modification = eval("model." + component_name)[model_component_i_indexes].value
                        value_after_modification = eval(component_delta) if component_operation == "=" else eval(str(value_for_modification) + component_operation + component_delta)
                        model_component[model_component_i_indexes].set_value(value_after_modification)
                        changed_or_fixed = " is changed to "
                    elif component_name in queried_model_dict["components"]["variables"]:
                        value_for_modification = eval("model." + component_name)[model_component_i_indexes].value
                        value_after_modification = eval(component_delta) if component_operation == "=" else eval(str(value_for_modification) + component_operation + component_delta)
                        model_component[model_component_i_indexes].fix(value_after_modification)
                        changed_or_fixed = " is fixed to "
                    feedback += f"{component_name} at {str(model_component_i_indexes)}" + changed_or_fixed + str(value_after_modification) + ".\n"
                    description += f"{component_name} at {str(model_component_i_indexes)}" + changed_or_fixed + str(value_after_modification) + ".\n"

            else:
                if component_name in queried_model_dict["components"]["parameters"]:
                    value_for_modification = eval("model." + component_name)[component_indexes].value
                    value_after_modification = eval(component_delta) if component_operation == "=" else eval(str(value_for_modification) + component_operation + component_delta)
                    model_component[component_indexes].set_value(value_after_modification)
                    changed_or_fixed = " is changed to "
                elif component_name in queried_model_dict["components"]["variables"]:
                    value_for_modification = eval("model." + component_name)[component_indexes].value
                    value_after_modification = eval(component_delta) if component_operation == "=" else eval(str(value_for_modification) + component_operation + component_delta)
                    model_component[component_indexes].fix(value_after_modification)
                    changed_or_fixed = " is fixed to "
                feedback += f"{component_name} at {str(component_indexes)}" + changed_or_fixed + str(value_after_modification) + ".\n"
                description += f"{component_name} at {str(component_indexes)}" + changed_or_fixed + str(value_after_modification) + ".\n"

        elif isinstance(component_indexes, slice):
            for model_component_i in model_component[component_indexes]:
                model_component_i_indexes = model_component_i.index()
                if component_name in queried_model_dict["components"]["parameters"]:
                    value_for_modification = eval("model." + component_name)[model_component_i_indexes].value
                    value_after_modification = eval(component_delta) if component_operation == "=" else eval(str(value_for_modification) + component_operation + component_delta)
                    model_component[model_component_i_indexes].set_value(value_after_modification)
                    changed_or_fixed = " is changed to "
                elif component_name in queried_model_dict["components"]["variables"]:
                    value_for_modification = eval("model." + component_name)[model_component_i_indexes].value
                    value_after_modification = eval(component_delta) if component_operation == "=" else eval(str(value_for_modification) + component_operation + component_delta)
                    model_component[model_component_i_indexes].fix(value_after_modification)
                    changed_or_fixed = " is fixed to "
                feedback += f"{component_name} at {str(model_component_i_indexes)}" + changed_or_fixed + str(value_after_modification) + ".\n"
                description += f"{component_name} at {str(model_component_i_indexes)}" + changed_or_fixed + str(value_after_modification) + ".\n"

        elif isinstance(component_indexes, int) or isinstance(component_indexes, str) or component_indexes == None:
            if component_name in queried_model_dict["components"]["parameters"]:
                value_for_modification = eval("model." + component_name)[component_indexes].value
                value_after_modification = eval(component_delta) if component_operation == "=" else eval(str(value_for_modification) + component_operation + component_delta)
                model_component[component_indexes].set_value(value_after_modification)
                changed_or_fixed = " is changed to "
            elif component_name in queried_model_dict["components"]["variables"]:
                value_for_modification = eval("model." + component_name)[component_indexes].value
                value_after_modification = eval(component_delta) if component_operation == "=" else eval(str(value_for_modification) + component_operation + component_delta)
                model_component[component_indexes].fix(value_after_modification)
                changed_or_fixed = " is fixed to "
            idx = "" if component_indexes == None else f" at {str(component_indexes)}"
            feedback += f"{component_name}{idx}" + changed_or_fixed + str(value_after_modification) + ".\n"
            description += f"{component_name}{idx}" + changed_or_fixed + str(value_after_modification) + ".\n"

        # resolve the model
        opt = SolverFactory('gurobi')
        opt.options['nonConvex'] = 2
        opt.options['TimeLimit'] = 300  # 5min time limit
        results = opt.solve(model, tee=True)
        # construct technical feedback
        termination_condition = results.solver.termination_condition
        new_model_dict = pyomo2json(model, termination_condition=termination_condition)

        new_model_name = get_new_model_name(queried_model)

        if termination_condition == TerminationCondition.maxTimeLimit:
            feedback += f"\n\nThe model is not solved due to time limit, and the new model will be referred to as {new_model_name}."
            feedback += f"The best objective value found so far is {results.Problem[0]['Upper bound']}.\n"
            feedback = feedback + f"\n\nHelp the user analyze the influence of these modifications."
            description = description[:7] + f", which is not solved due to time limit" + description[7:]
            new_model_dict["model description"] = description
            models_dict[new_model_name] = new_model_dict
        elif termination_condition == TerminationCondition.optimal:
            feedback += f"\n\nThe model now is feasible, and the new model will be referred to as {new_model_name}."
            feedback += f"The optimal objective value found is {results.Problem[0]['Lower bound']}.\n"
            feedback = feedback + f"\n\nHelp the user analyze the influence of these modifications. "
            description = description[:7] + f", which is feasible" + description[7:]
            new_model_dict["model description"] = description
            models_dict[new_model_name] = new_model_dict
        else:
            feedback += f"\n\nThe model now is infeasible, and the new model will be referred to as {new_model_name}."
            feedback = feedback + f"\n\nHelp the user analyze the influence of these modifications."
            description = description[:7] + f", which is infeasible" + description[7:]
            new_model_dict["model description"] = description
            models_dict[new_model_name] = new_model_dict

        reminder = f"Reminder: the status of old model, {queried_model}, was "
        if queried_model_dict['model status'] == TerminationCondition.maxTimeLimit:
            reminder += f"is not solved due to time limit with best objective value found as {original_obj_value}."
        elif queried_model_dict['model status'] == TerminationCondition.optimal:
            reminder += f"solved with optimal objective value found as {original_obj_value}."
        elif queried_model_dict['model status'] in [TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded]:
            reminder += f"infeasible."

        feedback += reminder
        feedback = "Feedback from internal tools: \n" + feedback

    return feedback


def alternative_solutions(queried_components: List[Dict], queried_model, models_dict,
                          n_solutions: int = 5, pool_gap: float = 1.0,
                          max_diffs_reported: int = 15, tol: float = 1e-5):
    """Generate near-optimal alternative solutions (counterfactuals) with Gurobi's
    solution pool, and contrast them with the incumbent optimal solution.

    Answers questions such as "why is this optimal", "are there other solutions",
    and "why are other solutions not better". The incumbent optimal solution is
    referred to as P; each pooled alternative is a Q. For every Q we report its
    objective value, the objective gap relative to P (the "price" of that
    alternative), and the decision variables whose values differ from P.

    queried_components (optional) focuses the comparison on the variables the user
    named; if none of them are variables, all model variables are compared.
    """
    queried_model_dict = models_dict[queried_model]

    if queried_model_dict['model status'] in [TerminationCondition.infeasible,
                                              TerminationCondition.infeasibleOrUnbounded]:
        feedback = ("Error: The model is infeasible, so there are no alternative solutions to compare. "
                    "Alternative-solution analysis can only be performed on a feasible/optimal model.")
        return "Feedback from internal tools: \n" + feedback

    model = queried_model_dict['model class'].clone()

    # Determine which variables to focus the counterfactual comparison on.
    focus_var_names = []
    for component in (queried_components or []):
        name = component.get('component_name')
        if name is not None and get_component_type(name, queried_model_dict) == 'variables':
            focus_var_names.append(name)
    focus_var_names = list(dict.fromkeys(focus_var_names))  # de-dup, keep order

    # objective sense: 1 = minimize, -1 = maximize
    obj = next(model.component_objects(pe.Objective, active=True))
    sense = obj.sense
    sense_word = "lower" if sense == pe.minimize else "higher"

    # Solve with the Gurobi solution pool (mode 2 = the n best solutions).
    try:
        from gurobipy import GRB
        opt = SolverFactory('gurobi_persistent')
        opt.set_instance(model)
        opt.set_gurobi_param('NonConvex', 2)
        opt.set_gurobi_param('TimeLimit', 300)
        opt.set_gurobi_param('Threads', 1)
        opt.set_gurobi_param('PoolSearchMode', 2)          # find the n best solutions
        opt.set_gurobi_param('PoolSolutions', int(n_solutions) + 1)  # +1 to include incumbent P
        opt.set_gurobi_param('PoolGap', float(pool_gap))  # relative: 1 = keep Q within 100% of P
        results = opt.solve(tee=True)
    except Exception as e:
        feedback = (f"Error: Could not generate a solution pool with the Gurobi persistent interface ({e}). "
                    f"This usually means 'gurobi_persistent' or gurobipy is unavailable.")
        return "Feedback from internal tools: \n" + feedback

    grb = opt._solver_model
    sol_count = grb.SolCount

    # Collect the (pyomo var, gurobi var) pairs we want to inspect.
    var_map = opt._pyomo_var_to_solver_var_map
    inspected = []  # list of (label, pyomo_vardata, gurobi_var)
    for var in model.component_objects(pe.Var, active=True):
        if focus_var_names and var.name not in focus_var_names:
            continue
        for idx in var:
            vardata = var[idx]
            gvar = var_map.get(id(vardata))
            if gvar is None:
                continue
            label = var.name if idx is None else f"{var.name}[{idx}]"
            inspected.append((label, vardata, gvar))

    def read_pool_solution(i):
        grb.setParam(GRB.Param.SolutionNumber, i)
        values = {label: gvar.Xn for (label, _v, gvar) in inspected}
        return grb.PoolObjVal, values

    # Incumbent optimal solution P = pool solution 0.
    obj_P, values_P = read_pool_solution(0)

    focus_note = (f" (focusing on: {', '.join(focus_var_names)})" if focus_var_names else "")
    feedback = (f"Solution-pool analysis of {queried_model}{focus_note}.\n"
                f"The incumbent optimal solution is referred to as P, with objective value {obj_P}.\n")

    if sol_count <= 1:
        feedback += ("\nGurobi found no alternative solutions within the search settings: the optimal solution "
                     "appears to be unique (or all alternatives are worse than the pool gap allows). "
                     "This is itself the explanation of optimality — help the user understand that no other "
                     "solution achieves the same objective.\n")
        feedback += "\nPlease explain to the user why the optimal solution is effectively unique. \n"
        return "Feedback from internal tools: \n" + feedback

    feedback += (f"\nGurobi returned {sol_count - 1} alternative solution(s) (Q). For a minimization the "
                 f"alternatives have a {sense_word if sense==pe.minimize else 'higher'} (worse) objective; "
                 f"the gap is the price of choosing that alternative over P.\n")

    for i in range(1, sol_count):
        obj_Q, values_Q = read_pool_solution(i)
        gap = obj_Q - obj_P  # for minimize, >= 0 (Q is worse); for maximize, <= 0
        rel = (abs(gap) / abs(obj_P) * 100) if abs(obj_P) > tol else float('nan')
        feedback += (f"\nAlternative Q{i}: objective = {obj_Q} "
                     f"(worse than P by {abs(gap)}, i.e. {rel:.2f}% ).\n")

        diffs = []
        for label in values_P:
            dP, dQ = values_P[label], values_Q[label]
            if abs(dP - dQ) > tol:
                diffs.append((abs(dP - dQ), label, dP, dQ))
        diffs.sort(reverse=True)
        if not diffs:
            feedback += "  This alternative has the same variable values as P within tolerance.\n"
        else:
            feedback += f"  Decision differences vs P ({len(diffs)} variable(s) differ, showing up to {max_diffs_reported}):\n"
            for _mag, label, dP, dQ in diffs[:max_diffs_reported]:
                feedback += f"    - {label}: P = {dP}, Q{i} = {dQ}\n"

    feedback += ("\nUse these counterfactuals to explain the optimality of P: for each alternative Q, state that "
                 "it IS a valid alternative but is worse by the reported gap, and point to the specific decision "
                 "changes that cause the worse objective. Frame it as 'P is better than Q because ...'. \n")
    return "Feedback from internal tools: \n" + feedback


# ---------------------------------------------------------------------------
# Stochastic-forecast tools
#
# The deterministic model treats forecast parameters (e.g. the inflow) as
# known. These two tools quantify what forecast uncertainty does to the
# solution, so that the Explainer never has to guess about risk:
#   * scenario_risk_assessment  — Monte-Carlo stress test of the incumbent
#     (deterministic) first-stage schedule under sampled forecast scenarios.
#   * stochastic_hedging_analysis — two-stage stochastic program (extensive
#     form) that produces a hedged first-stage schedule and the classic
#     stochastic-programming diagnostics (VSS, EVPI, wet/dry contrasts).
#
# The scenario generator is the SAME AR(1) log-normal process as in
# Scriptie_martijn/'stochastic approximation.py' (multiplicative noise around
# the deterministic forecast, mean-preserving), so SPSA results obtained
# outside OptiChat remain a valid cross-check of these tools.
# ---------------------------------------------------------------------------

def _ar1_lognormal_scenarios(forecast, n_scenarios, phi, sigma_eps, rng):
    """Sample forecast trajectories around the deterministic forecast.

    Z_t = phi*Z_{t-1} + eps_t (Z_0 = 0),  path[t] = forecast[t] * exp(Z_t - sigma_Y^2/2),
    with sigma_Y^2 the stationary variance of Z. For a constant forecast this is
    exactly the generator of the SPSA study; the cone starts at the current
    observation and widens with lead time, mean-preserving in the long run.
    """
    forecast = np.asarray(forecast, dtype=float)
    horizon = forecast.size
    sigma_y2 = sigma_eps ** 2 / (1.0 - phi ** 2)
    z = np.zeros((n_scenarios, horizon))
    eps = rng.normal(0.0, sigma_eps, size=(n_scenarios, horizon))
    for t in range(1, horizon):
        z[:, t] = phi * z[:, t - 1] + eps[:, t]
    return forecast * np.exp(z - 0.5 * sigma_y2)


def _trajectory_indexes(component):
    return sorted(component.index_set())


def _set_trajectory(model, param_name, path):
    param = getattr(model, param_name)
    for value, idx in zip(path, _trajectory_indexes(param)):
        param[idx].set_value(float(value))


def _solve_quietly(model, time_limit=300):
    opt = SolverFactory('gurobi')
    opt.options['nonConvex'] = 2
    opt.options['TimeLimit'] = time_limit
    results = opt.solve(model, tee=False)
    return results.solver.termination_condition


def _active_objective(model):
    return next(model.component_objects(pe.Objective, active=True))


def _first_stage_variable_names(model, queried_model_dict, queried_components):
    """Variables held fixed while scenarios play out (the here-and-now schedule).

    The user/LLM can name them explicitly in queried_components; the default is
    the set of decision variables that appear in the objective (the cost-bearing
    controls, e.g. Q_pump), everything else being recourse/state.
    """
    obj = _active_objective(model)
    objective_vars = list(dict.fromkeys(
        v.parent_component().name for v in identify_variables(obj.expr, include_fixed=True)))
    # A first-stage decision is a CONTROL the operator commits to before the
    # uncertainty is realised — in these cost-minimising models exactly the
    # variables the objective pays for (e.g. Q_pump). Variables the user names in
    # the query normally identify the RISK they are asking about (e.g. H_storage
    # in a flooding question), NOT a control to freeze: fixing a state variable
    # across scenarios forces the mass balance to absorb every extra unit of
    # inflow into the remaining flows, which makes wet scenarios infeasible for
    # the wrong reason. So only honour named variables that are objective controls.
    named = []
    for component in (queried_components or []):
        name = component.get('component_name')
        if name is not None and get_component_type(name, queried_model_dict) == 'variables' \
                and name in objective_vars:
            named.append(name)
    named = list(dict.fromkeys(named))
    return named if named else objective_vars


def _uncertain_parameter_name(queried_model_dict, queried_components):
    """Pick the forecast parameter to perturb. Returns (name, None) on success,
    or (None, candidates) when the tool cannot decide on its own."""
    params_meta = queried_model_dict['components']['parameters']
    named = []
    for component in (queried_components or []):
        name = component.get('component_name')
        if name is not None and get_component_type(name, queried_model_dict) == 'parameters':
            named.append(name)
    named = list(dict.fromkeys(named))
    named_indexed = [n for n in named if params_meta[n]['is_indexed']]
    if named_indexed:
        return named_indexed[0], None
    if named:
        # only scalar parameters were named — uncertainty needs a trajectory
        return None, [n for n, meta in params_meta.items() if meta['is_indexed']]
    # fall back on the Interpreter's adjustability classification, if present
    forecast_params = [n for n, meta in params_meta.items()
                       if meta.get('adjustability') == 'forecast' and meta['is_indexed']]
    if len(forecast_params) == 1:
        return forecast_params[0], None
    candidates = forecast_params or [n for n, meta in params_meta.items() if meta['is_indexed']]
    return None, candidates


def _soften_recourse_constraints(model, first_stage_names, violation_penalty):
    """Make the model relatively complete recourse: inequality constraints that
    involve only continuous non-first-stage (state/recourse) variables — e.g. a
    storage level limit — become soft with a penalised violation slack, and the
    native bounds of those state variables are softened the same way. Physics
    (equalities), logic (anything with binaries) and first-stage capacity
    constraints stay hard. Returns one label per slack, aligned with the
    soft_violation VarList indexes 1..n (labels survive model.clone()).
    """
    first_stage = set(first_stage_names)
    candidates = []
    for cons in list(model.component_objects(pe.Constraint, active=True)):
        for idx in cons:
            con = cons[idx]
            if con.equality:
                continue
            vars_in = list(identify_variables(con.body, include_fixed=True))
            if not vars_in:
                continue
            if any(v.parent_component().name in first_stage for v in vars_in):
                continue
            if any(v.is_binary() or v.is_integer() for v in vars_in):
                continue
            candidates.append((cons, idx, con))

    model.soft_violation = pe.VarList(domain=pe.NonNegativeReals)
    model.soft_constraints = pe.ConstraintList()
    records = []

    softened_var_names = set()
    for cons, idx, con in candidates:
        for v in identify_variables(con.body, include_fixed=True):
            softened_var_names.add(v.parent_component().name)
        suffix = "" if idx is None else f"[{idx}]"
        if con.has_ub():
            slack = model.soft_violation.add()
            model.soft_constraints.add(con.body - slack <= con.upper)
            records.append({'label': f"{cons.name}{suffix} (upper)", 'index': idx})
        if con.has_lb():
            slack = model.soft_violation.add()
            model.soft_constraints.add(con.body + slack >= con.lower)
            records.append({'label': f"{cons.name}{suffix} (lower)", 'index': idx})
        con.deactivate()

    # native bounds of the state variables must be softened too, otherwise a
    # fixed schedule can make a scenario infeasible (e.g. pumping the basin dry)
    for var_name in sorted(softened_var_names):
        var = getattr(model, var_name)
        for idx in var:
            vardata = var[idx]
            if vardata.is_binary() or vardata.is_integer():
                continue
            suffix = "" if idx is None else f"[{idx}]"
            if vardata.lb is not None:
                slack = model.soft_violation.add()
                model.soft_constraints.add(vardata + slack >= vardata.lb)
                records.append({'label': f"{var_name}{suffix} (below lower bound)", 'index': idx})
                vardata.domain = pe.Reals
                vardata.setlb(None)
            if vardata.ub is not None:
                slack = model.soft_violation.add()
                model.soft_constraints.add(vardata - slack <= vardata.ub)
                records.append({'label': f"{var_name}{suffix} (above upper bound)", 'index': idx})
                vardata.setub(None)

    obj = _active_objective(model)
    sign = 1.0 if obj.sense == pe.minimize else -1.0
    obj.expr = obj.expr + sign * violation_penalty * sum(model.soft_violation.values())
    return records


def _binding_candidates(model):
    """Hard inequality constraints (no binaries, still active after softening)
    whose tightness is worth reporting, e.g. the pump capacity limit."""
    candidates = []
    for cons in model.component_objects(pe.Constraint, active=True):
        if cons.name in ('soft_constraints', 'nonanticipativity'):
            continue
        for idx in cons:
            con = cons[idx]
            if con.equality:
                continue
            vars_in = list(identify_variables(con.body, include_fixed=True))
            if not vars_in or any(v.is_binary() or v.is_integer() for v in vars_in):
                continue
            suffix = "" if idx is None else f"[{idx}]"
            candidates.append({'cons_name': cons.name, 'index': idx,
                               'label': f"{cons.name}{suffix}"})
    return candidates


def _is_binding(model, candidate, tol=1e-5):
    con = getattr(model, candidate['cons_name'])[candidate['index']]
    body = pe.value(con.body)
    if con.has_ub() and abs(pe.value(con.upper) - body) <= tol:
        return True
    if con.has_lb() and abs(body - pe.value(con.lower)) <= tol:
        return True
    return False


def _evaluate_fixed_schedule(model, param_name, scenarios, records, tol):
    """Solve the (softened) model once per scenario with the first stage already
    fixed, and collect cost and violation statistics. Returns one dict per
    scenario; 'solved' is False when the solver failed on that scenario."""
    obj = _active_objective(model)
    out = []
    for path in scenarios:
        _set_trajectory(model, param_name, path)
        termination = _solve_quietly(model)
        if termination != TerminationCondition.optimal:
            out.append({'solved': False, 'termination': str(termination)})
            continue
        violations = []
        for i, record in enumerate(records, start=1):
            slack_value = model.soft_violation[i].value or 0.0
            if slack_value > tol:
                violations.append((record['label'], slack_value))
        out.append({'solved': True,
                    'cost': pe.value(obj),
                    'violations': violations,
                    'total_violation': sum(v for _, v in violations)})
    return out


def _violation_statistics(per_scenario):
    solved = [s for s in per_scenario if s['solved']]
    n_solved = len(solved)
    n_failed = len(per_scenario) - n_solved
    violated = [s for s in solved if s['violations']]
    p_violation = len(violated) / n_solved if n_solved else float('nan')
    half_width = 1.96 * np.sqrt(p_violation * (1 - p_violation) / n_solved) if n_solved else float('nan')
    frequency = {}
    for s in solved:
        for label, _value in s['violations']:
            frequency[label] = frequency.get(label, 0) + 1
    worst = max(solved, key=lambda s: s['total_violation'], default=None)
    return {'n_solved': n_solved, 'n_failed': n_failed,
            'p_violation': p_violation, 'ci_half_width': half_width,
            'frequency': frequency, 'worst': worst,
            'expected_cost': float(np.mean([s['cost'] for s in solved])) if solved else float('nan')}


def _format_violation_block(stats, n_solved, max_lines=12):
    lines = []
    by_frequency = sorted(stats['frequency'].items(), key=lambda kv: kv[1], reverse=True)
    for label, count in by_frequency[:max_lines]:
        lines.append(f"    {label}: violated in {count / n_solved:.1%} of scenarios")
    if len(by_frequency) > max_lines:
        lines.append(f"    ... and {len(by_frequency) - max_lines} more constraint/bound locations")
    return "\n".join(lines) if lines else "    (none)"


# How uncertain the forecast is, as the user phrased it. The LLM only picks the
# LABEL; the numbers live here so they are documented, reproducible and auditable
# rather than invented per call.
UNCERTAINTY_LEVELS = {
    'low':      {'sigma_eps': 0.15, 'phrasing': '"a bit", "slightly", "small" uncertainty'},
    'moderate': {'sigma_eps': 0.30, 'phrasing': 'unspecified or "some" uncertainty'},
    'high':     {'sigma_eps': 0.50, 'phrasing': '"a lot", "very uncertain", storm conditions'},
}

# AR(1) persistence of the forecast error. This describes how strongly wet/dry
# hours cluster in the catchment — a property of the weather, NOT of how uncertain
# the user says the forecast is — so it is a fixed documented constant and is
# deliberately NOT selectable by the LLM (only sigma_eps varies with the level).
PHI_PERSISTENCE = 0.8


def scenario_risk_assessment(queried_components: List[Dict], queried_model, models_dict,
                             uncertainty_level: str = 'moderate',
                             n_scenarios: int = 200, seed: int = 2026):
    """Monte-Carlo stress test of the committed schedule under forecast uncertainty.

    The first-stage schedule is held FIXED at the deterministic optimum, the
    uncertain forecast parameter is resampled per scenario, and every constraint
    of the model stays HARD. A scenario in which the model is then infeasible is
    one in which the committed schedule cannot be operated at all — for a water
    model that is exactly the flooding case. The tool therefore reports the
    fraction of scenarios in which the plan fails, over the FULL set of sampled
    scenarios (never a filtered denominator).
    """
    start_time = time.time()
    queried_model_dict = models_dict[queried_model]

    if queried_model_dict['model status'] in [TerminationCondition.infeasible,
                                              TerminationCondition.infeasibleOrUnbounded]:
        feedback = ("Error: The model is infeasible, so there is no incumbent schedule to stress-test. "
                    "Restore feasibility first (e.g. with feasibility_restoration).")
        return "Feedback from internal tools: \n" + feedback

    level = str(uncertainty_level or 'moderate').strip().lower()
    if level not in UNCERTAINTY_LEVELS:
        level = 'moderate'
    sigma_eps = UNCERTAINTY_LEVELS[level]['sigma_eps']
    phi = PHI_PERSISTENCE

    param_name, candidates = _uncertain_parameter_name(queried_model_dict, queried_components)
    if param_name is None:
        feedback = ("Error: Could not determine which forecast parameter is uncertain. "
                    f"Ask the user (or infer from the query) which one of these indexed parameters "
                    f"should be treated as uncertain and query again: {candidates}.")
        return "Feedback from internal tools: \n" + feedback

    model = queried_model_dict['model class'].clone()
    first_stage = _first_stage_variable_names(model, queried_model_dict, queried_components)

    # make sure the incumbent solution is loaded before committing to the schedule
    if any(vardata.value is None
           for name in first_stage for vardata in getattr(model, name).values()):
        _solve_quietly(model)
    for name in first_stage:
        getattr(model, name).fix()

    forecast = [pe.value(getattr(model, param_name)[idx])
                for idx in _trajectory_indexes(getattr(model, param_name))]
    n_scenarios = int(n_scenarios)
    rng = np.random.default_rng(seed)
    scenarios = _ar1_lognormal_scenarios(forecast, n_scenarios, phi, sigma_eps, rng)

    # Replay the committed schedule against every scenario with all limits hard.
    feasible_idx, infeasible_idx, unknown = [], [], []
    for s, path in enumerate(scenarios):
        _set_trajectory(model, param_name, path)
        termination = _solve_quietly(model)
        if termination == TerminationCondition.optimal:
            feasible_idx.append(s)
        elif termination in (TerminationCondition.infeasible,
                             TerminationCondition.infeasibleOrUnbounded):
            infeasible_idx.append(s)
        else:
            unknown.append((s, str(termination)))

    n_fail = len(infeasible_idx)
    p_fail = n_fail / n_scenarios
    half_width = 1.96 * np.sqrt(p_fail * (1.0 - p_fail) / n_scenarios)

    # How wet does it have to get? Grounded, IIS-free explanation of the driver.
    totals = scenarios.sum(axis=1)
    forecast_total = float(np.sum(forecast))
    ok_totals = totals[feasible_idx] if feasible_idx else np.array([])
    bad_totals = totals[infeasible_idx] if infeasible_idx else np.array([])
    if bad_totals.size and ok_totals.size:
        driver_line = (
            f"- the failures are the wet scenarios: scenarios in which the plan holds have a total "
            f"{param_name} of {ok_totals.mean():.1f} on average (max {ok_totals.max():.1f}), while the "
            f"failing ones average {bad_totals.mean():.1f} (min {bad_totals.min():.1f}); the "
            f"deterministic forecast total is {forecast_total:.1f}\n")
    else:
        driver_line = (f"- deterministic forecast total {param_name} = {forecast_total:.1f}; "
                       f"sampled totals ranged {totals.min():.1f} to {totals.max():.1f}\n")

    unknown_line = ""
    if unknown:
        reasons = sorted({r for _s, r in unknown})
        unknown_line = (
            f"- CAUTION: {len(unknown)} of {n_scenarios} scenarios returned neither optimal nor "
            f"infeasible ({', '.join(reasons)}); they are counted as NOT failing, so the reported "
            f"probability is a lower bound (upper bound {(n_fail + len(unknown)) / n_scenarios:.3f})\n")

    duration = time.time() - start_time
    feedback = (
        f"Scenario-based risk assessment of {queried_model} (Monte-Carlo stress test of the committed schedule).\n"
        f"Provenance — every number below comes from exactly this computation: uncertain parameter "
        f"{param_name} resampled with AR(1) log-normal multiplicative noise around its current "
        f"deterministic forecast; uncertainty level '{level}' ({UNCERTAINTY_LEVELS[level]['phrasing']}) "
        f"-> sigma_eps={sigma_eps}, phi={phi} (fixed constant, not selectable); the noise is "
        f"MEAN-PRESERVING, so scenarios are not biased wet — they spread symmetrically in log space "
        f"around the same expected forecast; {n_scenarios} scenarios, seed {seed}; first-stage schedule "
        f"{first_stage} held FIXED at the incumbent optimal solution; ALL model constraints kept HARD "
        f"(no penalties, no relaxation); solver Gurobi; runtime {duration:.1f}s.\n\n"
        f"Results over all {n_scenarios} sampled scenarios (nothing excluded):\n"
        f"- the committed schedule admits NO feasible operation in {n_fail} of {n_scenarios} scenarios "
        f"= {p_fail:.3f} (95% CI +/- {half_width:.3f})\n"
        f"- it can be operated within every limit in the remaining {len(feasible_idx)} scenarios\n"
        f"{driver_line}"
        f"{unknown_line}"
        f"\nGuidance for the explainer: 'no feasible operation' means that with this schedule committed "
        f"there is NO admissible way to run the remaining structures within the model's physical limits "
        f"— for a water model that is the flooding case (the storage limit cannot be respected). Report "
        f"{p_fail:.3f} as the probability the plan fails, state the uncertainty level it assumes, and use "
        f"the model's own component descriptions to say which physical limit is at stake. The tool "
        f"deliberately does NOT identify the hour of failure — do not invent timing, magnitudes, or "
        f"return periods. Every quantitative claim MUST come from the numbers above. If the user asks "
        f"what to do about the risk, suggest the hedged schedule (stochastic_hedging_analysis).\n"
    )
    return "Feedback from internal tools: \n" + feedback


def stochastic_hedging_analysis(queried_components: List[Dict], queried_model, models_dict,
                                n_scenarios: int = 30, phi: float = PHI_PERSISTENCE, sigma_eps: float = 0.3,
                                violation_penalty: float = 1.0e6, seed: int = 2026, tol: float = 1e-5):
    """Two-stage stochastic program over sampled forecast scenarios (extensive
    form): one shared first-stage schedule, per-scenario recourse with penalised
    soft-constraint violations. Contrasts the hedged schedule with the
    deterministic one and reports the classic diagnostics — VSS (value of the
    stochastic solution), EVPI (expected value of perfect information),
    violation probabilities of both schedules, per-hour spread of the
    scenario-optimal (wait-and-see) schedules, and which hard constraints bind
    in wet versus dry scenarios. Answers "what should we do under uncertainty,
    why does it differ from the deterministic plan, and is it worth it?".
    """
    start_time = time.time()
    queried_model_dict = models_dict[queried_model]

    if queried_model_dict['model status'] in [TerminationCondition.infeasible,
                                              TerminationCondition.infeasibleOrUnbounded]:
        feedback = ("Error: The model is infeasible; there is no deterministic solution to hedge against. "
                    "Restore feasibility first (e.g. with feasibility_restoration).")
        return "Feedback from internal tools: \n" + feedback

    param_name, candidates = _uncertain_parameter_name(queried_model_dict, queried_components)
    if param_name is None:
        feedback = ("Error: Could not determine which forecast parameter is uncertain. "
                    f"Ask the user (or infer from the query) which one of these indexed parameters "
                    f"should be treated as uncertain and query again: {candidates}.")
        return "Feedback from internal tools: \n" + feedback

    n_scenarios = int(n_scenarios)
    base = queried_model_dict['model class'].clone()
    first_stage = _first_stage_variable_names(base, queried_model_dict, queried_components)
    if any(vardata.value is None
           for name in first_stage for vardata in getattr(base, name).values()):
        _solve_quietly(base)
    sense = _active_objective(base).sense
    deterministic_schedule = {name: {idx: pe.value(getattr(base, name)[idx])
                                     for idx in getattr(base, name)}
                              for name in first_stage}

    template = base.clone()
    records = _soften_recourse_constraints(template, first_stage, violation_penalty)
    binding_candidates = _binding_candidates(template)
    forecast = [pe.value(getattr(base, param_name)[idx])
                for idx in _trajectory_indexes(getattr(base, param_name))]
    rng = np.random.default_rng(seed)
    scenarios = _ar1_lognormal_scenarios(forecast, n_scenarios, phi, sigma_eps, rng)

    # wet/dry terciles by total realised forecast over the horizon
    totals = scenarios.sum(axis=1)
    order = np.argsort(totals)
    tercile = n_scenarios // 3
    dry_set = set(order[:tercile].tolist())
    wet_set = set(order[-tercile:].tolist()) if tercile else set()

    # --- EEV: the deterministic schedule evaluated against the scenarios -----
    eev_model = template.clone()
    for name in first_stage:
        var = getattr(eev_model, name)
        for idx in var:
            var[idx].fix(deterministic_schedule[name][idx])
    eev_scenarios = _evaluate_fixed_schedule(eev_model, param_name, scenarios, records, tol)
    eev_stats = _violation_statistics(eev_scenarios)
    eev = eev_stats['expected_cost']

    # --- WS: per-scenario wait-and-see optima (free first stage) -------------
    ws_model = template.clone()
    ws_objective = _active_objective(ws_model)
    ws_costs, ws_schedules = [], []
    binding_counts = {c['label']: {'all': 0, 'wet': 0, 'dry': 0} for c in binding_candidates}
    primary_var = first_stage[0]
    primary_indexes = _trajectory_indexes(getattr(ws_model, primary_var))
    for s, path in enumerate(scenarios):
        _set_trajectory(ws_model, param_name, path)
        if _solve_quietly(ws_model) != TerminationCondition.optimal:
            continue
        ws_costs.append(pe.value(ws_objective))
        ws_schedules.append([pe.value(getattr(ws_model, primary_var)[idx]) for idx in primary_indexes])
        for candidate in binding_candidates:
            if _is_binding(ws_model, candidate, tol):
                binding_counts[candidate['label']]['all'] += 1
                if s in wet_set:
                    binding_counts[candidate['label']]['wet'] += 1
                elif s in dry_set:
                    binding_counts[candidate['label']]['dry'] += 1
    ws = float(np.mean(ws_costs)) if ws_costs else float('nan')

    # --- RP: the extensive form (shared first stage, scenario blocks) --------
    ef = pe.ConcreteModel(name=f"extensive_form_{queried_model}")
    ef.scen = pe.Block(range(n_scenarios))
    scenario_objectives = []
    for s in range(n_scenarios):
        sub = template.clone()
        _set_trajectory(sub, param_name, scenarios[s])
        ef.scen[s].transfer_attributes_from(sub)
        obj_s = next(ef.scen[s].component_objects(pe.Objective))
        obj_s.deactivate()
        scenario_objectives.append(obj_s.expr)
    ef.nonanticipativity = pe.ConstraintList()
    for name in first_stage:
        reference = getattr(ef.scen[0], name)
        for s in range(1, n_scenarios):
            other = getattr(ef.scen[s], name)
            for idx in reference:
                ef.nonanticipativity.add(other[idx] == reference[idx])
    ef.expected_cost = pe.Objective(expr=sum(scenario_objectives) / n_scenarios, sense=sense)
    ef_termination = _solve_quietly(ef, time_limit=300)
    if ef_termination not in [TerminationCondition.optimal, TerminationCondition.maxTimeLimit]:
        feedback = (f"Error: The two-stage extensive form could not be solved ({ef_termination}). "
                    f"Try fewer scenarios (n_scenarios) or check the model.")
        return "Feedback from internal tools: \n" + feedback
    rp = pe.value(ef.expected_cost)
    hedged_schedule = {name: {idx: pe.value(getattr(ef.scen[0], name)[idx])
                              for idx in getattr(ef.scen[0], name)}
                       for name in first_stage}

    # --- risk of the hedged schedule under the SAME scenarios ----------------
    hedged_model = template.clone()
    for name in first_stage:
        var = getattr(hedged_model, name)
        for idx in var:
            var[idx].fix(hedged_schedule[name][idx])
    hedged_scenarios = _evaluate_fixed_schedule(hedged_model, param_name, scenarios, records, tol)
    hedged_stats = _violation_statistics(hedged_scenarios)

    if sense == pe.minimize:
        vss, evpi = eev - rp, rp - ws
    else:
        vss, evpi = rp - eev, ws - rp

    # deterministic-schedule violation rate, wet vs dry terciles
    def _tercile_violation_rate(per_scenario, member_set):
        member = [r for s, r in enumerate(per_scenario) if s in member_set and r['solved']]
        return (sum(1 for r in member if r['violations']) / len(member)) if member else float('nan')

    det_wet_rate = _tercile_violation_rate(eev_scenarios, wet_set)
    det_dry_rate = _tercile_violation_rate(eev_scenarios, dry_set)

    # per-hour comparison table for the primary first-stage variable
    ws_array = np.array(ws_schedules) if ws_schedules else np.zeros((0, len(primary_indexes)))
    schedule_lines = [f"    {'idx':>5} | {'deterministic':>13} | {'hedged':>10} | WS p10-p90"]
    for j, idx in enumerate(primary_indexes):
        det_value = deterministic_schedule[primary_var][idx]
        hedged_value = hedged_schedule[primary_var][idx]
        if ws_array.size:
            p10, p90 = np.percentile(ws_array[:, j], [10, 90])
            spread = f"{p10:8.3f} - {p90:8.3f}"
        else:
            spread = "n/a"
        schedule_lines.append(f"    {str(idx):>5} | {det_value:13.3f} | {hedged_value:10.3f} | {spread}")
    det_total = sum(deterministic_schedule[primary_var].values())
    hedged_total = sum(hedged_schedule[primary_var].values())

    binding_lines = []
    n_ws = len(ws_costs)
    for label, counts in sorted(binding_counts.items(), key=lambda kv: kv[1]['all'], reverse=True):
        if counts['all'] == 0:
            continue
        wet_share = counts['wet'] / max(len(wet_set), 1)
        dry_share = counts['dry'] / max(len(dry_set), 1)
        binding_lines.append(f"    {label}: binding in {counts['all'] / n_ws:.0%} of scenario optima "
                             f"(wet tercile {wet_share:.0%}, dry tercile {dry_share:.0%})")
        if len(binding_lines) >= 10:
            break

    duration = time.time() - start_time
    time_limit_note = (" NOTE: the extensive form hit the time limit; the hedged schedule is the best "
                       "feasible one found, not proven optimal." if ef_termination == TerminationCondition.maxTimeLimit else "")

    feedback = (
        f"Two-stage stochastic (hedging) analysis of {queried_model}.\n"
        f"Provenance — every number below comes from exactly this computation: uncertain parameter "
        f"{param_name}, AR(1) log-normal scenarios around the deterministic forecast (phi={phi}, "
        f"sigma_eps={sigma_eps}, mean-preserving), {n_scenarios} scenarios, seed {seed}; first-stage "
        f"(here-and-now) variables {first_stage}; violation penalty {violation_penalty:g} per unit; "
        f"solver Gurobi; runtime {duration:.1f}s.{time_limit_note}\n\n"
        f"1. Stochastic-programming diagnostics (objective units of the model, penalty included):\n"
        f"    RP  (two-stage optimum, hedged schedule):        {rp:.6g}\n"
        f"    EEV (deterministic schedule under uncertainty):  {eev:.6g}\n"
        f"    WS  (average of per-scenario perfect-forecast optima): {ws:.6g}\n"
        f"    VSS = EEV - RP = {vss:.6g}  -> what using the hedged schedule instead of the "
        f"deterministic one is worth on average.\n"
        f"    EVPI = RP - WS = {evpi:.6g}  -> what a perfect forecast would be worth on top of hedging.\n\n"
        f"2. Risk comparison under the SAME {n_scenarios} scenarios:\n"
        f"    deterministic schedule: violates a (soft) limit in {eev_stats['p_violation']:.1%} of scenarios "
        f"(wet tercile {det_wet_rate:.1%}, dry tercile {det_dry_rate:.1%})\n"
        f"    hedged schedule:        violates a (soft) limit in {hedged_stats['p_violation']:.1%} of scenarios\n\n"
        f"3. Schedules, {primary_var} per index (deterministic vs hedged; WS spread = 10th-90th "
        f"percentile of the {n_ws} per-scenario optima — wide spread means that hour is "
        f"scenario-dependent):\n" + "\n".join(schedule_lines) + "\n"
        f"    totals: deterministic {det_total:.3f}, hedged {hedged_total:.3f}\n\n"
        f"4. Hard constraints binding in the per-scenario optima (why the hedged schedule acts early: "
        f"if a capacity binds in wet scenarios, reacting later is impossible):\n"
        + ("\n".join(binding_lines) if binding_lines else "    (none binding)") + "\n\n"
        f"Guidance for the explainer: explain the hedged schedule by (a) pointing to the indexes where "
        f"it differs most from the deterministic one, (b) linking the extra/earlier action to the "
        f"violation frequencies and to capacity constraints that bind in wet scenarios, and (c) "
        f"quantifying whether it is worth it with VSS (and EVPI for 'what would a better forecast be "
        f"worth'). Every quantitative claim about uncertainty MUST come from the numbers above.\n"
    )
    return "Feedback from internal tools: \n" + feedback


