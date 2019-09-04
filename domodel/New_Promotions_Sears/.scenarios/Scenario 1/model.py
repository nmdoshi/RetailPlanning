from docplex.mp.model import *
from docplex.mp.utils import *
from docloud.status import JobSolveStatus
from docplex.mp.conflict_refiner import ConflictRefiner, VarUbConstraintWrapper, VarLbConstraintWrapper
import time
import sys
import operator

import pandas as pd
import numpy as np

import codecs
import sys

# Handle output of unicode strings
if sys.version_info[0] < 3:
    sys.stdout = codecs.getwriter('utf8')(sys.stdout)


# Label constraint
def helper_add_labeled_cplex_constraint(mdl, expr, label, context=None, columns=None):
    global expr_counter
    if isinstance(expr, bool):
        pass  # Adding a trivial constraint: if infeasible, docplex will raise an exception it is added to the model
    else:
        expr.name = '_L_EXPR_' + str(len(expr_to_info) + 1)
        if columns:
            ctxt = ", ".join(str(getattr(context, col)) for col in columns)
        else:
            if context:
                ctxt = context.Index if isinstance(context.Index, str) is not None else ", ".join(context.Index)
            else:
                ctxt = None
        expr_to_info[expr.name] = (label, ctxt)
    mdl.add(expr)

def helper_get_index_names_for_type(dataframe, type):
    if not is_pandas_dataframe(dataframe):
        return None
    return [name for name in dataframe.index.names if name in helper_concept_id_to_index_names_map.get(type, [])]


helper_concept_id_to_index_names_map = {
    'cItem': ['id_of_Customer'],
    'Customer': ['id_of_Customer']}


# Data model definition for each table
# Data collection: list_of_Customer ['CostToServ', 'Offer_id', 'Revenue', 'line']

# Create a pandas Dataframe for each data table
list_of_Customer = inputs[u'Customer']
list_of_Customer = list_of_Customer[[u'CostToServ', u'Offer id', u'Revenue']].copy()
list_of_Customer.rename(columns={u'CostToServ': 'CostToServ', u'Offer id': 'Offer_id', u'Revenue': 'Revenue'}, inplace=True)

# Set index when a primary key is defined
list_of_Customer.index.name = 'id_of_Customer'






def build_model():
    mdl = Model()

    # Definition of model variables
    list_of_Customer['selectionVar'] = mdl.binary_var_list(len(list_of_Customer))


    # Definition of model
    # Objective cMaximizeGoalSelect-
    # Combine weighted criteria: 
    # 	cMaximizeGoalSelect cMaximizeGoalSelect{
    # 	cSingleCriterionGoal.goalFilter = null,
    # 	cSingleCriterionGoal.numericExpr = total cSelection[Customer] / Customer / Revenue,
    # 	cScaledGoal.scaleFactorExpr = 1} with weight 5.0
    list_of_Customer['conditioned_Revenue'] = list_of_Customer.selectionVar * list_of_Customer.Revenue
    agg_Customer_conditioned_Revenue_SG1 = mdl.sum(list_of_Customer.conditioned_Revenue)
    
    kpis_expression_list = [
        (1, 1.0, agg_Customer_conditioned_Revenue_SG1, 1, 0, u'total Revenue of Customers over all selections')]
    custom_code.update_goals_list(kpis_expression_list)
    
    for _, kpi_weight, kpi_expr, kpi_factor, kpi_offset, kpi_name in kpis_expression_list:
        mdl.add_kpi(kpi_weight * ((kpi_expr * kpi_factor) - kpi_offset), publish_name=kpi_name)
    
    mdl.maximize(sum([kpi_sign * kpi_weight * ((kpi_expr * kpi_factor) - kpi_offset) for kpi_sign, kpi_weight, kpi_expr, kpi_factor, kpi_offset, kpi_name in kpis_expression_list]))
    
    # [ST_1] Constraint : cIterativeRelationalConstraint_cIterativeRelationalConstraint
    # For each Customer selection, Offer id of selected Customer is less than or equal to 2
    # Label: CT_1_For_each_Customer_selection__Offer_id_of_selected_Customer_is_less_than_or_equal_to_2
    list_of_Customer['conditioned_Offer_id'] = list_of_Customer.selectionVar * list_of_Customer.Offer_id
    for row in list_of_Customer.itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.conditioned_Offer_id <= 2, u'For each Customer selection, Offer id of selected Customer is less than or equal to 2', row)
    
    # [ST_2] Constraint : cGlobalRelationalConstraint_cGlobalRelationalConstraint
    # total CostToServ of Customers over all selections is less than 200000
    # Label: CT_2_total_CostToServ_of_Customers_over_all_selections_is_less_than_200000
    list_of_Customer['conditioned_CostToServ'] = list_of_Customer.selectionVar * list_of_Customer.CostToServ
    agg_Customer_conditioned_CostToServ_lhs = mdl.sum(list_of_Customer.conditioned_CostToServ)
    helper_add_labeled_cplex_constraint(mdl, agg_Customer_conditioned_CostToServ_lhs <= -0.001 + 200000, u'total CostToServ of Customers over all selections is less than 200000')


    return mdl


def solve_model(mdl):
    mdl.parameters.timelimit = 300.0
    # Call to custom code to update parameters value
    custom_code.update_solver_params(mdl.parameters)
    # Update parameters value based on environment variables definition
    cplex_param_env_prefix = 'ma.cplex.'
    cplex_params = [name.qualified_name for name in mdl.parameters.generate_params()]
    for param in cplex_params:
        env_param = cplex_param_env_prefix + param
        param_value = get_environment().get_parameter(env_param)
        if param_value:
            # Updating parameter value
            print("Updated value for parameter %s = %s" % (param, param_value))
            parameters = mdl.parameters
            for p in param.split('.')[1:]:
                parameters = parameters.__getattribute__(p)
            parameters.set(param_value)

    msol = mdl.solve(log_output=True)
    if not msol:
        print("!!! Solve of the model fails")
        if mdl.get_solve_status() == JobSolveStatus.INFEASIBLE_SOLUTION or mdl.get_solve_status() == JobSolveStatus.INFEASIBLE_OR_UNBOUNDED_SOLUTION:
            crefiner = ConflictRefiner()
            conflicts = crefiner.refine_conflict(model, log_output=True)
            export_conflicts(conflicts)
    print('Solve status: %s' % mdl.get_solve_status())
    mdl.report()
    return msol


expr_to_info = {}


def export_conflicts(conflicts):
    # Display conflicts in console
    print('Conflict set:')
    list_of_conflicts = pd.DataFrame(columns=['constraint', 'context', 'detail'])
    for conflict, index in zip(conflicts, range(len(conflicts))):
        st = conflict.status
        ct = conflict.element
        label, context = expr_to_info.get(conflict.name, ('N/A', conflict.name))
        label_type = type(conflict.element)
        if isinstance(conflict.element, VarLbConstraintWrapper) \
                or isinstance(conflict.element, VarUbConstraintWrapper):
            ct = conflict.element.get_constraint()

        # Print conflict information in console
        print("Conflict involving constraint: %s, \tfor: %s -> %s" % (label, context, ct))
        list_of_conflicts = list_of_conflicts.append(pd.DataFrame({'constraint': label, 'context': str(context), 'detail': ct},
                                                                  index=[index], columns=['constraint', 'context', 'detail']))

    # Update of the ``outputs`` dict must take the 'Lock' to make this action atomic,
    # in case the job is aborted
    global output_lock
    with output_lock:
        outputs['list_of_conflicts'] = list_of_conflicts


def export_solution(msol):
    start_time = time.time()
    list_of_Customer_solution = pd.DataFrame(index=list_of_Customer.index)
    list_of_Customer_solution['selectionVar'] = msol.get_values(list_of_Customer.selectionVar.values)

    # Update of the ``outputs`` dict must take the 'Lock' to make this action atomic,
    # in case the job is aborted
    global output_lock
    with output_lock:
        outputs['list_of_Customer_solution'] = list_of_Customer_solution.reset_index()
        custom_code.post_process_solution(msol, outputs)

    elapsed_time = time.time() - start_time
    print('solution export done in ' + str(elapsed_time) + ' secs')
    return


# Import custom code definition if module exists
try:
    from custom_code import CustomCode
    custom_code = CustomCode(globals())
except ImportError:
    # Create a dummy anonymous object for custom_code
    custom_code = type('', (object,), {'preprocess': (lambda *args: None),
                                       'update_goals_list': (lambda *args: None),
                                       'update_model': (lambda *args: None),
                                       'update_solver_params': (lambda *args: None),
                                       'post_process_solution': (lambda *args: None)})()

# Custom pre-process
custom_code.preprocess()

print('* building wado model')
start_time = time.time()
model = build_model()

# Model customization
custom_code.update_model(model)

elapsed_time = time.time() - start_time
print('model building done in ' + str(elapsed_time) + ' secs')

print('* running wado model')
start_time = time.time()
msol = solve_model(model)
elapsed_time = time.time() - start_time
print('model solve done in ' + str(elapsed_time) + ' secs')
if msol:
    export_solution(msol)
