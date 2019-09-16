from docplex.mp.model import *
from docplex.mp.utils import *
from docloud.status import JobSolveStatus
from docplex.mp.conflict_refiner import ConflictRefiner, VarUbConstraintWrapper, VarLbConstraintWrapper
from docplex.mp.relaxer import Relaxer
import time
import sys
import operator

import pandas as pd
import numpy as np
import math

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

def helper_get_column_name_for_property(property):
    return helper_property_id_to_column_names_map.get(property, 'unknown')


def helper_get_index_names_for_type(dataframe, type):
    if not is_pandas_dataframe(dataframe):
        return None
    return [name for name in dataframe.index.names if name in helper_concept_id_to_index_names_map.get(type, [])]


helper_concept_id_to_index_names_map = {
    'cItem': ['id_of_Product'],
    'Product': ['id_of_Product']}
helper_property_id_to_column_names_map = {
    'Product.Profit': 'Profit',
    'Product.Promotion': 'Promotion',
    'Product.CCost': 'CCost',
    'Product.addquantitycost': 'addquantitycost',
    'Product.Id': 'Id',
    'Product.NOUB': 'NOUB'}


# Data model definition for each table
# Data collection: list_of_Product ['Id', 'CCost', 'Promotion', 'NOUB', 'Profit', 'addquantitycost']

# Create a pandas Dataframe for each data table
list_of_Product = inputs[u'Product']
list_of_Product = list_of_Product[[u'Id', u'CCost', u'Promotion', u'NOUB', u'Profit', u'addquantitycost']].copy()
list_of_Product.rename(columns={u'Id': 'Id', u'CCost': 'CCost', u'Promotion': 'Promotion', u'NOUB': 'NOUB', u'Profit': 'Profit', u'addquantitycost': 'addquantitycost'}, inplace=True)

# Set index when a primary key is defined
list_of_Product.set_index('Id', inplace=True)
list_of_Product.sort_index(inplace=True)
list_of_Product.index.name = 'id_of_Product'






def build_model():
    mdl = Model()

    # Definition of model variables
    list_of_Product['allocationVar'] = mdl.integer_var_list(len(list_of_Product))
    list_of_Product['selectionVar'] = mdl.binary_var_list(len(list_of_Product))


    # Definition of model
    # Objective cMaximizeSelections-
    # Combine weighted criteria: 
    # 	cMaximizeSelections cMaximizeSelections 1.2{
    # 	cMaximizeSelections.selection = cSelection[Product],
    # 	cScaledGoal.scaleFactorExpr = 1,
    # 	cSingleCriterionGoal.goalFilter = null,
    # 	cSingleCriterionGoal.numericExpr = decisionPath(cSelection[Product])} with weight 9.0
    # 	cMaximizeGoalSelect cMaximizeGoalSelect 1.2{
    # 	cSingleCriterionGoal.numericExpr = total cAllocation[Product] / Product / Profit,
    # 	cScaledGoal.scaleFactorExpr = 1,
    # 	cSingleCriterionGoal.goalFilter = null} with weight 5.0
    agg_Product_selectionVar_SG1 = mdl.sum(list_of_Product.selectionVar)
    list_of_Product['conditioned_Profit'] = list_of_Product.allocationVar * list_of_Product.Profit
    agg_Product_conditioned_Profit_SG2 = mdl.sum(list_of_Product.conditioned_Profit)
    
    kpis_expression_list = [
        (1, 256.0, agg_Product_selectionVar_SG1, 1, 0, u'the number of Product selections'),
        (1, 16.0, agg_Product_conditioned_Profit_SG2, 1, 0, u'total Profit of Products over all allocations')]
    custom_code.update_goals_list(kpis_expression_list)
    
    for _, kpi_weight, kpi_expr, kpi_factor, kpi_offset, kpi_name in kpis_expression_list:
        mdl.add_kpi(kpi_weight * ((kpi_expr * kpi_factor) - kpi_offset), publish_name=kpi_name)
    
    mdl.maximize(sum([kpi_sign * kpi_weight * ((kpi_expr * kpi_factor) - kpi_offset) for kpi_sign, kpi_weight, kpi_expr, kpi_factor, kpi_offset, kpi_name in kpis_expression_list]))
    
    # [ST_1] Constraint : cBasicSelectionLimitMin_cGlobalRelationalConstraint
    # The number of Product selections  is greater than or equal to 1
    # Label: CT_1_The_number_of_Product_selections__is_greater_than_or_equal_to_1
    agg_Product_selectionVar_lhs = mdl.sum(list_of_Product.selectionVar)
    helper_add_labeled_cplex_constraint(mdl, agg_Product_selectionVar_lhs >= 1, u'The number of Product selections  is greater than or equal to 1')
    
    # [ST_2] Constraint : cLinkSelectionToAllocationConstraint_cIterativeRelationalConstraint
    # Synchronize selection with Product allocations
    # Label: CT_2_Synchronize_selection_with_Product_allocations
    join_Product = list_of_Product.reset_index().merge(list_of_Product.reset_index(), left_on=['id_of_Product'], right_on=['id_of_Product'], suffixes=('', '_right')).set_index(['id_of_Product'])
    groupbyLevels = ['id_of_Product']
    groupby_Product = join_Product.allocationVar.groupby(level=groupbyLevels[0]).sum().to_frame(name='allocationVar')
    list_of_Product_maxValueAllocation = pd.Series([100000000] * len(list_of_Product)).to_frame('maxValueAllocation').set_index(list_of_Product.index)
    join_Product_2 = list_of_Product.join(list_of_Product_maxValueAllocation.maxValueAllocation, how='inner')
    join_Product_2['conditioned_maxValueAllocation'] = join_Product_2.selectionVar * join_Product_2.maxValueAllocation
    join_Product_3 = groupby_Product.join(join_Product_2.conditioned_maxValueAllocation, how='inner')
    for row in join_Product_3.itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.allocationVar <= row.conditioned_maxValueAllocation, u'Synchronize selection with Product allocations', row)
    
    # [ST_3] Constraint : cLinkSelectionToAllocationConstraint_cIterativeRelationalConstraint
    # Synchronize selection with Product allocations
    # Label: CT_3_Synchronize_selection_with_Product_allocations
    join_Product = list_of_Product.reset_index().merge(list_of_Product.reset_index(), left_on=['id_of_Product'], right_on=['id_of_Product'], suffixes=('', '_right')).set_index(['id_of_Product'])
    groupbyLevels = ['id_of_Product']
    groupby_Product = join_Product.allocationVar.groupby(level=groupbyLevels[0]).sum().to_frame(name='allocationVar')
    list_of_Product_minValueAllocationForAssignment = pd.Series([0.10000000149011612] * len(list_of_Product)).to_frame('minValueAllocationForAssignment').set_index(list_of_Product.index)
    join_Product_2 = list_of_Product.join(list_of_Product_minValueAllocationForAssignment.minValueAllocationForAssignment, how='inner')
    join_Product_2['conditioned_minValueAllocationForAssignment'] = join_Product_2.selectionVar * join_Product_2.minValueAllocationForAssignment
    join_Product_3 = groupby_Product.join(join_Product_2.conditioned_minValueAllocationForAssignment, how='inner')
    for row in join_Product_3.itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.allocationVar >= row.conditioned_minValueAllocationForAssignment, u'Synchronize selection with Product allocations', row)
    
    # [ST_4] Constraint : cBasicSelectionLimitMin_cGlobalRelationalConstraint
    # The number of Product selections  is less than or equal to total Promotion over all Products
    # Label: CT_4_The_number_of_Product_selections__is_less_than_or_equal_to_total_Promotion_over_all_Products
    agg_Product_selectionVar_lhs = mdl.sum(list_of_Product.selectionVar)
    agg_Product_Promotion_rhs = sum(list_of_Product.Promotion)
    helper_add_labeled_cplex_constraint(mdl, agg_Product_selectionVar_lhs <= agg_Product_Promotion_rhs, u'The number of Product selections  is less than or equal to total Promotion over all Products')
    
    # [ST_5] Constraint : cGlobalRelationalConstraint_cGlobalRelationalConstraint
    # total CCost of Products over all allocations is less than 10000
    # Label: CT_5_total_CCost_of_Products_over_all_allocations_is_less_than_10000
    list_of_Product['conditioned_CCost'] = list_of_Product.allocationVar * list_of_Product.CCost
    agg_Product_conditioned_CCost_lhs = mdl.sum(list_of_Product.conditioned_CCost)
    helper_add_labeled_cplex_constraint(mdl, agg_Product_conditioned_CCost_lhs <= -0.001 + 10000, u'total CCost of Products over all allocations is less than 10000')
    
    # [ST_6] Constraint : cIterativeRelationalConstraint_cIterativeRelationalConstraint
    # For each Product, allocation is less than or equal to NOUB
    # Label: CT_6_For_each_Product__allocation_is_less_than_or_equal_to_NOUB
    for row in list_of_Product[list_of_Product.NOUB.notnull()].itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.allocationVar <= row.NOUB, u'For each Product, allocation is less than or equal to NOUB', row)
    
    # [ST_7] Constraint : cGlobalRelationalConstraint_cGlobalRelationalConstraint
    # total addquantitycost of Products over all allocations is less than or equal to 45000000
    # Label: CT_7_total_addquantitycost_of_Products_over_all_allocations_is_less_than_or_equal_to_45000000
    list_of_Product['conditioned_addquantitycost'] = list_of_Product.allocationVar * list_of_Product.addquantitycost
    agg_Product_conditioned_addquantitycost_lhs = mdl.sum(list_of_Product.conditioned_addquantitycost)
    helper_add_labeled_cplex_constraint(mdl, agg_Product_conditioned_addquantitycost_lhs <= 45000000, u'total addquantitycost of Products over all allocations is less than or equal to 45000000')


    return mdl


def solve_model(mdl):
    mdl.parameters.timelimit = 120
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
            label = 'Upper/lower bound conflict for variable: {}'.format(conflict.element._var)
            context = 'Decision variable definition'
            ct = conflict.element.get_constraint()

        # Print conflict information in console
        print("Conflict involving constraint: %s, \tfor: %s -> %s" % (label, context, ct))
        list_of_conflicts = list_of_conflicts.append({'constraint': label, 'context': str(context), 'detail': ct},
                                                     ignore_index=True)

    # Update of the ``outputs`` dict must take the 'Lock' to make this action atomic,
    # in case the job is aborted
    global output_lock
    with output_lock:
        outputs['list_of_conflicts'] = list_of_conflicts


def export_solution(msol):
    start_time = time.time()
    list_of_Product_solution = pd.DataFrame(index=list_of_Product.index)
    list_of_Product_solution['allocationVar'] = msol.get_values(list_of_Product.allocationVar.values)
    list_of_Product_solution = list_of_Product_solution.round({'allocationVar': 2})
    list_of_Product_solution['selectionVar'] = msol.get_values(list_of_Product.selectionVar.values)

    # Update of the ``outputs`` dict must take the 'Lock' to make this action atomic,
    # in case the job is aborted
    global output_lock
    with output_lock:
        outputs['list_of_Product_solution'] = list_of_Product_solution.reset_index()
        custom_code.post_process_solution(msol, outputs)

    elapsed_time = time.time() - start_time
    print('solution export done in ' + str(elapsed_time) + ' secs')
    return


# Instantiate CustomCode class if definition exists
try:
    custom_code = CustomCode(globals())
except NameError:
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
