#dd-cell
import pandas as pd

df_customers = inputs['Customer']
customers = df_customers['CustomerID'].values
df_customers.set_index('CustomerID', inplace=True)

df_offers = inputs['Offer'] 
offers = df_offers['OfferID'].values
df_offers.set_index('OfferID', inplace=True)

df_channels = inputs['Channel'] 
channels = df_channels['ChannelID'].values
df_channels.set_index('ChannelID', inplace=True)


df_probability = inputs['Probability']
candidates = []
for index, row in inputs['Probability'].iterrows():
    candidate = {}
    candidates.append( (row['CustomerID'], row['OfferID'],row['ChannelID']) )
df_probability.set_index(["CustomerID", "OfferID","ChannelID"], inplace = True)

df_revenue = inputs['Revenue']
df_revenue.set_index(["CustomerID", "OfferID","ChannelID"], inplace = True)
#dd-cell
from docplex.mp.model import Model
mdl = Model("RetailPlanning")

selected = mdl.binary_var_dict(candidates, name="selected")
df_selected = pd.DataFrame({'selected': selected})
df_selected.index.names=['CustomerID', 'OfferID', 'ChannelID']
df_selected.head()
#dd-cell
expected_revenue = mdl.sum( selected[(cu, of, ch)] * df_revenue.Revenue[cu][of][ch] for (cu, of, ch) in candidates) 
mdl.add_kpi(expected_revenue, "Expected revenue");
mdl.maximize(expected_revenue);
#dd-cell
nb_offer = mdl.sum( selected[(cu, of, ch)] for (cu, of, ch) in candidates) 
mdl.add_kpi(nb_offer, "Nb Offer");
#dd-cell
for customer in customers:
    mdl.add_constraint( mdl.sum( selected[(cu, of, ch)] for (cu, of, ch) in candidates if cu == customer) <= 1)
#dd-cell
mdl.print_information()
#dd-cell
assert mdl.solve(log_output=True), "!!! Solve of the model fails"
#dd-cell
mdl.report()
#dd-cell
df_selected = df_selected.selected.apply(lambda v: v.solution_value)
print(df_selected)
#dd-cell
df_selected = df_selected.reset_index()
outputs['selected'] = df_selected
#dd-cell

