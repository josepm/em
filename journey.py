"""
__author__: josep ferrandiz

Input data format:
- DF with columns: customer_id, timestamp, state (URL/activity), other flags?
- Processing
  - sort by time stamp
  - flags are derived from state visits and change future states
  - Group by customer_id, shift and count transitions
  - Number the states from 0 to D with,
    - D - 2: C state
    - D - 1: L state
  - Output: DF with customer_id, init_state, i_state, j_state, count.
            - 'i_state' is the state we are in and 'j_state' the state we transition to
            - For every customer in the data, the output DF will have:
              - constant value for init_state
"""

import em
import sql
import pre_processing as pp
from config.logger import logger


if __name__ == "__main__":
    n_abs = 2                                      # number of absorbing states
    sql_data, states_map = sql.get_data()
    n_data_ = pp.set_data(sql_data)                # journey_id, init_state, i_state, j_state, count
    n_data_['journey_id'] = n_data_['journey_id'].astype('string')
    for c in ['init_state', 'i_state', 'j_state', 'count']:
        n_data_[c] = n_data_[c].astype(int)

    n_states = max(n_data_['i_state'].nunique(), n_data_['j_state'].nunique())   # unique states
    n_data = n_data_.groupby('journey_id').apply(em.check_counts, n_states=n_states, n_abs=n_abs).reset_index(drop=True)     # check input data
    j_obj = em.process_segment(n_states, n_abs, n_data, weighted=True, min_var=0.25, stochastic=True)

