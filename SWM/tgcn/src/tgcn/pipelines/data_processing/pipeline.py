from kedro.pipeline import Pipeline, node

from .nodes import *

def create_pipeline(**kwargs):
    return Pipeline(
[
     #node(
      #func=data_preprocess,
      #inputs=['data','params:initial_date','params:final_date','params:latitude_boundary','params:longitude_boundary'],
      #outputs='preprocessed_data',
      #name='data_preprocess',),
      #node(
      #func=add_column_h3id,
      #inputs='preprocessed_data',
      #outputs='preprocessed_with_h3',
      #name='add_column_h3id',),
      #node(
      #func=data_filtering,
      #inputs=['preprocessed_with_h3','hex_loc'],
      #outputs='filtered_data',
      #name='data_filtering',),
      #node(
      #func=data_split_hourlybasis,
      #inputs=['filtered_data','hex_loc','params:initial_datetime','params:final_datetime'],
      #outputs='hourly_data',
      #name='data_split_hourlybasis',),
      #node(
      #func=get_adjacency_matrix,
      #inputs='hex_loc',
      #outputs='matrix',
      #name='get_adjacency_matrix',),
      #node(
      func=compare_passenger_capacity,
      inputs='filtered_data',
      outputs='plot',
      name='plot',),
      node(
      func=train_test_split,
      inputs=['hourly_data','params:train_portion'],
      outputs=['train_data','test_data','plot'],
      name='train_test_split',),
      node(
      func=scale_data,
      inputs=['train_data','test_data'],
      outputs=['train_scaled','test_scaled'],
      name='scale_data',),
      node(
      func=sequence_data_preparation1,
      inputs=['params:seq_len','params:pre_len','train_scaled','test_scaled'],
      outputs=['trainX','trainY','testX','testY'],
      name='sequential_data_preparation',),
      node(
      func=model_initialize,
      inputs=['matrix','hex_loc','params:learning_rate','trainX','trainY','testX','testY'],
      outputs='y_pred_testX',
      name='model_define_initialize',),
       ]
      )
