import tensorflow as tf
# generate datasets from data
def identify_data_types(input_list):
  numeric_features = []
  categorical_features = []
  embedding_list = []
  for feat in input_list:
    size = feat.shape[-1]
    if size > 1:
      embedding_list.append(feat.name)
    else:
      if feat.dtype == tf.string:
        categorical_features.append(feat.name)
      else:
        numeric_features.append(feat.name)
  return numeric_features, categorical_features, embedding_list