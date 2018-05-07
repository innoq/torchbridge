# torchbridge
collection of lightweight tools to enable usage of functions from numpy, pandas and sklearn in the context of pytorch
without reinventing the wheel

## General approach

* Do as much as possible in numpy / pandas
  * Conversion to tensors just before feeding data into network
  * loss function and simple evaluations run on tensors, in more complex cases convert back to numpy

For preprocessing, there are tools to statically split CSVs and DataFrames (e.g. into train and test data) as well as a thin layer to simply use [sklearn scalers](http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing) on dataframes.

DFDataSet is a way to use pandas DataFrame with pytorch DataLoader. CrossValidationData can be used to wrap such a DataSet and to dynamically split it into train/validation on each epoch.