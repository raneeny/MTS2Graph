# MTS2Graph
MTS2Graph introduce a new framework for interpreting multivariate time series data by extracting and clustering the input representative patterns that highly activate CNN neurons. This way, we identify each signal’s role and dependencies, considering all possible combinations of signals in the MTS input. Then, we construct a graph that captures the temporal relationship between the extracted patterns for each layer. An effective graph merging strategy finds the connection of each node to the previous layer’s nodes. Finally, a graph embedding algorithm generates new representations of the created interpretable time-series features. To evaluate the performance of our proposed framework, we run extensive experiments on eight datasets of Baydogan’s archive. The experiments indicate the benefit of our time-aware graph-based representation in MTS classification while enriching them with more interpretability.

To Install the dataset datasets link:http://www.mustafabaydogan.com/files/viewcategory/20-data-sets.html

To run the code using python Train_example_dataset.py -f <foldar_name_data> -d <dataset name>
