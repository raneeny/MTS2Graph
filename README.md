# MTS2Graph
MTS2Graph introduces a new framework for interpreting multivariate time series data by extracting and clustering the input representative patterns that highly activate CNN neurons. This way, we identify each signal’s role and dependencies, considering all possible combinations of signals in the MTS input. Then, we construct a graph that captures the temporal relationship between the extracted patterns for each layer. An effective graph merging strategy finds the connection of each node to the previous layer’s nodes. Finally, a graph embedding algorithm generates new representations of the created interpretable time-series features. To evaluate the performance of our proposed framework, we run extensive experiments on eight datasets of Baydogan’s archive. The experiments indicate the benefit of our time-aware graph-based representation in MTS classification while enriching them with more interpretability.


Mustafa Baydogan's MTS datasets can be obtained from: http://www.mustafabaydogan.com/files/viewdownload/20-data-sets/69-multivariate-time-series-classification-data-sets-in-matlab-format.html or contact [Mustafa Baydogan](http://www.mustafabaydogan.com/contact/) 


## Code
The code is divided as follows:

* The Example_dataset.py python file contains the necessary code to run an experiment.
* The Clustering.py contains the necessary functions to apply k-shape clustering.
* The ConvNet_Model.py contains the CNN-based neural network model.
* The High_Activated_Filters_CNN.py contains the necessary functions to extract the MHAPs and build the graph.
* The Graph_embading.py contains the necessary functions to apply graph embedding.
* The Data_Preprocessing.py contains the necessary functions to preprocess the data.

To run a model on a dataset with the default parameters, you should issue the following command:

```bash
To run the code using python Example_dataset.py -f <foldar_name_data> -d <dataset name>
```

To set other hyperparameters

-c or --cluster_numbers: Sets the cluster numbers as a comma-separated list (e.g., 35,25,15).

-s or --segment_length: Sets the segment lengths (e.g.,10).

-a or --activation_threshold: Sets the activation threshold (e.g., 0.95).

-e or --embedding_size: Sets the embedding size (e.g., 100).

## Experement setup
The FCN architecture used in MTS2Graph is comprised of three convolutional blocks with 32, 64, and 128 filters of varying lengths, ReLU activation functions, and batch normalization for regularization. 
For all datasets, we used $80\%$ for training, 10$\%$ for validation, and 10$\%$ for test data. Each experiment is run for $500$ epochs, with $100$ embedding size, a segment length of size $10$, activation_threshold of 0.95 and cluster_numbers of 35,25,15.

## Prerequisites
The python packages needed are:
* numpy
* pandas
* sklearn
* scipy
* matplotlib
* tensorflow
* keras
