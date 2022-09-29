# Spatio-Temporal Data Interpolation Models

This directory contains two major implementations of spatio-temporal interpolation models [Inductive Graph Neural Network for Spatiotemporal Kriging](https://ojs.aaai.org/index.php/AAAI/article/view/16575)(IGNNK) and [Deep Learning based spatio-temporal time series prediction framework](https://www.nature.com/articles/s41598-020-79148-7). 

## Dataset
Air Quality Dataset from Indian city of Pune. Input data is preprocessed to remove outliers and fill all the missing values.


## Inductive Graph Neural Network for Spatiotemporal Kriging (IGNNK)
The Inductive Graph Neural Network Kriging (IGNNK) model is developed to recover data for unsampled sensors on a network/graph structure. IGNNK generate random subgraphs as samples and the corresponding adjacency matrix for each sample. By reconstructing all signals on each sample subgraph, IGNNK can effectively learn the spatial message passing mechanism. In addition, learned model can be successfully transferred to the same type of kriging tasks on an various dataset.

### Dependencies
- Numpy
- Pandas
- Matplotlib
- Torch
- Geopandas
- Scipy
- Seaborn
- Joblib
- Geopy
- Scikit-learn
- [IGNNK](https://github.com/Kaimaoge/IGNNK)

### Files
`GRIN_Imputed_Pune_Data.csv` file: Pune pm2.5 data after performing imputation using GRIN model.\
`IGNNK_Interpolation_pm2p5.ipynb` file: Jupyter notebook contains pytorch implementation of IGNNK interpolation.\
`IGNNK_Model_Training_pm2p5.ipynb` file: Jupyter notebook contains pytorch implementation of IGNNK training.\
`IGNNKKriging.yml` file: Configuration file for IGNNK implementation.

## Deep Learning Framework
Originally, deep learning framework is developed for spatio-temporal prediction of climate and environmental data. This approach has two key advantages. First, the decomposition of the spatio-temporal signal into fixed temporal bases and stochastic spatial coefficients. Second, Deep learning algorithms are particularly well suited to solve this problem because of their automatic feature representation learning. Furthermore, this framework is able to capture non-linear patterns in the data, as it models spatio-temporal fields as a combination of products of temporal bases by spatial coefficients. This framework succeeds at recovering spatial, temporal and spatio-temporal dependencies in both simulated and real-world data. Furthermore, this framework can eventually be generalized to study other climate fields and environmental spatio-temporal phenomena—e.g. air pollution or wind speed—or to solve missing data imputation problems in spatio-temporal datasets collected by satellites for earth observation or resulting from climate models.

### Dependencies
- Numpy
- Pandas
- Geopandas
- Tensorflow
- Keras
- Scikit-learn
- Matplotlib

### Files
`Deep_Learning_Interpolation.ipynb` file: Tensorflow implementation of deep learning framework interpolation.\
`Deep_Learning_Testing.ipynb` file: Tensorflow implementation of deep learning framework training.\
`Deep_Learning_ST.yml` file: Configuration file for deep learning framework implementation.\
`GRIN_Imputed_Pune_Data.csv` file: Pune pm2.5 data after performing imputation using GRIN model



## References
<a id="1">[1]</a> 
Wu, Y., Zhuang, D., Labbe, A., & Sun, L. (2021). Inductive Graph Neural Networks for Spatiotemporal Kriging. Proceedings of the AAAI Conference on Artificial Intelligence, 35(5), 4478-4485. https://doi.org/10.1609/aaai.v35i5.16575.

<a id="1">[2]</a> 
Amato, F., Guignard, F., Robert, S. et al. A novel framework for spatio-temporal prediction of environmental data using deep learning. Sci Rep 10, 22243 (2020). https://doi.org/10.1038/s41598-020-79148-7.
