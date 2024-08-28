# Joint Regularized Deep Neural Network Method for Network Estimation (JRDNN-KM)

## Abstract
Network estimation is a pivotal component in the analysis of single-cell transcriptomic data,
 offering insights into the complex interactions among genes at single-cell resolution. 
Our proposed method, JRDNN-KM, utilizes a joint regularized deep neural network combined with Mahalanobis distance-based K-means clustering to estimate multiple networks for various cell subgroups simultaneously. 
This approach effectively handles unknown cellular heterogeneity, zero-inflation, and complex nonlinear gene relationships.

**Advanced Handling of Data Characteristics**: Effectively addresses cellular heterogeneity, zero-inflation, and nonlinear gene relationships.

## Installation
Ensure that you have Python 3.8 installed on your machine. You can install the required packages using the following commands:

```bash
pip install numpy==1.21.2
pip install torch==1.9.1
