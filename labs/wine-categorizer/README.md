# Wine categorizer
This repo is used as an introduction for a course in Machine Learning and touches slightly the steps of a ML project (see: https://ictresearchmethods.nl/machine-learning/). The case focuses on a [classification](https://www.geeksforgeeks.org/machine-learning/getting-started-with-classification/) problem, utilizing the [Wine dataset](https://archive.ics.uci.edu/dataset/109/wine) to determine to which cultivar a wine belongs to.

![house-price-predictor](https://github.com/Fontys/wine-categorizer/blob/main/BANNER.jpeg)
*Image by Stable Diffusion: a robot tasting wine*

This notebook is designed as a foundational starting point and does not strictly adhere to established best practices as it is meant as a learning opportunity. 

## 📚 Preparation
Please ensure that you are familiar with the following aspects in order to successfully work with this repo and notebook.
 - You know the basics of [scikit-learn](https://scikit-learn.org/stable/getting_started.html)
 - You know the basic structure of k-Nearest Neighbors (k-NN) [theoretically](https://www.geeksforgeeks.org/machine-learning/k-nearest-neighbours/) and in [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
 - You understand the reasons for making [train-test-splits](https://www.statology.org/understanding-train-test-splits/).
 - You understand the purpose and value of classification evaluation metrics like [accuracy, precision, recall and F1-scores](https://medium.com/@ml_dl_explained/an-overview-of-classification-model-metrics-8e25432d36ea)

## 🎯 Learning opportunities
The following aspects of machine learning are part of this example:
- A brief analysis of the data
- A k-Nearest Neighbours classifier model used for training
- Evaluating the model's performance using standardized evaluation metrics for classification problems
- An inference to see how the model would respond.

## 🤔 Considerations for improvement
This notebook is an example on how to get started, it is open for improvements and enhancements. Feel free to clone my work and use it to study and learn. Things to consider if you want to improve this work:
- Domain Understanding: who would be possible stakeholders for such a case? What is the added value of such a model? Which features of a wine determine the cultivar it belongs to?
- Data requirements: what kind of data is needed to identify the cultivar of the wine better?
- The model only uses two features for input, but more features can be considered. Based on what reasoning would you include which features as well?
- The constructor of the kNeighborsClassifier has an optional hyperparameter named n_neighbors, which by default is 5. How does the result change with different hyperparameter values?
- Design a prototype or a wire frame that uses the AI-model and can be used by stakeholders. What elements should such a prototype contain?

## ⭐ Citation & Star
If you use my work please cite and star ⭐ this repo. Thanks!
> Konings, Hans H.H.M. (2026) "Wine categorizer" GitHub: https://github.com/Fontys/wine-categorizer/
