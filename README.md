# House-Price-Prediction

This repository was a project aimed at predicting sale prices of houses in Ames, Iowa from the years of 2006-2010. The dataset was supplied by [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data). By using different models including stacked regression to build an ensembled final model, we were able to achieve a score of 0.11591 which placed us at a rank of 640 on the Kaggle leaderboard at the time. Feel free to clone this to your own machine and improving on our methods.

Take a look at our blog for more details: [Which house shall we invest in?](https://nycdatascience.com/blog/student-works/which-real-estate-shall-we-invest-in/)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installing

In order to run the ipynb files, make sure to have the following libraries:
* [Jupyter](https://jupyter.readthedocs.io/en/latest/install.html)
* [Numpy](https://docs.scipy.org/doc/numpy/user/install.html)
* [Matplotlib](https://matplotlib.org/)
* [Scikit-Learn](https://scikit-learn.org/stable/)

Then, clone this repository into your local machine and try manipulating our tools yourself! The dataset is included in this repo.

```
git clone https://github.com/wonchankim97/House-Price-Prediction
```

## Running the tests

To modify the results, there are several ways to try and output results (for better or worse). The scripts inside the module can be altered according to various features or models you believe were ommitted or superfluous.

Even if the features or models aren't changed themselves, you can look into changing the weights that are present for various models in our final ensembled model.

Finally, in order to evaluate the model's performance, there are several segments in the jupyter notebook, predict.ipynb, where you can uncomment the train test split line as well as the RMSE code at the end. This is in lieu of submitting to Kaggle and finding out your score of course.

## Authors

* **Alyssa Wei** - [Alyssa Wei](https://github.com/AlyssaWei)
* **Hyelee Lee** - [Hyelee](https://github.com/hayley01145553)
* **Kisoo Cho** - [Necronia](https://github.com/necronia)
* **Wonchan Kim** - [Wonchan Kim](https://github.com/wonchankim97)
