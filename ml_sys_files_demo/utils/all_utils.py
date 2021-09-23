import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
plt.style.use("fivethirtyeight")

def prepare_data(df):
  """ split the data to input and output features

  Args:
      df (pd.dataframe): pandas dataframe containing the dataset

  Returns:
      pd.Datafrane, pd.Series: returns the input and output features as X and y
  """

  X = df.drop("y", axis=1)

  y = df["y"]

  return X,y


def save_model(model, filename):
  """ saves the model as pickle file to a location with given filename

  Args:
      model (python object): [model instance fitted to the dataset]
      filename (str): [name for the pickle file that will be created]
  """
  model_dir = "models"
  os.makedirs(model_dir, exist_ok=True)
  filepath = os.path.join(model_dir, filename)
  joblib.dump(model, filepath)



def save_plot(df, file_name, model):
  """ Visualize the model output after training

  Args:
      df (pd.dataframe): dataframe with features
      file_name (str): name for the image file to be saved as
      model (python obj): model instance
  """
  def _create_base_plot(df):
    """for creating a base plot with the data
    """

    df.plot(kind="scatter", x="X1", y="X2", c="y", s=100, cmap="winter")
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(10, 8)

  def _plot_decision_regions(X, y, classfier, resolution=0.02):
    """plotting the decision boundaries predicted that would separate the classes
    """

    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    X = X.values # as a array
    x1 = X[:, 0]
    x2 = X[:, 1]
    x1_min, x1_max = x1.min() -1 , x1.max() + 1
    x2_min, x2_max = x2.min() -1 , x2.max() + 1  

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                           np.arange(x2_min, x2_max, resolution))
    print(xx1)
    print(xx1.ravel())
    Z = classfier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.plot()



  X, y = prepare_data(df)

  _create_base_plot(df)
  _plot_decision_regions(X, y, model)

  plot_dir = "plots"
  os.makedirs(plot_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
  plotPath = os.path.join(plot_dir, file_name) # model/filename
  plt.savefig(plotPath)