from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import numpy as np
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "running_logs.log"), level=logging.INFO, format=logging_str)


def main(data, model_name, plot_name, eta, epochs):
    df_or = pd.DataFrame(data)

    X,y = prepare_data(df_or)

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()


    save_model(model, filename=model_name)
    save_plot(df_or, plot_name,  model)


if __name__ == "__main__":

    OR = {
    "X1":[0, 0, 1, 1],
    "X2":[0, 1, 0, 1],
    "y":[0, 1, 1, 1]
}


    ETA = 0.1
    EPOCHS = 10

    try:
        logging.info(">>>>> starting training >>>>>")
        main(data=OR, model_name="or.model", plot_name="or.png", eta=ETA, epochs=EPOCHS)
        logging.info("<<<<< training done successfully<<<<<\n")
        
    except Exception as e:
        logging.exception(e)
        raise e