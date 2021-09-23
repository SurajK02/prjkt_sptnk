from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import numpy as np
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(logs_dir,"running_logs.log"), level=logging.INFO, format=logging_str, filemode='a')



def main(data, modelName, plotName, eta, epochs):
    df_and = pd.DataFrame(AND)

    X,y = prepare_data(df_and)

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()


    save_model(model, filename=modelName)
    save_plot(df_and, plotName,  model)

if __name__ == '__main__' :
    AND = {
    "X1":[0, 0, 1, 1],
    "X2":[0, 1, 0, 1],
    "y":[0, 0, 0, 1]
}


    ETA = 0.1
    EPOCHS = 10
    try :
        logging.info(">>>>>>>>>> Starting Training >>>>>>>>>>")
        main(data=AND, modelName="and.model", plotName="and.png", eta=ETA, epochs=EPOCHS)
        logging.info("<<<<<<<<<< End of Training <<<<<<<<<<\n")

    except Exception as e:
        logging.exception(e)
        raise e