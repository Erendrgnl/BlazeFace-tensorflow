from params import get_hyper_parameters
from myDataLoader import get_data

from model import BlazeFace
from params import get_hyper_parameters

hp = get_hyper_parameters()

blazeFace = BlazeFace(hp)

train_data = get_data()

blazeFace.train(train_data)