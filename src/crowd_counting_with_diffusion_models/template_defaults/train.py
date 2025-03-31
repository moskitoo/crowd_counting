from crowd_counting_with_diffusion_models.model import Model
from crowd_counting_with_diffusion_models.template_defaults.data import MyDataset

def train():
    dataset = MyDataset("data/raw")
    model = Model()
    # add rest of your training code here

if __name__ == "__main__":
    train()
