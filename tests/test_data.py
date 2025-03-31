from torch.utils.data import Dataset

from crowd_counting_with_diffusion_models.template_defaults.data import MyDataset


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset("data/raw")
    assert isinstance(dataset, Dataset)
