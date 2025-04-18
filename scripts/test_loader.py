from .loader import loader3D

# Assuming args is an object with the necessary parameters
class Args:
    def __init__(self, image_size, target_name, data_directory, optional_meta):
        self.image_size = image_size
        self.target_name = target_name
        self.data_directory = data_directory
        self.optional_meta = optional_meta

# Setup args as an instance of Args
args = Args(
    image_size=(128, 128, 128),
    target_name='duration',
    data_directory='/mimer/NOBACKUP/groups/brainage/data/oasis3',
    optional_meta='age'
)

# Initialize the dataset
dataset = loader3D(args)

# Check if dataset length matches the number of pairs
print(f"Dataset length: {len(dataset)}")