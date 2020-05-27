clean_path = "./image/"

SIZE_INPUT = 64  # size of the training patch
SIZE_KERNEL = 61  # size of the kernel
SIZE_MATRIX = 10000
PER_FOR_MATRIX = 0.99
NUM_CHANNEL = 2000  # number of the input's channels.
NUM_FILES = 3  # total ( num_files - 1 ) training h5 files, the last one is used for validation
NUM_BATCH = 500  # number of patches in each h5 file.
UPDATE_OPS = 'update_ops'
DATA_PATH = './data_generation/h5data/'
EPOCH = 3
LEARNING_RATE = 0.001
SIZE_BATCH = 20
LAYER = 4
EPS = 0.00001
