import os
from shutil import copyfile


def _create_model_training_folder(writer, files_to_same):
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
    root_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            src = os.path.join(root_dir, file)
            dst = os.path.join(model_checkpoints_folder, os.path.basename(file)) 
            copyfile(src, dst)
