from imports.common_imports import *

class DataCopy:

    def __init__(self) -> None:
        pass

    def __create_folder(self, root_dir, folder_name):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            os.mkdir(os.path.join(folder_path, folder_name))
        else:
            print(f"El directorio {folder_name} ya existe.")

    def __copy_files(self, files, from_dir, to_dir):

        # si el directorio contiene archivos, pass
        if len(os.listdir(to_dir)) > 0:
            print("El directorio ya contiene archivos. No se copiarán nuevos archivos.")
            return
        
        with tqdm(total=len(files)) as pbar:
            for file in files:
                shutil.copy(os.path.join(from_dir, file), os.path.join(to_dir, file))
                tqdm.update(pbar)

    def create_directory_structure(self, from_dir, root_dir):
        """
        Copy data from src to dst.
        """
        train_dir = os.path.join(root_dir, "train", "train")
        test_dir = os.path.join(root_dir, "test", "test")
        val_dir = os.path.join(root_dir, "val", "val")
        self.__create_folder(root_dir, "train")
        self.__create_folder(root_dir, "test")
        self.__create_folder(root_dir, "val")

        np.random.seed(42)
        files = os.listdir(from_dir)
        np.random.shuffle(files)
        num_files = len(files)
        train_files = files[:int(0.8*num_files)]
        test_files = files[int(0.8*num_files):int(0.9*num_files)]
        val_files = files[int(0.9*num_files):]

        self.__copy_files(train_files, from_dir, train_dir)
        self.__copy_files(test_files, from_dir, test_dir)
        self.__copy_files(val_files, from_dir, val_dir)


class DataLoader:
    def __init__(
            self,
            target_size=(256, 256),
            batch_size = 360,
            seed = 24,
            data_augmentation = None,
    ) -> None:
        
        self.batch_size = batch_size
        self.data_augmentation = ImageDataGenerator(preprocessing_function = self.custom_preprocessing)\
                if data_augmentation is None else data_augmentation
        self.seed = seed
        self.color_mode = "rgb"
        self.target_size = target_size

    def custom_preprocessing(self, image):
        return (image - 127.5) / 127.5

    def generator(self, directory):
        return self.data_augmentation.flow_from_directory(
            directory,
            target_size=self.target_size,
            color_mode = self.color_mode,
            batch_size=self.batch_size,
            class_mode=None,
            seed=self.seed
        )

    def pair_generator(self, generator1, generator2):
        train_generator = zip(generator1, generator2)
        for (I1, I2) in train_generator:
            yield (I1, I2)


        

