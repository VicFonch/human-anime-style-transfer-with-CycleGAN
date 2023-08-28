from imports.common_imports import *

from model.gen_disc import *
from model.trainer import *

from data_processing.copy_load import *

def equal_len_dirs(dir1, dir2):

    n_dir1 = len(os.listdir(dir1))
    n_dir2 = len(os.listdir(dir2))
    np.random.seed(42)
    dir1_files = os.listdir(dir1)
    dir2_files = os.listdir(dir2)
    np.random.shuffle(dir1_files)
    np.random.shuffle(dir2_files)

    if n_dir1 == n_dir2:
        print("Los directorios tienen el mismo tamaño.")
        return

    if n_dir1 > n_dir2:
        dir1_files = dir1_files[n_dir2:]
        for file in dir1_files:
            os.remove(os.path.join(dir1, file))
    elif n_dir2 < n_dir1:
        dir2_files = dir2_files[n_dir1:]
        for file in dir2_files:
            os.remove(os.path.join(dir2, file))

if __name__ == "__main__":

    anime_from_dir = "/home/vfonte/Proyectos/CycleGAN/_data/anime_from"
    anime_root_path = "/home/est_posgrado_victor.fonte/Proyectos_y_tareas/cycleGAN/_data/anime"

    celeb_from_dir = "/home/vfonte/Proyectos/CycleGAN/_data/celeb_from"
    celeb_root_path = "/home/est_posgrado_victor.fonte/Proyectos_y_tareas/cycleGAN/_data/celeb"

    # copy_objeto = DataCopy()
    # copy_objeto.create_directory_structure(anime_from_dir, anime_root_path)
    # copy_objeto.create_directory_structure(celeb_from_dir, celeb_root_path)

    # shutil.rmtree(anime_from_dir)
    # shutil.rmtree(celeb_from_dir)

    anime_train_dir = os.path.join(anime_root_path, "train", "train")
    anime_test_dir = os.path.join(anime_root_path, "test", "test")
    anime_val_dir = os.path.join(anime_root_path, "val", "val")

    celeb_train_dir = os.path.join(celeb_root_path, "train", "train")
    celeb_test_dir = os.path.join(celeb_root_path, "test", "test")
    celeb_val_dir = os.path.join(celeb_root_path, "val", "val")

    equal_len_dirs(anime_train_dir, celeb_train_dir)
    equal_len_dirs(anime_test_dir, celeb_test_dir)
    equal_len_dirs(anime_val_dir, celeb_val_dir)

    dataloader = DataLoader(
        target_size=(256, 256),
        batch_size = 12,
        seed = 24
    )

    train_anime_generator  = dataloader.generator(os.path.join(anime_root_path, "train"))
    val_anime_generator  = dataloader.generator(os.path.join(anime_root_path, "val"))
    test_anime_generator  = dataloader.generator(os.path.join(anime_root_path, "test"))
    train_celeb_generator  = dataloader.generator(os.path.join(celeb_root_path, "train"))
    val_celeb_generator  = dataloader.generator(os.path.join(celeb_root_path, "val"))
    test_celeb_generator  = dataloader.generator(os.path.join(celeb_root_path, "test"))

    train_generator = dataloader.pair_generator(train_anime_generator, train_celeb_generator)
    val_generator = dataloader.pair_generator(val_anime_generator, val_celeb_generator)
    test_generator = dataloader.pair_generator(test_anime_generator, test_celeb_generator)

    ### Training ###
    checkpoint = "/home/est_posgrado_victor.fonte/Proyectos_y_tareas/cycleGAN/_checkpoints"
    dataloader = DataLoader()

    trainer = Trainer(
        train_generator,
        epochs=130,
        n_train=24000,
        batch_train_per_epoch=250,     
    )
    trainer.train(train_generator, test_generator)

    ### Testing ###
    # model.test(test_datagen=test_generator)
    # print("Testing completed.")

    # ### Outputs ###
    # anime_img, celeb_img = next(test_generator)
    # anime_img = anime_img[0]
    # celeb_img = c~eleb_img[0]
    # output_dir = "/home/est_posgrado_victor.fonte/Proyectos_y_tareas/cycleGAN/_outputs"
    # model.generate_output(output_dir, anime_img, celeb_img, dir = "celeb2anime")

    print("Outputs completed.")
    print("All completed.")