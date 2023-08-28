from imports.common_imports import *
from model.gen_disc import *

class Trainer(tf.Module):
    def __init__(
        self,
        dataloader,
        epochs: int = 100,
        n_train: int = 24000,
        batch_train_per_epoch: int = 720,
    ):        

        self.batch_size = next(dataloader)[0].shape[0]
        self.epochs = epochs
        self.n_train = n_train
        self.batch_train_per_epoch = batch_train_per_epoch

        self.generator_input_shape = (256, 256, 3)
        self.discriminator_output_shape = (self.batch_size, 16, 16, 1)
        
        self.generator_AtoB = Generator()
        self.generator_BtoA = Generator()
        self.discriminator_A = Discriminator()
        self.discriminator_B = Discriminator()
        self.cycle_AtoB = CycleModel(self.generator_AtoB, self.generator_BtoA, self.discriminator_B,
                                    self.generator_input_shape)
        self.cycle_BtoA = CycleModel(self.generator_BtoA, self.generator_AtoB, self.discriminator_A,
                                    self.generator_input_shape)  
    

    def generate_real_class_labels(self, n_samples, patch_shape):
        y = np.ones((n_samples, patch_shape, patch_shape, 1))
        return y
    
    def generate_fake_samples(self, generator_model, trainX, patch_shape):
        X = generator_model.predict(trainX)
        y = np.zeros((len(X), patch_shape, patch_shape, 1))
        return X, y
    
    def save_models(self, step, generator_AtoB, generator_BtoA):
        # save the first generator model
        filename1 = './_checkpoints/generator_AtoB_%06d.h5' % (step+1)
        generator_AtoB.save(filename1)
        # save the second generator model
        filename2 = './_checkpoints/generator_BtoA_%06d.h5' % (step+1)
        generator_BtoA.save(filename2)
        print('>Saved: %s and %s' % (filename1, filename2))

    def summarize_performance(self, step, generator_model, trainX, name, n_samples=5):

        X_in = trainX 
        X_out = generator_model.predict(X_in)

        X_in = (X_in + 1) / 2.0
        X_out = (X_out + 1) / 2.0

        for i in range(n_samples):
            plt.subplot(2, n_samples, 1 + i)
            plt.axis('off')
            plt.imshow(X_in[i])
        for i in range(n_samples):
            plt.subplot(2, n_samples, 1 + n_samples + i)
            plt.axis('off')
            plt.imshow(X_out[i])
        # save plot to file
        filename1 = './_outputs/%s_generated_plot_%06d.png' % (name, (step+1))
        plt.savefig(filename1)
        plt.close()

    def update_image_pool(self, pool, images, max_size=50):
        selected = list()
        for image in images:
            if len(pool) < max_size:
                # stock the pool
                pool.append(image)
                selected.append(image)
            elif random() < 0.5:
                # use image, but don't add it to the pool
                selected.append(image)
            else:
                # replace an existing image and use replaced image
                ix = randint(0, len(pool))
                selected.append(pool[ix])
                pool[ix] = image
        return asarray(selected)

    def train(self, train_dataloader, test_dataloader):
        
        epochs, batch_size, = self.epochs, self.batch_size  
        poolA, poolB = list(), list()
        num_batches = int(self.n_train / batch_size)
        n_patch = self.discriminator_output_shape[1]

        # manually enumerate epochs
        for epoch in range(epochs):
            
            epoch_dA_loss1, epoch_dA_loss2 = 0, 0
            epoch_dB_loss1, epoch_dB_loss2 = 0, 0
            epoch_g_loss1, epoch_g_loss2 = 0, 0
        
            with tqdm(total=self.batch_train_per_epoch, unit='batch') as pbar:
                for i, (X_realA, X_realB) in enumerate(train_dataloader):

                    y_realA = self.generate_real_class_labels(batch_size, n_patch)
                    y_realB = self.generate_real_class_labels(batch_size, n_patch)
                    
                    X_fakeA, y_fakeA = self.generate_fake_samples(self.generator_BtoA, X_realB, n_patch)
                    X_fakeB, y_fakeB = self.generate_fake_samples(self.generator_AtoB, X_realA, n_patch)
                    
                    X_fakeA = self.update_image_pool(poolA, X_fakeA)
                    X_fakeB = self.update_image_pool(poolB, X_fakeB)
                    
                    # update generator B->A via the composite model
                    g_loss2, _, _, _, _  = self.cycle_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
                    # update discriminator for A -> [real/fake]
                    dA_loss1 = self.discriminator_A.train_on_batch(X_realA, y_realA)
                    dA_loss2 = self.discriminator_A.train_on_batch(X_fakeA, y_fakeA)
                    
                    # update generator A->B via the composite model
                    g_loss1, _, _, _, _ = self.cycle_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
                    # update discriminator for B -> [real/fake]
                    dB_loss1 = self.discriminator_B.train_on_batch(X_realB, y_realB)
                    dB_loss2 = self.discriminator_B.train_on_batch(X_fakeB, y_fakeB)

                    epoch_dA_loss1 += dA_loss1
                    epoch_dA_loss2 += dA_loss2
                    epoch_dB_loss1 += dB_loss1
                    epoch_dB_loss2 += dB_loss2
                    epoch_g_loss1 += g_loss1
                    epoch_g_loss2 += g_loss2

                    pbar.set_postfix({'dA_loss1': dA_loss1, 'dB_loss1': dB_loss1, 'g_loss1': g_loss1, 'g_loss2': g_loss2})
                    pbar.update()

                    if i >= self.batch_train_per_epoch:
                        break

            epoch_dA_loss1 /= num_batches
            epoch_dA_loss2 /= num_batches
            epoch_dB_loss1 /= num_batches
            epoch_dB_loss2 /= num_batches
            epoch_g_loss1 /= num_batches
            epoch_g_loss2 /= num_batches

            print('Epoch>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (epoch+1, epoch_dA_loss1,epoch_dA_loss2, epoch_dB_loss1,epoch_dB_loss2, epoch_g_loss1,epoch_g_loss2))
            
            X_realA, X_realB = next(test_dataloader)

            self.summarize_performance(epoch, self.generator_AtoB, X_realA, 'Anime to Celeb')
            self.summarize_performance(epoch, self.generator_BtoA, X_realB, 'Celeb to Anime')
            self.save_models(epoch, self.generator_AtoB, self.generator_BtoA)

            # if (epoch+1) % 1 == 0 or epoch == 5:
            #     self.summarize_performance(epoch, self.generator_AtoB, X_realA, 'Anime to Celeb')
            #     self.summarize_performance(epoch, self.generator_BtoA, X_realB, 'Celeb to Anime')
            # if (epoch+1) % 1 == 0:
            #     self.save_models(epoch, self.generator_AtoB, self.generator_BtoA)


    


