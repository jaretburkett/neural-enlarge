import os
import sys
import time
import itertools
import numpy as np
import scipy.misc
import scipy.ndimage
from libs import models
import scipy.interpolate
from libs.args import args
from libs.loader import DataLoader
from libs.console import ansi, error

deblur = False

if "vgg19b34" in args.model:
    Model = models.VGG19b34
elif "vgg19b13" in args.model:
    Model = models.VGG19b13
elif "deblur" in args.model:
    Model = models.DEBLUR
    deblur = True
else:
    Model = models.LasagneModel

# Support ansi colors in Windows too.
if sys.platform == 'win32':
    import colorama
    colorama.init()


class NeuralEnhancer(object):
    def __init__(self, loader):
        if args.train:
            print('{}Training {} epochs on random image sections with batch size {}.{}' \
                  .format(ansi.BLUE_B, args.epochs, args.batch_size, ansi.BLUE))
        else:
            if len(args.files) == 0: error("Specify the image(s) to enhance on the command-line.")
            print('{}Enhancing {} image(s) specified on the command-line.{}' \
                  .format(ansi.BLUE_B, len(args.files), ansi.BLUE))

        self.thread = DataLoader() if loader else None
        self.model = Model()

        print('{}'.format(ansi.ENDC))

    def imsave(self, fn, img):
        scipy.misc.toimage(np.transpose(img + 0.5, (1, 2, 0)).clip(0.0, 1.0) * 255.0, cmin=0, cmax=255).save(fn)

    def show_progress(self, orign, scald, repro):
        os.makedirs('valid', exist_ok=True)
        os.makedirs('valid/' + args.model, exist_ok=True)
        for i in range(args.batch_size):
            self.imsave('valid/%s/%03i_1pixels.png' % (args.model, i), scald[i])
            self.imsave('valid/%s/%03i_2reprod.png' % (args.model, i), repro[i])
            self.imsave('valid/%s/%03i_3origin.png' % (args.model, i), orign[i])

    def show_progress_per_layer(self, in_img, reprod_img, layer_names, each_layer):
        os.makedirs('valid_layers', exist_ok=True)
        os.makedirs('valid_layers/' + args.model, exist_ok=True)
        self.imsave('valid_layers/%s/%03i_input.png' % (args.model, 0), in_img)
        self.imsave('valid_layers/%s/%03i_out.png' % (args.model, len(layer_names) + 1), reprod_img)

        for x in range(0, len(layer_names)):
            layer_name = layer_names[x]
            for i in range(0, len(each_layer[x])):
                img = np.transpose(np.squeeze(each_layer[x][i]), (2, 0, 1))
                if img.shape[0] > 3:
                    for o in range(0, img.shape[0]):
                        bw_img = np.zeros((3, img.shape[1], img.shape[2]))
                        bw_img[0, :, :] = img[o]
                        bw_img[1, :, :] = img[o]
                        bw_img[2, :, :] = img[o]
                        self.imsave('valid_layers/%s/%03i_%s_%04i_%04i.png' % (args.model, x + 1, layer_name, i, o), bw_img)
                else :
                    # rgb image
                    self.imsave('valid_layers/%s/%03i_%s_%04i_0000.png' % (args.model, x + 1, layer_name, i), img)

    def decay_learning_rate(self):
        l_r, t_cur = args.learning_rate, 0

        while True:
            yield l_r
            t_cur += 1
            if t_cur % args.learning_period == 0: l_r *= args.learning_decay

    def train(self):
        seed_size = args.batch_shape // args.zoom
        images = np.zeros((args.batch_size, 3, args.batch_shape, args.batch_shape), dtype=np.float32)
        seeds = np.zeros((args.batch_size, 3, seed_size, seed_size), dtype=np.float32)
        learning_rate = self.decay_learning_rate()
        # print(seeds.shape)
        # try:
        learning_rate_decay_multiplier = .01
        max_learning_rate = 0.01
        min_learning_rate = 0.0000001
        l_r = 0.0001

        previous_loss = None
        average, start = None, time.time()
        for epoch in range(args.epochs):
            total, stats = None, None
            if epoch >= args.discriminator_start:
                self.model.set_learning_rate(l_r)

            for _ in range(args.epoch_size):
                self.thread.copy(images, seeds)
                # print(images.shape)
                training_images = np.transpose(images, (0, 2, 3, 1))
                # print(images.shape)
                training_seeds = np.transpose(seeds, (0, 2, 3, 1))
                loss, stat = self.model.fit(training_images, training_seeds)
                losses = np.array([loss], dtype=np.float32)
                stats = (stats + stat) if stats is not None else np.array([stat], dtype=np.float32)
                total = total + losses if total is not None else losses
                l = np.sum(losses)
                assert not np.isnan(losses).any()
                average = l if average is None else average * 0.95 + 0.05 * l
                print('*' if l > average else '.', end='', flush=True)

            scald, repro = self.model.predict(seeds)
            self.show_progress(images, scald, repro)
            layer_names, each_layer = self.model.output_per_layer(seeds)
            self.show_progress_per_layer(scald[0], repro[0], layer_names, each_layer)
            total /= args.epoch_size

            # adjust learning rate
            this_loss = sum(total)
            if not previous_loss:
                previous_loss = this_loss
            elif deblur or (epoch > 50):
                if this_loss > previous_loss:
                    # increase learning rate
                    l_r += l_r * learning_rate_decay_multiplier
                    if l_r > max_learning_rate:
                        l_r = max_learning_rate
                    #  less magic
                    self.thread.less_magic()
                else:
                    # decay learning rate
                    l_r -= l_r * learning_rate_decay_multiplier * 2
                    if l_r < min_learning_rate:
                        l_r = min_learning_rate
                    # more magic
                    self.thread.more_magic()

            # stats /= args.epoch_size
            totals, labels = [sum(total)] + list(total), ['total', 'prcpt', 'smthn', 'advrs']
            gen_info = ['{}{}{}={:4.2e}'.format(ansi.WHITE_B, k, ansi.ENDC, v) for k, v in zip(labels, totals)]
            print('\rEpoch #{} at {:4.1f}s, magic:{:2.1f}, lr={:4.2e}{}'.format(epoch + 1, time.time() - start, self.thread.get_magic(), l_r,
                                                                 ' ' * (args.epoch_size - 30)))
            print('  - generator {}'.format(' '.join(gen_info)))

            # real, fake = stats[:args.batch_size], stats[args.batch_size:]
            # print('  - discriminator', real.mean(), len(np.where(real > 0.5)[0]),
            #       fake.mean(), len(np.where(fake < -0.5)[0]))
            # if epoch == args.adversarial_start - 1:
            #     print('  - generator now optimizing against discriminator.')
            #     self.model.adversary_weight.set_value(args.adversary_weight)
            #     running = None
            if (epoch + 1) % args.save_every == 0:
                print('  - saving current generator layers to disk...')
                self.model.save()


        # except KeyboardInterrupt:
        #     pass

        print('\n{}Trained {}x super-resolution for {} epochs.{}' \
              .format(ansi.CYAN_B, args.zoom, epoch + 1, ansi.CYAN))
        self.model.save()
        print(ansi.ENDC)

    def match_histograms(self, A, B, rng=(0.0, 255.0), bins=64):
        (Ha, Xa), (Hb, Xb) = [np.histogram(i, bins=bins, range=rng, density=True) for i in [A, B]]
        X = np.linspace(rng[0], rng[1], bins, endpoint=True)
        Hpa, Hpb = [np.cumsum(i) * (rng[1] - rng[0]) ** 2 / float(bins) for i in [Ha, Hb]]
        inv_Ha = scipy.interpolate.interp1d(X, Hpa, bounds_error=False, fill_value='extrapolate')
        map_Hb = scipy.interpolate.interp1d(Hpb, X, bounds_error=False, fill_value='extrapolate')
        return map_Hb(inv_Ha(A).clip(0.0, 255.0))

    def process(self, original):
        # Snap the image to a shape that's compatible with the generator (2x, 4x)
        s = 2 ** max(args.generator_upscale, args.generator_downscale)
        by, bx = original.shape[0] % s, original.shape[1] % s
        original = original[by - by // 2:original.shape[0] - by // 2, bx - bx // 2:original.shape[1] - bx // 2, :]

        # Prepare paded input image as well as output buffer of zoomed size.
        s, p, z = args.rendering_tile, args.rendering_overlap, args.zoom
        # default = s=80, p=24, z=2
        image = np.pad(original, ((p, p), (p, p), (0, 0)), mode='reflect')
        output = np.zeros((original.shape[0] * z, original.shape[1] * z, 3), dtype=np.float32)

        wh = args.rendering_tile + (args.rendering_overlap * 2)

        # Iterate through the tile coordinates and pass them through the network.
        for y, x in itertools.product(range(0, original.shape[0], s), range(0, original.shape[1], s)):
            img = np.transpose(image[y:y + p * 2 + s, x:x + p * 2 + s, :] / 255.0 - 0.5, (2, 0, 1))[np.newaxis].astype(
                np.float32)
            # add pad to the stitch to make it match input of model
            b, c, w, h = img.shape
            wp = wh - w
            hp = wh - h
            # prepare output depad
            unpad = np.zeros((w, h, c))
            img = np.pad(img, ((0, 0), (0, 0), (0, wp), (0, hp)), mode='reflect')

            unused, repro = self.model.predict(img)
            # remove any padding
            if (wp > 0) and (hp > 0):
                unpad = repro[0][:, 0: -wp * z, 0: -hp * z]
            elif hp > 0:
                unpad = repro[0][:, :, 0: -hp * z]
            elif wp > 0:
                unpad = repro[0][:, 0: -wp * z, :]
            else:
                unpad = repro[0]
            # print('unpad output', unpad.shape)
            output[y * z:(y + s) * z, x * z:(x + s) * z, :] = np.transpose(unpad + 0.5, (1, 2, 0))[p * z:-p * z,
                                                              p * z:-p * z, :]
            print('.', end='', flush=True)
        output = output.clip(0.0, 1.0) * 255.0

        # Match color histograms if the user specified this option.
        if args.rendering_histogram:
            for i in range(3):
                output[:, :, i] = self.match_histograms(output[:, :, i], original[:, :, i])

        return scipy.misc.toimage(output, cmin=0, cmax=255)
