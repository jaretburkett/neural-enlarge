import os
import sys
import time
import itertools
import numpy as np
import scipy.misc
import scipy.ndimage
import scipy.interpolate

from libs.args import args
from libs.ann import Model
from libs.loader import DataLoader
from libs.console import ansi, error


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
        for i in range(args.batch_size):
            self.imsave('valid/%s_%03i_origin.png' % (args.model, i), orign[i])
            self.imsave('valid/%s_%03i_pixels.png' % (args.model, i), scald[i])
            self.imsave('valid/%s_%03i_reprod.png' % (args.model, i), repro[i])

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
        try:
            average, start = None, time.time()
            for epoch in range(args.epochs):
                total, stats = None, None
                l_r = next(learning_rate)
                if epoch >= args.generator_start: self.model.gen_lr.set_value(l_r)
                if epoch >= args.discriminator_start: self.model.disc_lr.set_value(l_r)

                for _ in range(args.epoch_size):
                    self.thread.copy(images, seeds)
                    output = self.model.fit(images, seeds)
                    losses = np.array(output[:3], dtype=np.float32)
                    stats = (stats + output[3]) if stats is not None else output[3]
                    total = total + losses if total is not None else losses
                    l = np.sum(losses)
                    assert not np.isnan(losses).any()
                    average = l if average is None else average * 0.95 + 0.05 * l
                    print('*' if l > average else '.', end='', flush=True)

                scald, repro = self.model.predict(seeds)
                self.show_progress(images, scald, repro)
                total /= args.epoch_size
                stats /= args.epoch_size
                totals, labels = [sum(total)] + list(total), ['total', 'prcpt', 'smthn', 'advrs']
                gen_info = ['{}{}{}={:4.2e}'.format(ansi.WHITE_B, k, ansi.ENDC, v) for k, v in zip(labels, totals)]
                print('\rEpoch #{} at {:4.1f}s, lr={:4.2e}{}'.format(epoch + 1, time.time() - start, l_r,
                                                                     ' ' * (args.epoch_size - 30)))
                print('  - generator {}'.format(' '.join(gen_info)))

                real, fake = stats[:args.batch_size], stats[args.batch_size:]
                print('  - discriminator', real.mean(), len(np.where(real > 0.5)[0]),
                      fake.mean(), len(np.where(fake < -0.5)[0]))
                if epoch == args.adversarial_start - 1:
                    print('  - generator now optimizing against discriminator.')
                    self.model.adversary_weight.set_value(args.adversary_weight)
                    running = None
                if (epoch + 1) % args.save_every == 0:
                    print('  - saving current generator layers to disk...')
                    self.model.save_generator()

        except KeyboardInterrupt:
            pass

        print('\n{}Trained {}x super-resolution for {} epochs.{}' \
              .format(ansi.CYAN_B, args.zoom, epoch + 1, ansi.CYAN))
        self.model.save_generator()
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
        image = np.pad(original, ((p, p), (p, p), (0, 0)), mode='reflect')
        output = np.zeros((original.shape[0] * z, original.shape[1] * z, 3), dtype=np.float32)

        # Iterate through the tile coordinates and pass them through the network.
        for y, x in itertools.product(range(0, original.shape[0], s), range(0, original.shape[1], s)):
            img = np.transpose(image[y:y + p * 2 + s, x:x + p * 2 + s, :] / 255.0 - 0.5, (2, 0, 1))[np.newaxis].astype(
                np.float32)
            *_, repro = self.model.predict(img)
            output[y * z:(y + s) * z, x * z:(x + s) * z, :] = np.transpose(repro[0] + 0.5, (1, 2, 0))[p * z:-p * z,
                                                              p * z:-p * z, :]
            print('.', end='', flush=True)
        output = output.clip(0.0, 1.0) * 255.0

        # Match color histograms if the user specified this option.
        if args.rendering_histogram:
            for i in range(3):
                output[:, :, i] = self.match_histograms(output[:, :, i], original[:, :, i])

        return scipy.misc.toimage(output, cmin=0, cmax=255)
