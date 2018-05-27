import os
import io
import sys
import glob
from libs import magic
import random
import threading
import scipy.misc
import numpy as np
from libs.args import args
from libs.console import error, warn
from PIL import Image, ImageFilter

# Support ansi colors in Windows too.
if sys.platform == 'win32':
    pass


class DataLoader(threading.Thread):
    def __init__(self):
        super(DataLoader, self).__init__(daemon=True)
        self.data_ready = threading.Event()
        self.data_copied = threading.Event()

        self.orig1x_shape = args.batch_shape // 3
        self.orig2x_shape = args.batch_shape // 3 * 2
        self.orig3x_shape = args.batch_shape
        # self.orig_shape = args.batch_shape
        self.seed_shape = args.batch_shape // 3

        self.orig1x_buffer = np.zeros((args.buffer_size, 3, self.orig1x_shape, self.orig1x_shape), dtype=np.float32)
        self.orig2x_buffer = np.zeros((args.buffer_size, 3, self.orig2x_shape, self.orig2x_shape), dtype=np.float32)
        self.orig3x_buffer = np.zeros((args.buffer_size, 3, self.orig3x_shape, self.orig3x_shape), dtype=np.float32)
        self.seed_buffer = np.zeros((args.buffer_size, 3, self.seed_shape, self.seed_shape), dtype=np.float32)
        print('finding files for %s' % args.train)
        self.files = glob.glob(os.path.abspath('data/*.jpg'))
        print(self.files)
        if len(self.files) == 0:
            error("There were no files found to train from searching for `{}`".format(args.train),
                  "  - Try putting all your images in one folder and using `--train=data/*.jpg`")

        self.available = set(range(args.buffer_size))
        self.ready = set()
        self.magic = 0

        self.cwd = os.getcwd()
        print('starting dataloader')
        self.start()
        # self.run()

    def more_magic(self):
        max_magic = 10
        if self.magic < max_magic:
            self.magic += 1

    def less_magic(self):
        min_magic = 0
        if self.magic > min_magic:
            self.magic -= 1

    def run(self):
        while True:
            random.shuffle(self.files)
            for f in self.files:
                # print('adding %s to buffer' % f)
                self.add_to_buffer(f)

    def add_to_buffer(self, f):
        filename = os.path.join(self.cwd, f)
        try:
            orig = Image.open(filename).convert('RGB')
            scale = 3 ** random.randint(0, args.train_scales)
            if scale > 1 and all(s // scale >= args.batch_shape for s in orig.size):
                orig = orig.resize((orig.size[0] // scale, orig.size[1] // scale), resample=Image.LANCZOS)
            if any(s < args.batch_shape for s in orig.size):
                raise ValueError('Image is too small for training with size {}'.format(orig.size))
        except Exception as e:
            # warn('Could not load `{}` as image.'.format(filename),
            #      '  - Try fixing or removing the file before next run.')
            self.files.remove(f)
            return

        if args.train_magic is not 0:
            orig = magic.random_flip(orig)

        seed = orig

        orig1x = orig
        orig2x = orig
        orig3x = orig

        orig1x = orig1x.resize((orig.size[0] // 3, orig.size[1] // 3), resample=Image.LANCZOS)
        orig2x = orig2x.resize((orig.size[0] // 3 * 2, orig.size[1] // 3 *2), resample=Image.LANCZOS)

        seed = seed.resize((orig.size[0] // 3, orig.size[1] // 3), resample=Image.LANCZOS)

        # magic
        if self.magic is not 0:
            seed = magic.random_magic(seed, self.magic)

        orig1x = scipy.misc.fromimage(orig1x).astype(np.float32)
        orig2x = scipy.misc.fromimage(orig2x).astype(np.float32)
        orig3x = scipy.misc.fromimage(orig3x).astype(np.float32)
        seed = scipy.misc.fromimage(seed).astype(np.float32)

        for _ in range(seed.shape[0] * seed.shape[1] // (args.buffer_fraction * self.seed_shape ** 2)):
            h = random.randint(0, seed.shape[0] - self.seed_shape)
            w = random.randint(0, seed.shape[1] - self.seed_shape)
            seed_chunk = seed[h:h + self.seed_shape, w:w + self.seed_shape]

            h1, w1 = h, w
            h2, w2 = h * 2, w * 2
            h3, w3 = h * 3, w * 3
            orig1x_chunk = orig1x[h1:h1 + self.orig1x_shape, w1:w1 + self.orig1x_shape]
            orig2x_chunk = orig2x[h2:h2 + self.orig2x_shape, w2:w2 + self.orig2x_shape]
            orig3x_chunk = orig3x[h3:h3 + self.orig3x_shape, w3:w3 + self.orig3x_shape]

            while len(self.available) == 0:
                self.data_copied.wait()
                self.data_copied.clear()

            i = self.available.pop()
            self.orig1x_buffer[i] = np.transpose(orig1x_chunk.astype(np.float32) / 255.0 - 0.5, (2, 0, 1))
            self.orig2x_buffer[i] = np.transpose(orig2x_chunk.astype(np.float32) / 255.0 - 0.5, (2, 0, 1))
            self.orig3x_buffer[i] = np.transpose(orig3x_chunk.astype(np.float32) / 255.0 - 0.5, (2, 0, 1))
            self.seed_buffer[i] = np.transpose(seed_chunk.astype(np.float32) / 255.0 - 0.5, (2, 0, 1))
            self.ready.add(i)

            if len(self.ready) >= args.batch_size:
                self.data_ready.set()

    def copy(self, origs1x_out, origs2x_out, origs3x_out, seeds_out):
        self.data_ready.wait()
        # print('copying')
        self.data_ready.clear()

        for i, j in enumerate(random.sample(self.ready, args.batch_size)):
            origs1x_out[i] = self.orig1x_buffer[j]
            origs2x_out[i] = self.orig2x_buffer[j]
            origs3x_out[i] = self.orig3x_buffer[j]
            seeds_out[i] = self.seed_buffer[j]
            self.available.add(j)
        self.data_copied.set()
