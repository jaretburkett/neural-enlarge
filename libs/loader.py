import os
import io
import sys
import glob
import math
from libs import magic
import random
import threading
import scipy.misc
import numpy as np
from libs.args import args
from libs.console import error, warn
from PIL import Image, ImageFilter

deblur = False
if "deblur" in args.model:
    deblur = True

# Support ansi colors in Windows too.
if sys.platform == 'win32':
    pass


class DataLoader(threading.Thread):
    def __init__(self):
        super(DataLoader, self).__init__(daemon=True)
        self.data_ready = threading.Event()
        self.data_copied = threading.Event()
        self.magic = 0.0

        if deblur:
            self.magic = 5.0

        self.orig_shape, self.seed_shape = args.batch_shape, args.batch_shape // args.zoom

        self.orig_buffer = np.zeros((args.buffer_size, 3, self.orig_shape, self.orig_shape), dtype=np.float32)
        self.seed_buffer = np.zeros((args.buffer_size, 3, self.seed_shape, self.seed_shape), dtype=np.float32)
        print('finding files for %s' % args.train)
        self.files = glob.glob(os.path.abspath('data/*.jpg'))
        print(self.files)
        if len(self.files) == 0:
            error("There were no files found to train from searching for `{}`".format(args.train),
                  "  - Try putting all your images in one folder and using `--train=data/*.jpg`")

        self.available = set(range(args.buffer_size))
        self.ready = set()

        self.cwd = os.getcwd()
        print('starting dataloader')
        self.start()
        # self.run()

    def more_magic(self):
        max_magic = 10
        # if self.magic < max_magic:
        #     self.magic = self.magic + 0.1
        m = random.randint(50, 100)
        self.magic = m * 0.1

    def less_magic(self):
        min_magic = 0
        if deblur:
            min_magic = 30
        # if self.magic > min_magic:
        #     self.magic = self.magic - 0.1
        m = random.randint(min_magic, 50)
        self.magic = m * 0.1


    def get_magic(self):
        return self.magic

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
            scale = 2 ** random.randint(0, args.train_scales)
            if scale > 1 and all(s // scale >= args.batch_shape for s in orig.size):
                orig = orig.resize((orig.size[0] // scale, orig.size[1] // scale), resample=Image.LANCZOS)
            if any(s < args.batch_shape for s in orig.size):
                raise ValueError('Image is too small for training with size {}'.format(orig.size))
        except Exception as e:
            # warn('Could not load `{}` as image.'.format(filename),
            #      '  - Try fixing or removing the file before next run.')
            self.files.remove(f)
            return

        orig = magic.random_flip(orig)

        seed = orig

        if args.zoom > 1:
            seed = seed.resize((orig.size[0] // args.zoom, orig.size[1] // args.zoom), resample=Image.LANCZOS)
        # magic
        if self.magic > 0.1:
            seed = magic.random_magic(seed, round(self.magic))

        orig = scipy.misc.fromimage(orig).astype(np.float32)
        seed = scipy.misc.fromimage(seed).astype(np.float32)

        if args.train_magic == 0:
            if args.train_noise is not None:
                seed += scipy.random.normal(scale=args.train_noise, size=(seed.shape[0], seed.shape[1], 1))

        for _ in range(seed.shape[0] * seed.shape[1] // (args.buffer_fraction * self.seed_shape ** 2)):
            h = random.randint(0, seed.shape[0] - self.seed_shape)
            w = random.randint(0, seed.shape[1] - self.seed_shape)
            seed_chunk = seed[h:h + self.seed_shape, w:w + self.seed_shape]
            h, w = h * args.zoom, w * args.zoom
            orig_chunk = orig[h:h + self.orig_shape, w:w + self.orig_shape]

            while len(self.available) == 0:
                self.data_copied.wait()
                self.data_copied.clear()

            i = self.available.pop()
            self.orig_buffer[i] = np.transpose(orig_chunk.astype(np.float32) / 255.0 - 0.5, (2, 0, 1))
            self.seed_buffer[i] = np.transpose(seed_chunk.astype(np.float32) / 255.0 - 0.5, (2, 0, 1))
            self.ready.add(i)

            if len(self.ready) >= args.batch_size:
                self.data_ready.set()

    def copy(self, origs_out, seeds_out):
        self.data_ready.wait()
        # print('copying')
        self.data_ready.clear()

        for i, j in enumerate(random.sample(self.ready, args.batch_size)):
            origs_out[i] = self.orig_buffer[j]
            seeds_out[i] = self.seed_buffer[j]
            self.available.add(j)
        self.data_copied.set()
