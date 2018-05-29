
"""
 _   _                      _   _____      _
| \ | |                    | | |  ___|    | |
|  \| | ___ _   _ _ __ __ _| | | |__ _ __ | | __ _ _ __ __ _  ___
| . ` |/ _ \ | | | '__/ _` | | |  __| '_ \| |/ _` | '__/ _` |/ _ |
| |\  |  __/ |_| | | | (_| | | | |__| | | | | (_| | | | (_| |  __/
\_| \_/\___|\__,_|_|  \__,_|_| \____/_| |_|_|\__,_|_|  \__, |\___|
                                                        __/ |
                                                       |___/
"""

#
# Copyright (c) 2018, Jaret Burkett
#
# Neural Enlarge is based on Neural Enhance and is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License version 3. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.
#

#
# Copyright (c) 2016, Alex J. Champandard.
#
# Neural Enhance is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General
# Public License version 3. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#

# version 0.0.1

import os
import sys
import scipy.ndimage

from libs.args import args
from libs.console import ansi
from libs.video import enhance_video
# from libs.enhance import NeuralEnhancer
from libs.newenhance import NeuralEnhancer

# Support ansi colors in Windows too.
if sys.platform == 'win32':
    import colorama

# ----------------------------------------------------------------------------------------------------------------------


print("""{}   {}Super Resolution for images and videos powered by Deep Learning!{}
  - Code licensed as AGPLv3, models under CC BY-NC-SA.{}""".format(ansi.CYAN_B, __doc__, ansi.CYAN, ansi.ENDC))


if __name__ == "__main__":
    if args.train:
        # args.zoom = 2 ** (args.generator_upscale - args.generator_downscale)
        enhancer = NeuralEnhancer(loader=True)
        enhancer.train()
    else:
        enhancer = NeuralEnhancer(loader=False)
        for filename in args.files:
            print(filename, end=' ')
            if filename.lower().endswith(('.mp4', '.mov', '.mpg', '.avi', '.flv')):
                enhance_video(filename, enhancer)
            else:
                img = scipy.ndimage.imread(filename, mode='RGB')
                out = enhancer.process(img)

                if args.append:
                    output_filename = os.path.splitext(filename)[0] + '_%s.png' % args.append
                else:
                    output_filename = os.path.splitext(filename)[0] + '_ne%ix_%s.png' % (args.zoom, args.model)

                if args.output_type == 'jpg':
                    out.save(output_filename, format='jpeg', quality=80)
                else:
                    out.save(output_filename)
            print(flush=True)
        print(ansi.ENDC)
