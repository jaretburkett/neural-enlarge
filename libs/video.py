import os
import sys
import cv2
import scipy.misc
from libs import magic
import numpy as np
from libs.args import args

# Support ansi colors in Windows too.
if sys.platform == 'win32':
    pass


def enhance_video(filename, enhancer):
    if args.append:
        output_filename = os.path.splitext(filename)[0] + '_%s.mp4' % args.append
    else:
        output_filename = os.path.splitext(filename)[0] + '_ne%ix_%s.mp4' % (args.zoom, args.model)

    tmp_filename = os.path.splitext(filename)[0] + '_ne%ix_%s_tmp.mp4' % (args.zoom, args.model)
    cap = cv2.VideoCapture(filename)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # width = int(cap.get(3))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    framerate = cap.get(cv2.CAP_PROP_FPS)
    # height = int(cap.get(4))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('\nwidth:%i height:%i' % (width, height))
    out_file = cv2.VideoWriter(
        filename=tmp_filename,
        fourcc=fourcc,
        apiPreference=cv2.CAP_ANY,
        fps=framerate,
        frameSize=(width * args.zoom, height * args.zoom))

    frame_num = 0
    while cap.isOpened():
        percent_of = frame_num / total_frames * 100

        ret, frame = cap.read()
        if ret:
            print('\n%s frame %i of %i (%3.2f%%)' % (filename, frame_num, total_frames, percent_of), end=' ')
            # convert the frame to something the neural net can understand
            frame_img = scipy.misc.fromimage(magic.cv_to_pil(frame)).astype(np.float32)
            # process the frame
            frame_out = enhancer.process(frame_img)

            # write the processed image
            out_file.write(magic.pil_to_cv(frame_out))

            frame_num += 1

            # wait on open cv for next frame if needed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out_file.release()
    cv2.destroyAllWindows()

    print('\nCompiling new frames and audio into %s' % output_filename)
    # compile audio
    os.system("ffmpeg -y -i %s -i %s -c:v libx264 -preset medium -crf 22 -strict -2 -map 0:v:0 -map 1:a:0 -shortest %s" %
              (tmp_filename, filename, output_filename))

    # todo check for errors

    print('\nDone with %s' % output_filename)

    # remove tmp file
    os.remove(tmp_filename)

