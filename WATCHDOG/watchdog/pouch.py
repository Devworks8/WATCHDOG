import os
import datetime
import argparse
import yaml
import cv2

# TODO: Implement config file

parser = argparse.ArgumentParser(description='Watchdog!', prog='Watchdog')
parser.add_argument('--save', nargs=1, metavar='<filename>', dest='save', help='Save occupancy captures to file.')
parser.add_argument('--notify', nargs=1, metavar='<email address>', dest='send', help='Email file.')
parser.add_argument('--encrypt', action='store_true', dest='encrypt', help='Encrypt file.')
parser.add_argument('--version', action='version', version='%(prog)s 0.1')
args = parser.parse_args()
# args is a list containing [save=None, send=None, encrypt=False]
_CWD = os.path.abspath(os.path.curdir)
cascPath = _CWD + '/haarcascade_frontalface_default.xml'

with open(_CWD + '/watchdog.yml') as fstream:
    settings = yaml.load(fstream)

_FRATE = settings['defaults']['framerate']
_RESOLUTION = settings['defaults']['resolution']
_RSTOPDELAY = settings['defaults']['rstopdelay']

faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)

# Check arguments
if args.save is not None:
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    stop_time = datetime.datetime.now()
    reset = True

while True:

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    text = 'Unoccupied'

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = 'Occupied'

    if args.save is not None:
        if text == 'Occupied':
            if reset:
                stamp = '-' + datetime.datetime.now().strftime('%A %d %B %Y %I:%M:%S%p') + '.avi'
                out = cv2.VideoWriter(args.save[0] + stamp, fourcc, _FRATE, _RESOLUTION)
                reset = False

            stop_time = datetime.datetime.now()
            out.write(frame)

        elif datetime.datetime.now().minute - stop_time.minute >= _RSTOPDELAY:
            out.release()
            reset = True

    # draw the text and timestamp on the frame
    cv2.putText(frame, 'Room Status: {}'.format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime('%A %d %B %Y %I:%M:%S%p'),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # Display the resulting frame
    cv2.imshow('Security Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
if args.save is not None:
    out.release()
video_capture.release()
cv2.destroyAllWindows()
