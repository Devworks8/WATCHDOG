import os
import datetime
import argparse
import smtplib
from email.mime.text import MIMEText

import yaml
import cv2


def runparser():
    parser = argparse.ArgumentParser(description='Watchdog!', prog='Watchdog')
    parser.add_argument('--save', nargs=1, metavar='<filename>', dest='save', help='Save occupancy captures to file.')
    parser.add_argument('--nsave', action='store_true', dest='nsave', help='Do not save capture to file.')
    parser.add_argument('--res', nargs='?', metavar='<width> <height>', dest='res', help='Set recording resolution.')
    parser.add_argument('--frate', nargs=1, metavar='<frame rate>', dest='frate', help='Set recording frame rate.')
    parser.add_argument('--notify', nargs=1, metavar='<email address>', dest='send', help='Email file.')
    parser.add_argument('--smtp', nargs='?', metavar='<SMTP server> <username> <password>', dest='smtp', help='SMTP server to send thru.')
    parser.add_argument('--rsdelay', nargs=1, metavar='<# mins>', dest='rsdelay', help='Set record stop delay.')
    parser.add_argument('--version', action='version', version='%(prog)s 0.0.4')
    return parser.parse_args()


def sendmsg(fromadder, toadder, server, user, pwd):
    _MSG = MIMEText('Watchdog has made a capture.\n' + datetime.datetime.now().strftime('%A %d %B %Y %I:%M:%S%p'))
    _MSG['Subject'] = '--ALERT!--'
    _MSG['From'] = 'Watchdog'
    _MSG['To'] = toadder
    _MSG.preamble = 'Watchdog has made a capture.\n' + datetime.datetime.now().strftime('%A %d %B %Y %I:%M:%S%p')
    mailserver = smtplib.SMTP_SSL(server, 465)
    mailserver.ehlo()
    mailserver.login(user, pwd)
    mailserver.sendmail(fromadder, toadder, _MSG.as_string())
    mailserver.close()


def main():
    args = runparser()
    _CWD = os.path.abspath(os.path.curdir)
    cascPath = _CWD + '/haarcascade_frontalface_default.xml'

    # Load settings from config file.
    with open(_CWD + '/watchdog.yml') as fstream:
        settings = yaml.load(fstream)

    _FRATE = settings['defaults']['framerate']
    _RESOLUTION = (settings['defaults']['resolution']['width'], settings['defaults']['resolution']['height'])
    _RSTOPDELAY = settings['defaults']['rstopdelay']
    _SDIR = settings['defaults']['savedir'] + '/secvlog'
    _MAIL = settings['defaults']['address']
    _SMTP = settings['defaults']['smtp']
    _USER = settings['defaults']['username']
    _PWD = settings['defaults']['password']

    # Check arguments, and override defaults if necessary
    if args.save is not None:
        _SDIR = args.save[0]

    if args.frate is not None:
        _FRATE = args.frate[0]

    if args.res is not None:
        _RESOLUTION = (args.res[0], args.res[1])

    if args.send is not None:
        _MAIL = args.send[0]

    if args.rsdelay is not None:
        _RSTOPDELAY = args.rsdelay[0]

    if args.smtp is not None:
        _SMTP = args.smtp[0]
        _USER = args.smtp[1]
        _PWD = args.smtp[2]

    faceCascade = cv2.CascadeClassifier(cascPath)
    video_capture = cv2.VideoCapture(0)

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

        if not args.nsave:
            if text == 'Occupied':
                if reset:
                    stamp = '-' + datetime.datetime.now().strftime('%A %d %B %Y %I:%M:%S%p') + '.avi'
                    out = cv2.VideoWriter(_SDIR + stamp, fourcc, _FRATE, _RESOLUTION)
                    reset = False

                stop_time = datetime.datetime.now()
                out.write(frame)

            elif datetime.datetime.now().minute - stop_time.minute >= _RSTOPDELAY:
                out.release()
                if not reset:
                    sendmsg('Watchdog', _MAIL, _SMTP, _USER, _PWD)
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
    if not args.nsave:
        out.release()
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()