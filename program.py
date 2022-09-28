from distutils.cmd import Command
from queue import Empty, Queue,LifoQueue

from re import T
from threading import Thread
import time
import cv2
import subprocess
import pyaudio
import datetime


def audio_stream():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ",
                  p.get_device_info_by_host_api_device_index(0, i).get('name'))

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    return (stream, p, CHUNK)


def cv2_draw_contours(image):
    imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # gray scale image
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    thresh, imagegray = cv2.threshold(
        imagegray, 70, 255, cv2.THRESH_BINARY_INV)

    edges = cv2.Canny(imagegray, 100, 200)

    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #imagegray= cv2.cvtColor( imagegray, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(image, contours, -1, (255, 0, 255), 2)
    
    cv2.putText(image, f"Nguyen Phan Du {datetime.datetime.now()}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    return image

rtmp = f'rtmp://a.rtmp.youtube.com/live2/xxx'

cameraip = 0
# cameraip="rtsp://..."

(streamAud, pAud, chunk) = audio_stream()

cap = cv2.VideoCapture(cameraip)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# width=480
# height=320

print(width, height, fps)


class App:
    def __init__(self):
        self.appIsStop = False
        self.framequeue = Queue()
        self.framequeue_preprocess = Queue()

        # self.vidH=width
        # self.vidH=int(720*2)
        # self.vidW= int(int(self.vidH*width)/height)
        # self.vidCropH= int(self.vidH/4)
        # self.vidCropW= int(self.vidW/4)

        self.vidH = height
        self.vidW = width
        self.fps = fps
        self.fps = 24

        self.fps_g = self.fps * 2
        self.fps_sleep = (30 / self.fps) * 0.03

        print(self.vidW, self.vidH)
        self.command = ['ffmpeg',
                        '-threads', '0',
                        '-y',
                        '-re',
                        '-f', 'rawvideo',
                        '-pixel_format', 'bgr24',
                        '-s', f"{self.vidW}x{self.vidH}",
                        # '-s','320x240',
                        '-framerate', f"{self.fps}",
                        #'-i', 'pipe:0',
                        '-i', '-',
                        '-stream_loop', '-1',
                        '-i', '1.mp3',
                        '-c:v', 'libx264',
                        '-vf', 'format=yuv420p,setsar=1:1,scale=-1:720',
                        #'-vf', 'format=yuv420p,setsar=1:1,scale=-1:720',
                        #'-vf', f'format=yuv420p,setsar=1:1,crop={self.vidCropW}:{self.vidCropH}:0:0',
                        # '-s','320x240',
                        '-tune', 'zerolatency',
                        '-vprofile', 'baseline',
                        '-preset', 'veryfast',
                        '-async', '1',
                        '-c:a', 'aac',
                        '-g', f"{self.fps_g}",
                        '-b:v', '1984k',
                        '-b:a', '96k',
                        '-r',f"{self.fps}",
                        '-crf', '28',  # https://trac.ffmpeg.org/wiki/Encode/H.264
                        '-maxrate', '960k',
                        '-bufsize', '1920k',  # https://trac.ffmpeg.org/wiki/EncodingForStreamingSites
                        #'-strict', 'experimental',
                        '-movflags', '+faststart',
                        '-flvflags', 'no_duration_filesize',
                        '-flags','+global_header',

                         '-f', 'flv',
                         rtmp

                        # '-an',
                        #"-f" ,"rtp",
                        # "rtp://127.0.0.1:7234"
                        # ffplay rtp://127.0.0.1:7234

                         #"/work/cameraip2youtubelive/program.mp4"
            ]
        
        print(self.command )
        pass


app = App()

# https://trac.ffmpeg.org/wiki/StreamingGuide
"""
https://gist.github.com/travelhawk/4537a79c11fa5e308e6645a0b434cf4f
"""

def stream_youtube():

    pipe = subprocess.Popen(app.command, shell=False, stdin=subprocess.PIPE,
                            #stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
                            )
    lastframe = cv2.imread("du.png")
    
    lastframe=cv2.resize(lastframe,(app.vidW, app.vidH), cv2.INTER_CUBIC)

    while app.appIsStop == False:
        try:
            qsize = app.framequeue_preprocess.qsize()

            if qsize <= 0:
                frame = lastframe
            else:                
                rem = qsize - 24
                if rem > 0:
                    print(
                        f"app.frame/process: {app.framequeue.qsize()} / {app.framequeue_preprocess.qsize()}")
                    
                    for i in range(rem):
                        app.framequeue_preprocess.get()
                
                frame = app.framequeue_preprocess.get()
            
            pipe.stdin.write(frame.tobytes())
            lastframe = frame

            #cv2.imshow("", frame)
            # cv2.waitKey(20)
            time.sleep(app.fps_sleep)  # keep fps
        except Exception as ex:
            print(ex)
            pass

    pipe.terminate()

def video_preprocess():

    while app.appIsStop == False:
        frame = app.framequeue.get()


        frame = cv2_draw_contours(frame)

        # frame = cv2.resize(frame, (app.vidH, app.vidH),
        #                      interpolation=cv2.INTER_AREA)

        app.framequeue_preprocess.put(frame)

        time.sleep(0.01)
        pass


def video_capture():

    while app.appIsStop == False:
        try:
            if cap.isOpened() == False:
                time.sleep(1)
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                app.appIsStop = True
                break

            for i in range(5):
                success, frame = cap.read()

            if success:
                app.framequeue.put(frame)
                # pipe.communicate(frame.tobytes())
                # pipe.stdin.write(streamAud.read(chunk))
                cv2.imshow("123",frame)
                cv2.waitKey(1)
        except Exception as ex:
            print(ex)
            pass
    # streamAud.stop_stream()
    # streamAud.close()
    # pAud.terminate()

    cap.release()


tyoutube = Thread(target=stream_youtube, args=(), daemon=True)
tvid = Thread(target=video_capture, args=(), daemon=True)
tpreprocess = Thread(target=video_preprocess, args=(), daemon=True)


tvid.start()

tpreprocess.start()

tyoutube.start()

tvid.join()

# https://stackoverflow.com/questions/68527762/pipe-opencv-and-pyaudio-to-ffmpeg-streaming-youtube-rtmp-from-python
