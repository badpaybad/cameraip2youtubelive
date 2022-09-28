from distutils.cmd import Command
from queue import Empty, Queue, LifoQueue

from re import T
from threading import Thread
import time
import cv2
import subprocess
import pyaudio
import datetime
import os


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
        
        self.pipeVid = "pipeVid.mp4"
        self.pipeAud = "pipeAud.mp3"        
        
        self.UnlinkPipe()
        
        try:
            os.mkfifo(self.pipeVid)
        except:
            pass
        try:
            os.mkfifo(self.pipeAud)
        except:
            pass

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
                        #'-i',   f"{self.pipeVid}",
                        '-i', '-',

                        '-stream_loop', '-1',
                        '-i', '1.mp3',
                        # '-f', 'alsa',
                        # '-ac', '2' ,
                        # '-itsoffset', '00:00:00.1',
                        # '-i','default',
                        # '-re',
                        # '-f', 'lavfi',
                        # #'-i', f"{self.pipeAud}",
                        # '-i','-',

                        '-c:v', 'libx264',
                        '-vf', 'format=yuv420p,setsar=1:1',
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
                        #'-b:a', '96k',
                        '-r', f"{self.fps}",
                        '-crf', '28',  # https://trac.ffmpeg.org/wiki/Encode/H.264
                        '-maxrate', '960k',
                        '-bufsize', '1920k',  # https://trac.ffmpeg.org/wiki/EncodingForStreamingSites
                        #'-strict', 'experimental',
                        '-strict', '-2',
                        '-movflags', '+faststart',
                        '-flvflags', 'no_duration_filesize',
                        '-flags', '+global_header',

                        # '-f', 'flv',
                        # rtmp

                        # '-an',
                        #"-f" ,"rtp",
                        # "rtp://127.0.0.1:7234"
                        # ffplay rtp://127.0.0.1:7234

                        "/work/cameraip2youtubelive/program.mp4"
                        ]

        # capture mic : ffmpeg -f alsa -ac 2 -itsoffset 00:00:00.5 -i default  -f video4linux2 -s 320x240 -r 25 -i /dev/video0 out.mpg
        # xxx=f"ffmpeg -f alsa -ac 2 -itsoffset 00:00:00.5 -i default  -f video4linux2 -s 320x240 -r 25 -i /dev/video0 -c:a aac -c:v libx264 -vf format=yuv420p,setsar=1:1 -movflags +faststart -f flv  rtmp://a.rtmp.youtube.com/live2/keytube"
        # print(xxx)
        print(self.command)
        pass
    
    def UnlinkPipe(self):       
        
        try:
            os.unlink(self.pipeVid)
        except:
            pass
        try:
            os.unlink(self.pipeAud)
        except:
            pass
       
    def __del__(self):
        self.UnlinkPipe()

app = App()

# https://trac.ffmpeg.org/wiki/StreamingGuide
"""
https://gist.github.com/travelhawk/4537a79c11fa5e308e6645a0b434cf4f
"""


def proc_write_pipe(proc_pipe, pipe_name, dataInBytes):
    # # https://stackoverflow.com/questions/67388548/multiple-named-pipes-in-ffmpeg
    
    # # Open the pipes as opening files (open for "open for writing only").
    # # fd_pipe1 is a file descriptor (an integer)
    # fd_pipe = os.open(pipe_name, os.O_WRONLY)
    
    # print(f"fd_pipe: {fd_pipe}")

    # os.write(fd_pipe, dataInBytes)
    # #os.close(fd_pipe)
    
    proc_pipe.stdin.write(dataInBytes)
    

    pass


def stream_youtube():

    pipeFfmpegProc = subprocess.Popen(app.command, shell=False, stdin=subprocess.PIPE, 
                                      #stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
                                      )
    lastframe = cv2.imread("du.png")

    lastframe = cv2.resize(lastframe, (app.vidW, app.vidH), cv2.INTER_CUBIC)

    while app.appIsStop == False:
        try:
            qsize = app.framequeue_preprocess.qsize()

            if qsize <= 0:
                frame = lastframe
            else:
                rem = qsize - app.fps
                if rem > 0:
                    print(
                        f"check to remove: app.frame/process: {app.framequeue.qsize()} / {app.framequeue_preprocess.qsize()}")

                    for i in range(rem):
                        app.framequeue_preprocess.get()

                frame = app.framequeue_preprocess.get()

            proc_write_pipe(pipeFfmpegProc, app.pipeVid, frame.tobytes())
            #proc_write_pipe(pipeFfmpegProc, app.pipeAud, streamAud.read(chunk))

            lastframe = frame

            #cv2.imshow("", frame)
            # cv2.waitKey(20)
            time.sleep(app.fps_sleep)  # keep fps
        except Exception as ex:
            print(ex)
            raise(ex)
            pass

    pipeFfmpegProc.terminate()


def video_preprocess():

    while app.appIsStop == False:
        frame = app.framequeue.get()

        frame = cv2_draw_contours(frame)

        # frame = cv2.resize(frame, (app.vidH, app.vidH),
        #                      interpolation=cv2.INTER_AREA)

        app.framequeue_preprocess.put(frame)

        # print(
        #     f"app.frame/process: {app.framequeue.qsize()} / {app.framequeue_preprocess.qsize()}")

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
                cv2.imshow("123", frame)
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
