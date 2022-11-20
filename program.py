from signal import signal, SIGPIPE, SIG_DFL

signal(SIGPIPE,SIG_DFL)

from distutils.cmd import Command
from math import sqrt
from queue import Empty, Queue, LifoQueue

from threading import Thread
import time
import cv2
import subprocess
import pyaudio
import datetime
import os
import numpy
import imutils

import multiprocessing


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


cameraip = 0
# cameraip="rtsp://..."

(streamAud, pAud, chunk) = audio_stream()


sharedCameraQueueCaptured= multiprocessing.Queue()

shared_framequeue_preprocess= multiprocessing.Queue()

sharedIsStoped= multiprocessing.Value("i",0)
sharedVidWidth=multiprocessing.Value("i",640)
sharedVidHeight=multiprocessing.Value("i",480)
sharedVidFps=multiprocessing.Value("i",24)
sharedVidFps_g = multiprocessing.Value("i",int(sharedVidFps.value * 5))
sharedVidFps_sleep =multiprocessing.Value("d", (30 / sharedVidFps.value) * 0.03)


class App:
    def __init__(self,vid_width,vid_height,vid_fps):
        self.appIsStop = False
        
        # self.vidH=width
        # self.vidH=int(720*2)
        # self.vidW= int(int(self.vidH*width)/height)
        # self.vidCropH= int(self.vidH/4)
        # self.vidCropW= int(self.vidW/4)

        self.vidH = vid_height
        self.vidW = vid_width
        self.fps = vid_fps
        self.fps = 24

        # https://sites.google.com/site/linuxencoding/x264-ffmpeg-mapping
        self.fps_g = self.fps * 5
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

        print("display:w:h:", self.vidW, self.vidH)
        self.command = ['ffmpeg',
                        '-threads', '0',
                        '-thread_queue_size','4096',
                        '-y',                        
                         '-re',
                        '-f', 'rawvideo',  # fake video input then write image to stdin late
                        '-pixel_format', 'bgr24',
                        '-s', f"{self.vidW}x{self.vidH}",
                        # '-s','320x240',
                        '-framerate', f"{self.fps}",
                        #'-i',   f"{self.pipeVid}",
                        #'-hwaccel','auto',
                        '-i', '-',  # write image input  with stdin

                        # # slience sound
                        #'-re',
                        #'-f', 'lavfi',
                        #'-i', 'anullsrc',

                        # sound from file
                        #'-re',
                        # '-stream_loop', '-1',
                        # '-i', '1.mp3',#from file

                        # from mic
                        '-re',
                          '-f', 'alsa',
                          '-ac', '2' ,
                          '-itsoffset', '00:00:00.1',
                          '-i','default',

                        # '-re',
                        # '-f', 'lavfi',
                        # #'-i', f"{self.pipeAud}",#not working yet, try to write stream from stdin
                        # '-i','-',
                        
                        '-c:v', 'libx264',  # mp4 format
                        '-vf', 'yadif,format=yuv420p,setsar=1:1',
                        # key frame https://support.google.com/youtube/answer/2853702?hl=en#zippy=%2Cp
                        
                        '-force_key_frames', 'expr:gte(t,n_forced/2)',
                        
                       # '-force_key_frames', 'expr:gte(t,n_forced*2)'
                        
                        '-keyint_min', f"{ self.fps}",
                        '-x264opts', f"keyint={self.fps_g}:min-keyint={self.fps}:no-scenecut",
                        '-sc_threshold', '40',
                        '-reorder_queue_size', '4096',
                        '-max_delay', '500000',  # 0.5sec
                        '-pix_fmt', 'yuv420p',  # optimize mp4 format
                        #'-vf', 'format=yuv420p,setsar=1:1,scale=-1:720',
                        #'-vf', f'format=yuv420p,setsar=1:1,crop={self.vidCropW}:{self.vidCropH}:0:0',
                        # '-s','320x240',
                        '-tune', 'zerolatency',
                        '-vprofile', 'baseline',                        
                        '-preset', 'veryfast',
                        '-async', '1',
                        '-bf', '2',
                        '-use_editlist', '0',
                        # https://wiki.multimedia.cx/index.php/FFmpeg_Metadata
                        '-metadata', 'copyright=dunp',
                        '-metadata', 'author=dunp',
                        '-metadata', 'album_artist=dunp',
                        '-metadata', 'album=dunp',
                        '-metadata', 'comment=dunp',
                        '-metadata', 'title=dunp',
                        '-metadata', 'year=2010',

                        '-c:a', 'aac',
                        '-g', f"{self.fps_g}",
                        '-b:v', '2048k',  # https://support.google.com/youtube/answer/2853702?hl=en#zippy=%2Cp
                        #'-b:a', '96k',
                        '-r', f"{ self.fps}",
                        '-crf', '28',  # https://trac.ffmpeg.org/wiki/Encode/H.264
                        '-maxrate', '960k',
                        '-bufsize', '2048k',  # https://trac.ffmpeg.org/wiki/EncodingForStreamingSites
                        '-strict', 'experimental',
                        #'-strict', '-2',
                        '-movflags', '+faststart',  # support MAC os quick time to play
                        '-flvflags', 'no_duration_filesize',                        
                        '-flags', '+global_header',
                        "-x264opts","opencl",
                        
                        #'-fflags','nobuffer',
                        #'-probesize','32',
                        #'-analyzeduration','0',
                        

                        ## youtube live ok
                        #'-f', 'flv',
                         #rtmp

                        # # convert to .gif work oki                    https://superuser.com/questions/556029/how-do-i-convert-a-video-to-gif-using-ffmpeg-with-reasonable-quality
                        # '-an',
                        # ##'-vf', "scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
                        # '-f', 'image2pipe',
                        # 'convert ',
                        # '-delay', '10',
                        # '-loop', '0',
                        # "xxx.gif"

                        # # localhost live ok
                        '-f', 'mpegts',
                        "udp://127.0.0.1:7234"
                        # #    #ffplay udp://127.0.0.1:7234
                        # #    #vlc udp://@127.0.0.1:7234
                        
                        ## stream through http m3u8stream/index.html
                        # '-f','hls',
                        # '-hls_time','10',
                        # '-segment_time','10',
                        # '-hls_list_size','10',
                        # '/m3u8stream/stream.m3u8'

                        # #save to file
                        #"/work/cameraip2youtubelive/program.mp4"
                        ]
        """
        ffmpeg option 264 https://sites.google.com/site/linuxencoding/x264-ffmpeg-mapping
        https://www.wowza.com/docs/how-to-restream-using-ffmpeg-with-wowza-streaming-engine
        https://github.com/Noxalus/Multi-Streaming-Server/blob/master/nginx/conf/nginx.template.conf
        ffmpeg -f lavfi -i anullare rtsp transport udp - "etap://admin:cam@192.168.7.185:554/cam/realmonitor?channel-16subtype-0 force key frames "expr:gte(t,n_forced 2)" -vf scale-1920:1080 - reorder_queue_size 4000 -max_delay 10000000 -vcodec 11bx264 -b:v 4500k - pix_fmt yuv420p -f fly "cyoutube_stream_url>
        """
        # capture mic : ffmpeg -f alsa -ac 2 -itsoffset 00:00:00.5 -i default  -f video4linux2 -s 320x240 -r 25 -i /dev/video0 out.mpg
        # xxx=f"ffmpeg -f alsa -ac 2 -itsoffset 00:00:00.5 -i default  -f video4linux2 -s 320x240 -r 25 -i /dev/video0 -c:a aac -c:v libx264 -vf format=yuv420p,setsar=1:1 -movflags +faststart -f flv  rtmp://a.rtmp.youtube.com/live2/keytube"
        # print(xxx)
        print(' '.join(self.command))
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



# https://trac.ffmpeg.org/wiki/StreamingGuide
"""
https://gist.github.com/travelhawk/4537a79c11fa5e308e6645a0b434cf4f
"""


def cv2_draw_contours(image):
    imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # gray scale image
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    thresh, imagegray = cv2.threshold(
        imagegray, 70, 255, cv2.THRESH_BINARY_INV)

    edges = cv2.Canny(imagegray, 100, 155)

    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    image = cv2.cvtColor(imagegray, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(image, contours, -1, (255, 0, 255), 1)

    return image


def draw_overlay(frame, img, x, y, color2remove=(0, 0, 0)):
    th, tw, tc = img.shape

    if color2remove == None:
        frame[y:y+th, x:x+tw] = img
        return frame

    mth = 3*th/4
    mtw = 3*tw/4
    kth = th/4
    ktw = tw/4

    #frame=cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    # print("x:tw+x",x,tw+x)
    rth = range(th)
    rtw = range(tw)
    for i in rth:
        for j in rtw:
            iy = i+y
            jx = j+x
            if (iy > kth and jx > ktw) and (iy < mth and jx < mtw):
                frame[iy, jx, :] = img[i, j, :]
                continue

            if color2remove != None:
                if img[i, j, :][0] == color2remove[0] and img[i, j, :][1] == color2remove[1] and img[i, j, :][2] == color2remove[2]:
                    continue
                frame[iy, jx, :] = img[i, j, :]
            else:
                frame[iy, jx, :] = img[i, j, :]

    return frame


def proc_write_pipe(proc_pipe: subprocess.Popen, pipe_name, dataInBytes):
    # # https://stackoverflow.com/questions/67388548/multiple-named-pipes-in-ffmpeg

    # # Open the pipes as opening files (open for "open for writing only").
    # # fd_pipe1 is a file descriptor (an integer)
    # fd_pipe = os.open(pipe_name, os.O_WRONLY)

    # print(f"fd_pipe: {fd_pipe}")

    # os.write(fd_pipe, dataInBytes)
    # #os.close(fd_pipe)

    proc_pipe.stdin.write(dataInBytes)

    pass


def transparent_black(src):
    # Convert image to image gray
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # Applying thresholding technique
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)

    # Using cv2.split() to split channels
    # of coloured image
    b, g, r = cv2.split(src)

    # Making list of Red, Green, Blue
    # Channels and alpha
    rgba = [b, g, r, alpha]

    # Using cv2.merge() to merge rgba
    # into a coloured/multi-channeled image
    dst = cv2.merge(rgba, 4)
    return (dst, alpha)


def rotate_image(image, angle):

    #image = imutils.rotate_bound(image, angle)
    # return image
    h, w, c = image.shape
    r = int(sqrt(h*h+w*w))+2
    nx = int((r-w)/2)
    ny = int((r-h)/2)

    blank = numpy.zeros((r, r, 3), numpy.uint8)
    # blank[:,:]=(69,255,0)#green in film
    blank[ny:ny+h, nx:nx+w] = image
    image = blank

    try:
        image_center = tuple(numpy.array(image.shape[1::-1]) / 2)

        rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
        result = cv2.warpAffine(
            image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return (result, r, r)
    except:
        return (image, h, w)


def stream_youtube():
    
    
    command = ['ffmpeg',
                        '-threads', '0',
                        '-thread_queue_size','4096',
                        '-y',                        
                         '-re',
                        '-f', 'rawvideo',  # fake video input then write image to stdin late
                        '-pixel_format', 'bgr24',
                        '-s', f"{sharedVidWidth.value}x{sharedVidHeight.value}",
                        # '-s','320x240',
                        '-framerate', f"{sharedVidFps.value}",
                        #'-i',   f"{self.pipeVid}",
                        #'-hwaccel','auto',
                        '-i', '-',  # write image input  with stdin

                        # # slience sound
                        #'-re',
                        #'-f', 'lavfi',
                        #'-i', 'anullsrc',

                        # sound from file
                        #'-re',
                        # '-stream_loop', '-1',
                        # '-i', '1.mp3',#from file

                        # from mic
                        '-re',
                          '-f', 'alsa',
                          '-ac', '2' ,
                          '-itsoffset', '00:00:00.1',
                          '-i','default',

                        # '-re',
                        # '-f', 'lavfi',
                        # #'-i', f"{self.pipeAud}",#not working yet, try to write stream from stdin
                        # '-i','-',
                        
                        '-c:v', 'libx264',  # mp4 format
                        '-vf', 'yadif,format=yuv420p,setsar=1:1',
                        # key frame https://support.google.com/youtube/answer/2853702?hl=en#zippy=%2Cp
                        
                        '-force_key_frames', 'expr:gte(t,n_forced/2)',
                        
                       # '-force_key_frames', 'expr:gte(t,n_forced*2)'
                        
                        '-keyint_min', f"{sharedVidFps.value}",
                        '-x264opts', f"keyint={sharedVidFps_g.value}:min-keyint={sharedVidFps.value}:no-scenecut",
                        '-sc_threshold', '40',
                        '-reorder_queue_size', '4096',
                        '-max_delay', '500000',  # 0.5sec
                        '-pix_fmt', 'yuv420p',  # optimize mp4 format
                        #'-vf', 'format=yuv420p,setsar=1:1,scale=-1:720',
                        #'-vf', f'format=yuv420p,setsar=1:1,crop={self.vidCropW}:{self.vidCropH}:0:0',
                        # '-s','320x240',
                        '-tune', 'zerolatency',
                        '-vprofile', 'baseline',                        
                        '-preset', 'veryfast',
                        '-async', '1',
                        '-bf', '2',
                        '-use_editlist', '0',
                        # https://wiki.multimedia.cx/index.php/FFmpeg_Metadata
                        '-metadata', 'copyright=dunp',
                        '-metadata', 'author=dunp',
                        '-metadata', 'album_artist=dunp',
                        '-metadata', 'album=dunp',
                        '-metadata', 'comment=dunp',
                        '-metadata', 'title=dunp',
                        '-metadata', 'year=2010',

                        '-c:a', 'aac',
                        '-g', f"{sharedVidFps.value}",
                        '-b:v', '2048k',  # https://support.google.com/youtube/answer/2853702?hl=en#zippy=%2Cp
                        #'-b:a', '96k',
                        '-r', f"{ sharedVidFps.value}",
                        '-crf', '28',  # https://trac.ffmpeg.org/wiki/Encode/H.264
                        '-maxrate', '960k',
                        '-bufsize', '2048k',  # https://trac.ffmpeg.org/wiki/EncodingForStreamingSites
                        '-strict', 'experimental',
                        #'-strict', '-2',
                        '-movflags', '+faststart',  # support MAC os quick time to play
                        '-flvflags', 'no_duration_filesize',                        
                        '-flags', '+global_header',
                        "-x264opts","opencl",
                        
                        #'-fflags','nobuffer',
                        #'-probesize','32',
                        #'-analyzeduration','0',
                        

                        # ## youtube live ok
                        # '-f', 'flv',
                        #  rtmp

                        # # convert to .gif work oki                    https://superuser.com/questions/556029/how-do-i-convert-a-video-to-gif-using-ffmpeg-with-reasonable-quality
                        # '-an',
                        # ##'-vf', "scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
                        # '-f', 'image2pipe',
                        # 'convert ',
                        # '-delay', '10',
                        # '-loop', '0',
                        # "xxx.gif"

                        # # localhost live ok
                         '-f', 'mpegts',
                         "udp://127.0.0.1:7234"
                        # #    #ffplay udp://127.0.0.1:7234
                        # #    #vlc udp://@127.0.0.1:7234
                        
                        ## stream through http m3u8stream/index.html
                        # '-f','hls',
                        # '-hls_time','10',
                        # '-segment_time','10',
                        # '-hls_list_size','10',
                        # '/m3u8stream/stream.m3u8'

                        # #save to file
                        #"/work/cameraip2youtubelive/program.mp4"
                        ]

    lastframe = cv2.imread("du.png")

    lastframe = cv2.resize(lastframe, (sharedVidWidth.value, sharedVidHeight.value), cv2.INTER_CUBIC)
    
    qsize= shared_framequeue_preprocess.qsize()
    print("shared_framequeue_preprocess", qsize)    
    for i in range(qsize):
        shared_framequeue_preprocess.get()        
        
    qsize= sharedCameraQueueCaptured.qsize()
    print("sharedCameraQueueCaptured", qsize)    
    for i in range(qsize):
        sharedCameraQueueCaptured.get()

    print("reset done -> begin stream")
    
    print(command)
    
    pipeFfmpegProc = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE,
                                      #stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
                                      )
    
    time.sleep(1)
    
    dtcheckEmptyQueue=datetime.datetime.now()

    while sharedIsStoped.value == 0:
        try:
            
            qsize = shared_framequeue_preprocess.qsize()
            
            if qsize <= 0:
                frame = lastframe
                # dtnow= datetime.datetime.now()
                # timedif= dtnow- dtcheckEmptyQueue
                # if timedif.total_seconds()>3:
                #     dtcheckEmptyQueue=datetime.datetime.now()
                #     frame = lastframe
            else:
                rem = qsize - sharedVidFps.value
                if rem > 0:
                    print(
                        f"check to remove: app.frame/process: {sharedCameraQueueCaptured.qsize()} / {shared_framequeue_preprocess.qsize()}")

                    for i in range(rem):
                        shared_framequeue_preprocess.get()

                frame = shared_framequeue_preprocess.get()

            #proc_write_pipe(pipeFfmpegProc, "pipeVid.mp4", frame.tobytes())            
            #proc_write_pipe(pipeFfmpegProc, app.pipeAud, streamAud.read(chunk))
            stdinBytes=frame.tobytes()
            if len(stdinBytes)==0:
                print("stream_youtube", "no bytes")
                continue
            pipeFfmpegProc.stdin.write(stdinBytes)

            lastframe = frame

            #cv2.imshow("", frame)
            # cv2.waitKey(20)
            time.sleep(sharedVidFps_sleep.value)  # keep fps
        except Exception as ex:
            print("stream_youtube", ex)
            
            pass

    pipeFfmpegProc.terminate()


def video_preprocess_zoom(frameW,frameH, zoomQueue: Queue):
    du = cv2.imread("du.png")
    du = cv2.resize(du, (120, 126), cv2.INTER_LINEAR)    
    tw=120
    th=126
    ts=10
    td=-1
    mw=10
    mh=40
    while app.appIsStop == False:
        if zoomQueue.qsize()>1000:
            time.sleep(1)
            continue
        
        if tw>=120:
            td=-1
        if tw<=10:
            td=1
            
        tw=tw+td*ts
        th=th+td*ts
        
        mw=mw-td*5
        mh=mh-td*5
        img= cv2.resize(du,(tw,th), cv2.INTER_LINEAR)
                
        zoomQueue.put((img, mh,mw))
        pass

def video_preprocess_rotate(frameW,frameH, rotateQueue: Queue):

    du = cv2.imread("du.png")
    du = cv2.resize(du, (120, 126))
    print("du.png", du.shape)
    angle = 1
    while app.appIsStop == False:
        if rotateQueue.qsize()>1000:
            time.sleep(1)
            continue
        (img, hi, wi) = rotate_image(du, angle)
        #(img, alpha_data)= transparent_black(img)

        if angle >= 360:
            angle = 0
        angle = angle+1
        
        rotateQueue.put((img, 0,frameW- wi))

        pass


def video_preprocess():


    zoomQueue = Queue()
    rotateQueue = Queue()

    # trotate = Thread(target=video_preprocess_rotate,
    #                  args=(app.vidW,app.vidH, rotateQueue,), daemon=True)
    # #trotate.start()
    
    
    # tzoom = Thread(target=video_preprocess_zoom,
    #                  args=(app.vidW,app.vidH, zoomQueue,), daemon=True)
    # #tzoom.start()
    
    while sharedIsStoped.value == 0:
        
        frame = sharedCameraQueueCaptured.get()
        
        try:
            
            frame = cv2_draw_contours(frame)
                    
            # try:
            #     (imgr, hir, wir) = rotateQueue.get()
            #     draw_overlay(frame, imgr, wir, hir, (0, 0, 0))
            # except:
            #     pass
            
            # try:
            #     (imgz, hiz, wiz) = zoomQueue.get()
            #     draw_overlay(frame, imgz, wiz, hiz, None)
            # except:
                # pass
            
            #frame=cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            #frame=cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            tnow = datetime.datetime.now()
            cv2.putText(frame, f"Nguyen Phan Du {tnow}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f"Nguyen Phan Du {tnow}", (11, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (69, 255, 0), 1)
        except Exception as ex:
            print(ex)
            pass

        shared_framequeue_preprocess.put(frame)

        # print(
        #          f"app.frame/process: {sharedCameraQueueCaptured.qsize()} / {shared_framequeue_preprocess.qsize()}")

        #time.sleep(0.01)
        pass


def video_capture():

    cap = cv2.VideoCapture(cameraip)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print("org camera info:w:h:fps:", width, height, fps)

    width=640
    height=480
    fps=24
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
        
    sharedVidWidth.value= width
    sharedVidHeight.value= height
    sharedVidFps.value=fps

    print("camera info:w:h:fps:", width, height, fps)
    
    while sharedIsStoped.value == 0:
        try:
            if cap.isOpened() == False:
                time.sleep(1)
                continue

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     sharedIsStoped.value=1
            #     break

            for i in range(3):
                success, frame = cap.read()

            if success:
                #frame= cv2.flip(frame,1)

                #app.framequeue.put(frame)
                sharedCameraQueueCaptured.put(frame)
                # pipe.communicate(frame.tobytes())
                # pipe.stdin.write(streamAud.read(chunk))

                #cv2.imshow("123", frame)
                # cv2.waitKey(1)
                
            #time.sleep(app.fps_sleep)  # keep fps
            #time.sleep(sharedVidFps_sleep.value) 
        except Exception as ex:
            print(ex)
            pass
    # streamAud.stop_stream()
    # streamAud.close()
    # pAud.terminate()

    cap.release()



tvid =  multiprocessing.Process(target=video_capture, args=(), daemon=True)

tvid.start()

time.sleep(2)

tyoutube = multiprocessing.Process(target=stream_youtube, args=(), daemon=True)

tyoutube.start()

app = App(sharedVidWidth.value, sharedVidHeight.value, sharedVidFps.value)

tpreprocess = Thread(target=video_preprocess, args=(), daemon=True)

tpreprocess.start()


tvid.join()

# https://stackoverflow.com/questions/68527762/pipe-opencv-and-pyaudio-to-ffmpeg-streaming-youtube-rtmp-from-python
#https://sonnati.wordpress.com/2011/08/30/ffmpeg-%e2%80%93-the-swiss-army-knife-of-internet-streaming-%e2%80%93-part-iv/