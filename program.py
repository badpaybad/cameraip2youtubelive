from distutils.cmd import Command
import cv2
import subprocess
import pyaudio


def audio_stream():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
            if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    return (stream,p, CHUNK)

def cv2_draw(image  ):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #gray scale image
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    thresh, image = cv2.threshold(image, 125, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(image, 100, 300)
    contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    for idx, c in enumerate( contours):
        cv2.drawContours(image, contours, idx, (0, 230, 255), 1)
    
    
    
    return image

rtmp = f'rtmp://a.rtmp.youtube.com/live2/youkey'


cameraip=0 
## cameraip="rtsp://..."

(streamAud,pAud, chunk)= audio_stream()

cap = cv2.VideoCapture(cameraip)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(width, height,fps)
# width=1280
# height=720
# fps=16

command = ['ffmpeg',
           '-y',
           '-f', 'rawvideo',
           '-pixel_format', 'bgr24',
           '-video_size', f"{width}x{height}",
           '-framerate', str(fps),
           #'-re',           
           '-i', 'pipe:0',
           '-preset', 'veryfast' ,
           '-maxrate', '500k',
           '-bufsize', '2500k',
           #'-re',
           '-f', 'lavfi',
           #'-i', 'pipe:1',
           '-c:v', 'libx264',
           #'-c:a', 'aac',
           '-vf', 'format=yuv420p',
           '-f', 'flv',
           rtmp]


# command=[
    
#     'ffmpeg',
#     '-re',
#     '-video_size', f"{width}x{height}",
#     '-framerate', str(fps),
#     '-i' ,'pipe:0' ,
#     '-vcodec', 'libx264' ,
#     '-preset', 'veryfast' ,
#     '-maxrate', '500k',
#     '-bufsize', '2500k',
#     '-vf', "format=yuv420p",
#     '-g' ,'60' ,
#     #-acodec libmp3lame -b:a 198k -ar 44100 
#     '-f' ,'flv' ,
#     rtmp    
# ]

pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

while cap.isOpened():
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    success, frame = cap.read()
    if success:
        frame= cv2_draw(frame)
        
        pipe.stdin.write(frame.tobytes())
        #pipe.stdin.write(streamAud.read(chunk))
        
        cv2.imshow("cam0", frame)
        cv2.waitKey(1)
        
#streamAud.stop_stream()
#streamAud.close()
#pAud.terminate()

cap.release()
pipe.terminate()
        