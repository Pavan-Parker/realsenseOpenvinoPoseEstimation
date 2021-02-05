# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import collections
from functools import partial
import re
import time

import numpy as np
from PIL import Image
from numpy.core.records import _deprecate_shape_0_as_None
from numpy.lib.function_base import average
import svgwrite
import gstreamer

from pose_engine import PoseEngine


################################
import pyrealsense2 as rs
import cv2
import math
#from  sympy import *

#################################
 
EDGES = (
#    ('nose', 'left eye'),
#    ('nose', 'right eye'),
#    ('nose', 'left ear'),
#    ('nose', 'right ear'),
#    ('left ear', 'left eye'),
#    ('right ear', 'right eye'),
#    ('left eye', 'right eye'),
    ('left shoulder', 'right shoulder'),
    ('left shoulder', 'left elbow'),
    ('left shoulder', 'left hip'),
    ('right shoulder', 'right elbow'),
    ('right shoulder', 'right hip'),
    ('left elbow', 'left wrist'),
    ('right elbow', 'right wrist'),
    ('left hip', 'right hip')
#    ('left hip', 'left knee'),
#    ('right hip', 'right knee'),
#    ('left knee', 'left ankle'),
#    ('right knee', 'right ankle'),
)

elbowAngles=(
    ('right shoulder','right elbow','right wrist'),
    ('left shoulder','left elbow','left wrist')
)
shoulderAngles=(
    ('right hip','left shoulder','right elbow','right shoulder'),
    ('left hip','right shoulder','left elbow','left shoulder')

)
timestamps=[]
checkfpsNow=0
fps=0
np.seterr(divide='ignore', invalid='ignore') 

def angle(v1, v2, acute):
# v1 is your firsr vector
# v2 is your second vector
    oldSettings=np.seterr(all='ignore')
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    try:
        return 180-round(angle*(180.0/np.pi))
    except ValueError:
        return -1
def projectedAngle(v1,hip,shoulder2,shoulder1):
## v1 is your firsr vector
## v2 is your second vector
    angles=["NaN","NaN"]
    oldSettings=np.seterr(all='ignore')
    l1=shoulder1[:-1]-hip[:-1]
    l2=shoulder1[1:]-hip[1:]
    #v2=np.cross(l1,l2)
##    v11,v12=v1,v1
    angle1 = np.arccos(np.dot(v1[:-1], l1) / (np.linalg.norm(v1[:-1]) * np.linalg.norm(l1)))
    angle2 = np.arccos(np.dot(v1[1:],l2)  /  (np.linalg.norm(v1[1:]) * np.linalg.norm(l2)))
    try:
        angles[0]= 180-round(angle1*(180.0/np.pi))
    except ValueError:
        pass
    try:
        angles[1]= 180-round(angle2*(180.0/np.pi))
    except ValueError:
        pass
    
    """
    if(not math.isnan(angle1)):
        ret= 180-round(angle1*(180.0/np.pi))
        if(not math.isnan(ret)):angles[0]=ret
    if(not math.isnan(angle2)):
        ret= 180-round(angle2*(180.0/np.pi))
        if(not math.isnan(ret)):angles[1]=ret
    """
    return angles
"""    
def projectedAngle(v1,hip,shoulder2,shoulder1):
## v1 is your firsr vector
## v2 is your second vector
#    angles=["NaN","NaN"]
    oldSettings=np.seterr(all='ignore')
#    l1=shoulder1-hip
#    l2=shoulder2-hip
#    v2=np.cross(l1,l2)
##    v11,v12=v1,v1
    plane1=Plane(Point3D(shoulder1),Point3D(shoulder2),normal_vector=hip)
    plane2=Plane(Point3D(hip),Point3D(shoulder1),normal_vector=shoulder2)
    v1=Line3D(Point3D(shoulder1),(Point3D(v1)))
    angle1=N(deg(plane1.angle_between(v1)),2)
    angle2=N(deg(plane2.angle_between(v1)),2)
#    angle1 = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
#    angle2=np.arccos(np.dot(l2-l1,v1) / (np.linalg.norm(v1) * np.linalg.norm(l2-l1)))
    angles=[angle1,angle2]

    if(angle1 is not np.nan):
        ret= 180-round(angle1*(180.0/np.pi))
        if(ret is not np.nan):angles[0]=ret
    if(angle2 is not np.nan):
        ret= 90-round(angle2*(180.0/np.pi))
        if(ret is not np.nan):angles[1]=ret
    return angles
"""    

def draw_pose(dwg, pose, src_size, inference_box, depthFrame,intrinsics,color='yellow', threshold=0.2):
    global timestamps
    global checkfpsNow
    global fps
    box_x, box_y, box_w, box_h = inference_box
    scale_x, scale_y = src_size[0] / box_w, src_size[1] / box_h
    xys = {}
    cv2.namedWindow('motion', cv2.WINDOW_AUTOSIZE)

#    print(pose)
#    print(dwg.shape)
#    print(len(depthFrame[0]))
    realXYZ={}
    oldPoints={}
    for label, keypoint in pose.keypoints.items():
        if keypoint.score < threshold : continue
        # Offset and scale to source coordinate space.
        #print(keypoint.yx[0],keypoint.yx[1])
        
        kp_y = int((keypoint.yx[0] - box_y) * scale_y)
        kp_x = int((keypoint.yx[1] - box_x) * scale_x)
        #print(kp_x,kp_y)
        scale=10
        kp_x=int(kp_x-(kp_x%scale))
        kp_y=int(kp_y-(kp_y%scale))
        xys[label] = (kp_x, kp_y)
        dwg=cv2.circle(dwg,(int(kp_x), int(kp_y)), 3,(0,0,255),5)
        realX,realY,realZ=rs.rs2_deproject_pixel_to_point(intrinsics,[kp_y,kp_x],depthFrame[int(kp_y)-1][int(kp_x)])
        realXYZ[label]=[realX,realY,realZ]

        #dwg=cv2.putText(dwg,"({0},{1},{2})".format(round(realY),round(realX),round(realZ)),(int(kp_x),int(kp_y) ),(cv2.FONT_HERSHEY_COMPLEX),0.6,(0,255,0),1,cv2.LINE_AA)
#        dwg=cv2.putText(dwg,"({0},{1},{2})".format((kp_x),(kp_y),str(depthFrame[int(kp_y)-1][int(kp_x)])),(int(kp_x),int(kp_y) ),(cv2.FONT_HERSHEY_COMPLEX),0.6,(0,255,0),1,cv2.LINE_AA)
    for a, b in EDGES:
        if a not in xys or b not in xys: continue
        ax, ay = xys[a]
        bx, by = xys[b]
        dwg=cv2.line(dwg,(ax, ay),(bx, by), (0,255,0),2)

    for A,B,C,D in shoulderAngles:
        if  B not in xys or C not in xys or D not in xys: continue
        if("right hip"  in xys): A="right hip"
        elif("left hip"  in xys): A="left hip"
        else: continue
#        print(list(xys[A])[1],xys[B],xys[C])
#       print((depthFrame[list(xys[A])[1]-1] [list(xys[A])[0]]))
#        a,b,c,d=list(map(lambda p: ((np.asarray(xys[p]+tuple([depthFrame[list(xys[p])[1]-1][list(xys[p])[0]]])))),[A,B,C,D]))
        a,b,c,d=list(map(lambda p: np.asarray(realXYZ[p]),[A,B,C,D]))
#        print(a,b,c)
        ANGLES=projectedAngle(c-d,a,b,d)
        dwg=cv2.putText(dwg,str(ANGLES),(xys[D]),(cv2.FONT_HERSHEY_COMPLEX),0.6,(0,255,0),1,cv2.LINE_AA)
    
    for A,B,C in elbowAngles:
        if A not in xys or B not in xys or C not in xys: continue
#        print(list(xys[A])[1],xys[B],xys[C])
#        print((depthFrame[list(xys[A])[1]-1] [list(xys[A])[0]]))
#        realX,realY,realZ=rs.rs2_deproject_pixel_to_point(intrinsics,[kp_y,kp_x],depthFrame[int(kp_y)-1][int(kp_x)])
#        a,b,c=list(map(lambda p: ((np.asarray(xys[p]+tuple([depthFrame[list(xys[p])[1]-1][list(xys[p])[0]]])))),[A,B,C]))
        a,b,c=list(map(lambda p: np.asarray(realXYZ[p]),[A,B,C]))

#        a,b,c=list(map(lambda p: ((np.asarray(xys[p]+tuple([depthFrame[list(xys[p])[1]-1][list(xys[p])[0]]])))),[A,B,C]))
#        print(a,b,c)
        ANGLE=str(angle(b-a,c-b,True))
        if(ANGLE!=-1):
            dwg=cv2.putText(dwg,ANGLE,(xys[B]),(cv2.FONT_HERSHEY_COMPLEX),1,(0,255,0),2,cv2.LINE_AA)
    
    if(len(timestamps)>30 and checkfpsNow>5):
        checkfpsNow=0
        fps=1/average(timestamps)
    checkfpsNow+=1
    dwg=cv2.putText(dwg,str(round(fps))+' fps',(10,20),(cv2.FONT_HERSHEY_COMPLEX),1,(255,0,0),2,cv2.LINE_AA)

    cv2.imshow('motion', dwg)
                 
def mine():
    global timestamps
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mirror', help='flip video horizontally', action='store_true')
    parser.add_argument('--model', help='.tflite model path.', required=False)
    parser.add_argument('--res', help='Resolution', default='640x480',
                        choices=['480x360', '640x480', '1280x720'])
    parser.add_argument('--videosrc', help='Which video source to use', default='/dev/video0')
    parser.add_argument('--h264', help='Use video/x-h264 input', action='store_true')
    parser.add_argument('--jpeg', help='Use image/jpeg input', action='store_true')
    args = parser.parse_args()

    default_model = 'models/mobilenet/posenet_mobilenet_v1_075_%d_%d_quant_decoder_edgetpu.tflite'
    if args.res == '480x360':
        src_size = (640, 480)
        appsink_size = (480, 360)
        model = args.model or default_model % (353, 481)
    elif args.res == '640x480':
        src_size = (640, 480)
        appsink_size = (640, 480)
        model = args.model or default_model % (481, 641)
    elif args.res == '1280x720':
        src_size = (1280, 720)
        appsink_size = (1280, 720)
        model = args.model or default_model % (721, 1281)
####################################
    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, src_size[0],  src_size[1], rs.format.z16, 30)
    config.enable_stream(rs.stream.color,  src_size[0],  src_size[1], rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)
####################################
    print('Loading model: ', model)
    engine = PoseEngine(model)
    input_shape = engine.get_input_tensor_shape()
    inference_size = (input_shape[2], input_shape[1])
####################################    
    try:
        while(True):
            start=time.time()
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue
            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            poses, inference_time = engine.DetectPosesInImage(color_image)
            for pose in poses:    
                draw_pose(color_image,pose,src_size,(0,0, src_size[0]+1, src_size[1]),depth_image,depth_intrin)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            end=time.time()
            #print(end-start)
            timestamps.append(end-start)
            if(len(timestamps)>45):
                del timestamps[0]
    finally:
        pipeline.stop()
###################################
"""
def main():
    n = 0
    sum_process_time = 0
    sum_inference_time = 0
    ctr = 0
    fps_counter  = avg_fps_counter(30)
    def run_inference(engine, input_tensor):
        return engine.run_inference(input_tensor)

    def render_overlay(engine, output, src_size, inference_box):
        nonlocal n, sum_process_time, sum_inference_time, fps_counter

        svg_canvas = svgwrite.Drawing('', size=src_size)
        start_time = time.monotonic()
        outputs, inference_time = engine.ParseOutput(output)
        end_time = time.monotonic()
        n += 1
        sum_process_time += 1000 * (end_time - start_time)
        sum_inference_time += inference_time

        avg_inference_time = sum_inference_time / n
        text_line = 'PoseNet: %.1fms (%.2f fps) TrueFPS: %.2f Nposes %d' % (
            avg_inference_time, 1000 / avg_inference_time, next(fps_counter), len(outputs)
        )

        shadow_text(svg_canvas, 10, 20, text_line)
        for pose in outputs:
            draw_pose(svg_canvas, pose, src_size, inference_box)
        return (svg_canvas.tostring(), False)

    run(run_inference, render_overlay)

"""
if __name__ == '__main__':
    mine()
