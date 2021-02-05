import os
from os import system
import io
import time
from os.path import isfile
from os.path import join
import argparse
import platform
import sys
from functools import partial
import re
import time

import numpy as np
from numpy.lib.function_base import average

################################
import pyrealsense2 as rs
import cv2
import math
#################################
np.seterr(divide='ignore', invalid='ignore') 

try:
    from armv7l.openvino.inference_engine import IENetwork, IEPlugin
except:
    from openvino.inference_engine import IENetwork, IEPlugin



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

                
def mine():
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
def getKeypoints(probMap, threshold=0.1):

    mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)
    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []
    contours = None
    try:
        #OpenCV4.x
        contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        #OpenCV3.x
        _, contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints


def getValidPairs(outputs, w, h):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7

    for k in range(len(mapIdx)):
        pafA = outputs[0, mapIdx[k][0], :, :]
        pafB = outputs[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (w, h))
        pafB = cv2.resize(pafB, (w, h))

        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            valid_pairs.append(valid_pair)
        else:
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs


def getPersonwiseKeypoints(valid_pairs, invalid_pairs):
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints


fps = ""
detectfps = ""
framecount = 0
time1 = 0

camera_width = 320
camera_height = 240
src_size=[camera_height,camera_width]
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9], [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16], [0,15], [15,17], [2,17], [5,16]]
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], [55,56], [37,38], [45,46]]
colors = [[0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255], [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255], [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]
"""""
cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
"""""
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", help="Specify the target device to infer on; CPU, GPU, MYRIAD is acceptable. (Default=CPU)", default="CPU", type=str)
parser.add_argument("-b", "--boost", help="Setting it to True will make it run faster instead of sacrificing accuracy. (Default=False)", default=False, type=bool)
args = parser.parse_args()

plugin = IEPlugin(device=args.device)

if "CPU" == args.device:
    if platform.processor() == "x86_64":
        plugin.add_cpu_extension("lib/libcpu_extension.so")
    if args.boost == False:
        model_xml = "models/train/test/openvino/mobilenet_v2_1.4_224/FP32/frozen-model.xml"
    else:
        model_xml = "models/train/test/openvino/mobilenet_v2_0.5_224/FP32/frozen-model.xml"

elif "GPU" == args.device or "MYRIAD" == args.device:
    if args.boost == False:
        model_xml = "models/train/test/openvino/mobilenet_v2_1.4_224/FP16/frozen-model.xml"
    else:
        model_xml = "models/train/test/openvino/mobilenet_v2_0.5_224/FP16/frozen-model.xml"

else:
    print("Specify the target device to infer on; CPU, GPU, MYRIAD is acceptable.")
    sys.exit(0)

model_bin = os.path.splitext(model_xml)[0] + ".bin"
net = IENetwork(model=model_xml, weights=model_bin)
input_blob = next(iter(net.inputs))
exec_net = plugin.load(network=net)
inputs = net.inputs["image"]

h = inputs.shape[2] #368
w = inputs.shape[3] #432

threshold = 0.1
nPoints = 18
####################################
# Create a pipeline
pipeline = rs.pipeline()
src_size=[480,640]
#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, src_size[1],  src_size[0], rs.format.z16, 30)
config.enable_stream(rs.stream.color,  src_size[1],  src_size[0], rs.format.bgr8, 30)

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
try:
    while True:
        t1 = time.perf_counter()
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

        #if not ret:
        #    break

        colw = color_image.shape[1]
        colh = color_image.shape[0]
        new_w = int(colw * min(w/colw, h/colh))
        new_h = int(colh * min(w/colw, h/colh))

        resized_image = cv2.resize(color_image, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
        resized_image_depth=cv2.resize(depth_image, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
        canvas = np.full((h, w, 3), 128)
        canvas_depth = np.full((h, w, 1), 128)
        canvas[(h - new_h)//2:(h - new_h)//2 + new_h,(w - new_w)//2:(w - new_w)//2 + new_w, :] = resized_image
        canvas_depth[(h - new_h)//2:(h - new_h)//2 + new_h,(w - new_w)//2:(w - new_w)//2 + new_w, :] = resized_image_depth

        prepimg = canvas
        prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
        prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW, (1, 3, 368, 432)
        outputs = exec_net.infer(inputs={input_blob: prepimg})["Openpose/concat_stage7"]

        detected_keypoints = []
        keypoints_list = np.zeros((0, 3))
        keypoint_id = 0

        for part in range(nPoints):
            probMap = outputs[0, part, :, :]
            probMap = cv2.resize(probMap, (canvas.shape[1], canvas.shape[0])) # (432, 368)
            keypoints = getKeypoints(probMap, threshold)
            keypoints_with_id = []

            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                keypoint_id += 1

            detected_keypoints.append(keypoints_with_id)

        frameClone = np.uint8(canvas.copy())
        for i in range(nPoints):
            for j in range(len(detected_keypoints[i])):
                realX,realY,realZ=rs.rs2_deproject_pixel_to_point(depth_intrin,detected_keypoints[i][j][0:2],depth_image[detected_keypoints[i][j][0]][detected_keypoints[i][j][1]])
                cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
                cv2.putText(frameClone,str([realX,realY,realZ]),(detected_keypoints[i][j][0:2]),(cv2.FONT_HERSHEY_COMPLEX),0.6,(0,255,0),1,cv2.LINE_AA)
        valid_pairs, invalid_pairs = getValidPairs(outputs, w, h)
        personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)

        for i in range(17):
            for n in range(len(personwiseKeypoints)):
                index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                if -1 in index:
                    continue
                B = np.int32(keypoints_list[index.astype(int), 0])
                A = np.int32(keypoints_list[index.astype(int), 1])
                cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

        cv2.putText(frameClone, fps, (w-170,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)

        cv2.namedWindow("USB Camera", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("USB Camera" , frameClone)

        if cv2.waitKey(1)&0xFF == ord('q'):
            break

        # FPS calculation
        framecount += 1
        if framecount >= 15:
            fps = "(Playback) {:.1f} FPS".format(time1/15)
            framecount = 0
            time1 = 0
        t2 = time.perf_counter()
        elapsedTime = t2-t1
        time1 += 1/elapsedTime

except:
    import traceback
    traceback.print_exc()

finally:

    print("\n\nFinished\n\n")

