# thermal should be our template image for registration

import os
import cv2
import time
import sys
import numpy as np

from skimage.exposure import adjust_gamma
from imutils.video import FPS
from align import ecc_registration
from utils import mutual_information_2d, fixBorder, non_max_suppression
from yolo import YOLO
from PIL import Image
from detection import Detection

# import dlib
# from skimage.measure import shannon_entropy, compare_ssim
# from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
# from mi_reg import main_mi_reg
# from poc_reg import main_poc_reg

def get_box_from_object_detection(yolo, image):
    (h, w) = image.shape[:2]
    center_point = (int(w/2), int(h/2))
    final_detections = []

    image = Image.fromarray(image[...,::-1])  # bgr to rgb
    boxes, confidence, classes = yolo.detect_image(image)

    detections = [Detection(bbox, confidence, cls) for bbox, confidence, cls in
                      zip(boxes, confidence, classes)]

    # Run non-maxima suppression.
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.cls for d in detections])
    indices = non_max_suppression(boxes, 1.0, scores)
    detections = [detections[i] for i in indices if detections[i].cls == "person"]
    
    # cv2.imshow("Bounding Box", cv2.resize(after_registration_object, (800, 500)))
    # cv2.waitKey(1)

    if len(detections) == 0 :
        print("No human object detection!")
        return None

    for detection in detections:
        start_x, _, end_x, _ = detection.to_tlbr() 

        # Registration only on the center area of image
        if end_x < center_point[0] or start_x > center_point[0]:
            continue
        final_detections.append(detection)

    # get biggest bounding boxes (closest object)
    if len(final_detections) >= 1:
        final_detections = sorted(final_detections, key=sort_by_bb_area, reverse=True)
    else:
        return None

    return final_detections[0].tlwh

def sort_by_bb_area(object_detection):
    _, _, w, h = object_detection.tlwh
    return w*h

def finetune_registration(thermal, visible, is_thermal_reference, initial_transformation_matrix, previous_finetune_matrix, box):
    (h, w) = visible.shape[:2]
    finetune_matrix = None

    # there is target area for finetune registration
    if box is not None:
        if previous_finetune_matrix is not None:
            temp_visible = cv2.warpAffine(visible, previous_finetune_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            temp_visible = visible

        (x, y, del_x, del_y) = [int(v) for v in box]
        x1 = x if x > 0 else 0
        y1 = y if y > 0 else 0
        x2 = x1 + del_x
        y2 = y1 + del_y
        x2 = x2 if x2 < w-1 else w-1
        y2 = y2 if y2 < h-1 else h-1
        start_point = (x1, y1)
        end_point = (x2, y2)

        object_thermal = thermal[start_point[1]:end_point[1], start_point[0]:end_point[0]]
        temp_object_visible = temp_visible[start_point[1]:end_point[1], start_point[0]:end_point[0]]
        object_visible = visible[start_point[1]:end_point[1], start_point[0]:end_point[0]]

        object_thermal_gray = cv2.cvtColor(object_thermal, cv2.COLOR_RGB2GRAY)
        temp_object_visible_gray = cv2.cvtColor(temp_object_visible, cv2.COLOR_RGB2GRAY)
        object_visible_gray = cv2.cvtColor(object_visible, cv2.COLOR_RGB2GRAY)

        # cv2.imshow("Object Thermal", object_thermal)
        # cv2.imshow("Object Visible", object_visible)
            
        # get finetune matrix for object
        finetune_matrix = ecc_registration(object_thermal, temp_object_visible, is_thermal_reference, warp_mode=cv2.MOTION_TRANSLATION)
        if finetune_matrix is not None:
            if previous_finetune_matrix is not None:
                finetune_matrix[0, 2] =  finetune_matrix[0, 2] + previous_finetune_matrix[0, 2]
                finetune_matrix[1, 2] =  finetune_matrix[1, 2] + previous_finetune_matrix[1, 2]
            
            finetuned_temp_object_visible = cv2.warpAffine(object_visible, finetune_matrix, (object_thermal.shape[1], object_thermal.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            finetuned_temp_object_visible_gray = cv2.cvtColor(finetuned_temp_object_visible, cv2.COLOR_RGB2GRAY)
            
             # check mutual score and return the result
            print("*******************************")
            print(mutual_information_2d(fixBorder(object_thermal_gray).ravel(), fixBorder(finetuned_temp_object_visible_gray).ravel(), normalized=True))
            print(mutual_information_2d(fixBorder(object_thermal_gray).ravel(), fixBorder(temp_object_visible_gray).ravel(), normalized=True))
            print(mutual_information_2d(fixBorder(object_thermal_gray).ravel(), fixBorder(object_visible_gray).ravel(), normalized=True))

            results = [(object_thermal, finetuned_temp_object_visible, finetune_matrix, "1"), (object_thermal, temp_object_visible, previous_finetune_matrix, "2"), 
                        (object_thermal, object_visible, previous_finetune_matrix, "3")]

            results = sorted(results, key = sort_by_MI, reverse=True)
            for r in results:
                print(r[3])

            # get the best score of result
            final_finetune_matrix = results[0][2]
            case = results[0][3]

            cv2.imshow("1", cv2.addWeighted(fixBorder(object_thermal), 0.5, fixBorder(finetuned_temp_object_visible), 0.5, 0.0))
            cv2.imshow("2", cv2.addWeighted(fixBorder(object_thermal), 0.5, fixBorder(temp_object_visible), 0.5, 0.0))
            cv2.imshow("3", cv2.addWeighted(fixBorder(object_thermal), 0.5, fixBorder(object_visible), 0.5, 0.0))
            cv2.imshow("Result", cv2.addWeighted(fixBorder(object_thermal), 0.5, fixBorder(results[0][1]), 0.5, 0.0))
            cv2.waitKey(1)

            if case == "3" or final_finetune_matrix is None:
                finetuned_visible = visible
            else:
                finetuned_visible = cv2.warpAffine(visible, final_finetune_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            return finetuned_visible, final_finetune_matrix  

    # there is no target area, apply ecc only
    if box is None or finetune_matrix is None:
        return visible, None

def sort_by_MI(result):
    object1 = cv2.cvtColor(fixBorder(result[0]), cv2.COLOR_RGB2GRAY)
    object2 = cv2.cvtColor(fixBorder(result[1]), cv2.COLOR_RGB2GRAY)
    return mutual_information_2d(object1.ravel(), object2.ravel(), normalized=True)

def get_initial_transformaton_matrix(thermal_cap, visible_cap, frame_count, is_thermal_reference):
    initial_matrix = None
    # frame_num = int(frame_count/4)

    frame_indexes = np.random.choice(range(0, frame_count), size = 30, replace = False)
    warp_matrix_list = []

    print()
    print("*************************************")
    print("Get initial transformation matrix....")
    print(frame_indexes)
    print("Processing " + str(len(frame_indexes)) + " selected frames")

    for i, frame_index in enumerate(frame_indexes):
        print("Frame " + str(i+1) + "/" + str(len(frame_indexes)))
        thermal_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        visible_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        _, thermal_frame = thermal_cap.read()
        _, visible_frame = visible_cap.read()

        (h, w) = visible_frame.shape[:2]
        thermal_frame = cv2.resize(thermal_frame, (w, h))
        warp_matrix = ecc_registration(thermal_frame, visible_frame, is_thermal_reference)
        if warp_matrix is None:
            print("Got error with ECC registration! Cannot get initial transformation matrix!")
        else:
            warp_matrix_list.append(warp_matrix)

    initial_matrix = np.mean(np.array(warp_matrix_list), axis=0)
    print("Initial transformatrion matrix:")
    print(initial_matrix)
    print("*************************************")
    print()

    return initial_matrix

def run(yolo, thermal_video_path, visible_video_path, output_video, combined_output_video, is_thermal_reference):
    thermal_cap = cv2.VideoCapture(thermal_video_path)
    visible_cap = cv2.VideoCapture(visible_video_path)

    # get thermal and visible video primary info
    video_fps = thermal_cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(thermal_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Video FPS:", video_fps)
    print("Number of frames:", frame_count)

    # resolution for both thermal and visible
    fps = FPS().start() 
    (w, h) = int(visible_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(visible_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # init video writer
    combined_out = cv2.VideoWriter(combined_output_video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (2*w, h), True)
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (w, h), True)

    initial_transformation_matrix = get_initial_transformaton_matrix(thermal_cap, visible_cap, frame_count, is_thermal_reference)
    # initial_transformation_matrix = np.array([[ 9.8124236e-01 ,-4.4611096e-03 , 1.6780478e+01], [ 8.2092045e-04 , 1.0321504e+00 , -1.3029246e+01]])
    # initial_transformation_matrix = np.array([[ 9.8314154e-01 , 6.4334326e-04 , 1.6971518e+01], [ 4.5000976e-03 , 1.0378873e+00 , -1.6411327e+01]])
    # initial_transformation_matrix = np.array([[ 9.7655469e-01 , 6.9244271e-03 , 1.8063793e+01], [-1.3102370e-02 , 1.0067332e+00 , 1.0388448e+00]])
    # initial_transformation_matrix = np.array([[ 9.94109511e-01 , 1.15506705e-02 , 7.52761555e+00], [ 7.38697872e-03 , 1.03959394e+00 ,-2.04733562e+01]])
    # initial_transformation_matrix = np.array([[ 9.7559845e-01, -4.8294739e-04,  2.0626131e+01], [ 3.5860003e-03,  1.0423101e+00, -1.8106863e+01]])
    # initial_transformation_matrix = np.array([[ 9.7559845e-01, -4.8294739e-04,  2.0626131e+01], [ 3.5860003e-03,  1.0423101e+00, -1.8106863e+01]])
    # initial_transformation_matrix = np.array([[ 9.9236757e-01 , 6.4181895e-03 , 1.2159206e+01], [ 4.3100063e-03 , 1.0310320e+00, -1.3821500e+01]])

    print("Processing thermal and visible videos...")
    print("+++++++++++++++++++++++++++++++++++++")

    # init params
    index = 1
    finetune_matrix = None
    initBB = None
    tracking_success = False
    box = None
    tracker = None

    # set camera capture to init position
    destinate_index = 0
    thermal_cap.set(cv2.CAP_PROP_POS_FRAMES, destinate_index)
    visible_cap.set(cv2.CAP_PROP_POS_FRAMES, destinate_index)

    while True:
        _, thermal_frame = thermal_cap.read()
        _, visible_frame = visible_cap.read()
        
        if thermal_frame is None or visible_frame is None:
            break
        print("Frame " + str(index) + "/" + str(frame_count))

        # resize thermal to  visible resolution
        thermal_frame = cv2.resize(thermal_frame, (w, h))
        visible_frame = cv2.resize(visible_frame, (w, h))
        
        # visualize before registration
        print("Mutual Information Before Registration:", mutual_information_2d(fixBorder(thermal_frame).ravel(), fixBorder(visible_frame).ravel()))
        before_registration = cv2.addWeighted(thermal_frame, 0.5, visible_frame, 0.5, 0.0)
        cv2.imshow("Before Registration", cv2.resize(before_registration, (800, 500)))
        cv2.waitKey(1)

        # get registered visible frame
        registered_visible_frame = cv2.warpAffine(visible_frame, initial_transformation_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        # visualize after applying initial transformation matrix
        after_registration = cv2.addWeighted(thermal_frame, 0.5, registered_visible_frame, 0.5, 0.0)
        print("Mutual Information After Initial Registration:", mutual_information_2d(fixBorder(thermal_frame).ravel(), fixBorder(registered_visible_frame).ravel()))
        cv2.imshow("After Initial Registration", cv2.resize(after_registration, (800, 500)))
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("s"):
            initBB = cv2.selectROI("Init BB", after_registration, fromCenter=False, showCrosshair=True)
            tracker = cv2.TrackerCSRT_create()
            # tracker = cv2.TrackerKCF_create()
            # tracker = dlib.correlation_tracker()

            tracker.init(thermal_frame, initBB)
            # rect = dlib.rectangle(initBB[0], initBB[1], initBB[0] + initBB[2], initBB[1] + initBB[3])
            # tracker.start_track(thermal_frame, rect)
            
            box = initBB
            tracking_success = True
            finetune_matrix = None
            cv2.destroyWindow("Init BB")
        
        if key == ord("x"):
            tracker.clear()
            initBB = None
            tracking_success = False

        if initBB is not None:
            (tracking_success, box) = tracker.update(thermal_frame)
            # tracker.update(after_registration)
            # pos = tracker.get_position()
            
            # startX = int(pos.left())
            # startY = int(pos.top())
            # endX = int(pos.right())
            # endY = int(pos.bottom())

            # box = (startX, startY, endX-startX, endY-startY)

            if not tracking_success:
                box = get_box_from_object_detection(yolo, after_registration)
        else:
            box = get_box_from_object_detection(yolo, after_registration)

        # finetune registration
        finetuned_visible_frame, finetune_matrix = finetune_registration(thermal_frame, registered_visible_frame, is_thermal_reference, initial_transformation_matrix, finetune_matrix, box)
        after_finetune = cv2.addWeighted(thermal_frame, 0.5, finetuned_visible_frame, 0.5, 0.0)
        
        # draw bounding box on target object
        if tracking_success and box is not None:
            # green color for tracking target
            (x, y, del_x, del_y) = [int(v) for v in box]
            cv2.rectangle(after_finetune, (x, y), (x + del_x, y + del_y), (0, 255, 0), 2)
        elif box is not None:
            # red color for YOLO target
            (x, y, del_x, del_y) = [int(v) for v in box]
            cv2.rectangle(after_finetune, (x, y), (x + del_x, y + del_y), (0, 0, 255), 2)
        elif box is None:
            initBB = None
            tracking_success = False
        
        print("Mutual Information After Finetune Registration:", mutual_information_2d(fixBorder(thermal_frame).ravel(), fixBorder(finetuned_visible_frame).ravel()))
        cv2.imshow("After Finetune Registration", cv2.resize(after_finetune, (800, 500)))
        cv2.waitKey(1)

        combined_frame_out = cv2.hconcat([after_finetune, after_registration])
        combined_out.write(combined_frame_out)
        out.write(after_finetune)

        index += 1
        fps.update()
        print("+++++++++++++++++++++++++++++++++++++")

    fps.stop()
    print("Elasped time: {:.2f}".format(fps.elapsed()))
    print("Approx. FPS: {:.2f}".format(fps.fps()))
    thermal_cap.release()
    visible_cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # set if thermal is template image
    is_thermal_reference = True

    # init YOLOv4 object detector for person
    yolo = YOLO()

    # paths to videos
    # thermal_video_path = "../440/thermal_440.avi"
    # visible_video_path = "../440/visible_440.avi"
    # combined_output_video = "../result_videos/combined_auto_registration_440.avi"
    # output_video = "../result_videos/auto_registration_440.avi"

    # thermal_video_path = "../560/thermal_560.avi"
    # visible_video_path = "../560/visible_560.avi"
    # combined_output_video = "../result_videos/combined_auto_registration_560.avi"
    # output_video = "../result_videos/auto_registration_560.avi"

    # thermal_video_path = "../data/Test_Videossneaking_thermal.mp4"
    # visible_video_path = "../data/Test_Videossneaking_visible.avi"
    # combined_output_video = "../result_videos/combined_auto_registration_sneaking.avi"
    # output_video = "../result_videos/auto_registration_sneaking.avi"
    
    # thermal_video_path = "../data/Test_Videosfighting_thermal.mp4"
    # visible_video_path = "../data/Test_Videosfighting_visible.avi"
    # combined_output_video = "../result_videos/combined_auto_registration_fighting.avi"
    # output_video = "../result_videos/auto_registration_fighting.avi"

    # thermal_video_path = "../data/Test_Videos/multi_targets_thermal.avi"
    # visible_video_path = "../data/Test_Videos/multi_targets_visible.avi"
    # combined_output_video = "../result_videos/combined_multi_targets_fused.avi"
    # output_video = "../result_videos/multi_targets_fused.avi"

    thermal_video_path = "../data/Normal/Outdoor/2020-06-03T21.57.11/AX5.avi"
    visible_video_path = "../data/Normal/Outdoor/2020-06-03T21.57.11/BLA.avi"
    combined_output_video = "../result_videos/combined_ped1.avi"
    output_video = "../result_videos/ped1.avi"

    # run auto registration
    run(yolo, thermal_video_path, visible_video_path, output_video, combined_output_video, is_thermal_reference)