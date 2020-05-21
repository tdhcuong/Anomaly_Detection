# thermal is our template image for registration

import os
import cv2
import time
import numpy as np
from skimage.measure import shannon_entropy, compare_ssim
from imageai.Detection import ObjectDetection
from skimage.exposure import adjust_gamma
from imutils.video import VideoStream, FileVideoStream, FPS

def ecc_registration(thermal, visible, warp_mode = cv2.MOTION_AFFINE):
    is_thermal_reference = True
    template_img = None
    register_img = None

    if is_thermal_reference:
        template_img = thermal
        register_img = visible
    else:
        template_img = visible
        register_img = thermal

    (h,w) = template_img.shape[:2]
    register_img = cv2.resize(register_img, (w,h))

    # template_img_gray = template_img
    # register_img_gray = register_img

    template_img_gray = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
    register_img_gray = cv2.cvtColor(register_img, cv2.COLOR_RGB2GRAY)

    # cv2.imshow("Template Image", template_img_gray)
    # cv2.imshow("Register Image", register_img_gray)
    # cv2.waitKey()

    # Preprocessing
    if is_thermal_reference:
        register_img_gray = adjust_gamma(register_img_gray, 0.5)
        # kernel = np.array([[-1, -1, -1],
        #             [-1, 9, -1],
        #             [-1, -1, -1]])
        # register_img_gray = cv2.filter2D(register_img_gray, -1, kernel)
    else:
        template_img = adjust_gamma(template_img, 0.5)
        # kernel = np.array([[-1, -1, -1],
        #             [-1, 9, -1],
        #             [-1, -1, -1]])
        # template_img = cv2.filter2D(template_img, -1, kernel)

    # Find warp matrix
    warp_matrix = align(template_img_gray, register_img_gray, warp_mode, 50, 1e-3, 2)

    # if warp_mode == cv2.MOTION_HOMOGRAPHY:
    #     # Use warpPerspective for Homography
    #     aligned_img = cv2.warpPerspective(register_img, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    # else:
    #     # Use warpAffine for Translation, Euclidean and Affine
    #     aligned_img = cv2.warpAffine(register_img, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # print(warp_matrix)
    # dst = cv2.addWeighted(thermal, 0.5, aligned_img, 0.5, 0.0)
    # cv2.imshow("Blending Image", dst)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return warp_matrix

def align(ref_img, match_img, warp_mode=cv2.MOTION_AFFINE, max_iterations=300, epsilon_threshold=1e-10, pyramid_levels=2,
          is_video=False):
    if pyramid_levels is None:
        w = ref_img.shape[1]
        nol = int(w / (1280 / 3)) - 1
    else:
        nol = pyramid_levels

    # Initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # warp_matrix = np.eye(3, 3, dtype=np.float32)
        warp_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    else:
        # warp_matrix = np.eye(2, 3, dtype=np.float32)
        warp_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    ref_img = cv2.fastNlMeansDenoising(ref_img, None, 5, 21)
    match_img = cv2.fastNlMeansDenoising(match_img, None, 5, 21)

    kernel = np.ones((11, 11), np.uint8)
    ref_img = cv2.morphologyEx(ref_img, cv2.MORPH_OPEN, kernel)
    match_img = cv2.morphologyEx(match_img, cv2.MORPH_OPEN, kernel)

    ref_img = gradient(ref_img)
    match_img = gradient(match_img)

    ref_img_pyr = [ref_img]
    match_img_pyr = [match_img]

    for level in range(nol):
        ref_img_pyr[0] = normalize(ref_img_pyr[0])
        ref_img_pyr.insert(0, cv2.resize(ref_img_pyr[0], None, fx=1/2, fy=1/2, interpolation=cv2.INTER_LINEAR))
        match_img_pyr[0] = normalize(match_img_pyr[0])
        match_img_pyr.insert(0, cv2.resize(match_img_pyr[0], None, fx=1/2, fy=1/2, interpolation=cv2.INTER_LINEAR))

    # Terminate the optimizer if either the max iterations or the threshold are reached
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, epsilon_threshold)
    # run pyramid ECC
    for level in range(nol):
        ref_img_grad = ref_img_pyr[level]
        match_img_grad = match_img_pyr[level]
        try:
            cc, warp_matrix = cv2.findTransformECC(ref_img_grad, match_img_grad, warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)
        except TypeError:
            cc, warp_matrix = cv2.findTransformECC(ref_img_grad, match_img_grad, warp_matrix, warp_mode, criteria)

        if level != nol:  # scale up only the offset by a factor of 2 for the next (larger image) pyramid level
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                warp_matrix = warp_matrix * np.array([[1, 1, 2], [1, 1, 2], [0.5, 0.5, 1]], dtype=np.float32)
            else:
                warp_matrix = warp_matrix * np.array([[1, 1, 2], [1, 1, 2]], dtype=np.float32)
    
    # return warp_matrix
    return warp_matrix

def gradient(im, ksize=15):
    im = normalize(im)
    grad_x = cv2.Sobel(im, cv2.CV_32FC1, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(im, cv2.CV_32FC1, 0, 1, ksize=ksize)
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return convert_to_img(grad)

def normalize(im, min=None, max=None):
    width, height = im.shape
    norm = np.zeros((width, height), dtype=np.float32)
    if min is not None and max is not None:
        norm = (im - min) / (max - min)
    else:
        cv2.normalize(im, dst=norm, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    norm[norm < 0.0] = 0.0
    norm[norm > 1.0] = 1.0
    return norm

def convert_to_img(img):
    img = np.multiply(np.divide(img - np.min(img), (np.max(img) - np.min(img))), 255)
    img = img.astype(np.uint8)
    return img

def corr2d(a,b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    r = np.sum(a*b) / float(np.sqrt(np.sum(a**2) * np.sum(b**2)))
    return r

def calculate_correlation_coef(thermal, visible):
    thermal_gray = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)
    visible_gray = cv2.cvtColor(visible, cv2.COLOR_BGR2GRAY)
    return corr2d(visible_gray, thermal_gray)

def fixBorder(frame):
  s = frame.shape
  # Scale the image 4% without moving the center
  T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
  frame = cv2.warpAffine(frame, T, (s[1], s[0]))
  return frame

def get_initial_transformaton_matrix(thermal_cap, visible_cap, frame_count):
    initial_matrix = None
    # frame_num = int(frame_count/4)
    frame_indexes = np.random.choice(range(0, frame_count), size = 10, replace = False)
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
        warp_matrix_list.append(ecc_registration(thermal_frame, visible_frame))
    
    initial_matrix = np.mean(np.array(warp_matrix_list), axis=0)

    print("Initial transformatrion matrix:")
    print(initial_matrix)
    print("*************************************")
    print()

    # set camera capture to init position
    thermal_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    visible_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return initial_matrix

def finetune_registration(detector, custom, thermal, visible, fused, corr_coef_after_registration):
    (h, w) = visible.shape[:2]
    
    after_registration_object, after_registration_detections = detector.detectCustomObjectsFromImage(custom_objects = custom,
            input_type = "array",
            input_image = fused,
            output_type = "array",
            minimum_percentage_probability=30)
    
    # cv2.imshow("Bounding Box", cv2.resize(after_registration_object, (800, 500)))
    # cv2.waitKey()

    if len(after_registration_detections) == 0 :
        print("No human object detection!")
        return visible

    # get biggest bounding boxes (closest object)
    final_bounding_box = after_registration_detections[0]["box_points"]
    max_area = (final_bounding_box[2] - final_bounding_box[0]) * (final_bounding_box[3] - final_bounding_box[1])
    if len(after_registration_detections) > 1:
        for bounding_box in after_registration_detections:
            local_width = bounding_box["box_points"][2] - bounding_box["box_points"][0]
            local_height = bounding_box["box_points"][3] - bounding_box["box_points"][1]
            local_area = local_width * local_height
            if(local_area > max_area):
                final_bounding_box = bounding_box["box_points"]
                max_area = local_area

    # set bounding boxes location
    x1 = final_bounding_box[0] - 50
    y1 = final_bounding_box[1] - 50
    x2 = final_bounding_box[2] + 50
    y2 = final_bounding_box[3] + 50

    start_point = (x1 if x1 > 0 else 0, y1 if y1 > 0 else 0)
    end_point = (x2 if x2 < w-1 else w-1, y2 if y2 < h-1 else h-1)

    object_thermal = thermal[start_point[1]:end_point[1], start_point[0]:end_point[0]]
    object_visible = visible[start_point[1]:end_point[1], start_point[0]:end_point[0]]

    # get fintune matrix for object
    finetune_matrix = ecc_registration(object_thermal, object_visible, warp_mode=cv2.MOTION_TRANSLATION)
    finetuned_visible = cv2.warpAffine(visible, finetune_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    if np.abs(calculate_correlation_coef(fixBorder(thermal), fixBorder(finetuned_visible))) > np.abs(corr_coef_after_registration):
        return finetuned_visible
    return visible

def run(detector, custom, thermal_video_path, visible_video_path, output_video):
    thermal_cap = cv2.VideoCapture(thermal_video_path)
    visible_cap = cv2.VideoCapture(visible_video_path)

    # get thermal and visible video primary info
    video_fps = thermal_cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(thermal_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Video FPS:", video_fps)
    print("Number of frames:", frame_count)

    # resolution for both thermal and visible
    fps = FPS().start() 
    # (w, h) = (2048, 1536)
    (w, h) = int(visible_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(visible_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # init video writer
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (2*w, h), True)
    
    initial_transformation_matrix = get_initial_transformaton_matrix(thermal_cap, visible_cap, frame_count)
    # initial_transformation_matrix = np.array([[9.9126685e-01, 5.3021866e-03, 1.8741880e+01], [9.3280076e-04, 1.0312355e+00, -2.4526514e+01]])

    print("Processing thermal and visible videos...")
    print("+++++++++++++++++++++++++++++++++++++")

    index = 1
    while True:
        _, thermal_frame = thermal_cap.read()
        _, visible_frame = visible_cap.read()

        # if index < 89:
        #     index += 1
        #     continue

        if thermal_frame is None or visible_frame is None:
            break
        
        print("Frame " + str(index) + "/" + str(frame_count))
        # resize thermal to  visible resolution
        thermal_frame = cv2.resize(thermal_frame, (w, h))

        # visualize before registration
        print("Correlation Coef Before Registration:", calculate_correlation_coef(fixBorder(thermal_frame), fixBorder(visible_frame)))
        before_registration = cv2.addWeighted(thermal_frame, 0.5, visible_frame, 0.5, 0.0)
        cv2.imshow("Before Registration", cv2.resize(before_registration, (800, 500)))
        cv2.waitKey(1)

        # get registered visible frame
        registered_visible_frame = cv2.warpAffine(visible_frame, initial_transformation_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        
        # visualize after applying initial transformation matrix
        after_registration = cv2.addWeighted(thermal_frame, 0.5, registered_visible_frame, 0.5, 0.0)
        corr_coef_after_registration = calculate_correlation_coef(fixBorder(thermal_frame), fixBorder(registered_visible_frame))
        print("Correlation Coef After Initial Registration:", corr_coef_after_registration)
        cv2.imshow("After Initial Registration", cv2.resize(after_registration, (800, 500)))
        cv2.waitKey(1)
        

        # finetune registration
        finetuned_visible_frame = finetune_registration(detector, custom, thermal_frame, registered_visible_frame, after_registration, corr_coef_after_registration)
        after_finetune = cv2.addWeighted(thermal_frame, 0.5, finetuned_visible_frame, 0.5, 0.0)
        print("Correlation Coef After Finetune Registration:", calculate_correlation_coef(fixBorder(thermal_frame), fixBorder(finetuned_visible_frame)))
        cv2.imshow("After Finetune Registration", cv2.resize(after_finetune, (800, 500)))
        cv2.waitKey(1)

        frame_out = cv2.hconcat([after_finetune, after_registration])
        out.write(frame_out)
        index += 1
        fps.update()
        # cv2.destroyAllWindows()
        print("+++++++++++++++++++++++++++++++++++++")

    fps.stop()
    print("Elasped time: {:.2f}".format(fps.elapsed()))
    print("Approx. FPS: {:.2f}".format(fps.fps()))
    thermal_cap.release()
    visible_cap.release()
    # out.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    # init YOLOv3 object detector for person
    execution_path = os.getcwd()
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(execution_path , "yolo.h5"))
    detector.loadModel()
    custom = detector.CustomObjects(person=True)
    
    # paths to videos
    thermal_video_path = "./440/thermal_440.avi"
    visible_video_path = "./440/visible_440.avi"
    output_video = "./auto_registration_440.avi"

    # run auto registration
    run(detector, custom, thermal_video_path, visible_video_path, output_video)