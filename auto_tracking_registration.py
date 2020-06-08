# thermal is our template image for registration

import os
import cv2
import time
import sys
import numpy as np
import dlib
from skimage.measure import shannon_entropy, compare_ssim
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from imageai.Detection import ObjectDetection
from skimage.exposure import adjust_gamma
from imutils.video import VideoStream, FileVideoStream, FPS
from scipy import ndimage
# from mi_reg import main_mi_reg
# from poc_reg import main_poc_reg

def mutual_information_2d(x, y, sigma=1, normalized=False):
    EPS = np.finfo(float).eps
    bins = (256, 256)
    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
                                    output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                / np.sum(jh * np.log(jh))) - 1
    else:
        mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
                - np.sum(s2 * np.log(s2)))

    return mi

def gradient(im, ksize=5):
    im = normalize(im)
    grad_x = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=ksize)

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

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

def preprocess_images(template_img, register_img, is_thermal_reference):

    # convert to gray image
    template_img_gray = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
    register_img_gray = cv2.cvtColor(register_img, cv2.COLOR_RGB2GRAY)

    template_img_gray = cv2.fastNlMeansDenoising(template_img_gray, None, 5, 21)
    register_img_gray = cv2.fastNlMeansDenoising(register_img_gray, None, 5, 21)

    #  apply Gaussian bluring to remove noise
    template_img_gray = cv2.GaussianBlur(template_img_gray, (3, 3), 0)
    register_img_gray = cv2.GaussianBlur(register_img_gray, (3, 3), 0)

    if is_thermal_reference:
        # apply image histogram equaliztion
        register_img_gray = cv2.equalizeHist(register_img_gray)

        # ajust gamma for dark environment
        register_img_gray = adjust_gamma(register_img_gray, 0.5)
        
        # kernel = np.array([[-1, -1, -1],
        #             [-1, 8, -1],
        #             [-1, -1, -1]])
        # register_img_gray = cv2.filter2D(register_img_gray, -1, kernel)

        # invert visible image
        # register_img_gray = cv2.bitwise_not(register_img_gray)

        # convert to binary image
        # _, register_img_gray = cv2.threshold(register_img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
    else:
        # apply image histogram equaliztion
        template_img_gray = cv2.equalizeHist(template_img_gray)

        # ajust gamma for dark environment
        template_img_gray = adjust_gamma(template_img_gray, 0.5)
        
        # kernel = np.array([[-1, -1, -1],
        #             [-1, 9, -1],
        #             [-1, -1, -1]])
        # template_img_gray = cv2.filter2D(template_img_gray, -1, kernel)

        # invert visible image
        # template_img_gray = cv2.bitwise_not(template_img_gray)

        # convert to binary image
        # _, template_img_gray = cv2.threshold(template_img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cv2.imshow("Template Image", template_img_gray)
    cv2.imshow("Register Image", register_img_gray)
    cv2.waitKey(1)

    # template_img_gray = cv2.fastNlMeansDenoising(template_img_gray, None, 5, 21)
    # register_img_gray = cv2.fastNlMeansDenoising(register_img_gray, None, 5, 21)

    kernel = np.ones((5, 5), np.uint8)
    template_img_gray = cv2.morphologyEx(template_img_gray, cv2.MORPH_OPEN, kernel)
    register_img_gray = cv2.morphologyEx(register_img_gray, cv2.MORPH_OPEN, kernel)

    # template_img_gray = auto_canny(template_img_gray)
    # register_img_gray = auto_canny(register_img_gray)

    # Convert to gradient image
    template_img_gray = gradient(template_img_gray)
    register_img_gray = gradient(register_img_gray)

    cv2.imshow("Template Image Gradient", template_img_gray)
    cv2.imshow("Registration Image Gradient", register_img_gray)
    cv2.waitKey(1)

    return template_img_gray, register_img_gray

def align(template_img, register_img, warp_mode=cv2.MOTION_AFFINE, max_iterations=300, epsilon_threshold=1e-10, pyramid_levels=2,
          is_video=False):

    if pyramid_levels is None:
        w = template_img.shape[1]
        nol = int(w / (1280 / 3)) - 1
    else:
        nol = pyramid_levels

    # Initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    template_img_pyr = [template_img]
    register_img_pyr = [register_img]

    for level in range(nol):
        template_img_pyr[0] = normalize(template_img_pyr[0])
        template_img_pyr.insert(0, cv2.resize(template_img_pyr[0], None, fx=1/2, fy=1/2, interpolation=cv2.INTER_LINEAR))
        register_img_pyr[0] = normalize(register_img_pyr[0])
        register_img_pyr.insert(0, cv2.resize(register_img_pyr[0], None, fx=1/2, fy=1/2, interpolation=cv2.INTER_LINEAR))

    # Terminate the optimizer if either the max iterations or the threshold are reached
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, epsilon_threshold)
    # run pyramid ECC
    for level in range(nol):
        template_img_grad = template_img_pyr[level]
        register_img_grad = register_img_pyr[level]
        try:
            cc, warp_matrix = cv2.findTransformECC(template_img_grad, register_img_grad, warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)
        except Exception as e:
            print("################## Error ECC ##################")
            print(e)
            print("Press Enter to try register the whole images....")
            print("###############################################")
            cv2.waitKey()
            return None
            # sys.exit(1)
            # cc, warp_matrix = cv2.findTransformECC(template_img_grad, register_img_grad, warp_matrix, warp_mode, criteria)

        if level != nol:  # scale up only the offset by a factor of 2 for the next (larger image) pyramid level
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                warp_matrix = warp_matrix * np.array([[1, 1, 2], [1, 1, 2], [0.5, 0.5, 1]], dtype=np.float32)
            else:
                warp_matrix = warp_matrix * np.array([[1, 1, 2], [1, 1, 2]], dtype=np.float32)
    
    # return warp_matrix
    return warp_matrix

def ecc_registration(thermal, visible, is_thermal_reference, warp_mode = cv2.MOTION_AFFINE):
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

    template_img_gray, register_img_gray = preprocess_images(template_img, register_img, is_thermal_reference)    

    # Find warp matrix
    warp_matrix = align(template_img_gray, register_img_gray, warp_mode, 50, 1e-3, 2)
    
    if warp_matrix is not None:
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            aligned_img = cv2.warpPerspective(register_img, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            aligned_img = cv2.warpAffine(register_img, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        # print(warp_matrix)
        dst = cv2.addWeighted(thermal, 0.5, aligned_img, 0.5, 0.0)
        cv2.imshow("Blending Image", dst)
        # cv2.moveWindow("Blending Image", 200, 200)
        cv2.waitKey(1)
        # cv2.destroyAllWindows()
    
    return warp_matrix

def get_box_from_object_detection(detector, custom, fused):
    (h, w) = fused.shape[:2]
    after_registration_object, after_registration_detections = detector.detectCustomObjectsFromImage(custom_objects = custom,
            input_type = "array",
            input_image = fused,
            output_type = "array",
            minimum_percentage_probability=50)
    
    cv2.imshow("Bounding Box", cv2.resize(after_registration_object, (800, 500)))
    cv2.waitKey(1)

    if len(after_registration_detections) == 0 :
        print("No human object detection!")
        return None

    # get biggest bounding boxes (closest object)
    if len(after_registration_detections) > 1:
        after_registration_detections = sorted(after_registration_detections, key=sort_by_bb_area, reverse=True)
    
    final_bounding_box = after_registration_detections[0]["box_points"]
    print((final_bounding_box[2] - final_bounding_box[0])*(final_bounding_box[3] - final_bounding_box[1]))

    if (final_bounding_box[2] - final_bounding_box[0])*(final_bounding_box[3] - final_bounding_box[1]) < 2500:
        return None

    # set bounding boxes location
    x1 = final_bounding_box[0] - 30
    y1 = final_bounding_box[1] - 30
    x2 = final_bounding_box[2] + 30
    y2 = final_bounding_box[3] + 30
    
    start_point = (x1 if x1 > 0 else 0, y1 if y1 > 0 else 0)
    end_point = (x2 if x2 < w-1 else w-1, y2 if y2 < h-1 else h-1)

    return (start_point[0], start_point[1], end_point[0] - start_point[0], end_point[1] - start_point[1])

def sort_by_bb_area(object_detection):
    bounding_box = object_detection["box_points"]
    return (bounding_box[2] - bounding_box[0]) * (bounding_box[3] - bounding_box[1])

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

        # template_img_gray, register_img_gray = preprocess_images(object_thermal, temp_object_visible, is_thermal_reference)    
        # dx, dy, match_height = main_mi_reg(template_img_gray, register_img_gray)
        # # print(main_poc_reg(template_img_gray, register_img_gray))
        # print(dx, dy, match_height)
        # translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        # aaa = cv2.warpAffine(temp_object_visible, translation_matrix, (temp_object_visible.shape[1], temp_object_visible.shape[0]))
        # cv2.imshow("aaa", cv2.addWeighted(fixBorder(object_thermal), 0.5, fixBorder(aaa), 0.5, 0.0))

        # get finetune matrix for object
        finetune_matrix = ecc_registration(object_thermal, temp_object_visible, is_thermal_reference, warp_mode=cv2.MOTION_TRANSLATION)
        if finetune_matrix is not None:
            if previous_finetune_matrix is not None:
                finetune_matrix[0, 2] =  finetune_matrix[0, 2] + previous_finetune_matrix[0, 2]
                finetune_matrix[1, 2] =  finetune_matrix[1, 2] + previous_finetune_matrix[1, 2]
            
            # check mutual score and return the result
            finetuned_temp_object_visible = cv2.warpAffine(object_visible, finetune_matrix, (object_thermal.shape[1], object_thermal.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            finetuned_temp_object_visible_gray = cv2.cvtColor(finetuned_temp_object_visible, cv2.COLOR_RGB2GRAY)
            
            print("*******************************")
            print(mutual_information_2d(fixBorder(object_thermal_gray).ravel(), fixBorder(finetuned_temp_object_visible_gray).ravel(), normalized=True))
            print(mutual_information_2d(fixBorder(object_thermal_gray).ravel(), fixBorder(temp_object_visible_gray).ravel(), normalized=True))
            print(mutual_information_2d(fixBorder(object_thermal_gray).ravel(), fixBorder(object_visible_gray).ravel(), normalized=True))

            results = [(object_thermal, finetuned_temp_object_visible, finetune_matrix, box, "1"), (object_thermal, temp_object_visible, previous_finetune_matrix, box, "2"), (object_thermal, object_visible, previous_finetune_matrix, box, "3")]
            results = sorted(results, key = sort_by_MI, reverse=True)

            for r in results:
                print(r[4])

            final_finetune_matrix = results[0][2]
            box = results[0][3]
            case = results[0][4]

            cv2.imshow("1", cv2.addWeighted(fixBorder(object_thermal), 0.5, fixBorder(finetuned_temp_object_visible), 0.5, 0.0))
            cv2.imshow("2", cv2.addWeighted(fixBorder(object_thermal), 0.5, fixBorder(temp_object_visible), 0.5, 0.0))
            cv2.imshow("3", cv2.addWeighted(fixBorder(object_thermal), 0.5, fixBorder(object_visible), 0.5, 0.0))
            cv2.imshow("Result", cv2.addWeighted(object_thermal, 0.5, results[0][1], 0.5, 0.0))
            cv2.waitKey(1)

            if case == "3" or final_finetune_matrix is None:
                finetuned_visible = visible
                # final_finetune_matrix = None
            else:
                finetuned_visible = cv2.warpAffine(visible, final_finetune_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            return finetuned_visible, final_finetune_matrix, box    

    # there is no target area, apply ecc only
    if box is None or finetune_matrix is None:
        return visible, None, None

def sort_by_coef_score(result):
    return np.abs(calculate_correlation_coef(fixBorder(result[0]), fixBorder(result[1])))

def sort_by_SSIM(result):
    object1 = cv2.cvtColor(fixBorder(result[0]), cv2.COLOR_RGB2GRAY)
    object2 = cv2.cvtColor(fixBorder(result[1]), cv2.COLOR_RGB2GRAY)
    return compare_ssim(object1, object2)

def sort_by_MI(result):
    object1 = cv2.cvtColor(fixBorder(result[0]), cv2.COLOR_RGB2GRAY)
    object2 = cv2.cvtColor(fixBorder(result[1]), cv2.COLOR_RGB2GRAY)
    return mutual_information_2d(object1.ravel(), object2.ravel(), normalized=True)

def get_initial_transformaton_matrix(thermal_cap, visible_cap, frame_count, is_thermal_reference):
    initial_matrix = None
    frame_num = int(frame_count/4)

    frame_indexes = np.random.choice(range(0, frame_count), size = 20, replace = False)
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

        # (h, w) = visible_frame.shape[:2]
        (w, h) = (1024, 768)
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

    # set camera capture to init position
    thermal_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    visible_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return initial_matrix

def run(detector, custom, thermal_video_path, visible_video_path, output_video, is_thermal_reference):
    thermal_cap = cv2.VideoCapture(thermal_video_path)
    visible_cap = cv2.VideoCapture(visible_video_path)

    # get thermal and visible video primary info
    video_fps = thermal_cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(thermal_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Video FPS:", video_fps)
    print("Number of frames:", frame_count)

    # resolution for both thermal and visible
    fps = FPS().start() 
    (w, h) = (1024, 768)
    # (w, h) = int(visible_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(visible_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # init video writer
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (2*w, h), True)

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
                box = get_box_from_object_detection(detector, custom, after_registration)
        else:
            box = get_box_from_object_detection(detector, custom, after_registration)

        # finetune registration
        finetuned_visible_frame, finetune_matrix, box = finetune_registration(thermal_frame, registered_visible_frame, is_thermal_reference, initial_transformation_matrix, finetune_matrix, box)
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

        frame_out = cv2.hconcat([after_finetune, after_registration])
        # frame_out = after_finetune

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
    
    # set if thermal is template image
    is_thermal_reference = True

    # init YOLOv3 object detector for person
    execution_path = os.getcwd()
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(execution_path , "yolo.h5"))
    detector.loadModel()
    custom = detector.CustomObjects(person=True)

    # paths to videos
    # thermal_video_path = "./440/thermal_440.avi"
    # visible_video_path = "./440/visible_440.avi"
    # output_video = "./auto_registration_440.avi"

    # thermal_video_path = "./560/thermal_560.avi"
    # visible_video_path = "./560/visible_560.avi"
    # output_video = "./auto_registration_560.avi"

    # thermal_video_path = "./sneaking_thermal.mp4"
    # visible_video_path = "./sneaking_visible.avi"
    # output_video = "./auto_registration_sneaking.avi"
    
    # thermal_video_path = "./fighting_thermal.mp4"
    # visible_video_path = "./fighting_visible.avi"
    # output_video = "./auto_registration_fighting.avi"

    # thermal_video_path = "./multi_targets_thermal.avi"
    # visible_video_path = "./multi_targets_visible.avi"
    # output_video = "./multi_targets_fused.avi"

    thermal_video_path = "./data/2020-06-03T21.57.11/AX5-0000.avi"
    visible_video_path = "./data/2020-06-03T21.57.11/BLA-0000.mp4"
    output_video = "./ped1.avi"

    # run auto registration
    run(detector, custom, thermal_video_path, visible_video_path, output_video, is_thermal_reference)