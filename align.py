import cv2
import numpy as np

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

    # template_img_gray = cv2.fastNlMeansDenoising(template_img_gray, None, 5, 21)
    # register_img_gray = cv2.fastNlMeansDenoising(register_img_gray, None, 5, 21)

    template_img_gray = cv2.medianBlur(template_img_gray, 5)
    register_img_gray = cv2.medianBlur(register_img_gray, 5)

    #  apply Gaussian bluring to remove noise
    template_img_gray = cv2.GaussianBlur(template_img_gray, (3, 3), 0)
    register_img_gray = cv2.GaussianBlur(register_img_gray, (3, 3), 0)

    if is_thermal_reference:
        # apply image histogram equaliztion
        register_img_gray = cv2.equalizeHist(register_img_gray)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        register_img_gray = clahe.apply(register_img_gray)

        # ajust gamma for dark environment
        # register_img_gray = adjust_gamma(register_img_gray, 0.5)
        
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

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        template_img_gray = clahe.apply(template_img_gray)

        # ajust gamma for dark environment
        # template_img_gray = adjust_gamma(template_img_gray, 0.5)
        
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
            # print("Trying register the whole images....")
            print("###############################################")
            # cv2.waitKey(1)
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