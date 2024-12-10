import numpy as np
import cv2
from relativeOrientation import relative_orientation

def histogram_matching(source, template):
    matched = np.zeros_like(source)
    for i in range(source.shape[0]):
        source_hist, _ = np.histogram(source[i, :], bins=256, range=(0, 256))
        template_hist, _ = np.histogram(template[i, :], bins=256, range=(0, 256))

        source_cdf = np.cumsum(source_hist)
        source_cdf = (source_cdf - source_cdf.min()) * 255 / (source_cdf.max() - source_cdf.min())
        source_cdf = source_cdf.astype(np.uint8)

        template_cdf = np.cumsum(template_hist)
        template_cdf = (template_cdf - template_cdf.min()) * 255 / (template_cdf.max() - template_cdf.min())
        template_cdf = template_cdf.astype(np.uint8)

        inverse_cdf = np.zeros(256, dtype=np.uint8)
        for j in range(256):
            inverse_cdf[j] = np.searchsorted(template_cdf, source_cdf[j])

        matched[i, :] = inverse_cdf[source[i, :]]

    return matched

def morcvec_feature_extraction(image, window_size, threshold):
    def calculate_interest_value(image):
        rows, cols = image.shape
        interest_values = np.zeros((rows, cols), dtype=np.float32)
        image = image.astype(np.float32)

        V1 = (image[1:-1, 1:-1] - image[:-2, 1:-1]) ** 2
        V2 = (image[1:-1, 1:-1] - image[2:, 1:-1]) ** 2
        V3 = (image[1:-1, 1:-1] - image[1:-1, :-2]) ** 2
        V4 = (image[1:-1, 1:-1] - image[1:-1, 2:]) ** 2
        interest_values[1:-1, 1:-1] = np.minimum(np.minimum(V1, V2), np.minimum(V3, V4))

        return interest_values

    def select_candidate_points(interest_values, window_size, threshold):
        rows, cols = interest_values.shape
        candidate_points = []

        for i in range(window_size//2, rows-window_size//2):
            for j in range(window_size//2, cols-window_size//2):
                if interest_values[i, j] > threshold:
                    candidate_points.append((i, j, interest_values[i, j]))

        return candidate_points

    def non_maximum_suppression(candidate_points, window_size):
        feature_points = []
        candidate_points.sort(key=lambda x: x[2], reverse=True)

        for i, (x, y, value) in enumerate(candidate_points):
            is_max = True
            for j in range(max(0, x-window_size//2), min(x+window_size//2+1, len(image))):
                for k in range(max(0, y-window_size//2), min(y+window_size//2+1, len(image[0]))):
                    if (j, k) != (x, y) and (j, k, interest_values[j, k]) in candidate_points:
                        if interest_values[j, k] > value:
                            is_max = False
                            break
                if not is_max:
                    break
            if is_max:
                feature_points.append((x, y))

        return feature_points

    # Ensure the image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    interest_values = calculate_interest_value(image)
    candidate_points = select_candidate_points(interest_values, window_size, threshold)
    feature_points = non_maximum_suppression(candidate_points, window_size)

    return feature_points

# Function to compute the corresponding point in the right image
def compute_corresponding_point(left_point, relative_orientation, df, bx):
    xL, yL = left_point
    dphi, domega, dkappa, dnumbda, dX = relative_orientation  # Adjusted to match the number of elements

    xR = xL - bx + df * (dphi * xL + domega * yL + dkappa)
    yR = yL + df * (dphi * yL - domega * xL)

    return (xR, yR)

# Mouse callback function to capture the point on the left or right image
def mark_point(event, x, y, flags, param):
    global left, right, relative_orientation, df, bx

    if event == cv2.EVENT_LBUTTONDOWN:
        left_point = (x, y)
        right_point = compute_corresponding_point(left_point, relative_orientation, df, bx)

        # Mark the points on the images
        cv2.circle(left, left_point, 1, (0, 0, 255), -1)
        cv2.circle(right, (int(right_point[0]), int(right_point[1])), 1, (0, 0, 255), -1)
        cv2.circle(right, (int(right_point[0]), int(right_point[1])), 30, (0, 255, 255), 1)
        # Display the images with the marked points
        combined = np.hstack((left, right))
        cv2.imshow("Left and Right Images", combined)

def compute_corresponding_point(left_point, relative_orientation, df, bx):
    xL, yL = left_point
    dphi, domega, dkappa, dnumbda, dX = relative_orientation

    xR = xL - bx + df * (dphi * xL + domega * yL + dkappa)
    yR = yL + df * (dphi * yL - domega * xL)

    return (xR, yR)

def extract_patch(image, center, size):
    x, y = center
    half_size = size // 2
    return image[y-half_size:y+half_size+1, x-half_size:x+half_size+1]

def match_points(left_image, right_image, left_features, right_features, relative_orientation, df, bx, radius=40, threshold=0.9):
    matched_points = []
    unmatched_left_points = []
    used_right_points = set()

    for left_point in left_features:
        right_point_est = compute_corresponding_point(left_point, relative_orientation, df, bx)
        xR_est, yR_est = map(int, right_point_est)

        best_match = None
        best_corr = -1

        for right_point in right_features:
            if right_point in used_right_points:
                continue

            x, y = right_point
            if np.sqrt((x - xR_est)**2 + (y - yR_est)**2) <= radius:
                left_patch = extract_patch(left_image, left_point, 7)
                right_patch = extract_patch(right_image, (x, y), 7)

                if left_patch.shape == right_patch.shape:
                    corr = np.corrcoef(left_patch.flatten(), right_patch.flatten())[0, 1]
                    if corr > best_corr:
                        best_corr = corr
                        best_match = (x, y)

        if best_corr >= threshold:
            matched_points.append((left_point, best_match))
            used_right_points.add(best_match)
        else:
            unmatched_left_points.append(left_point)

    return matched_points, unmatched_left_points

if __name__ == '__main__':
    window_size = 5
    threshold = 500
    l = np.array([[220.0019, 184.0242],
                  [187.0071, 387.0079],
                  [256.9686, 202.005],
                  [441.0076, 261.0358],
                  [463.0410, 473.0818],
                  [550.9522, 522.0760],
                  [70.9394, 22.4494]])
    r = np.array([[207.0075, 184.0486],
                 [190.0083, 387.0102],
                 [252.0076, 202.0754],
                 [433.0286, 261.5824],
                 [470.0779, 473.1367],
                 [553.0257, 521.3618],
                 [23.0119, 22.4648]])
    relative_orientation = np.array([1.0215501283596062e-06, 2.84745998228201e-07, -0.003346271609082209, 0.06944056882842992, -1.7706741742335887e-05])
    df = 0.15
    bx = np.mean(l[:, 0] - r[:, 0])

    left_image_path = 'data/left.tif'
    right_image_path = 'data/right.tif'

    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    left = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    right_image_matched = histogram_matching(right_image, left_image)

    left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    cv2.namedWindow("Left and Right Images")
    cv2.setMouseCallback("Left and Right Images", mark_point)
    combined = np.hstack((left, right))
    cv2.imshow("Left and Right Images", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    left_features = morcvec_feature_extraction(left_image, window_size, threshold)
    right_features = morcvec_feature_extraction(right_image_matched, window_size, threshold)

    left_image_bgr = cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
    right_image_bgr = cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR)
    for (x, y) in left_features:
        cv2.circle(left_image_bgr, (y, x), 1, (0, 255, 0), -1)

    for (x, y) in right_features:
        cv2.circle(right_image_bgr, (y, x), 1, (0, 255, 0), -1)

    cv2.imshow("Left and Right with Features", np.hstack((left_image_bgr, right_image_bgr)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    matched_points, unmatched_left_points = match_points(left_image, right_image_matched, left_features, right_features, relative_orientation, df, bx)
    left_image_match = cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
    right_image_match = cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR)
    combined_image = np.hstack((left_image_match, right_image_match))
    right_image_offset = left_image.shape[1]

    for left_point, right_point in matched_points:
        cv2.line(combined_image, (left_point[1], left_point[0]), (right_point[1] + right_image_offset, right_point[0]),
                 (0, 255, 0), 1)
        cv2.circle(combined_image, (left_point[1], left_point[0]), 2, (0, 0, 255), -1)
        cv2.circle(combined_image, (right_point[1] + right_image_offset, right_point[0]), 2, (0, 0, 255), -1)
    cv2.imshow("Matched Points", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()