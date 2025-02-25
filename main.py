import argparse
import cv2
import imutils
import skimage
import numpy as np
import pandas as pd
from utils.warp import four_point_transform
from tabulate import tabulate

# Letter paper ratio
PAPER_RATIO = 8.5 / 11
MARGIN_CROP = 75
TABLE_PADDING = 5

# CLI arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input image")
args = vars(ap.parse_args())

# load the image
img = cv2.imread(args["image"])
original_img = img.copy()


# ============================== Paper detection ============================= #

# we don't need a huge image to detect edges, so resize it
ratio = img.shape[0] / 500
resized_img = imutils.resize(img, height=500)

# convert the image to grayscale, blur it, and find edges
grayscale_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
blurred_img = cv2.GaussianBlur(grayscale_img, (5, 5), 0)
edges_img = cv2.Canny(blurred_img, 75, 200)

# find contours and keep the largest 5 by area
contours, _ = cv2.findContours(edges_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

# filter our a contour with 4 sides
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(
        contour, 0.02 * perimeter, True
    )  # reduce the number of points in the contour

    if len(approx) == 4:
        paper_contour = approx
        break
if paper_contour is None:
    print("No paper contour found")
    exit(0)
paper_corners = []
for corner in paper_contour:
    paper_corners.append(tuple(corner[0]))
    cv2.circle(resized_img, tuple(corner[0]), 3, (0, 255, 0), 4)


# ======================== Paper warp and enhancement ======================== #

# warp the image
warped_img = four_point_transform(original_img, paper_contour.reshape(4, 2) * ratio)

# change the aspect ratio of the image to match the paper_ratio
warped_img = cv2.resize(
    warped_img, (warped_img.shape[1], int(warped_img.shape[1] / PAPER_RATIO))
)

# crop the image, removing the margins
cropped_img = warped_img[
    MARGIN_CROP : warped_img.shape[0] - MARGIN_CROP,
    MARGIN_CROP : warped_img.shape[1] - MARGIN_CROP,
]

# enhance colors and contrast
enhance_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
cropped_threshold = skimage.filters.threshold_local(
    enhance_img, 21, offset=10, method="gaussian"
)
enhance_img = (enhance_img > cropped_threshold).astype("uint8") * 255


# ============================== Table detection ============================= #

# detect edges
enhance_resized_img = imutils.resize(enhance_img, height=500)
enhance_edges_img = cv2.Canny(enhance_resized_img, 75, 200)

# remove double edges
kernel = np.ones((3, 3), np.uint8)
enhance_edges_img = cv2.morphologyEx(enhance_edges_img, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(
    enhance_edges_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# extract rectangles
rect_contours = []
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(approx) == 4:
        rect_contours.append(approx)
        if len(rect_contours) == 7:
            break

if len(rect_contours) < 7:
    print("No all tables were detected")
    exit(0)

# extract corners of rectangles
rect_corners = []
for contour in rect_contours:
    corners = []
    for corner in contour:
        corners.append(tuple(corner[0]))
    rect_corners.append(corners)


# =========================== Rotation compensation ========================== #

# get all horizontal/vertical lines (point pairs)
horizontal_lines = []
for corners in rect_corners:
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            if abs(corners[i][1] - corners[j][1]) < 10:
                horizontal_lines.append((corners[i], corners[j]))
vertical_lines = []
for corners in rect_corners:
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            if abs(corners[i][0] - corners[j][0]) < 10:
                vertical_lines.append((corners[i], corners[j]))

# calculate slopes
horizontal_slopes = []
for line in horizontal_lines:
    horizontal_slopes.append(
        ((line[1][1] - line[0][1]) / (line[1][0] - line[0][0]), line)
    )
vertical_slopes = []
for line in vertical_lines:
    # rotate the vertical slope 90 deg to avoid division by zero
    vertical_slopes.append(
        (-(line[1][0] - line[0][0]) / (line[1][1] - line[0][1]), line)
    )
slopes = horizontal_slopes + vertical_slopes


# remove outliers in slopes
def remove_outliers(data, z_threshold=1):
    values = [d[0] for d in data]
    if len(data) < 2:
        return data
    mean = np.mean(values)
    std = np.std(values)
    return [d for d in data if abs(d[0] - mean) / std < z_threshold]


slopes = remove_outliers(slopes)

# find how much we need to rotate the image to make the lines horizontal/vertical
avg_slope = np.mean([d[0] for d in slopes])
angle = np.arctan(avg_slope) * 180 / np.pi

# rotate the image
enhance_resized_img = imutils.rotate(enhance_resized_img, angle)
rotated_enhanced_img = imutils.rotate(enhance_img, angle)


def rotate_point(point, center, angle_degrees):
    # Convert angle to radians
    angle = np.radians(angle_degrees)
    # Shift point so the center is the origin
    shifted_x = point[0] - center[0]
    shifted_y = point[1] - center[1]
    # Apply 2D rotation
    rotated_x = shifted_x * np.cos(angle) - shifted_y * np.sin(angle)
    rotated_y = shifted_x * np.sin(angle) + shifted_y * np.cos(angle)
    # Shift back
    return (
        int(rotated_x + center[0]),
        int(rotated_y + center[1]),
    )


# rotate the rect_corners
height, width = enhance_resized_img.shape[:2]
center = (width / 2, height / 2)
rotated_rect_corners = []
for corners in rect_corners:
    rotated_corners = []
    for corner in corners:
        # rotate about the center of the image
        rotated_corners.append(rotate_point(corner, center, -angle))
    rotated_rect_corners.append(rotated_corners)
annotated_enhanced_img = imutils.resize(rotated_enhanced_img, height=500)
annotated_enhanced_img = cv2.cvtColor(annotated_enhanced_img, cv2.COLOR_GRAY2BGR)

# draw the rotated rectangles
for corners in rotated_rect_corners:
    cv2.drawContours(annotated_enhanced_img, [np.array(corners)], -1, (0, 255, 0), 2)


# ============================= Table extraction ============================= #

bounding_boxes = []
for corners in rotated_rect_corners:
    x, y, w, h = cv2.boundingRect(np.array(corners))
    bounding_boxes.append(((x, y, w, h), (x + w / 2, y + h / 2)))

# sort bounding boxes from top to bottom
bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1][1])
# order: basic_info, leave, auto, end_game, teleop, disconnects, overall

# draw the bounding boxes
for i in range(len(bounding_boxes)):
    box = bounding_boxes[i][0]
    center = bounding_boxes[i][1]
    x, y, w, h = box
    cv2.rectangle(annotated_enhanced_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.circle(
        annotated_enhanced_img, (int(center[0]), int(center[1])), 3, (0, 0, 255), 4
    )
    cv2.putText(
        annotated_enhanced_img,
        str(i),
        (int(center[0]) - 10, int(center[1]) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )

# add padding to the bounding boxes
for i in range(len(bounding_boxes)):
    box = bounding_boxes[i][0]
    x, y, w, h = box
    x -= TABLE_PADDING
    y -= TABLE_PADDING
    w += 2 * TABLE_PADDING
    h += 2 * TABLE_PADDING
    bounding_boxes[i] = ((x, y, w, h), bounding_boxes[i][1])

# crop the tables
tables = []
for box in bounding_boxes:
    x, y, w, h = box[0]
    ratio = rotated_enhanced_img.shape[0] / 500
    x = int(x * ratio)
    y = int(y * ratio)
    w = int(w * ratio)
    h = int(h * ratio)
    tables.append(rotated_enhanced_img[y : y + h, x : x + w])

# resize the tables
resized_tables = []
for table in tables:
    resized_tables.append(imutils.resize(table, height=250))

for i in range(len(resized_tables)):
    cv2.imshow(f"{i} Table", resized_tables[i])


def table_detection(img):
    img_gray = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    (thresh, img_bin) = cv2.threshold(
        img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    img_bin = cv2.bitwise_not(img_bin)

    kernel_length_v = (np.array(img_gray).shape[1]) // 120
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_v))
    im_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=3)
    vertical_lines_img = cv2.dilate(im_temp1, vertical_kernel, iterations=3)

    kernel_length_h = (np.array(img_gray).shape[1]) // 40
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length_h, 1))
    im_temp2 = cv2.erode(img_bin, horizontal_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(im_temp2, horizontal_kernel, iterations=3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    table_segment = cv2.addWeighted(
        vertical_lines_img, 0.5, horizontal_lines_img, 0.5, 0.0
    )
    table_segment = cv2.erode(cv2.bitwise_not(table_segment), kernel, iterations=2)
    thresh, table_segment = cv2.threshold(table_segment, 0, 255, cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(
        table_segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    count = 0

    full_list = []
    row = []
    data = []
    first_iter = 0
    firsty = -1

    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)

        if h > 9 and h < 100:
            if first_iter == 0:
                first_iter = 1
                firsty = y
            if firsty != y:
                row.reverse()
                full_list.append(row)
                row = []
                data = []
            # print(x, y, w, h)
            cropped = img[y : y + h, x : x + w]
            # cv2.imshow(str(i), cropped)
            # bounds = reader.readtext(cropped)

            try:
                data.append(bounds[0][1])
                data.append(w)
                row.append(data)
                data = []
            except:
                data.append("--")
                data.append(w)
                row.append(data)
                data = []
            firsty = y
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow(str(i), img)
    full_list.reverse()
    # print(full_list)

    new_data = []
    new_row = []
    for i in full_list:
        for j in i:
            new_row.append(j[0])
        new_data.append(new_row)
        new_row = []
    # print(new_data)

    # Convert list of lists into a DataFrame
    df = pd.DataFrame(new_data)
    df = df.applymap(lambda x: "" if pd.isna(x) else x)

    table = tabulate(df, headers="firstrow", tablefmt="grid")

    # Print DataFrame
    print(table)
    cv2.imshow("Table Detection", imutils.resize(img, height=500))


# table_detection(enhance_img.copy())

cv2.imshow("Original Image", resized_img)
# cv2.imshow("Edges Image", edges_img)
# cv2.imshow("Warped Image", imutils.resize(warped_img, height=500))
cv2.imshow("Cropped Image", imutils.resize(cropped_img, height=500))
cv2.imshow("Enhanced Image", annotated_enhanced_img)
cv2.imshow("Enhanced Edges Image", enhance_edges_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
