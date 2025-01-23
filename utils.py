import os
import glob
import cv2

def write_file_into_folder(from_folder, to_folder, change_conf=0):
    """
    Writing file into folder with modified confidence score

    INPUT     

    from_folder   : Folder source
    to_folder     : Folder destination
    change_conf   : New confidence score value to change, Default 0 (No change)
                    If defined, it will change all confidence to the value. 
    
    EXAMPLE USE

    # Change all confidence scores to 0.3
    write_file_into_folder(from_folder, to_folder, change_conf=0.3)

    # Do not change confidence scores
    write_file_into_folder(from_folder, to_folder)

    """
    # Ensure the to_folder exists
    os.makedirs(to_folder, exist_ok=True)

    for file_name in os.listdir(from_folder):
        if file_name.endswith('.txt'):
            base_name = os.path.splitext(file_name)[0]
            with open(os.path.join(from_folder, file_name), 'r') as file1:
                with open(os.path.join(to_folder, f'{base_name}.txt'), 'a') as merged_file:
                    for line in file1:
                        # Split the line into columns
                        columns = line.strip().split()
                        # Change the last column value to 0.1 if change_conf is True
                        if columns and change_conf:  
                            # Ensure there is at least one column and change_conf is True
                            columns[-1] = str(change_conf) # change the last column with conf number
                        # Join the columns back into a line
                        new_line = ' '.join(columns)
                        # Write the modified or unmodified line to the output file
                        merged_file.write(new_line + '\n')


def merge_files_from_folders(folder1_path, folder2_path, merged_folder_path, change_conf=0):
    """
    Merging files from two folders into a single folder. 
    Files from the first folder are written as they are.
    Files from the second folder are appended with modified confidence score if specified.

    INPUT     

    folder1_path       : First folder path
    folder2_path       : Second folder path
    merged_folder_path : Merged folder path
    change_conf        : New confidence score value to change in folder2_path, Default 0 (No change)
                         If defined, it will change all confidence to the value.
    """

    if not os.path.exists(merged_folder_path):
        os.makedirs(merged_folder_path)
    
    # The first folder we do not change the confidence score
    write_file_into_folder(folder1_path, merged_folder_path)

    # The second folder we change confidence score
    write_file_into_folder(folder2_path, merged_folder_path, change_conf=change_conf)

def parse_boxes_from_txt(file_path):
    """
    Parsing xywh from txt file
    """
    boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.split()
            class_index = int(values[0])
            x_center = float(values[1])
            y_center = float(values[2])
            width = float(values[3])
            height = float(values[4])
            confidence = float(values[5])
            box = [class_index, x_center, y_center, width, height, confidence]
            boxes.append(box)
    return boxes


def apply_nms_to_folder(folder_path, output_folder_path, score_threshold=0.05, nms_threshold=0.5):
    """
    This function applies Non-Maximum Suppression (NMS) to all bounding box detections
    in text files within a specified folder and saves the results to an output folder.

    Parameters:
    - folder_path: The folder containing text files with bounding box detections.
    - output_folder_path: The folder where the filtered detections will be saved.
    - score_threshold: The confidence score threshold for filtering detections before NMS.
    - nms_threshold: The IoU (Intersection over Union) threshold for NMS.

    The function reads bounding box detections from each text file, applies NMS to
    filter out overlapping boxes, and writes the filtered detections to new text files
    in the specified output folder.
    """

    # Iterate over all text files in the specified folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            # Parse the bounding boxes from the text file
            boxes = parse_boxes_from_txt(file_path)

            # Convert detections to format expected by NMSBoxes (x1, y1, x2, y2)
            bounding_boxes = [[x1, y1, x2, y2] for _, x1, y1, x2, y2, _ in boxes]
            # Extract confidence scores for each bounding box
            confidence_scores = [confidence for _, _, _, _, _, confidence in boxes]

            # Apply Non-Maximum Suppression
            indices = cv2.dnn.NMSBoxes(bounding_boxes, confidence_scores, score_threshold, nms_threshold)

            # Check if any indices were returned
            if len(indices) > 0:
                indices = indices.flatten()
                # Filter out detections based on NMS results
                filtered_detections = [boxes[i] for i in indices]

                # Format the filtered detections for writing to a text file
                formatted_data = ""
                for bbox in filtered_detections:
                    formatted_data += " ".join(map(str, bbox)) + "\n"

                # Ensure the output folder exists, create if not
                os.makedirs(output_folder_path, exist_ok=True)

                # Define the path for the output text file
                output_file_path = os.path.join(output_folder_path, file_name)
                # Write the filtered detections to the output text file
                with open(output_file_path, "w") as file:
                    file.write(formatted_data)

def append_bboxes_to_image(image_path, bounding_boxes, output_folder):
    """
    This function appends bounding boxes to an image and saves the result.

    Parameters:
    - image_path: The file path of the image to which bounding boxes will be added.
    - bounding_boxes: A list of bounding boxes, where each bounding box is represented as
                      a tuple (class_index, x_center, y_center, width, height, conf).
    - output_folder: The folder where the modified image will be saved.

    The function reads the image, draws bounding boxes with corresponding class names and
    confidence scores, and saves the new image in the specified output folder.
    """
    
    # Define color map for different class indices
    color_map = {
        0: (0, 255, 0),    # Green
        1: (255, 0, 0),    # Blue
        2: (0, 0, 255),    # Red
        3: (255, 255, 0),  # Yellow
        4: (0, 255, 255),  # Cyan
        5: (150, 0, 255)   # Purple
        # Add more colors as needed
    }

    # Define class names for different class indices
    class_map = {
        0: "Abnormal",
        1: "Buah Busuk",
        2: "Buah Lewat Masak",
        3: "Buah Masak",
        4: "Buah Mentah",
        5: "Tandan Kosong",
    }

    # Read the image from the specified path
    image = cv2.imread(image_path)
    # Extract the file name without extension from the image path
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    # Ensure the output folder exists, create if not
    os.makedirs(output_folder, exist_ok=True)
    # Define the output path for the modified image
    output_path = os.path.join(output_folder, image_name + ".jpg")

    # Draw bounding boxes on the image
    for bbox in bounding_boxes:
        class_index, x_center, y_center, width, height, conf = bbox
        # Calculate the top-left and bottom-right coordinates of the bounding box
        x_min = int((x_center - width / 2) * image.shape[1])
        y_min = int((y_center - height / 2) * image.shape[0])
        x_max = int((x_center + width / 2) * image.shape[1])
        y_max = int((y_center + height / 2) * image.shape[0])

        # Get the color for the current class index, default to white if not found
        color = color_map.get(class_index, (255, 255, 255))

        # Draw the bounding box on the image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        # Prepare the text for the bounding box (class name and confidence score)
        text = "{} {}".format(class_map.get(class_index), round(conf, 2))
        # Draw the text on the image, above the bounding box
        cv2.putText(image, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the image with the bounding boxes drawn on it
    cv2.imwrite(output_path, image)


def append_bboxes_to_image2(image_path, bounding_boxes, output_folder):
    """
    This function appends bounding boxes to an image and saves the result.

    Parameters:
    - image_path: The file path of the image to which bounding boxes will be added.
    - bounding_boxes: A list of bounding boxes, where each bounding box is represented as
                      a tuple (class_index, x_center, y_center, width, height, conf).
    - output_folder: The folder where the modified image will be saved.

    The function reads the image, draws bounding boxes with corresponding class names and
    confidence scores, and saves the new image in the specified output folder.
    """
    
    # Define color map for different class indices
    color_map = {
        0: (0, 255, 0),    # Green
        1: (255, 0, 0),    # Blue
        3: (0, 0, 255),    # Red
        2: (255, 255, 0),  # Yellow
        4: (0, 255, 255),  # Cyan
        5: (150, 0, 255)   # Purple
        # Add more colors as needed
    }

    # Define class names for different class indices
    class_map = {
        0: "Abnormal bunch",
        1: "Damaged bunch",
        2: "Overripe bunch",
        3: "Ripe bunch",
        4: "Unripe bunch",
        5: "Empty bunch",
    }

    # Read the image from the specified path
    image = cv2.imread(image_path)
    # Extract the file name without extension from the image path
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    # Ensure the output folder exists, create if not
    os.makedirs(output_folder, exist_ok=True)
    # Define the output path for the modified image
    output_path = os.path.join(output_folder, image_name + ".jpg")

    # Draw bounding boxes on the image
    for bbox in bounding_boxes:
        class_index, x_center, y_center, width, height, conf = bbox
        # Calculate the top-left and bottom-right coordinates of the bounding box
        x_min = int((x_center - width / 2) * image.shape[1])
        y_min = int((y_center - height / 2) * image.shape[0])
        x_max = int((x_center + width / 2) * image.shape[1])
        y_max = int((y_center + height / 2) * image.shape[0])

        # Get the color for the current class index, default to white if not found
        color = color_map.get(class_index, (255, 255, 255))

        # Draw the bounding box on the image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        # Prepare the text for the bounding box (class name and confidence score)
        text = "{} {}".format(class_map.get(class_index), round(conf, 2))
        # Draw the text on the image, above the bounding box
        cv2.putText(image, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the image with the bounding boxes drawn on it
    cv2.imwrite(output_path, image)


def apply_bboxes_to_images(image_folder, bbox_folder, output_folder):
    """
    This function applies bounding boxes to all images in the specified folder.

    Parameters:
    - image_folder: The folder containing the images.
    - bbox_folder: The folder containing the bounding box text files.
    - output_folder: The folder where the modified images will be saved.

    The function reads each bounding box text file, finds the corresponding image,
    applies the bounding boxes to the image, and saves the result in the output folder.
    """
    
    # Iterate over all text files in the bounding box folder
    for txt_file in os.listdir(bbox_folder):
        if txt_file.endswith('.txt'):
            # Extract the image name (without extension) from the text file name
            image_name = os.path.splitext(txt_file)[0]
            # Define the path to the corresponding image file (assuming .jpg format)
            image_path = os.path.join(image_folder, image_name + '.jpg')
            # Define the path to the bounding box text file
            bbox_path = os.path.join(bbox_folder, txt_file)
            # Parse the bounding boxes from the text file
            bboxes = parse_boxes_from_txt(bbox_path)  # Assuming you have implemented this function
            # Append the bounding boxes to the image and save the result
            append_bboxes_to_image2(image_path, bboxes, output_folder)
