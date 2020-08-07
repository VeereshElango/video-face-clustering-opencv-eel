# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import FPS
from sklearn.cluster import DBSCAN
import imutils
import math
import numpy as np
import cv2
import os
import eel
import time

eel.init("web")
OUTPUT_DIR_PATH = "web/output"

supporting_files_path = "supporting_files/"

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([supporting_files_path, "deploy.prototxt"])
modelPath = os.path.sep.join([supporting_files_path, "res10_300x300_ssd_iter_140000.caffemodel"])
embedding_model_path = os.path.sep.join([supporting_files_path, "openface_nn4.small2.v1.t7"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
print("[INFO] loading face embedder...")
embedder = cv2.dnn.readNetFromTorch(embedding_model_path)


def detect_faces_from_frame(frame, frame_no, width, height):
    facecodes = []
    face_embeddings = []
    detected_faces = []
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            facecode = str(frame_no) + "_" + str(i)

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            facecodes.append(facecode)
            face_embeddings.append(vec.flatten())
            detected_faces.append(face)

    return {"faces": detected_faces, "face_embeddings": face_embeddings, "facecodes": facecodes}


def face_detection_in_video(path2video, seconds_per_frame=1):
    print(path2video)
    detected_faces = []
    face_embeddings = []
    facecodes = []

    print("[INFO] starting video file thread...")
    fvs = FileVideoStream(path2video)
    org_frame_rate = fvs.stream.get(5)
    width = fvs.stream.get(3)  # float
    height = fvs.stream.get(4)  # float
    no_of_frames_to_skip = org_frame_rate * seconds_per_frame
    no_of_frames_to_skip = no_of_frames_to_skip if no_of_frames_to_skip >= 1 else 1

    print("[INFO] original frame rate: {}".format(org_frame_rate))
    print("[INFO] no_of_frames_to_skip: {}".format(no_of_frames_to_skip))
    fvs.start()

    # start the FPS timer
    fps = FPS().start()

    frame_no = 1

    # loop over frames from the video file stream
    while fvs.more():
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale (while still retaining 3
        # channels)
        frame = fvs.read()

        if frame is not None:
            if frame_no % math.floor(no_of_frames_to_skip) == 0:
                data = detect_faces_from_frame(frame, frame_no, width, height)
                detected_faces.extend(data["faces"])
                face_embeddings.extend(data["face_embeddings"])
                facecodes.extend(data["facecodes"])
            frame_no += 1

            # update the FPS counter
            fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    print("[INFO] total faces detected: {} ".format(len(detected_faces)))
    print("[INFO] total frames: {} ".format(frame_no))

    return {"faces": detected_faces, "face_embeddings": face_embeddings, "facecodes": facecodes}


def cluster_faces(encodings):
    # cluster the embeddings
    print("[INFO] clustering...")
    clt = DBSCAN(metric="euclidean", n_jobs=-1)
    clt.fit(encodings)
    # determine the total number of unique faces found in the dataset
    return clt.labels_


def save_images(face_detections, labels, output_path):
    clustered_faces = {}
    labelIDs = np.unique(labels)
    numUniqueFaces = len(np.where(labelIDs > -1)[0])

    faces = face_detections["faces"]
    facecodes = face_detections["facecodes"]
    print("[INFO] # unique faces: {}".format(numUniqueFaces))
    # loop over the unique face integers
    for labelID in labelIDs:
        file_paths = []
        label_path = os.path.sep.join([output_path, str(labelID)])
        os.mkdir(label_path)

        # find all indexes into the `data` array that belong to the
        # current label ID, then randomly sample a maximum of 25 indexes
        # from the set
        print("[INFO] faces for face ID: {}".format(labelID))
        idxs = np.where(labels == labelID)[0]
        for id in idxs:
            filename = os.path.sep.join([label_path,facecodes[id] + ".jpg"])
            cv2.imwrite(filename, faces[id])
            file_paths.append(filename[4:])
        cluster_name = "Cluster "+str(labelID) if labelID != -1 else "Unknown"
        clustered_faces[cluster_name] = file_paths
    return clustered_faces


@eel.expose
def detect_faces_from_video(video_file_path, seconds_per_frame = 1):
    try:
        video_filename = video_file_path.split(os.path.sep)[-1].split(".")[0]
        dir_name = "_".join([video_filename, str(seconds_per_frame), str(int(time.time()))])
        output_dir_path = os.path.sep.join([OUTPUT_DIR_PATH, dir_name])
        os.mkdir(output_dir_path)
        eel.addText("Face Detection is on progress ...")
        data = face_detection_in_video(video_file_path, seconds_per_frame)
        eel.addText("Face Clustering is on progress ...")
        labels = cluster_faces(data["face_embeddings"])
        eel.addText("Saving images is on progress ...")
        clustered_faces = save_images(data, labels, output_dir_path)
        eel.addText("Completed!")
        return clustered_faces
    except Exception as e:
        eel.addErrorMessage(str(e))


eel.start('index.html', size=(1000, 600))

