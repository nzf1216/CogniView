import cv2

from landmarks import EyeLandmarkDetector


camera = cv2.VideoCapture(0)

detector = EyeLandmarkDetector()

while True:

    success, frame = camera.read()

    if not success:
        break

    face, _ = detector.detect(frame)

    if face:

        points = detector.get_landmark_points(
            face,
            frame
        )

        for x, y in points:

            cv2.circle(
                frame,
                (x, y),
                1,
                (0, 255, 0),
                -1
            )

    cv2.imshow(
        "CogniView Landmarks",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()