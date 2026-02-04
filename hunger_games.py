import cv2
import mediapipe as mp #computer vision to track body info

cam = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose  #display all pose info
mp_hands = mp.solutions.hands #hand info
mp_drawing = mp.solutions.drawing_utils  #drawing on the vid

with mp_pose.Pose(static_image_mode=False) as pose, mp_hands.Hands(static_image_mode=False) as hands:
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #convert to rgb bc mediapipe wants
        results = pose.process(frame_rgb)  #get all pose info from the vid
        hand_results = hands.process(frame_rgb) #hand info

        above_head = False

        if results.pose_landmarks: #if detecting body
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)  # draw pose landmarks
            landmarks = results.pose_landmarks.landmark  # landmark positions
            body = mp_pose.PoseLandmark

            # get wrist and head y-coords
            right = landmarks[body.RIGHT_WRIST].y
            left = landmarks[body.LEFT_WRIST].y
            head = landmarks[body.NOSE].y

            # convert to pixel location
            img_height = frame.shape[0]
            right_pixel = right * img_height
            left_pixel = left * img_height
            head_pixel = head * img_height

            # check if above
            if right_pixel < head_pixel or left_pixel < head_pixel:
                above_head = True
                cv2.putText(frame, "hand is raised above head!", (10, 60), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                            fontScale=1.5, color=(0, 0, 255))

        if hand_results.multi_hand_landmarks: #if theres a hand in the frame
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) #draw hand landmarks

                #get fingers positions
                middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                #check if saluting
                if (index.x < middle.x < ring.x) and (pinky.x > ring.x) and (thumb.x > ring.x) and above_head:
                    cv2.putText(frame, "I VOLUNTEER AS TRIBUTE", (10, 120), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1.5, color=(0, 0, 255))

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
