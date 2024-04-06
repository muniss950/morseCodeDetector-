
import cv2
import mediapipe as mp
import time

def calculate_ear(faceLm, index1, index2, index3, index4, index5, index6):
    # Calculate numerator part 1
    diff_x_1 = (faceLm.landmark[index1].x - faceLm.landmark[index2].x) ** 2
    diff_y_1 = (faceLm.landmark[index1].y - faceLm.landmark[index2].y) ** 2
    numerator_part1 = abs(diff_x_1 - diff_y_1)

    # Calculate numerator part 2
    diff_x_2 = (faceLm.landmark[index3].x - faceLm.landmark[index4].x) ** 2
    diff_y_2 = (faceLm.landmark[index3].y - faceLm.landmark[index4].y) ** 2
    numerator_part2 = abs(diff_x_2 - diff_y_2)

    # Calculate denominator
    diff_x_denom = (faceLm.landmark[index5].x - faceLm.landmark[index6].x) ** 2
    diff_y_denom = (faceLm.landmark[index5].y - faceLm.landmark[index6].y) ** 2
    denominator = abs(diff_x_denom - diff_y_denom)

    # Calculate ear
    ear = (numerator_part1 + numerator_part2) / denominator

    return ear

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def process_frame(frame, facemesh):
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = facemesh.process(imgRGB)
    return results

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)

    pTime = 0
    last_blink_time = 0

    mp_draw = mp.solutions.drawing_utils
    mp_facemesh = mp.solutions.face_mesh
    facemesh = mp_facemesh.FaceMesh(max_num_faces=2)
    draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=2)

    blink_threshold = 0.3  # Blink duration threshold in seconds
    blink_separator = ' '  # Separator to distinguish between short and long blinks

    blink_sequence = ""
    detected_letters = ""

    MorseCode = {
        "SL": "A",
        "LSSS": "B",
        "LSLS": "C",
        "LSS": "D",
        "S": "E",
        "SSLS": "F",
        "LLS": "G",
        "SSSS": "H",
        "SS": "I",
        "SLLL": "J",
        "LSL": "K",
        "SLSS": "L",
        "LL": "M",
        "LS": "N",
        "LLL": "O",
        "SLLS": "P",
        "LLSL": "Q",
        "SLS": "R",
        "SSS": "S",
        "L": "T",
        "SSL": "U",
        "SSSL": "V",
        "SLL": "W",
        "LSSL": "X",
        "LSLL": "Y",
        "LLSS": "Z"
    }

    while True:
        success, img = cap.read()
        scaled = rescale_frame(img, 150)

        results = process_frame(scaled, facemesh)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        if results.multi_face_landmarks:
            for faceLm in results.multi_face_landmarks:
                ear1 = calculate_ear(faceLm, 160, 144, 158, 153, 33, 133)
                ear2 = calculate_ear(faceLm, 385, 380, 387, 373, 362, 263)

                if ear1 < 0.2 and ear2 < 0.2:  # Closed eyes
                    if last_blink_time == 0:
                        last_blink_time = time.time()
                else:
                    if last_blink_time != 0:
                        blinked_duration = time.time() - last_blink_time
                        if blinked_duration >= blink_threshold:
                            if blinked_duration > 0.8:  # Long blink
                                print('L',end=' ')
                                blink_sequence += "L"
                            else:  # Short blink
                                print('S',end=' ')
                                blink_sequence += "S"
                            last_blink_time = 0

        # Detecting letter after a 2-second pause
        if last_blink_time != 0 and (time.time() - last_blink_time) > 2:
            if blink_sequence:
                if blink_sequence in MorseCode:
                    detected_letters += MorseCode[blink_sequence] + blink_separator
                    # print("Detected letter: ",detected_letters)
                blink_sequence = ""
                last_blink_time = 0
            print()
        scaled=cv2.flip(scaled,1)

        cv2.putText(scaled, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
        cv2.putText(scaled, detected_letters, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Image", scaled)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
